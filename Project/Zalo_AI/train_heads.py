# train_heads.py
import os, json, time, random, argparse
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import cv2

from matcher import FrozenEmbedder
from proposals import YOLOProposals
from utils import list_episodes, load_refs_for_episode, load_annotations, video_path_for_episode, iou_xyxy

class CosineScaleHead(nn.Module):
    """
    Minimal head: learnable positive scale sigma; score = sigma * cosine(f, w_template)
    """
    def __init__(self, init_sigma=10.0):
        super().__init__()
        # store log-sigma to keep positivity and stable optimization
        self.log_sigma = nn.Parameter(torch.log(torch.tensor([init_sigma], dtype=torch.float32)))

    def forward(self, cos_scores: torch.Tensor):
        # cos_scores: (N,) in [-1,1]
        sigma = torch.exp(self.log_sigma)
        return sigma * cos_scores  # (N,)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--split_file", required=True)  # JSON with {"train":[vid,...], "val":[...]}
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--freeze_backbone", action="store_true")
    ap.add_argument("--thaw_last", action="store_true")  # unfreeze RN18 layer4
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--max_props", type=int, default=80)
    ap.add_argument("--conf", type=float, default=0.01)
    ap.add_argument("--nms_iou", type=float, default=0.60)
    ap.add_argument("--yolo_ckpt", default=None)
    ap.add_argument("--save_ckpt", default="./ckpts/heads_freeze.pt")
    ap.add_argument("--load_ckpt", default="")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--debug", action="store_true")
    return ap.parse_args()

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def cosine(a, b):  # a:(N,D), b:(1,D) -> (N,)
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    return (a @ b.T).squeeze(-1)

def build_split(split_file):
    with open(split_file, "r", encoding="utf-8") as f:
        js = json.load(f)
    train_ids = list(js.get("train", []))
    val_ids   = list(js.get("val",   [])) or list(js.get("dev", []))
    return train_ids, val_ids

def sample_batch(
    data_root,
    vids,
    matcher,                # FrozenEmbedder
    yolo,                   # YOLOProposals
    batch_size=128,
    stride=2,
    iou_pos=0.5,
    iou_neg=0.2,
    debug=False,
):
    """
    Build one training batch of (features, labels).
    - Chooses random videos/frames
    - Gets YOLO proposals (falls back internally if needed)
    - Labels a proposal POS if IoU>=iou_pos with GT on that frame; otherwise NEG (hard negative preferred)
    - Embeds the selected crop via matcher.encode_np(...)
    Returns:
        X: torch.FloatTensor (B, 512)  |  y: torch.FloatTensor (B,)
        or (None, None) if unable to assemble a batch within the retry budget
    """
    import random
    import numpy as np
    import torch
    import cv2
    from utils import load_annotations, video_path_for_episode, iou_xyxy

    max_retries = 200
    retries = 0
    X_feats, y_lab = [], []

    while len(X_feats) < batch_size and retries < max_retries:
        if not vids:
            break

        vid = random.choice(vids)

        # Central GT map: {frame_idx: [[x1,y1,x2,y2], ...], ...}
        gt_by_f = load_annotations(data_root, vid) or {}

        # Pick a frame
        vpath = video_path_for_episode(data_root, vid)
        cap = cv2.VideoCapture(vpath)
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if nframes <= 0:
            cap.release()
            retries += 1
            continue

        fidx = random.randrange(1, max(2, nframes // max(1, stride))) * max(1, stride)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx - 1)
        ok, frame_bgr = cap.read()
        cap.release()
        if not ok or frame_bgr is None:
            retries += 1
            continue

        if debug:
            print(f"[DBG] sample vid={vid} frame={fidx} (n={nframes})")

        # Proposals (YOLOProposals already applies NMS/conf/max_det and grid fallback if needed)
        boxes = yolo(frame_bgr)  # list of (x1,y1,x2,y2)
        if not boxes:
            retries += 1
            continue

        # Decide POS/NEG using first GT of this frame (single-target assumption)
        gt_boxes = gt_by_f.get(fidx, [])
        label = 0
        bx = None

        if gt_boxes:
            # use the first GT box on that frame
            gt0 = tuple(gt_boxes[0])
            ious = [iou_xyxy(b, gt0) for b in boxes]

            # positive half the time if any IoU high enough; else hard negative
            if max(ious) >= iou_pos and np.random.rand() < 0.5:
                j = int(np.argmax(ious))
                bx = boxes[j]
                label = 1
            else:
                # hard negative: pick proposal with highest IoU that is still < iou_neg,
                # otherwise pick a low-scoring one
                order = np.argsort(-np.array(ious))
                neg_idx = None
                for k in order:
                    if ious[k] < iou_neg:
                        neg_idx = int(k)
                        break
                if neg_idx is None:
                    neg_idx = int(order[-1]) if len(order) > 0 else 0
                bx = boxes[neg_idx]
                label = 0
        else:
            # no GT on this frame → background
            bx = random.choice(boxes)
            label = 0

        # Crop and embed (convert BGR→RGB for the embedder)
        x1, y1, x2, y2 = bx
        crop = frame_bgr[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            retries += 1
            continue

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        feat = matcher.encode_np([crop_rgb])  # (1,512) float32
        if feat.shape[0] == 0:
            retries += 1
            continue

        if debug:
            print(f"[DBG]  choose box={bx} label={'POS' if label==1 else 'NEG'}")

        X_feats.append(feat[0])
        y_lab.append(label)

    if len(X_feats) == 0:
        if debug:
            print(f"[WARN] sample_batch() gave up after {retries} retries")
        return None, None

    import torch
    X = torch.from_numpy(np.stack(X_feats, axis=0)).float()     # (B,512)
    y = torch.tensor(y_lab, dtype=torch.float32)                # (B,)
    return X, y


def sample_batch1(data_root, vids, matcher, yolo, batch_size=128, stride=2, iou_pos=0.5, iou_neg=0.2, debug=False):
    X_feats = []
    y_lab   = []
    for _ in range(batch_size):
        vid = random.choice(vids)
        ann = load_annotations(data_root, vid)
        vpath = video_path_for_episode(data_root, vid)
        cap = cv2.VideoCapture(vpath)
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if nframes <= 0: 
            cap.release(); continue
        # pick a frame
        fidx = random.randrange(1, max(2, nframes//stride))*stride
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx-1)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None: 
            continue
        if debug:
            print(f"[DBG] sample vid={vid} frame={fidx} (n={nframes})")

        # gt boxes for this frame
        gt_by_f = load_annotations(data_root, vid) or {}
        gt_boxes = gt_by_f.get(fidx, [])

        # proposals
        boxes = yolo(frame)
        # choose pos/neg
        label = 0
        if len(gt_boxes) > 0 and len(boxes) > 0:
            # IoU against the first gt (single target per video)
            ious = [iou_xyxy(b, gt_boxes[0]) for b in boxes]
            # decide: 50% chance positive if any IoU>=0.5, else hard negative
            if max(ious) >= iou_pos and random.random() < 0.5:
                j = int(np.argmax(ious)); bx = boxes[j]; label = 1
            else:
                # hard negative near highest IoU but below threshold or random
                order = np.argsort(-np.array(ious))
                neg_idx = None
                for k in order:
                    if ious[k] < iou_neg:
                        neg_idx = int(k); break
                if neg_idx is None:
                    neg_idx = int(order[-1]) if len(order)>0 else 0
                bx = boxes[neg_idx]
        elif len(boxes) > 0:
            bx = random.choice(boxes); label = 0
        else:
            # fallback: center crop
            H,W = frame.shape[:2]; w=H//8; h=H//8
            x1 = max(0, W//2 - w); y1 = max(0, H//2 - h)
            bx = (x1,y1,x1+w,y1+h); label=0
        
        if debug:
            print(f"[DBG]  choose box={bx} label={'POS' if label==1 else 'NEG'}")
                
        crop = frame[bx[1]:bx[3], bx[0]:bx[2]]
        if crop.size == 0:
            continue
        feat = matcher.encode_np([cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)])
        X_feats.append(feat[0]); y_lab.append(label)

    if len(X_feats) == 0:
        return None, None
    X = torch.from_numpy(np.stack(X_feats, axis=0)).float()
    y = torch.tensor(y_lab, dtype=torch.float32)
    return X, y

def main():
    import os, time, json, random
    import numpy as np
    import torch
    import torch.nn.functional as F
    from matcher import FrozenEmbedder
    from proposals import YOLOProposals
    from utils import list_episodes, load_refs_for_episode

    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build split
    train_ids, val_ids = build_split(args.split_file)
    print(f"[SPLIT] train={len(train_ids)} val={len(val_ids)}")
    if len(train_ids) == 0:
        raise SystemExit("Empty train split — please regenerate ./splits/dev_split.json")

    # Models
    matcher = FrozenEmbedder(device=device)
    yolo = YOLOProposals(
        yolo_ckpt=args.yolo_ckpt,
        imgsz=args.imgsz,
        conf=args.conf,
        nms_iou=args.nms_iou,
        max_props=args.max_props,
        debug=args.debug,
    )
    head = CosineScaleHead(init_sigma=10.0).to(device)

    # (Optional) resume head
    if args.load_ckpt and os.path.isfile(args.load_ckpt):
        sd = torch.load(args.load_ckpt, map_location=device)
        if isinstance(sd, dict) and "head" in sd and sd["head"] is not None:
            head.load_state_dict(sd["head"], strict=False)
            print(f"[LOAD] head from {args.load_ckpt}")

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)

    # Steps per epoch (bounded so one epoch always finishes)
    steps_per_epoch = 200  # good default; change if you add a CLI flag

    best_sigma = float(torch.exp(head.log_sigma).item())

    for epoch in range(1, args.epochs + 1):
        print(f"\n[Epoch {epoch}/{args.epochs}]")
        t0 = time.time()
        head.train()
        losses = []

        for step in range(1, steps_per_epoch + 1):
            # Build a batch (may return None if sampler gives up this step)
            X, y = sample_batch(
                data_root=args.data_root,
                vids=train_ids,
                matcher=matcher,
                yolo=yolo,
                batch_size=args.batch_size,
                stride=2,
                iou_pos=0.5,
                iou_neg=0.2,
                debug=args.debug,
            )
            if X is None:
                if step % 10 == 0 or args.debug:
                    print(f"[SKIP] empty batch at step {step}/{steps_per_epoch}")
                continue

            # Build a template from a random video's refs (RGB arrays)
            vid_for_tmpl = random.choice(train_ids)
            refs = load_refs_for_episode(args.data_root, vid_for_tmpl)
            tmpl = matcher.encode_np(refs)                     # (M,512)
            tmpl = tmpl.mean(axis=0, keepdims=True)            # (1,512)

            # Cosine similarity (normalize in numpy for stability, then back to torch)
            Xn = X.numpy()
            Xn = Xn / (np.linalg.norm(Xn, axis=-1, keepdims=True) + 1e-8)
            Tn = tmpl / (np.linalg.norm(tmpl, axis=-1, keepdims=True) + 1e-8)
            cos = (Xn @ Tn.T).squeeze(-1)                      # (B,)
            cos = torch.from_numpy(cos).to(device)
            yb  = y.to(device)

            opt.zero_grad(set_to_none=True)
            logits = head(cos)                                 # (B,)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()
            opt.step()

            losses.append(float(loss.detach().cpu()))

            if step % 20 == 0:
                dt = time.time() - t0
                sigma_now = float(torch.exp(head.log_sigma).item())
                print(f"[Epoch {epoch}] step {step}/{steps_per_epoch}  "
                      f"loss={np.mean(losses):.4f}  sigma={sigma_now:.3f}  dt={dt:.1f}s")

        # End of epoch — save checkpoint
        os.makedirs(os.path.dirname(args.save_ckpt), exist_ok=True)
        torch.save({"head": head.state_dict(), "epoch": epoch}, args.save_ckpt)

        sigma_now = float(torch.exp(head.log_sigma).item())
        print(f"[EPOCH {epoch}] loss={np.mean(losses):.4f} sigma={sigma_now:.3f} "
              f"-> saved {args.save_ckpt}")

        if sigma_now > best_sigma:
            best_sigma = sigma_now

    print(f"[DONE] best_sigma={best_sigma:.3f} -> {args.save_ckpt}")
    
if __name__ == "__main__":
    main()
