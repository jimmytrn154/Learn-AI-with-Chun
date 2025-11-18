# inference.py
import os, json, time, math, argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F

from matcher import ImageMatcher          # your existing embedder wrapper
from proposals import YOLOProposals
from utils import list_episode_dirs, load_refs_for_episode, ensure_dir

# --------------- helpers -----------------

def crop_xyxy(img, box, pad=0.0):
    x1, y1, x2, y2 = box
    H, W = img.shape[:2]
    if pad > 0:
        w = x2 - x1; h = y2 - y1
        cx = (x1 + x2) / 2.0; cy = (y1 + y2) / 2.0
        w2 = w * (1.0 + pad); h2 = h * (1.0 + pad)
        x1 = int(max(0, math.floor(cx - w2/2))); y1 = int(max(0, math.floor(cy - h2/2)))
        x2 = int(min(W, math.ceil(cx + w2/2)));  y2 = int(min(H, math.ceil(cy + h2/2)))
    return img[y1:y2, x1:x2]

def roi_expand(box, pad, W, H):
    if box is None:
        return (0, 0, W, H)
    x1, y1, x2, y2 = box
    w = x2 - x1; h = y2 - y1
    cx = (x1 + x2) / 2.0; cy = (y1 + y2) / 2.0
    w2 = w * (1.0 + pad); h2 = h * (1.0 + pad)
    X1 = int(max(0, math.floor(cx - w2/2))); Y1 = int(max(0, math.floor(cy - h2/2)))
    X2 = int(min(W, math.ceil(cx + w2/2)));  Y2 = int(min(H, math.ceil(cy + h2/2)))
    return (X1, Y1, X2, Y2)

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter == 0: return 0.0
    area_a = max(0, ax2-ax1) * max(0, ay2-ay1)
    area_b = max(0, bx2-bx1) * max(0, by2-by1)
    return inter / (area_a + area_b - inter + 1e-6)

def percentile(x, p):
    if len(x) == 0: return 0.0
    return float(np.percentile(np.asarray(x, dtype=np.float32), p))

# --------------- warm-up -----------------

def warmup_thresholds(matcher, props, cap, num_frames=120, stride=10, roi_pad=0.0):
    """
    Sample some frames, compute cosine scores for top few proposals to pick per-video thresholds.
    """
    cosvals = []
    fidx = 0
    while True and len(cosvals) < 200:
        ok, frame = cap.read()
        if not ok: break
        fidx += 1
        if (fidx-1) % stride != 0: continue
        boxes = props(frame)
        confs = getattr(props, "last_scores", None) or [0.0] * len(boxes)
        if len(boxes) == 0: continue
        # take top-N by YOLO conf so we're sampling "likely" regions
        order = np.argsort(-np.asarray(confs, dtype=np.float32))
        take = order[: min(10, len(order))]
        crops = [crop_xyxy(frame, boxes[i]) for i in take]
        embs = matcher.encode_np(crops)  # (N, D)
        tmpl = matcher.get_template().reshape(1, -1)  # (1, D)
        cos = F.linear(torch.from_numpy(embs), torch.from_numpy(tmpl)).squeeze(1).numpy().tolist()
        cosvals.extend(cos)
        if fidx >= num_frames: break

    # conservative defaults if nothing gathered
    if len(cosvals) == 0:
        return 0.34, 0.20   # fallback

    hi = percentile(cosvals, 72)  # ~70-75th percentile
    lo = percentile(cosvals, 52)  # ~50-55th percentile
    return float(hi), float(lo)

# --------------- main inference per episode -----------------

def run_episode(root, vid, matcher, props, C):
    """
    C: config dict with keys like alpha_fuse, temporal_T, tau_high/low, roi_pad, etc.
    """
    vpath = os.path.join(root, "samples", vid, "drone_video.mp4")
    cap = cv2.VideoCapture(vpath)
    if not cap.isOpened():
        raise FileNotFoundError(vpath)

    # Build per-video template once
    refs = load_refs_for_episode(root, vid)  # list of BGR ref images
    tmpl = matcher.encode_np(refs, augs_per_ref=16).mean(axis=0, keepdims=True)  # (1,D)
    matcher.set_template(tmpl)

    # Warm-up to pick tau_* automatically
    cap_warm = cv2.VideoCapture(vpath)
    tau_hi_auto, tau_lo_auto = warmup_thresholds(matcher, props, cap_warm,
                                                 num_frames=C.get("warmup_frames", 120),
                                                 stride=max(1, C.get("frame_stride", 2)),
                                                 roi_pad=0.0)
    cap_warm.release()
        # AFTER (safe even if JSON has "null")
    def _coalesce(val, fallback):
        return fallback if val is None else val

    tau_high_cos = _coalesce(C.get("tau_high_cos"), tau_hi_auto)
    tau_low_cos  = _coalesce(C.get("tau_low_cos"),  tau_lo_auto)
    # If tau_high/low are None, fall back to the cosine thresholds
    tau_high     = _coalesce(C.get("tau_high"), tau_high_cos)
    tau_low      = _coalesce(C.get("tau_low"),  tau_low_cos)

    if C.get("topk_debug", 0):
        print(f"[THRESH] auto_hi_cos={tau_hi_auto:.3f} auto_lo_cos={tau_lo_auto:.3f} "
            f"=> tau_high_cos={tau_high_cos:.3f} tau_low_cos={tau_low_cos:.3f} "
            f"tau_high={tau_high:.3f} tau_low={tau_low:.3f}")
    alpha_fuse   = C.get("alpha_fuse", 0.88)
    roi_pad      = C.get("roi_pad", 0.40)
    min_commit   = C.get("min_commit", 1)
    max_lost     = C.get("max_lost", 25)
    topk_debug   = C.get("topk_debug", 0)

    detections = []
    lost = 0
    fidx = 0
    last_box = None
    active = False
    committed = 0

    while True:
        ok, frame = cap.read()
        if not ok: break
        fidx += 1
        if C.get("frame_stride", 2) > 1 and ((fidx-1) % C["frame_stride"]) != 0:
            continue

        H, W = frame.shape[:2]

        # proposals
        boxes = props(frame)
        confs = getattr(props, "last_scores", None) or [0.0] * len(boxes)

        # compute an adaptive pad based on last displacement
        if last_box is not None and "roi_pad" in C:
            if "_prev_center" not in locals(): _prev_center = None
            cx = 0.5*(last_box[0]+last_box[2]); cy = 0.5*(last_box[1]+last_box[3])
            if _prev_center is not None:
                dx = abs(cx - _prev_center[0]) / max(1, W)
                dy = abs(cy - _prev_center[1]) / max(1, H)
                jump = max(dx, dy)
            else:
                jump = 0.0
            _prev_center = (cx, cy)
            roi_pad_eff = float(C.get("roi_pad", 0.55)) * (1.0 + 1.5 * min(0.2, jump))
        else:
            roi_pad_eff = float(C.get("roi_pad", 0.55))

        # ROI filtering when a track exists
        if last_box is not None:
            X1, Y1, X2, Y2 = roi_expand(last_box, roi_pad_eff, W, H)
            keep = []
            for i, (x1, y1, x2, y2) in enumerate(boxes):
                cx = 0.5*(x1+x2); cy = 0.5*(y1+y2)
                if (X1 <= cx <= X2) and (Y1 <= cy <= Y2):
                    keep.append(i)
            if len(keep) > 0:
                boxes = [boxes[i] for i in keep]
                confs = [confs[i] for i in keep]

        if len(boxes) == 0:
            # miss
            if active:
                lost += 1
                if lost > max_lost:
                    active = False; last_box = None; committed = 0
            continue

        crops = [crop_xyxy(frame, b) for b in boxes]
        embs = matcher.encode_np(crops)  # (N,D)
        cos = F.linear(torch.from_numpy(embs), torch.from_numpy(matcher.get_template())).squeeze(1).numpy()
        cos = cos.astype(np.float32)
        confs = np.asarray(confs, dtype=np.float32)

        # normalize YOLO conf into [0,1] (already is), fuse
        fused = alpha_fuse * cos + (1.0 - alpha_fuse) * confs

        # pick best by fused, but gate by both
        order = np.argsort(-fused)
        def diverse_topk(boxes, scores, K=10, iou_thr=0.7):
            keep = []
            for i in order:
                ok = True
                for j in keep:
                    if iou_xyxy(boxes[i], boxes[j]) > iou_thr:
                        ok = False; break
                if ok: keep.append(i)
                if len(keep) >= K: break
            return keep

        div_idx = diverse_topk(boxes, fused, K=min(10, len(boxes)), iou_thr=0.7)
        bidx = div_idx[0]  # best among diverse set
        
        EMA_BETA = C.get("ema_beta", 0.10)
        if (bcos >= tau_high_cos) or (bfuse >= tau_high):
            # update template with current best crop
            emb_best = torch.from_numpy(matcher.encode_np([crop_xyxy(frame, b)])).squeeze(0).numpy()
            tmpl = matcher.get_template()
            tmpl = (1.0 - EMA_BETA) * tmpl + EMA_BETA * emb_best[None, :]
            # L2 normalize
            tmpl = tmpl / max(1e-6, np.linalg.norm(tmpl))
            matcher.set_template(tmpl)
        
        bcos = float(cos[bidx]); bfuse = float(fused[bidx])
        b = boxes[bidx]

        # optional debug top-K
        if topk_debug > 0:
            K = min(topk_debug, len(order))
            print(f"[TOPK] {vid} f={fidx} K={K} margin={bfuse - float(fused[order[1]]) if len(order)>1 else 1.0:.3f}")
            for k in range(K):
                i = order[k]
                x1,y1,x2,y2 = boxes[i]
                print(f"  #{k+1}: fused={fused[i]:+.3f}  cos={cos[i]:+.3f}  y={confs[i]:+.3f}  "
                      f"wh=({x2-x1}, {y2-y1})  box=({x1}, {y1}, {x2}, {y2})")

        # hysteresis logic
        if not active:
            if (bcos >= tau_high_cos) or (bfuse >= tau_high):
                active = True
                last_box = b
                committed = 0
            else:
                # stay idle
                continue
        else:
            if (bcos < tau_low_cos) and (bfuse < tau_low):
                lost += 1
                if lost > max_lost:
                    active = False; last_box = None; committed = 0
                continue
            else:
                lost = 0
                last_box = b

        # commit if active and beyond warm-up length
        committed += 1
        if committed >= min_commit:
            detections.append({"bboxes": [{"frame": fidx, "x1": last_box[0], "y1": last_box[1],
                                           "x2": last_box[2], "y2": last_box[3]}]})

    cap.release()
    return {"video_id": vid, "detections": detections}


# --------------- top-level -----------------

def run_inference_on_split(data_root, vids, ckpt, cfg, yolo_ckpt, debug=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # matcher: frozen feature extractor (e.g., RN18/RN50/CLIP/DINOv2 behind ImageMatcher)
    matcher = ImageMatcher(device=device)

    # proposals: now with your trained YOLO + multi-scale TTA
    tta_sizes = cfg.get("tta_sizes", [cfg.get("imgsz", 960)])
    props = YOLOProposals(
        yolo_ckpt=yolo_ckpt,
        imgsz=cfg.get("imgsz", 960),
        conf=cfg.get("conf", 0.02),
        nms_iou=cfg.get("nms_iou", 0.60),
        max_props=cfg.get("max_props", 80),
        tta_sizes=tta_sizes,
        device=0 if torch.cuda.is_available() else None,
        half=True,
        allow_fallback=True,
        min_wh=cfg.get("min_wh", 8),
        min_area=cfg.get("min_area", 36),
        debug=debug,
    )

    preds = []
    for vid in vids:
        t0 = time.time()
        out = run_episode(data_root, vid, matcher, props, cfg)
        dt = time.time() - t0
        print(f"[DONE] {vid}: {len(out['detections'])} segments, {dt:.1f}s")
        preds.append(out)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--ckpt", required=True, help="(kept for compatibility; matcher is frozen)")
    ap.add_argument("--config", required=True, help="JSON with all thresholds/hparams")
    ap.add_argument("--yolo_ckpt", default=None, help="Your trained YOLO weights (*.pt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        C = json.load(f)

    vids = list_episode_dirs(args.data_root)  # returns list of folder names under samples/
    ensure_dir(os.path.dirname(args.out))

    preds = run_inference_on_split(args.data_root, vids, args.ckpt, C, args.yolo_ckpt, debug=args.debug)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(preds, f, indent=2)
    print(f"[SAVE] predictions -> {args.out}")


if __name__ == "__main__":
    main()
