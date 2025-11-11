"""
baseline_predictor.py
Baseline: tiny YOLO (class-agnostic proposals) + ResNet18 embedding matcher + IoU/appearance tracking
Outputs submission-style predictions.json and (optionally) ST-IoU on a validation split.

Directory layout expected (per your spec):
dataset/
 ├─ samples/
 │   ├─ drone_video_001/
 │   │   ├─ object_images/ img_1.jpg img_2.jpg img_3.jpg
 │   │   └─ drone_video.mp4
 │   ├─ drone_video_002/ ...
 │   └─ ...
 └─ annotations/ annotations.json          # optional (only for eval / viz)

Run:
python baseline_predictor.py --data_root dataset --out predictions.json
Add --eval to compute ST-IoU against annotations/annotations.json
"""

import os, json, math, argparse, random, time
from pathlib import Path
from typing import List, Tuple, Dict, Iterable

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

# Optional (tiny) YOLO proposals
try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except Exception:
    _HAS_YOLO = False


# ------------------------------
# Utility
# ------------------------------
def set_seed(seed=1337):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def to_device(x):
    return x.cuda() if torch.cuda.is_available() else x

def l2_normalize(x, dim=-1, eps=1e-6):
    return x / (x.norm(dim=dim, keepdim=True) + eps)

def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    xx1, yy1 = max(ax1,bx1), max(ay1,by1)
    xx2, yy2 = min(ax2,bx2), min(ay2,by2)
    w, h = max(0, xx2-xx1), max(0, yy2-yy1)
    inter = w*h
    if inter == 0: return 0.0
    area_a = max(0, ax2-ax1) * max(0, ay2-ay1)
    area_b = max(0, bx2-bx1) * max(0, by2-by1)
    return inter / float(area_a + area_b - inter + 1e-6)

def nms_xyxy(boxes, scores, iou_thr=0.5):
    if not boxes:
        return []
    boxes = np.asarray(boxes, dtype=float)
    scores = np.asarray(scores, dtype=float)
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        inds = np.where(ovr <= iou_thr)[0]
        order = order[inds + 1]
    return keep


# ------------------------------
# Data helpers
# ------------------------------
def find_episodes(data_root: str) -> List[str]:
    """Return sorted list of video_id directories under samples/"""
    p = Path(data_root) / "samples"
    vids = [d.name for d in p.iterdir() if d.is_dir()]
    vids.sort()
    return vids

def load_refs_for_episode(data_root: str, video_id: str) -> List[np.ndarray]:
    ref_dir = Path(data_root) / "samples" / video_id / "object_images"
    imgs = []
    for p in sorted(ref_dir.glob("*")):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            im = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if im is not None:
                imgs.append(im)
    if len(imgs) == 0:
        raise RuntimeError(f"No reference images for {video_id}")
    return imgs[:3]

def video_path_for_episode(data_root: str, video_id: str) -> str:
    return str(Path(data_root) / "samples" / video_id / "drone_video.mp4")

def load_annotations_json(data_root: str) -> list:
    p = Path(data_root) / "annotations" / "annotations.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def frame_to_boxes(entries: list, video_id: str, key: str) -> Dict[int, List[Tuple[int,int,int,int]]]:
    """Handle multi-segment annotations/detections."""
    if not entries: return {}
    by_vid = {e.get("video_id"): e for e in entries if "video_id" in e}
    if video_id not in by_vid: return {}
    rec = by_vid[video_id]
    segs = rec.get(key, [])
    # normalize shapes
    if isinstance(segs, dict) and "bboxes" in segs:
        segs = [segs]
    if isinstance(segs, list) and segs and "frame" in segs[0]:
        segs = [{"bboxes": segs}]
    out = {}
    for seg in segs:
        for b in seg.get("bboxes", []):
            k = int(b["frame"])
            out.setdefault(k, []).append((int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])))
    return out


# ------------------------------
# Proposal generator (YOLO nano)
# ------------------------------
class YOLOProposals:
    """Class-agnostic proposals using a tiny YOLO; if YOLO unavailable, falls back to sliding windows."""
    def __init__(self, conf=0.05, iou=0.7, imgsz=640, max_candidates=200):
        self.conf = conf; self.iou = iou; self.imgsz = imgsz; self.max_candidates = max_candidates
        self.model = None
        if _HAS_YOLO:
            try:
                # yolov8n.pt is ~3.2M params
                self.model = YOLO("yolov8n.pt")
            except Exception:
                self.model = None

    def __call__(self, frame_bgr: np.ndarray) -> List[Tuple[int,int,int,int]]:
        H, W = frame_bgr.shape[:2]
        if self.model is None:
            # simple sliding window fallback (sparse to stay cheap)
            sizes = [64, 96, 128, 160, 192]
            stride = 0.5  # 50% overlap
            props = []
            for s in sizes:
                sx = max(16, int(s))
                sy = max(16, int(s))
                step_x = max(8, int(sx * (1 - stride)))
                step_y = max(8, int(sy * (1 - stride)))
                for y in range(0, H - sy, step_y):
                    for x in range(0, W - sx, step_x):
                        props.append((x, y, x + sx, y + sy))
                        if len(props) >= self.max_candidates:
                            return props
            return props[: self.max_candidates]
        # YOLO forward
        res = self.model.predict(frame_bgr, conf=self.conf, iou=self.iou, imgsz=self.imgsz, verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0,4))
        # class-agnostic: ignore cls; cap candidates
        boxes = boxes[: self.max_candidates]
        out = []
        for x1,y1,x2,y2 in boxes:
            x1 = int(max(0, min(W-1, x1))); y1 = int(max(0, min(H-1, y1)))
            x2 = int(max(0, min(W-1, x2))); y2 = int(max(0, min(H-1, y2)))
            if x2 > x1 and y2 > y1:
                out.append((x1,y1,x2,y2))
        return out


# ------------------------------
# Embedding model (ResNet-18 + tiny head)
# ------------------------------
class EmbeddingMatcher(nn.Module):
    """
    ResNet-18 backbone (frozen) + projection head 512->256 with L2 normalize.
    """
    def __init__(self, out_dim=256):
        super().__init__()
        rn18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        for p in rn18.parameters():
            p.requires_grad_(False)
        # take features before FC
        self.backbone = nn.Sequential(*list(rn18.children())[:-1])  # (B,512,1,1)
        self.head = nn.Linear(512, out_dim, bias=False)
        self.prep = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ColorJitter(0.2,0.2,0.2,0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])
        ])

    @torch.no_grad()
    def encode_np(self, np_imgs: List[np.ndarray]) -> torch.Tensor:
        """np_imgs: list of HxWxC BGR or RGB? We'll convert BGR->RGB."""
        if len(np_imgs) == 0:
            return torch.empty(0, 256)
        tensors = []
        for img in np_imgs:
            if img.shape[-1] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensors.append(self.prep(img))
        batch = torch.stack(tensors, dim=0)
        batch = to_device(batch).float()
        feats = self.backbone(batch).flatten(1)         # (B,512)
        proj  = self.head(feats)                        # (B,256)
        proj  = l2_normalize(proj, dim=1)
        return proj

def build_template(matcher: EmbeddingMatcher, ref_imgs: List[np.ndarray], augs_per_ref=16) -> torch.Tensor:
    """Augment refs, embed, and average → template vector (1, D)."""
    # Simple augmentation: random rescale + color jitter + small blur
    auged = []
    for im in ref_imgs:
        h,w = im.shape[:2]
        for _ in range(augs_per_ref):
            scale = np.random.uniform(0.8, 1.2)
            nh, nw = max(16, int(h*scale)), max(16, int(w*scale))
            resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR)
            if np.random.rand() < 0.3:
                k = np.random.choice([3,5])
                resized = cv2.GaussianBlur(resized, (k,k), 0)
            auged.append(resized)
    with torch.no_grad():
        embs = matcher.encode_np(auged)  # (N, D)
        tmpl = embs.mean(0, keepdim=True)  # (1, D)
        tmpl = l2_normalize(tmpl, dim=1)
    return tmpl  # (1, D)


# ------------------------------
# Segmenter / Tracker logic
# ------------------------------
class SingleTargetTracker:
    """
    Keeps a single track using IoU + appearance similarity.
    Hysteresis thresholds for start/keep; min segment length & gap fill.
    """
    def __init__(self,
                 tau_high=0.45, tau_low=0.35,
                 assoc_lambda=0.5,
                 search_roi_pad=0.35,
                 max_lost=10,
                 min_commit=3,
                 gap_fill=2):
        self.tau_high = tau_high
        self.tau_low  = tau_low
        self.assoc_lambda = assoc_lambda
        self.search_roi_pad = search_roi_pad
        self.max_lost = max_lost
        self.min_commit = min_commit
        self.gap_fill = gap_fill

        self.active = False
        self.last_box = None
        self.lost = 0
        self.buffer = []     # pending boxes before commit
        self.current_segment = []  # committed boxes for the ongoing segment
        self.detections = []  # committed for entire video (flat list)
        self.prev_frame = None

    def _search_window(self, frame_shape):
        H,W = frame_shape[:2]
        if self.last_box is None:
            return (0,0,W-1,H-1)
        x1,y1,x2,y2 = self.last_box
        pad = int(self.search_roi_pad * max(x2-x1, y2-y1))
        rx1 = max(0, x1 - pad); ry1 = max(0, y1 - pad)
        rx2 = min(W-1, x2 + pad); ry2 = min(H-1, y2 + pad)
        return (rx1,ry1,rx2,ry2)

    def _associate(self, boxes, sims):
        if not boxes:
            return None, None
        if self.last_box is None or self.lost >= self.max_lost:
            # pick best by similarity
            idx = int(np.argmax(sims))
            return boxes[idx], sims[idx]
        best_val = -1e9; best_idx = -1
        for i,(b,s) in enumerate(zip(boxes, sims)):
            val = self.assoc_lambda * iou_xyxy(b, self.last_box) + (1.0 - self.assoc_lambda) * float(s)
            if val > best_val:
                best_val = val; best_idx = i
        return boxes[best_idx], sims[best_idx]

    def _commit_buffer_if_ready(self):
        if len(self.buffer) >= self.min_commit:
            self.current_segment.extend(self.buffer)
            self.detections.extend(self.buffer)
            self.buffer = []
            self.active = True

    def _flush_segment(self):
        self.buffer = []
        self.current_segment = []
        self.active = False

    def update(self, frame_idx: int, frame_shape, candidate_boxes: List[Tuple[int,int,int,int]], candidate_scores: List[float]):
        # select one candidate (or none)
        chosen, sim = self._associate(candidate_boxes, candidate_scores) if candidate_boxes else (None, None)

        # gating with hysteresis
        if chosen is not None:
            if not self.active:
                # pre-commit buffer with tau_high
                if sim >= self.tau_high:
                    self.buffer.append({"frame": int(frame_idx), "x1": int(chosen[0]), "y1": int(chosen[1]), "x2": int(chosen[2]), "y2": int(chosen[3])})
                    self._commit_buffer_if_ready()
            else:
                # keep segment with tau_low
                if sim >= self.tau_low:
                    # fill small gaps if any
                    if self.prev_frame is not None and frame_idx - self.prev_frame > 1 and (frame_idx - self.prev_frame - 1) <= self.gap_fill and len(self.current_segment)>0:
                        # simple linear interp of boxes
                        last = self.current_segment[-1]
                        num_gap = frame_idx - self.prev_frame - 1
                        for g in range(1, num_gap+1):
                            t = g / (num_gap+1)
                            interp = (
                                int((1-t)*last["x1"] + t*chosen[0]),
                                int((1-t)*last["y1"] + t*chosen[1]),
                                int((1-t)*last["x2"] + t*chosen[2]),
                                int((1-t)*last["y2"] + t*chosen[3]),
                            )
                            self.current_segment.append({"frame": int(self.prev_frame+g), "x1": interp[0], "y1": interp[1], "x2": interp[2], "y2": interp[3]})
                            self.detections.append(self.current_segment[-1])
                    self.current_segment.append({"frame": int(frame_idx), "x1": int(chosen[0]), "y1": int(chosen[1]), "x2": int(chosen[2]), "y2": int(chosen[3])})
                    self.detections.append(self.current_segment[-1])
                else:
                    # similarity too low -> close the segment
                    self._flush_segment()

            self.last_box = chosen
            self.lost = 0
        else:
            # no candidate
            self.lost += 1
            if self.lost >= self.max_lost:
                self._flush_segment()

        self.prev_frame = frame_idx

    def finalize(self) -> List[dict]:
        # If we never committed (buffer only), drop it
        self.buffer = []
        return self.detections


# ------------------------------
# ST-IoU evaluator
# ------------------------------
def _extract_segments(record, key):
    segs = record.get(key, [])
    if isinstance(segs, dict) and "bboxes" in segs: return [segs]
    if isinstance(segs, list) and segs and "frame" in segs[0]: return [{"bboxes": segs}]
    return segs if isinstance(segs, list) else []

def st_iou_one(gt_entries, pred_entries, video_id: str) -> float:
    # frame -> list of boxes
    gt_map   = frame_to_boxes(gt_entries,   video_id, key="annotations")
    pred_map = frame_to_boxes(pred_entries, video_id, key="detections")
    inter = sorted(set(gt_map.keys()).intersection(pred_map.keys()))
    union = sorted(set(gt_map.keys()).union(pred_map.keys()))
    if not union: return 0.0
    num = 0.0
    for f in inter:
        # single-target: credit best match
        best = 0.0
        for g in gt_map[f]:
            for p in pred_map[f]:
                best = max(best, iou_xyxy(g,p))
        num += best
    den = float(len(union))  # sum of 1 over union frames
    return num / den

def st_iou_mean(gt_entries, pred_entries, video_ids: Iterable[str]) -> float:
    vals = [st_iou_one(gt_entries, pred_entries, vid) for vid in video_ids]
    return float(np.mean(vals)) if vals else 0.0


# ------------------------------
# Main predictor
# ------------------------------
class Predictor:
    def __init__(self,
                 conf=0.05, nms_iou=0.7, imgsz=640, max_props=200,
                 tau_high=0.45, tau_low=0.35, assoc_lambda=0.5,
                 roi_pad=0.35, max_lost=10, min_commit=3, gap_fill=2,
                 frame_stride=1, nms_final_iou=0.55):
        self.props = YOLOProposals(conf=conf, iou=nms_iou, imgsz=imgsz, max_candidates=max_props)
        self.matcher = to_device(EmbeddingMatcher(out_dim=256).eval())
        self.tau_high = tau_high; self.tau_low = tau_low
        self.assoc_lambda = assoc_lambda
        self.roi_pad = roi_pad; self.max_lost = max_lost
        self.min_commit = min_commit; self.gap_fill = gap_fill
        self.frame_stride = frame_stride
        self.nms_final_iou = nms_final_iou

        self.crop_prep = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])
        ])

    @torch.no_grad()
    def _embed_crops(self, crops: List[np.ndarray]) -> torch.Tensor:
        tensors = []
        for c in crops:
            if c is None or c.size == 0: continue
            img = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
            tensors.append(self.crop_prep(img))
        if not tensors:
            return torch.empty(0, 256)
        batch = torch.stack(tensors, dim=0)
        batch = to_device(batch).float()
        feats = self.matcher.backbone(batch).flatten(1)
        proj  = l2_normalize(self.matcher.head(feats), dim=1)
        return proj

    def _cosine_scores(self, embs: torch.Tensor, tmpl: torch.Tensor) -> np.ndarray:
        if embs.numel() == 0:
            return np.zeros((0,), dtype=np.float32)
        sims = F.linear(embs, tmpl)  # (N,1)
        return sims.squeeze(-1).detach().float().cpu().numpy()

    def _segmentize(self, dets: List[dict], max_gap=1) -> List[dict]:
        """Split flat detections into segments by frame gaps > max_gap."""
        dets = sorted(dets, key=lambda d: d["frame"])
        segments, cur = [], []
        prev = None
        for d in dets:
            if prev is not None and d["frame"] - prev["frame"] > max_gap:
                if cur: segments.append({"bboxes": cur}); cur = []
            cur.append(d)
            prev = d
        if cur: segments.append({"bboxes": cur})
        return segments

    def run_episode(self, data_root: str, video_id: str) -> dict:
        # build template
        refs = load_refs_for_episode(data_root, video_id)
        template = build_template(self.matcher, refs, augs_per_ref=16)  # (1,256)

        # video
        vpath = video_path_for_episode(data_root, video_id)
        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {vpath}")

        tracker = SingleTargetTracker(
            tau_high=self.tau_high, tau_low=self.tau_low, assoc_lambda=self.assoc_lambda,
            search_roi_pad=self.roi_pad, max_lost=self.max_lost,
            min_commit=self.min_commit, gap_fill=self.gap_fill
        )

        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok: break
            if self.frame_stride > 1 and (frame_idx % self.frame_stride) != 0:
                frame_idx += 1
                continue

            # proposals
            props = self.props(frame)  # list of (x1,y1,x2,y2)

            # crops & embeddings
            crops = [frame[y1:y2, x1:x2] for (x1,y1,x2,y2) in props]
            embs  = self._embed_crops(crops)
            sims  = self._cosine_scores(embs, template)

            # final NMS by similarity
            keep = nms_xyxy(props, sims, iou_thr=self.nms_final_iou)
            props_kept = [props[i] for i in keep]
            sims_kept  = [float(sims[i]) for i in keep]

            tracker.update(frame_idx, frame.shape, props_kept, sims_kept)
            frame_idx += 1

        cap.release()
        dets = tracker.finalize()  # flat list of dicts with frame,x1,y1,x2,y2
        segments = self._segmentize(dets, max_gap=1)
        return {"video_id": video_id, "detections": segments}


# ------------------------------
# CLI / main
# ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Path to dataset root (containing samples/ and annotations/)")
    parser.add_argument("--out", type=str, default="predictions.json", help="Output predictions JSON")
    parser.add_argument("--eval", action="store_true", help="Compute ST-IoU vs annotations if available")
    parser.add_argument("--seed", type=int, default=1337)
    # quick knobs
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--nms_iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max_props", type=int, default=200)
    parser.add_argument("--tau_high", type=float, default=0.45)
    parser.add_argument("--tau_low", type=float, default=0.35)
    parser.add_argument("--assoc_lambda", type=float, default=0.5)
    parser.add_argument("--roi_pad", type=float, default=0.35)
    parser.add_argument("--max_lost", type=int, default=10)
    parser.add_argument("--min_commit", type=int, default=3)
    parser.add_argument("--gap_fill", type=int, default=2)
    parser.add_argument("--frame_stride", type=int, default=1)
    args = parser.parse_args()

    set_seed(args.seed)

    predictor = Predictor(
        conf=args.conf, nms_iou=args.nms_iou, imgsz=args.imgsz, max_props=args.max_props,
        tau_high=args.tau_high, tau_low=args.tau_low, assoc_lambda=args.assoc_lambda,
        roi_pad=args.roi_pad, max_lost=args.max_lost, min_commit=args.min_commit, gap_fill=args.gap_fill,
        frame_stride=args.frame_stride
    )

    video_ids = find_episodes(args.data_root)
    print(f"[INFO] Found {len(video_ids)} episodes:", video_ids)

    preds = []
    for vid in video_ids:
        t0 = time.time()
        entry = predictor.run_episode(args.data_root, vid)
        dt = time.time() - t0
        preds.append(entry)
        print(f"[DONE] {vid}: {sum(len(s['bboxes']) for s in entry['detections'])} boxes, {len(entry['detections'])} segments, {dt:.1f}s")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(preds, f, indent=2)
    print(f"[SAVE] predictions -> {args.out}")

    if args.eval:
        gt_entries = load_annotations_json(args.data_root)
        if gt_entries:
            score = st_iou_mean(gt_entries, preds, video_ids)
            print(f"[ST-IoU] mean over {len(video_ids)} videos: {score:.4f}")
        else:
            print("[WARN] annotations/annotations.json not found; skipping ST-IoU.")


if __name__ == "__main__":
    main()
