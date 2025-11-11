"""
baseline_temporal.py
Tiny-YOLO proposals + ResNet18 (frozen) matcher + Temporal Head (GRU) + IoU Head + optional LTO re-scoring
Outputs predictions.json and ST-IoU. Compatible with your dataset layout.

Run (from inside Zalo_AI/):
CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 python baseline_temporal.py \
  --data_root ./train \
  --out ./viz/predictions.json \
  --eval \
  --frame_stride 2 --imgsz 640 --max_props 150 --conf 0.12 --nms_iou 0.65 \
  --tau_high 0.55 --tau_low 0.45 --lto

Param budget (approx):
- YOLOv8-n ~3.2M
- ResNet18 frozen ~11.7M
- Projection head ~0.13M
- TemporalHead (GRU) ~0.20-0.30M
- IoUHead ~0.01M
Total ~15-16M << 50M
"""

import os, json, math, argparse, random, time
from pathlib import Path
from typing import List, Tuple, Dict, Iterable, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import models, transforms

# Optional YOLO proposals
try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except Exception:
    _HAS_YOLO = False

cudnn.benchmark = True

# ------------------------------
# Utils
# ------------------------------
def set_seed(seed=1337):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

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
    if not entries: return {}
    by_vid = {e.get("video_id"): e for e in entries if "video_id" in e}
    if video_id not in by_vid: return {}
    rec = by_vid[video_id]
    segs = rec.get(key, [])
    if isinstance(segs, dict) and "bboxes" in segs:
        segs = [segs]
    if isinstance(segs, list) and segs and isinstance(segs[0], dict) and "frame" in segs[0]:
        segs = [{"bboxes": segs}]
    out = {}
    for seg in segs:
        for b in seg.get("bboxes", []):
            k = int(b["frame"])
            out.setdefault(k, []).append((int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])))
    return out

# ------------------------------
# Proposals (YOLO nano)
# ------------------------------
class YOLOProposals:
    def __init__(self, conf=0.05, iou=0.7, imgsz=640, max_candidates=200, device=0, use_half=True):
        self.conf = conf; self.iou = iou; self.imgsz = imgsz; self.max_candidates = max_candidates
        self.device = device; self.use_half = use_half and torch.cuda.is_available()
        self.model = None
        if _HAS_YOLO:
            try:
                # Prefer local file if present
                local = Path(__file__).with_name("yolov8n.pt")
                self.model = YOLO(str(local)) if local.exists() else YOLO("yolov8n.pt")
            except Exception:
                self.model = None

    def __call__(self, frame_bgr: np.ndarray) -> List[Tuple[int,int,int,int]]:
        H, W = frame_bgr.shape[:2]
        if self.model is None:
            # Sliding windows fallback
            sizes = [64, 96, 128, 160, 192]
            stride = 0.5
            props = []
            for s in sizes:
                sx = max(16, int(s)); sy = max(16, int(s))
                step_x = max(8, int(sx*(1-stride))); step_y = max(8, int(sy*(1-stride)))
                for y in range(0, H - sy, step_y):
                    for x in range(0, W - sx, step_x):
                        props.append((x, y, x+sx, y+sy))
                        if len(props) >= self.max_candidates: return props
            return props[: self.max_candidates]
        res = self.model.predict(
            frame_bgr, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
            device=self.device, half=self.use_half, verbose=False
        )[0]
        boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0,4))
        boxes = boxes[: self.max_candidates]
        out = []
        for x1,y1,x2,y2 in boxes:
            x1 = int(max(0, min(W-1, x1))); y1 = int(max(0, min(H-1, y1)))
            x2 = int(max(0, min(W-1, x2))); y2 = int(max(0, min(H-1, y2)))
            if x2 > x1 and y2 > y1:
                out.append((x1,y1,x2,y2))
        return out

# ------------------------------
# Embedding matcher (ResNet18 + head)
# ------------------------------
class EmbeddingMatcher(nn.Module):
    """
    ResNet-18 backbone (frozen) + projection head 512->256 with L2 normalize.
    """
    def __init__(self, out_dim=256, use_half=True):
        super().__init__()
        rn18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        for p in rn18.parameters():
            p.requires_grad_(False)
        self.backbone = nn.Sequential(*list(rn18.children())[:-1])  # (B,512,1,1)
        self.head = nn.Linear(512, out_dim, bias=False)
        self.prep = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        # self.use_half = use_half and torch.cuda.is_available()
        self.use_half = False

    @torch.no_grad()
    def encode_np(self, np_imgs: List[np.ndarray]) -> torch.Tensor:
        if len(np_imgs) == 0:
            return torch.empty(0, 256, device=next(self.head.parameters()).device)
        device = next(self.head.parameters()).device
        tensors = []
        for img in np_imgs:
            if img.shape[-1] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensors.append(self.prep(img))
        batch = torch.stack(tensors, dim=0).to(device, non_blocking=True)
        # if self.use_half:
        #     batch = batch.half()
        # else:
        #     batch = batch.float()
        batch = batch.float()    
        feats = self.backbone(batch).flatten(1)         # (B,512)
        proj  = self.head(feats)                        # (B,256)
        proj  = l2_normalize(proj, dim=1)
        return proj

def build_template(matcher: EmbeddingMatcher, ref_imgs: List[np.ndarray], augs_per_ref=8) -> torch.Tensor:
    """Augment refs, embed, average -> template vector (1, D)."""
    auged = []
    for im in ref_imgs:
        h,w = im.shape[:2]
        for _ in range(augs_per_ref):
            scale = np.random.uniform(0.85, 1.15)
            nh, nw = max(16, int(h*scale)), max(16, int(w*scale))
            resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR)
            if np.random.rand() < 0.25:
                k = np.random.choice([3,5])
                resized = cv2.GaussianBlur(resized, (k,k), 0)
            auged.append(resized)
    with torch.no_grad():
        embs = matcher.encode_np(auged)  # (N, D)
        tmpl = embs.mean(0, keepdim=True)  # (1, D)
        tmpl = l2_normalize(tmpl, dim=1)
    return tmpl  # (1, D)

# ------------------------------
# Temporal & IoU heads
# ------------------------------
class TemporalHead(nn.Module):
    """
    Tiny GRU temporal head.
    Input per frame: [cosine_score (1), dx, dy, ds, dh, dw, (optional) reduced embed dims]
    We keep it minimal: 1 + 5 geometry = 6 dims -> project to 32 -> GRU(64) -> MLP -> sigmoid score in [0,1].
    """
    def __init__(self, in_dim=6, proj_dim=32, hidden=64, out_dim=1):
        super().__init__()
        self.proj = nn.Linear(in_dim, proj_dim)
        self.gru  = nn.GRU(input_size=proj_dim, hidden_size=hidden, num_layers=1, batch_first=True)
        self.mlp  = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
            nn.Sigmoid()
        )

    def forward(self, seq_feats: torch.Tensor) -> torch.Tensor:
        """
        seq_feats: (B, T, in_dim)
        returns: (B, 1) score in [0,1] for the last step (we use final hidden)
        """
        x = self.proj(seq_feats)  # (B,T,proj_dim)
        out, h = self.gru(x)      # h: (1,B,hidden)
        last = h[-1]              # (B,hidden)
        s = self.mlp(last)        # (B,1)
        return s

class IoUHead(nn.Module):
    """
    Small MLP to predict IoU quality from [cosine_score, box geometry features].
    """
    def __init__(self, in_dim=6, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ------------------------------
# Tracker with hysteresis (unchanged core)
# ------------------------------
class SingleTargetTracker:
    def __init__(self,
                 tau_high=0.55, tau_low=0.45,
                 assoc_lambda=0.5,
                 search_roi_pad=0.35,
                 max_lost=10,
                 min_commit=3,
                 gap_fill=1):
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
        self.buffer = []
        self.current_segment = []
        self.detections = []
        self.prev_frame = None

    def _associate(self, boxes, sims):
        if not boxes:
            return None, None
        if self.last_box is None or self.lost >= self.max_lost:
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

    def update(self, frame_idx: int, candidate_boxes: List[Tuple[int,int,int,int]], fused_scores: List[float]):
        chosen, sim = self._associate(candidate_boxes, fused_scores) if candidate_boxes else (None, None)
        if chosen is not None:
            if not self.active:
                if sim >= self.tau_high:
                    self.buffer.append({"frame": int(frame_idx), "x1": int(chosen[0]), "y1": int(chosen[1]), "x2": int(chosen[2]), "y2": int(chosen[3])})
                    self._commit_buffer_if_ready()
            else:
                if sim >= self.tau_low:
                    # handle small gaps
                    if self.prev_frame is not None and frame_idx - self.prev_frame > 1 and (frame_idx - self.prev_frame - 1) <= self.gap_fill and len(self.current_segment)>0:
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
                    self._flush_segment()
            self.last_box = chosen
            self.lost = 0
        else:
            self.lost += 1
            if self.lost >= self.max_lost:
                self._flush_segment()
        self.prev_frame = frame_idx

    def finalize(self) -> List[dict]:
        self.buffer = []
        return self.detections

# ------------------------------
# ST-IoU evaluator
# ------------------------------
def st_iou_one(gt_entries, pred_entries, video_id: str) -> float:
    gt_map   = frame_to_boxes(gt_entries,   video_id, key="annotations")
    pred_map = frame_to_boxes(pred_entries, video_id, key="detections")
    inter = sorted(set(gt_map.keys()).intersection(pred_map.keys()))
    union = sorted(set(gt_map.keys()).union(pred_map.keys()))
    if not union: return 0.0
    num = 0.0
    for f in inter:
        best = 0.0
        for g in gt_map[f]:
            for p in pred_map[f]:
                best = max(best, iou_xyxy(g,p))
        num += best
    den = float(len(union))
    return num / den

def st_iou_mean(gt_entries, pred_entries, video_ids: Iterable[str]) -> float:
    vals = [st_iou_one(gt_entries, pred_entries, vid) for vid in video_ids]
    return float(np.mean(vals)) if vals else 0.0

# ------------------------------
# LTO-style tube re-scoring (simple)
# ------------------------------
def lto_rescore(entry: dict, top_percent: float = 0.2) -> dict:
    """
    After first pass, boost scores by tube consistency.
    Here we don't store per-frame scores in JSON; we approximate by smoothing segments:
    - For each segment, compute the mean of top-P% fused scores we recorded in-memory (passed via side dict).
    - Since JSON doesn't carry scores, this function is a placeholder to show where you'd rescore
      if you store scores alongside boxes. Kept here to match the paper's spirit.
    """
    # In this minimal JSON-only version we skip actual mutation as we didn't store scores in entry.
    # Hook provided for future extension (store per-box score in det dict and adjust here).
    return entry

# ------------------------------
# Predictor
# ------------------------------
class Predictor:
    def __init__(self,
                 conf=0.12, nms_iou=0.65, imgsz=640, max_props=150,
                 tau_high=0.55, tau_low=0.45, assoc_lambda=0.5,
                 roi_pad=0.35, max_lost=10, min_commit=3, gap_fill=1,
                 frame_stride=2, nms_final_iou=0.5, device=0, use_half=True,
                 temporal_T=15, alpha_fuse=0.5, use_lto=False):
        self.props = YOLOProposals(conf=conf, iou=nms_iou, imgsz=imgsz, max_candidates=max_props, device=device, use_half=use_half)
        self.matcher = EmbeddingMatcher(out_dim=256, use_half=use_half).eval().to(device)
        self.use_half = use_half and torch.cuda.is_available()
        self.tau_high = tau_high; self.tau_low = tau_low
        self.assoc_lambda = assoc_lambda
        self.roi_pad = roi_pad; self.max_lost = max_lost
        self.min_commit = min_commit; self.gap_fill = gap_fill
        self.frame_stride = frame_stride
        self.nms_final_iou = nms_final_iou
        self.temporal_T = temporal_T
        self.alpha_fuse = alpha_fuse
        self.use_lto = use_lto

        # heads
        self.temporal_head = TemporalHead(in_dim=6, proj_dim=32, hidden=64, out_dim=1).to(device).eval()
        self.iou_head      = IoUHead(in_dim=6, hidden=64).to(device).eval()

        # crop prep
        self.crop_prep = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        self.device = device

    @torch.no_grad()
    def _embed_crops(self, crops: List[np.ndarray]) -> torch.Tensor:
        if len(crops) == 0:
            return torch.empty(0, 256, device=self.device)
        tensors = []
        for c in crops:
            if c is None or c.size == 0: continue
            img = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
            tensors.append(self.crop_prep(img))
        if not tensors:
            return torch.empty(0, 256, device=self.device)
        batch = torch.stack(tensors, dim=0).to(self.device, non_blocking=True)
        # if self.use_half:
        #     batch = batch.half()
        # else:
        #     batch = batch.float()
        batch = batch.float()
        batch = batch.to(self.device, non_blocking=True).float()
        feats = self.matcher.backbone(batch).flatten(1)
        proj  = l2_normalize(self.matcher.head(feats), dim=1)
        return proj

    @torch.no_grad()
    def _cosine_scores(self, embs: torch.Tensor, tmpl: torch.Tensor) -> np.ndarray:
        # if embs.numel() == 0:
        #     return np.zeros((0,), dtype=np.float32)
        if embs.numel() == 0:
            return np.zeros((0,), dtype=np.float32)
        tmpl = tmpl.to(embs.dtype)                                     # <- match dtype
        sims = F.linear(embs, tmpl)  # (N,1)
        return sims.squeeze(-1).detach().float().cpu().numpy()

    def _segmentize(self, dets: List[dict], max_gap=1) -> List[dict]:
        dets = sorted(dets, key=lambda d: d["frame"])
        segments, cur = [], []
        prev = None
        for d in dets:
            if prev is not None and d["frame"] - prev["frame"] > max_gap:
                if cur: segments.append({"bboxes": cur}); cur = []
            cur.append(d); prev = d
        if cur: segments.append({"bboxes": cur})
        return segments

    def run_episode(self, data_root: str, video_id: str) -> dict:
        print(f"[{video_id}] building template...")
        refs = load_refs_for_episode(data_root, video_id)
        
        template = build_template(self.matcher, refs, augs_per_ref=8)  # (1,256)
        template = template.to(self.device, dtype=torch.float32) 
        
        if self.use_half:
            template = template.half().to(self.device)
        else:
            template = template.float().to(self.device)
        print(f"[{video_id}] template ready.")

        vpath = video_path_for_episode(data_root, video_id)
        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {vpath}")

        tracker = SingleTargetTracker(
            tau_high=self.tau_high, tau_low=self.tau_low, assoc_lambda=self.assoc_lambda,
            search_roi_pad=self.roi_pad, max_lost=self.max_lost,
            min_commit=self.min_commit, gap_fill=self.gap_fill
        )

        # ring buffer for temporal head inputs
        seq_feats: List[List[float]] = []

        # last box for geometry deltas (cx, cy, w, h)
        last_geom: Optional[Tuple[float,float,float,float]] = None

        frame_idx = 0
        dets_with_score = []  # if you want to store per-box scores for LTO
        while True:
            ok, frame = cap.read()
            if not ok: break
            if self.frame_stride > 1 and (frame_idx % self.frame_stride) != 0:
                frame_idx += 1
                continue
            H, W = frame.shape[:2]

            # proposals
            props = self.props(frame)  # list of (x1,y1,x2,y2)
            crops = [frame[y1:y2, x1:x2] for (x1,y1,x2,y2) in props]

            # match with template
            embs  = self._embed_crops(crops)
            cos_sims  = self._cosine_scores(embs, template)

            # geometry features for temporal/IoU heads
            # For each proposal, compute normalized deltas vs last box geometry; if none, set zeros.
            def geom_feats(box):
                x1,y1,x2,y2 = box
                cx = (x1 + x2) / 2.0; cy = (y1 + y2) / 2.0
                w  = max(1.0, x2 - x1); h  = max(1.0, y2 - y1)
                if last_geom is None:
                    dx=dy=ds=dh=dw=0.0
                else:
                    lcx,lcy,lw,lh = last_geom
                    dx = (cx - lcx) / max(1.0, W)
                    dy = (cy - lcy) / max(1.0, H)
                    ds = math.log(w / max(1.0, lw))
                    dh = math.log(h / max(1.0, lh))
                    dw = math.log((w/h) / max(1e-6, lw/lh))
                return [dx,dy,ds,dh,dw], (cx,cy,w,h)

            geom_list = []
            feat_rows = []
            for b, s in zip(props, cos_sims):
                g, _geom_abs = geom_feats(b)
                # temporal head input per candidate at this frame is built from a sequence,
                # but we need a single fused score now, so we append this step into the ring.
                curr = [float(s)] + g  # [cos, dx,dy,ds,dh,dw] -> 6 dims
                feat_rows.append(curr)
                geom_list.append(_geom_abs)

            # final NMS on cosine first (reduce cost)
            keep_idx = nms_xyxy(props, cos_sims, iou_thr=self.nms_final_iou)
            props_kept = [props[i] for i in keep_idx]
            sims_kept  = [float(cos_sims[i]) for i in keep_idx]
            feats_kept = [feat_rows[i] for i in keep_idx]

            # Build temporal sequences for kept candidates:
            # Append current-step features into ring buffer (global), then for each candidate we
            # use the last T steps (per-frame we reuse the same ring; it's a cheap approximation).
            # For stronger modeling, you'd track per-candidate sequences, but that adds complexity.
            seq_feats.append([0.0]*6 if len(feats_kept)==0 else feats_kept[0])  # use top-1 as the video-level cue
            if len(seq_feats) > self.temporal_T:
                seq_feats.pop(0)
            # Prepare (B=K, T, 6) by repeating the shared sequence for each kept candidate
            if len(props_kept) > 0 and len(seq_feats) > 0:
                # seq = torch.tensor(seq_feats, device=self.device, dtype=torch.float16 if self.use_half else torch.float32)  # (T,6)
                seq = torch.tensor(seq_feats, device=self.device, dtype=torch.float32)
                seq = seq.unsqueeze(0).repeat(len(props_kept), 1, 1)  # (K,T,6)
                with torch.no_grad():
                    s_temp = self.temporal_head(seq).squeeze(-1).detach().float().cpu().numpy()  # (K,)
            else:
                s_temp = np.zeros((len(props_kept),), dtype=np.float32)

            # IoU head (quality) using the *current* candidate features
            if len(props_kept) > 0:
                # x_iou = torch.tensor(feats_kept, device=self.device, dtype=torch.float16 if self.use_half else torch.float32)
                x_iou = torch.tensor(feats_kept, device=self.device, dtype=torch.float32)
                with torch.no_grad():
                    s_iou = self.iou_head(x_iou).squeeze(-1).detach().float().cpu().numpy()
            else:
                s_iou = np.zeros((0,), dtype=np.float32)

            # Fuse scores: cosine vs temporal (alpha_fuse) and multiply IoU quality as a weight
            fused = []
            for c, t, q in zip(sims_kept, s_temp, s_iou):
                s = self.alpha_fuse * c + (1.0 - self.alpha_fuse) * float(t)
                s = float(s) * float(q)  # quality-aware
                fused.append(s)

            # Update tracker
            tracker.update(frame_idx, props_kept, fused)

            # Update last_geom from chosen top-1 (if any)
            if len(props_kept) > 0:
                # approximate last as best fused
                bi = int(np.argmax(fused))
                _, (cx,cy,w,h) = geom_feats(props_kept[bi])
                last_geom = (cx,cy,w,h)

            if frame_idx % 100 == 0 and frame_idx > 0:
                print(f"[{video_id}] frame {frame_idx}")

            frame_idx += 1

        cap.release()
        dets = tracker.finalize()
        segments = self._segmentize(dets, max_gap=1)
        entry = {"video_id": video_id, "detections": segments}

        if self.use_lto:
            entry = lto_rescore(entry)

        return entry

# ------------------------------
# CLI / main
# ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Path containing samples/ and annotations/")
    parser.add_argument("--out", type=str, default="./viz/predictions.json", help="Output predictions JSON")
    parser.add_argument("--eval", action="store_true", help="Compute ST-IoU vs annotations if available")
    parser.add_argument("--seed", type=int, default=1337)
    # proposals
    parser.add_argument("--conf", type=float, default=0.12)
    parser.add_argument("--nms_iou", type=float, default=0.65)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max_props", type=int, default=150)
    # tracker / matcher thresholds
    parser.add_argument("--tau_high", type=float, default=0.55)
    parser.add_argument("--tau_low", type=float, default=0.45)
    parser.add_argument("--assoc_lambda", type=float, default=0.5)
    parser.add_argument("--roi_pad", type=float, default=0.35)
    parser.add_argument("--max_lost", type=int, default=10)
    parser.add_argument("--min_commit", type=int, default=3)
    parser.add_argument("--gap_fill", type=int, default=1)
    parser.add_argument("--frame_stride", type=int, default=2)
    parser.add_argument("--nms_final_iou", type=float, default=0.5)
    # temporal head
    parser.add_argument("--temporal_T", type=int, default=15)
    parser.add_argument("--alpha_fuse", type=float, default=0.5)
    parser.add_argument("--lto", action="store_true", help="Enable simple LTO tube re-scoring")
    args = parser.parse_args()

    set_seed(args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    device = 0 if torch.cuda.is_available() else "cpu"
    use_half = torch.cuda.is_available()

    predictor = Predictor(
        conf=args.conf, nms_iou=args.nms_iou, imgsz=args.imgsz, max_props=args.max_props,
        tau_high=args.tau_high, tau_low=args.tau_low, assoc_lambda=args.assoc_lambda,
        roi_pad=args.roi_pad, max_lost=args.max_lost, min_commit=args.min_commit, gap_fill=args.gap_fill,
        frame_stride=args.frame_stride, nms_final_iou=args.nms_final_iou,
        device=device, use_half=use_half,
        temporal_T=args.temporal_T, alpha_fuse=args.alpha_fuse, use_lto=args.lto
    )

    video_ids = find_episodes(args.data_root)
    print(f"[INFO] Found {len(video_ids)} episodes:", video_ids)

    preds = []
    out_path = Path(args.out)

    # stream-save after each video (safe to interrupt/restart)
    for vid in video_ids:
        t0 = time.time()
        entry = predictor.run_episode(args.data_root, vid)
        dt = time.time() - t0
        preds.append(entry)

        # incremental write
        if out_path.exists():
            try:
                with open(out_path, "r", encoding="utf-8") as f:
                    cur = json.load(f)
            except Exception:
                cur = []
        else:
            cur = []
        cur = [e for e in cur if e.get("video_id") != entry.get("video_id")]
        cur.append(entry)
        tmp = out_path.with_suffix(".tmp.json")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cur, f, indent=2)
        tmp.replace(out_path)

        n_boxes = sum(len(s["bboxes"]) for s in entry["detections"])
        print(f"[DONE] {vid}: {n_boxes} boxes, {len(entry['detections'])} segments, {dt:.1f}s")

    # final ST-IoU (if requested)
    if args.eval:
        gt_entries = load_annotations_json(args.data_root)
        if gt_entries:
            score = st_iou_mean(gt_entries, preds, video_ids)
            print(f"[ST-IoU] mean over {len(video_ids)} videos: {score:.4f}")
            # per-video breakdown
            for vid in video_ids:
                s = st_iou_one(gt_entries, preds, vid)
                print(f"[ST-IoU] {vid}: {s:.4f}")
        else:
            print("[WARN] annotations/annotations.json not found; skipping ST-IoU.")

if __name__ == "__main__":
    main()
