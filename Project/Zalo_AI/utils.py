# utils.py
import os, json, glob, cv2
import numpy as np
from matcher import FrozenEmbedder

# utils.py
import os, json, glob, cv2
import numpy as np

def list_episodes(data_root):
    samples = os.path.join(data_root, "samples")
    return sorted([d for d in os.listdir(samples) if os.path.isdir(os.path.join(samples, d))])

def video_path_for_episode(data_root, vid):
    cand = [
        os.path.join(data_root, "samples", vid, "drone_video.mp4"),
        os.path.join(data_root, "samples", vid, "video.mp4")
    ]
    for p in cand:
        if os.path.isfile(p): return p
    raise FileNotFoundError(f"video not found for {vid}")

def load_refs_for_episode(data_root, vid):
    """
    Look for samples/<vid>/object_images/*.jpg first,
    else any images inside samples/<vid>,
    else extract ~3 frames from the video as references.
    Returns a list of RGB numpy arrays.
    """
    base = os.path.join(data_root, "samples", vid)
    refdir = os.path.join(base, "object_images")  # << correct folder name
    exts = ("*.jpg","*.jpeg","*.png","*.bmp")

    # 1) object_images/
    paths = []
    if os.path.isdir(refdir):
        for e in exts:
            paths.extend(sorted(glob.glob(os.path.join(refdir, e))))
    if len(paths) >= 1:
        arrs = []
        for p in paths[:3]:
            im = cv2.imread(p)
            if im is not None:
                arrs.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        if len(arrs) > 0:
            return arrs

    # 2) any images directly under the video folder
    img_candidates = []
    for e in exts:
        img_candidates.extend(sorted(glob.glob(os.path.join(base, e))))
    if len(img_candidates) >= 1:
        arrs = []
        for p in img_candidates[:3]:
            im = cv2.imread(p)
            if im is not None:
                arrs.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        if len(arrs) > 0:
            return arrs

    # 3) fallback: extract a few frames from the video
    vpath = video_path_for_episode(data_root, vid)
    cap = cv2.VideoCapture(vpath)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if nframes <= 0:
        cap.release()
        raise FileNotFoundError(f"no object_images and cannot read video {vpath}")

    picks = sorted({max(1, int(nframes*0.25)), max(1, int(nframes*0.50)), max(1, int(nframes*0.75))})
    arrs = []
    for f in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f-1)
        ok, fr = cap.read()
        if ok and fr is not None:
            arrs.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(arrs) == 0:
        raise FileNotFoundError(f"no object_images and failed to extract frames from {vpath}")
    return arrs

def _central_annotations_path(data_root):
    return os.path.join(data_root, "annotations", "annotations.json")

def load_annotations(data_root, vid):
    """
    Read central JSON: train/annotations/annotations.json
    Expected formats (seen in your screenshots):
      [
        {
          "video_id": "Backpack_0",
          "annotations": [
            {"bboxes":[ {"frame":370,"x1":..,"y1":..,"x2":..,"y2":..}, ... ]},
            {"bboxes":[ ... ]},
            ...
          ]
        },
        ...
      ]
    Returns: dict frame_idx -> list of [x1,y1,x2,y2]
    """
    path = _central_annotations_path(data_root)
    if not os.path.isfile(path): 
        return None
    with open(path, "r", encoding="utf-8") as f:
        js = json.load(f)

    # find this video
    rec = None
    for item in js:
        if str(item.get("video_id","")) == str(vid):
            rec = item; break
    if rec is None:
        return {}

    by_frame = {}
    for seg in rec.get("annotations", []):
        for bb in seg.get("bboxes", []):
            f = int(bb.get("frame", 0))
            x1 = int(bb.get("x1", 0)); y1 = int(bb.get("y1", 0))
            x2 = int(bb.get("x2", 0)); y2 = int(bb.get("y2", 0))
            if x2 > x1 and y2 > y1:
                by_frame.setdefault(f, []).append([x1,y1,x2,y2])
    return by_frame

def iou_xyxy(a, b):
    x1=max(a[0], b[0]); y1=max(a[1], b[1])
    x2=min(a[2], b[2]); y2=min(a[3], b[3])
    inter=max(0, x2-x1)*max(0, y2-y1)
    aa=(a[2]-a[0])*(a[3]-a[1])
    bb=(b[2]-b[0])*(b[3]-b[1])
    den=aa+bb-inter
    return inter/den if den>0 else 0.0


def build_template(matcher: FrozenEmbedder, refs, device="cuda", augs_per_ref=12, debug=False):
    # simple jitter augmentations for template robustness
    import random
    import numpy as np
    auged=[]
    for r in refs:
        h,w = r.shape[:2]
        for _ in range(augs_per_ref):
            sx = random.uniform(0.85, 1.0)
            sy = random.uniform(0.85, 1.0)
            dx = int((1-sx)*w*random.uniform(0,1))
            dy = int((1-sy)*h*random.uniform(0,1))
            x1=dx; y1=dy; x2=min(w, int(dx+sx*w)); y2=min(h, int(dy+sy*h))
            crop = r[y1:y2, x1:x2].copy()
            auged.append(crop)
    embs = matcher.encode_np(auged, device=device)
    if embs.shape[0] == 0:
        return np.zeros((1,512), dtype=np.float32)
    proto = embs.mean(axis=0, keepdims=True)
    # L2 norm
    proto = proto / (np.linalg.norm(proto, axis=-1, keepdims=True) + 1e-8)
    if debug: print(f"[TEMPLATE] augs={len(auged)} kept={embs.shape[0]} (edge var median≈)")
    return proto

def list_episode_dirs(data_root):
    """Alias for compatibility with inference.py"""
    return list_episodes(data_root)

def ensure_dir(path):
    """mkdir -p for a file's parent or a folder path"""
    if path is None or path == "":
        return
    import os
    os.makedirs(path, exist_ok=True)