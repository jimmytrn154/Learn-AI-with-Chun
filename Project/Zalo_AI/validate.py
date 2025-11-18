# validate.py
import os, json, argparse, cv2, numpy as np
from inference import run_inference_on_split
from utils import load_annotations, iou_xyxy

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--split_file", required=True)  # JSON with {"val":[...]} or {"dev":[...]}
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--frame_stride", type=int, default=2)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--max_props", type=int, default=80)
    ap.add_argument("--conf", type=float, default=0.01)
    ap.add_argument("--nms_iou", type=float, default=0.60)
    ap.add_argument("--nms_final_iou", type=float, default=0.50)
    ap.add_argument("--assoc_lambda", type=float, default=0.50)
    ap.add_argument("--alpha_fuse", type=float, default=0.88)
    ap.add_argument("--alpha_start", type=float, default=0.70)
    ap.add_argument("--temporal_T", type=int, default=15)
    ap.add_argument("--tau_high", type=float, default=0.30)
    ap.add_argument("--tau_low", type=float, default=0.16)
    ap.add_argument("--tau_high_cos", type=float, default=0.24)
    ap.add_argument("--tau_low_cos", type=float, default=0.12)
    ap.add_argument("--min_commit", type=int, default=1)
    ap.add_argument("--max_lost", type=int, default=25)
    ap.add_argument("--beta_iou", type=float, default=0.0)
    ap.add_argument("--margin_gate", type=float, default=0.06)
    ap.add_argument("--roi_pad", type=float, default=0.50)
    ap.add_argument("--yolo_ckpt", default=None)
    ap.add_argument("--debug", action="store_true")
    return ap.parse_args()

def st_iou_for_video(pred_segs, data_root, vid):
    gt_by_f = load_annotations(data_root, vid) or {}
    # build first-box-per-frame map from predictions
    pred_by_f = {}
    for seg in pred_segs:
        for bb in seg.get("bboxes", []):
            f = int(bb["frame"])
            # keep the first box for that frame
            pred_by_f.setdefault(f, (bb["x1"],bb["y1"],bb["x2"],bb["y2"]))

    frames = sorted(set(pred_by_f.keys()) | set(gt_by_f.keys()))
    if not frames: return 0.0
    vals = []
    for f in frames:
        if f in pred_by_f and f in gt_by_f and len(gt_by_f[f])>0:
            # single-target; use the first gt box
            vals.append(iou_xyxy(pred_by_f[f], gt_by_f[f][0]))
        else:
            vals.append(0.0)
    return float(np.mean(vals))

def main():
    args = parse_args()
    with open(args.split_file, "r", encoding="utf-8") as f:
        split = json.load(f)
    val_ids = split.get("val", []) or split.get("dev", [])
    if not val_ids: 
        print("[VAL] no val/dev ids in split file"); return

    cfg = {
        "frame_stride": args.frame_stride,
        "imgsz": args.imgsz,
        "max_props": args.max_props,
        "conf": args.conf,
        "nms_iou": args.nms_iou,
        "nms_final_iou": args.nms_final_iou,
        "assoc_lambda": args.assoc_lambda,
        "alpha_fuse": args.alpha_fuse,
        "alpha_start": args.alpha_start,
        "temporal_T": args.temporal_T,
        "tau_high": args.tau_high,
        "tau_low": args.tau_low,
        "tau_high_cos": args.tau_high_cos,
        "tau_low_cos": args.tau_low_cos,
        "min_commit": args.min_commit,
        "max_lost": args.max_lost,
        "beta_iou": args.beta_iou,
        "margin_gate": args.margin_gate,
        "roi_pad": args.roi_pad,
        "topk_debug": 0
    }

    preds = run_inference_on_split(args.data_root, val_ids, args.ckpt, cfg, args.yolo_ckpt, debug=args.debug)
    # preds: list of {"video_id": vid, "detections":[{"bboxes":[...]}]}
    totals = []
    for entry in preds:
        vid = entry["video_id"]
        ann = load_annotations(args.data_root, vid)
        if ann is None:
            print(f"[VAL] {vid}: no GT, skip"); continue
        st = st_iou_for_video(entry["detections"], args.data_root, vid)
        totals.append(st)
        print(f"[ST-IoU] {vid}: {st:.4f}")
    mean = float(np.mean(totals)) if totals else 0.0
    print(f"[ST-IoU] mean over {len(totals)} videos: {mean:.4f}")

if __name__ == "__main__":
    main()
