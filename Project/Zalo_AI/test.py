import json, argparse
import numpy as np
import torch
import cv2, math, time
from pathlib import Path

from common import (
    find_episodes, load_refs_for_episode, video_path_for_episode,
    YOLOProposals, EmbeddingMatcher, build_template, nms_xyxy,
    TemporalHead, IoUHead, SingleTargetTracker, segmentize
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="./viz_test/predictions.json")
    ap.add_argument("--yolo_ckpt", type=str, default=None)
    ap.add_argument("--yolo_off", action='store_true')
    ap.add_argument("--debug", action='store_true')
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.config,"r",encoding="utf-8") as f: C = json.load(f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    props = YOLOProposals(conf=C["conf"], iou=C["nms_iou"], imgsz=C["imgsz"],
                          max_candidates=C["max_props"], device=0,
                          yolo_on=not args.yolo_off, yolo_ckpt=args.yolo_ckpt, debug=args.debug)
    matcher = EmbeddingMatcher(out_dim=256).to(device).eval()
    temp_head = TemporalHead().to(device).eval()
    iou_head  = IoUHead().to(device).eval()
    sd = torch.load(args.ckpt, map_location=device)
    temp_head.load_state_dict(sd["temporal_head"]) ; iou_head.load_state_dict(sd["iou_head"])

    vids = find_episodes(args.data_root)
    preds=[]
    for vid in vids:
        print(f"[RUN] {vid}")
        refs = load_refs_for_episode(args.data_root, vid)
        tmpl = build_template(matcher, refs, device=device, augs_per_ref=12, use_adapter=True, debug=args.debug)
        cap = cv2.VideoCapture(video_path_for_episode(args.data_root, vid))
        tracker = SingleTargetTracker(
                tau_high=C["tau_high"], tau_low=C["tau_low"],
                assoc_lambda=C["assoc_lambda"],
                max_lost=int(C.get("max_lost", max(10, 3*C["frame_stride"]))),
                min_commit=int(C.get("min_commit", 2)),
                gap_fill=max(1, C["frame_stride"]-1),
                frame_stride=C["frame_stride"],
                debug=args.debug
            )
        seq_buf=[]; last_geom=None; fidx=0
        while True:
            ok, frame = cap.read()
            if not ok: break
            fidx += 1  # 1-based
            if C["frame_stride"]>1 and ((fidx-1) % C["frame_stride"])!=0:
                continue
            t0=time.time(); boxes = props(frame); t_prop=time.time()-t0
            # ---- Debug: YOLO proposal confidences (if available) ----
            yolo_confs = getattr(props, "last_scores", None)
            if args.debug and yolo_confs is not None and len(yolo_confs) == len(boxes):
                # After NMS and keep-indexing, print top few base confidences too
                # We'll align them AFTER you compute 'keep' so they match 'boxes' order:
                pass
            # --------------------------------
            crops = [frame[y1:y2, x1:x2] for (x1,y1,x2,y2) in boxes]
            t1=time.time(); embs  = matcher.encode_np(crops, device); t_emb=time.time()-t1
            cos   = (embs @ tmpl.T).squeeze(1).detach().cpu().numpy() if embs.numel()>0 else np.zeros((0,),dtype=np.float32)
            H,W = frame.shape[:2]; feat_rows=[]
            for b,s in zip(boxes, cos):
                x1,y1,x2,y2=b
                cx=(x1+x2)/2; cy=(y1+y2)/2; w=max(1.0,x2-x1); h=max(1.0,y2-y1)
                if last_geom is None:
                    dx=dy=ds=dh=dw=0.0
                else:
                    lcx,lcy,lw,lh = last_geom
                    dx=(cx-lcx)/max(1.0,W); dy=(cy-lcy)/max(1.0,H)
                    ds=math.log(w/max(1.0,lw)); dh=math.log(h/max(1.0,lh))
                    dw=math.log((w/h)/max(1e-6,lw/lh))
                feat_rows.append([float(s),dx,dy,ds,dh,dw])
            keep = nms_xyxy(boxes, cos, iou_thr=C["nms_final_iou"]) 
            boxes = [boxes[i] for i in keep]; sims=[float(cos[i]) for i in keep]
            
            # ---- Debug: YOLO confidences aligned with kept boxes ----
            if args.debug and yolo_confs is not None and len(yolo_confs) > 0:
                # If props() concatenates YOLO boxes + fallback grid, make sure
                # last_scores only correspond to the YOLO subset. If you stored
                # scores only for YOLO, guard by min length:
                if len(yolo_confs) == len(keep):              # simplest case
                    kept_confs = [float(yolo_confs[i]) for i in keep]
                else:
                    # mixed YOLO + fallback: map only indices < len(yolo_confs)
                    kept_confs = [float(yolo_confs[i]) for i in keep if i < len(yolo_confs)]

                if kept_confs:
                    top = max(kept_confs)
                    print(f"[YOLO] {vid} f={fidx} kept={len(kept_confs)} top_conf={top:.3f}")
            # -----------------------------------------------
            
            feats=[feat_rows[i] for i in keep]
            if feats:
                seq_buf.append(feats[0])
                if len(seq_buf)>C["temporal_T"]: seq_buf.pop(0)
                seq = torch.tensor(seq_buf, dtype=torch.float32, device=device).unsqueeze(0).repeat(len(feats),1,1)
                s_temp = temp_head(seq).squeeze(-1).detach().cpu().numpy()
                s_iou  = iou_head(torch.tensor(feats, dtype=torch.float32, device=device)).squeeze(-1).detach().cpu().numpy()
                fused = (C["alpha_fuse"]*np.array(sims) + (1-C["alpha_fuse"]) * s_temp) * 0.5 + 0.5*s_iou
            else:
                fused=[]
            
            # ---- Debug: top-K breakdown ----
            if args.debug and len(boxes) > 0:
                K = min(5, len(boxes))
                # pack rows: (fused, cosine, temp, iou_head, (w,h), (x1,y1,x2,y2))
                rows = []
                for (x1,y1,x2,y2), c, t, q, f in zip(boxes, sims, s_temp if len(s_temp) else [0]*len(sims),
                                                    s_iou if len(s_iou) else [1]*len(sims),
                                                    fused if isinstance(fused, (list, np.ndarray)) else []):
                    w = max(1, x2-x1); h = max(1, y2-y1)
                    rows.append((float(f), float(c), float(t), float(q), (w,h), (x1,y1,x2,y2)))
                rows.sort(key=lambda r: r[0], reverse=True)
                # compute margin between best and second-best fused
                margin = (rows[0][0] - rows[1][0]) if len(rows) > 1 else 1.0
                print(f"[TOPK] {vid} f={fidx} K={len(rows)} margin={margin:.3f}")
                for i, (F, Cc, Tt, Qq, wh, bb) in enumerate(rows[:K]):
                    print(f"  #{i+1}: fused={F:+.3f}  cos={Cc:+.3f}  temp={Tt:+.3f}  qIoU={Qq:+.3f}  wh={wh}  box={bb}")
            # --------------------------------    
                
            # ---- Tracker debug (every 20th kept frame) ----
            if isinstance(fused, list):
                top_score = -1.0 if len(fused) == 0 else float(np.max(fused))
            else:
                top_score = -1.0 if (fused is None or len(fused) == 0) else float(np.max(fused))
            if args.debug and (fidx % (20 * max(1, C["frame_stride"])) == 0):
                state = getattr(tracker, "state", "NA")
                curr_len = getattr(tracker, "curr_len", 0)
                print(
                    f"[TRK] vid={vid} f={fidx} top_s={top_score:.3f} "
                    f"state={state} tauH={C['tau_high']:.2f} tauL={C['tau_low']:.2f} "
                    f"len={curr_len}"
                )
            # -----------------------------------------------
            tracker.update(fidx, boxes, fused)
            if boxes:
                bi = int(np.argmax(fused)); x1,y1,x2,y2 = boxes[bi]
                last_geom=((x1+x2)/2,(y1+y2)/2,max(1.0,x2-x1),max(1.0,y2-y1))
            if args.debug and (fidx % (5*C["frame_stride"]) == 1):
                print(f"[DBG] {vid} f={fidx} props={len(boxes)} t_prop={t_prop:.3f}s t_emb={t_emb:.3f}s")
        cap.release()
        segs = segmentize(tracker.detections, max_gap=C["frame_stride"]) 
        preds.append({"video_id": vid, "detections": segs})
        out_path = Path(args.out); tmp = out_path.with_suffix(".tmp.json")
        cur = []
        if out_path.exists():
            try:
                with open(out_path,"r",encoding="utf-8") as f: cur=json.load(f)
            except Exception: cur=[]
        cur = [e for e in cur if e.get("video_id") != vid]
        cur.append({"video_id": vid, "detections": segs})
        with open(tmp,"w",encoding="utf-8") as f: json.dump(cur, f, indent=2)
        tmp.replace(out_path)
        print(f"[DONE] {vid}: {sum(len(s['bboxes']) for s in segs)} boxes, {len(segs)} segments")
    with open(args.out,"w",encoding="utf-8") as f: json.dump(preds, f, indent=2)
    print(f"[SAVE] {args.out}")

if __name__ == "__main__":
    main()
