# proposals.py
import numpy as np
from ultralytics import YOLO

class YOLOProposals:
    def __init__(self, yolo_ckpt=None, imgsz=1280, conf=0.01, nms_iou=0.60, max_props=80, debug=False):
        self.model = YOLO(yolo_ckpt or "yolov8n.pt")
        self.imgsz = imgsz
        self.conf = conf
        self.nms_iou = nms_iou
        self.max_props = max_props
        self.debug = debug
        self.last_scores = None

    def __call__(self, frame):
        res = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, iou=self.nms_iou,
                                 max_det=self.max_props, verbose=False)[0]
        # raw = res[0].boxes  # ultralytics Boxes
        # raw_conf = [] if raw is None else raw.conf.detach().cpu().numpy().tolist()
        # raw_xyxy = [] if raw is None else raw.xyxy.detach().cpu().numpy().tolist()
        # rawK = 0 if raw is None else len(raw_xyxy)
        # if self.debug:
        #     # print raw K and few confidences BEFORE your own filtering/NMS
        #     print(f"[YOLO] rawK={rawK} raw_conf(sample)={raw_conf[:5]}")
        #     print(f"[YOLO] preds={len(boxes)} (returned after conf/NMS/topK), conf_thr={self.conf}, nms_iou={self.nms_iou}, max_props={self.max_props}")
        
        boxes, scores = [], []
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            conf = res.boxes.conf.cpu().numpy()
            order = np.argsort(-conf)
            xyxy = xyxy[order][:self.max_props]
            conf = conf[order][:self.max_props]
            for (x1,y1,x2,y2), c in zip(xyxy, conf):
                boxes.append((int(x1),int(y1),int(x2),int(y2)))
                scores.append(float(c))
        self.last_scores = scores

        if len(boxes) == 0:
            # grid fallback
            H, W = frame.shape[:2]
            gx, gy = 10, 8
            gw, gh = W//gx, H//gy
            for j in range(gy):
                for i in range(gx):
                    x1 = i*gw; y1 = j*gh; x2 = min(W, x1+gw); y2 = min(H, y1+gh)
                    boxes.append((x1,y1,x2,y2))
                    if len(boxes) >= self.max_props: break
                if len(boxes) >= self.max_props: break
            if self.debug: print(f"[FALLBACK] props={len(boxes)}")
        else:
            if self.debug: print(f"[YOLO] preds={len(boxes)}")
        return boxes
