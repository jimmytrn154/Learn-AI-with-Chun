# common.py
import numpy as np

def nms_xyxy(boxes, scores, iou_thr=0.5):
    order = np.argsort(-scores)
    keep = []
    for i in order:
        bi = boxes[i]
        ok = True
        for j in keep:
            bj = boxes[j]
            xx1 = max(bi[0], bj[0]); yy1 = max(bi[1], bj[1])
            xx2 = min(bi[2], bj[2]); yy2 = min(bi[3], bj[3])
            inter = max(0, xx2-xx1)*max(0, yy2-yy1)
            ai = (bi[2]-bi[0])*(bi[3]-bi[1])
            aj = (bj[2]-bj[0])*(bj[3]-bj[1])
            den = ai + aj - inter
            iou = inter/den if den>0 else 0.0
            if iou > iou_thr:
                ok = False; break
        if ok: keep.append(i)
    return keep

class SingleTargetTracker:
    def __init__(self, tau_high, tau_low, assoc_lambda=0.5,
                 max_lost=25, min_commit=1, gap_fill=1, frame_stride=2, debug=False):
        self.tau_high = tau_high
        self.tau_low  = tau_low
        self.assoc_lambda = assoc_lambda
        self.max_lost = max_lost
        self.min_commit = min_commit
        self.gap_fill = gap_fill
        self.frame_stride = frame_stride
        self.debug = debug
        self.reset()

    def reset(self):
        self.state = "IDLE"        # IDLE | TENT | LIVE
        self.last_box = None
        self.last_score = 0.0
        self.curr = []
        self.curr_len = 0
        self.detections = []
        self.lost = 0
        self.just_committed = False
        self.last_segment_meta = None

    def update(self, fidx, boxes, scores, tau_high=None, tau_low=None):
        self.just_committed = False
        if tau_high is None: tau_high = self.tau_high
        if tau_low  is None: tau_low  = self.tau_low

        if len(boxes) == 0:
            self._step_none(fidx)
            return

        i = int(np.argmax(scores))
        b = boxes[i]; s = float(scores[i])

        if self.state in ("IDLE","TENT"):
            if s >= tau_high:
                self.state = "LIVE"
                self.curr = [{"frame": fidx, "x1":int(b[0]), "y1":int(b[1]), "x2":int(b[2]), "y2":int(b[3])}]
                self.curr_len = 1
                self.last_box = b; self.last_score = s; self.lost = 0
            else:
                self.state = "TENT"
                self.last_box = b; self.last_score = s
                self.lost += 1
        else:
            if s >= tau_low:
                self.curr.append({"frame": fidx, "x1":int(b[0]), "y1":int(b[1]), "x2":int(b[2]), "y2":int(b[3])})
                self.curr_len += 1
                self.last_box = b; self.last_score = s; self.lost = 0
            else:
                self.lost += 1
                if self.lost > self.max_lost:
                    self._commit_current(); self.state = "IDLE"
                    self.last_box = None; self.last_score = 0.0; self.lost = 0

    def _step_none(self, fidx):
        if self.state == "LIVE":
            self.lost += 1
            if self.lost > self.max_lost:
                self._commit_current(); self.state = "IDLE"
                self.last_box = None; self.last_score = 0.0; self.lost = 0
        elif self.state == "TENT":
            self.lost += 1
            if self.lost > self.max_lost:
                self.state = "IDLE"
                self.last_box = None; self.last_score = 0.0; self.lost = 0

    def _commit_current(self):
        if self.curr_len >= self.min_commit:
            self.detections.extend(self.curr)
            self.just_committed = True
            self.last_segment_meta = {"len": self.curr_len, "score": self.last_score}
        self.curr = []; self.curr_len = 0

    def flush(self):
        seg = self.curr[:] if self.curr_len >= self.min_commit else None
        self.curr = []; self.curr_len = 0; self.state = "IDLE"
        return seg
