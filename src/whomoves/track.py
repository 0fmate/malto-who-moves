"""Link detections across frames into tracks (the camera is fixed).

Method, a simplified ByteTrack:
0. Detections below min_det_score are ignored (a low-threshold detection run can be re-tracked at any cut).
1. Per frame, merge duplicate boxes of the same competition class (e.g. "car" and "truck" on the
   same vehicle) with a class-aware non-maximum suppression.
2. Each live track predicts its box in the new frame with a constant-velocity model: the last box
   shifted by an exponentially smoothed per-frame velocity of its center.
3. Hungarian matching between predicted boxes and detections of the same class, in three rounds:
   confident detections by IoU, then low-score detections by IoU (a far-away person often scores
   only 0.25-0.5), then a center-distance round for small or fast objects whose boxes barely overlap.
4. Unmatched detections with enough score start new tracks; tracks unmatched for more than
   max_age_frames frames are closed. Tracks shorter than min_track_len detections are dropped as noise.
The class of a track is the majority class of its detections.
"""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

INF = 1e6


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between (n, 4) and (m, 4) boxes given as x1, y1, x2, y2."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def nms_per_class(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou_thr: float) -> np.ndarray:
    """Indices of the boxes kept after greedy NMS within each class."""
    keep = []
    for c in np.unique(classes):
        idx = np.flatnonzero(classes == c)
        idx = idx[np.argsort(-scores[idx])]
        while len(idx):
            best, rest = idx[0], idx[1:]
            keep.append(best)
            idx = rest[iou_xyxy(boxes[best:best + 1], boxes[rest])[0] < iou_thr]
    return np.array(sorted(keep), dtype=int)


def _assign(cost: np.ndarray, max_cost: float) -> list[tuple[int, int]]:
    """Minimum-cost one-to-one matching, keeping only pairs under max_cost."""
    if cost.size == 0:
        return []
    rows, cols = linear_sum_assignment(cost)
    return [(r, c) for r, c in zip(rows, cols) if cost[r, c] <= max_cost]


class _Track:
    def __init__(self, track_id: int, box: np.ndarray, frame: int, cls: str):
        self.id, self.box, self.last_frame, self.cls = track_id, box, frame, cls
        self.vel = np.zeros(2)  # center velocity, pixels per frame

    def predict(self, frame: int) -> np.ndarray:
        shift = self.vel * (frame - self.last_frame)
        return self.box + np.concatenate([shift, shift])

    def update(self, box: np.ndarray, frame: int, smoothing: float) -> None:
        dt = frame - self.last_frame
        center_shift = (box[:2] + box[2:]) / 2 - (self.box[:2] + self.box[2:]) / 2
        self.vel = smoothing * center_shift / dt + (1 - smoothing) * self.vel
        self.box, self.last_frame = box, frame


def _cost_iou(preds, boxes, track_cls, det_cls):
    cost = 1 - iou_xyxy(preds, boxes)
    cost[track_cls[:, None] != det_cls[None, :]] = INF
    return cost


def _cost_center(preds, boxes, track_cls, det_cls):
    """Center distance in units of box height; also requires similar heights."""
    pc, dc = (preds[:, :2] + preds[:, 2:]) / 2, (boxes[:, :2] + boxes[:, 2:]) / 2
    ph, dh = preds[:, 3] - preds[:, 1], boxes[:, 3] - boxes[:, 1]
    scale = np.maximum(np.maximum(ph[:, None], dh[None, :]), 1.0)
    cost = np.linalg.norm(pc[:, None, :] - dc[None, :, :], axis=2) / scale
    ratio = np.maximum(ph[:, None], dh[None, :]) / np.maximum(np.minimum(ph[:, None], dh[None, :]), 1.0)
    cost[(ratio > 2.0) | (track_cls[:, None] != det_cls[None, :])] = INF
    return cost


def track_video(detections: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add a track_id column to the detections of one video (cfg = config["tracker"]).

    Detections that are suppressed or never join a track are dropped. The column `class` becomes
    the track's majority class; the detection's own class is kept as `det_class`.
    """
    det = detections[detections["score"] >= cfg.get("min_det_score", 0.0)]
    det = det.sort_values("frame", kind="stable").reset_index(drop=True)
    frames = det["frame"].to_numpy()
    xyxy = det[["x", "y", "w", "h"]].to_numpy(float)
    xyxy[:, 2:] += xyxy[:, :2]
    scores = det["score"].to_numpy(float)
    classes = det["class"].to_numpy()
    assigned = np.full(len(det), -1)

    live: list[_Track] = []
    next_id = 0
    bounds = np.flatnonzero(np.diff(frames)) + 1
    for rows in np.split(np.arange(len(det)), bounds):
        if not len(rows):
            continue
        frame = int(frames[rows[0]])
        rows = rows[nms_per_class(xyxy[rows], scores[rows], classes[rows], cfg["nms_iou"])]
        live = [t for t in live if frame - t.last_frame <= cfg["max_age_frames"]]

        free_tracks = list(range(len(live)))
        free_dets = list(rows)
        preds = np.array([t.predict(frame) for t in live]).reshape(-1, 4)
        track_cls = np.array([t.cls for t in live], dtype=object)

        low_iou = cfg.get("low_iou_threshold", cfg["iou_threshold"])
        center_min = cfg.get("center_min_score", 0.0)
        rounds = [
            (_cost_iou, lambda r: scores[r] >= cfg["high_score"], 1 - cfg["iou_threshold"]),
            (_cost_iou, lambda r: scores[r] < cfg["high_score"], 1 - low_iou),
            (_cost_center, lambda r: scores[r] >= center_min, cfg["center_gate"]),
        ]
        for cost_fn, select, max_cost in rounds:
            cand = [r for r in free_dets if select(r)]
            if not free_tracks or not cand:
                continue
            cost = cost_fn(preds[free_tracks], xyxy[cand], track_cls[free_tracks], classes[cand])
            pairs = _assign(cost, max_cost)
            for ti, di in pairs:
                t, r = live[free_tracks[ti]], cand[di]
                t.update(xyxy[r], frame, cfg["velocity_smoothing"])
                assigned[r] = t.id
            matched_t = {free_tracks[ti] for ti, _ in pairs}
            matched_d = {cand[di] for _, di in pairs}
            free_tracks = [i for i in free_tracks if i not in matched_t]
            free_dets = [r for r in free_dets if r not in matched_d]

        for r in free_dets:
            if scores[r] >= cfg["new_track_score"]:
                live.append(_Track(next_id, xyxy[r], frame, classes[r]))
                assigned[r] = next_id
                next_id += 1

    out = det[assigned >= 0].assign(track_id=assigned[assigned >= 0]).rename(columns={"class": "det_class"})
    length = out.groupby("track_id")["frame"].transform("size")
    out = out[length >= cfg["min_track_len"]]
    majority = out.groupby("track_id")["det_class"].agg(lambda s: s.value_counts().index[0])
    out = out.assign(**{"class": out["track_id"].map(majority)})
    return out.reset_index(drop=True)
