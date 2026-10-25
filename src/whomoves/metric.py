"""Local re-implementation of the competition scorer (see CLAUDE.md, "How the score works").

Rules (from the Evaluation page):
- Targets are the rows of train_alarms.csv: role "target" (counted in recall) and "bonus"
  (only adds points).
- An alarm hits a target if t is in [t_move_start - 1, t_end + 3] and (x, y) falls inside the
  target's box at that time, enlarged by 20% on each side.
- Alarms are processed in time order. First hit on a target: 1.0 (right class) or 0.6 (wrong).
  Later alarms on an already-hit target are ignored.
- Any other alarm is a false alarm: weight 2 if it falls on something that must not alarm
  (static, disturbance, driver, a target that has not started to move yet), weight 1 otherwise.
  Alarms on "ambiguous" / "other" objects are neutral.
- F1 over all videos together: P = points / (hits + weighted FA), R = points / n_targets.

Assumptions where the official rules are not explicit:
- When the time is outside the frames where the target box exists (e.g. in the 3 s after it
  leaves), the box of the nearest annotated frame is used.
- If an alarm could hit several targets, it goes to a not-yet-hit one, preferring the right
  class and then the closest box center.
- Boxes of objects that must not alarm are not enlarged.
- Precedence when the alarm is on several objects: hit > duplicate > weight-2 FA > neutral > weight-1 FA.
- Recall is capped at 1, so bonus hits cannot push the score above 1.
- "Areas the original annotation does not cover" are unknown to us: such alarms count as weight-1 FA,
  so the local score is slightly pessimistic.

Usage:
    gt = GroundTruth(tracks, targets, videos)   # build once (a few seconds)
    result = score(alarms, gt)                  # fast, call as often as needed
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

TIME_BEFORE_S = 1.0
TIME_AFTER_S = 3.0
BOX_MARGIN = 0.20
WRONG_CLASS_POINTS = 0.6
NEGATIVE_ROLES = {"static", "disturbance", "driver"}
NEUTRAL_ROLES = {"ambiguous", "other"}
MOVER_ROLES = {"target", "bonus"}
HIT_OUTCOMES = {"hit", "hit_wrong_class"}


def _xyxy(df: pd.DataFrame) -> np.ndarray:
    """Boxes as (n, 4) arrays of x1, y1, x2, y2 in pixels."""
    x, y = df["x"].to_numpy(float), df["y"].to_numpy(float)
    return np.stack([x, y, x + df["w"].to_numpy(float), y + df["h"].to_numpy(float)], axis=1)


def _inside(boxes: np.ndarray, px: float, py: float, margin: float = 0.0) -> np.ndarray:
    """Whether the point falls inside each box, enlarged by `margin` of its size on each side."""
    x1, y1, x2, y2 = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    mx, my = margin * (x2 - x1), margin * (y2 - y1)
    return (px >= x1 - mx) & (px <= x2 + mx) & (py >= y1 - my) & (py <= y2 + my)


def _center_distance(box: np.ndarray, px: float, py: float) -> float:
    """Distance from the box center, in units of box height (scale-free tie-break)."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return float(np.hypot(px - cx, py - cy) / max(box[3] - box[1], 1.0))


@dataclass
class _Target:
    track_id: str
    t_start: float
    t_end: float
    cls: str
    bonus: bool
    frames: np.ndarray  # sorted frame indices where the box is annotated
    boxes: np.ndarray   # (n, 4) boxes at those frames


class _VideoGT:
    """Ground truth of one video, indexed by frame for fast lookups."""

    def __init__(self, rows: pd.DataFrame, targets: pd.DataFrame, width: int, height: int, fps: float):
        rows = rows.sort_values(["frame", "track_id"], kind="stable")
        self.width, self.height, self.fps = width, height, fps
        self.frames = rows["frame"].to_numpy()
        self.roles = rows["role"].to_numpy()
        self.track_ids = rows["track_id"].to_numpy()
        self.boxes = _xyxy(rows)
        self.move_start = dict(zip(targets["track_id"], targets["t_move_start"]))
        self.targets = []
        for tgt in targets.itertuples(index=False):
            mask = self.track_ids == tgt.track_id  # rows are sorted by frame, so is the selection
            self.targets.append(_Target(
                track_id=tgt.track_id, t_start=tgt.t_move_start, t_end=tgt.t_end,
                cls=tgt.cls, bonus=bool(tgt.bonus),
                frames=self.frames[mask], boxes=self.boxes[mask],
            ))

    def rows_at(self, t: float) -> slice:
        """Slice of the rows annotated at the frame closest to time t."""
        f = int(round(t * self.fps))
        return slice(np.searchsorted(self.frames, f, "left"), np.searchsorted(self.frames, f, "right"))

    def target_box(self, tgt: _Target, t: float) -> np.ndarray:
        """Box of a target at time t, from the nearest annotated frame."""
        f = t * self.fps
        i = int(np.clip(np.searchsorted(tgt.frames, f), 1, len(tgt.frames) - 1)) if len(tgt.frames) > 1 else 0
        if i > 0 and abs(tgt.frames[i - 1] - f) <= abs(tgt.frames[i] - f):
            i -= 1
        return tgt.boxes[i]


class GroundTruth:
    """All the annotations needed to score alarms. Build once, reuse for every score() call."""

    def __init__(self, tracks: pd.DataFrame, targets: pd.DataFrame, videos: pd.DataFrame):
        targets = targets[["video_id", "track_id", "t_move_start", "t_end", "class", "bonus"]]
        targets = targets.rename(columns={"class": "cls"})  # "class" is a Python keyword
        targets_by_video = dict(tuple(targets.groupby("video_id")))
        rows_by_video = dict(tuple(tracks.groupby("video_id")))
        self.videos = {}
        for v in videos.itertuples(index=False):
            self.videos[v.video_id] = _VideoGT(
                rows_by_video.get(v.video_id, tracks.iloc[:0]),
                targets_by_video.get(v.video_id, targets.iloc[:0]),
                v.width, v.height, v.fps,
            )


def _judge(v: _VideoGT, hit: np.ndarray, t: float, px: float, py: float, cls: str) -> dict:
    """Decide what one alarm is worth, updating `hit` in place."""
    candidates = []
    for k, tgt in enumerate(v.targets):
        if tgt.t_start - TIME_BEFORE_S <= t <= tgt.t_end + TIME_AFTER_S and len(tgt.frames):
            box = v.target_box(tgt, t)
            if _inside(box, px, py, BOX_MARGIN):
                candidates.append((k, box))

    fresh = [(k, box) for k, box in candidates if not hit[k]]
    if fresh:
        k, _ = min(fresh, key=lambda kb: (v.targets[kb[0]].cls != cls, _center_distance(kb[1], px, py)))
        hit[k] = True
        tgt = v.targets[k]
        right = tgt.cls == cls
        return {"outcome": "hit" if right else "hit_wrong_class", "points": 1.0 if right else WRONG_CLASS_POINTS,
                "fa_weight": 0, "track_id": tgt.track_id, "bonus": tgt.bonus}
    if candidates:
        k = candidates[0][0]
        return {"outcome": "duplicate", "points": 0.0, "fa_weight": 0,
                "track_id": v.targets[k].track_id, "bonus": v.targets[k].bonus}

    # Not on a scorable target: what is under the alarm at this moment?
    sl = v.rows_at(t)
    under = _inside(v.boxes[sl], px, py)
    neutral_tid = None
    for role, tid in zip(v.roles[sl][under], v.track_ids[sl][under]):
        if role in NEGATIVE_ROLES:
            return {"outcome": f"fa_{role}", "points": 0.0, "fa_weight": 2, "track_id": tid, "bonus": False}
        if role in MOVER_ROLES and tid in v.move_start and t < v.move_start[tid] - TIME_BEFORE_S:
            return {"outcome": "fa_premotion", "points": 0.0, "fa_weight": 2, "track_id": tid, "bonus": False}
        if role in NEUTRAL_ROLES:
            neutral_tid = tid
    if neutral_tid is not None:
        return {"outcome": "neutral", "points": 0.0, "fa_weight": 0, "track_id": neutral_tid, "bonus": False}
    return {"outcome": "fa_background", "points": 0.0, "fa_weight": 1, "track_id": None, "bonus": False}


def score(alarms: pd.DataFrame, gt: GroundTruth, video_ids=None) -> dict:
    """Score alarms (video_id, t, x, y, class; x, y normalized to [0, 1]).

    video_ids restricts scoring to a subset of videos (e.g. a validation fold): targets of the
    other videos are not counted and their alarms are dropped.

    Returns f1, precision, recall and counts per outcome, plus two DataFrames for error analysis:
    "alarms" (one row per alarm with its outcome) and "targets" (one row per target, hit or missed).
    """
    video_ids = list(gt.videos) if video_ids is None else list(video_ids)
    alarms = alarms[alarms["video_id"].isin(video_ids)].sort_values("t", kind="stable")

    hit = {vid: np.zeros(len(gt.videos[vid].targets), dtype=bool) for vid in video_ids}
    records = []
    for vid, t, x, y, cls in alarms[["video_id", "t", "x", "y", "class"]].itertuples(index=False):
        v = gt.videos[vid]
        records.append({"video_id": vid, "t": t, "x": x, "y": y, "class": cls,
                        **_judge(v, hit[vid], t, x * v.width, y * v.height, cls)})
    details = pd.DataFrame(records, columns=["video_id", "t", "x", "y", "class",
                                             "outcome", "points", "fa_weight", "track_id", "bonus"])

    targets = pd.DataFrame([
        {"video_id": vid, "track_id": tgt.track_id, "class": tgt.cls, "bonus": tgt.bonus, "hit": bool(h)}
        for vid in video_ids for tgt, h in zip(gt.videos[vid].targets, hit[vid])
    ], columns=["video_id", "track_id", "class", "bonus", "hit"])

    n_targets = int((~targets["bonus"]).sum())
    points = float(details["points"].sum())
    n_hits = int(details["outcome"].isin(HIT_OUTCOMES).sum())
    fa_weighted = float(details["fa_weight"].sum())
    precision = points / (n_hits + fa_weighted) if n_hits + fa_weighted > 0 else 0.0
    recall = min(points / n_targets, 1.0) if n_targets > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return {
        "f1": f1, "precision": precision, "recall": recall,
        "points": points, "n_targets": n_targets, "n_hits": n_hits, "fa_weighted": fa_weighted,
        "n_missed": int((~targets["hit"] & ~targets["bonus"]).sum()),
        "outcomes": details["outcome"].value_counts().to_dict(),
        "alarms": details, "targets": targets,
    }


def reference_alarms(gt: GroundTruth) -> pd.DataFrame:
    """The ideal answer: one alarm per target, at its box center when it starts to move."""
    rows = []
    for vid, v in gt.videos.items():
        for tgt in v.targets:
            if not len(tgt.frames):
                continue
            box = v.target_box(tgt, tgt.t_start)
            rows.append({"video_id": vid, "t": tgt.t_start,
                         "x": (box[0] + box[2]) / 2 / v.width, "y": (box[1] + box[3]) / 2 / v.height,
                         "class": tgt.cls})
    return pd.DataFrame(rows, columns=["video_id", "t", "x", "y", "class"])
