"""Match predicted tracks to ground-truth tracks: the training labels for the motion model.

1. Per frame, Hungarian matching on IoU between predicted and GT boxes, with any role and any class:
   a "person" detection on a shadow must learn the label "disturbance".
2. Per predicted track, the GT track it matched most often becomes its gt_track_id, if that covers at
   least min_share of the track's rows. Otherwise the track is unmatched: a false detection, or an
   object nobody annotated (gt_role "none").
3. Every row of a matched track gets the GT role, and the GT `moving` value of that GT track at that
   frame (NaN where the GT track has no box in that frame).
"""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .track import iou_xyxy


def _xyxy(df: pd.DataFrame) -> np.ndarray:
    b = df[["x", "y", "w", "h"]].to_numpy(float)
    b[:, 2:] += b[:, :2]
    return b


def match_tracks(pred: pd.DataFrame, gt: pd.DataFrame, iou_threshold: float = 0.3,
                 min_share: float = 0.5) -> pd.DataFrame:
    """pred and gt are the rows of ONE video. Returns pred with added columns:
    row_gt_track_id (frame-level match), gt_track_id, gt_role, gt_moving (track-level)."""
    pred = pred.reset_index(drop=True)
    row_match = np.full(len(pred), None, dtype=object)
    gt_by_frame = {f: g for f, g in gt.groupby("frame")}
    for f, p in pred.groupby("frame"):
        g = gt_by_frame.get(f)
        if g is None:
            continue
        iou = iou_xyxy(_xyxy(p), _xyxy(g))
        rows, cols = linear_sum_assignment(-iou)
        ok = iou[rows, cols] >= iou_threshold
        row_match[p.index[rows[ok]]] = g["track_id"].to_numpy()[cols[ok]]
    pred["row_gt_track_id"] = row_match

    # Track level: majority GT track, if it covers enough of the predicted track.
    matched = pred.dropna(subset=["row_gt_track_id"])
    counts = matched.groupby(["track_id", "row_gt_track_id"]).size().rename("n").reset_index()
    best = counts.sort_values("n", ascending=False).drop_duplicates("track_id")
    best = best.merge(pred.groupby("track_id").size().rename("len").reset_index(), on="track_id")
    best = best[best["n"] >= min_share * best["len"]]
    pred["gt_track_id"] = pred["track_id"].map(dict(zip(best["track_id"], best["row_gt_track_id"])))

    roles = gt.drop_duplicates("track_id").set_index("track_id")["role"]
    pred["gt_role"] = pred["gt_track_id"].map(roles).fillna("none")
    moving = gt.set_index(["track_id", "frame"])["moving"]
    moving = moving[~moving.index.duplicated()]
    key = pd.MultiIndex.from_arrays([pred["gt_track_id"], pred["frame"]])
    pred["gt_moving"] = moving.reindex(key).to_numpy()
    return pred
