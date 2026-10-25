"""Motion features per track and frame, computed from the track's own boxes (no pixels yet).

All distances are divided by the box height, so the same numbers mean the same thing for a car
next to the camera and for a person 30 px tall. Box centers are first smoothed with a centered
rolling median over 5 detections, to remove the detector's jitter.

Real time is not required, so features may look into the future of the track ("will it be far
away in 2 s?"): the alarm itself is still placed at the moment motion is confirmed.

No absolute image position is used: it belongs to a scene, and test scenes are new (some mirrored).
"""

import numpy as np
import pandas as pd

WINDOWS_S = (1.0, 2.0, 5.0)
SMOOTH_ROWS = 5


def _rolling_median(a: np.ndarray, k: int) -> np.ndarray:
    return pd.Series(a).rolling(k, center=True, min_periods=1).median().to_numpy()


def _track_features(t: np.ndarray, box: np.ndarray, width: int, height: int) -> dict:
    """Features for one track; t sorted, box (n, 4) as x, y, w, h."""
    n = len(t)
    c = np.stack([_rolling_median(box[:, 0] + box[:, 2] / 2, SMOOTH_ROWS),
                  _rolling_median(box[:, 1] + box[:, 3] / 2, SMOOTH_ROWS)], axis=1)
    h = np.maximum(_rolling_median(box[:, 3], 2 * SMOOTH_ROWS + 1), 1.0)
    step = np.r_[0.0, np.linalg.norm(np.diff(c, axis=0), axis=1)]
    path = np.cumsum(step)  # distance travelled along the smoothed trajectory

    f = {}
    for k in WINDOWS_S:
        past = np.clip(np.searchsorted(t, t - k, side="left"), 0, n - 1)
        future = np.clip(np.searchsorted(t, t + k, side="right") - 1, 0, n - 1)
        f[f"disp_past_{k:g}s"] = np.linalg.norm(c - c[past], axis=1) / h
        f[f"disp_future_{k:g}s"] = np.linalg.norm(c[future] - c, axis=1) / h
        if k == 2.0:
            net = np.linalg.norm(c - c[past], axis=1)
            f["straightness_2s"] = net / np.maximum(path - path[past], 1e-6)
            f["log_size_change_2s"] = np.log(h / h[past])

    from_start = np.linalg.norm(c - c[0], axis=1) / h
    f["disp_from_start"] = from_start
    f["max_disp_from_start"] = np.maximum.accumulate(from_start)
    f["track_range"] = np.full(n, from_start.max())
    f["age_s"] = t - t[0]
    f["remaining_s"] = t[-1] - t
    f["track_len_s"] = np.full(n, t[-1] - t[0])
    f["first_t"] = np.full(n, t[0])
    x, y, w, bh = box[0]
    f["first_border_dist"] = np.full(n, min(x, y, width - x - w, height - y - bh) / max(bh, 1.0))
    f["h_rel"] = h / height
    f["n_dets_per_s"] = np.full(n, n / max(t[-1] - t[0], 0.1))
    return f


def _inside_vehicle(df: pd.DataFrame) -> np.ndarray:
    """For person rows: the largest share of the box covered by a vehicle box in the same frame."""
    out = np.zeros(len(df))
    is_person = (df["class"] == "person").to_numpy()
    is_vehicle = (df["class"] == "vehicle").to_numpy()
    frames = df["frame"].to_numpy()
    both = np.intersect1d(frames[is_person], frames[is_vehicle])
    if not len(both):
        return out
    xyxy = df[["x", "y", "w", "h"]].to_numpy(float)
    xyxy[:, 2:] += xyxy[:, :2]
    order = np.argsort(frames, kind="stable")
    bounds = np.searchsorted(frames[order], both, side="left"), np.searchsorted(frames[order], both, side="right")
    for lo, hi in zip(*bounds):
        rows = order[lo:hi]
        p, v = rows[is_person[rows]], rows[is_vehicle[rows]]
        a, b = xyxy[p], xyxy[v]
        iw = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
        ih = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
        area = np.maximum((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), 1e-6)
        out[p] = (iw * ih / area[:, None]).max(axis=1)
    return out


def track_features(tracks: pd.DataFrame, width: int, height: int) -> pd.DataFrame:
    """Tracks of ONE video in, same rows (sorted by track, time) with feature columns added."""
    df = tracks.sort_values(["track_id", "t"], kind="stable").reset_index(drop=True)
    parts = []
    for _, g in df.groupby("track_id", sort=False):
        parts.append(pd.DataFrame(_track_features(g["t"].to_numpy(float), g[["x", "y", "w", "h"]].to_numpy(float),
                                                  width, height), index=g.index))
    feats = pd.concat(parts).loc[df.index]
    df = pd.concat([df, feats], axis=1)
    df["inside_vehicle"] = _inside_vehicle(df)
    df["score_mean"] = df.groupby("track_id")["score"].transform("mean")
    return df


FEATURE_COLUMNS = (
    [f"disp_past_{k:g}s" for k in WINDOWS_S] + [f"disp_future_{k:g}s" for k in WINDOWS_S]
    + ["straightness_2s", "log_size_change_2s", "disp_from_start", "max_disp_from_start", "track_range",
       "age_s", "remaining_s", "track_len_s", "first_t", "first_border_dist", "h_rel", "n_dets_per_s",
       "inside_vehicle", "score", "score_mean"]
)
