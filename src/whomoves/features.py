"""Motion features per track, over a sliding time window.

Ideas (all scale-free, so they work for near and far objects):
- net displacement of the box center / box height
- path length / net displacement (jitter vs real travel)
- box size change (approaching vs flicker)
- track age, detector score
- pixel change inside the box vs around it (background subtraction / optical flow)
"""

import pandas as pd


def track_features(tracks: pd.DataFrame, window_s: float) -> pd.DataFrame:
    """Return one row per (track_id, frame) with the motion features."""
    raise NotImplementedError
