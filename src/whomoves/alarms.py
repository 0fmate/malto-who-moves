"""Turn tracks + P(moving) into alarms.

Policy:
- one alarm per track, at the first frame where motion is confirmed for
  cfg["motion"]["confirm_frames"] consecutive frames; position = box center (normalized)
- drop person boxes mostly inside a vehicle box (drivers)
- earliness is not rewarded, false alarms cost double: prefer waiting to guessing
"""

import pandas as pd


def make_alarms(tracks: pd.DataFrame, p_moving: pd.Series, videos: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Return alarms: video_id, t, x, y, class (x, y in [0, 1])."""
    raise NotImplementedError
