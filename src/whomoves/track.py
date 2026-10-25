"""Link detections across frames into tracks (the camera is fixed)."""

import pandas as pd


def track_video(detections: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add a track_id column to the detections of one video."""
    raise NotImplementedError
