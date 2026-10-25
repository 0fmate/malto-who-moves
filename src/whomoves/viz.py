"""Draw boxes, tracks and alarms on frames, for error analysis and the report.

Show only a few frames, only to illustrate the method (competition rules).
"""

from pathlib import Path

import pandas as pd


def save_overlay(video_path: Path, t: float, boxes: pd.DataFrame, out_path: Path) -> None:
    """Save the frame at time t with the given boxes (and their labels) drawn on it."""
    raise NotImplementedError
