"""Match predicted tracks to ground-truth tracks, frame by frame, by IoU.

Gives each predicted box the GT role and `moving` label: these are the training labels
for the motion model, and also tell which GT objects the detector never finds.
"""

import pandas as pd


def match_tracks(pred: pd.DataFrame, gt: pd.DataFrame, iou_threshold: float = 0.3) -> pd.DataFrame:
    """Return pred with added columns gt_track_id, role, moving (NaN when unmatched)."""
    raise NotImplementedError
