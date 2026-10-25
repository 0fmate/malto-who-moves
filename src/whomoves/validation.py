"""Validation that looks like the test set.

- Folds grouped by scene / source dataset: test scenes never appear in training.
- Test-like clips: 1-3 minute cuts, some mirrored horizontally (x -> 1 - x), possibly
  starting while something is already moving.
"""

import pandas as pd


def make_folds(videos: pd.DataFrame, n_folds: int, group_by: str, seed: int) -> pd.Series:
    """Return a fold index per video_id, never splitting a group across folds."""
    raise NotImplementedError


def make_clips(videos: pd.DataFrame, min_s: float, max_s: float, seed: int) -> pd.DataFrame:
    """Return clips (video_id, t_start, t_end, mirrored) covering the training videos."""
    raise NotImplementedError
