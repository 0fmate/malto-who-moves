"""Validation that looks like the test set: folds never share a scene.

Scenes come from scripts/scene_groups.py (automatic clustering of video backgrounds). Test scenes are
not in the training set, so a model must always be judged on scenes it has not seen.
"""

from pathlib import Path

import pandas as pd


def load_scenes(cache_dir: Path) -> pd.Series:
    """video_id -> scene id, as computed by scripts/scene_groups.py."""
    df = pd.read_csv(Path(cache_dir) / "scene_groups.csv", dtype={"video_id": str})
    return df.set_index("video_id")["scene"]


def make_folds(scenes: pd.Series, weights: pd.Series, n_folds: int) -> pd.Series:
    """video_id -> fold, keeping each scene in one fold.

    Greedy balancing: scenes sorted by total weight (e.g. number of targets), each one goes to the
    fold that is currently lightest. Deterministic, so every experiment uses the same folds.
    """
    scene_weight = weights.groupby(scenes).sum().sort_values(ascending=False)
    load = [0.0] * n_folds
    scene_fold = {}
    for scene, w in scene_weight.items():
        fold = min(range(n_folds), key=lambda i: load[i])
        scene_fold[scene] = fold
        load[fold] += w
    return scenes.map(scene_fold).rename("fold")
