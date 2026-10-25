"""Load the competition CSVs and iterate over video frames."""

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import pandas as pd


def load_csv(data_dir: Path, name: str) -> pd.DataFrame:
    """Load one of the competition CSVs, e.g. load_csv(data_dir, "train_tracks")."""
    return pd.read_csv(Path(data_dir) / f"{name}.csv", dtype={"video_id": str})


def video_path(data_dir: Path, video_id: str) -> Path:
    """All videos sit flat in data_dir; train ids already carry the "train_" prefix."""
    return Path(data_dir) / f"{video_id}.mp4"


def iter_frames(path: Path, stride: int = 1) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (frame_index, t_seconds, BGR image) for every `stride`-th frame."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                yield idx, idx / fps, frame
            idx += 1
    finally:
        cap.release()


def read_frame_at(path: Path, t: float) -> np.ndarray:
    """Return the frame closest to time t (seconds)."""
    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"cannot read frame at t={t}s from {path}")
        return frame
    finally:
        cap.release()
