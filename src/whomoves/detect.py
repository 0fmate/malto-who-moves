"""Pretrained object detector (no vision-language models).

Maps COCO classes to person / vehicle / animal with config["detector"]["class_map"];
everything else (including birds) is dropped. Results are cached as parquet per video:
video_id, frame, t, x, y, w, h (pixels, top-left), score, class.
"""

from pathlib import Path

import pandas as pd


def detect_video(video_path: Path, cfg: dict) -> pd.DataFrame:
    """Run the detector on one video and return one row per detection."""
    raise NotImplementedError
