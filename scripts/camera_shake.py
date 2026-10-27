"""Measure how much the camera itself moves (wind, vibrations) in each video.

Phase correlation: the Fourier transform turns a shift of the whole image into a phase change, so
comparing the spectra of two frames gives the translation between them with sub-pixel precision.
Moving objects are a small part of the frame, so the background (the camera) dominates the answer.

For every video: shift of each sampled frame from the first one (in full-resolution pixels).
Prints per-split quantiles over videos and saves the per-video table to <cache_dir>/camera_shake.csv.

Usage: python scripts/camera_shake.py [--stride 5] [--max-frames 3000]
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whomoves.config import load_config  # noqa: E402
from whomoves.io import load_csv, video_path  # noqa: E402

WORK_WIDTH = 640


def video_shake(path: Path, stride: int, max_frames: int) -> dict:
    cap = cv2.VideoCapture(str(path))
    ref, window, scale, shifts, idx = None, None, 1.0, [], 0
    while idx < max_frames:
        if not cap.grab():
            break
        if idx % stride == 0:
            _, bgr = cap.retrieve()
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            scale = gray.shape[1] / WORK_WIDTH
            gray = cv2.resize(gray, (WORK_WIDTH, round(gray.shape[0] / scale)), interpolation=cv2.INTER_AREA)
            gray = gray.astype(np.float32)
            if ref is None:
                ref = gray
                window = cv2.createHanningWindow(gray.shape[::-1], cv2.CV_32F)
            else:
                (dx, dy), _ = cv2.phaseCorrelate(ref, gray, window)
                shifts.append(np.hypot(dx, dy) * scale)
        idx += 1
    cap.release()
    s = np.array(shifts) if shifts else np.zeros(1)
    return {"p50": np.median(s), "p95": np.quantile(s, 0.95), "max": s.max()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=3000)
    args = parser.parse_args()
    cfg = load_config()
    data_dir = cfg["paths"]["data_dir"]
    rows = []
    for split in ["train", "test"]:
        for vid in load_csv(data_dir, f"{split}_videos")["video_id"]:
            rows.append({"split": split, "video_id": vid,
                         **video_shake(video_path(data_dir, vid), args.stride, args.max_frames)})
    df = pd.DataFrame(rows)
    df.to_csv(cfg["paths"]["cache_dir"] / "camera_shake.csv", index=False)
    print("camera shift from the first frame, full-resolution px (quantiles over videos):")
    for split, g in df.groupby("split"):
        q = g[["p50", "p95", "max"]].quantile([0.5, 0.9, 1.0]).round(1)
        print(f"\n{split}: {len(g)} videos; videos with p95 > 2 px: {int((g['p95'] > 2).sum())}, > 5 px: {int((g['p95'] > 5).sum())}")
        print(q.to_string())


if __name__ == "__main__":
    main()
