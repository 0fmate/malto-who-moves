"""Compute motion features for the cached tracks of every video, into one parquet file.

Output: <cache_dir>/features/<detections run>.parquet (train rows keep the gt_* label columns).

Usage: python scripts/build_features.py [--det-run fasterrcnn_resnet50_fpn_v2_s1]
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whomoves.config import load_config  # noqa: E402
from whomoves.features import track_features  # noqa: E402
from whomoves.io import load_csv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--det-run", default="fasterrcnn_resnet50_fpn_v2_s1")
    args = parser.parse_args()

    cfg = load_config()
    data_dir, cache = cfg["paths"]["data_dir"], cfg["paths"]["cache_dir"]
    videos = pd.concat([load_csv(data_dir, "train_videos"), load_csv(data_dir, "test_videos")]).set_index("video_id")

    t0, parts = time.time(), []
    for path in sorted((cache / "tracks" / args.det_run).glob("*.parquet")):
        v = videos.loc[path.stem]
        parts.append(track_features(pd.read_parquet(path), int(v["width"]), int(v["height"])))
    feats = pd.concat(parts, ignore_index=True)

    out = cache / "features" / f"{args.det_run}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out, index=False)
    print(f"{len(parts)} videos, {len(feats):,} rows, {len(feats.drop_duplicates(['video_id', 'track_id']))} tracks "
          f"in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
