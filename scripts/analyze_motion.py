"""What does "moving" look like on our tracks? Numbers for the report and for the hand-written rule.

All distances in box heights, measured on the smoothed track centers (see features.py):
1. How far does each kind of track get from where it first appeared (track_range)?
   Parked cars should stay near 0 (detector jitter); movers should go far.
2. For targets, how far have they gone 1, 2 and 3 s after the annotated start of motion?
   This bounds how quickly a distance rule can fire.
3. For targets visible while still waiting (e.g. a parked car that later leaves), how far had they
   gone before the start? A rule must stay silent there: alarms before the start weigh 2.

Usage: python scripts/analyze_motion.py [--det-run fasterrcnn_resnet50_fpn_v2_s1]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whomoves.config import load_config  # noqa: E402
from whomoves.io import load_csv  # noqa: E402

Q = [0.1, 0.5, 0.9, 0.99]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--det-run", default="fasterrcnn_resnet50_fpn_v2_s1")
    args = parser.parse_args()
    cfg = load_config()
    f = pd.read_parquet(cfg["paths"]["cache_dir"] / "features" / f"{args.det_run}.parquet")
    f = f[f["video_id"].str.startswith("train_")]
    alarms = load_csv(cfg["paths"]["data_dir"], "train_alarms")

    tracks = f.groupby(["video_id", "track_id"]).agg(
        role=("gt_role", "first"), cls=("class", "first"), range=("track_range", "first"),
        len_s=("track_len_s", "first"))
    print("1. How far each track gets from its first position (box heights), by matched GT role:")
    table = tracks.groupby("role")["range"].describe(percentiles=Q)[["count", "10%", "50%", "90%", "99%"]]
    print(table.round(2).to_string(), "\n")
    long_static = tracks[(tracks["role"] == "static") & (tracks["len_s"] > 30)]
    print(f"   static tracks longer than 30 s: {len(long_static)}, range 90%/99% = "
          f"{np.quantile(long_static['range'], [0.9, 0.99]).round(2)}\n")

    # Targets: rows of our tracks matched to a target, with its annotated start of motion.
    tg = f[f["gt_role"] == "target"].merge(
        alarms[["video_id", "track_id", "t_move_start"]].rename(columns={"track_id": "gt_track_id"}),
        on=["video_id", "gt_track_id"])
    tg["since_start"] = tg["t"] - tg["t_move_start"]
    # Distance travelled since the start, relative to the track position at the start (or first seen).
    print("2. Targets: max distance from their first position, k seconds after the annotated start:")
    rows = []
    for k in [1.0, 2.0, 3.0]:
        at = tg[(tg["since_start"] - k).abs() <= 0.05]
        rows.append(at.groupby("class")["max_disp_from_start"].quantile([0.1, 0.25, 0.5]).unstack()
                    .assign(after_s=k).reset_index())
    print(pd.concat(rows).set_index(["after_s", "class"]).round(2).to_string(), "\n")

    waiting = tg[(tg["since_start"] < -1.0) & (tg["t_move_start"] - tg["first_t"] > 3.0)]
    before = waiting.groupby(["video_id", "track_id"]).agg(cls=("class", "first"), max_before=("max_disp_from_start", "max"))
    print(f"3. Targets seen waiting > 3 s before starting: {len(before)} tracks; max distance before the start "
          f"(box heights) 50/90/99% = {np.quantile(before['max_before'], [0.5, 0.9, 0.99]).round(2)}")
    print(before.groupby("cls")["max_before"].describe(percentiles=[0.5, 0.9])[["count", "50%", "90%", "max"]]
          .round(2).to_string())


if __name__ == "__main__":
    main()
