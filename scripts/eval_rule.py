"""Tune the hand-written motion rule with the competition metric on the training videos.

Rule: a track moves once it is more than THETA box heights away from where it first appeared, for
CONFIRM consecutive detections. Prints a grid of scores on all training videos, then, for the best
setting, the score of each scene fold (are the numbers stable across parking lots?) and the outcome
breakdown (what kind of mistakes are left).

Usage: python scripts/eval_rule.py [--det-run fasterrcnn_resnet50_fpn_v2_s1]
"""

import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whomoves.alarms import make_alarms, rule_moving  # noqa: E402
from whomoves.config import load_config  # noqa: E402
from whomoves.io import load_csv  # noqa: E402
from whomoves.metric import GroundTruth, score  # noqa: E402
from whomoves.validation import load_scenes, make_folds  # noqa: E402

THETAS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
CONFIRMS = [1, 5, 10]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--det-run", default="fasterrcnn_resnet50_fpn_v2_s1")
    parser.add_argument("--videos", nargs="*", help="score only these train videos")
    args = parser.parse_args()
    cfg = load_config()
    data_dir, cache = cfg["paths"]["data_dir"], cfg["paths"]["cache_dir"]

    videos = load_csv(data_dir, "train_videos")
    targets = load_csv(data_dir, "train_alarms")
    gt = GroundTruth(load_csv(data_dir, "train_tracks"), targets, videos)
    feats = pd.read_parquet(cache / "features" / f"{args.det_run}.parquet")
    feats = feats[feats["video_id"].str.startswith("train_")]
    # A detection run may cover only some train videos: score those only.
    vids = sorted(feats["video_id"].unique())
    if args.videos:
        vids = [v for v in vids if v in set(args.videos)]
        feats = feats[feats["video_id"].isin(vids)]
    driver = cfg["motion"]["driver_inside"]

    results = []
    for theta, confirm in itertools.product(THETAS, CONFIRMS):
        a = make_alarms(feats, rule_moving(feats, theta), videos, confirm, driver)
        r = score(a, gt, video_ids=vids)
        results.append({"theta": theta, "confirm": confirm, "f1": r["f1"], "precision": r["precision"],
                        "target_recall": r["target_recall"], "alarms": len(a)})
    grid = pd.DataFrame(results)
    print(f"F1 on {len(vids)} training videos (rows: theta in box heights, columns: confirm rows):")
    print(grid.pivot(index="theta", columns="confirm", values="f1").round(3).to_string(), "\n")

    best = grid.sort_values("f1", ascending=False).iloc[0]
    theta, confirm = float(best["theta"]), int(best["confirm"])
    alarms = make_alarms(feats, rule_moving(feats, theta), videos, confirm, driver)
    r = score(alarms, gt, video_ids=vids)
    print(f"best: theta={theta} confirm={confirm}  F1 {r['f1']:.3f}  P {r['precision']:.3f}  "
          f"target recall {r['target_recall']:.3f}  alarms {len(alarms)}")
    print("outcomes:", dict(sorted(r["outcomes"].items(), key=lambda kv: -kv[1])))

    scenes_path = cache / "scene_groups.csv"
    if scenes_path.exists() and len(vids) == len(videos):
        scenes = load_scenes(cache)
        weights = targets[targets["bonus"] == 0].groupby("video_id").size().reindex(scenes.index, fill_value=0)
        folds = make_folds(scenes, weights + 1, cfg["validation"]["n_folds"])
        print("\nper scene fold (same rule, no refit):")
        for fold, vids in folds.groupby(folds).groups.items():
            rf = score(alarms, gt, video_ids=list(vids))
            print(f"  fold {fold}: {len(vids):2d} videos, scenes {sorted(int(s) for s in scenes[vids].unique())}, "
                  f"targets {rf['n_targets']:4d}  F1 {rf['f1']:.3f}  P {rf['precision']:.3f}  "
                  f"target recall {rf['target_recall']:.3f}")


if __name__ == "__main__":
    main()
