"""Build submission.csv for the test videos from the cached features.

Method from config["motion"]["method"] (or --method):
- rule: moving once a track is rule_theta box heights away from where it first appeared;
- model: the LightGBM model saved by train_motion.py --save (trained on --model-run features),
  with its tuned threshold and confirmation.

Output: <output_dir>/submission_<name>.csv, validated against test_videos.csv.

Usage: python scripts/make_submission.py --name rule_v1 [--theta 2.0] [--confirm 5] [--det-run ...]
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whomoves.alarms import make_alarms, rule_moving  # noqa: E402
from whomoves.config import load_config  # noqa: E402
from whomoves.io import load_csv  # noqa: E402
from whomoves.motion_model import load, predict, smooth  # noqa: E402
from whomoves.submission import write_submission  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="suffix of the output file, e.g. rule_v1")
    parser.add_argument("--det-run", default="fasterrcnn_resnet50_fpn_v2_s1")
    parser.add_argument("--theta", type=float, help="override motion.rule_theta")
    parser.add_argument("--confirm", type=int, help="override motion.confirm_rows")
    parser.add_argument("--method", choices=["rule", "model"], help="override motion.method")
    parser.add_argument("--model-run", default="fasterrcnn_resnet50_fpn_v2_s1", help="features the model was trained on")
    args = parser.parse_args()
    cfg = load_config()
    m = cfg["motion"]
    if args.theta is not None:
        m["rule_theta"] = args.theta
    if args.confirm is not None:
        m["confirm_rows"] = args.confirm

    test_videos = load_csv(cfg["paths"]["data_dir"], "test_videos")
    feats = pd.read_parquet(cfg["paths"]["cache_dir"] / "features" / f"{args.det_run}.parquet")
    feats = feats[feats["video_id"].isin(test_videos["video_id"])]
    missing = set(test_videos["video_id"]) - set(feats["video_id"])
    if missing:
        print(f"warning: no tracks for {len(missing)} test videos (they get no alarm): {sorted(missing)[:5]}")

    method = args.method or m["method"]
    if method == "rule":
        alarms = make_alarms(feats, rule_moving(feats, m["rule_theta"]), test_videos, m["confirm_rows"], m["driver_inside"])
    else:
        model_dir = cfg["paths"]["cache_dir"] / "models"
        policy = json.loads((model_dir / f"lgbm_{args.model_run}.json").read_text())
        feats = feats.reset_index(drop=True)
        p = smooth(predict(load(model_dir / f"lgbm_{args.model_run}.txt"), feats), feats)
        print(f"model policy: P >= {policy['thr']} for {policy['confirm']} rows")
        alarms = make_alarms(feats, p >= policy["thr"], test_videos, policy["confirm"], m["driver_inside"])

    out = cfg["paths"]["output_dir"] / f"submission_{args.name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_submission(alarms, test_videos, out)
    per_video = alarms.groupby("video_id").size().reindex(test_videos["video_id"], fill_value=0)
    print(f"{len(alarms)} alarms on {len(test_videos)} test videos {alarms['class'].value_counts().to_dict()}; "
          f"per video: min {per_video.min()}, median {per_video.median():.0f}, max {per_video.max()}; "
          f"videos with no alarm: {(per_video == 0).sum()}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
