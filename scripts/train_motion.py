"""Train the LightGBM motion model with scene folds, tune the alarm policy, compare with the rule.

1. Out-of-fold predictions: for each scene fold, a model trained on the other folds predicts it, so every
   training video is scored by a model that never saw its parking lot.
2. P is smoothed over 5 rows; an alarm is raised once P >= THRESHOLD for CONFIRM rows in a row.
   The pair is chosen on the out-of-fold F1 of all training videos.
3. --eval-run: the same fold models applied to another detection run (e.g. tiled boxes) of the train
   videos it covers, to check that a model trained on full-frame boxes transfers.
4. --save: a final model on all training videos, plus the chosen policy, for make_submission.py.

Usage: python scripts/train_motion.py [--det-run fasterrcnn_resnet50_fpn_v2_s1]
                                      [--eval-run fasterrcnn_resnet50_fpn_v2_s1_t2x2] [--save]
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whomoves.alarms import make_alarms, rule_moving  # noqa: E402
from whomoves.config import load_config  # noqa: E402
from whomoves.io import load_csv  # noqa: E402
from whomoves.metric import GroundTruth, score  # noqa: E402
from whomoves.motion_model import importance, predict, smooth, train  # noqa: E402
from whomoves.validation import load_scenes, make_folds  # noqa: E402

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
CONFIRMS = [1, 3, 5, 10]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--det-run", default="fasterrcnn_resnet50_fpn_v2_s1")
    parser.add_argument("--eval-run", help="another detection run of (some) train videos to test transfer on")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    cfg = load_config()
    data_dir, cache, m = cfg["paths"]["data_dir"], cfg["paths"]["cache_dir"], cfg["motion"]

    videos = load_csv(data_dir, "train_videos")
    targets = load_csv(data_dir, "train_alarms")
    gt = GroundTruth(load_csv(data_dir, "train_tracks"), targets, videos)
    feats = pd.read_parquet(cache / "features" / f"{args.det_run}.parquet")
    feats = feats[feats["video_id"].str.startswith("train_")].reset_index(drop=True)

    scenes = load_scenes(cache)
    weights = targets[targets["bonus"] == 0].groupby("video_id").size().reindex(scenes.index, fill_value=0)
    folds = make_folds(scenes, weights + 1, cfg["validation"]["n_folds"])
    row_fold = feats["video_id"].map(folds)

    oof, models = pd.Series(index=feats.index, dtype=float), {}
    for k in sorted(folds.unique()):
        models[k] = train(feats[row_fold != k])
        oof[row_fold == k] = predict(models[k], feats[row_fold == k])
    p = smooth(oof, feats)

    results = []
    for thr, confirm in itertools.product(THRESHOLDS, CONFIRMS):
        r = score(make_alarms(feats, p >= thr, videos, confirm, m["driver_inside"]), gt)
        results.append({"thr": thr, "confirm": confirm, "f1": r["f1"]})
    grid = pd.DataFrame(results)
    print("Out-of-fold F1 on 90 train videos (rows: P threshold, columns: confirm rows):")
    print(grid.pivot(index="thr", columns="confirm", values="f1").round(3).to_string(), "\n")
    best = grid.sort_values("f1", ascending=False).iloc[0]
    thr, confirm = float(best["thr"]), int(best["confirm"])

    model_alarms = make_alarms(feats, p >= thr, videos, confirm, m["driver_inside"])
    rule_alarms = make_alarms(feats, rule_moving(feats, m["rule_theta"]), videos, m["confirm_rows"], m["driver_inside"])
    print(f"{'':10s} {'model thr=%.1f c=%d' % (thr, confirm):>34s}   {'rule theta=%.1f c=%d' % (m['rule_theta'], m['confirm_rows']):>34s}")
    for name, vids in [("all", list(folds.index))] + [(f"fold {k}", list(v)) for k, v in folds.groupby(folds).groups.items()]:
        a, b = score(model_alarms, gt, video_ids=vids), score(rule_alarms, gt, video_ids=vids)
        print(f"{name:10s} F1 {a['f1']:.3f} P {a['precision']:.3f} tR {a['target_recall']:.3f}      "
              f"F1 {b['f1']:.3f} P {b['precision']:.3f} tR {b['target_recall']:.3f}")
    r = score(model_alarms, gt)
    print("model outcomes:", dict(sorted(r["outcomes"].items(), key=lambda kv: -kv[1])))

    if args.eval_run:
        ev = pd.read_parquet(cache / "features" / f"{args.eval_run}.parquet")
        ev = ev[ev["video_id"].str.startswith("train_")].reset_index(drop=True)
        ev_fold = ev["video_id"].map(folds)
        pe = pd.Series(index=ev.index, dtype=float)
        for k, model in models.items():
            if (ev_fold == k).any():
                pe[ev_fold == k] = predict(model, ev[ev_fold == k])
        pe = smooth(pe, ev)
        vids = sorted(ev["video_id"].unique())
        a = score(make_alarms(ev, pe >= thr, videos, confirm, m["driver_inside"]), gt, video_ids=vids)
        b = score(make_alarms(ev, rule_moving(ev, m["rule_theta"]), videos, m["confirm_rows"], m["driver_inside"]),
                  gt, video_ids=vids)
        print(f"\ntransfer to {args.eval_run} ({len(vids)} train videos): model F1 {a['f1']:.3f} "
              f"(P {a['precision']:.3f} tR {a['target_recall']:.3f})   rule F1 {b['f1']:.3f} "
              f"(P {b['precision']:.3f} tR {b['target_recall']:.3f})")

    final = train(feats)
    print("\nfeature importance (gain share, model on all train):")
    print(importance(final).round(3).to_string())
    if args.save:
        out = cache / "models"
        out.mkdir(parents=True, exist_ok=True)
        final.booster_.save_model(str(out / f"lgbm_{args.det_run}.txt"))
        (out / f"lgbm_{args.det_run}.json").write_text(json.dumps({"thr": thr, "confirm": confirm}))
        print(f"saved -> {out / f'lgbm_{args.det_run}.txt'}")


if __name__ == "__main__":
    main()
