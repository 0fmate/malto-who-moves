"""Alarm filter: the rule proposes candidate alarms, a LightGBM model decides which ones to keep.

Why: a per-frame "is it moving?" model alarms at the first frame where it says yes, so one wrong
frame among thousands of a parked car is enough for a false alarm (see experiments.md). Here the
unit is the alarm itself, exactly what the metric scores:
1. candidates: the distance rule with a looser threshold (THETA_LOW), so few real movers are lost;
2. label of each candidate, from the metric on the training videos: 1 if it hits a target (or a
   target already hit, which is free), 0 if it is a false alarm; neutral ones are left out;
3. LightGBM on the features of the candidate row, with scene folds; keep candidates with P >= THR.

Usage: python scripts/train_filter.py [--det-run fasterrcnn_resnet50_fpn_v2_s1]
                                      [--eval-run fasterrcnn_resnet50_fpn_v2_s1_t2x2] [--save]
"""

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whomoves.alarms import alarm_rows, rows_to_alarms, rule_moving  # noqa: E402
from whomoves.config import load_config  # noqa: E402
from whomoves.io import load_csv  # noqa: E402
from whomoves.metric import GroundTruth, score  # noqa: E402
from whomoves.motion_model import model_columns, model_inputs  # noqa: E402
from whomoves.validation import load_scenes, make_folds  # noqa: E402

THETAS_LOW = [0.75, 1.0, 1.5]
THRESHOLDS = np.round(np.arange(0.5, 1.0, 0.02), 2)
PARAMS = dict(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=20,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=42, verbose=-1)
POSITIVE = {"hit", "hit_wrong_class", "duplicate"}


def candidates(feats, theta, m):
    return alarm_rows(feats, rule_moving(feats, theta), m["confirm_rows"], m["driver_inside"])


def labels(rows, videos, gt):
    """1 = the candidate lands on a mover, 0 = false alarm, NaN = neutral (left out)."""
    d = score(rows_to_alarms(rows, videos), gt, video_ids=sorted(rows["video_id"].unique()))["alarms"]
    out = d.set_index("alarm_index")["outcome"]
    return out.map(lambda o: 1.0 if o in POSITIVE else (np.nan if o == "neutral" else 0.0)).reindex(rows.index)


def f1_at(rows, p, thr, videos, gt, vids):
    return score(rows_to_alarms(rows[p >= thr], videos), gt, video_ids=vids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--det-run", default="fasterrcnn_resnet50_fpn_v2_s1")
    parser.add_argument("--eval-run")
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
    all_vids = list(folds.index)
    rule = score(rows_to_alarms(candidates(feats, m["rule_theta"], m), videos), gt)
    print(f"rule theta={m['rule_theta']}: F1 {rule['f1']:.3f} P {rule['precision']:.3f} tR {rule['target_recall']:.3f}\n")

    best = None
    for theta in THETAS_LOW:
        rows = candidates(feats, theta, m)
        y = labels(rows, videos, gt)
        use = y.notna()
        fold = rows["video_id"].map(folds)
        p = pd.Series(np.nan, index=rows.index)
        models = {}
        for k in sorted(folds.unique()):
            tr, te = use & (fold != k), fold == k
            models[k] = lgb.LGBMClassifier(**PARAMS).fit(model_inputs(rows[tr]), y[tr].astype(int))
            p[te] = models[k].predict_proba(model_inputs(rows[te]))[:, 1]
        cand = score(rows_to_alarms(rows, videos), gt)
        f1s = [f1_at(rows, p, thr, videos, gt, all_vids)["f1"] for thr in THRESHOLDS]
        i = int(np.argmax(f1s))
        print(f"candidates theta={theta}: {len(rows)} ({int(y.sum())} on movers, {int((y == 0).sum())} false), "
              f"candidate recall {cand['target_recall']:.3f}; best out-of-fold F1 {f1s[i]:.3f} at P >= {THRESHOLDS[i]}")
        if best is None or f1s[i] > best["f1"]:
            best = dict(f1=f1s[i], theta=theta, thr=float(THRESHOLDS[i]), rows=rows, p=p, y=y, models=models)

    theta, thr, rows, p = best["theta"], best["thr"], best["rows"], best["p"]
    print(f"\nchosen: candidates theta={theta}, keep P >= {thr}")
    for name, vids in [("all", all_vids)] + [(f"fold {k}", list(v)) for k, v in folds.groupby(folds).groups.items()]:
        a = f1_at(rows, p, thr, videos, gt, vids)
        b = score(rows_to_alarms(candidates(feats, m["rule_theta"], m), videos), gt, video_ids=vids)
        print(f"  {name:7s} filter F1 {a['f1']:.3f} P {a['precision']:.3f} tR {a['target_recall']:.3f}   "
              f"rule F1 {b['f1']:.3f} P {b['precision']:.3f} tR {b['target_recall']:.3f}")
    a = f1_at(rows, p, thr, videos, gt, all_vids)
    print("  filter outcomes:", dict(sorted(a["outcomes"].items(), key=lambda kv: -kv[1])))

    if args.eval_run:
        ev = pd.read_parquet(cache / "features" / f"{args.eval_run}.parquet")
        ev = ev[ev["video_id"].str.startswith("train_")].reset_index(drop=True)
        er = candidates(ev, theta, m)
        pe = pd.Series(np.nan, index=er.index)
        for k, model in best["models"].items():
            sel = er["video_id"].map(folds) == k
            if sel.any():
                pe[sel] = model.predict_proba(model_inputs(er[sel]))[:, 1]
        vids = sorted(ev["video_id"].unique())
        a = f1_at(er, pe, thr, videos, gt, vids)
        b = score(rows_to_alarms(candidates(ev, m["rule_theta"], m), videos), gt, video_ids=vids)
        print(f"\ntransfer to {args.eval_run} ({len(vids)} videos): filter F1 {a['f1']:.3f} "
              f"(P {a['precision']:.3f} tR {a['target_recall']:.3f})   rule F1 {b['f1']:.3f} "
              f"(P {b['precision']:.3f} tR {b['target_recall']:.3f})")

    use = best["y"].notna()
    final = lgb.LGBMClassifier(**PARAMS).fit(model_inputs(rows[use]), best["y"][use].astype(int))
    gain = final.booster_.feature_importance("gain")
    print("\nfeature importance (gain share):")
    print(pd.Series(gain / gain.sum(), index=model_columns()).sort_values(ascending=False).head(8).round(3).to_string())
    if args.save:
        out = cache / "models"
        out.mkdir(parents=True, exist_ok=True)
        final.booster_.save_model(str(out / f"filter_{args.det_run}.txt"))
        (out / f"filter_{args.det_run}.json").write_text(json.dumps({"theta": theta, "thr": thr}))
        print(f"saved -> {out / f'filter_{args.det_run}.txt'}")


if __name__ == "__main__":
    main()
