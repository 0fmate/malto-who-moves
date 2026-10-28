"""Learned alarm filter on test-like clips of the tiled detector: does it beat the hand-written gate?

The hand-written gate (sweep_lookahead.py) drops an alarm when its track moves little over its whole life
(< 2 box heights for people, < 4 for vehicles) AND the detector is unsure (mean score < 0.7). Here a LightGBM
model learns the same decision from more inputs: it scores each alarm of the rule as "lands on a mover" (hit or
free duplicate) vs "false alarm", from the features of the alarm row and the whole-track measures.

Differences from train_filter.py (dropped on 2026-10-27): trained on the tiled detector, whose false alarms look
like the test ones, and on 35-120 s clips like the test videos; whole-track measures are added.

Out-of-fold: 3 scene folds, so a clip is always scored by a model that never saw its scene.
Decisions compared: drop when P < thr (every alarm), or only weak alarms can be dropped.

Usage: python scripts/learn_alarm_filter.py [--theta-person 0.75]
"""

import argparse
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sweep_lookahead as sw  # noqa: E402
from whomoves.alarms import alarm_rows, rows_to_alarms, rule_moving  # noqa: E402
from whomoves.config import load_config  # noqa: E402
from whomoves.io import load_csv  # noqa: E402
from whomoves.metric import score  # noqa: E402
from whomoves.motion_model import model_inputs  # noqa: E402
from whomoves.validation import load_scenes, make_folds  # noqa: E402

KEY = ["video_id", "track_id"]
MEASURES = ["straight", "jump_share", "size_jump", "hold"]  # score_mean and track_range are already features
POSITIVE = {"hit", "hit_wrong_class", "duplicate"}
THRESHOLDS = np.r_[np.round(np.arange(0.2, 0.96, 0.05), 2), 0.97, 0.98]
PARAMS = dict(n_estimators=200, learning_rate=0.05, num_leaves=7, min_child_samples=20,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=42, verbose=-1)


def candidate_rows(feats: pd.DataFrame, theta_person: float, driver: float) -> pd.DataFrame:
    """Rule alarm rows (index kept) + whole-track measures + weak flag."""
    rows = alarm_rows(feats, rule_moving(feats, sw.THETA, theta_person), sw.CONFIRM, driver)
    alarmed = feats.merge(rows[KEY + ["t"]].rename(columns={"t": "t_alarm"}), on=KEY)
    m = []
    for (vid, tid), g in alarmed.sort_values(KEY + ["t"], kind="stable").groupby(KEY, sort=False):
        theta = theta_person if g["class"].iloc[0] == "person" else sw.THETA
        m.append({"video_id": vid, "track_id": tid, **sw.track_measures(g, g["t_alarm"].iloc[0], theta)})
    m = pd.DataFrame(m).drop(columns=["score_mean", "total"])
    out = rows.reset_index(names="_row").merge(m, on=KEY, how="left").set_index("_row")
    out.index.name = None
    out["weak"] = out["track_range"] < np.where(out["class"] == "person", 2.0, 4.0)
    return out


DROP: list[str] = []  # input columns left out (--drop)


def inputs(rows: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([model_inputs(rows), rows[MEASURES]], axis=1).drop(columns=DROP)


def labels(rows: pd.DataFrame, videos: pd.DataFrame, gt) -> pd.Series:
    """1 = lands on a mover, 0 = false alarm, NaN = neutral (left out of training)."""
    d = score(rows_to_alarms(rows, videos), gt)["alarms"].set_index("alarm_index")["outcome"]
    return d.map(lambda o: 1.0 if o in POSITIVE else (np.nan if o == "neutral" else 0.0)).reindex(rows.index)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--theta-person", type=float, default=1.0)
    parser.add_argument("--drop", nargs="*", default=[], help="input columns to leave out, e.g. h_rel")
    args = parser.parse_args()
    DROP[:] = args.drop
    cfg = load_config()
    cache, driver = cfg["paths"]["cache_dir"], cfg["motion"]["driver_inside"]

    scenes = load_scenes(cache)
    targets = load_csv(cfg["paths"]["data_dir"], "train_alarms")
    weights = targets[targets["bonus"] == 0].groupby("video_id").size().reindex(scenes.index, fill_value=0)
    folds = make_folds(scenes, weights + 1, cfg["validation"]["n_folds"])

    data = {}
    for name in ["t90", "s45"]:
        c = pickle.loads((cache / sw.CLIP_SETS[name][1]).read_bytes())
        rows = candidate_rows(c["feats"], args.theta_person, driver)
        rows["fold"] = rows["video_id"].str.split("@").str[0].map(folds)
        data[name] = (rows, c["videos"], c["variants"])
    rows, videos, variants = data["t90"]
    y = labels(rows, videos, variants["strict"])
    print(f"t90 candidates: {len(rows)} ({int((y == 1).sum())} on movers, {int((y == 0).sum())} false, "
          f"{int(y.isna().sum())} neutral); false among weak: {int(((y == 0) & rows['weak']).sum())}")

    models = {}
    for k in sorted(folds.unique()):
        tr = y.notna() & (rows["fold"] != k)
        models[k] = lgb.LGBMClassifier(**PARAMS).fit(inputs(rows[tr]), y[tr].astype(int))
    for name, (r, _, _) in data.items():  # out-of-fold probabilities
        r["p"] = np.nan
        for k, model in models.items():
            sel = r["fold"] == k
            if sel.any():
                r.loc[sel, "p"] = model.predict_proba(inputs(r[sel]))[:, 1]

    def f1(name, keep, variant="strict", fold=None):
        r, v, g = data[name]
        vids = None if fold is None else sorted(v["video_id"][v["video_id"].str.split("@").str[0].map(folds) == fold])
        return score(rows_to_alarms(r[keep], v), g[variant], video_ids=vids)["f1"]

    def hand(r):
        return ~r["weak"] | (r["score_mean"] >= 0.7)

    print("\nF1 on clips (strict): t90 | t90 lenient | s45")
    print(f"  no filter        {f1('t90', np.ones(len(rows), bool)):.3f} | "
          f"{f1('t90', np.ones(len(rows), bool), 'lenient'):.3f} | {f1('s45', np.ones(len(data['s45'][0]), bool)):.3f}")
    print(f"  hand gate        {f1('t90', hand(rows)):.3f} | {f1('t90', hand(rows), 'lenient'):.3f} | "
          f"{f1('s45', hand(data['s45'][0])):.3f}")
    best = {}
    keep_rule = {
        "all": lambda r, thr: r["p"] >= thr,
        "weak only": lambda r, thr: (r["p"] >= thr) | ~r["weak"],
        "and hand": lambda r, thr: (r["p"] >= thr) & hand(r),  # dropped if either filter drops it
    }
    for mode, rule in keep_rule.items():
        res = []
        for thr in THRESHOLDS:
            keeps = {n: rule(r, thr) for n, (r, _, _) in data.items()}
            res.append((thr, f1("t90", keeps["t90"]), f1("t90", keeps["t90"], "lenient"), f1("s45", keeps["s45"]),
                        int((~keeps["t90"]).sum())))
        res = pd.DataFrame(res, columns=["thr", "t90", "t90_len", "s45", "dropped"])
        b = res.loc[res["t90"].idxmax()]
        best[mode] = b["thr"]
        print(f"  model ({mode:9s}) best thr {b['thr']:.2f}: {b['t90']:.3f} | {b['t90_len']:.3f} | {b['s45']:.3f}"
              f"  (drops {int(b['dropped'])} of {len(rows)})")
        print("    curve t90: " + "  ".join(f"{t:.2f}:{v:.3f}" for t, v in zip(res["thr"], res["t90"]) if t >= 0.6))

    print("\nper fold (t90 strict): no filter / hand gate / model all / model weak-only")
    for k in sorted(folds.unique()):
        ka = rows["p"] >= best["all"]
        kw = (rows["p"] >= best["weak only"]) | ~rows["weak"]
        print(f"  fold {k}: {f1('t90', np.ones(len(rows), bool), fold=k):.3f} / {f1('t90', hand(rows), fold=k):.3f} / "
              f"{f1('t90', ka, fold=k):.3f} / {f1('t90', kw, fold=k):.3f}")

    use = y.notna()
    final = lgb.LGBMClassifier(**PARAMS).fit(inputs(rows[use]), y[use].astype(int))
    gain = final.booster_.feature_importance("gain")
    print("\nfeature importance (gain share):")
    print(pd.Series(gain / gain.sum(), index=inputs(rows).columns).sort_values(ascending=False).head(8).round(3).to_string())


if __name__ == "__main__":
    main()
