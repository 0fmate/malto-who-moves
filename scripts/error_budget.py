"""Where the final method still loses points on t90, and how much each kind of error is worth.

Method: person 0.75 / others 2.5, confirm 5, driver fix, hand gate AND out-of-fold model P >= 0.85.
For each kind of error: its count and the F1 if that kind alone were fixed (an upper bound on what an idea that
targets it can gain). Also the class confusion of wrong-class hits and the missed targets that a filtered-out
candidate would have hit.

Usage: python scripts/error_budget.py
"""

import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import learn_alarm_filter as L  # noqa: E402
import sweep_lookahead as sw  # noqa: E402
from whomoves.alarms import rows_to_alarms  # noqa: E402
from whomoves.config import load_config  # noqa: E402
from whomoves.io import load_csv  # noqa: E402
from whomoves.metric import WRONG_CLASS_POINTS, score  # noqa: E402
from whomoves.validation import load_scenes, make_folds  # noqa: E402

THETA_PERSON, P_THR = 0.75, 0.85


def f1_of(points, hits, fa, n_targets):
    p, r = points / (hits + fa), min(points / n_targets, 1.0)
    return 2 * p * r / (p + r)


def main() -> None:
    cfg = load_config()
    cache, driver = cfg["paths"]["cache_dir"], cfg["motion"]["driver_inside"]
    scenes = load_scenes(cache)
    targets = load_csv(cfg["paths"]["data_dir"], "train_alarms")
    weights = targets[targets["bonus"] == 0].groupby("video_id").size().reindex(scenes.index, fill_value=0)
    folds = make_folds(scenes, weights + 1, cfg["validation"]["n_folds"])
    c = pickle.loads((cache / sw.CLIP_SETS["t90"][1]).read_bytes())
    videos, gt = c["videos"], c["variants"]["strict"]

    rows = L.candidate_rows(c["feats"], THETA_PERSON, driver, True)
    rows["fold"] = rows["video_id"].str.split("@").str[0].map(folds)
    y = L.labels(rows, videos, gt)
    rows["p"] = np.nan
    for k in sorted(folds.unique()):
        tr = y.notna() & (rows["fold"] != k)
        m = lgb.LGBMClassifier(**L.PARAMS).fit(L.inputs(rows[tr]), y[tr].astype(int))
        sel = rows["fold"] == k
        rows.loc[sel, "p"] = m.predict_proba(L.inputs(rows[sel]))[:, 1]
    keep = (~rows["weak"] | (rows["score_mean"] >= 0.7)) & (rows["p"] >= P_THR)

    s = score(rows_to_alarms(rows[keep], videos), gt)
    a, tg = s["alarms"], s["targets"]
    pts, hits, fa, n = s["points"], s["n_hits"], s["fa_weighted"], s["n_targets"]
    print(f"F1 {s['f1']:.4f}  P {s['precision']:.3f}  R {s['recall']:.3f}  points {pts:.1f}  hits {hits}  "
          f"fa weighted {fa:.0f}  targets {n}  missed {s['n_missed']}")
    print("outcomes:", a["outcome"].value_counts().to_dict())

    print("\nF1 if one kind of error were fixed:")
    for o in sorted(a["outcome"].unique()):
        if o.startswith("fa_"):
            w = a.loc[a["outcome"] == o, "fa_weight"].sum()
            print(f"  remove {o:16s} n={int((a['outcome'] == o).sum()):4d} weight {w:4.0f} -> {f1_of(pts, hits, fa - w, n):.4f}")
    nw = int((a["outcome"] == "hit_wrong_class").sum())
    print(f"  right class for wrong-class hits n={nw} -> {f1_of(pts + nw * (1 - WRONG_CLASS_POINTS), hits, fa, n):.4f}")
    miss = tg[~tg["hit"] & ~tg["bonus"]]
    for cls, g in miss.groupby("class"):
        print(f"  hit all missed {cls:8s} n={len(g):4d} -> {f1_of(pts + len(g), hits + len(g), fa, n):.4f}")

    wc = a[a["outcome"] == "hit_wrong_class"].merge(tg[["video_id", "track_id", "class"]], on=["video_id", "track_id"],
                                                     suffixes=("_pred", "_gt"))
    print("\nwrong-class hits, predicted -> true:", wc.groupby(["class_pred", "class_gt"]).size().to_dict())

    # missed targets that some candidate dropped by the filters would have hit
    s_all = score(rows_to_alarms(rows, videos), gt)
    t_all = s_all["targets"].set_index(["video_id", "track_id"])["hit"]
    lost = miss.set_index(["video_id", "track_id"]).index
    print(f"\nmissed targets hit by an unfiltered candidate: {int(t_all.reindex(lost).sum())} of {len(miss)}; "
          f"unfiltered F1 {s_all['f1']:.4f}")
    fb = a[a["outcome"] == "fa_background"].merge(rows[["video_id", "track_id", "h", "score_mean", "p"]]
                                                  .reset_index(names="alarm_index"), on=["alarm_index", "video_id"])
    hv = videos.set_index("video_id")["height"]
    fb["h_rel"] = fb["h"] / fb["video_id"].map(hv)
    print("fa_background: class", fb["class"].value_counts().to_dict(),
          " h_rel quartiles", np.round(fb["h_rel"].quantile([0.25, 0.5, 0.75]).to_numpy(), 3).tolist(),
          " p quartiles", np.round(fb["p"].quantile([0.25, 0.5, 0.75]).to_numpy(), 3).tolist())


if __name__ == "__main__":
    main()
