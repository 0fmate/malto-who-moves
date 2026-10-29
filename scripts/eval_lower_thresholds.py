"""With the learned filter and the driver fix, can the motion thresholds go lower?

A lower threshold catches movers that move little (t90: 189 missed targets move < theta), but adds false alarms on
jittery parked cars and still people. Without the learned filter vehicle 2.0 lost (experiments.md, hand gate sweep);
the model may now remove those extra false alarms. Each setting: candidates, out-of-fold model (3 scene folds),
keep = hand gate AND P >= thr. Compared on t90 (strict / lenient / per fold) and s45, plus test alarms kept.

Usage: python scripts/eval_lower_thresholds.py
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
from whomoves.metric import score  # noqa: E402
from whomoves.validation import load_scenes, make_folds  # noqa: E402

SETTINGS = [(0.75, 2.5), (0.75, 2.0), (0.75, 1.5), (0.6, 2.5), (0.6, 2.0)]  # (person, others) box heights
P_GRID = [0.8, 0.85, 0.9]


def main() -> None:
    cfg = load_config()
    cache, driver = cfg["paths"]["cache_dir"], cfg["motion"]["driver_inside"]
    scenes = load_scenes(cache)
    targets = load_csv(cfg["paths"]["data_dir"], "train_alarms")
    weights = targets[targets["bonus"] == 0].groupby("video_id").size().reindex(scenes.index, fill_value=0)
    folds = make_folds(scenes, weights + 1, cfg["validation"]["n_folds"])
    sets = {n: pickle.loads((cache / sw.CLIP_SETS[n][1]).read_bytes()) for n in ["t90", "s45"]}
    tv = load_csv(cfg["paths"]["data_dir"], "test_videos")
    ft = pd.read_parquet(cache / "features" / f"{sw.TILED}.parquet")
    ft = ft[ft["video_id"].isin(tv["video_id"])]

    def hand(r):
        return ~r["weak"] | (r["score_mean"] >= 0.7)

    print("hand & model, F1: t90 strict | lenient | folds | s45 | test alarms")
    for tp, to in SETTINGS:
        data = {}
        for n, c in sets.items():
            r = L.candidate_rows(c["feats"], tp, driver, True, to)
            r["fold"] = r["video_id"].str.split("@").str[0].map(folds)
            data[n] = (r, c)
        rows, c = data["t90"]
        y = L.labels(rows, c["videos"], c["variants"]["strict"])
        models = {k: lgb.LGBMClassifier(**L.PARAMS).fit(L.inputs(rows[y.notna() & (rows["fold"] != k)]),
                                                         y[y.notna() & (rows["fold"] != k)].astype(int))
                  for k in sorted(folds.unique())}
        for r, _ in data.values():
            r["p"] = np.nan
            for k, m in models.items():
                sel = r["fold"] == k
                if sel.any():
                    r.loc[sel, "p"] = m.predict_proba(L.inputs(r[sel]))[:, 1]
        t = L.candidate_rows(ft, tp, driver, True, to)
        full = lgb.LGBMClassifier(**L.PARAMS).fit(L.inputs(rows[y.notna()]), y[y.notna()].astype(int))
        t["p"] = full.predict_proba(L.inputs(t))[:, 1]
        cf = c["videos"]["video_id"].str.split("@").str[0].map(folds)
        print(f"  person {tp} / others {to}: {len(rows)} t90 candidates ({int((y == 0).sum())} false), "
              f"{len(t)} test candidates")
        for thr in P_GRID:
            def f1(n, var="strict", fold=None):
                r, cc = data[n]
                keep = hand(r) & (r["p"] >= thr)
                vids = None if fold is None else cc["videos"]["video_id"][cf == fold]
                return score(rows_to_alarms(r[keep], cc["videos"]), cc["variants"][var], video_ids=vids)["f1"]
            per = "/".join(f"{f1('t90', fold=k):.3f}" for k in sorted(folds.unique()))
            print(f"    P >= {thr:<4} {f1('t90'):.3f} | {f1('t90', 'lenient'):.3f} | {per} | {f1('s45'):.3f} | "
                  f"{int((hand(t) & (t['p'] >= thr)).sum())}", flush=True)


if __name__ == "__main__":
    main()
