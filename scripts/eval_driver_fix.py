"""Driver suppression: drop the whole track, or only wait until the person is out of the vehicle box?

Current rule (alarms.alarm_rows): the alarm row is the first confirmed moving row; if at that row a person box is
covered >= driver_inside by a vehicle box, the track never alarms. In parking lots a person walking behind a parked
car is covered at that moment and is then lost for good (t90: 388 such tracks, only 2 annotated drivers).
Fix: a person row inside a vehicle box does not count as moving, so the alarm comes at the first confirmed moving row
outside vehicle boxes. A real driver stays inside the vehicle box for the whole track and still never alarms.

Compared on t90 (strict / lenient / per fold) and s45, hand gate and hand gate AND learned model (out-of-fold).

Usage: python scripts/eval_driver_fix.py
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

P_THR = 0.85


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

    print("F1: t90 strict | lenient | folds | s45 | test alarms")
    for tp in [1.0, 0.75]:
        for fix in [False, True]:
            data = {}
            for n, c in sets.items():
                r = L.candidate_rows(c["feats"], tp, driver, fix)
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
            t = L.candidate_rows(ft, tp, driver, fix)
            full = lgb.LGBMClassifier(**L.PARAMS).fit(L.inputs(rows[y.notna()]), y[y.notna()].astype(int))
            t["p"] = full.predict_proba(L.inputs(t))[:, 1]
            cf = c["videos"]["video_id"].str.split("@").str[0].map(folds)
            for name, rule in {"hand": hand, "hand & model": lambda r: hand(r) & (r["p"] >= P_THR)}.items():
                def f1(n, var="strict", fold=None):
                    r, cc = data[n]
                    vids = None if fold is None else cc["videos"]["video_id"][cf == fold]
                    return score(rows_to_alarms(r[rule(r)], cc["videos"]), cc["variants"][var], video_ids=vids)["f1"]
                per = "/".join(f"{f1('t90', fold=k):.3f}" for k in sorted(folds.unique()))
                print(f"  person {tp} {'fix' if fix else 'old'} {name:13s} {f1('t90'):.3f} | {f1('t90', 'lenient'):.3f} | "
                      f"{per} | {f1('s45'):.3f} | {int(rule(t).sum())}")


if __name__ == "__main__":
    main()
