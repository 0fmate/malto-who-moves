"""Is the hand gate still needed now that the learned filter decides (confirm 2)?

Hand gate: drop when the track moves little over its whole life (< 2 person / < 4 others box heights) AND the mean
detector score is < 0.7. It was written before the model; the model sees the same inputs (track_range, score_mean).
Compared, person 0.75 / others 2.5, confirm 2, driver fix, out-of-fold model (3 scene folds):
hand AND P >= thr (current, thr 0.9) vs P >= thr alone. t90 strict / lenient / folds / s45 / test alarms, and a
bootstrap over scenes of the gain vs the current method.

Usage: python scripts/eval_no_hand.py
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
from robustness_check import N_BOOT, SEED, clip_counts, f1  # noqa: E402
from whomoves.alarms import rows_to_alarms  # noqa: E402
from whomoves.config import load_config  # noqa: E402
from whomoves.io import load_csv  # noqa: E402
from whomoves.metric import score  # noqa: E402
from whomoves.validation import load_scenes, make_folds  # noqa: E402

THETA_PERSON, CONFIRM = 0.75, 2
P_GRID = [0.85, 0.9, 0.92, 0.94, 0.96]


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

    data = {}
    for n, c in sets.items():
        r = L.candidate_rows(c["feats"], THETA_PERSON, driver, True, confirm=CONFIRM)
        r["fold"] = r["video_id"].str.split("@").str[0].map(folds)
        data[n] = (r, c)
    rows, c = data["t90"]
    y = L.labels(rows, c["videos"], c["variants"]["strict"])
    for r, _ in data.values():
        r["p"] = np.nan
    for k in sorted(folds.unique()):
        tr = y.notna() & (rows["fold"] != k)
        m = lgb.LGBMClassifier(**L.PARAMS).fit(L.inputs(rows[tr]), y[tr].astype(int))
        for r, _ in data.values():
            sel = r["fold"] == k
            if sel.any():
                r.loc[sel, "p"] = m.predict_proba(L.inputs(r[sel]))[:, 1]
    t = L.candidate_rows(ft, THETA_PERSON, driver, True, confirm=CONFIRM)
    t["p"] = lgb.LGBMClassifier(**L.PARAMS).fit(L.inputs(rows[y.notna()]), y[y.notna()].astype(int)) \
        .predict_proba(L.inputs(t))[:, 1]

    def hand(r):
        return ~r["weak"] | (r["score_mean"] >= 0.7)

    rules = {"hand & P": lambda r, thr: hand(r) & (r["p"] >= thr), "P alone": lambda r, thr: r["p"] >= thr}
    cf = c["videos"]["video_id"].str.split("@").str[0].map(folds)

    def f1s(n, keep, var="strict", fold=None):
        r, cc = data[n]
        vids = None if fold is None else cc["videos"]["video_id"][cf == fold]
        return score(rows_to_alarms(r[keep], cc["videos"]), cc["variants"][var], video_ids=vids)["f1"]

    print(f"confirm {CONFIRM}: t90 strict | lenient | folds | s45 | test alarms")
    for name, rule in rules.items():
        for thr in P_GRID:
            k = rule(rows, thr)
            per = "/".join(f"{f1s('t90', k, fold=q):.3f}" for q in sorted(folds.unique()))
            print(f"  {name:9s} P >= {thr:<4} {f1s('t90', k):.3f} | {f1s('t90', k, 'lenient'):.3f} | {per} | "
                  f"{f1s('s45', rule(data['s45'][0], thr)):.3f} | {int(rule(t, thr).sum())}", flush=True)
    drop = hand(rows) == False  # noqa: E712
    print(f"hand gate drops {int(drop.sum())} t90 candidates: {int((y[drop] == 1).sum())} on movers, "
          f"{int((y[drop] == 0).sum())} false; their P quartiles "
          f"{np.round(rows.loc[drop, 'p'].quantile([0.25, 0.5, 0.75]).to_numpy(), 3).tolist()}")

    gt, videos = c["variants"]["strict"], c["videos"]
    base = clip_counts(rows, rules["hand & P"](rows, 0.9), videos, gt)
    clip_scene = pd.Series(videos["video_id"].str.split("@").str[0].map(scenes).to_numpy(), index=videos["video_id"])
    by_scene = {s: clip_scene.index[clip_scene == s] for s in clip_scene.unique()}
    rng = np.random.default_rng(SEED)
    print(f"\nbootstrap over scenes ({N_BOOT} draws), gain of 'P alone' vs 'hand & P >= 0.9': mean [5%, 95%], share > 0")
    for thr in [0.9, 0.92, 0.94]:
        alt = clip_counts(rows, rules["P alone"](rows, thr), videos, gt)
        g = []
        for _ in range(N_BOOT):
            ids = np.concatenate([by_scene[s] for s in rng.choice(list(by_scene), len(by_scene))])
            g.append(f1(alt.loc[ids]) - f1(base.loc[ids]))
        g = np.array(g)
        print(f"  P >= {thr:<4} {g.mean():+.4f} [{np.quantile(g, 0.05):+.4f}, {np.quantile(g, 0.95):+.4f}]  {(g > 0).mean():.0%}")


if __name__ == "__main__":
    main()
