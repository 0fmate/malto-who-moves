"""How robust are the filter gains, and how noisy is a leaderboard computed on ~30% of the test?

Filters compared on the tiled test-like clips (t90, person 0.75 / others 2.5, confirm 5):
none, hand gate (weak track AND mean score < 0.7 -> drop), hand gate AND learned model (P >= 0.85, out-of-fold).

1. Per scene fold: does each filter help in every fold, or only on average?
2. Bootstrap over scenes (resample whole scenes with replacement): spread of the F1 gain of each filter.
3. Public-like subsets: random draws of N_PUBLIC clips (the public LB is ~30% of 48 test videos, ~15 clips).
   Spread of F1 on such a subset, and how often the ranking of two filters flips compared to the full set.

Usage: python scripts/robustness_check.py
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
from whomoves.alarms import rows_to_alarms  # noqa: E402
from whomoves.config import load_config  # noqa: E402
from whomoves.io import load_csv  # noqa: E402
from whomoves.metric import score  # noqa: E402
from whomoves.validation import load_scenes, make_folds  # noqa: E402

THETA_PERSON, P_THR = 0.75, 0.85
N_BOOT, N_PUBLIC, SEED = 2000, 15, 0


def clip_counts(rows: pd.DataFrame, keep: pd.Series, videos: pd.DataFrame, gt) -> pd.DataFrame:
    """Per clip: points, hits, weighted false alarms, targets (F1 of any set of clips = from the summed counts)."""
    s = score(rows_to_alarms(rows[keep], videos), gt)
    d = s["alarms"].assign(hit=lambda x: x["outcome"].isin(["hit", "hit_wrong_class"]))
    c = d.groupby("video_id").agg(points=("points", "sum"), hits=("hit", "sum"), fa=("fa_weight", "sum"))
    c["targets"] = s["targets"][~s["targets"]["bonus"]].groupby("video_id").size()
    return c.reindex(videos["video_id"]).fillna(0.0)


def f1(c: pd.DataFrame) -> float:
    pts, den, tgt = c["points"].sum(), c["hits"].sum() + c["fa"].sum(), c["targets"].sum()
    p, r = pts / max(den, 1e-9), min(pts / max(tgt, 1e-9), 1.0)
    return 2 * p * r / max(p + r, 1e-9)


def main() -> None:
    cfg = load_config()
    cache, driver = cfg["paths"]["cache_dir"], cfg["motion"]["driver_inside"]
    scenes = load_scenes(cache)
    targets = load_csv(cfg["paths"]["data_dir"], "train_alarms")
    weights = targets[targets["bonus"] == 0].groupby("video_id").size().reindex(scenes.index, fill_value=0)
    folds = make_folds(scenes, weights + 1, cfg["validation"]["n_folds"])

    c = pickle.loads((cache / "clips90_t2x2.pkl").read_bytes())
    videos, gt = c["videos"], c["variants"]["strict"]
    rows = L.candidate_rows(c["feats"], THETA_PERSON, driver)
    rows["fold"] = rows["video_id"].str.split("@").str[0].map(folds)
    y = L.labels(rows, videos, gt)
    p = pd.Series(np.nan, index=rows.index)
    for k in sorted(folds.unique()):
        tr = y.notna() & (rows["fold"] != k)
        m = lgb.LGBMClassifier(**L.PARAMS).fit(L.inputs(rows[tr]), y[tr].astype(int))
        sel = rows["fold"] == k
        p[sel] = m.predict_proba(L.inputs(rows[sel]))[:, 1]
    hand = ~rows["weak"] | (rows["score_mean"] >= 0.7)
    filters = {"none": pd.Series(True, index=rows.index), "hand": hand, "hand+model": hand & (p >= P_THR)}
    counts = {n: clip_counts(rows, k, videos, gt) for n, k in filters.items()}

    base = videos["video_id"].str.split("@").str[0]
    clip_scene = pd.Series(base.map(scenes).to_numpy(), index=videos["video_id"])
    clip_fold = pd.Series(base.map(folds).to_numpy(), index=videos["video_id"])
    print(f"{len(videos)} clips, {clip_scene.nunique()} scenes, {int(counts['none']['targets'].sum())} targets")
    print("\nF1 per scene fold:       " + "  ".join(f"{n:>10s}" for n in filters))
    for k in sorted(clip_fold.unique()):
        ids = clip_fold.index[clip_fold == k]
        print(f"  fold {k} ({len(ids):3d} clips)      " + "  ".join(f"{f1(c.loc[ids]):10.3f}" for c in counts.values()))
    print("  all                     " + "  ".join(f"{f1(c):10.3f}" for c in counts.values()))

    rng = np.random.default_rng(SEED)
    scene_ids = clip_scene.unique()
    by_scene = {s: clip_scene.index[clip_scene == s] for s in scene_ids}
    gains = {n: [] for n in ["hand - none", "hand+model - hand", "hand+model - none"]}
    for _ in range(N_BOOT):
        ids = np.concatenate([by_scene[s] for s in rng.choice(scene_ids, len(scene_ids))])
        f = {n: f1(c.loc[ids]) for n, c in counts.items()}
        gains["hand - none"].append(f["hand"] - f["none"])
        gains["hand+model - hand"].append(f["hand+model"] - f["hand"])
        gains["hand+model - none"].append(f["hand+model"] - f["none"])
    print(f"\nbootstrap over {len(scene_ids)} scenes ({N_BOOT} draws): F1 gain, mean [5%, 95%], share of draws > 0")
    for n, g in gains.items():
        g = np.array(g)
        print(f"  {n:18s} {g.mean():+.3f} [{np.quantile(g, 0.05):+.3f}, {np.quantile(g, 0.95):+.3f}]  {(g > 0).mean():.0%}")

    print(f"\npublic-like subsets: {N_BOOT} random draws of {N_PUBLIC} clips")
    sub = {n: [] for n in counts}
    for _ in range(N_BOOT):
        ids = rng.choice(videos["video_id"], N_PUBLIC, replace=False)
        for n, c in counts.items():
            sub[n].append(f1(c.loc[ids]))
    sub = pd.DataFrame(sub)
    for n in counts:
        print(f"  {n:10s} F1 on full set {f1(counts[n]):.3f}; on a subset: std {sub[n].std():.3f}, "
              f"5-95% [{sub[n].quantile(0.05):.3f}, {sub[n].quantile(0.95):.3f}]")
    for a, b in [("hand", "none"), ("hand+model", "hand")]:
        d = sub[a] - sub[b]
        print(f"  {a} vs {b}: full-set gain {f1(counts[a]) - f1(counts[b]):+.3f}; on a subset std {d.std():.3f}, "
              f"subset says the opposite (or tie) in {(d <= 0).mean():.0%} of draws")


if __name__ == "__main__":
    main()
