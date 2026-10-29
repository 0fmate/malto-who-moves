"""Is the gain of a shorter confirmation (eval_lower_thresholds.py --confirm) robust, or a lucky pick from the grid?

Current method: confirm 5, P >= 0.85. Shorter: confirm 4 / 2 / 1 with P >= 0.9 (person 0.75 / others 2.5, driver fix,
hand gate AND out-of-fold model). Paired on the same t90 clips: F1 gain per scene (6 scenes) and bootstrap over
scenes (share of draws where the shorter confirmation wins). Also what the extra alarms are.

Usage: python scripts/eval_confirm_robust.py
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

THETA_PERSON = 0.75
CONFIGS = {"c5 P.85": (5, 0.85), "c4 P.9": (4, 0.9), "c2 P.9": (2, 0.9), "c1 P.9": (1, 0.9)}


def main() -> None:
    cfg = load_config()
    cache, driver = cfg["paths"]["cache_dir"], cfg["motion"]["driver_inside"]
    scenes = load_scenes(cache)
    targets = load_csv(cfg["paths"]["data_dir"], "train_alarms")
    weights = targets[targets["bonus"] == 0].groupby("video_id").size().reindex(scenes.index, fill_value=0)
    folds = make_folds(scenes, weights + 1, cfg["validation"]["n_folds"])
    c = pickle.loads((cache / sw.CLIP_SETS["t90"][1]).read_bytes())
    videos, gt = c["videos"], c["variants"]["strict"]

    counts, outcomes = {}, {}
    for name, (confirm, thr) in CONFIGS.items():
        sw.CONFIRM = confirm  # read by L.candidate_rows
        rows = L.candidate_rows(c["feats"], THETA_PERSON, driver, True)
        rows["fold"] = rows["video_id"].str.split("@").str[0].map(folds)
        y = L.labels(rows, videos, gt)
        p = pd.Series(np.nan, index=rows.index)
        for k in sorted(folds.unique()):
            tr = y.notna() & (rows["fold"] != k)
            m = lgb.LGBMClassifier(**L.PARAMS).fit(L.inputs(rows[tr]), y[tr].astype(int))
            sel = rows["fold"] == k
            p[sel] = m.predict_proba(L.inputs(rows[sel]))[:, 1]
        keep = (~rows["weak"] | (rows["score_mean"] >= 0.7)) & (p >= thr)
        counts[name] = clip_counts(rows, keep, videos, gt)
        outcomes[name] = score(rows_to_alarms(rows[keep], videos), gt)["outcomes"]

    clip_scene = pd.Series(videos["video_id"].str.split("@").str[0].map(scenes).to_numpy(), index=videos["video_id"])
    scene_ids = clip_scene.unique()
    by_scene = {s: clip_scene.index[clip_scene == s] for s in scene_ids}
    base = "c5 P.85"
    print("F1 per scene (targets):  " + "  ".join(f"{n:>8s}" for n in CONFIGS))
    for s in scene_ids:
        ids = by_scene[s]
        print(f"  scene {s!s:12.12s} ({int(counts[base].loc[ids, 'targets'].sum()):4d})  "
              + "  ".join(f"{f1(cc.loc[ids]):8.3f}" for cc in counts.values()))
    print("  all                       " + "  ".join(f"{f1(cc):8.3f}" for cc in counts.values()))

    rng = np.random.default_rng(SEED)
    gains = {n: [] for n in CONFIGS if n != base}
    for _ in range(N_BOOT):
        ids = np.concatenate([by_scene[s] for s in rng.choice(scene_ids, len(scene_ids))])
        f0 = f1(counts[base].loc[ids])
        for n in gains:
            gains[n].append(f1(counts[n].loc[ids]) - f0)
    print(f"\nbootstrap over scenes ({N_BOOT} draws), gain vs {base}: mean [5%, 95%], share > 0")
    for n, g in gains.items():
        g = np.array(g)
        print(f"  {n:8s} {g.mean():+.4f} [{np.quantile(g, 0.05):+.4f}, {np.quantile(g, 0.95):+.4f}]  {(g > 0).mean():.0%}")

    print("\noutcomes:")
    keys = ["hit", "hit_wrong_class", "duplicate", "fa_background", "fa_static", "fa_premotion", "fa_driver", "neutral"]
    for n, o in outcomes.items():
        print(f"  {n:8s} " + "  ".join(f"{k} {o.get(k, 0)}" for k in keys if o.get(k, 0) or k in keys[:4]))


if __name__ == "__main__":
    main()
