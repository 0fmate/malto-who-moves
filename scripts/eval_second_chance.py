"""Second chance: when the learned filter rejects a track at its first candidate row, ask again later.

Today each track is judged once, at its first confirmed moving row; if the model says no, the track never alarms,
even if it is clearly moving a few seconds later. Here each track also gets candidate rows at the first moving row
>= OFFSET seconds after the first one (moving = rule, persons outside vehicle boxes). The model is trained on all of
them (a later row on a mover is a free duplicate, so a positive; on a parked car it is a false alarm). The alarm goes
at the first row of the track that passes hand gate AND P >= thr: earliness is not rewarded, duplicates are free.

Whole-track measures are the same at every row; row features and `hold` (share of later rows still moving) change.
Compared on t90 / s45 (person 0.75 / others 2.5, confirm 2, driver fix, out-of-fold model) for several offset sets;
offsets [0] is the current method. Bootstrap over scenes vs the current method.

Usage: python scripts/eval_second_chance.py
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
from whomoves.alarms import KEY, rows_to_alarms, rule_moving  # noqa: E402
from whomoves.config import load_config  # noqa: E402
from whomoves.io import load_csv  # noqa: E402
from whomoves.metric import score  # noqa: E402
from whomoves.validation import load_scenes, make_folds  # noqa: E402

THETA_PERSON, CONFIRM = 0.75, 2
OFFSET_SETS = {"[0]": [0], "[0,1,2,4]": [0, 1, 2, 4], "[0,2,4,8]": [0, 2, 4, 8], "[0,1,2,3,4,6,8]": [0, 1, 2, 3, 4, 6, 8]}
P_GRID = [0.9, 0.92, 0.94, 0.96]
TRACK_COLS = ["straight", "jump_share", "size_jump", "weak"]  # same at every row of a track


def chance_rows(feats: pd.DataFrame, theta_person: float, driver: float, offsets: list[float],
                confirm: int = CONFIRM) -> pd.DataFrame:
    """Candidate rows of L.candidate_rows plus, per track, the first moving row >= offset s after the first one."""
    base = L.candidate_rows(feats, theta_person, driver, True, confirm=confirm)
    person = feats["class"] == "person"
    moving = rule_moving(feats, sw.THETA, theta_person) & ~(person & (feats["inside_vehicle"] >= driver))
    theta = np.where(person, theta_person, sw.THETA)
    # hold at every row: share of the track's rows from this one on that are >= theta from the start
    f = feats.assign(_far=(feats["disp_from_start"] >= theta).astype(float)).sort_values(KEY + ["t"], kind="stable")
    rev = f.iloc[::-1].groupby(KEY, sort=False)["_far"]
    hold = (rev.cumsum() / (rev.cumcount() + 1)).iloc[::-1]

    parts = [base.assign(chance=0.0)]
    mov = feats.loc[moving, KEY + ["t"]].reset_index(names="_row")
    mov = mov.merge(base[KEY + ["t"]].rename(columns={"t": "t0"}), on=KEY).sort_values("t", kind="stable")
    extra = base[KEY + TRACK_COLS].drop_duplicates(KEY)
    for off in sorted(o for o in offsets if o > 0):
        pick = mov[mov["t"] >= mov["t0"] + off].groupby(KEY, sort=False).head(1)
        r = feats.loc[pick["_row"]].merge(extra, on=KEY, how="left").set_axis(pick["_row"].to_numpy())
        r["hold"] = hold.reindex(r.index).to_numpy()
        parts.append(r.assign(chance=float(off)))
    out = pd.concat(parts)
    return out[~out.index.duplicated(keep="first")]


def first_pass(rows: pd.DataFrame, keep: pd.Series) -> pd.Series:
    """Per track, only the first row (in time) that passes `keep`."""
    k = rows[keep].sort_values("t", kind="stable").groupby(KEY, sort=False).head(1).index
    return pd.Series(rows.index.isin(k), index=rows.index)


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
    videos, gt = sets["t90"]["videos"], sets["t90"]["variants"]
    cf = videos["video_id"].str.split("@").str[0].map(folds)
    clip_scene = pd.Series(videos["video_id"].str.split("@").str[0].map(scenes).to_numpy(), index=videos["video_id"])
    by_scene = {s: clip_scene.index[clip_scene == s] for s in clip_scene.unique()}

    def hand(r):
        return ~r["weak"] | (r["score_mean"] >= 0.7)

    base_counts = None
    print("hand & P, first passing row per track: t90 strict | lenient | folds | s45 | test alarms | from later chances")
    for name, offsets in OFFSET_SETS.items():
        data = {}
        for n, c in sets.items():
            r = chance_rows(c["feats"], THETA_PERSON, driver, offsets)
            r["fold"] = r["video_id"].str.split("@").str[0].map(folds)
            data[n] = r
        rows = data["t90"]
        y = L.labels(rows, videos, gt["strict"])
        for r in data.values():
            r["p"] = np.nan
        for k in sorted(folds.unique()):
            tr = y.notna() & (rows["fold"] != k)
            m = lgb.LGBMClassifier(**L.PARAMS).fit(L.inputs(rows[tr]), y[tr].astype(int))
            for r in data.values():
                sel = r["fold"] == k
                if sel.any():
                    r.loc[sel, "p"] = m.predict_proba(L.inputs(r[sel]))[:, 1]
        t = chance_rows(ft, THETA_PERSON, driver, offsets)
        t["p"] = lgb.LGBMClassifier(**L.PARAMS).fit(L.inputs(rows[y.notna()]), y[y.notna()].astype(int)) \
            .predict_proba(L.inputs(t))[:, 1]
        print(f"  offsets {name}: {len(rows)} t90 rows ({int((y == 0).sum())} false), {len(t)} test rows")
        for thr in P_GRID:
            keeps = {n: first_pass(r, hand(r) & (r["p"] >= thr)) for n, r in data.items()}
            kt = first_pass(t, hand(t) & (t["p"] >= thr))

            def f1s(n, var="strict", fold=None):
                r = data[n]
                cc = sets[n]
                vids = None if fold is None else videos["video_id"][cf == fold]
                return score(rows_to_alarms(r[keeps[n]], cc["videos"]), cc["variants"][var], video_ids=vids)["f1"]
            per = "/".join(f"{f1s('t90', fold=q):.3f}" for q in sorted(folds.unique()))
            later = int((rows.loc[keeps["t90"], "chance"] > 0).sum())
            line = (f"    P >= {thr:<4} {f1s('t90'):.3f} | {f1s('t90', 'lenient'):.3f} | {per} | {f1s('s45'):.3f} | "
                    f"{int(kt.sum())} | {later} t90 / {int((t.loc[kt, 'chance'] > 0).sum())} test")
            cnt = clip_counts(rows, keeps["t90"], videos, gt["strict"])
            if base_counts is None and thr == 0.9:
                base_counts = cnt  # offsets [0], P >= 0.9 = current method
            elif base_counts is not None:
                rng = np.random.default_rng(SEED)
                g = np.array([f1(cnt.loc[ids]) - f1(base_counts.loc[ids]) for ids in
                              (np.concatenate([by_scene[s] for s in rng.choice(list(by_scene), len(by_scene))])
                               for _ in range(N_BOOT))])
                line += f" | boot {g.mean():+.4f} [{np.quantile(g, 0.05):+.4f}, {np.quantile(g, 0.95):+.4f}] {(g > 0).mean():.0%}"
            print(line, flush=True)


if __name__ == "__main__":
    main()
