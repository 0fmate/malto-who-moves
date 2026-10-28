"""Sweep look-ahead filters: drop an alarm whose track moves little over its whole life AND moves incoherently.

LB evidence (reports/experiments.md): on test, alarms on tracks that move < 2 (person) / 4 (vehicle) box heights
over their whole life are mostly false alarms, while on train they are mostly hits. Hypothesis: on test those
small displacements are fake (a track jumping to a nearby object, a parked car drifting), on train they are real
slow motion. So a weak alarm is kept only if its motion looks coherent; strong alarms are always kept.

Whole-track measures (track resampled every STEP_S seconds, centers and heights smoothed, distances / median height):
- straight:   farthest distance from the start / path length up to there (1 = straight line, low = wandering);
- jump_share: largest single-step displacement / farthest distance (high = the motion is one sudden jump);
- size_jump:  largest |log height ratio| between steps (a switch to another object changes the box size);
- hold:       share of the track after the alarm that stays at least theta away from the start (low = goes back);
- score_mean: mean detector score of the track.

Data: test-like clips of the 90 train videos (plain detector), of the 10 small-target videos with 2x2 tiles at
full size and shrunk to 60% / 45%, and the 48 test videos (only how many alarms each filter keeps).
Clip sets are cached as pickles in cache_dir. Output: <cache_dir>/sweep_lookahead.csv + a printed summary.

Usage: python scripts/sweep_lookahead.py [--quick]
"""

import argparse
import itertools
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_clips import build_clips  # noqa: E402
from whomoves.alarms import alarm_rows, rows_to_alarms, rule_moving  # noqa: E402
from whomoves.config import load_config  # noqa: E402
from whomoves.io import load_csv  # noqa: E402
from whomoves.metric import score  # noqa: E402

KEY = ["video_id", "track_id"]
THETA, THETA_PERSON, CONFIRM = 2.5, 1.0, 5
STEP_S = 0.5
TILED = "fasterrcnn_resnet50_fpn_v2_s1_t2x2"
CLIP_SETS = {  # name -> (detection run, cache file)
    "c90": ("fasterrcnn_resnet50_fpn_v2_s1", "clips90.pkl"),
    "s100": (TILED, "clips_sub10_t2x2.pkl"),
    "s60": (TILED + "_x60", "clips_sub10_t2x2_x60.pkl"),
    "s45": (TILED + "_x45", "clips_sub10_t2x2_x45.pkl"),
}
TESTS = {
    "straight": [0.3, 0.4, 0.5, 0.6, 0.7],
    "jump_share": [0.3, 0.4, 0.5, 0.6, 0.8],
    "size_jump": [0.15, 0.25, 0.4, 0.6],
    "hold": [0.3, 0.5, 0.7, 0.9],
    "score_mean": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85],
}
KEEP_IF_ABOVE = {"straight", "hold", "score_mean"}  # the others: keep if below
K_GRID = [(1.5, 3.0), (2.0, 3.0), (2.0, 4.0), (3.0, 5.0), (3.0, 6.0),  # (person, vehicle) whole-track displacement
          (np.inf, np.inf)]  # every alarm must pass the conditions


def _smooth(a: np.ndarray) -> np.ndarray:
    return pd.Series(a).rolling(5, center=True, min_periods=1).median().to_numpy()


def track_measures(g: pd.DataFrame, t_alarm: float, theta: float) -> dict:
    """Coherence measures of one track (rows sorted by t)."""
    t = g["t"].to_numpy(float)
    cx = _smooth((g["x"] + g["w"] / 2).to_numpy(float))
    cy = _smooth((g["y"] + g["h"] / 2).to_numpy(float))
    hs = _smooth(g["h"].to_numpy(float))
    grid = np.arange(t[0], t[-1] + 1e-9, STEP_S) if t[-1] > t[0] else t[:1]
    c = np.c_[np.interp(grid, t, cx), np.interp(grid, t, cy)]
    h = np.maximum(np.interp(grid, t, hs), 1.0)
    h_med = max(float(np.median(hs)), 1.0)
    from_start = np.linalg.norm(c - c[0], axis=1) / h_med
    steps = np.r_[0.0, np.linalg.norm(np.diff(c, axis=0), axis=1) / h_med]
    far = int(from_start.argmax())
    reach = from_start[far]
    d = g["disp_from_start"].to_numpy(float)
    after = d[t >= t_alarm]
    return {
        "straight": reach / max(steps[:far + 1].sum(), 1e-6) if reach > 0 else 0.0,
        "jump_share": steps.max() / max(reach, 1e-6) if reach > 0 else 1.0,
        "size_jump": float(np.abs(np.diff(np.log(h))).max()) if len(h) > 1 else 0.0,
        "hold": float((after >= theta).mean()) if len(after) else 0.0,
        "score_mean": float(g["score"].mean()),
        "total": float(g["track_range"].iloc[0]),
    }


def alarms_with_measures(feats: pd.DataFrame, videos: pd.DataFrame, driver: float) -> pd.DataFrame:
    """Alarms of the base rule (person 1.0 / others 2.5, confirm 5) + whole-track measures of their tracks."""
    rows = alarm_rows(feats, rule_moving(feats, THETA, THETA_PERSON), CONFIRM, driver)
    a = rows_to_alarms(rows, videos)
    alarmed = feats.merge(rows[KEY + ["t"]].rename(columns={"t": "t_alarm"}), on=KEY)
    m = []
    for (vid, tid), g in alarmed.sort_values(KEY + ["t"], kind="stable").groupby(KEY, sort=False):
        theta = THETA_PERSON if g["class"].iloc[0] == "person" else THETA
        m.append({"video_id": vid, "track_id": tid, **track_measures(g, g["t_alarm"].iloc[0], theta)})
    return a.merge(pd.DataFrame(m), on=KEY, how="left")


def keep_mask(a: pd.DataFrame, k: tuple[float, float] | None, conds: tuple) -> np.ndarray:
    """Keep strong alarms (whole-track displacement >= k); keep a weak one only if it passes every condition.

    k None: no filter. No conditions: every weak alarm is dropped.
    """
    if k is None:
        return np.ones(len(a), dtype=bool)
    strong = (a["total"] >= np.where(a["class"] == "person", k[0], k[1])).to_numpy()
    ok = np.full(len(a), bool(conds))
    for name, thr in conds:
        v = a[name].to_numpy()
        ok &= v >= thr if name in KEEP_IF_ABOVE else v <= thr
    return strong | ok


def load_clip_set(name: str, cache: Path) -> dict:
    det_run, fname = CLIP_SETS[name]
    p = cache / fname
    if p.exists():
        return pickle.loads(p.read_bytes())
    c = build_clips(det_run)
    p.write_bytes(pickle.dumps(c))
    return c


def configs(quick: bool) -> list[tuple]:
    out = [(None, ())]  # no filter at all
    for k in K_GRID[2:3] if quick else K_GRID:
        out.append((k, ()))  # drop every weak alarm
        for name, thrs in TESTS.items():
            out += [(k, ((name, thr),)) for thr in (thrs[1:2] if quick else thrs)]
        if not quick:
            for (n1, t1), (n2, t2) in itertools.combinations(TESTS.items(), 2):
                out += [(k, ((n1, a), (n2, b))) for a in t1[1:-1] for b in t2[1:-1]]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="a few configs only (smoke test)")
    args = parser.parse_args()
    cfg = load_config()
    cache, driver = cfg["paths"]["cache_dir"], cfg["motion"]["driver_inside"]
    t0 = time.time()

    sets = {}
    for name in CLIP_SETS:
        c = load_clip_set(name, cache)
        sets[name] = (alarms_with_measures(c["feats"], c["videos"], driver), c["variants"])
        print(f"{name}: {len(sets[name][0])} alarms ({time.time() - t0:.0f} s)", flush=True)
    test_videos = load_csv(cfg["paths"]["data_dir"], "test_videos")
    ft = pd.read_parquet(cache / "features" / f"{TILED}.parquet")
    test = alarms_with_measures(ft[ft["video_id"].isin(test_videos["video_id"])], test_videos, driver)
    weak_lb = test["total"] < np.where(test["class"] == "person", 2.0, 4.0)  # the 79 alarms dropped on the LB

    print("\nmedian measures: train c90 weak (K 2/4) hits vs false alarms, test weak vs strong")
    a, v = sets["c90"]
    out = score(a, v["strict"])["alarms"]
    a_sorted = a.sort_values("t", kind="stable").reset_index(drop=True)
    weak = a_sorted["total"] < np.where(a_sorted["class"] == "person", 2.0, 4.0)
    hit = out["outcome"].str.startswith("hit").to_numpy()
    fa = out["outcome"].str.startswith("fa").to_numpy()
    cols = list(TESTS)
    print(pd.DataFrame({"train weak hit": a_sorted.loc[weak & hit, cols].median(),
                        "train weak FA": a_sorted.loc[weak & fa, cols].median(),
                        "test weak": test.loc[weak_lb, cols].median(),
                        "test strong": test.loc[~weak_lb, cols].median()}).round(2).to_string())
    print(f"counts: train weak hits {(weak & hit).sum()}, weak FA {(weak & fa).sum()}; test weak {weak_lb.sum()}")

    rows = []
    for i, (k, conds) in enumerate(configs(args.quick)):
        r = {"k": "none" if k is None else "all" if np.isinf(k[0]) else f"{k[0]:g}/{k[1]:g}",
             "conds": " & ".join(f"{n}{'>=' if n in KEEP_IF_ABOVE else '<='}{t}" for n, t in conds)
             or ("-" if k is None else "drop all weak")}
        for name, (a, variants) in sets.items():
            keep = keep_mask(a, k, conds)
            r[name] = score(a[keep], variants["strict"])["f1"]
            if name == "c90":
                r["c90_len"] = score(a[keep], variants["lenient"])["f1"]
        keep = keep_mask(test, k, conds)
        r["test_kept"] = int(keep.sum())
        r["lb_weak_dropped"] = int((weak_lb & ~keep).sum())
        rows.append(r)
        if i % 50 == 0:
            print(f"  {i} configs ({time.time() - t0:.0f} s)", flush=True)

    res = pd.DataFrame(rows)
    res["train_mean"] = res[["c90", "s100", "s45"]].mean(axis=1)
    res.to_csv(cache / "sweep_lookahead.csv", index=False)
    base = res.iloc[0]
    print(f"\n{len(res)} configs in {(time.time() - t0) / 60:.1f} min -> {cache / 'sweep_lookahead.csv'}")
    print(f"no filter: c90 {base['c90']:.3f}  s100 {base['s100']:.3f}  s45 {base['s45']:.3f}, test {base['test_kept']} alarms")
    show = ["k", "conds", "c90", "c90_len", "s100", "s60", "s45", "train_mean", "test_kept", "lb_weak_dropped"]
    print("\nreference filters (drop all weak):")
    print(res[res["conds"] == "drop all weak"][show].round(3).to_string(index=False))
    good = res[res["lb_weak_dropped"] >= 40]
    print("\nbest on train among filters that drop >= 40 of the 79 LB-weak test alarms:")
    print(good.sort_values("train_mean", ascending=False)[show].head(15).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
