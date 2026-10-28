"""Does the zoomed second look (zoom_verify.py) tell false alarms from hits on the tiled test-like clips?

Per candidate alarm (person 0.75 / others 2.5, confirm 5) two crops were re-detected: the alarm row and the middle
row of the track. Features: zoom_alarm, zoom_mid (same-class score), zoom_max, zoom_min, zany_max (any class).

Compared on t90 (strict / lenient), all on top of the current best filter (hand gate AND learned model P >= 0.85,
out-of-fold as in learn_alarm_filter.py):
- zoom gates: also drop when zoom_max < z (every alarm, weak alarms only, or small boxes only);
- the learned model retrained with the zoom features as extra inputs.
The s45 set has no zoom crops, so only t90 is used here. Test: only how many alarms each rule keeps.

Usage: python scripts/eval_zoom.py
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

KEY = ["video_id", "track_id"]
THETA_PERSON, P_THR, SMALL = 0.75, 0.85, 0.04  # SMALL: box height / frame height
ZOOM = ["zoom_alarm", "zoom_mid", "zoom_max", "zoom_min", "zany_max"]
Z_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def zoom_features(cache: Path) -> pd.DataFrame:
    z = pd.concat([pd.read_parquet(p) for p in sorted((cache / "zoom_verify").glob("*.parquet"))])
    w = z.pivot_table(index=KEY, columns="which", values="zoom_score", aggfunc="max")
    out = pd.DataFrame({"zoom_alarm": w["alarm"], "zoom_mid": w["mid"]})
    out["zoom_max"] = out.max(axis=1)
    out["zoom_min"] = out[["zoom_alarm", "zoom_mid"]].min(axis=1)
    out["zany_max"] = z.groupby(KEY)["zoom_any"].max()
    return out.reset_index()


def with_zoom(rows: pd.DataFrame, zf: pd.DataFrame) -> pd.DataFrame:
    out = rows.reset_index(names="_row").merge(zf, on=KEY, how="left").set_index("_row")
    out.index.name = None
    return out


def main() -> None:
    cfg = load_config()
    cache, driver = cfg["paths"]["cache_dir"], cfg["motion"]["driver_inside"]
    scenes = load_scenes(cache)
    targets = load_csv(cfg["paths"]["data_dir"], "train_alarms")
    weights = targets[targets["bonus"] == 0].groupby("video_id").size().reindex(scenes.index, fill_value=0)
    folds = make_folds(scenes, weights + 1, cfg["validation"]["n_folds"])
    zf = zoom_features(cache)

    c = pickle.loads((cache / "clips90_t2x2.pkl").read_bytes())
    videos, gt = c["videos"], c["variants"]
    rows = with_zoom(L.candidate_rows(c["feats"], THETA_PERSON, driver), zf)
    rows["fold"] = rows["video_id"].str.split("@").str[0].map(folds)
    y = L.labels(rows, videos, gt["strict"])
    tv = load_csv(cfg["paths"]["data_dir"], "test_videos")
    ft = pd.read_parquet(cache / "features" / f"{sw.TILED}.parquet")
    test = with_zoom(L.candidate_rows(ft[ft["video_id"].isin(tv["video_id"])], THETA_PERSON, driver), zf)
    print(f"t90 candidates {len(rows)}, missing zoom {int(rows['zoom_max'].isna().sum())}; "
          f"test candidates {len(test)}, missing zoom {int(test['zoom_max'].isna().sum())}")

    def oof(cols):
        p = pd.Series(np.nan, index=rows.index)
        for k in sorted(folds.unique()):
            tr = y.notna() & (rows["fold"] != k)
            m = lgb.LGBMClassifier(**L.PARAMS).fit(L.inputs(rows[tr]).join(rows.loc[tr, cols]), y[tr].astype(int))
            sel = rows["fold"] == k
            p[sel] = m.predict_proba(L.inputs(rows[sel]).join(rows.loc[sel, cols]))[:, 1]
        return p

    def fit_all(cols):
        use = y.notna()
        return lgb.LGBMClassifier(**L.PARAMS).fit(L.inputs(rows[use]).join(rows.loc[use, cols]), y[use].astype(int))

    def hand(r):
        return ~r["weak"] | (r["score_mean"] >= 0.7)

    def f1(keep, variant="strict"):
        return score(rows_to_alarms(rows[keep], videos), gt[variant])["f1"]

    # how the zoom scores split hits from false alarms
    small = rows["h_rel"] < SMALL
    groups = {"hit": y == 1, "false": y == 0, "false weak": (y == 0) & rows["weak"],
              "hit small": (y == 1) & small, "false small": (y == 0) & small}
    print(f"\nzoom score quantiles (25/50/75%), small = box height < {SMALL:.0%} of the frame")
    for name, m in groups.items():
        q = rows.loc[m, ["zoom_alarm", "zoom_mid", "zoom_max", "zany_max"]].quantile([0.25, 0.5, 0.75])
        print(f"  {name:12s} n={int(m.sum()):4d}  " + "  ".join(
            f"{col[5:] if col.startswith('zoom') else col}: {'/'.join(f'{v:.2f}' for v in q[col])}" for col in q))
    print(f"  test small: {int((test['h_rel'] < SMALL).sum())} of {len(test)}; test zoom_max median "
          f"{test['zoom_max'].median():.2f} (small {test.loc[test['h_rel'] < SMALL, 'zoom_max'].median():.2f})")

    rows["p"] = oof([])
    base = hand(rows) & (rows["p"] >= P_THR)
    tp = fit_all([]).predict_proba(L.inputs(test))[:, 1]
    tbase = hand(test) & (tp >= P_THR)
    print(f"\nF1 t90 strict | lenient | test kept")
    print(f"  hand gate                   {f1(hand(rows)):.3f} | {f1(hand(rows), 'lenient'):.3f} | {int(hand(test).sum())}")
    print(f"  hand & model {P_THR} (current)  {f1(base):.3f} | {f1(base, 'lenient'):.3f} | {int(tbase.sum())}")
    scopes = {"all": lambda r: np.ones(len(r), bool), "weak": lambda r: r["weak"],
              "small": lambda r: r["h_rel"] < SMALL}
    for scope, sel in scopes.items():
        res = []
        for z in Z_GRID:
            k = base & ~(sel(rows) & (rows["zoom_max"] < z))
            kt = tbase & ~(sel(test) & (test["zoom_max"] < z))
            res.append(f"{z}:{f1(k):.3f}/{f1(k, 'lenient'):.3f}/{int(kt.sum())}")
        print(f"  + drop {scope:5s} if zoom_max<z  " + "  ".join(res))

    for name, cols in {"model + zoom": ZOOM, "model + zoom_max": ["zoom_max"]}.items():
        rows["pz"] = oof(cols)
        tpz = fit_all(cols).predict_proba(L.inputs(test).join(test[cols]))[:, 1]
        res = []
        for thr in [0.7, 0.75, 0.8, 0.85, 0.9, 0.93, 0.95]:
            k = hand(rows) & (rows["pz"] >= thr)
            res.append(f"{thr}:{f1(k):.3f}/{f1(k, 'lenient'):.3f}/{int((hand(test) & (tpz >= thr)).sum())}")
        print(f"  hand & {name:16s} " + "  ".join(res))
    m = fit_all(ZOOM)
    gain = m.booster_.feature_importance("gain")
    print("\nfeature importance with zoom (gain share):")
    print(pd.Series(gain / gain.sum(), index=L.inputs(rows).columns.tolist() + ZOOM)
          .sort_values(ascending=False).head(8).round(3).to_string())


if __name__ == "__main__":
    main()
