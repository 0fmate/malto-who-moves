"""Does pixel evidence of motion (pixel_motion.py) tell false alarms from hits on the tiled test-like clips?

Per candidate (person 0.75 / others 2.5, confirm 5): fg_med / fg_max = median / max share of box pixels that differ
from the background over the sampled rows, ring_med = the same around the box, fg_contrast = fg_med - ring_med.

Compared on t90 (strict / lenient, and per scene fold), on top of the hand gate and of the current best filter
(hand gate AND learned model P >= 0.85, out-of-fold): drop when fg_med < z; and the learned model with the pixel
features as extra inputs. Test: only how many alarms each rule keeps.

Usage: python scripts/eval_pixel_motion.py
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
THETA_PERSON, P_THR = 0.75, 0.85
PIXEL = ["fg_med", "fg_max", "ring_med", "fg_contrast"]
Z_GRID = [0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3]


def pixel_features(cache: Path) -> pd.DataFrame:
    z = pd.concat([pd.read_parquet(p) for p in sorted((cache / "pixel_motion").glob("*.parquet"))])
    g = z.groupby(KEY)
    out = pd.DataFrame({"fg_med": g["fg_box"].median(), "fg_max": g["fg_box"].max(), "ring_med": g["fg_ring"].median()})
    out["fg_contrast"] = out["fg_med"] - out["ring_med"]
    return out.reset_index()


def with_pixels(rows: pd.DataFrame, pf: pd.DataFrame) -> pd.DataFrame:
    out = rows.reset_index(names="_row").merge(pf, on=KEY, how="left").set_index("_row")
    out.index.name = None
    return out


def main() -> None:
    cfg = load_config()
    cache, driver = cfg["paths"]["cache_dir"], cfg["motion"]["driver_inside"]
    scenes = load_scenes(cache)
    targets = load_csv(cfg["paths"]["data_dir"], "train_alarms")
    weights = targets[targets["bonus"] == 0].groupby("video_id").size().reindex(scenes.index, fill_value=0)
    folds = make_folds(scenes, weights + 1, cfg["validation"]["n_folds"])
    pf = pixel_features(cache)

    c = pickle.loads((cache / "clips90_t2x2.pkl").read_bytes())
    videos, gt = c["videos"], c["variants"]
    rows = with_pixels(L.candidate_rows(c["feats"], THETA_PERSON, driver), pf)
    rows["fold"] = rows["video_id"].str.split("@").str[0].map(folds)
    y = L.labels(rows, videos, gt["strict"])
    tv = load_csv(cfg["paths"]["data_dir"], "test_videos")
    ft = pd.read_parquet(cache / "features" / f"{sw.TILED}.parquet")
    test = with_pixels(L.candidate_rows(ft[ft["video_id"].isin(tv["video_id"])], THETA_PERSON, driver), pf)
    print(f"t90 candidates {len(rows)}, missing {int(rows['fg_med'].isna().sum())}; "
          f"test candidates {len(test)}, missing {int(test['fg_med'].isna().sum())}")

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

    clip_fold = videos["video_id"].str.split("@").str[0].map(folds)

    def f1(keep, variant="strict", fold=None):
        vids = None if fold is None else videos["video_id"][clip_fold == fold]
        return score(rows_to_alarms(rows[keep], videos), gt[variant], video_ids=vids)["f1"]

    def line(keep, tkeep):
        per = "/".join(f"{f1(keep, fold=k):.3f}" for k in sorted(folds.unique()))
        return f"{f1(keep):.3f} | {f1(keep, 'lenient'):.3f} | folds {per} | test {int(tkeep.sum())}"

    groups = {"hit": y == 1, "false": y == 0, "false weak": (y == 0) & rows["weak"],
              "hit weak": (y == 1) & rows["weak"]}
    print("\nquantiles 10/25/50/75%")
    for name, m in groups.items():
        q = rows.loc[m, PIXEL].quantile([0.1, 0.25, 0.5, 0.75])
        print(f"  {name:10s} n={int(m.sum()):4d}  " + "  ".join(f"{col}: {'/'.join(f'{v:.2f}' for v in q[col])}" for col in PIXEL))
    q = test[PIXEL].quantile([0.1, 0.25, 0.5, 0.75])
    print(f"  {'test all':10s} n={len(test):4d}  " + "  ".join(f"{col}: {'/'.join(f'{v:.2f}' for v in q[col])}" for col in PIXEL))

    rows["p"] = oof([])
    tp = fit_all([]).predict_proba(L.inputs(test))[:, 1]
    bases = {"hand": (hand(rows), hand(test)),
             "hand & model": (hand(rows) & (rows["p"] >= P_THR), hand(test) & (tp >= P_THR))}
    print("\nF1 t90 strict | lenient | per fold | test kept")
    for bname, (b, tb) in bases.items():
        print(f"  {bname:28s} {line(b, tb)}")
        for feat in ["fg_med", "fg_contrast"]:
            for z in Z_GRID:
                k, kt = b & ~(rows[feat] < z), tb & ~(test[feat] < z)
                print(f"    + drop {feat}<{z:<4}         {line(k, kt)}")

    for name, cols in {"model + pixels": PIXEL, "model + fg_med": ["fg_med"]}.items():
        rows["pz"] = oof(cols)
        tpz = fit_all(cols).predict_proba(L.inputs(test).join(test[cols]))[:, 1]
        for thr in [0.8, 0.85, 0.9]:
            print(f"  hand & {name} {thr:<4}  {line(hand(rows) & (rows['pz'] >= thr), hand(test) & (tpz >= thr))}")
    m = fit_all(PIXEL)
    gain = m.booster_.feature_importance("gain")
    print("\nfeature importance with pixels (gain share):")
    print(pd.Series(gain / gain.sum(), index=L.inputs(rows).columns.tolist() + PIXEL)
          .sort_values(ascending=False).head(8).round(3).to_string())


if __name__ == "__main__":
    main()
