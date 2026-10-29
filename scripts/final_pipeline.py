"""Final pipeline, after the tiled detections: tracks, features, alarms, the two final submissions.

Run on Kaggle after `run_detect.py --tiles 2x2` on the test videos and on the training subset (see --print-subset).
1. run_track.py and build_features.py on every video of the detection run;
   in both submissions a person covered by a vehicle box alarms once out of it (alarms.alarm_rows);
2. hand-gate submission (person 1.0 / others 2.5 box heights, confirm 5): drop an alarm whose track moves less than
   2 (person) / 4 (others) box heights over its whole life AND whose mean detector score is < 0.7;
3. learned-filter submissions (person 0.75): the same hand gate AND a LightGBM model (hit vs false alarm) trained on
   test-like clips of the training subset; confirm 5 rows and keep if P >= 0.85 (lgbm), or confirm 2 rows and keep if
   P >= 0.9 (lgbm_c2: the model takes over the job of the long confirmation, eval_confirm_robust.py).

Training subset: train videos sorted by id (ids are random hashes, so this is a random order) until TRAIN_BUDGET_S
seconds of video, so that detection on test + subset fits in the 9 h GPU limit. On the tiled clips the model trained on
this subset scores 0.873 (all 90 videos: 0.878, hand gate alone: 0.868).

Output: <output_dir>/submission_final_hand.csv, submission_final_lgbm.csv, submission_final_lgbm_c2.csv, and
submission.csv (= the lgbm_c2 one).

Usage:
  python scripts/final_pipeline.py --print-subset      # train video ids to detect, space separated
  python scripts/final_pipeline.py [--skip-track]      # --skip-track: reuse the cached tracks and features
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import lightgbm as lgb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import learn_alarm_filter as L  # noqa: E402
from eval_clips import build_clips  # noqa: E402
from whomoves.alarms import rows_to_alarms  # noqa: E402
from whomoves.config import load_config  # noqa: E402
from whomoves.io import load_csv  # noqa: E402
from whomoves.submission import write_submission  # noqa: E402

DET_RUN = "fasterrcnn_resnet50_fpn_v2_s1_t2x2"
TRAIN_BUDGET_S = 6000
HAND_THETA_PERSON, LGBM_THETA_PERSON = 1.0, 0.75
SCORE_MEAN = 0.7
LGBM_RUNS = {"lgbm": (5, 0.85), "lgbm_c2": (2, 0.9)}  # name: (confirm rows, P threshold)
WAIT_OUT_OF_VEHICLE = True  # see alarms.alarm_rows (eval_driver_fix.py)
PARAMS = dict(L.PARAMS, deterministic=True, force_row_wise=True, n_jobs=4)


def train_subset(data_dir: Path) -> list[str]:
    v = load_csv(data_dir, "train_videos").sort_values("video_id")
    return v.loc[v["duration_s"].cumsum() <= TRAIN_BUDGET_S, "video_id"].tolist()


def hand_keep(rows: pd.DataFrame) -> pd.Series:
    return ~rows["weak"] | (rows["score_mean"] >= SCORE_MEAN)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-subset", action="store_true")
    parser.add_argument("--skip-track", action="store_true")
    args = parser.parse_args()
    cfg = load_config()
    data_dir, cache, out_dir = cfg["paths"]["data_dir"], cfg["paths"]["cache_dir"], cfg["paths"]["output_dir"]
    driver = cfg["motion"]["driver_inside"]
    subset = train_subset(data_dir)
    if args.print_subset:
        print(" ".join(subset))
        return

    t0 = time.time()
    if not args.skip_track:
        for script in ["run_track.py", "build_features.py"]:
            cmd = [sys.executable, str(ROOT / "scripts" / script), "--det-run", DET_RUN]
            subprocess.run(cmd + (["--no-report"] if script == "run_track.py" else []), check=True)
            print(f"{script} done ({(time.time() - t0) / 60:.1f} min)", flush=True)

    tv = load_csv(data_dir, "test_videos")
    feats = pd.read_parquet(cache / "features" / f"{DET_RUN}.parquet")
    test_feats = feats[feats["video_id"].isin(tv["video_id"])]
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. hand gate
    t = L.candidate_rows(test_feats, HAND_THETA_PERSON, driver, WAIT_OUT_OF_VEHICLE)
    keep = hand_keep(t)
    write_submission(rows_to_alarms(t[keep], tv), tv, out_dir / "submission_final_hand.csv")
    print(f"hand: {len(t)} candidates, {int(keep.sum())} alarms {t[keep]['class'].value_counts().to_dict()}", flush=True)

    # 2. hand gate AND learned filter, trained on test-like clips of the training subset
    c = build_clips(DET_RUN, only_videos=subset)
    for name, (confirm, p_thr) in LGBM_RUNS.items():
        rows = L.candidate_rows(c["feats"], LGBM_THETA_PERSON, driver, WAIT_OUT_OF_VEHICLE, confirm=confirm)
        y = L.labels(rows, c["videos"], c["variants"]["strict"])
        use = y.notna()
        model = lgb.LGBMClassifier(**PARAMS).fit(L.inputs(rows[use]), y[use].astype(int))
        print(f"{name}: model trained on {int(use.sum())} candidates of "
              f"{c['videos']['video_id'].str.split('@').str[0].nunique()} train videos ({int((y == 0).sum())} false "
              f"alarms) ({(time.time() - t0) / 60:.1f} min)", flush=True)
        t = L.candidate_rows(test_feats, LGBM_THETA_PERSON, driver, WAIT_OUT_OF_VEHICLE, confirm=confirm)
        keep = hand_keep(t) & (model.predict_proba(L.inputs(t))[:, 1] >= p_thr)
        write_submission(rows_to_alarms(t[keep], tv), tv, out_dir / f"submission_final_{name}.csv")
        print(f"{name}: {len(t)} candidates, {int(keep.sum())} alarms {t[keep]['class'].value_counts().to_dict()}")
    shutil.copy(out_dir / "submission_final_lgbm_c2.csv", out_dir / "submission.csv")
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {out_dir}")


if __name__ == "__main__":
    main()
