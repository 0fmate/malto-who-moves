"""Concrete training cases for the report (video_id and time in seconds), one short list per phenomenon.

A. person behind a parked car: a target track the old driver rule dropped for good, alarmed by the fix;
B. short fragment: a target track that alarms with confirm 2 but never gets 5 moving rows in a row;
C. jittery parked car: a static track whose box center drifts > 0.5 box heights (the naive baseline's K) but stays
   below our vehicle threshold of 2.5.

Whole training videos, tiled detector, person 0.75 / others 2.5 box heights.

Usage: python scripts/find_cases.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import learn_alarm_filter as L  # noqa: E402
import sweep_lookahead as sw  # noqa: E402
from whomoves.alarms import KEY  # noqa: E402
from whomoves.config import load_config  # noqa: E402
from whomoves.io import load_csv  # noqa: E402

N = 4


def show(title: str, df: pd.DataFrame, cols: list[str]) -> None:
    print(f"\n{title}: {len(df)} tracks; largest boxes first")
    print(df.sort_values("h", ascending=False).head(N)[cols].round(2).to_string(index=False))


def main() -> None:
    cfg = load_config()
    cache, driver = cfg["paths"]["cache_dir"], cfg["motion"]["driver_inside"]
    tv = load_csv(cfg["paths"]["data_dir"], "train_videos")
    f = pd.read_parquet(cache / "features" / f"{sw.TILED}.parquet")
    f = f[f["video_id"].isin(tv["video_id"])]

    old = L.candidate_rows(f, 0.75, driver, False, confirm=5)
    new = L.candidate_rows(f, 0.75, driver, True, confirm=5)
    a = new[(new["gt_role"] == "target") & (new["class"] == "person")].merge(old[KEY], on=KEY, how="left", indicator=True)
    a = a[a["_merge"] == "left_only"]
    first_inside = f[(f["class"] == "person") & (f["inside_vehicle"] >= driver)].groupby(KEY)["t"].min().rename("t_inside")
    a = a.merge(first_inside, left_on=KEY, right_index=True, how="left")
    show("A. person behind a parked car (old rule: never; fix: alarm at t)", a,
         ["video_id", "track_id", "t_inside", "t", "h", "track_range"])

    c2 = L.candidate_rows(f, 0.75, driver, True, confirm=2)
    b = c2[c2["gt_role"] == "target"].merge(new[KEY], on=KEY, how="left", indicator=True)
    b = b[b["_merge"] == "left_only"]
    show("B. short fragment (confirm 5: never; confirm 2: alarm at t)", b,
         ["video_id", "track_id", "class", "t", "h", "track_len_s", "max_disp_from_start"])

    st = f[(f["gt_role"] == "static") & (f["class"] == "vehicle")]
    g = st.groupby(KEY).agg(dmax=("max_disp_from_start", "max"), h=("h", "median"))
    g = g[(g["dmax"] > 0.5) & (g["dmax"] < 2.5)]
    cross = st[st["disp_from_start"] > 0.5].groupby(KEY)["t"].min().rename("t_cross")
    g = g.join(cross).reset_index()
    show("C. parked car drifting > 0.5 heights (naive K) but < 2.5 (ours)", g, ["video_id", "track_id", "t_cross", "dmax", "h"])


if __name__ == "__main__":
    main()
