"""Turn tracks into alarms: one per track that is judged to move.

Policy:
- a track is "moving" in a row when the decision (a rule now, a model later) says so for
  `confirm_rows` consecutive rows of that track; the alarm goes at the first such row;
- the alarm position is the box center at that moment (the object may have moved since it started);
- person tracks mostly inside a vehicle box (drivers) never alarm;
- earliness is not rewarded and false alarms cost double: prefer waiting to guessing.
"""

import pandas as pd

KEY = ["video_id", "track_id"]


def rule_moving(feats: pd.DataFrame, theta: float) -> pd.Series:
    """Hand-written rule: the track is now more than `theta` box heights away from where it first appeared."""
    return feats["disp_from_start"] >= theta


def first_confirmed(feats: pd.DataFrame, moving: pd.Series, confirm_rows: int) -> pd.DataFrame:
    """For each track, the first row where `moving` has been true for confirm_rows rows in a row."""
    df = feats.assign(_moving=moving.to_numpy()).sort_values(KEY + ["t"], kind="stable")
    new_track = (df["video_id"] != df["video_id"].shift()) | (df["track_id"] != df["track_id"].shift())
    block = (new_track | (df["_moving"] != df["_moving"].shift())).cumsum()
    run = df.groupby(block).cumcount() + 1
    confirmed = df["_moving"] & (run >= confirm_rows)
    return df[confirmed].groupby(KEY, sort=False).head(1).drop(columns="_moving")


def make_alarms(feats: pd.DataFrame, moving: pd.Series, videos: pd.DataFrame, confirm_rows: int = 1,
                driver_inside: float = 0.7) -> pd.DataFrame:
    """Alarms (video_id, t, x, y, class, track_id) from per-row moving decisions."""
    rows = first_confirmed(feats, moving, confirm_rows)
    rows = rows[~((rows["class"] == "person") & (rows["inside_vehicle"] >= driver_inside))]
    rows = rows.merge(videos[["video_id", "width", "height", "duration_s"]], on="video_id")
    return pd.DataFrame({
        "video_id": rows["video_id"],
        "t": rows["t"].clip(0, rows["duration_s"]),
        "x": ((rows["x"] + rows["w"] / 2) / rows["width"]).clip(0, 1),
        "y": ((rows["y"] + rows["h"] / 2) / rows["height"]).clip(0, 1),
        "class": rows["class"],
        "track_id": rows["track_id"],
    }).reset_index(drop=True)
