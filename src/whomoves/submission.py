"""Write and validate submission.csv (format in DATA.md)."""

from pathlib import Path

import pandas as pd

CLASSES = {"person", "vehicle", "animal"}


def write_submission(alarms: pd.DataFrame, test_videos: pd.DataFrame, path: Path) -> pd.DataFrame:
    """One row per test video; alarms joined by ';'; a single ';' when there are none."""
    alarms = alarms.sort_values(["video_id", "t"])
    joined = alarms.groupby("video_id").apply(
        lambda g: ";".join(f"{r.t:.2f} {r.x:.4f} {r.y:.4f} {r['class']}" for _, r in g.iterrows())
    )
    sub = pd.DataFrame({"video_id": test_videos["video_id"]})
    sub["alarms"] = sub["video_id"].map(joined).fillna(";")
    validate_submission(sub, test_videos)
    sub.to_csv(path, index=False)
    return sub


def validate_submission(sub: pd.DataFrame, test_videos: pd.DataFrame) -> None:
    """Raise if the submission would be rejected or is obviously wrong."""
    assert list(sub.columns) == ["video_id", "alarms"]
    assert set(sub["video_id"]) == set(test_videos["video_id"]), "video_id mismatch"
    duration = test_videos.set_index("video_id")["duration_s"]
    for vid, cell in zip(sub["video_id"], sub["alarms"]):
        assert isinstance(cell, str) and cell, f"empty cell for {vid}"
        if cell == ";":
            continue
        for alarm in cell.split(";"):
            t, x, y, cls = alarm.split(" ")
            assert 0 <= float(t) <= duration[vid], f"{vid}: t={t} out of range"
            assert 0 <= float(x) <= 1 and 0 <= float(y) <= 1, f"{vid}: x, y out of [0, 1]"
            assert cls in CLASSES, f"{vid}: bad class {cls}"
