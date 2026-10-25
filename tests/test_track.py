"""Tests for the tracker on hand-made detections."""

import pandas as pd

from whomoves.config import load_config
from whomoves.track import track_video

CFG = load_config()["tracker"]


def dets(*rows):
    """rows: (frame, x, y, w, h, score, class)"""
    return pd.DataFrame([dict(video_id="v", frame=f, t=f / 10, x=x, y=y, w=w, h=h, score=s, label=c, **{"class": c})
                         for f, x, y, w, h, s, c in rows])


def test_walking_person_keeps_one_id():
    out = track_video(dets(*[(f, 100 + 3 * f, 50, 15, 30, 0.6, "person") for f in range(30)]), CFG)
    assert len(out) == 30 and out["track_id"].nunique() == 1


def test_parked_car_and_walker_get_two_ids():
    rows = [(f, 500, 300, 120, 60, 0.99, "vehicle") for f in range(20)]
    rows += [(f, 100 + 3 * f, 50, 15, 30, 0.6, "person") for f in range(20)]
    out = track_video(dets(*rows), CFG)
    assert out.groupby("class")["track_id"].nunique().to_dict() == {"person": 1, "vehicle": 1}


def test_short_gap_is_bridged_by_the_velocity_prediction():
    # A small fast walker missed for 8 frames: its boxes no longer overlap after the gap.
    rows = [(f, 100 + 6 * f, 50, 10, 20, 0.6, "person") for f in list(range(10)) + list(range(18, 30))]
    out = track_video(dets(*rows), CFG)
    assert out["track_id"].nunique() == 1


def test_long_gap_starts_a_new_track():
    rows = [(f, 100, 50, 15, 30, 0.6, "person") for f in list(range(10)) + list(range(40, 50))]
    assert track_video(dets(*rows), CFG)["track_id"].nunique() == 2


def test_car_and_truck_boxes_on_one_vehicle_are_merged():
    rows = [(f, 500, 300, 120, 60, 0.9, "vehicle") for f in range(10)]
    rows += [(f, 502, 301, 118, 60, 0.6, "vehicle") for f in range(10)]
    out = track_video(dets(*rows), CFG)
    assert len(out) == 10 and out["track_id"].nunique() == 1


def test_low_score_detections_extend_but_do_not_start_tracks():
    rows = [(f, 100 + 2 * f, 50, 15, 30, 0.6 if f < 5 else 0.27, "person") for f in range(20)]
    rows += [(f, 800, 400, 15, 30, 0.27, "person") for f in range(20)]  # never confident
    out = track_video(dets(*rows), CFG)
    assert out["track_id"].nunique() == 1 and len(out) == 20


def test_one_frame_blips_are_dropped():
    rows = [(f, 100, 50, 15, 30, 0.6, "person") for f in range(10)] + [(3, 700, 400, 20, 20, 0.9, "animal")]
    assert set(track_video(dets(*rows), CFG)["class"]) == {"person"}
