"""Tests for motion features and the hand-written rule on hand-made tracks."""

import numpy as np
import pandas as pd
import pytest

from whomoves.alarms import make_alarms, rule_moving
from whomoves.features import track_features

W, H = 1000, 500
VIDEOS = pd.DataFrame([dict(video_id="v", width=W, height=H, duration_s=20.0)])


def track(track_id, cls, box_at, frames=range(200), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for f in frames:
        x, y, w, h = box_at(f)
        jx, jy = rng.normal(0, 0.5, 2)  # detector jitter, half a pixel
        rows.append(dict(video_id="v", track_id=track_id, frame=f, t=f / 10, x=x + jx, y=y + jy, w=w, h=h,
                         score=0.9, **{"class": cls}))
    return rows


@pytest.fixture(scope="module")
def feats():
    rows = (
        track(0, "vehicle", lambda f: (100, 300, 120, 60))                             # parked car
        + track(1, "person", lambda f: (500 + max(0, f - 50) * 2.0, 100, 15, 30))      # waits 5 s, then walks
        + track(2, "vehicle", lambda f: (600, 300, 150, 80))                           # car with a driver...
        + track(3, "person", lambda f: (650 + max(0, f - 100) * 0.5, 320, 30, 40))     # ...who shifts inside it
    )
    return track_features(pd.DataFrame(rows), W, H)


def test_parked_car_stays_near_its_start(feats):
    assert feats.loc[feats["track_id"] == 0, "track_range"].max() < 0.1


def test_walker_alarms_once_after_two_heights(feats):
    a = make_alarms(feats, rule_moving(feats, 2.0), VIDEOS, confirm_rows=5)
    walker = a[a["track_id"] == 1]
    assert len(walker) == 1
    # 2 box heights = 60 px at 2 px per frame -> ~30 frames after the start at t=5 s, plus confirmation.
    assert 7.5 <= walker["t"].iloc[0] <= 8.5
    assert walker["x"].iloc[0] == pytest.approx((500 + (walker["t"].iloc[0] * 10 - 50) * 2 + 7.5) / W, abs=0.01)


def test_no_alarm_on_parked_car_or_driver(feats):
    a = make_alarms(feats, rule_moving(feats, 2.0), VIDEOS, confirm_rows=5)
    assert set(a["track_id"]) == {1}


def test_driver_is_covered_by_the_vehicle(feats):
    assert feats.loc[feats["track_id"] == 3, "inside_vehicle"].min() > 0.7


def test_future_features_see_the_walk_coming(feats):
    w = feats[feats["track_id"] == 1].set_index("frame")
    assert w.loc[45, "disp_past_1s"] < 0.1 and w.loc[45, "disp_future_2s"] > 0.5
