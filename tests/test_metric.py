"""Tests for the local scorer: one small synthetic scene per rule, plus the real training data."""

import pandas as pd
import pytest

from whomoves.config import load_config
from whomoves.io import load_csv
from whomoves.metric import GroundTruth, reference_alarms, score
from whomoves.submission import parse_submission, write_submission

W, H, FPS, N_FRAMES = 100, 100, 10.0, 100  # a 10-second, 100x100 px video


def _track(track_id, role, cls, box_at, frames=range(N_FRAMES), moving_from=None):
    rows = []
    for f in frames:
        x, y, w, h = box_at(f)
        moving = int(moving_from is not None and f / FPS >= moving_from)
        rows.append(dict(video_id="v", track_id=track_id, role=role, **{"class": cls},
                         frame=f, t=f / FPS, x=x, y=y, w=w, h=h, moving=moving))
    return rows


@pytest.fixture(scope="module")
def gt():
    # Person "a": stands at x=10 until t=3 s, then walks right at 0.5 px per frame.
    walk = lambda f: (10 + max(0, f - 30) * 0.5, 10, 10, 20)
    tracks = pd.DataFrame(
        _track("a", "target", "person", walk, moving_from=3.0)
        + _track("s", "static", "vehicle", lambda f: (60, 60, 20, 10))
        + _track("d", "disturbance", None, lambda f: (80, 10, 5, 5))
        + _track("m", "ambiguous", "vehicle", lambda f: (40, 80, 10, 10))
    )
    targets = pd.DataFrame([dict(video_id="v", track_id="a", t_move_start=3.0, t_end=9.9, bonus=0, **{"class": "person"})])
    videos = pd.DataFrame([dict(video_id="v", width=W, height=H, fps=FPS, duration_s=N_FRAMES / FPS)])
    return GroundTruth(tracks, targets, videos)


def alarms(*rows):
    return pd.DataFrame([dict(video_id="v", t=t, x=x / W, y=y / H, **{"class": c}) for t, x, y, c in rows],
                        columns=["video_id", "t", "x", "y", "class"])


def test_reference_scores_one(gt):
    assert score(reference_alarms(gt), gt)["f1"] == pytest.approx(1.0)


def test_empty_scores_zero(gt):
    assert score(alarms(), gt)["f1"] == 0.0


def test_wrong_class_gets_partial_points(gt):
    r = score(alarms((4.0, 15, 20, "vehicle")), gt)
    assert r["points"] == pytest.approx(0.6)
    assert r["outcomes"] == {"hit_wrong_class": 1}


def test_repeated_alarms_on_a_hit_target_are_free(gt):
    r = score(alarms((4.0, 15, 20, "person"), (6.0, 30, 20, "person")), gt)
    assert r["outcomes"] == {"hit": 1, "duplicate": 1}
    assert r["f1"] == pytest.approx(1.0)


def test_up_to_one_second_early_is_a_hit(gt):
    assert score(alarms((2.1, 15, 20, "person")), gt)["outcomes"] == {"hit": 1}


def test_too_early_on_a_waiting_target_weighs_two(gt):
    r = score(alarms((1.0, 15, 20, "person")), gt)
    assert r["outcomes"] == {"fa_premotion": 1}
    assert r["fa_weighted"] == 2


def test_alarm_on_parked_car_weighs_two_each_time(gt):
    r = score(alarms((4.0, 15, 20, "person"), (5.0, 70, 65, "vehicle"), (8.0, 70, 65, "vehicle")), gt)
    assert r["outcomes"] == {"hit": 1, "fa_static": 2}
    assert r["precision"] == pytest.approx(1 / (1 + 4))


def test_alarm_on_disturbance_weighs_two(gt):
    assert score(alarms((5.0, 82, 12, "person")), gt)["outcomes"] == {"fa_disturbance": 1}


def test_alarm_on_ambiguous_is_neutral(gt):
    r = score(alarms((4.0, 15, 20, "person"), (5.0, 45, 85, "vehicle")), gt)
    assert r["outcomes"] == {"hit": 1, "neutral": 1}
    assert r["f1"] == pytest.approx(1.0)


def test_alarm_on_empty_background_weighs_one(gt):
    assert score(alarms((5.0, 50, 40, "person")), gt)["fa_weighted"] == 1


def test_box_is_enlarged_by_20_percent(gt):
    # At t=4 the box is x in [15, 25], y in [10, 30]: 20% of 10 px wide -> 2 px of slack on x.
    assert score(alarms((4.0, 26.5, 20, "person")), gt)["outcomes"] == {"hit": 1}
    assert score(alarms((4.0, 28.0, 20, "person")), gt)["outcomes"] == {"fa_background": 1}


def test_submission_round_trip(gt, tmp_path):
    a = reference_alarms(gt)
    videos = pd.DataFrame([dict(video_id="v", duration_s=10.0)])
    sub = write_submission(a, videos, tmp_path / "sub.csv")
    assert score(parse_submission(sub), gt)["f1"] == pytest.approx(1.0)


def test_real_training_reference_scores_one():
    data_dir = load_config()["paths"]["data_dir"]
    if not (data_dir / "train_tracks.csv").exists():
        pytest.skip("competition data not available")
    real = GroundTruth(load_csv(data_dir, "train_tracks"), load_csv(data_dir, "train_alarms"),
                       load_csv(data_dir, "train_videos"))
    r = score(reference_alarms(real), real)
    assert r["n_targets"] == 1401
    assert r["f1"] == pytest.approx(1.0)
