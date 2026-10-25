# DATA.md — competition data

Copied from the Kaggle Data and Evaluation pages. Raw files live in `data_dir` (see `config.yaml`).
Real example rows are in `data_sample/` (run `python scripts/make_sample.py`).

## Files

All files sit flat in `data_dir` (no train/ and test/ subfolders). Train `video_id`s include the
`train_` prefix (e.g. `train_0001e6b8`), test ones do not (e.g. `10a86472`): the video file is always
`<video_id>.mp4`. About 2.5 GB in total.

| File | Content |
|---|---|
| `train_<id>.mp4` | 90 training videos, 218 minutes in total. Whole clips. |
| `train_videos.csv` | One row per training video: `video_id, width, height, fps, duration_s, source_dataset, notes` |
| `train_tracks.csv` | One row per object per frame: `video_id, track_id, role, class, frame, t, x, y, w, h, moving` |
| `train_alarms.csv` | Expected alarms on training videos, one per target: `video_id, track_id, t_move_start, t_end, class, bonus` |
| `<id>.mp4` | 48 test videos, 61 minutes in total. Cut from longer clips, some mirrored, all re-encoded. |
| `test_videos.csv` | One row per test video: `video_id, width, height, fps, duration_s` |
| `sample_submission.csv` | Valid submission with no alarms (`;` everywhere) |

## Annotations (`train_tracks.csv`)

- `x, y, w, h`: **pixels** of the delivered video; `x, y` is the **top-left** corner. `t` in seconds.
- `class`: person, vehicle, animal.
- `role` — what the object is for the evaluation:
  - `target`: must be alarmed once it moves
  - `bonus`: very small or very brief mover, optional
  - `static`: never really moves; alarming it is a false alarm (weight 2)
  - `disturbance`: leaves, shadows, snow, reflections; alarming it is a false alarm (weight 2)
  - `driver`: person inside a vehicle; must not alarm
  - `other`, `ambiguous`: not scored, or scored lightly
- `moving`: 1 from the moment a target starts to move. **This is the operational definition of "moving".**
- `train_alarms.csv` is derived from `train_tracks.csv` and is the reference answer for training.
- Annotations are good, not perfect: spotting doubtful ones is a plus for the report.

## Videos

| Source | Resolution | FPS | What to expect |
|---|---|---|---|
| Parking lots and campus roads | 1280x720 | 10 | Many parked cars, far people (20–40 px tall), vehicles entering/leaving |
| Outdoor yards and building entrances | 1280x720 | 10 | People walking in/out, shadows, long empty stretches |
| Street scenes in bad weather | 720x480 or 720x540 | 10 | Snow, leaves in the wind, wet-road reflections, low resolution |

- No audio. Test videos are 1–3 minute cuts and may start while something is already moving.
- Mirrored videos are valid scenes. **Test scenes do not appear in training.**

## Key facts from data_sample/SUMMARY.md

- Train is almost all VIRAT: 87 of 90 videos (196 of 218 min); MEVA 2 videos, CDnet 1 (720x480).
  Test has 6 low-resolution videos (3 at 720x480, 3 at 720x540; 720x540 never appears in train).
- 5062 tracks, 2.3 M rows. Targets: 730 person + 671 vehicle, **0 animal** (animals only as 5 bonus).
  Static: 1744 vehicle + 10 person. Disturbance (652) and other (1009) have no class and are tiny
  (median box height 7 px). Driver: 26.
- `moving` for a target is ~1 for almost its whole visible life (median 0.97): it means
  "has started moving", not "is moving in this frame". Static/disturbance are always 0.
- Target box height: min 20 px, median 64 px. Bonus median 18 px.
- ~7% of alarms have t_move_start < 0.5 s (already moving when the video starts).

## Submission format

```
video_id,alarms
a7f3c2,12.5 0.412 0.633 person;40.0 0.100 0.520 vehicle
b19e44,;
c08d71,3.0 0.871 0.402 animal
```

- One row per `video_id` of `test_videos.csv`. Alarms separated by `;`, fields by spaces.
- Video with no alarm: a single `;` (empty cells are rejected).
- `0 <= t <= duration`; `x, y` in [0, 1], x left→right, y top→bottom; `x, y` ideally the box center.
