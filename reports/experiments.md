# Experiments log

One row per experiment, including the ones that did not work (the report needs them).

| date | id | idea / change | CV F1 | CV P / R | LB public | kept? | notes |
|---|---|---|---|---|---|---|---|
| | baseline | organizers' naive notebook | | | | — | |

## Detector comparison (pillar 1)

2026-10-25, Kaggle T4, first 3 train videos (train_0001e6b8, train_04c8087d, train_087daea2; 2409 frames,
25974 annotated boxes). A box is "found" if a same-class detection has IoU >= 0.5 in that frame.

| detector | fps (1 T4) | target <30px | target 30-50px | target 50-100px | target >100px | static | moving persons found* | kept? |
|---|---|---|---|---|---|---|---|---|
| Faster R-CNN v2, short side 800 | 13.5 | 0.17 | 0.14 | 0.91 | 0.98 | 1.00 | 6/12 | **yes** |
| Faster R-CNN v2, short side 1080 | 7.6 | 0.20 | 0.20 | 0.91 | 0.98 | 1.00 | 6/12 | no: +3-6 pts on small targets for 1.8x the time |
| RT-DETR r50 (640x640) | 13.8 | 0.09 | 0.09 | 0.89 | 0.99 | 1.00 | 5/12 | no: half the small-target recall, not faster |

\* tracks found in >= 30% of their moving frames. Vehicles: 5/5 for all three.

Takeaways: big objects and parked cars are solved; **small moving people (< 50 px) are the bottleneck**
(1930 of 5800 target boxes here). Upscaling the input barely helps. To try: tiling the frame, motion blobs
(background subtraction) as extra candidates, and a looser match criterion (IoU 0.5 is harsh on 15x30 px boxes,
while the metric only needs the alarm point inside the box enlarged by 20%).

## Tracking (pillar 2)

2026-10-25, all 90 train videos, Faster R-CNN v2 detections, tracker settings in config.yaml (`run_track.py`).
Predicted tracks matched to GT tracks by IoU >= 0.3 in >= 50% of their frames.

| what | value |
|---|---|
| predicted tracks | 12146: 6362 match nothing (median 9 frames), 2850 target, 2598 static, 142 ambiguous, 117 bonus, 39 driver, 38 other |
| GT targets with >= 1 predicted track | 1339/1401 = 95.6% (person 669/730, vehicle 670/671) |
| pieces per found target | median 1, mean 2.1 (ID switches; cheap, since repeated alarms on a hit target are free) |
| ceiling: perfect motion decision on these tracks | F1 0.991 (target recall 0.955, bonus hits push the official recall to the cap) |
| no motion filter: one alarm per track | F1 0.211 (4167 alarms on parked cars, 1796 on unmatched tracks) |

Takeaway: pillars 1-2 leave room for F1 ~0.99 on train; everything is decided by pillar 3 (0.21 -> 0.99).
Of the 62 targets never tracked, 61 are people (sizes not checked yet).

## Scene groups for validation (pillar 3.1)

2026-10-26, `scene_groups.py`: median background of 15 frames per video, 96x54 gray, correlation (also mirrored),
average-linkage clustering. Merge distances jump from 0.39 to 0.61, so threshold 0.5 -> **6 scenes**:
3 VIRAT parking lots (47, 34 and 6 videos), the CDnet snow street, 2 MEVA clips. At threshold 0.35 one lot was
split in two (same view, different sun and shadows): checked by eye on outputs/scene_groups.jpg.
Folds (greedy by number of targets): fold 0 = lot of 47 videos, fold 1 = lot of 34, fold 2 = everything else.

## How far tracks move (pillar 3.2)

`analyze_motion.py`, distances in box heights from the first position of each predicted track (smoothed centers):

| matched GT role | 10% | median | 90% | 99% |
|---|---|---|---|---|
| static (2598 tracks) | 0.02 | 0.09 | 0.79 | 2.19 |
| none (6362) | 0.00 | 0.07 | 1.56 | 10.70 |
| target (2850) | 0.13 | 3.09 | 13.52 | 41.15 |

- After the annotated start, median distance reached: person 0.55 / 0.90 / 1.12 heights after 1 / 2 / 3 s,
  vehicle 0.83 / 1.84 / 2.68. In their own height units people are slow.
- 128 targets are seen waiting > 3 s before they start: before the start they had moved 0.23 (median) /
  0.91 (90%) heights. A rule must stay above ~1 height to keep quiet there (alarms before the start weigh 2).

## Hand-written rule (pillar 3.4)

Rule: a track moves once its smoothed center is more than THETA box heights from its first position, for
CONFIRM consecutive detections; one alarm per track at that moment; people >= 70% inside a vehicle box never alarm.
Grid on all 90 train videos (`eval_rule.py`), F1:

| theta \ confirm | 1 | 5 | 10 |
|---|---|---|---|
| 0.5 | 0.695 | 0.736 | 0.763 |
| 1.0 | 0.835 | 0.851 | 0.864 |
| 1.5 | 0.884 | 0.897 | 0.896 |
| **2.0** | 0.898 | **0.903** | 0.897 |
| 3.0 | 0.894 | 0.887 | 0.868 |

Best theta=2.0, confirm=5: F1 0.903, P 0.910, target recall 0.874; 66 alarms on unannotated background,
28 on parked cars, **0 before the start of motion**. Per scene fold (no refit): 0.921 / 0.877 / 0.871;
fold 2 (snow, MEVA, small lot) has the lowest precision (0.777).

| date | id | idea / change | CV F1 | CV P / R | LB public | kept? | notes |
|---|---|---|---|---|---|---|---|
| 2026-10-26 | rule_v1 | distance rule theta=2.0, confirm=5 | 0.903 (all train; folds 0.92/0.88/0.87) | 0.910 / 0.874 | **0.711** | baseline | 241 alarms on test; organizers' target_submission.csv scores 0.720 |
| 2026-10-26 | rule_t1.0 | same rule, theta=1.0 | 0.851 | | | probe | 318 alarms on test |
| 2026-10-26 | rule_t1.5 | same rule, theta=1.5 | 0.897 | | | probe | 273 alarms on test |

### Why 0.90 locally but 0.71 on the public LB? (2026-10-26)

- The test is dominated by **one scene absent from training**: about 30 of 48 videos show the same wide campus
  view (large lawn, roads around it; some mirrored). Others: building entrance (~5), street with trees and leaves
  (~3), snow street (~6), a parking lot, a gas-station-like building (~3). Train has only 3 VIRAT lots.
- Objects are smaller in test: median vehicle box height 25 px vs 40 px in train (1280x720 videos); fewer
  tracks per minute (23 vs 38). Small objects are exactly where the detector is weakest (pillar 1).
- On train, non-VIRAT videos are fine (CDnet F1 0.99, 38 targets; MEVA only 3 targets, too few to judge).
- So scene-grouped CV cannot see this gap: all 3 train lots are closer to each other than to the test campus.

### LB probes on the threshold (2026-10-26)

| id | theta | test alarms | LB public |
|---|---|---|---|
| rule_v1 | 2.0 | 241 | 0.711 |
| rule_t1.5 | 1.5 | 273 | 0.679 |
| rule_t1.0 | 1.0 | 318 | 0.701 |

Flat and not monotonic: the threshold is not the lever (public LB is ~14 videos, noisy). We produce ~4 alarms/min
on test vs ~9/min on train: the likely loss is objects never detected/tracked, which no rule can recover.
Test fps, frame size and duration match test_videos.csv (no timing bug). We do not inspect our alarms on test
frames by eye: error analysis stays on train (rules forbid hand-annotating test videos).

### Tiled detection, first look (2026-10-26)

Full frame + 2x2 tiles with 20% overlap, merged by class-aware NMS. CPU check on 18 frames of 3 small-people
videos (IoU >= 0.5): moving targets 30-50 px found 0.40 -> 0.60, < 30 px 0.59 -> 0.62; ~3.8x the time.
Rule baseline (full frame) on the 10 train videos with the smallest targets (median 29-35 px, 15.6 min,
172 targets): F1 0.906 at theta 2.0 / confirm 5. Next: same 10 videos + test with tiles on Kaggle.
