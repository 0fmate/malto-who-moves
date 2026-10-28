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

### Tiled detection on Kaggle (2026-10-27)

Full frame + 2x2 tiles, 48 test + 10 small-target train videos, 2 T4: ~2.5 h (2.5 fps per GPU, ~5x slower).
Test: +21% detections, +30% boxes under 30 px. On the 10 train videos (same rule, theta 2.0 / confirm 5):

| | full frame | tiled 2x2 |
|---|---|---|
| targets with >= 1 track | 166/172 | 167/172 |
| ceiling target recall | 0.959 | 0.971 |
| rule F1 | 0.906 | **0.918** (P 0.935, target recall 0.884) |
| test alarms | 241 | 299 (+24%) |

With tiles, low thresholds get much worse (theta 1.0: 0.713 vs 0.838): tile boxes add jitter and false
detections, so the rule must stay strict. **tiled_v1 on the LB: 0.748** (rule_v1 0.711, +0.037): the test
gap is at least partly small objects the full-frame detector misses. Now above the organizers' 0.720.

## Learned motion decision (pillar 3.5) — did not beat the rule

2026-10-27. Trained on full-frame features of the 90 train videos, 3 scene folds (out-of-fold scores);
transfer checked on the tiled features of the 10 small-target videos (rule: 0.918 there).

| approach | OOF F1 train | per fold | tiled 10 videos | notes |
|---|---|---|---|---|
| rule theta 2.0, confirm 5 | **0.903** | 0.921 / 0.877 / 0.871 | **0.918** | |
| per-frame LightGBM on `moving`, alarm at first P >= thr for k rows | 0.825 (thr 0.9, k 5) | 0.797 / 0.882 / 0.848 | 0.760 | 131 alarms on parked cars vs 28 |
| alarm filter: rule candidates (theta 1.5) + LightGBM on hit/false-alarm labels | 0.898 (keep P >= 0.96) | 0.911 / 0.885 / 0.864 | 0.870 | candidate recall 0.90 |

Why the per-frame model fails: an alarm fires at the first frame a track looks like moving, so one
high-P frame among thousands of a parked car is a false alarm; the training objective (per frame) is
not the decision (first frame per track). The filter fixes the objective, but with 136 false candidates
against 2004 good ones there is little to learn, and it transfers worse to tiled boxes (different jitter
and scores). Most important features: track_range (per-frame model); first_border_dist, track_len_s,
disp_future_2s, h_rel (filter). Decision: keep the rule, which is simpler, explainable and more robust;
spend the remaining time on detection (pillar 1), the lever that moved the LB.

## Run A: tiled 2x2 with detector threshold 0.05 (pillar 1) — no gain

2026-10-27, Kaggle, same 58 videos. Re-tracked on the PC at several score cuts (tracker.min_det_score),
rule grid on the 10 small-target train videos (best theta / confirm per row):

| detections kept | low-score matching | best F1 | P | target recall |
|---|---|---|---|---|
| >= 0.05 | as before (IoU 0.3, center round for all) | 0.701 | 0.609 | 0.820 |
| >= 0.10 | as before | 0.832 | 0.818 | 0.831 |
| >= 0.15 | as before | 0.877 | 0.910 | 0.831 |
| >= 0.10 | IoU >= 0.6, no center round below 0.25 | 0.898 | 0.929 | 0.855 |
| >= 0.15 | IoU >= 0.6, no center round below 0.25 | 0.908 | 0.965 | 0.843 |
| **>= 0.25** (= tiled_v1) | — | **0.918** | 0.935 | 0.884 |

Low-confidence boxes do not find new targets (recall does not rise); they get attached to nearby tracks
and make parked cars look like they move. Stricter matching limits the damage but never beats the 0.25 cut.
Kept 0.25. The new tracker options (low_iou_threshold, center_min_score) default to the old behaviour.

## Run B: tiled 3x3, and fusion with 2x2 (pillar 1)

2026-10-28, Kaggle, same 58 videos: 4.7 h on 2 T4 (1.4 fps per GPU; test alone ~3.7 h). 3x3 adds only
+5% boxes over 2x2 (+7% under 30 px). Fusion = all 2x2 boxes + 3x3 boxes shorter than H px, per-frame
class-aware NMS (`fuse_detections.py`, no GPU). Rule on the 10 small-target train videos (best theta/confirm):

| detections | F1 | P | target recall | test alarms |
|---|---|---|---|---|
| 2x2 (tiled_v1, LB 0.748) | **0.918** | 0.935 | 0.884 | 299 |
| 3x3 | 0.865 | 0.872 | 0.849 | |
| 2x2 + 3x3 boxes < 30 px | 0.894 | 0.869 | **0.901** | 339 |
| 2x2 + 3x3 boxes < 40 px | 0.885 | 0.904 | 0.855 | |
| 2x2 + 3x3 boxes < 60 px | 0.864 | 0.882 | 0.831 | |

3x3 alone is worse: smaller tiles cut big parked cars more often, their boxes jump and they look like they
move (1269 alarms on parked cars without a motion filter, vs 1004 with 2x2). Fusion with small boxes only
finds the most targets but adds false alarms; LB probe **fused_h30: 0.748, identical to tiled_v1** (36 of 48
videos differ, but on the ~14 public ones the extra alarms land on already-hit targets or neutral areas).
Pillar 1 stays at 2x2.

## Track stitching (pillar 2) — no gain

2026-10-28. `stitch_tracks`: join a piece that ends with one that starts up to 3 s later where the first,
at its last speed, should be (<= 1 box height), same class, height ratio <= 1.5. Idea: the rule measures distance
from the first position, so a slow walker split in pieces never reaches the threshold.

| | without | with stitching |
|---|---|---|
| 90 train videos, full frame (theta 2, confirm 5) | **0.903** | 0.901 |
| 10 small-target videos, tiled 2x2 (best rule) | **0.918** | 0.900 |
| pieces per found target | 2.1 | 1.9 |

Few pieces get joined, and some wrong joins make still objects look like they move. Kept off (tracker.stitch: false).

## Motion-guided zoom (pillar 1) — first check

2026-10-28. `motion_zoom.py`: MOG2 background subtraction (shadows dropped), cleaned mask, blobs >= 20 px,
up to 6 square crops (4x the blob, 96-320 px) every 2nd frame, enlarged by the detector. CPU check on 900
frames of 3 train videos: MOG2 at 60-150 fps; share of annotated boxes whose center falls in a crop:

| | < 30 px | 30-50 px | > 50 px |
|---|---|---|---|
| target, moving | **0.57** | **0.81** | **0.95** |
| static | 0.19 | 0.33 | 0.32 |
| disturbance | 0.15 | 0.27 | 0.41 |

The zoom goes where things move, not to parked cars. Next: detector on the crops (Kaggle), fused with 2x2.

## Motion-guided zoom (pillar 1) — detector on the crops: no gain

2026-10-27. Kaggle run `fasterrcnn_resnet50_fpn_v2_s1_mz` (test + 10 small-target train videos): 80,874 boxes on crops
(median height 29 px person, 36 px vehicle). Fused with the 2x2 boxes (`fuse_detections.py`), tracked, rule theta 2 / confirm 5:

| 10 small-target train videos | F1 | P | target recall | fa_static |
|---|---|---|---|---|
| 2x2 alone | **0.918** | 0.935 | 0.884 | 3 |
| 2x2 + all zoom boxes | 0.843 | 0.784 | 0.895 | 17 |
| 2x2 + zoom boxes < 40 px | 0.845 | | | |
| 2x2 + only new zoom boxes (IoU <= 0.1 with every 2x2 box) | 0.913 | 0.92 | 0.890 | 0 |
| 2x2 + only new zoom boxes < 40 px | 0.916 | 0.924 | 0.890 | 3 |

- All zoom boxes: parked cars near a mover get a second, slightly different box every 2nd frame; the track
  jumps between the two and the car seems to move (fa_static 3 -> 17).
- Only new boxes (`--only-new 0.1`) removes that, but adds nothing: GT targets reached by some track stay
  167/172 in both runs. On test the only-new submission has the same 299 alarms as tiled_v1 (3 change class);
  +532 tracks, almost all short (+4k rows), +6 small tracks that move >= 2 heights.
- Why: on these videos the 2x2 tiles already reach the small movers the zoom reaches; the 5 missed targets
  are not found by the zoom either (occluded / in no crop). Zoom kept out of the final pipeline.

## LB probes on the threshold, pixel floor, camera shake — no lever found

2026-10-27.

| tiled 2x2, confirm 5 | test alarms | public LB | 10 small-target train | 90 train (full frame) |
|---|---|---|---|---|
| theta 2.0 (tiled_v1) | 299 | 0.748 | 0.918 | **0.903** |
| theta 2.5 | 255 | 0.719 | **0.921** | 0.902 |
| theta 3.0 | 228 | **0.753** | 0.904 | 0.887 |

Not monotonic: the public part (~14 videos) moves +-0.03 for a handful of alarms, so this is noise, not a
precision-vs-recall signal. Threshold stays at 2.0 (CV); theta 3.0 is a candidate second final submission.

- Pixel floor (`disp >= theta * max(h, h_min)`, h_min 10-30 px): F1 within +-0.003 on both train sets. Dropped.
- Camera shake (`scripts/camera_shake.py`, phase correlation vs the first frame): test cameras are steady
  (median video p95 shift 0.6 px, only 2/48 videos above 2 px). Not the cause of the test gap; no compensation.

## Test-like clips: the validation was too kind; per-class threshold (pillar 3)

2026-10-27. `scripts/eval_clips.py`: every train video cut into consecutive 35-120 s clips (like the test), our
tracks cut too and features recomputed per clip; ground truth re-derived per clip (a target is a clip target if
`moving` = 1 in some clip frame; one that only moves after the clip becomes static). "strict" keeps targets
that moved before the clip and barely move inside it; "lenient" makes targets moving < 0.5 box heights neutral.

| rule theta 2, confirm 5 | F1 | P | target recall |
|---|---|---|---|
| 90 train videos, whole | 0.903 | 0.910 | 0.874 |
| same, 187 clips (median 64 s), strict | **0.855** | 0.921 | **0.780** |
| same, clips, lenient | 0.881 | 0.922 | 0.824 |

Target recall by how far the target moves inside its clip (box heights): <0.5: 0.03 (96), 0.5-1: 0.04 (52),
1-2: 0.16 (142), 2-4: 0.76 (262), >=4: 0.95 (1165). 25% of clip targets are already moving at the clip start
(recall 0.63 vs 0.83), mostly because many of them barely move inside the clip (parked after arriving, stopped).
Short clips hide part of each trajectory, so a 2-box-height rule misses slow or briefly visible movers.

Per-class threshold (`rule_moving(theta, theta_person)`), confirm 5:

| person, vehicle | clips strict | clips lenient | clip folds 0/1/2 | 90 whole videos | 10 small-target clips (2x2) |
|---|---|---|---|---|---|
| 2.0, 2.0 (current) | 0.855 | 0.881 | 0.881 0.828 0.812 | 0.903 | 0.905 |
| 1.0, 2.5 | **0.877** | **0.902** | 0.905 0.849 0.833 | **0.921** | **0.923** |
| 0.75, 2.5 | 0.885 | 0.906 | 0.916 0.851 0.836 | 0.920 | 0.922 |
| 1.5, 2.5 | 0.869 | 0.895 | 0.900 0.834 0.823 | 0.914 | 0.913 |

Better on every view and every fold. Why: a step is a small move in box heights for a tall, narrow person
box, while half-hidden parked cars jitter more. On train the extra person alarms are mostly duplicates (free)
and new hits (person hits 605 -> 666, false alarms 19 -> 41). Risk: train has only 10 static people; a campus
with people standing around could cost more. Test person alarms grow x1.52 (train x1.37).
Also tried (no gain on clips): an OR with a 2 s speed rule; a lower theta for high-score tracks.
Submissions: tiled_p1.0_v2.5 (358 alarms), tiled_p0.75_v2.5 (381).
Public LB: tiled_p1.0_v2.5 **0.766** (best so far; tiled_v1 0.748, theta 2.5 for all 0.719), tiled_p0.75_v2.5 0.757.
Same direction as the clip validation: lowering the person threshold is what gains (+0.047 over theta 2.5 for all).

## Smaller objects: shrunk training videos (validation of pillar 1 + 3 on test-like sizes)

2026-10-27. Kaggle: the 10 small-target train videos with `run_detect.py --tiles 2x2 --shrink 0.6 / 0.45` (frame scaled
down and pasted on a canvas of the original size, so objects get fewer pixels and a smaller share of each tile;
boxes mapped back). GT targets reached by some track: 167/172 (x100), 166/172 (x60), 166/172 (x45): detection still
finds them. Confirm 5; whole videos / test-like clips (strict):

| scale | theta 2 all | p1.0 v2.5 | p0.75 v2.5 | p1.0 v2.0 |
|---|---|---|---|---|
| x100 | 0.918 / 0.905 | **0.947 / 0.923** | 0.942 / 0.922 | 0.937 / 0.917 |
| x60 | 0.900 / 0.884 | 0.918 / 0.895 | **0.933 / 0.904** | 0.920 / 0.897 |
| x45 | 0.882 / 0.873 | 0.904 / 0.887 | **0.916 / 0.899** | 0.903 / 0.892 |

- Size alone costs ~0.03 (x45, clips), all in recall (0.853 -> 0.788); precision even rises. Tracks exist but
  move less than the threshold before breaking.
- The per-class threshold helps at every scale; at small scales person 0.75 is a bit better than 1.0
  (LB: p1.0 0.766, p0.75 0.757, within noise).
- Gap budget: clips ~-0.05, size ~-0.03 -> ~0.87-0.89 expected, LB 0.766: ~0.1 still unexplained by length
  or size (scene content, re-encoding, how the test GT treats targets that barely move inside a cut).

## Class-only LB probes: where does the public score go?

2026-10-28. The tiled_p1.0_v2.5 submission (LB 0.766) split by class, nothing else changed:
persons only (216 alarms) **0.576**, vehicles only (142) **0.442**. Ratios to the full score: 0.75 / 0.58
(train clips: 0.71 / 0.61 strict, 0.62 / 0.69 small x45).

Since F1 = 2 points / (D + T) (D = hits + weighted false alarms, T = targets, the same in all three runs), the
three scores fix T/D, and so overall precision P = F1 (1 + T/D) / 2 and recall R = P / (T/D):

| | T/D | precision | recall |
|---|---|---|---|
| train clips strict (true) | 1.11 | 0.926 | 0.834 |
| train clips lenient (true) | 1.05 | 0.925 | 0.880 |
| public LB (estimated) | ~0.92-1.02 | **~0.74-0.77** | ~0.74-0.80 |

- The estimate needs the person share of D, known only through the alarm count (0.60); on train that
  proxy moves T/D by only ~0.04, so P and R are robust. The per-class precision split is NOT robust
  (on train the alarm-count proxy gives 0.79 / 1.06 instead of 0.93 / 0.91), so the 0.75 vs 0.58 ratios
  cannot tell persons from vehicles: they also depend on the unknown class mix of the test targets.
- Main finding: on test about 1 weighted alarm in 4 earns nothing (train: 1 in 13). Precision drops
  ~0.17, recall ~0.05-0.10. Test alarms have detection scores like train hits, so they sit on real
  objects: likely objects the test GT counts as not moving (weight 2), or alarms outside the time window.

## Look-ahead filter: keep an alarm only if its track moves enough over its whole life

2026-10-28. Offline we see the whole video, so for each alarm (still raised at the first confirmed row) we can
check the track's maximum `disp_from_start` over its whole life. Filter: >= 2 box heights for people, >= 4 for
vehicles. Test-like clips (strict / lenient):

| K person / vehicle | alarms | strict F1 (P / R) | lenient F1 |
|---|---|---|---|
| none | 2240 | 0.877 (0.926 / 0.834) | 0.902 |
| 2 / 3 | 1976 | 0.856 (0.938 / 0.788) | 0.882 |
| 2 / 4 | 1918 | 0.841 (0.940 / 0.761) | 0.867 |
| 3 / 5 | 1628 | 0.802 (0.948 / 0.694) | 0.828 |

On train the weakly moving alarms are mostly hits (total < 3 heights: 211 hits, 23 false alarms, 243 duplicates),
so the filter costs. Test alarms move less over their life (median 4.9 vs 5.8 heights; K 2/4 keeps 78% of test
alarms vs 86% on train). Submission strong_k2_4 (279 alarms, 157 person, 122 vehicle).
Expected from train: ~0.96 x 0.766 = 0.73. Public LB: **0.776** (best so far, +0.010 over the unfiltered 0.766).
- Rough split (T/D ~0.96 from the class probes, weak alarms ~22% of D): the 79 weak alarms earn ~0.3-0.4
  points each on test (train: ~0.9), the strong ones ~0.85. The weak movers explain a good part of the
  precision gap: on test, "moved only a little" is often not a target.
- Within LB noise on its own (+0.010), but 0.04 above the train-based expectation, consistent with the
  precision estimate from the class probes.

## Look-ahead filter gated by motion coherence and detector confidence (scripts/sweep_lookahead.py)

2026-10-28. Strong alarms (whole-track displacement >= K) are always kept; a weak one is kept only if its track
passes coherence/confidence tests (straightness, largest single jump, box size jump, holding the displacement,
mean detector score). 745 configs on test-like clips: c90 = 90 train videos (plain detector), s100/s60/s45 = the
10 small-target videos with 2x2 tiles at full size / shrunk.

Median measures of weak (K 2/4) alarms: train hits score_mean 0.88, train false alarms 0.58, **test weak 0.61**,
test strong 0.92 (jump_share 0.21 / 0.31 / 0.26 / 0.11). The weak test alarms look like train false alarms.

| filter | c90 strict | c90 lenient | s100 | s60 | s45 | test alarms | LB-weak dropped |
|---|---|---|---|---|---|---|---|
| none | 0.877 | 0.902 | 0.923 | - | 0.887 | 358 | 0 |
| drop all weak (K 2/4) | 0.841 | 0.867 | 0.905 | 0.875 | 0.857 | 279 | 79 |
| K 2/4, weak kept if score_mean >= 0.7 | **0.877** | **0.902** | **0.933** | 0.899 | **0.885** | 307 | 51 |
| score_mean >= 0.7 on every alarm | 0.846 | 0.871 | 0.921 | 0.838 | 0.768 | 236 | 51 |

- On weak alarms the confidence gate is free on train for thresholds 0.5-0.75 (flat), costs from 0.8 up.
  Applied to every alarm it costs a lot (slow, small real movers have low scores too).
- Coherence tests add nothing on top of the score (best pairs within +-0.002).
- Submission p1.0_v2.5_k2_4_sm0.7 (307 alarms). Still to validate on the tiled detector for all 90 videos
  (Kaggle run of 2026-10-28 night).

## Tiled detector on all 90 train videos: validation of the confidence-gated filter

2026-10-28. Kaggle (2x T4): `run_detect.py --tiles 2x2` on the 80 train videos not done yet (same config as the
test run), merged into `fasterrcnn_resnet50_fpn_v2_s1_t2x2`, re-tracked and re-featured (test alarms unchanged:
358). Test-like clips of the 90 tiled videos: t90 (187 clips, 1717 targets).

- No filter, rule p1.0 v2.5: t90 **0.853** strict / 0.872 lenient (plain detector c90: 0.877). Tiles add weak
  false alarms: weak (K 2/4) alarms on t90 = 183 hits + 53 FA (c90: 170 + 22); their FA have score_mean 0.48.
- Filter K 2/4 + score_mean >= 0.7: t90 **0.861** / 0.881 (+0.008 / +0.009). Flat for score_mean 0.65-0.75.
  Dropping all weak alarms: 0.829.
- With the filter, a lower person threshold is better again (it was not without it):

| theta person / vehicle | t90 no filter | t90 filtered | t90 lenient filtered | s45 filtered |
|---|---|---|---|---|
| 0.75 / 2.0 | 0.844 | 0.865 | 0.883 | **0.903** |
| **0.75 / 2.5** | 0.853 | **0.868** | **0.886** | 0.893 |
| 1.00 / 2.5 | 0.853 | 0.861 | 0.881 | 0.885 |
| 1.25 / 2.5 | 0.851 | 0.860 | 0.881 | 0.890 |
| 1.00 / 3.0 | 0.854 | 0.859 | 0.880 | 0.879 |

Submissions: p1.0_v2.5_k2_4_sm0.7 (307 alarms), p0.75_v2.5_k2_4_sm0.7 (317).
