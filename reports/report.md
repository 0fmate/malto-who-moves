# Who Moves? — write-up

**Result:** public LB 0.837 (organizers' naive baseline: 0.603). Reproduction notebook: REPRODUCTION_NOTEBOOK_LINK ·
code: https://github.com/0fmate/malto-who-moves (pinned at commit `08b2ffc`).

## 1. Idea

An alarm must land once on every person, vehicle or animal that **really moves**, and never on parked cars, people
standing still or background. Two facts of the metric drive every choice: earliness is not rewarded (any time from
1 s before the move to 3 s after the object leaves is fine), and a false alarm on a static object costs double. So
the method waits until motion is certain, and since it runs offline it may look at the **whole track**, past and
future, before deciding.

The pipeline is two stages: a permissive **candidate generator** (detector, tracker, a simple motion rule) and a
strict **judge** (a hand gate plus a learned filter) that decides which candidates become alarms.

## 2. Method

**Detection.** torchvision Faster R-CNN ResNet-50 FPN v2 with COCO weights (a two-stage detector: a region proposal
network suggests boxes, a second head classifies and refines them). It runs on every frame on the full image **and
on 2x2 overlapping tiles** (20% overlap) that are enlarged, so a 25 px person is seen at about 50 px; boxes are
merged with class-aware NMS. COCO classes map to person / vehicle / animal (bicycle → person, birds dropped).
Tiling was chosen because small people are the bottleneck: on 3 training videos the detector found 17% of targets
under 30 px and 91% of those over 50 px, and test objects are smaller than training ones (median vehicle box 25 px
vs 40 px).

**Tracking.** A simplified ByteTrack: constant-velocity prediction, Hungarian matching by IoU first for confident
and then for low-score detections, then by center distance for small or fast objects; lost tracks live 2 s.

**Motion, in box heights.** Every displacement is divided by the box height, so "moved 2" means the same for a
near truck and a far pedestrian. A track becomes a **candidate** when it is 0.75 (person) / 2.5 (vehicle, animal)
box heights from where it first appeared, for **2 detections in a row**. People need a lower threshold: a step is
a small move for a tall, narrow box, while half-hidden parked cars jitter more. A person whose box is >= 70% covered
by a vehicle box does not count as moving until it leaves that box, so drivers never alarm but people walking
behind parked cars do. One alarm per track, at the box center at that moment.

**Judge.** (a) A hand gate drops tracks that move little over their whole life (< 2 / < 4 box heights) **and** have
low mean detector confidence (< 0.7). (b) A LightGBM model (gradient-boosted decision trees: each new small tree
corrects the errors of the previous ones; 200 trees, 7 leaves) scores each candidate as "lands on a real mover" vs
"false alarm" from 27 features of the candidate row and of the whole track (displacements over 1/2/5 s past and
future, straightness, size change, box size relative to the frame, distance from the border, track age and length,
detector score, share of later rows still moving). It is trained on test-like clips of 47 training videos, with
labels given by the competition metric itself (hit or free duplicate = 1, false alarm = 0, neutral = left out).
An alarm is kept if both agree, with P >= 0.9. The most useful input is box size: false alarms are tiny boxes.

## 3. Validation

The test scenes are not in the training set, so validation must not leak scenes. Training videos were grouped into
**6 scenes** by clustering their median backgrounds (3 parking lots, a snow street, 2 other clips) and split into
**3 scene folds**. Videos were cut into **187 test-like clips of 35–120 s**, with the ground truth re-derived per
clip (a clip may start while something already moves). A second set shrinks the 10 videos with the smallest
targets to test-like sizes. Every model is scored out-of-fold, on scenes it never saw.

**Decision rule:** keep a change only if it gains >= 0.005 F1 on the clips **and loses in no fold**. The public LB
(~15 videos) is only a sanity check: on random 15-clip subsets the F1 has std ~0.035, and a real +0.009 gain looks
like a loss in 23% of draws. The two main late gains were also checked with a bootstrap over scenes.

| step | clips CV F1 | public LB |
|---|---|---|
| organizers' naive baseline (mobilenet detector, 2 fps, alarm at 0.5 box heights) | — | 0.603 |
| our detector + tracker + rule (2 box heights, confirm 5), full frame | 0.855* | 0.711 |
| 2x2 tiles | — | 0.748 |
| per-class threshold (person 1.0 / vehicle 2.5) | 0.877* | 0.766 |
| hand gate (tiled) | 0.861 | 0.796 |
| + LightGBM filter, person 0.75 | 0.878 | 0.794 |
| + people behind parked cars no longer dropped as drivers | 0.883 | 0.812 |
| + confirm 2 instead of 5, P >= 0.9 (**final**) | **0.892** | **0.837** |

\* full-frame detector; the other CV rows use the tiled one, whose extra small detections add jitter (hence the
dip of the hand-gate row).

The last two gains are corrections of logic rather than tuned thresholds. The driver rule used to drop a track for
good if, at its first moving moment, the person was inside a car box: on the clips it dropped 388 tracks, of which
only 2 were real drivers. The 5-row confirmation was chosen before the learned filter existed; with the filter
judging, it only threw away short fragments of real movers (60 missed targets had a track that moved enough for
fewer than 5 rows). Confirm 2 wins in all 3 scenes that hold real numbers of targets, in 98% of scene-bootstrap
draws, and also with the final 47-video model (0.887 → 0.896).

## 4. What did not work

| idea | result |
|---|---|
| RT-DETR instead of Faster R-CNN; larger input; detector threshold 0.05; 3x3 tiles or fusion with 2x2 | half the small-target recall / small gain for 1.8x time / no gain / no LB gain |
| per-row learned motion classifier instead of the rule | did not beat the rule |
| stitching track fragments | only 16 missed targets would pass; no gain |
| second look: detector on a zoomed crop of each candidate | +0.002 CV, −0.018 LB: dropped real small movers |
| pixel evidence: background difference inside the box | +0.002, one fold worse |
| lower motion thresholds once the filter exists | ≤ +0.003, one fold worse: the filter cannot tell jittery parked cars |
| learned filter without the hand gate | −0.007: weak, low-score test tracks get through |
| second chance: re-ask the filter later in the track | ≤ +0.003, one fold worse: decisive inputs are whole-track |

**Lesson:** in a two-stage system, loosening a stage-one rule helps only where it lets in candidates the judge can
tell apart (short fragments of real movers), not where it lets in what the judge cannot judge (jittery parked cars,
weak low-score tracks).

## 5. Concrete cases (training videos)

- **train_728f0655, 17.0–19.6 s:** a walking person's box is >= 70% covered by a parked car's box from 17.0 s. The
  old driver rule dropped the track for good; now it alarms at 19.6 s, once out of the car box.
- **train_bd0dd247, 55.1 s:** a person tracked for only 1.7 s (a fragment of a longer walk) moves 1.75 box heights.
  Confirm 5 never alarms; confirm 2 plus the filter does.
- **train_6be8bd6d, 27.1 s:** a parked car whose box center drifts 1.67 box heights under partial occlusion. The
  naive baseline's 0.5 threshold would raise a weight-2 false alarm, again after every track restart; our 2.5
  vehicle threshold stays quiet.
- **train_0001e6b8, 0.0–0.3 s:** a moving person of ~15x33 px half hidden by a tree. The detector misses it at every
  input size while all 32 parked cars in the frame are found: no later step can recover an undetected object.

**Where the naive baseline fails:** 2 frames per second and IoU-only linking break tracks of anything moving sideways
faster than about a fifth of its height per sample; its mobilenet detector misses people 20–30 px tall; a 0.5 box
height threshold alarms on parked cars whose boxes drift, and every track restart alarms again; drivers alarm.

## 6. Limits

On validation clips the remaining loss is mostly (a) 115 false alarms on tiny moving objects outside the
annotations, which the filter rates as movers (P median 0.98; some may be unannotated real movers), and (b) 274
missed targets: 35 never tracked (mostly small people) and 168 that move less than the threshold inside the clip.
The training set has only 6 scenes, while most test videos show a wide campus view absent from training, so
validation is optimistic (clips CV 0.89 vs public LB 0.84).

## 7. Models, licenses, reproducibility

- Faster R-CNN ResNet-50 FPN v2, torchvision (BSD-3), COCO pretrained weights: the only pretrained model.
- LightGBM (MIT), trained from scratch on the competition training data. Everything else is our own code
  (numpy, pandas, scipy, OpenCV). No vision-language model, no external annotation.
- Same method and parameters for every video; no masks, zones or per-video settings.
- The reproduction notebook runs from the competition data in 352 min on 2x T4 (limit 540), with the code pinned
  to a commit and the environment pinned; it reproduces the submitted files exactly (319 / 309 / 333 alarms).
