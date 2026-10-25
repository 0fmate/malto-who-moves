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
