# Casebook

Concrete cases from the training videos. The report must cite at least three, with
`video_id` and time. Note also doubtful annotations.

| video_id | t (s) | what happens | why it is easy / hard | baseline behaviour | our behaviour |
|---|---|---|---|---|---|
| train_0001e6b8 | 0.0–0.3 | Moving person (target t5) at the top of the frame, ~15x33 px, partly hidden behind a tree and a lamp post | Hard: tiny, blurred and occluded. Faster R-CNN v2 misses it at input short side 800, 1080 and 1440, while all 32 parked-car boxes in the same frames are found | not checked yet | missed by the detector (pillar 1), so no alarm is possible downstream |
| train_728f0655 | 17.0–19.6 | Walking person (target, box ~120 px) whose box is >= 70% covered by a parked car's box from 17.0 s | Hard for a "driver" rule: in a parking lot, walking behind a parked car looks like sitting inside it | baseline has no driver logic: it alarms on real drivers too | old rule dropped the track for good; with the fix it alarms at 19.6 s, once out of the car box (scripts/find_cases.py) |
| train_bd0dd247 | 55.1 | Person (target, box ~175 px) whose track lives only 1.7 s (fragment of a longer walk) and moves 1.75 box heights | Hard: too short for a long confirmation | | confirm 5 never alarms; confirm 2 + model alarms at 55.1 s |
| train_6be8bd6d | 27.1 | Parked car (static) whose box center drifts up to 1.67 box heights (partial occlusion changes the box) | Easy for a size-aware rule, hard for a small threshold | the baseline's K = 0.5 box heights would alarm at 27.1 s (weight-2 false alarm, again after each track restart) | below our vehicle threshold of 2.5: no alarm |
