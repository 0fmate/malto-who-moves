# Casebook

Concrete cases from the training videos. The report must cite at least three, with
`video_id` and time. Note also doubtful annotations.

| video_id | t (s) | what happens | why it is easy / hard | baseline behaviour | our behaviour |
|---|---|---|---|---|---|
| train_0001e6b8 | 0.0–0.3 | Moving person (target t5) at the top of the frame, ~15x33 px, partly hidden behind a tree and a lamp post | Hard: tiny, blurred and occluded. Faster R-CNN v2 misses it at input short side 800, 1080 and 1440, while all 32 parked-car boxes in the same frames are found | not checked yet | missed by the detector (pillar 1), so no alarm is possible downstream |
