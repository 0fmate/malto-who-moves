# Who Moves? — MALTO Recruitment Challenge

Pipeline: detect → track → decide who moves → one alarm per mover.
Data schema: `DATA.md`. Write-up: `reports/report.md`. Experiment log: `reports/experiments.md`.
The final submission is reproduced on Kaggle by `scripts/final_pipeline.py` (see its docstring).

## Setup

1. The competition data is **outside** this folder, in `../data_code/malto-who-moves` (flat: videos + CSVs).
2. Set `paths.data_dir` in `config.yaml` (or the env var `WHOMOVES_DATA_DIR`).
3. `python -m venv .venv`, activate it, `pip install -r requirements.txt`.
4. `python scripts/make_sample.py` → fills `data_sample/` with small excerpts and `SUMMARY.md`.
5. Look at a frame: `python scripts/extract_frame.py <video_id> <t> --gt`.

## Layout

```
config.yaml            paths and parameters (single source of truth)
DATA.md                data schema and what the videos look like
src/whomoves/          the package
  config.py            load config.yaml (+ env override)
  io.py                load CSVs, iterate video frames
  metric.py            local re-implementation of the competition scorer
  validation.py        scene-grouped folds, test-like clips
  detect.py            pretrained detector (full frame + tiles), COCO -> {person, vehicle, animal}
  track.py             multi-object tracker (simplified ByteTrack)
  match.py             match predicted tracks to ground-truth tracks
  features.py          motion features per track row
  motion_model.py      model inputs
  motion_zoom.py       detector on zoomed crops (tried, not in the final pipeline)
  alarms.py            turn tracks into alarms (one per mover)
  submission.py        write and validate submission.csv
  viz.py               overlays and figures
scripts/               command-line entry points; final_pipeline.py builds the final submissions
reports/               write-up, experiment log, concrete cases
tests/                 tests (the metric scores train_alarms.csv as 1.0)
```

## After the competition

Delete all copies of the competition data and derived files (rules), except the MEVA clips (CC BY 4.0).
