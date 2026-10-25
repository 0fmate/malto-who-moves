# Who Moves? — MALTO Recruitment Challenge

Pipeline: detect → track → decide who moves → one alarm per mover.
Context, rules and conventions for working with Claude Code are in `CLAUDE.md`; the data schema is in `DATA.md`.

## Setup

1. The competition data is **outside** this folder, in `../data&code/malto-who-moves` (flat: videos + CSVs).
2. Set `paths.data_dir` in `config.yaml` (or the env var `WHOMOVES_DATA_DIR`).
3. `python -m venv .venv`, activate it, `pip install -r requirements.txt`.
4. `python scripts/make_sample.py` → fills `data_sample/` with small excerpts and `SUMMARY.md`.
5. Look at a frame: `python scripts/extract_frame.py <video_id> <t> --gt`.

## Layout

See `CLAUDE.md`, section "Project layout".

## After the competition

Delete all copies of the competition data and derived files (rules), except the MEVA clips (CC BY 4.0).
