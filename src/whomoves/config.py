"""Load config.yaml, with the data directory overridable by an env var."""

import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # On Kaggle the data is mounted elsewhere: WHOMOVES_DATA_DIR wins over the file.
    env_data_dir = os.environ.get("WHOMOVES_DATA_DIR")
    if env_data_dir:
        cfg["paths"]["data_dir"] = env_data_dir

    # Relative paths are relative to the repo root.
    for key, value in cfg["paths"].items():
        p = Path(value)
        cfg["paths"][key] = p if p.is_absolute() else REPO_ROOT / p
    return cfg
