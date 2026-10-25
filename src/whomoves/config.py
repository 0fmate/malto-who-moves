"""Load config.yaml, with paths overridable by env vars (used on Kaggle / Colab)."""

import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"

# Env var -> key in cfg["paths"]. An env var that is set wins over config.yaml.
ENV_OVERRIDES = {
    "WHOMOVES_DATA_DIR": "data_dir",
    "WHOMOVES_CACHE_DIR": "cache_dir",
    "WHOMOVES_OUTPUT_DIR": "output_dir",
}


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for env_var, key in ENV_OVERRIDES.items():
        value = os.environ.get(env_var)
        if value:
            cfg["paths"][key] = value

    # Relative paths are relative to the repo root.
    for key, value in cfg["paths"].items():
        p = Path(value)
        cfg["paths"][key] = p if p.is_absolute() else REPO_ROOT / p
    return cfg
