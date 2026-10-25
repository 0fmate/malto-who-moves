"""Classifier: is this track moving at this frame? Trained on the `moving` label."""

import pandas as pd


def train(features: pd.DataFrame, labels: pd.Series, cfg: dict):
    """Fit the model (e.g. LightGBM) and return it."""
    raise NotImplementedError


def predict(model, features: pd.DataFrame) -> pd.Series:
    """Return P(moving) per row."""
    raise NotImplementedError
