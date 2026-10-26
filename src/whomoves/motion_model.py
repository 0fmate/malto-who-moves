"""Classifier: has this track started to move, at this frame? Trained on the GT `moving` label.

LightGBM (gradient-boosted decision trees): many small trees are added one after the other, each
correcting the errors of the previous ones. It works well on tabular features like ours, needs no
feature scaling, and trains in seconds.

Labels come from match.py (our tracks matched to the annotated ones):
- target / bonus: the GT `moving` value at that frame ("has started moving");
- static / disturbance / driver: 0 (must never alarm);
- none (no annotated object): 0 with a lower weight, since it may be a real mover nobody annotated;
- ambiguous / other: left out (not scored by the competition).
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS

NONE_WEIGHT = 0.3
PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=200,
              subsample=0.5, subsample_freq=1, colsample_bytree=0.8, random_state=42, verbose=-1)


def model_columns() -> list[str]:
    # first_t is left out: train videos are whole clips, test videos are 1-3 min cuts, so the time a
    # track appears means different things; "present from the start" is kept as a binary flag.
    return [c for c in FEATURE_COLUMNS if c != "first_t"] + ["at_video_start", "is_person", "is_vehicle"]


def model_inputs(feats: pd.DataFrame) -> pd.DataFrame:
    x = feats.reindex(columns=[c for c in FEATURE_COLUMNS if c != "first_t"])
    return x.assign(at_video_start=(feats["first_t"] < 1.0).astype(float),
                    is_person=(feats["class"] == "person").astype(float),
                    is_vehicle=(feats["class"] == "vehicle").astype(float))


def training_labels(feats: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(label, weight) per row; rows that must not be used get weight 0."""
    role = feats["gt_role"]
    label = pd.Series(0.0, index=feats.index)
    weight = pd.Series(1.0, index=feats.index)
    movers = role.isin(["target", "bonus"])
    label[movers] = feats.loc[movers, "gt_moving"]
    weight[movers & feats["gt_moving"].isna()] = 0.0   # GT track has no box in this frame
    weight[role == "none"] = NONE_WEIGHT
    weight[role.isin(["ambiguous", "other"])] = 0.0
    return label.fillna(0.0), weight


def train(feats: pd.DataFrame, params: dict | None = None) -> lgb.LGBMClassifier:
    label, weight = training_labels(feats)
    keep = weight > 0
    model = lgb.LGBMClassifier(**{**PARAMS, **(params or {})})
    model.fit(model_inputs(feats[keep]), label[keep].astype(int), sample_weight=weight[keep])
    return model


def predict(model, feats: pd.DataFrame) -> pd.Series:
    """P(has started moving) per row; model is an LGBMClassifier or a Booster loaded from file."""
    if not len(feats):
        return pd.Series(dtype=float)
    x = model_inputs(feats)
    p = model.predict_proba(x)[:, 1] if hasattr(model, "predict_proba") else model.predict(x)
    return pd.Series(p, index=feats.index)


def load(path) -> lgb.Booster:
    return lgb.Booster(model_file=str(path))


def importance(model: lgb.LGBMClassifier, top: int = 10) -> pd.Series:
    gain = model.booster_.feature_importance("gain")
    return pd.Series(gain / gain.sum(), index=model_columns()).sort_values(ascending=False).head(top)


def smooth(p: pd.Series, feats: pd.DataFrame, rows: int = 5) -> pd.Series:
    """Centered rolling mean of P within each track (rows sorted by time), to remove flicker."""
    order = feats.sort_values(["video_id", "track_id", "t"], kind="stable").index
    s = p.loc[order].groupby([feats.loc[order, "video_id"], feats.loc[order, "track_id"]])
    out = s.transform(lambda v: v.rolling(rows, center=True, min_periods=1).mean())
    return out.reindex(p.index).astype(np.float64)
