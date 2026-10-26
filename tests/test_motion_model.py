"""Tests for the training labels of the motion model."""

import numpy as np
import pandas as pd

from whomoves.motion_model import NONE_WEIGHT, training_labels


def test_labels_and_weights_by_role():
    feats = pd.DataFrame({
        "gt_role": ["target", "target", "target", "static", "none", "ambiguous", "driver"],
        "gt_moving": [0.0, 1.0, np.nan, 0.0, np.nan, np.nan, 0.0],
    })
    label, weight = training_labels(feats)
    assert label.tolist() == [0, 1, 0, 0, 0, 0, 0]
    # target without a GT box in that frame, and ambiguous objects, are not used
    assert weight.tolist() == [1, 1, 0, 1, NONE_WEIGHT, 0, 1]
