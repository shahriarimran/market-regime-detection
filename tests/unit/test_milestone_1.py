from __future__ import annotations

import numpy as np
import pytest
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from milestone_1_regime_detection.chronological_validation import (
    FEATURE_COLUMNS,
    causal_filter,
    get_state_mapping,
)


pytestmark = pytest.mark.unit


def deterministic_hmm() -> GaussianHMM:
    model = GaussianHMM(n_components=3, covariance_type="diag", init_params="")
    model.startprob_ = np.array([0.6, 0.3, 0.1])
    model.transmat_ = np.array([[0.8, 0.15, 0.05], [0.1, 0.8, 0.1], [0.05, 0.15, 0.8]])
    model.means_ = np.array([[-1.0], [0.0], [1.0]])
    model.n_features = 1
    model.covars_ = np.array([[0.4], [0.4], [0.4]])
    return model


def test_causal_filter_probabilities_and_future_independence():
    model = deterministic_hmm()
    x = np.array([[-0.8], [-0.2], [0.4], [1.2], [0.1]])
    changed = x.copy()
    changed[3:] += 100
    probs, likelihood = causal_filter(model, x)
    changed_probs, _ = causal_filter(model, changed)
    assert probs.shape == (5, 3)
    assert np.isfinite(likelihood)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(probs[:3], changed_probs[:3], atol=1e-12)


def test_state_mapping_uses_volatility_order_not_numeric_state_id():
    scaler = StandardScaler().fit(np.vstack([np.zeros(8), np.ones(8)]))
    model = type("Model", (), {})()
    model.means_ = np.zeros((3, len(FEATURE_COLUMNS)))
    vol_index = FEATURE_COLUMNS.index("Volatility_20D")
    model.means_[:, vol_index] = [1.0, -1.0, 0.0]
    names, means = get_state_mapping(model, scaler)
    assert names[1] == "Low-Volatility Trend"
    assert names[2] == "Elevated-Volatility Transition"
    assert names[0] == "High-Volatility Stress"
    assert set(means["HMM_State"]) == {0, 1, 2}



