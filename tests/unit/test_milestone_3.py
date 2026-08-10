from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from milestone_3_direction.direction_feature_engineering import FEATURE_COLUMNS
from milestone_3_direction.operational_inference import temperature_scale
from milestone_3_direction.purged_validation import HORIZON, purge_training_boundary
from milestone_3_direction.target_engineering import classify_direction


pytestmark = pytest.mark.unit


def test_target_threshold_boundaries_and_missing_values():
    values = pd.Series([-0.006, -0.005, 0.0, 0.005, 0.006, np.nan])
    labels = classify_direction(values, 0.005).tolist()
    assert labels == ["DOWN", "FLAT", "FLAT", "FLAT", "UP", None]


def test_five_day_future_return_uses_observation_horizon():
    prices = pd.Series([100, 101, 102, 103, 104, 110, 120, 130, 140, 150], dtype=float)
    future = prices.shift(-5) / prices - 1
    assert future.iloc[0] == pytest.approx(0.10)
    assert future.iloc[4] == pytest.approx(150 / 104 - 1)
    assert future.tail(HORIZON).isna().all()


def test_production_feature_list_contains_no_target_or_cross_milestone_features():
    assert len(FEATURE_COLUMNS) == 8
    forbidden = ("Target", "Future", "M1_", "M2_")
    assert not any(name.startswith(forbidden) for name in FEATURE_COLUMNS)


def test_purge_removes_exactly_the_target_horizon():
    dates = pd.bdate_range("2024-01-01", periods=10)
    train = pd.DataFrame({"Date": dates[::-1], "value": range(10)})
    purged = purge_training_boundary(train)
    assert len(purged) == 10 - HORIZON
    assert purged["Date"].max() == sorted(dates)[-HORIZON - 1]


def test_purge_rejects_insufficient_history():
    with pytest.raises(ValueError, match="Insufficient training observations"):
        purge_training_boundary(pd.DataFrame({"Date": pd.bdate_range("2024-01-01", periods=HORIZON)}))


def test_temperature_scaling_identity_and_normalization():
    probabilities = np.array([[0.2, 0.3, 0.5], [0.0, 0.5, 0.5]])
    calibrated = temperature_scale(probabilities, 1.0)
    np.testing.assert_allclose(calibrated.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(calibrated[0], probabilities[0], atol=1e-12)
    assert np.isfinite(calibrated).all()


@pytest.mark.parametrize("temperature", [0.0, -1.0])
def test_temperature_must_be_positive(temperature):
    with pytest.raises(ValueError, match="positive"):
        temperature_scale([[0.2, 0.3, 0.5]], temperature)

