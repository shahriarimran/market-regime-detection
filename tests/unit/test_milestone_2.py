from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from milestone_2_anomaly_detection.prepare_anomaly_features import (
    add_anomaly_features,
    estimate_volatility_floor,
)
from milestone_2_anomaly_detection.statistical_baseline import calculate_baseline
from milestone_2_anomaly_detection.train_final_anomaly_model import empirical_percentile


pytestmark = pytest.mark.unit


def anomaly_frame(**overrides):
    values = {
        "Return_ZScore_20D": 0.0,
        "Volatility_Ratio_5D_60D": 1.0,
        "Drawdown_Change_5D": 0.0,
        "Absolute_Return_1D": 0.0,
    }
    values.update(overrides)
    return pd.DataFrame([values])


@pytest.mark.parametrize(
    "column, threshold",
    [
        ("Return_ZScore_20D", 4.0),
        ("Volatility_Ratio_5D_60D", 2.0),
        ("Drawdown_Change_5D", 0.05),
        ("Absolute_Return_1D", 0.04),
    ],
)
def test_each_baseline_threshold_is_inclusive(column, threshold):
    at = calculate_baseline(anomaly_frame(**{column: threshold})).iloc[0]
    below = calculate_baseline(anomaly_frame(**{column: np.nextafter(threshold, 0.0)})).iloc[0]
    assert bool(at["Baseline_Anomaly"])
    assert not bool(below["Baseline_Anomaly"])


def test_baseline_uses_max_normalized_rule_score():
    result = calculate_baseline(anomaly_frame(Return_ZScore_20D=-2.0, Absolute_Return_1D=0.03)).iloc[0]
    assert result["Baseline_Anomaly_Score"] == pytest.approx(0.75)
    assert result["Rules_Breached"] == 0


def test_empirical_percentile_uses_right_inclusive_rank():
    reference = np.array([1.0, 2.0, 2.0, 4.0])
    assert empirical_percentile(2.0, reference) == pytest.approx(0.75)
    assert empirical_percentile(0.0, reference) == 0.0
    assert empirical_percentile(5.0, reference) == 1.0


def test_anomaly_features_are_causal():
    n = 80
    base = pd.DataFrame({
        "Return_1D": 0.01 * np.sin(np.arange(n)),
        "Volatility_5D": np.linspace(0.01, 0.03, n),
        "Volatility_60D": np.linspace(0.02, 0.04, n),
        "Drawdown_60D": np.linspace(-0.1, 0.0, n),
    })
    changed = base.copy()
    changed.loc[61:, "Return_1D"] += 10
    training = base.iloc[:61].copy()
    volatility_floor = estimate_volatility_floor(training)
    left = add_anomaly_features(
        base, volatility_floor=volatility_floor
    ).iloc[:61]
    right = add_anomaly_features(
        changed, volatility_floor=volatility_floor
    ).iloc[:61]
    pd.testing.assert_frame_equal(left, right)


