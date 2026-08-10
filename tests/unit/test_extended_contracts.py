from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from milestone_1_regime_detection.chronological_validation import causal_filter
from milestone_1_regime_detection.feature_engineering import create_features
from milestone_2_anomaly_detection.prepare_anomaly_features import add_anomaly_features
from milestone_2_anomaly_detection.statistical_baseline import calculate_baseline
from milestone_3_direction.operational_inference import temperature_scale

from test_milestone_1 import deterministic_hmm


pytestmark = pytest.mark.unit


def test_feature_warmup_is_dropped_and_first_valid_row_is_causal(synthetic_prices):
    result = create_features(synthetic_prices.iloc[:80])
    assert len(result) == 20
    assert result.iloc[0]["Date"] == synthetic_prices.iloc[60]["Date"]
    assert result.isna().sum().sum() == 0


def test_hmm_filter_matches_hmmlearn_forward_log_likelihood():
    model = deterministic_hmm()
    x = np.array([[-0.8], [-0.2], [0.4], [1.2], [0.1]])
    probabilities, likelihood = causal_filter(model, x)
    assert (probabilities >= 0).all()
    assert likelihood == pytest.approx(model.score(x), abs=1e-10)


def test_m2_anomaly_math_for_contemporaneous_columns():
    source = pd.DataFrame({
        "Return_1D": [0.01] * 25,
        "Volatility_5D": [0.02] * 25,
        "Volatility_60D": [0.01] * 25,
        "Drawdown_60D": np.linspace(-0.10, -0.02, 25),
    })
    result = add_anomaly_features(source, volatility_floor=0.001)
    row = result.iloc[-1]
    assert row["Absolute_Return_1D"] == pytest.approx(0.01)
    assert row["Volatility_Ratio_5D_60D"] == pytest.approx(2.0)
    assert row["Drawdown_Change_5D"] == pytest.approx(source["Drawdown_60D"].iloc[-1] - source["Drawdown_60D"].iloc[-6])


def test_m2_primary_labels_derive_only_from_statistical_rules():
    frame = pd.DataFrame({
        "Return_ZScore_20D": [0.0, 4.0],
        "Volatility_Ratio_5D_60D": [1.0, 1.0],
        "Drawdown_Change_5D": [0.0, 0.0],
        "Absolute_Return_1D": [0.0, 0.0],
        "IF_Anomaly_Score": [999.0, -999.0],
    })
    result = calculate_baseline(frame)
    labels = np.where(result["Baseline_Anomaly"], "ANOMALOUS", "NORMAL").tolist()
    assert labels == ["NORMAL", "ANOMALOUS"]


def test_m3_predictors_are_causal_even_when_future_prices_change(synthetic_prices):
    cutoff = synthetic_prices.loc[100, "Date"]
    changed = synthetic_prices.copy()
    changed.loc[changed["Date"] > cutoff, "USDTRY"] *= np.linspace(2, 20, len(changed.loc[changed["Date"] > cutoff]))
    original = create_features(synthetic_prices).query("Date == @cutoff").iloc[0]
    perturbed = create_features(changed).query("Date == @cutoff").iloc[0]
    pd.testing.assert_series_equal(original, perturbed)


def test_temperature_scaling_preserves_class_count_order_and_is_deterministic():
    probabilities = np.array([[0.1, 0.7, 0.2]])
    first = temperature_scale(probabilities, 0.8418048930199643)
    second = temperature_scale(probabilities, 0.8418048930199643)
    assert first.shape == probabilities.shape
    assert first.argmax(axis=1).tolist() == [1]
    np.testing.assert_array_equal(first, second)


