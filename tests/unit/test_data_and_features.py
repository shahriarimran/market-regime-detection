from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from milestone_1_regime_detection.feature_engineering import create_features
from milestone_1_regime_detection.load_data import load_usdtry_csv


pytestmark = pytest.mark.unit


def test_loader_sorts_and_parses_valid_data(tmp_path):
    path = tmp_path / "prices.csv"
    path.write_text("Date,USDTRY\n01/03/2024,30.0\n01/02/2024,29.5\n", encoding="utf-8")
    result = load_usdtry_csv(path)
    assert result["Date"].tolist() == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert result["USDTRY"].dtype.kind in "fi"


@pytest.mark.parametrize(
    "content, match",
    [
        ("Date,Other\n01/02/2024,1\n", "Missing required columns"),
        ("Date,USDTRY\nnot-a-date,1\n", "time data|format"),
        ("Date,USDTRY\n01/02/2024,x\n", "Unable to parse|numeric"),
        ("Date,USDTRY\n01/02/2024,1\n01/02/2024,2\n", "Duplicate dates"),
    ],
)
def test_loader_rejects_malformed_input(tmp_path, content, match):
    path = tmp_path / "bad.csv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises((ValueError, TypeError), match=match):
        load_usdtry_csv(path)


def test_feature_formulas_match_independent_calculation(synthetic_prices):
    result = create_features(synthetic_prices)
    row = result.iloc[-1]
    prices = synthetic_prices["USDTRY"]
    returns = prices.pct_change()
    ma20 = prices.rolling(20).mean()
    expected = {
        "Return_1D": returns.iloc[-1],
        "Return_5D": prices.pct_change(5).iloc[-1],
        "Volatility_5D": returns.rolling(5).std().iloc[-1],
        "Volatility_20D": returns.rolling(20).std().iloc[-1],
        "Volatility_60D": returns.rolling(60).std().iloc[-1],
        "MA_Distance_20D": prices.iloc[-1] / ma20.iloc[-1] - 1,
        "MA_Slope_20D": ma20.pct_change(20).iloc[-1],
        "Drawdown_60D": prices.iloc[-1] / prices.rolling(60).max().iloc[-1] - 1,
    }
    for name, value in expected.items():
        assert row[name] == pytest.approx(value, abs=1e-12)
    assert np.isfinite(result[list(expected)]).all().all()


def test_m1_features_are_causal(synthetic_prices):
    cutoff = synthetic_prices.loc[100, "Date"]
    changed = synthetic_prices.copy()
    changed.loc[changed["Date"] > cutoff, "USDTRY"] *= 10
    left = create_features(synthetic_prices).query("Date <= @cutoff").reset_index(drop=True)
    right = create_features(changed).query("Date <= @cutoff").reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)

