from __future__ import annotations

import json

import pandas as pd
import pytest


pytestmark = [pytest.mark.regression, pytest.mark.slow]


def test_m1_documented_profiles_and_model_comparison(project_root):
    profiles = pd.read_csv(project_root / "outputs/model_state_feature_profiles.csv").query("Model == 'HMM'").set_index("State")
    expected = {
        0: ("Elevated-Volatility Transition", [0.00089, 0.00425, 0.00558, 0.00649, 0.00760, 0.00768, 0.01561, -0.02618]),
        1: ("Low-Volatility Trend", [0.00069, 0.00336, 0.00119, 0.00139, 0.00182, 0.00634, 0.01293, -0.00038]),
        2: ("High-Volatility Stress", [0.00281, 0.01526, 0.01791, 0.02009, 0.02023, 0.02715, 0.06459, -0.07742]),
    }
    columns = ["Return_1D", "Return_5D", "Volatility_5D", "Volatility_20D", "Volatility_60D", "MA_Distance_20D", "MA_Slope_20D", "Drawdown_60D"]
    for state, (name, values) in expected.items():
        assert profiles.loc[state, "Regime_Name"] == name
        for column, value in zip(columns, values):
            assert profiles.loc[state, column] == pytest.approx(value, abs=1e-3)
    comparison = pd.read_csv(project_root / "outputs/model_comparison_summary.csv", index_col=0)
    assert comparison.loc["K-Means", "Median_Episode_Duration"] == 3
    assert comparison.loc["HMM", "Median_Episode_Duration"] == 14
    assert comparison.loc["K-Means", "Mean_Self_Transition"] == pytest.approx(0.9070, abs=5e-5)
    assert comparison.loc["HMM", "Mean_Self_Transition"] == pytest.approx(0.9674, abs=5e-5)


def test_m1_walk_forward_stability_and_operational_reference(project_root):
    walk = pd.read_csv(project_root / "outputs/validation/walk_forward_summary.csv")
    ordering = pd.read_csv(project_root / "outputs/validation/state_ordering_checks.csv")
    separation = pd.read_csv(project_root / "outputs/validation/state_separation.csv")
    latest = json.loads((project_root / "outputs/operational/latest_regime.json").read_text(encoding="utf-8"))
    assert walk["Switch_Rate"].mean() == pytest.approx(0.0327, abs=5e-4)
    assert walk["Median_Episode_Duration"].median() == 19
    assert ordering["Passed_Checks"].sum() / ordering["Total_Checks"].sum() == pytest.approx(0.9583, abs=5e-5)
    assert separation["Minimum_Separation"].min() == pytest.approx(1.0175, abs=5e-5)
    assert separation["Minimum_Separation"].mean() == pytest.approx(1.8574, abs=5e-5)
    stress_years = set(pd.read_csv(project_root / "outputs/validation/walk_forward_predictions.csv").query("Regime_Name == 'High-Volatility Stress'")["Test_Year"])
    assert stress_years == {2021, 2022, 2023}
    assert latest["date"] == "2026-08-07"
    assert latest["usdtry"] == pytest.approx(47.7111)
    assert latest["regime_code"] == "LOW_VOL"
    assert latest["confidence"] > 0.999999
    assert latest["inference_mode"] == "TRAINING_REFERENCE"


def test_m2_documented_robustness(project_root):
    path = project_root / "src/milestone_2_anomaly_detection/outputs/stability_sensitivity/if_stability_summary.csv"
    summary = pd.read_csv(path).set_index("Metric")
    assert summary.loc["Mean IF seed pairwise Jaccard", "Value"] == pytest.approx(0.97, abs=0.005)


def test_m3_benchmark_robustness_selection_and_first_prediction(project_root):
    validation = project_root / "outputs/milestone_3/validation"
    summary = pd.read_csv(validation / "purged_rf_vs_momentum_summary.csv").set_index("Model")
    per_class = pd.read_csv(validation / "purged_rf_vs_momentum_per_class.csv")
    seeds = pd.read_csv(validation / "rf_seed_stability.csv")
    agreement = pd.read_csv(validation / "rf_seed_prediction_agreement.csv")
    selection = pd.read_csv(validation / "final_model_selection.csv").iloc[0]
    latest = json.loads((project_root / "outputs/milestone_3/operational/latest_direction.json").read_text(encoding="utf-8"))
    rf, momentum = summary.loc["RandomForest"], summary.loc["Momentum_5D"]
    assert rf["Accuracy"] == pytest.approx(0.6456, abs=5e-5)
    assert rf["Balanced_Accuracy"] == pytest.approx(0.5341, abs=5e-5)
    assert rf["Macro_F1"] == pytest.approx(0.5349, abs=5e-5)
    down_f1 = per_class.query("Model == 'RandomForest' and Class == 'DOWN'")["F1"].iloc[0]
    assert down_f1 == pytest.approx(0.2809, abs=5e-5)
    assert momentum["Balanced_Accuracy"] == pytest.approx(0.5006, abs=5e-5)
    assert momentum["Macro_F1"] == pytest.approx(0.5003, abs=5e-5)
    assert rf["Balanced_Accuracy"] - momentum["Balanced_Accuracy"] == pytest.approx(0.0335, abs=1e-4)
    assert (seeds["Delta_BA_vs_Momentum"] > 0).all()
    assert agreement["Prediction_Agreement"].mean() == pytest.approx(0.9780, abs=5e-5)
    assert int(selection["Acceptance_Gates_Passed"]) == int(selection["Acceptance_Gates_Total"]) == 11
    assert latest["direction"] == "FLAT"
    assert latest["confidence"] == pytest.approx(0.8794, abs=5e-5)
    assert latest["calibrated_probabilities"]["DOWN"] == pytest.approx(0.0027, abs=5e-5)
    assert latest["calibrated_probabilities"]["FLAT"] == pytest.approx(0.8794, abs=5e-5)
    assert latest["calibrated_probabilities"]["UP"] == pytest.approx(0.1179, abs=5e-5)


