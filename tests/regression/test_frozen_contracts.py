from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from milestone_2_anomaly_detection.prepare_anomaly_features import (
    add_anomaly_features,
    estimate_volatility_floor,
)


pytestmark = [pytest.mark.regression, pytest.mark.slow]


BASE_FEATURES = [
    "Return_1D", "Return_5D", "Volatility_5D", "Volatility_20D",
    "Volatility_60D", "MA_Distance_20D", "MA_Slope_20D", "Drawdown_60D",
]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_m1_metadata_and_artifact_contract(project_root):
    metadata = read_json(project_root / "models/usdtry_hmm3_metadata.json")
    artifact = joblib.load(project_root / "models/usdtry_hmm3.joblib")
    assert metadata["n_states"] == 3
    assert metadata["features"] == BASE_FEATURES
    assert metadata["validation_status"] == "PASSED"
    assert metadata["acceptance_gates"] == "7/7"
    assert artifact["feature_columns"] == BASE_FEATURES
    assert artifact["training_observations"] == metadata["training_observations"]
    assert pd.Timestamp(artifact["training_start_date"]).strftime("%Y-%m-%d") == metadata["training_start_date"]
    assert pd.Timestamp(artifact["training_end_date"]).strftime("%Y-%m-%d") == metadata["training_end_date"]


def test_m2_metadata_matches_reference_outputs(project_root):
    base = project_root / "src/milestone_2_anomaly_detection"
    metadata = read_json(base / "models/usdtry_anomaly_metadata.json")
    latest = read_json(base / "outputs/operational/latest_anomaly.json")
    summary = pd.read_csv(base / "outputs/final_training/final_training_summary.csv").iloc[0]
    assert metadata["primary_operational_detector"] == "STATISTICAL_BASELINE"
    assert metadata["secondary_ml_signal"] == "ISOLATION_FOREST_SCORE"
    assert metadata["model_version"] == "M2-v0.2.0"
    assert metadata["volatility_floor"]["quantile"] == pytest.approx(0.05)
    assert metadata["volatility_floor"]["estimation_sample"] == "training_only"
    assert int(summary["Observations"]) == metadata["training_observations"]
    assert int(summary["Baseline_Anomalies"]) == metadata["historical_baseline_anomaly_count"]
    assert summary["Baseline_Anomaly_Rate"] == pytest.approx(metadata["historical_baseline_anomaly_rate"])
    assert latest["date"] == metadata["training_cutoff"]
    assert latest["baseline_anomaly"] == (latest["baseline_anomaly_score"] >= 1.0)
    assert metadata["volatility_floor"]["value"] == pytest.approx(0.0007028294836738612, abs=1e-15)
    assert summary["IF_Score_Median"] == pytest.approx(0.3662563024470112, abs=1e-12)
    assert summary["IF_Score_P95"] == pytest.approx(0.5363940192620982, abs=1e-12)
    assert latest["baseline_anomaly_score"] == pytest.approx(0.396268979573304, abs=1e-12)
    assert latest["if_anomaly_score"] == pytest.approx(0.3472304544707059, abs=1e-12)
    assert latest["if_training_percentile"] == pytest.approx(0.2338144329896907, abs=1e-12)


def test_m3_metadata_model_and_operational_output_contract(project_root):
    model_dir = project_root / "models/milestone_3"
    metadata = read_json(model_dir / "usdtry_direction_metadata.json")
    temperature = read_json(model_dir / "usdtry_direction_temperature.json")
    latest = read_json(project_root / "outputs/milestone_3/operational/latest_direction.json")
    model = joblib.load(model_dir / "usdtry_direction_rf.joblib")
    params = metadata["random_forest_parameters"]
    for name in ("n_estimators", "max_depth", "min_samples_leaf", "max_features", "class_weight", "random_state"):
        assert model.get_params()[name] == params[name]
    assert metadata["features"] == BASE_FEATURES
    assert metadata["use_m1_features"] is False
    assert metadata["use_m2_features"] is False
    assert temperature["temperature"] > 0
    assert temperature["class_order"] == ["DOWN", "FLAT", "UP"]
    assert latest["model_version"] == metadata["model_version"]
    assert latest["training_cutoff"] == metadata["training"]["end_date"]
    assert latest["direction"] == max(latest["calibrated_probabilities"], key=latest["calibrated_probabilities"].get)
    assert sum(latest["calibrated_probabilities"].values()) == pytest.approx(1.0)


def test_documented_validation_metrics_match_frozen_summary(project_root):
    metadata = read_json(project_root / "models/milestone_3/usdtry_direction_metadata.json")
    summary = pd.read_csv(project_root / "outputs/milestone_3/validation/purged_rf_vs_momentum_summary.csv")
    rf = summary.loc[summary["Model"] == "RandomForest"].iloc[0]
    reference = metadata["validation_reference"]
    assert rf["Balanced_Accuracy"] == pytest.approx(reference["purged_oos_balanced_accuracy"], abs=1e-12)
    assert rf["Macro_F1"] == pytest.approx(reference["purged_oos_macro_f1"], abs=1e-12)

def test_m2_future_observations_cannot_change_prior_features(project_root):
    raw = pd.read_csv(
        project_root / "data/processed/usdtry_features.csv",
        parse_dates=["Date"],
    )
    cutoff = pd.Timestamp("2022-12-30")
    training = raw[raw["Date"] <= cutoff].copy()
    floor = estimate_volatility_floor(training)

    changed = raw.copy()
    future = changed["Date"] > cutoff
    changed.loc[future, "Return_1D"] *= 100.0
    changed.loc[future, "Drawdown_60D"] -= 10.0

    original_features = add_anomaly_features(
        raw,
        volatility_floor=floor,
    ).query("Date <= @cutoff").reset_index(drop=True)
    changed_features = add_anomaly_features(
        changed,
        volatility_floor=floor,
    ).query("Date <= @cutoff").reset_index(drop=True)

    pd.testing.assert_frame_equal(
        original_features,
        changed_features,
    )


