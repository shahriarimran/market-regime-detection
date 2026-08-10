from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from milestone_1_regime_detection.chronological_validation import FEATURE_COLUMNS, causal_filter
from milestone_2_anomaly_detection.train_final_anomaly_model import empirical_percentile
from milestone_3_direction.operational_inference import temperature_scale


pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_m1_frozen_artifact_runs_causal_inference(project_root):
    artifact = joblib.load(project_root / "models/usdtry_hmm3.joblib")
    features = pd.read_csv(project_root / "data/processed/usdtry_features.csv")
    x = artifact["scaler"].transform(features[FEATURE_COLUMNS].tail(10))
    probs, likelihood = causal_filter(artifact["model"], x, artifact["final_filtered_probability"])
    assert probs.shape == (10, 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-10)
    assert np.isfinite(likelihood)


def test_m2_frozen_artifact_and_percentile_are_deterministic(project_root):
    model_dir = project_root / "src/milestone_2_anomaly_detection/models"
    artifact = joblib.load(model_dir / "usdtry_isolation_forest.joblib")
    model = artifact["model"]
    reference = np.load(model_dir / "usdtry_if_training_scores.npy")
    metadata = json.loads((model_dir / "usdtry_anomaly_metadata.json").read_text(encoding="utf-8"))
    assert model.random_state == 42
    assert artifact["feature_columns"] == metadata["features"]
    assert artifact["volatility_floor"] == pytest.approx(
        metadata["volatility_floor"]["value"]
    )
    assert artifact["volatility_floor_quantile"] == pytest.approx(0.05)
    assert len(reference) == metadata["training_observations"]
    assert 0.0 <= empirical_percentile(reference[-1], np.sort(reference)) <= 1.0


def test_m3_frozen_artifact_predicts_with_declared_schema(project_root):
    model_dir = project_root / "models/milestone_3"
    model = joblib.load(model_dir / "usdtry_direction_rf.joblib")
    metadata = json.loads((model_dir / "usdtry_direction_metadata.json").read_text(encoding="utf-8"))
    temperature = json.loads((model_dir / "usdtry_direction_temperature.json").read_text(encoding="utf-8"))
    data = pd.read_csv(project_root / "data/processed/usdtry_direction_features.csv")
    raw = model.predict_proba(data[metadata["features"]].tail(1))
    calibrated = temperature_scale(raw, temperature["temperature"])
    assert list(model.classes_) == temperature["class_order"]
    np.testing.assert_allclose(calibrated.sum(axis=1), 1.0, atol=1e-12)
    assert model.predict(data[metadata["features"]].tail(1))[0] in temperature["class_order"]


def test_artifacts_round_trip_without_prediction_drift(project_root, tmp_path):
    paths = [
        project_root / "models/usdtry_hmm3.joblib",
        project_root / "src/milestone_2_anomaly_detection/models/usdtry_isolation_forest.joblib",
        project_root / "models/milestone_3/usdtry_direction_rf.joblib",
    ]
    for index, path in enumerate(paths):
        loaded = joblib.load(path)
        copy = tmp_path / f"artifact-{index}.joblib"
        joblib.dump(loaded, copy)
        reloaded = joblib.load(copy)
        assert type(reloaded) is type(loaded)


