from __future__ import annotations

import json
import os
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

from milestone_1_regime_detection.chronological_validation import FEATURE_COLUMNS, causal_filter
from milestone_2_anomaly_detection.statistical_baseline import calculate_baseline


pytestmark = pytest.mark.integration


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_real_processed_market_data_quality(project_root):
    data = pd.read_csv(project_root / "data/processed/usdtry_features.csv", parse_dates=["Date"])
    assert np.isfinite(data["USDTRY"]).all()
    assert (data["USDTRY"] > 0).all()
    assert (data["Date"].dt.dayofweek < 5).all()
    assert data[FEATURE_COLUMNS].notna().all().all()
    assert np.isfinite(data[FEATURE_COLUMNS]).all().all()


def test_m1_operational_schema_and_semantics(project_root):
    output = read_json(project_root / "outputs/operational/latest_regime.json")
    assert set(output) >= {"date", "usdtry", "state", "regime_code", "regime_name", "confidence", "probabilities", "inference_mode"}
    assert output["regime_code"] in {"LOW_VOL", "ELEVATED_VOL", "HIGH_VOL_STRESS"}
    assert not isinstance(output["regime_code"], int)
    assert sum(output["probabilities"].values()) == pytest.approx(1.0)


def test_m2_operational_schema_and_primary_detector(project_root):
    base = project_root / "src/milestone_2_anomaly_detection"
    output = read_json(base / "outputs/operational/latest_anomaly.json")
    metadata = read_json(base / "models/usdtry_anomaly_metadata.json")
    assert output["anomaly_state"] in {"NORMAL", "ANOMALOUS"}
    assert output["primary_detector"] == metadata["primary_operational_detector"] == "STATISTICAL_BASELINE"
    assert output["secondary_ml_signal"] == "ISOLATION_FOREST_SCORE"
    assert output["anomaly_state"] == ("ANOMALOUS" if output["baseline_anomaly"] else "NORMAL")


def test_shared_base_features_match_m1_and_m2_persisted_data(project_root):
    m1 = pd.read_csv(project_root / "data/processed/usdtry_features.csv")
    m2 = pd.read_csv(project_root / "src/milestone_2_anomaly_detection/data/usdtry_anomaly_features.csv")
    merged = m1[["Date", *FEATURE_COLUMNS]].merge(m2[["Date", *FEATURE_COLUMNS]], on="Date", suffixes=("_m1", "_m2"))
    assert len(merged) > 2000
    for feature in FEATURE_COLUMNS:
        np.testing.assert_allclose(merged[f"{feature}_m1"], merged[f"{feature}_m2"], rtol=0, atol=1e-14)


def test_m3_validation_purge_prevents_target_overlap(project_root):
    predictions = pd.read_csv(project_root / "outputs/milestone_3/validation/purged_rf_predictions.csv", parse_dates=["Date"])
    features = pd.read_csv(project_root / "data/processed/usdtry_direction_features.csv", parse_dates=["Date"])
    dates = features["Date"].sort_values().reset_index(drop=True)
    for year in sorted(predictions["Test_Year"].unique()):
        first_test = predictions.loc[predictions["Test_Year"] == year, "Date"].min()
        first_position = dates[dates == first_test].index[0]
        last_legal_train = dates.iloc[first_position - 6]
        assert last_legal_train < first_test


def test_source_modules_import_from_repository_root(project_root):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src")
    modules = [
        "milestone_1_regime_detection.operational_inference",
        "milestone_2_anomaly_detection.operational_anomaly_inference",
        "milestone_3_direction.operational_inference",
    ]
    result = subprocess.run(
        [sys.executable, "-c", ";".join(f"import {module}" for module in modules)],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr

