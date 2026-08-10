from __future__ import annotations

import json

import pandas as pd
import pytest


pytestmark = pytest.mark.integration


def test_processed_dataset_contracts_align(project_root):
    m1 = pd.read_csv(project_root / "data/processed/usdtry_features.csv", parse_dates=["Date"])
    m3 = pd.read_csv(project_root / "data/processed/usdtry_direction_features.csv", parse_dates=["Date"])
    assert m1["Date"].is_monotonic_increasing and m1["Date"].is_unique
    assert m3["Date"].is_monotonic_increasing and m3["Date"].is_unique
    assert set(m3["Date"]).issubset(set(m1["Date"]))
    assert m3[["Return_1D", "Return_5D", "Volatility_5D", "Volatility_20D", "Volatility_60D", "MA_Distance_20D", "MA_Slope_20D", "Drawdown_60D"]].notna().all().all()


def test_no_project_4_source_or_artifacts_exist(project_root):
    names = [path.name.lower() for path in project_root.rglob("*")]
    assert not any("milestone_4" in name or "project_4" in name for name in names)


def test_frozen_training_dates_do_not_exceed_operational_observation(project_root):
    m1 = json.loads((project_root / "models/usdtry_hmm3_metadata.json").read_text(encoding="utf-8"))
    m2 = json.loads((project_root / "src/milestone_2_anomaly_detection/models/usdtry_anomaly_metadata.json").read_text(encoding="utf-8"))
    m3 = json.loads((project_root / "models/milestone_3/usdtry_direction_metadata.json").read_text(encoding="utf-8"))
    latest = json.loads((project_root / "outputs/milestone_3/operational/latest_direction.json").read_text(encoding="utf-8"))
    observation = pd.Timestamp(latest["observation_date"])
    assert pd.Timestamp(m1["training_end_date"]) <= observation
    assert pd.Timestamp(m2["training_cutoff"]) <= observation
    assert pd.Timestamp(m3["training"]["end_date"]) < observation

