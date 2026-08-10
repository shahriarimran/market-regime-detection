from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=140)
    returns = 0.001 + 0.004 * np.sin(np.arange(len(dates)) / 7.0)
    prices = 6.0 * np.cumprod(1.0 + returns)
    return pd.DataFrame({"Date": dates, "USDTRY": prices})

