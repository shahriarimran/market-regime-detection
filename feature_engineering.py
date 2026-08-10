from pathlib import Path

import pandas as pd

from load_data import load_usdtry_csv

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create market-regime features from USD/TRY daily prices.
    """

    features = df.copy()

    # ---------------------------------
    # 1. Returns / momentum
    # ---------------------------------

    features["Return_1D"] = features["USDTRY"].pct_change(1)

    features["Return_5D"] = features["USDTRY"].pct_change(5)

    # ---------------------------------
    # 2. Rolling volatility
    # ---------------------------------

    features["Volatility_5D"] = (
        features["Return_1D"]
        .rolling(window=5)
        .std()
    )

    features["Volatility_20D"] = (
        features["Return_1D"]
        .rolling(window=20)
        .std()
    )

    features["Volatility_60D"] = (
        features["Return_1D"]
        .rolling(window=60)
        .std()
    )

    # ---------------------------------
    # 3. Moving averages
    # ---------------------------------

    ma20 = features["USDTRY"].rolling(window=20).mean()

    features["MA_Distance_20D"] = (
        features["USDTRY"] / ma20 - 1
    )

    # ---------------------------------
    # 4. 20-day MA slope
    # ---------------------------------

    features["MA_Slope_20D"] = ma20.pct_change(20)

    # ---------------------------------
    # 5. 60-day drawdown
    # ---------------------------------

    rolling_high_60 = (
        features["USDTRY"]
        .rolling(window=60)
        .max()
    )

    features["Drawdown_60D"] = (
        features["USDTRY"] / rolling_high_60 - 1
    )

    # ---------------------------------
    # Remove rows where rolling windows
    # do not yet have enough history
    # ---------------------------------

    features = features.dropna().reset_index(drop=True)

    return features

if __name__ == "__main__":

    csv_file = Path(
        "data/raw/Financial_Trackers_AutoRun_ML_clean.csv"
    )

    df = load_usdtry_csv(csv_file)

    features = create_features(df)

    print("\nFeature engineering complete.")
    print(f"Raw observations: {len(df):,}")
    print(f"Usable observations: {len(features):,}")

    print("\nFeature columns:")
    print(features.columns.tolist())

    print("\nFirst 5 feature rows:")
    print(features.head())

    print("\nSummary statistics:")
    print(features.describe())

    output_file = Path(
        "data/processed/usdtry_features.csv"
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    features.to_csv(
        output_file,
        index=False
    )

    print(f"\nSaved to: {output_file}")