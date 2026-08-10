from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

M2_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = M2_DIR.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "usdtry_features.csv"
)

OUTPUT_DIR = M2_DIR / "data"

OUTPUT_FILE = (
    OUTPUT_DIR
    / "usdtry_anomaly_features.csv"
)


# ============================================================
# Milestone 1 feature set
# ============================================================

M1_FEATURES = [
    "Return_1D",
    "Return_5D",
    "Volatility_5D",
    "Volatility_20D",
    "Volatility_60D",
    "MA_Distance_20D",
    "MA_Slope_20D",
    "Drawdown_60D",
]


# ============================================================
# Milestone 2 candidate features
# ============================================================

M2_EXTRA_FEATURES = [
    "Absolute_Return_1D",
    "Return_ZScore_20D",
    "Volatility_Ratio_5D_60D",
    "Drawdown_Change_5D",
]


ALL_FEATURES = (
    M1_FEATURES
    + M2_EXTRA_FEATURES
)


# ============================================================
# Data loading
# ============================================================

def load_m1_features() -> pd.DataFrame:

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Milestone 1 feature file not found:\n"
            f"{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["Date"],
    )

    required_columns = {
        "Date",
        "USDTRY",
        *M1_FEATURES,
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            "Input dataset is missing required "
            f"columns: {sorted(missing)}"
        )

    df = (
        df
        .sort_values("Date")
        .reset_index(drop=True)
    )

    return df


# ============================================================
# Feature engineering
# ============================================================

def add_anomaly_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    # --------------------------------------------------------
    # 1. Absolute daily return
    #
    # Measures magnitude of the daily move regardless
    # of direction.
    # --------------------------------------------------------

    df["Absolute_Return_1D"] = (
        df["Return_1D"].abs()
    )

    # --------------------------------------------------------
    # 2. Rolling 20-day return z-score
    #
    # Today's return is compared against the PRIOR
    # 20 observations only.
    #
    # This is strictly causal and prevents the observation
    # being scored from influencing its own reference
    # distribution.
    # --------------------------------------------------------

    prior_returns = (
        df["Return_1D"]
        .shift(1)
    )

    rolling_mean = (
        prior_returns
        .rolling(
            window=20,
            min_periods=20,
        )
        .mean()
    )

    rolling_std = (
        prior_returns
        .rolling(
            window=20,
            min_periods=20,
        )
        .std()
    )

    # Prevent extremely small local standard deviations from
    # producing numerically explosive z-scores.
    #
    # The floor is based on the 5th percentile of valid
    # historical 20-day return standard deviations.
    valid_std = rolling_std.dropna()

    std_floor = float(
        valid_std.quantile(0.05)
    )

    effective_std = (
        rolling_std.clip(
            lower=std_floor
        )
    )

    df["Return_ZScore_20D"] = (
        (
            df["Return_1D"]
            - rolling_mean
        )
        / effective_std
    )

    print(
        f"\nReturn z-score volatility floor: "
        f"{std_floor:.6f}"
    )

    # --------------------------------------------------------
    # 3. Short / long volatility ratio
    #
    # > 1:
    # short-term volatility exceeds long-term volatility
    #
    # < 1:
    # short-term environment is quieter than the
    # longer-term volatility background.
    # --------------------------------------------------------

    df["Volatility_Ratio_5D_60D"] = (
        df["Volatility_5D"]
        / df["Volatility_60D"]
    )

    # --------------------------------------------------------
    # 4. Five-day change in drawdown
    #
    # Detects rapid movement toward or away from
    # recent highs.
    # --------------------------------------------------------

    df["Drawdown_Change_5D"] = (
        df["Drawdown_60D"]
        .diff(periods=5)
    )

    return df


# ============================================================
# Validation
# ============================================================

def validate_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    # Replace any accidental infinities before
    # checking / removing incomplete rows.
    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    print(
        "\n--- MISSING VALUES BEFORE CLEANING ---"
    )

    missing = (
        df[
            ["Date", "USDTRY", *ALL_FEATURES]
        ]
        .isna()
        .sum()
    )

    print(
        missing.to_string()
    )

    rows_before = len(df)

    # New rolling features introduce expected NaNs
    # at the beginning of the dataset.
    df = (
        df.dropna(
            subset=ALL_FEATURES
        )
        .reset_index(drop=True)
    )

    rows_after = len(df)

    print(
        "\nRows removed because of "
        f"rolling/difference initialization: "
        f"{rows_before - rows_after}"
    )

    # --------------------------------------------------------
    # Duplicate dates
    # --------------------------------------------------------

    duplicate_dates = (
        df["Date"]
        .duplicated()
        .sum()
    )

    if duplicate_dates:

        raise ValueError(
            f"Found {duplicate_dates} "
            "duplicate dates."
        )

    # --------------------------------------------------------
    # Remaining missing / infinite values
    # --------------------------------------------------------

    if (
        df[ALL_FEATURES]
        .isna()
        .any()
        .any()
    ):

        raise ValueError(
            "Missing feature values remain "
            "after cleaning."
        )

    numeric_values = (
        df[ALL_FEATURES]
        .to_numpy(
            dtype=float
        )
    )

    if not np.isfinite(
        numeric_values
    ).all():

        raise ValueError(
            "Infinite feature values remain."
        )

    return df


# ============================================================
# Diagnostics
# ============================================================

def print_summary(
    df: pd.DataFrame,
) -> None:

    print(
        "\n--- MILESTONE 2 ANOMALY FEATURE DATASET ---"
    )

    print(
        f"Observations: {len(df):,}"
    )

    print(
        f"Start date: "
        f"{df['Date'].min().date()}"
    )

    print(
        f"End date: "
        f"{df['Date'].max().date()}"
    )

    print(
        f"Candidate features: "
        f"{len(ALL_FEATURES)}"
    )

    print(
        "\n--- FEATURES ---"
    )

    for feature in ALL_FEATURES:
        print(
            f"  {feature}"
        )

    print(
        "\n--- NEW FEATURE SUMMARY ---"
    )

    summary = (
        df[M2_EXTRA_FEATURES]
        .describe()
        .T
    )

    print(
        summary.round(6)
        .to_string()
    )

    print(
        "\n--- EXTREME ABSOLUTE RETURNS ---"
    )

    extreme_returns = (
        df[
            [
                "Date",
                "USDTRY",
                "Return_1D",
                "Absolute_Return_1D",
                "Return_ZScore_20D",
            ]
        ]
        .sort_values(
            "Absolute_Return_1D",
            ascending=False,
        )
        .head(10)
    )

    print(
        extreme_returns
        .to_string(
            index=False
        )
    )

    print(
        "\n--- EXTREME RETURN Z-SCORES ---"
    )

    extreme_z = (
        df[
            [
                "Date",
                "USDTRY",
                "Return_1D",
                "Return_ZScore_20D",
            ]
        ]
        .assign(
            Absolute_Z=lambda x:
                x[
                    "Return_ZScore_20D"
                ].abs()
        )
        .sort_values(
            "Absolute_Z",
            ascending=False,
        )
        .head(10)
    )

    print(
        extreme_z
        .to_string(
            index=False
        )
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "\n--- PREPARING MILESTONE 2 FEATURES ---"
    )

    print(
        f"Input: {INPUT_FILE}"
    )

    df = load_m1_features()

    print(
        f"Loaded M1 observations: "
        f"{len(df):,}"
    )

    df = add_anomaly_features(
        df
    )

    df = validate_features(
        df
    )

    print_summary(
        df
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_columns = [
        "Date",
        "USDTRY",
        *ALL_FEATURES,
    ]

    df[
        output_columns
    ].to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        "\n--- OUTPUT ---"
    )

    print(
        f"Saved: {OUTPUT_FILE}"
    )

    print(
        "\nAnomaly feature preparation complete."
    )


if __name__ == "__main__":
    main()