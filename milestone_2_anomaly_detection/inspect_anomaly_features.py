from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler


# ============================================================
# Paths
# ============================================================

M2_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    M2_DIR
    / "data"
    / "usdtry_anomaly_features.csv"
)

OUTPUT_DIR = (
    M2_DIR
    / "outputs"
    / "feature_diagnostics"
)

CORRELATION_FILE = (
    OUTPUT_DIR
    / "feature_correlations.csv"
)

HIGH_CORRELATION_FILE = (
    OUTPUT_DIR
    / "high_correlation_pairs.csv"
)

DISTRIBUTION_FILE = (
    OUTPUT_DIR
    / "feature_distribution_summary.csv"
)

SCALING_FILE = (
    OUTPUT_DIR
    / "scaling_comparison.csv"
)

CORRELATION_FIGURE = (
    OUTPUT_DIR
    / "01_anomaly_feature_correlation.png"
)


# ============================================================
# Candidate features
# ============================================================

FEATURES = [
    "Return_1D",
    "Return_5D",
    "Volatility_5D",
    "Volatility_20D",
    "Volatility_60D",
    "MA_Distance_20D",
    "MA_Slope_20D",
    "Drawdown_60D",
    "Absolute_Return_1D",
    "Return_ZScore_20D",
    "Volatility_Ratio_5D_60D",
    "Drawdown_Change_5D",
]


HIGH_CORRELATION_THRESHOLD = 0.80


# ============================================================
# Load
# ============================================================

def load_data() -> pd.DataFrame:

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Feature dataset not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["Date"],
    )

    missing = (
        set(FEATURES)
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            f"Missing features: {sorted(missing)}"
        )

    values = (
        df[FEATURES]
        .to_numpy(dtype=float)
    )

    if not np.isfinite(values).all():

        raise ValueError(
            "Dataset contains NaN or infinite values."
        )

    return df


# ============================================================
# Correlation analysis
# ============================================================

def correlation_analysis(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    correlation = (
        df[FEATURES]
        .corr()
    )

    pairs = []

    for i in range(len(FEATURES)):

        for j in range(i + 1, len(FEATURES)):

            feature_a = FEATURES[i]
            feature_b = FEATURES[j]

            r = correlation.loc[
                feature_a,
                feature_b,
            ]

            if abs(r) >= HIGH_CORRELATION_THRESHOLD:

                pairs.append(
                    {
                        "Feature_A": feature_a,
                        "Feature_B": feature_b,
                        "Correlation": r,
                        "Absolute_Correlation": abs(r),
                    }
                )

    high_corr = pd.DataFrame(pairs)

    if not high_corr.empty:

        high_corr = (
            high_corr
            .sort_values(
                "Absolute_Correlation",
                ascending=False,
            )
            .reset_index(drop=True)
        )

    return correlation, high_corr


# ============================================================
# Distribution diagnostics
# ============================================================

def distribution_analysis(
    df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for feature in FEATURES:

        x = df[feature]

        rows.append(
            {
                "Feature": feature,
                "Mean": x.mean(),
                "Std": x.std(),
                "Skewness": x.skew(),
                "Kurtosis": x.kurt(),
                "Min": x.min(),
                "P01": x.quantile(0.01),
                "P05": x.quantile(0.05),
                "Median": x.median(),
                "P95": x.quantile(0.95),
                "P99": x.quantile(0.99),
                "Max": x.max(),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Scaling diagnostics
# ============================================================

def scaling_analysis(
    df: pd.DataFrame,
) -> pd.DataFrame:

    X = (
        df[FEATURES]
        .to_numpy(dtype=float)
    )

    standard_scaler = StandardScaler()

    robust_scaler = RobustScaler()

    X_standard = (
        standard_scaler
        .fit_transform(X)
    )

    X_robust = (
        robust_scaler
        .fit_transform(X)
    )

    rows = []

    for i, feature in enumerate(FEATURES):

        standard_abs = np.abs(
            X_standard[:, i]
        )

        robust_abs = np.abs(
            X_robust[:, i]
        )

        rows.append(
            {
                "Feature": feature,

                "Standard_MaxAbs":
                    standard_abs.max(),

                "Standard_P99Abs":
                    np.quantile(
                        standard_abs,
                        0.99,
                    ),

                "Robust_MaxAbs":
                    robust_abs.max(),

                "Robust_P99Abs":
                    np.quantile(
                        robust_abs,
                        0.99,
                    ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Correlation figure
# ============================================================

def plot_correlation_matrix(
    correlation: pd.DataFrame,
) -> None:

    fig, ax = plt.subplots(
        figsize=(12, 10)
    )

    image = ax.imshow(
        correlation.values,
        vmin=-1,
        vmax=1,
        aspect="auto",
    )

    ax.set_xticks(
        range(len(FEATURES))
    )

    ax.set_xticklabels(
        FEATURES,
        rotation=90,
    )

    ax.set_yticks(
        range(len(FEATURES))
    )

    ax.set_yticklabels(
        FEATURES
    )

    ax.set_title(
        "Milestone 2 Candidate Feature Correlations"
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Pearson correlation",
    )

    fig.tight_layout()

    fig.savefig(
        CORRELATION_FIGURE,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():

    print(
        "\n--- MILESTONE 2 FEATURE DIAGNOSTICS ---"
    )

    df = load_data()

    print(
        f"Observations: {len(df):,}"
    )

    print(
        f"Features: {len(FEATURES)}"
    )

    print(
        f"Period: "
        f"{df['Date'].min().date()} "
        f"to "
        f"{df['Date'].max().date()}"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Correlations
    # --------------------------------------------------------

    (
        correlation,
        high_corr,
    ) = correlation_analysis(df)

    correlation.to_csv(
        CORRELATION_FILE
    )

    high_corr.to_csv(
        HIGH_CORRELATION_FILE,
        index=False,
    )

    print(
        "\n--- HIGH CORRELATION PAIRS ---"
    )

    print(
        f"Threshold: "
        f"|r| >= {HIGH_CORRELATION_THRESHOLD:.2f}"
    )

    if high_corr.empty:

        print(
            "No high-correlation pairs found."
        )

    else:

        print(
            high_corr[
                [
                    "Feature_A",
                    "Feature_B",
                    "Correlation",
                ]
            ]
            .round(4)
            .to_string(index=False)
        )

    # --------------------------------------------------------
    # Distribution analysis
    # --------------------------------------------------------

    distributions = (
        distribution_analysis(df)
    )

    distributions.to_csv(
        DISTRIBUTION_FILE,
        index=False,
    )

    print(
        "\n--- DISTRIBUTION DIAGNOSTICS ---"
    )

    print(
        distributions[
            [
                "Feature",
                "Skewness",
                "Kurtosis",
                "P01",
                "Median",
                "P99",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Most skewed
    # --------------------------------------------------------

    most_skewed = (
        distributions
        .assign(
            Absolute_Skew=lambda x:
                x["Skewness"].abs()
        )
        .sort_values(
            "Absolute_Skew",
            ascending=False,
        )
    )

    print(
        "\n--- MOST SKEWED FEATURES ---"
    )

    print(
        most_skewed[
            [
                "Feature",
                "Skewness",
                "Kurtosis",
            ]
        ]
        .head(6)
        .round(4)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Scaling
    # --------------------------------------------------------

    scaling = (
        scaling_analysis(df)
    )

    scaling.to_csv(
        SCALING_FILE,
        index=False,
    )

    print(
        "\n--- STANDARD VS ROBUST SCALING ---"
    )

    print(
        scaling
        .round(3)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Correlation figure
    # --------------------------------------------------------

    plot_correlation_matrix(
        correlation
    )

    # --------------------------------------------------------
    # Preliminary diagnostics
    # --------------------------------------------------------

    print(
        "\n--- PRELIMINARY FLAGS ---"
    )

    flagged_skew = (
        distributions.loc[
            distributions[
                "Skewness"
            ].abs() >= 2.0,
            "Feature",
        ]
        .tolist()
    )

    if flagged_skew:

        print(
            "Highly skewed features:"
        )

        for feature in flagged_skew:
            print(
                f"  {feature}"
            )

    else:

        print(
            "No features exceed |skew| >= 2."
        )

    if not high_corr.empty:

        print(
            "\nRedundancy candidates:"
        )

        for _, row in (
            high_corr.iterrows()
        ):

            print(
                f"  {row['Feature_A']} "
                f"<-> "
                f"{row['Feature_B']} "
                f"(r={row['Correlation']:.3f})"
            )

    print(
        "\n--- OUTPUT FILES ---"
    )

    print(CORRELATION_FILE)
    print(HIGH_CORRELATION_FILE)
    print(DISTRIBUTION_FILE)
    print(SCALING_FILE)
    print(CORRELATION_FIGURE)

    print(
        "\nFeature diagnostics complete."
    )


if __name__ == "__main__":
    main()