from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# ============================================================
# Paths
# ============================================================

M2_DIR = Path(__file__).resolve().parent

FEATURE_FILE = (
    M2_DIR
    / "data"
    / "usdtry_anomaly_features.csv"
)

BASELINE_FILE = (
    M2_DIR
    / "outputs"
    / "statistical_baseline"
    / "statistical_baseline_scores.csv"
)

OUTPUT_DIR = (
    M2_DIR
    / "outputs"
    / "isolation_forest"
)

SCORES_FILE = (
    OUTPUT_DIR
    / "isolation_forest_scores.csv"
)

TOP_FILE = (
    OUTPUT_DIR
    / "top_isolation_forest_anomalies.csv"
)

COMPARISON_FILE = (
    OUTPUT_DIR
    / "baseline_vs_isolation_forest.csv"
)

# ============================================================
# Modeling features
#
# Absolute_Return_1D is intentionally excluded because it is
# deterministically derived from Return_1D and would effectively
# double-weight one-day return magnitude.
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
    "Return_ZScore_20D",
    "Volatility_Ratio_5D_60D",
    "Drawdown_Change_5D",
]

# ============================================================
# Configuration
# ============================================================

RANDOM_STATE = 42

N_ESTIMATORS = 500

MAX_SAMPLES = 512

# ============================================================
# Load
# ============================================================

def load_data():

    if not FEATURE_FILE.exists():

        raise FileNotFoundError(
            f"Feature file not found:\n"
            f"{FEATURE_FILE}"
        )

    if not BASELINE_FILE.exists():

        raise FileNotFoundError(
            f"Statistical baseline not found:\n"
            f"{BASELINE_FILE}"
        )

    df = pd.read_csv(
        FEATURE_FILE,
        parse_dates=["Date"],
    )

    baseline = pd.read_csv(
        BASELINE_FILE,
        parse_dates=["Date"],
    )

    missing = (
        set(FEATURES)
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            f"Missing model features: "
            f"{sorted(missing)}"
        )

    # --------------------------------------------------------
    # Align baseline output by Date
    # --------------------------------------------------------

    baseline_columns = [
        "Date",
        "Baseline_Anomaly",
        "Baseline_Anomaly_Score",
        "Rules_Breached",
        "Primary_Reason",
    ]

    df = df.merge(
        baseline[baseline_columns],
        on="Date",
        how="inner",
        validate="one_to_one",
    )

    df = (
        df
        .sort_values("Date")
        .reset_index(drop=True)
    )

    if df["Baseline_Anomaly"].dtype != bool:

        df["Baseline_Anomaly"] = (
            df["Baseline_Anomaly"]
            .astype(str)
            .str.lower()
            .map(
                {
                    "true": True,
                    "false": False,
                }
            )
        )

    if df["Baseline_Anomaly"].isna().any():

        raise ValueError(
            "Could not parse Baseline_Anomaly."
        )

    values = (
        df[FEATURES]
        .to_numpy(dtype=float)
    )

    if not np.isfinite(values).all():

        raise ValueError(
            "Model features contain "
            "NaN or infinite values."
        )

    return df

# ============================================================
# Fit Isolation Forest
# ============================================================

def fit_isolation_forest(df):

    X = (
        df[FEATURES]
        .to_numpy(dtype=float)
    )

    # --------------------------------------------------------
    # Match the anomaly rate of the transparent
    # statistical baseline.
    #
    # This is for fair exploratory comparison only.
    # The final contamination / threshold will later
    # be selected using chronological validation.
    # --------------------------------------------------------

    baseline_anomaly_rate = float(
        df["Baseline_Anomaly"].mean()
    )

    print(
        "\n--- ISOLATION FOREST CONFIGURATION ---"
    )

    print(
        f"Baseline anomaly rate: "
        f"{baseline_anomaly_rate:.4%}"
    )

    print(
        f"Features: {len(FEATURES)}"
    )

    print(
        f"Trees: {N_ESTIMATORS}"
    )

    print(
        f"Max samples per tree: "
        f"{min(MAX_SAMPLES, len(df))}"
    )

    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        max_samples=min(
            MAX_SAMPLES,
            len(df),
        ),
        contamination=baseline_anomaly_rate,
        max_features=1.0,
        bootstrap=False,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X)

    # --------------------------------------------------------
    # sklearn:
    #
    # score_samples:
    # lower values = more abnormal
    #
    # We reverse the sign so that:
    # larger value = more anomalous
    # --------------------------------------------------------

    raw_score = (
        model.score_samples(X)
    )

    anomaly_score = (
        -raw_score
    )

    prediction = (
        model.predict(X)
    )

    anomaly_flag = (
        prediction == -1
    )

    df = df.copy()

    df["IF_Raw_Score"] = (
        raw_score
    )

    df["IF_Anomaly_Score"] = (
        anomaly_score
    )

    df["IF_Anomaly"] = (
        anomaly_flag
    )

    return df, model

# ============================================================
# Comparison with statistical baseline
# ============================================================

def compare_models(df):

    baseline = (
        df["Baseline_Anomaly"]
        .astype(bool)
    )

    isolation = (
        df["IF_Anomaly"]
        .astype(bool)
    )

    both = (
        baseline
        & isolation
    )

    baseline_only = (
        baseline
        & ~isolation
    )

    isolation_only = (
        isolation
        & ~baseline
    )

    neither = (
        ~baseline
        & ~isolation
    )

    intersection = int(
        both.sum()
    )

    union = int(
        (
            baseline
            | isolation
        ).sum()
    )

    jaccard = (
        intersection / union
        if union > 0
        else np.nan
    )

    # --------------------------------------------------------
    # Rank correlation between anomaly scores
    #
    # We use rank correlation because the two scores
    # have completely different numeric scales.
    # --------------------------------------------------------

    baseline_rank = (
        df["Baseline_Anomaly_Score"]
        .rank()
    )

    isolation_rank = (
        df["IF_Anomaly_Score"]
        .rank()
    )

    score_rank_correlation = (
        baseline_rank
        .corr(isolation_rank)
    )

    print(
        "\n--- BASELINE VS ISOLATION FOREST ---"
    )

    print(
        f"Baseline anomalies: "
        f"{baseline.sum():,} "
        f"({baseline.mean():.2%})"
    )

    print(
        f"Isolation Forest anomalies: "
        f"{isolation.sum():,} "
        f"({isolation.mean():.2%})"
    )

    print(
        f"\nBoth models flag: "
        f"{both.sum():,}"
    )

    print(
        f"Baseline only: "
        f"{baseline_only.sum():,}"
    )

    print(
        f"Isolation Forest only: "
        f"{isolation_only.sum():,}"
    )

    print(
        f"Neither: "
        f"{neither.sum():,}"
    )

    print(
        f"\nJaccard agreement: "
        f"{jaccard:.4f}"
    )

    print(
        f"Anomaly-score rank correlation: "
        f"{score_rank_correlation:.4f}"
    )

    df = df.copy()

    df["Comparison_Class"] = np.select(
        [
            both,
            baseline_only,
            isolation_only,
        ],
        [
            "BOTH",
            "BASELINE_ONLY",
            "ISOLATION_FOREST_ONLY",
        ],
        default="NEITHER",
    )

    return df

# ============================================================
# Top anomalies
# ============================================================

def print_top_anomalies(df):

    columns = [
        "Date",
        "USDTRY",
        "Return_1D",
        "Return_5D",
        "Volatility_20D",
        "Return_ZScore_20D",
        "Volatility_Ratio_5D_60D",
        "Drawdown_Change_5D",
        "IF_Anomaly_Score",
        "Baseline_Anomaly_Score",
        "Rules_Breached",
        "Primary_Reason",
        "Comparison_Class",
    ]

    top = (
        df
        .sort_values(
            "IF_Anomaly_Score",
            ascending=False,
        )
        .head(25)
    )

    printable = (
        top[columns]
        .copy()
    )

    numeric_columns = (
        printable
        .select_dtypes(
            include=[np.number]
        )
        .columns
    )

    printable[
        numeric_columns
    ] = (
        printable[
            numeric_columns
        ]
        .round(4)
    )

    print(
        "\n--- TOP 25 ISOLATION FOREST ANOMALIES ---"
    )

    print(
        printable
        .to_string(index=False)
    )

    return top

# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\n--- MILESTONE 2: ISOLATION FOREST ---"
    )

    df = load_data()

    print(
        f"Observations: {len(df):,}"
    )

    print(
        f"Period: "
        f"{df['Date'].min().date()} "
        f"to "
        f"{df['Date'].max().date()}"
    )

    df, model = (
        fit_isolation_forest(df)
    )

    df = compare_models(df)

    top = print_top_anomalies(df)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    df.to_csv(
        SCORES_FILE,
        index=False,
    )

    top.to_csv(
        TOP_FILE,
        index=False,
    )

    comparison_columns = [
        "Date",
        "USDTRY",
        "Baseline_Anomaly",
        "Baseline_Anomaly_Score",
        "IF_Anomaly",
        "IF_Anomaly_Score",
        "Comparison_Class",
    ]

    df[
        comparison_columns
    ].to_csv(
        COMPARISON_FILE,
        index=False,
    )

    print(
        "\n--- OUTPUT ---"
    )

    print(SCORES_FILE)
    print(TOP_FILE)
    print(COMPARISON_FILE)

    print(
        "\nIsolation Forest exploratory "
        "analysis complete."
    )

if __name__ == "__main__":
    main()