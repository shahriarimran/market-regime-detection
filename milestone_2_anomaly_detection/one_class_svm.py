from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


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

IF_FILE = (
    M2_DIR
    / "outputs"
    / "isolation_forest"
    / "isolation_forest_scores.csv"
)

OUTPUT_DIR = (
    M2_DIR
    / "outputs"
    / "one_class_svm"
)

SCORES_FILE = (
    OUTPUT_DIR
    / "one_class_svm_scores.csv"
)

TOP_FILE = (
    OUTPUT_DIR
    / "top_one_class_svm_anomalies.csv"
)

COMPARISON_FILE = (
    OUTPUT_DIR
    / "three_model_comparison.csv"
)


# ============================================================
# Modeling features
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
# OCSVM configuration
# ============================================================

KERNEL = "rbf"

# gamma="scale" will be evaluated after our preprocessing.
GAMMA = "scale"


# ============================================================
# Load
# ============================================================

def load_data():

    for path in [
        FEATURE_FILE,
        BASELINE_FILE,
        IF_FILE,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    df = pd.read_csv(
        FEATURE_FILE,
        parse_dates=["Date"],
    )

    baseline = pd.read_csv(
        BASELINE_FILE,
        parse_dates=["Date"],
    )

    isolation = pd.read_csv(
        IF_FILE,
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

    # --------------------------------------------------------
    # Merge baseline
    # --------------------------------------------------------

    baseline_columns = [
        "Date",
        "Baseline_Anomaly",
        "Baseline_Anomaly_Score",
    ]

    df = df.merge(
        baseline[baseline_columns],
        on="Date",
        how="inner",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Merge Isolation Forest
    # --------------------------------------------------------

    if_columns = [
        "Date",
        "IF_Anomaly",
        "IF_Anomaly_Score",
    ]

    df = df.merge(
        isolation[if_columns],
        on="Date",
        how="inner",
        validate="one_to_one",
    )

    df = (
        df
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Parse possible serialized booleans
    # --------------------------------------------------------

    for column in [
        "Baseline_Anomaly",
        "IF_Anomaly",
    ]:

        if df[column].dtype != bool:

            df[column] = (
                df[column]
                .astype(str)
                .str.lower()
                .map(
                    {
                        "true": True,
                        "false": False,
                    }
                )
            )

        if df[column].isna().any():

            raise ValueError(
                f"Could not parse {column}."
            )

    X = (
        df[FEATURES]
        .to_numpy(dtype=float)
    )

    if not np.isfinite(X).all():

        raise ValueError(
            "Feature matrix contains "
            "NaN or infinite values."
        )

    return df


# ============================================================
# Tail-compressing preprocessing
# ============================================================

def preprocess_features(X):

    # --------------------------------------------------------
    # Robust location and scale
    # --------------------------------------------------------

    median = np.median(
        X,
        axis=0,
    )

    q25 = np.quantile(
        X,
        0.25,
        axis=0,
    )

    q75 = np.quantile(
        X,
        0.75,
        axis=0,
    )

    iqr = (
        q75 - q25
    )

    # Avoid division by zero.
    iqr = np.where(
        iqr > 1e-12,
        iqr,
        1.0,
    )

    robust_position = (
        (X - median)
        / iqr
    )

    # --------------------------------------------------------
    # Tail compression
    #
    # asinh(x):
    # approximately x near zero
    # approximately log(2|x|) for large |x|
    # --------------------------------------------------------

    X_compressed = np.arcsinh(
        robust_position
    )

    # --------------------------------------------------------
    # Final standardization
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_scaled = (
        scaler.fit_transform(
            X_compressed
        )
    )

    return (
        X_scaled,
        median,
        iqr,
        scaler,
    )


# ============================================================
# Fit
# ============================================================

def fit_one_class_svm(df):

    X = (
        df[FEATURES]
        .to_numpy(dtype=float)
    )

    (
        X_scaled,
        median,
        iqr,
        scaler,
    ) = preprocess_features(X)

    # --------------------------------------------------------
    # Use baseline anomaly rate as initial nu.
    #
    # nu does NOT guarantee exactly this anomaly percentage.
    # It controls the OCSVM solution itself.
    # --------------------------------------------------------

    baseline_rate = float(
        df["Baseline_Anomaly"].mean()
    )

    print(
        "\n--- ONE-CLASS SVM CONFIGURATION ---"
    )

    print(
        f"Features: {len(FEATURES)}"
    )

    print(
        f"Kernel: {KERNEL}"
    )

    print(
        f"Gamma: {GAMMA}"
    )

    print(
        f"Nu: {baseline_rate:.4%}"
    )

    print(
        "Preprocessing: "
        "median/IQR -> asinh -> StandardScaler"
    )

    model = OneClassSVM(
        kernel=KERNEL,
        gamma=GAMMA,
        nu=baseline_rate,
    )

    model.fit(
        X_scaled
    )

    # --------------------------------------------------------
    # Native OCSVM output
    #
    # decision_function:
    # more negative = more anomalous
    # --------------------------------------------------------

    decision = (
        model.decision_function(
            X_scaled
        )
        .ravel()
    )

    native_prediction = (
        model.predict(
            X_scaled
        )
    )

    native_anomaly = (
        native_prediction == -1
    )

    df = df.copy()

    df["OCSVM_Decision"] = (
        decision
    )

    df["OCSVM_Anomaly_Score"] = (
        -decision
    )

    df["OCSVM_Native_Anomaly"] = (
        native_anomaly
    )

    # --------------------------------------------------------
    # Rate-matched classification
    #
    # For exploratory comparison only, also identify exactly
    # the same number of anomalies as the statistical baseline.
    #
    # Final thresholds will be selected using chronological
    # validation, not full-history rankings.
    # --------------------------------------------------------

    target_count = int(
        df["Baseline_Anomaly"].sum()
    )

    order = np.argsort(
        df[
            "OCSVM_Anomaly_Score"
        ].to_numpy()
    )[::-1]

    matched_flag = np.zeros(
        len(df),
        dtype=bool,
    )

    matched_flag[
        order[:target_count]
    ] = True

    df["OCSVM_Anomaly"] = (
        matched_flag
    )

    return (
        df,
        model,
        median,
        iqr,
        scaler,
    )


# ============================================================
# Pairwise agreement
# ============================================================

def agreement_metrics(a, b):

    a = (
        pd.Series(a)
        .astype(bool)
        .to_numpy()
    )

    b = (
        pd.Series(b)
        .astype(bool)
        .to_numpy()
    )

    both = (
        a & b
    )

    union = (
        a | b
    )

    intersection_count = int(
        both.sum()
    )

    union_count = int(
        union.sum()
    )

    jaccard = (
        intersection_count
        / union_count
        if union_count
        else np.nan
    )

    return {
        "Both": intersection_count,
        "A_Only": int(
            (a & ~b).sum()
        ),
        "B_Only": int(
            (~a & b).sum()
        ),
        "Jaccard": jaccard,
    }


# ============================================================
# Comparison
# ============================================================

def compare_models(df):

    baseline = (
        df["Baseline_Anomaly"]
    )

    isolation = (
        df["IF_Anomaly"]
    )

    ocsvm = (
        df["OCSVM_Anomaly"]
    )

    native_ocsvm = (
        df["OCSVM_Native_Anomaly"]
    )

    print(
        "\n--- ONE-CLASS SVM RESULTS ---"
    )

    print(
        f"Native OCSVM anomalies: "
        f"{native_ocsvm.sum():,} "
        f"({native_ocsvm.mean():.2%})"
    )

    print(
        f"Rate-matched OCSVM anomalies: "
        f"{ocsvm.sum():,} "
        f"({ocsvm.mean():.2%})"
    )

    comparisons = {
        "Baseline vs OCSVM":
            agreement_metrics(
                baseline,
                ocsvm,
            ),

        "Isolation Forest vs OCSVM":
            agreement_metrics(
                isolation,
                ocsvm,
            ),

        "Baseline vs Isolation Forest":
            agreement_metrics(
                baseline,
                isolation,
            ),
    }

    print(
        "\n--- PAIRWISE ANOMALY AGREEMENT ---"
    )

    for name, metrics in (
        comparisons.items()
    ):

        print(
            f"\n{name}"
        )

        print(
            f"  Both:    "
            f"{metrics['Both']}"
        )

        print(
            f"  A only:  "
            f"{metrics['A_Only']}"
        )

        print(
            f"  B only:  "
            f"{metrics['B_Only']}"
        )

        print(
            f"  Jaccard: "
            f"{metrics['Jaccard']:.4f}"
        )

    # --------------------------------------------------------
    # Score-rank relationships
    # --------------------------------------------------------

    score_frame = pd.DataFrame(
        {
            "Baseline":
                df[
                    "Baseline_Anomaly_Score"
                ],

            "IsolationForest":
                df[
                    "IF_Anomaly_Score"
                ],

            "OneClassSVM":
                df[
                    "OCSVM_Anomaly_Score"
                ],
        }
    )

    rank_correlation = (
        score_frame
        .rank()
        .corr()
    )

    print(
        "\n--- ANOMALY SCORE RANK CORRELATION ---"
    )

    print(
        rank_correlation
        .round(4)
        .to_string()
    )

    return comparisons


# ============================================================
# Three-way classification
# ============================================================

def assign_three_way_class(df):

    b = (
        df["Baseline_Anomaly"]
        .astype(bool)
    )

    i = (
        df["IF_Anomaly"]
        .astype(bool)
    )

    o = (
        df["OCSVM_Anomaly"]
        .astype(bool)
    )

    count = (
        b.astype(int)
        + i.astype(int)
        + o.astype(int)
    )

    df = df.copy()

    df["Models_Flagging"] = (
        count
    )

    conditions = [
        count == 3,
        count == 2,
        count == 1,
    ]

    labels = [
        "ALL_THREE",
        "TWO_MODELS",
        "ONE_MODEL",
    ]

    df["Consensus_Class"] = (
        np.select(
            conditions,
            labels,
            default="NONE",
        )
    )

    print(
        "\n--- THREE-MODEL CONSENSUS ---"
    )

    print(
        df[
            "Consensus_Class"
        ]
        .value_counts()
        .to_string()
    )

    return df


# ============================================================
# Top OCSVM anomalies
# ============================================================

def print_top(df):

    columns = [
        "Date",
        "USDTRY",
        "Return_1D",
        "Return_5D",
        "Volatility_20D",
        "Return_ZScore_20D",
        "Volatility_Ratio_5D_60D",
        "Drawdown_Change_5D",
        "OCSVM_Anomaly_Score",
        "IF_Anomaly_Score",
        "Baseline_Anomaly_Score",
        "Models_Flagging",
        "Consensus_Class",
    ]

    top = (
        df
        .sort_values(
            "OCSVM_Anomaly_Score",
            ascending=False,
        )
        .head(25)
    )

    printable = (
        top[columns]
        .copy()
    )

    numeric = (
        printable
        .select_dtypes(
            include=[np.number]
        )
        .columns
    )

    printable[
        numeric
    ] = (
        printable[numeric]
        .round(4)
    )

    print(
        "\n--- TOP 25 ONE-CLASS SVM ANOMALIES ---"
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
        "\n--- MILESTONE 2: ONE-CLASS SVM ---"
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

    (
        df,
        model,
        median,
        iqr,
        scaler,
    ) = fit_one_class_svm(df)

    compare_models(df)

    df = assign_three_way_class(
        df
    )

    top = print_top(df)

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
        "IF_Anomaly",
        "OCSVM_Anomaly",
        "Baseline_Anomaly_Score",
        "IF_Anomaly_Score",
        "OCSVM_Anomaly_Score",
        "Models_Flagging",
        "Consensus_Class",
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
        "\nOne-Class SVM exploratory "
        "analysis complete."
    )


if __name__ == "__main__":
    main()