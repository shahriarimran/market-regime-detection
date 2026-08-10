from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


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
    / "walk_forward_validation"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "walk_forward_summary.csv"
)

PREDICTIONS_FILE = (
    OUTPUT_DIR
    / "walk_forward_predictions.csv"
)


# ============================================================
# Walk-forward years
# ============================================================

TEST_YEARS = [
    2021,
    2022,
    2023,
    2024,
    2025,
    2026,
]


# ============================================================
# Features
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
# Baseline thresholds
# ============================================================

RETURN_Z_THRESHOLD = 4.0
VOL_RATIO_THRESHOLD = 2.0
DRAWDOWN_CHANGE_THRESHOLD = 0.05
ABS_RETURN_THRESHOLD = 0.04


# ============================================================
# ML configuration
# ============================================================

RANDOM_STATE = 42
N_ESTIMATORS = 500
MAX_SAMPLES = 512


# ============================================================
# Load
# ============================================================

def load_data():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found:\n"
            f"{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["Date"],
    )

    required = {
        "Date",
        "USDTRY",
        "Absolute_Return_1D",
        *FEATURES,
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing)}"
        )

    df = (
        df
        .sort_values("Date")
        .reset_index(drop=True)
    )

    values = (
        df[FEATURES]
        .to_numpy(dtype=float)
    )

    if not np.isfinite(values).all():

        raise ValueError(
            "Feature matrix contains "
            "NaN or infinite values."
        )

    return df


# ============================================================
# Statistical baseline
# ============================================================

def baseline_score_and_flag(df):

    score_return_z = (
        df["Return_ZScore_20D"].abs()
        / RETURN_Z_THRESHOLD
    )

    score_volatility = (
        df["Volatility_Ratio_5D_60D"]
        / VOL_RATIO_THRESHOLD
    )

    score_drawdown = (
        df["Drawdown_Change_5D"].abs()
        / DRAWDOWN_CHANGE_THRESHOLD
    )

    score_return = (
        df["Absolute_Return_1D"]
        / ABS_RETURN_THRESHOLD
    )

    matrix = np.column_stack(
        [
            score_return_z,
            score_volatility,
            score_drawdown,
            score_return,
        ]
    )

    score = (
        matrix.max(axis=1)
    )

    flag = (
        score >= 1.0
    )

    return (
        score.astype(float),
        flag.astype(bool),
    )


# ============================================================
# OCSVM preprocessing
# ============================================================

def fit_ocsvm_preprocessor(X_train):

    median = np.median(
        X_train,
        axis=0,
    )

    q25 = np.quantile(
        X_train,
        0.25,
        axis=0,
    )

    q75 = np.quantile(
        X_train,
        0.75,
        axis=0,
    )

    iqr = (
        q75 - q25
    )

    iqr = np.where(
        iqr > 1e-12,
        iqr,
        1.0,
    )

    robust = (
        (X_train - median)
        / iqr
    )

    compressed = np.arcsinh(
        robust
    )

    scaler = StandardScaler()

    X_scaled = (
        scaler.fit_transform(
            compressed
        )
    )

    return {
        "median": median,
        "iqr": iqr,
        "scaler": scaler,
        "X_train": X_scaled,
    }


def transform_ocsvm(
    X,
    prep,
):

    robust = (
        (X - prep["median"])
        / prep["iqr"]
    )

    compressed = np.arcsinh(
        robust
    )

    return (
        prep["scaler"]
        .transform(
            compressed
        )
    )


# ============================================================
# Fit models
# ============================================================

def fit_models(train):

    X_train = (
        train[FEATURES]
        .to_numpy(dtype=float)
    )

    (
        train_baseline_score,
        train_baseline_flag,
    ) = baseline_score_and_flag(
        train
    )

    target_rate = float(
        train_baseline_flag.mean()
    )

    # --------------------------------------------------------
    # Isolation Forest
    # --------------------------------------------------------

    isolation = IsolationForest(
        n_estimators=N_ESTIMATORS,
        max_samples=min(
            MAX_SAMPLES,
            len(train),
        ),
        contamination="auto",
        max_features=1.0,
        bootstrap=False,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    isolation.fit(
        X_train
    )

    if_train_score = (
        -isolation.score_samples(
            X_train
        )
    )

    if_threshold = float(
        np.quantile(
            if_train_score,
            1.0 - target_rate,
        )
    )

    # --------------------------------------------------------
    # One-Class SVM
    # --------------------------------------------------------

    prep = (
        fit_ocsvm_preprocessor(
            X_train
        )
    )

    nu = float(
        np.clip(
            target_rate,
            0.01,
            0.25,
        )
    )

    ocsvm = OneClassSVM(
        kernel="rbf",
        gamma="scale",
        nu=nu,
    )

    ocsvm.fit(
        prep["X_train"]
    )

    ocsvm_train_score = (
        -ocsvm
        .decision_function(
            prep["X_train"]
        )
        .ravel()
    )

    ocsvm_threshold = float(
        np.quantile(
            ocsvm_train_score,
            1.0 - target_rate,
        )
    )

    return {
        "target_rate":
            target_rate,

        "isolation":
            isolation,

        "if_threshold":
            if_threshold,

        "ocsvm":
            ocsvm,

        "ocsvm_prep":
            prep,

        "ocsvm_threshold":
            ocsvm_threshold,
    }


# ============================================================
# Score test fold
# ============================================================

def score_test(
    test,
    fitted,
):

    result = test.copy()

    X = (
        result[FEATURES]
        .to_numpy(dtype=float)
    )

    # Baseline
    (
        baseline_score,
        baseline_flag,
    ) = baseline_score_and_flag(
        result
    )

    result[
        "Baseline_Score"
    ] = baseline_score

    result[
        "Baseline_Anomaly"
    ] = baseline_flag

    # Isolation Forest
    if_score = (
        -fitted[
            "isolation"
        ]
        .score_samples(X)
    )

    result[
        "IF_Score"
    ] = if_score

    result[
        "IF_Anomaly"
    ] = (
        if_score
        >= fitted[
            "if_threshold"
        ]
    )

    # OCSVM
    X_ocsvm = (
        transform_ocsvm(
            X,
            fitted[
                "ocsvm_prep"
            ],
        )
    )

    ocsvm_score = (
        -fitted[
            "ocsvm"
        ]
        .decision_function(
            X_ocsvm
        )
        .ravel()
    )

    result[
        "OCSVM_Score"
    ] = ocsvm_score

    result[
        "OCSVM_Anomaly"
    ] = (
        ocsvm_score
        >= fitted[
            "ocsvm_threshold"
        ]
    )

    return result


# ============================================================
# Jaccard
# ============================================================

def jaccard(a, b):

    a = np.asarray(
        a,
        dtype=bool,
    )

    b = np.asarray(
        b,
        dtype=bool,
    )

    union = (
        a | b
    ).sum()

    if union == 0:

        return np.nan

    return float(
        (
            a & b
        ).sum()
        / union
    )


# ============================================================
# Episode statistics
# ============================================================

def episode_statistics(flags):

    flags = np.asarray(
        flags,
        dtype=bool,
    )

    lengths = []

    current = 0

    for flag in flags:

        if flag:

            current += 1

        elif current > 0:

            lengths.append(
                current
            )

            current = 0

    if current > 0:

        lengths.append(
            current
        )

    if not lengths:

        return {
            "Episodes": 0,
            "Median_Episode_Length": 0.0,
            "Max_Episode_Length": 0,
        }

    return {
        "Episodes":
            len(lengths),

        "Median_Episode_Length":
            float(
                np.median(lengths)
            ),

        "Max_Episode_Length":
            int(
                np.max(lengths)
            ),
    }


# ============================================================
# Run one fold
# ============================================================

def run_fold(
    df,
    test_year,
):

    test_start = pd.Timestamp(
        f"{test_year}-01-01"
    )

    test_end = pd.Timestamp(
        f"{test_year}-12-31"
    )

    train = (
        df[
            df["Date"]
            < test_start
        ]
        .copy()
        .reset_index(drop=True)
    )

    test = (
        df[
            (
                df["Date"]
                >= test_start
            )
            &
            (
                df["Date"]
                <= test_end
            )
        ]
        .copy()
        .reset_index(drop=True)
    )

    if len(train) < 500:

        raise RuntimeError(
            f"Insufficient training data "
            f"for {test_year}: "
            f"{len(train)}"
        )

    if len(test) == 0:

        raise RuntimeError(
            f"No test data for "
            f"{test_year}."
        )

    fitted = fit_models(
        train
    )

    scored = score_test(
        test,
        fitted,
    )

    target_rate = (
        fitted[
            "target_rate"
        ]
    )

    baseline_rate = float(
        scored[
            "Baseline_Anomaly"
        ].mean()
    )

    if_rate = float(
        scored[
            "IF_Anomaly"
        ].mean()
    )

    ocsvm_rate = float(
        scored[
            "OCSVM_Anomaly"
        ].mean()
    )

    if_episode = (
        episode_statistics(
            scored[
                "IF_Anomaly"
            ]
        )
    )

    oc_episode = (
        episode_statistics(
            scored[
                "OCSVM_Anomaly"
            ]
        )
    )

    baseline_episode = (
        episode_statistics(
            scored[
                "Baseline_Anomaly"
            ]
        )
    )

    summary = {
        "Test_Year":
            test_year,

        "Train_Start":
            train[
                "Date"
            ].min(),

        "Train_End":
            train[
                "Date"
            ].max(),

        "Train_N":
            len(train),

        "Test_N":
            len(test),

        "Training_Target_Rate":
            target_rate,

        "Baseline_Test_Rate":
            baseline_rate,

        "IF_Test_Rate":
            if_rate,

        "OCSVM_Test_Rate":
            ocsvm_rate,

        "IF_Target_Gap":
            abs(
                if_rate
                - target_rate
            ),

        "OCSVM_Target_Gap":
            abs(
                ocsvm_rate
                - target_rate
            ),

        "IF_vs_TestBaseline_Gap":
            abs(
                if_rate
                - baseline_rate
            ),

        "OCSVM_vs_TestBaseline_Gap":
            abs(
                ocsvm_rate
                - baseline_rate
            ),

        "Jaccard_Baseline_IF":
            jaccard(
                scored[
                    "Baseline_Anomaly"
                ],
                scored[
                    "IF_Anomaly"
                ],
            ),

        "Jaccard_Baseline_OCSVM":
            jaccard(
                scored[
                    "Baseline_Anomaly"
                ],
                scored[
                    "OCSVM_Anomaly"
                ],
            ),

        "Jaccard_IF_OCSVM":
            jaccard(
                scored[
                    "IF_Anomaly"
                ],
                scored[
                    "OCSVM_Anomaly"
                ],
            ),

        "IF_Threshold":
            fitted[
                "if_threshold"
            ],

        "OCSVM_Threshold":
            fitted[
                "ocsvm_threshold"
            ],

        "Baseline_Episodes":
            baseline_episode[
                "Episodes"
            ],

        "Baseline_Median_Episode":
            baseline_episode[
                "Median_Episode_Length"
            ],

        "Baseline_Max_Episode":
            baseline_episode[
                "Max_Episode_Length"
            ],

        "IF_Episodes":
            if_episode[
                "Episodes"
            ],

        "IF_Median_Episode":
            if_episode[
                "Median_Episode_Length"
            ],

        "IF_Max_Episode":
            if_episode[
                "Max_Episode_Length"
            ],

        "OCSVM_Episodes":
            oc_episode[
                "Episodes"
            ],

        "OCSVM_Median_Episode":
            oc_episode[
                "Median_Episode_Length"
            ],

        "OCSVM_Max_Episode":
            oc_episode[
                "Max_Episode_Length"
            ],
    }

    scored[
        "Test_Year"
    ] = test_year

    scored[
        "Training_Target_Rate"
    ] = target_rate

    return (
        summary,
        scored,
    )


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\n--- MILESTONE 2: "
        "WALK-FORWARD VALIDATION ---"
    )

    df = load_data()

    summaries = []
    predictions = []

    for year in TEST_YEARS:

        print(
            f"\n--- FOLD: TEST {year} ---"
        )

        (
            summary,
            scored,
        ) = run_fold(
            df,
            year,
        )

        summaries.append(
            summary
        )

        predictions.append(
            scored
        )

        print(
            f"Train: "
            f"{summary['Train_Start'].date()} "
            f"to "
            f"{summary['Train_End'].date()} "
            f"({summary['Train_N']:,})"
        )

        print(
            f"Test observations: "
            f"{summary['Test_N']:,}"
        )

        print(
            f"Training target rate: "
            f"{summary['Training_Target_Rate']:.2%}"
        )

        print(
            f"Baseline test rate: "
            f"{summary['Baseline_Test_Rate']:.2%}"
        )

        print(
            f"IF test rate: "
            f"{summary['IF_Test_Rate']:.2%}"
        )

        print(
            f"OCSVM test rate: "
            f"{summary['OCSVM_Test_Rate']:.2%}"
        )

        print(
            f"IF target gap: "
            f"{summary['IF_Target_Gap']:.2%}"
        )

        print(
            f"OCSVM target gap: "
            f"{summary['OCSVM_Target_Gap']:.2%}"
        )

    summary_df = pd.DataFrame(
        summaries
    )

    predictions_df = pd.concat(
        predictions,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Full summary
    # --------------------------------------------------------

    print(
        "\n--- WALK-FORWARD SUMMARY ---"
    )

    printable_columns = [
        "Test_Year",
        "Train_N",
        "Test_N",
        "Training_Target_Rate",
        "Baseline_Test_Rate",
        "IF_Test_Rate",
        "OCSVM_Test_Rate",
        "IF_Target_Gap",
        "OCSVM_Target_Gap",
        "Jaccard_Baseline_IF",
        "Jaccard_Baseline_OCSVM",
        "Jaccard_IF_OCSVM",
    ]

    printable = (
        summary_df[
            printable_columns
        ]
        .copy()
    )

    rate_columns = [
        "Training_Target_Rate",
        "Baseline_Test_Rate",
        "IF_Test_Rate",
        "OCSVM_Test_Rate",
        "IF_Target_Gap",
        "OCSVM_Target_Gap",
    ]

    for column in rate_columns:

        printable[column] *= 100

    print(
        printable
        .round(3)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    print(
        "\n--- AGGREGATE WALK-FORWARD METRICS ---"
    )

    print(
        f"Mean baseline test rate: "
        f"{summary_df['Baseline_Test_Rate'].mean():.2%}"
    )

    print(
        f"Mean IF test rate: "
        f"{summary_df['IF_Test_Rate'].mean():.2%}"
    )

    print(
        f"Mean OCSVM test rate: "
        f"{summary_df['OCSVM_Test_Rate'].mean():.2%}"
    )

    print(
        f"\nMean IF target gap: "
        f"{summary_df['IF_Target_Gap'].mean():.2%}"
    )

    print(
        f"Mean OCSVM target gap: "
        f"{summary_df['OCSVM_Target_Gap'].mean():.2%}"
    )

    print(
        f"\nMedian IF target gap: "
        f"{summary_df['IF_Target_Gap'].median():.2%}"
    )

    print(
        f"Median OCSVM target gap: "
        f"{summary_df['OCSVM_Target_Gap'].median():.2%}"
    )

    print(
        f"\nMean baseline-IF Jaccard: "
        f"{summary_df['Jaccard_Baseline_IF'].mean():.4f}"
    )

    print(
        f"Mean baseline-OCSVM Jaccard: "
        f"{summary_df['Jaccard_Baseline_OCSVM'].mean():.4f}"
    )

    print(
        f"Mean IF-OCSVM Jaccard: "
        f"{summary_df['Jaccard_IF_OCSVM'].mean():.4f}"
    )

    # --------------------------------------------------------
    # Annual episode behavior
    # --------------------------------------------------------

    print(
        "\n--- EPISODE SUMMARY ---"
    )

    episode_columns = [
        "Test_Year",
        "Baseline_Episodes",
        "Baseline_Median_Episode",
        "Baseline_Max_Episode",
        "IF_Episodes",
        "IF_Median_Episode",
        "IF_Max_Episode",
        "OCSVM_Episodes",
        "OCSVM_Median_Episode",
        "OCSVM_Max_Episode",
    ]

    print(
        summary_df[
            episode_columns
        ]
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    predictions_df.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    print(
        "\n--- OUTPUT ---"
    )

    print(SUMMARY_FILE)
    print(PREDICTIONS_FILE)

    print(
        "\nWalk-forward validation complete."
    )


if __name__ == "__main__":
    main()