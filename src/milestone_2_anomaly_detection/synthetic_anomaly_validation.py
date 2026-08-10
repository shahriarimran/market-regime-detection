from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from .prepare_anomaly_features import (
    M1_FEATURES,
    VOLATILITY_FLOOR_QUANTILE,
    add_anomaly_features,
    estimate_volatility_floor,
    validate_features,
)


# ============================================================
# Paths
# ============================================================

M2_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = M2_DIR.parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "usdtry_features.csv"
)

OUTPUT_DIR = (
    M2_DIR
    / "outputs"
    / "synthetic_validation"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "synthetic_validation_summary.csv"
)

DETAIL_FILE = (
    OUTPUT_DIR
    / "synthetic_validation_details.csv"
)


# ============================================================
# Chronological split
# ============================================================

TRAIN_END = pd.Timestamp("2022-12-30")


# ============================================================
# Features
# ============================================================

MODEL_FEATURES = [
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
# Statistical baseline thresholds
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

RNG = np.random.default_rng(
    RANDOM_STATE
)


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
        *M1_FEATURES,
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

    return df


# ============================================================
# Statistical baseline
# ============================================================

def baseline_flag(df):

    return (
        (
            df["Return_ZScore_20D"].abs()
            >= RETURN_Z_THRESHOLD
        )
        |
        (
            df["Volatility_Ratio_5D_60D"]
            >= VOL_RATIO_THRESHOLD
        )
        |
        (
            df["Drawdown_Change_5D"].abs()
            >= DRAWDOWN_CHANGE_THRESHOLD
        )
        |
        (
            df["Absolute_Return_1D"]
            >= ABS_RETURN_THRESHOLD
        )
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

    iqr = q75 - q25

    iqr = np.where(
        iqr > 1e-12,
        iqr,
        1.0,
    )

    robust_train = (
        (X_train - median)
        / iqr
    )

    compressed_train = np.arcsinh(
        robust_train
    )

    scaler = StandardScaler()

    X_train_scaled = (
        scaler.fit_transform(
            compressed_train
        )
    )

    return (
        X_train_scaled,
        median,
        iqr,
        scaler,
    )


def transform_ocsvm(
    X,
    median,
    iqr,
    scaler,
):

    robust = (
        (X - median)
        / iqr
    )

    compressed = np.arcsinh(
        robust
    )

    return scaler.transform(
        compressed
    )


# ============================================================
# Fit models using TRAINING DATA ONLY
# ============================================================

def fit_models(train):

    X_train = (
        train[MODEL_FEATURES]
        .to_numpy(dtype=float)
    )

    train_baseline = (
        baseline_flag(train)
    )

    anomaly_rate = float(
        train_baseline.mean()
    )

    print(
        "\n--- TRAINING CONFIGURATION ---"
    )

    print(
        f"Training observations: "
        f"{len(train):,}"
    )

    print(
        f"Training baseline anomaly rate: "
        f"{anomaly_rate:.2%}"
    )

    # --------------------------------------------------------
    # Isolation Forest
    #
    # Model fitting and threshold calibration are separated.
    # --------------------------------------------------------

    isolation = IsolationForest(
        n_estimators=N_ESTIMATORS,
        max_samples=min(
            MAX_SAMPLES,
            len(train),
        ),
        contamination="auto",
        max_features=1.0,
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
            1.0 - anomaly_rate,
        )
    )

    # --------------------------------------------------------
    # OCSVM
    # --------------------------------------------------------

    (
        X_train_scaled,
        median,
        iqr,
        scaler,
    ) = fit_ocsvm_preprocessor(
        X_train
    )

    ocsvm = OneClassSVM(
        kernel="rbf",
        gamma="scale",
        nu=anomaly_rate,
    )

    ocsvm.fit(
        X_train_scaled
    )

    ocsvm_train_score = (
        -ocsvm.decision_function(
            X_train_scaled
        ).ravel()
    )

    ocsvm_threshold = float(
        np.quantile(
            ocsvm_train_score,
            1.0 - anomaly_rate,
        )
    )

    return {
        "isolation": isolation,
        "if_threshold": if_threshold,

        "ocsvm": ocsvm,
        "ocsvm_threshold":
            ocsvm_threshold,

        "median": median,
        "iqr": iqr,
        "scaler": scaler,

        "anomaly_rate":
            anomaly_rate,
    }


# ============================================================
# Score arbitrary observations
# ============================================================

def score_models(df, fitted):

    X = (
        df[MODEL_FEATURES]
        .to_numpy(dtype=float)
    )

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    baseline = (
        baseline_flag(df)
        .to_numpy(dtype=bool)
    )

    # --------------------------------------------------------
    # Isolation Forest
    # --------------------------------------------------------

    if_score = (
        -fitted[
            "isolation"
        ].score_samples(X)
    )

    if_flag = (
        if_score
        >= fitted[
            "if_threshold"
        ]
    )

    # --------------------------------------------------------
    # OCSVM
    # --------------------------------------------------------

    X_scaled = transform_ocsvm(
        X,
        fitted["median"],
        fitted["iqr"],
        fitted["scaler"],
    )

    ocsvm_score = (
        -fitted[
            "ocsvm"
        ]
        .decision_function(
            X_scaled
        )
        .ravel()
    )

    ocsvm_flag = (
        ocsvm_score
        >= fitted[
            "ocsvm_threshold"
        ]
    )

    return {
        "Baseline_Flag":
            baseline,

        "IF_Flag":
            if_flag,

        "OCSVM_Flag":
            ocsvm_flag,

        "IF_Score":
            if_score,

        "OCSVM_Score":
            ocsvm_score,
    }


# ============================================================
# Training robust scales
# ============================================================

def feature_iqr(train):

    q25 = (
        train[MODEL_FEATURES]
        .quantile(0.25)
    )

    q75 = (
        train[MODEL_FEATURES]
        .quantile(0.75)
    )

    scale = (
        q75 - q25
    )

    scale = scale.replace(
        0,
        np.nan,
    )

    return scale.fillna(
        1.0
    )


# ============================================================
# Synthetic anomaly injection
# ============================================================

def inject_return_shock(
    df,
    scale,
    severity,
):

    out = df.copy()

    signs = RNG.choice(
        [-1.0, 1.0],
        size=len(out),
    )

    out["Return_1D"] += (
        signs
        * severity
        * 3.0
        * scale["Return_1D"]
    )

    out["Return_5D"] += (
        signs
        * severity
        * 2.0
        * scale["Return_5D"]
    )

    out["Return_ZScore_20D"] += (
        signs
        * severity
        * 3.0
    )

    # Keep interpretation field coherent.
    out["Absolute_Return_1D"] = (
        out["Return_1D"].abs()
    )

    return out


def inject_volatility_spike(
    df,
    severity,
):

    out = df.copy()

    factors = {
        1: (1.5, 1.3, 1.1),
        2: (2.5, 1.8, 1.3),
        3: (4.0, 2.5, 1.6),
    }

    f5, f20, f60 = (
        factors[severity]
    )

    out["Volatility_5D"] *= f5
    out["Volatility_20D"] *= f20
    out["Volatility_60D"] *= f60

    out[
        "Volatility_Ratio_5D_60D"
    ] = (
        out["Volatility_5D"]
        / out["Volatility_60D"]
    )

    return out


def inject_drawdown_break(
    df,
    scale,
    severity,
):

    out = df.copy()

    # Downward/correction-style shock.
    out["Drawdown_60D"] -= (
        severity
        * 3.0
        * abs(
            scale[
                "Drawdown_60D"
            ]
        )
    )

    out["Drawdown_Change_5D"] -= (
        severity
        * 3.0
        * abs(
            scale[
                "Drawdown_Change_5D"
            ]
        )
    )

    out["MA_Distance_20D"] -= (
        severity
        * 2.0
        * abs(
            scale[
                "MA_Distance_20D"
            ]
        )
    )

    return out


def inject_compound_stress(
    df,
    scale,
    severity,
):

    out = inject_return_shock(
        df,
        scale,
        severity,
    )

    out = inject_volatility_spike(
        out,
        severity,
    )

    out = inject_drawdown_break(
        out,
        scale,
        severity,
    )

    return out


# ============================================================
# Synthetic validation
# ============================================================

def run_synthetic_validation(
    reference_pool,
    train,
    fitted,
):

    scale = feature_iqr(
        train
    )

    injection_functions = {
        "RETURN_SHOCK":
            lambda x, s:
                inject_return_shock(
                    x,
                    scale,
                    s,
                ),

        "VOLATILITY_SPIKE":
            lambda x, s:
                inject_volatility_spike(
                    x,
                    s,
                ),

        "DRAWDOWN_BREAK":
            lambda x, s:
                inject_drawdown_break(
                    x,
                    scale,
                    s,
                ),

        "COMPOUND_STRESS":
            lambda x, s:
                inject_compound_stress(
                    x,
                    scale,
                    s,
                ),
    }

    summaries = []
    details = []

    for anomaly_type, function in (
        injection_functions.items()
    ):

        for severity in [
            1,
            2,
            3,
        ]:

            synthetic = function(
                reference_pool,
                severity,
            )

            scores = score_models(
                synthetic,
                fitted,
            )

            baseline_flagged = (
                scores[
                    "Baseline_Flag"
                ]
            )

            if_flagged = (
                scores[
                    "IF_Flag"
                ]
            )

            ocsvm_flagged = (
                scores[
                    "OCSVM_Flag"
                ]
            )

            any_ml = (
                if_flagged
                | ocsvm_flagged
            )

            all_three = (
                baseline_flagged
                & if_flagged
                & ocsvm_flagged
            )

            summaries.append(
                {
                    "Anomaly_Type":
                        anomaly_type,

                    "Severity":
                        severity,

                    "N_Injected":
                        len(synthetic),

                    "Baseline_Recall":
                        baseline_flagged.mean(),

                    "IF_Recall":
                        if_flagged.mean(),

                    "OCSVM_Recall":
                        ocsvm_flagged.mean(),

                    "Any_ML_Recall":
                        any_ml.mean(),

                    "All_Three_Recall":
                        all_three.mean(),
                }
            )

            detail = pd.DataFrame(
                {
                    "Date":
                        synthetic[
                            "Date"
                        ].to_numpy(),

                    "Anomaly_Type":
                        anomaly_type,

                    "Severity":
                        severity,

                    "Baseline_Flag":
                        baseline_flagged,

                    "IF_Flag":
                        if_flagged,

                    "OCSVM_Flag":
                        ocsvm_flagged,

                    "IF_Score":
                        scores[
                            "IF_Score"
                        ],

                    "OCSVM_Score":
                        scores[
                            "OCSVM_Score"
                        ],
                }
            )

            details.append(
                detail
            )

    return (
        pd.DataFrame(
            summaries
        ),
        pd.concat(
            details,
            ignore_index=True,
        ),
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
        "SYNTHETIC ANOMALY VALIDATION ---"
    )

    df = load_data()

    raw_train = (
        df[df["Date"] <= TRAIN_END]
        .copy()
        .reset_index(drop=True)
    )

    volatility_floor = estimate_volatility_floor(
        raw_train,
        quantile=VOLATILITY_FLOOR_QUANTILE,
    )

    featured = validate_features(
        add_anomaly_features(
            df,
            volatility_floor=volatility_floor,
        )
    )

    train = (
        featured[featured["Date"] <= TRAIN_END]
        .copy()
        .reset_index(drop=True)
    )

    test = (
        featured[featured["Date"] > TRAIN_END]
        .copy()
        .reset_index(drop=True)
    )

    print(
        "\n--- CHRONOLOGICAL SPLIT ---"
    )

    print(
        f"Train: "
        f"{train['Date'].min().date()} "
        f"to "
        f"{train['Date'].max().date()} "
        f"({len(train):,})"
    )

    print(
        f"Test:  "
        f"{test['Date'].min().date()} "
        f"to "
        f"{test['Date'].max().date()} "
        f"({len(test):,})"
    )

    fitted = fit_models(
        train
    )

    # --------------------------------------------------------
    # Score untouched holdout
    # --------------------------------------------------------

    original_scores = score_models(
        test,
        fitted,
    )

    test = test.copy()

    test["Baseline_Normal"] = (
        ~original_scores[
            "Baseline_Flag"
        ]
    )

    test["IF_Normal"] = (
        ~original_scores[
            "IF_Flag"
        ]
    )

    test["OCSVM_Normal"] = (
        ~original_scores[
            "OCSVM_Flag"
        ]
    )

    # --------------------------------------------------------
    # Use only observations originally classified
    # NORMAL by all three models.
    # --------------------------------------------------------

    normal_mask = (
        test["Baseline_Normal"]
        & test["IF_Normal"]
        & test["OCSVM_Normal"]
    )

    reference_pool = (
        test[
            normal_mask
        ]
        .copy()
        .reset_index(drop=True)
    )

    print(
        "\n--- CLEAN HOLDOUT REFERENCE POOL ---"
    )

    print(
        f"Untouched test observations: "
        f"{len(test):,}"
    )

    print(
        f"Normal by all three: "
        f"{len(reference_pool):,}"
    )

    print(
        f"Reference-pool share: "
        f"{len(reference_pool) / len(test):.2%}"
    )

    if len(reference_pool) < 100:

        raise RuntimeError(
            "Reference pool is unexpectedly "
            "small."
        )

    # --------------------------------------------------------
    # Validate synthetic anomalies
    # --------------------------------------------------------

    (
        summary,
        details,
    ) = run_synthetic_validation(
        reference_pool,
        train,
        fitted,
    )

    print(
        "\n--- SYNTHETIC DETECTION RECALL ---"
    )

    printable = (
        summary.copy()
    )

    percentage_columns = [
        "Baseline_Recall",
        "IF_Recall",
        "OCSVM_Recall",
        "Any_ML_Recall",
        "All_Three_Recall",
    ]

    for column in (
        percentage_columns
    ):

        printable[column] = (
            100
            * printable[column]
        )

    print(
        printable
        .round(2)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Aggregate by model
    # --------------------------------------------------------

    print(
        "\n--- MEAN RECALL ACROSS "
        "ALL SYNTHETIC TESTS ---"
    )

    for column in [
        "Baseline_Recall",
        "IF_Recall",
        "OCSVM_Recall",
    ]:

        print(
            f"{column:20s}: "
            f"{summary[column].mean():.2%}"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    details.to_csv(
        DETAIL_FILE,
        index=False,
    )

    print(
        "\n--- OUTPUT ---"
    )

    print(SUMMARY_FILE)
    print(DETAIL_FILE)

    print(
        "\nSynthetic anomaly validation "
        "complete."
    )


if __name__ == "__main__":
    main()

