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
    / "synthetic_flip_validation"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "synthetic_flip_validation_summary.csv"
)

DETAIL_FILE = (
    OUTPUT_DIR
    / "synthetic_flip_validation_details.csv"
)

MONOTONICITY_FILE = (
    OUTPUT_DIR
    / "synthetic_flip_monotonicity.csv"
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

def baseline_scores(df):

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

    score = matrix.max(
        axis=1
    )

    flag = score >= 1.0

    return (
        score.astype(float),
        flag.astype(bool),
    )


# ============================================================
# OCSVM preprocessing
# ============================================================

def fit_ocsvm_preprocessor(
    X_train,
):

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

    robust_train = (
        (X_train - median)
        / iqr
    )

    compressed_train = (
        np.arcsinh(
            robust_train
        )
    )

    scaler = StandardScaler()

    X_scaled = (
        scaler.fit_transform(
            compressed_train
        )
    )

    return (
        X_scaled,
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

    compressed = (
        np.arcsinh(
            robust
        )
    )

    return scaler.transform(
        compressed
    )


# ============================================================
# Fit models — TRAIN ONLY
# ============================================================

def fit_models(train):

    X_train = (
        train[MODEL_FEATURES]
        .to_numpy(dtype=float)
    )

    _, baseline_flag = (
        baseline_scores(train)
    )

    training_anomaly_rate = float(
        baseline_flag.mean()
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
        f"{training_anomaly_rate:.2%}"
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
            1.0
            - training_anomaly_rate,
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
        nu=training_anomaly_rate,
    )

    ocsvm.fit(
        X_train_scaled
    )

    ocsvm_train_score = (
        -ocsvm
        .decision_function(
            X_train_scaled
        )
        .ravel()
    )

    ocsvm_threshold = float(
        np.quantile(
            ocsvm_train_score,
            1.0
            - training_anomaly_rate,
        )
    )

    return {
        "isolation":
            isolation,

        "if_threshold":
            if_threshold,

        "ocsvm":
            ocsvm,

        "ocsvm_threshold":
            ocsvm_threshold,

        "median":
            median,

        "iqr":
            iqr,

        "scaler":
            scaler,

        "training_anomaly_rate":
            training_anomaly_rate,
    }


# ============================================================
# Score models
# ============================================================

def score_models(
    df,
    fitted,
):

    X = (
        df[MODEL_FEATURES]
        .to_numpy(dtype=float)
    )

    # Baseline
    baseline_score, baseline_flag = (
        baseline_scores(df)
    )

    # Isolation Forest
    if_score = (
        -fitted[
            "isolation"
        ]
        .score_samples(X)
    )

    if_flag = (
        if_score
        >= fitted[
            "if_threshold"
        ]
    )

    # OCSVM
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
        "Baseline_Score":
            baseline_score,

        "Baseline_Flag":
            baseline_flag,

        "IF_Score":
            if_score,

        "IF_Flag":
            if_flag,

        "OCSVM_Score":
            ocsvm_score,

        "OCSVM_Flag":
            ocsvm_flag,
    }


# ============================================================
# Training feature IQR
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

    scale = (
        scale
        .replace(0, np.nan)
        .fillna(1.0)
    )

    return scale


# ============================================================
# Synthetic perturbations
# ============================================================

def inject_return_shock(
    df,
    scale,
    severity,
    rng,
):

    out = df.copy()

    signs = rng.choice(
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
    rng,
):

    out = inject_return_shock(
        df,
        scale,
        severity,
        rng,
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
# Evaluate one model
# ============================================================

def evaluate_model(
    original_flag,
    original_score,
    perturbed_flag,
    perturbed_score,
):

    original_flag = np.asarray(
        original_flag,
        dtype=bool,
    )

    perturbed_flag = np.asarray(
        perturbed_flag,
        dtype=bool,
    )

    original_score = np.asarray(
        original_score,
        dtype=float,
    )

    perturbed_score = np.asarray(
        perturbed_score,
        dtype=float,
    )

    originally_normal = (
        ~original_flag
    )

    n_originally_normal = int(
        originally_normal.sum()
    )

    flips = (
        originally_normal
        & perturbed_flag
    )

    flip_rate = (
        flips.sum()
        / n_originally_normal
        if n_originally_normal > 0
        else np.nan
    )

    score_uplift = (
        perturbed_score
        - original_score
    )

    return {
        "Original_Anomaly_Rate":
            original_flag.mean(),

        "Perturbed_Detection_Rate":
            perturbed_flag.mean(),

        "Originally_Normal":
            n_originally_normal,

        "Normal_to_Anomaly_Flips":
            int(flips.sum()),

        "Flip_Rate":
            flip_rate,

        "Median_Score_Uplift":
            float(
                np.median(
                    score_uplift
                )
            ),

        "Mean_Score_Uplift":
            float(
                np.mean(
                    score_uplift
                )
            ),

        "Positive_Score_Uplift_Rate":
            float(
                (
                    score_uplift > 0
                ).mean()
            ),
    }


# ============================================================
# Validation
# ============================================================

def run_validation(
    test,
    train,
    fitted,
):

    scale = feature_iqr(
        train
    )

    original = score_models(
        test,
        fitted,
    )

    anomaly_types = [
        "RETURN_SHOCK",
        "VOLATILITY_SPIKE",
        "DRAWDOWN_BREAK",
        "COMPOUND_STRESS",
    ]

    summary_rows = []

    detail_frames = []

    for type_index, anomaly_type in enumerate(
        anomaly_types
    ):

        for severity in [
            1,
            2,
            3,
        ]:

            # Deterministic scenario-specific RNG
            rng = np.random.default_rng(
                RANDOM_STATE
                + type_index * 100
                + severity
            )

            if anomaly_type == "RETURN_SHOCK":

                synthetic = (
                    inject_return_shock(
                        test,
                        scale,
                        severity,
                        rng,
                    )
                )

            elif anomaly_type == "VOLATILITY_SPIKE":

                synthetic = (
                    inject_volatility_spike(
                        test,
                        severity,
                    )
                )

            elif anomaly_type == "DRAWDOWN_BREAK":

                synthetic = (
                    inject_drawdown_break(
                        test,
                        scale,
                        severity,
                    )
                )

            else:

                synthetic = (
                    inject_compound_stress(
                        test,
                        scale,
                        severity,
                        rng,
                    )
                )

            perturbed = score_models(
                synthetic,
                fitted,
            )

            for model_name in [
                "Baseline",
                "IF",
                "OCSVM",
            ]:

                metrics = evaluate_model(
                    original[
                        f"{model_name}_Flag"
                    ],
                    original[
                        f"{model_name}_Score"
                    ],
                    perturbed[
                        f"{model_name}_Flag"
                    ],
                    perturbed[
                        f"{model_name}_Score"
                    ],
                )

                summary_rows.append(
                    {
                        "Anomaly_Type":
                            anomaly_type,

                        "Severity":
                            severity,

                        "Model":
                            model_name,

                        **metrics,
                    }
                )

            # ------------------------------------------------
            # Detailed row-level output
            # ------------------------------------------------

            detail = pd.DataFrame(
                {
                    "Date":
                        test[
                            "Date"
                        ].to_numpy(),

                    "Anomaly_Type":
                        anomaly_type,

                    "Severity":
                        severity,
                }
            )

            for model_name in [
                "Baseline",
                "IF",
                "OCSVM",
            ]:

                detail[
                    f"{model_name}_Original_Flag"
                ] = original[
                    f"{model_name}_Flag"
                ]

                detail[
                    f"{model_name}_Perturbed_Flag"
                ] = perturbed[
                    f"{model_name}_Flag"
                ]

                detail[
                    f"{model_name}_Original_Score"
                ] = original[
                    f"{model_name}_Score"
                ]

                detail[
                    f"{model_name}_Perturbed_Score"
                ] = perturbed[
                    f"{model_name}_Score"
                ]

                detail[
                    f"{model_name}_Score_Uplift"
                ] = (
                    perturbed[
                        f"{model_name}_Score"
                    ]
                    - original[
                        f"{model_name}_Score"
                    ]
                )

            detail_frames.append(
                detail
            )

    return (
        pd.DataFrame(
            summary_rows
        ),
        pd.concat(
            detail_frames,
            ignore_index=True,
        ),
    )


# ============================================================
# Monotonicity
# ============================================================

def check_monotonicity(
    summary,
):

    rows = []

    for anomaly_type in (
        summary[
            "Anomaly_Type"
        ].unique()
    ):

        for model in [
            "Baseline",
            "IF",
            "OCSVM",
        ]:

            subset = (
                summary[
                    (
                        summary[
                            "Anomaly_Type"
                        ]
                        == anomaly_type
                    )
                    &
                    (
                        summary[
                            "Model"
                        ]
                        == model
                    )
                ]
                .sort_values(
                    "Severity"
                )
            )

            flip_rates = (
                subset[
                    "Flip_Rate"
                ]
                .to_numpy()
            )

            detection_rates = (
                subset[
                    "Perturbed_Detection_Rate"
                ]
                .to_numpy()
            )

            flip_monotonic = bool(
                np.all(
                    np.diff(
                        flip_rates
                    )
                    >= -1e-12
                )
            )

            detection_monotonic = bool(
                np.all(
                    np.diff(
                        detection_rates
                    )
                    >= -1e-12
                )
            )

            rows.append(
                {
                    "Anomaly_Type":
                        anomaly_type,

                    "Model":
                        model,

                    "Flip_Rate_Monotonic":
                        flip_monotonic,

                    "Detection_Rate_Monotonic":
                        detection_monotonic,
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# Print
# ============================================================

def print_summary(summary):

    printable = summary[
        [
            "Anomaly_Type",
            "Severity",
            "Model",
            "Original_Anomaly_Rate",
            "Perturbed_Detection_Rate",
            "Originally_Normal",
            "Normal_to_Anomaly_Flips",
            "Flip_Rate",
            "Median_Score_Uplift",
            "Positive_Score_Uplift_Rate",
        ]
    ].copy()

    for column in [
        "Original_Anomaly_Rate",
        "Perturbed_Detection_Rate",
        "Flip_Rate",
        "Positive_Score_Uplift_Rate",
    ]:

        printable[column] *= 100

    print(
        "\n--- FAIR SYNTHETIC FLIP VALIDATION ---"
    )

    print(
        printable
        .round(3)
        .to_string(index=False)
    )


# ============================================================
# Aggregate ranking
# ============================================================

def print_aggregate(summary):

    aggregate = (
        summary
        .groupby("Model")
        .agg(
            Mean_Flip_Rate=(
                "Flip_Rate",
                "mean",
            ),
            Mean_Detection_Rate=(
                "Perturbed_Detection_Rate",
                "mean",
            ),
            Mean_Positive_Uplift=(
                "Positive_Score_Uplift_Rate",
                "mean",
            ),
            Median_Score_Uplift=(
                "Median_Score_Uplift",
                "median",
            ),
        )
        .reset_index()
    )

    print(
        "\n--- AGGREGATE SYNTHETIC RESPONSE ---"
    )

    printable = (
        aggregate.copy()
    )

    for column in [
        "Mean_Flip_Rate",
        "Mean_Detection_Rate",
        "Mean_Positive_Uplift",
    ]:

        printable[column] *= 100

    print(
        printable
        .round(3)
        .to_string(index=False)
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
        "FAIR SYNTHETIC FLIP VALIDATION ---"
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

    summary, details = (
        run_validation(
            test,
            train,
            fitted,
        )
    )

    monotonicity = (
        check_monotonicity(
            summary
        )
    )

    print_summary(
        summary
    )

    print_aggregate(
        summary
    )

    print(
        "\n--- MONOTONICITY CHECK ---"
    )

    print(
        monotonicity
        .to_string(index=False)
    )

    print(
        "\nFlip monotonicity pass rate: "
        f"{monotonicity['Flip_Rate_Monotonic'].mean():.2%}"
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

    monotonicity.to_csv(
        MONOTONICITY_FILE,
        index=False,
    )

    print(
        "\n--- OUTPUT ---"
    )

    print(SUMMARY_FILE)
    print(DETAIL_FILE)
    print(MONOTONICITY_FILE)

    print(
        "\nFair synthetic flip validation "
        "complete."
    )


if __name__ == "__main__":
    main()
