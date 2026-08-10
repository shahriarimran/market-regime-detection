from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest

from .prepare_anomaly_features import (
    FINAL_TRAINING_CUTOFF,
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

MODEL_DIR = (
    M2_DIR
    / "models"
)

OUTPUT_DIR = (
    M2_DIR
    / "outputs"
    / "final_training"
)

MODEL_FILE = (
    MODEL_DIR
    / "usdtry_isolation_forest.joblib"
)

METADATA_FILE = (
    MODEL_DIR
    / "usdtry_anomaly_metadata.json"
)

TRAINING_SCORES_FILE = (
    MODEL_DIR
    / "usdtry_if_training_scores.npy"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "final_training_summary.csv"
)

LATEST_REFERENCE_FILE = (
    OUTPUT_DIR
    / "latest_training_reference.json"
)


# ============================================================
# Version
# ============================================================

MODEL_VERSION = "M2-v0.2.0"


# ============================================================
# Final Isolation Forest specification
#
# This is NOT the primary binary anomaly detector.
# It is retained as a continuous multivariate anomaly score.
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

N_ESTIMATORS = 500
MAX_SAMPLES = 512
RANDOM_STATE = 42


# ============================================================
# Final statistical baseline thresholds
#
# These control the operational NORMAL / ANOMALOUS decision.
# ============================================================

RETURN_Z_THRESHOLD = 4.0
VOLATILITY_RATIO_THRESHOLD = 2.0
DRAWDOWN_CHANGE_THRESHOLD = 0.05
ABS_RETURN_THRESHOLD = 0.04


# ============================================================
# Load
# ============================================================

def load_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input feature dataset not found:\n{INPUT_FILE}"
        )

    raw = pd.read_csv(
        INPUT_FILE,
        parse_dates=["Date"],
    )

    required = {
        "Date",
        "USDTRY",
        *M1_FEATURES,
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    raw = (
        raw[raw["Date"] <= FINAL_TRAINING_CUTOFF]
        .sort_values("Date")
        .reset_index(drop=True)
    )

    volatility_floor = estimate_volatility_floor(
        raw,
        quantile=VOLATILITY_FLOOR_QUANTILE,
    )

    df = validate_features(
        add_anomaly_features(
            raw,
            volatility_floor=volatility_floor,
        )
    )

    X = df[FEATURES].to_numpy(dtype=float)
    if not np.isfinite(X).all():
        raise ValueError(
            "Feature matrix contains NaN or infinite values."
        )

    return df, volatility_floor

# ============================================================
# Statistical baseline
# ============================================================

def calculate_baseline(df):

    result = df.copy()

    result["Score_Return_Z"] = (
        result["Return_ZScore_20D"].abs()
        / RETURN_Z_THRESHOLD
    )

    result["Score_Volatility_Ratio"] = (
        result["Volatility_Ratio_5D_60D"]
        / VOLATILITY_RATIO_THRESHOLD
    )

    result["Score_Drawdown_Change"] = (
        result["Drawdown_Change_5D"].abs()
        / DRAWDOWN_CHANGE_THRESHOLD
    )

    result["Score_Absolute_Return"] = (
        result["Absolute_Return_1D"]
        / ABS_RETURN_THRESHOLD
    )

    score_columns = [
        "Score_Return_Z",
        "Score_Volatility_Ratio",
        "Score_Drawdown_Change",
        "Score_Absolute_Return",
    ]

    result["Baseline_Anomaly_Score"] = (
        result[score_columns]
        .max(axis=1)
    )

    result["Baseline_Anomaly"] = (
        result["Baseline_Anomaly_Score"]
        >= 1.0
    )

    rule_columns = {
        "Score_Return_Z":
            "LOCAL_RETURN_SHOCK",

        "Score_Volatility_Ratio":
            "VOLATILITY_ACCELERATION",

        "Score_Drawdown_Change":
            "DRAWDOWN_SHIFT",

        "Score_Absolute_Return":
            "LARGE_ABSOLUTE_RETURN",
    }

    dominant = (
        result[
            list(rule_columns.keys())
        ]
        .idxmax(axis=1)
    )

    result["Primary_Reason"] = (
        dominant.map(
            rule_columns
        )
    )

    return result


# ============================================================
# Train final IF
# ============================================================

def train_isolation_forest(df):

    X = (
        df[FEATURES]
        .to_numpy(dtype=float)
    )

    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        max_samples=min(
            MAX_SAMPLES,
            len(df),
        ),
        contamination="auto",
        max_features=1.0,
        bootstrap=False,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X)

    # sklearn:
    # lower score_samples = more abnormal
    #
    # Reverse sign so:
    # higher score = more abnormal
    training_scores = (
        -model.score_samples(X)
    )

    return (
        model,
        training_scores,
    )


# ============================================================
# Empirical percentile
# ============================================================

def empirical_percentile(
    score,
    sorted_reference_scores,
):

    position = np.searchsorted(
        sorted_reference_scores,
        score,
        side="right",
    )

    return float(
        position
        / len(sorted_reference_scores)
    )


# ============================================================
# Main
# ============================================================

def main():

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\n--- MILESTONE 2: "
        "FINAL ANOMALY MODEL TRAINING ---"
    )

    (
        df,
        volatility_floor,
    ) = load_data()

    print(
        f"Observations: {len(df):,}"
    )

    print(
        f"Period: "
        f"{df['Date'].min().date()} "
        f"to "
        f"{df['Date'].max().date()}"
    )

    print(
        f"Isolation Forest features: "
        f"{len(FEATURES)}"
    )

    # --------------------------------------------------------
    # Primary binary detector
    # --------------------------------------------------------

    scored = calculate_baseline(
        df
    )

    baseline_count = int(
        scored[
            "Baseline_Anomaly"
        ].sum()
    )

    baseline_rate = float(
        scored[
            "Baseline_Anomaly"
        ].mean()
    )

    print(
        "\n--- PRIMARY STATISTICAL DETECTOR ---"
    )

    print(
        f"Historical anomalies: "
        f"{baseline_count:,}"
    )

    print(
        f"Historical anomaly rate: "
        f"{baseline_rate:.2%}"
    )

    # --------------------------------------------------------
    # Secondary ML score
    # --------------------------------------------------------

    print(
        "\n--- TRAINING SECONDARY "
        "ISOLATION FOREST ---"
    )

    print(
        f"Trees: {N_ESTIMATORS}"
    )

    print(
        f"Max samples: "
        f"{min(MAX_SAMPLES, len(df))}"
    )

    print(
        f"Random state: "
        f"{RANDOM_STATE}"
    )

    model, training_scores = (
        train_isolation_forest(
            scored
        )
    )

    sorted_scores = np.sort(
        training_scores
    )

    scored[
        "IF_Anomaly_Score"
    ] = training_scores

    scored[
        "IF_Training_Percentile"
    ] = [
        empirical_percentile(
            score,
            sorted_scores,
        )
        for score
        in training_scores
    ]

    print(
        "\n--- IF TRAINING SCORE DISTRIBUTION ---"
    )

    for percentile in [
        0.50,
        0.75,
        0.90,
        0.95,
        0.975,
        0.99,
        0.995,
    ]:

        value = float(
            np.quantile(
                training_scores,
                percentile,
            )
        )

        print(
            f"P{percentile * 100:5.1f}: "
            f"{value:.6f}"
        )

    # --------------------------------------------------------
    # Latest training observation
    # --------------------------------------------------------

    latest = scored.iloc[-1]

    latest_if_score = float(
        latest[
            "IF_Anomaly_Score"
        ]
    )

    latest_percentile = float(
        latest[
            "IF_Training_Percentile"
        ]
    )

    latest_state = (
        "ANOMALOUS"
        if bool(
            latest[
                "Baseline_Anomaly"
            ]
        )
        else "NORMAL"
    )

    print(
        "\n--- LATEST TRAINING REFERENCE ---"
    )

    print(
        f"Date: "
        f"{latest['Date'].date()}"
    )

    print(
        f"USDTRY: "
        f"{latest['USDTRY']:.4f}"
    )

    print(
        f"Operational anomaly state: "
        f"{latest_state}"
    )

    print(
        f"Baseline anomaly score: "
        f"{latest['Baseline_Anomaly_Score']:.4f}"
    )

    print(
        f"Primary reason: "
        f"{latest['Primary_Reason']}"
    )

    print(
        f"IF anomaly score: "
        f"{latest_if_score:.6f}"
    )

    print(
        f"IF historical percentile: "
        f"{latest_percentile:.2%}"
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_artifact = {
        "model": model,
        "feature_columns": FEATURES,
        "volatility_floor": volatility_floor,
        "volatility_floor_quantile": VOLATILITY_FLOOR_QUANTILE,
        "training_cutoff": str(FINAL_TRAINING_CUTOFF.date()),
        "model_version": MODEL_VERSION,
    }

    joblib.dump(
        model_artifact,
        MODEL_FILE,
    )

    # Exact empirical training distribution is saved so
    # operational inference can calculate a reference
    # percentile without pretending the IF score is a
    # probability.
    np.save(
        TRAINING_SCORES_FILE,
        sorted_scores,
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "model_version":
            MODEL_VERSION,

        "milestone":
            "M2_FX_ANOMALY_DETECTION",

        "training_start":
            str(
                df[
                    "Date"
                ].min().date()
            ),

        "training_cutoff":
            str(
                df[
                    "Date"
                ].max().date()
            ),

        "training_observations":
            int(len(df)),

        "volatility_floor": {
            "value": volatility_floor,
            "quantile": VOLATILITY_FLOOR_QUANTILE,
            "estimation_sample": "training_only",
            "training_cutoff": str(FINAL_TRAINING_CUTOFF.date()),
        },

        "primary_operational_detector":
            "STATISTICAL_BASELINE",

        "secondary_ml_signal":
            "ISOLATION_FOREST_SCORE",

        "binary_states": [
            "NORMAL",
            "ANOMALOUS",
        ],

        "features":
            FEATURES,

        "baseline_thresholds": {
            "absolute_return_zscore_20d":
                RETURN_Z_THRESHOLD,

            "volatility_ratio_5d_60d":
                VOLATILITY_RATIO_THRESHOLD,

            "absolute_drawdown_change_5d":
                DRAWDOWN_CHANGE_THRESHOLD,

            "absolute_return_1d":
                ABS_RETURN_THRESHOLD,
        },

        "historical_baseline_anomaly_count":
            baseline_count,

        "historical_baseline_anomaly_rate":
            baseline_rate,

        "isolation_forest": {
            "n_estimators":
                N_ESTIMATORS,

            "max_samples":
                min(
                    MAX_SAMPLES,
                    len(df),
                ),

            "random_state":
                RANDOM_STATE,

            "contamination":
                "auto",

            "operational_binary_threshold":
                None,

            "role":
                "continuous_secondary_score",
        },

        "validation_decision": {
            "isolation_forest_binary":
                "REJECTED_AS_PRIMARY",

            "one_class_svm_binary":
                "REJECTED_AS_PRIMARY",

            "hard_regime_conditioning":
                "REJECTED",

            "statistical_baseline_binary":
                "SELECTED_PRIMARY",

            "isolation_forest_score":
                "SELECTED_SECONDARY",
        },

        "score_interpretation": {
            "if_score":
                "Higher values indicate greater "
                "multivariate isolation.",

            "if_training_percentile":
                "Empirical percentile relative to "
                "the frozen training-history score "
                "distribution; not an anomaly "
                "probability.",
        },
    }

    METADATA_FILE.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Training summary
    # --------------------------------------------------------

    summary = pd.DataFrame(
        [
            {
                "Model_Version":
                    MODEL_VERSION,

                "Training_Start":
                    df[
                        "Date"
                    ].min(),

                "Training_Cutoff":
                    df[
                        "Date"
                    ].max(),

                "Observations":
                    len(df),

                "Volatility_Floor":
                    volatility_floor,

                "Volatility_Floor_Quantile":
                    VOLATILITY_FLOOR_QUANTILE,

                "Baseline_Anomalies":
                    baseline_count,

                "Baseline_Anomaly_Rate":
                    baseline_rate,

                "IF_Score_Median":
                    float(
                        np.median(
                            training_scores
                        )
                    ),

                "IF_Score_P95":
                    float(
                        np.quantile(
                            training_scores,
                            0.95,
                        )
                    ),

                "IF_Score_P99":
                    float(
                        np.quantile(
                            training_scores,
                            0.99,
                        )
                    ),

                "Latest_State":
                    latest_state,

                "Latest_Baseline_Score":
                    float(
                        latest[
                            "Baseline_Anomaly_Score"
                        ]
                    ),

                "Latest_IF_Score":
                    latest_if_score,

                "Latest_IF_Percentile":
                    latest_percentile,
            }
        ]
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Latest-reference JSON
    # --------------------------------------------------------

    latest_reference = {
        "date":
            str(
                latest[
                    "Date"
                ].date()
            ),

        "usdtry":
            float(
                latest[
                    "USDTRY"
                ]
            ),

        "anomaly_state":
            latest_state,

        "baseline_anomaly":
            bool(
                latest[
                    "Baseline_Anomaly"
                ]
            ),

        "volatility_floor":
            volatility_floor,

        "volatility_floor_quantile":
            VOLATILITY_FLOOR_QUANTILE,

        "baseline_anomaly_score":
            float(
                latest[
                    "Baseline_Anomaly_Score"
                ]
            ),

        "primary_reason":
            str(
                latest[
                    "Primary_Reason"
                ]
            ),

        "if_anomaly_score":
            latest_if_score,

        "if_training_percentile":
            latest_percentile,

        "inference_mode":
            "TRAINING_REFERENCE",
    }

    LATEST_REFERENCE_FILE.write_text(
        json.dumps(
            latest_reference,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n--- SAVED ARTIFACTS ---"
    )

    print(MODEL_FILE)
    print(METADATA_FILE)
    print(TRAINING_SCORES_FILE)
    print(SUMMARY_FILE)
    print(LATEST_REFERENCE_FILE)

    print(
        "\nFinal M2 anomaly model "
        "training complete."
    )


if __name__ == "__main__":
    main()
