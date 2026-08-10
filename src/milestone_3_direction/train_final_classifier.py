from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn

from scipy.optimize import minimize_scalar
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

TRAINING_FILE = (
    ROOT
    / "data"
    / "processed"
    / "usdtry_direction_features.csv"
)

OOS_PREDICTIONS_FILE = (
    ROOT
    / "outputs"
    / "milestone_3"
    / "validation"
    / "purged_rf_predictions.csv"
)

SELECTION_FILE = (
    ROOT
    / "outputs"
    / "milestone_3"
    / "validation"
    / "final_model_selection.json"
)

MODEL_DIR = (
    ROOT
    / "models"
    / "milestone_3"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "milestone_3"
    / "training"
)

MODEL_FILE = (
    MODEL_DIR
    / "usdtry_direction_rf.joblib"
)

CALIBRATION_FILE = (
    MODEL_DIR
    / "usdtry_direction_temperature.json"
)

METADATA_FILE = (
    MODEL_DIR
    / "usdtry_direction_metadata.json"
)

IMPORTANCE_FILE = (
    MODEL_DIR
    / "usdtry_direction_feature_importance.csv"
)

TRAINING_SUMMARY_FILE = (
    OUTPUT_DIR
    / "final_training_summary.csv"
)

CALIBRATION_DIAGNOSTICS_FILE = (
    OUTPUT_DIR
    / "final_temperature_fit_diagnostics.csv"
)


# ============================================================
# Frozen M3 specification
# ============================================================

MODEL_VERSION = "M3-v0.1.0"

TARGET = "Target_5D_0p5pct"

FEATURES = [
    "Return_1D",
    "Return_5D",
    "Volatility_5D",
    "Volatility_20D",
    "Volatility_60D",
    "MA_Distance_20D",
    "MA_Slope_20D",
    "Drawdown_60D",
]

LABELS = [
    "DOWN",
    "FLAT",
    "UP",
]

OOS_PROBABILITY_COLUMNS = [
    "RandomForest_P_DOWN",
    "RandomForest_P_FLAT",
    "RandomForest_P_UP",
]

RF_PARAMETERS = {
    "n_estimators": 500,
    "max_depth": 6,
    "min_samples_leaf": 10,
    "max_features": "sqrt",
    "class_weight": None,
    "random_state": 42,
    "n_jobs": -1,
}

EPSILON = 1e-12


# ============================================================
# Temperature scaling
# ============================================================

def softmax(logits):
    logits = np.asarray(
        logits,
        dtype=float,
    )

    logits = (
        logits
        - np.max(
            logits,
            axis=1,
            keepdims=True,
        )
    )

    exp_values = np.exp(logits)

    return (
        exp_values
        / exp_values.sum(
            axis=1,
            keepdims=True,
        )
    )


def temperature_scale(
    probabilities,
    temperature,
):
    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    probabilities = np.clip(
        probabilities,
        EPSILON,
        1.0,
    )

    logits = np.log(
        probabilities
    )

    return softmax(
        logits / temperature
    )


def fit_temperature(
    y_true,
    probabilities,
):
    y_true = np.asarray(
        y_true
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    def objective(
        log_temperature,
    ):
        temperature = np.exp(
            log_temperature
        )

        calibrated = temperature_scale(
            probabilities,
            temperature,
        )

        return log_loss(
            y_true,
            calibrated,
            labels=LABELS,
        )

    result = minimize_scalar(
        objective,
        bounds=(
            np.log(0.10),
            np.log(10.0),
        ),
        method="bounded",
        options={
            "xatol": 1e-7,
        },
    )

    if not result.success:
        raise RuntimeError(
            "Final temperature optimization failed."
        )

    return float(
        np.exp(result.x)
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

    # --------------------------------------------------------
    # Verify architecture selection
    # --------------------------------------------------------

    if not SELECTION_FILE.exists():
        raise FileNotFoundError(
            f"Missing final selection artifact: "
            f"{SELECTION_FILE}"
        )

    with open(
        SELECTION_FILE,
        "r",
        encoding="utf-8",
    ) as handle:
        selection = json.load(handle)

    if (
        selection.get("Status")
        != "SELECTED_FOR_FINAL_TRAINING"
    ):
        raise ValueError(
            "M3 architecture has not been approved "
            "for final training."
        )

    if (
        selection.get("Feature_Set")
        != "BASE_8_FEATURES"
    ):
        raise ValueError(
            "Unexpected selected feature architecture."
        )

    if selection.get("Use_M1"):
        raise ValueError(
            "Final M3 architecture must not use M1."
        )

    if selection.get("Use_M2"):
        raise ValueError(
            "Final M3 architecture must not use M2."
        )

    print(
        "\n"
        "=========================================="
    )
    print(
        "MILESTONE 3 — FINAL TRAINING"
    )
    print(
        "=========================================="
    )

    # ========================================================
    # Load final labeled training set
    # ========================================================

    df = pd.read_csv(
        TRAINING_FILE,
        parse_dates=["Date"],
    )

    df = (
        df
        .sort_values("Date")
        .reset_index(drop=True)
    )

    required = (
        ["Date", TARGET]
        + FEATURES
    )

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing training columns: {missing}"
        )

    if df[required].isna().any().any():
        raise ValueError(
            "Final training data contain missing values."
        )

    if (
        np.isinf(
            df[FEATURES]
            .to_numpy(dtype=float)
        )
        .any()
    ):
        raise ValueError(
            "Final training features contain "
            "infinite values."
        )

    observed_classes = set(
        df[TARGET].unique()
    )

    if observed_classes != set(LABELS):
        raise ValueError(
            "Final training set does not contain "
            "the expected three target classes."
        )

    X = df[FEATURES]
    y = df[TARGET]

    print(
        f"Training observations: "
        f"{len(df):,}"
    )

    print(
        f"Training period: "
        f"{df['Date'].min().date()} "
        f"to "
        f"{df['Date'].max().date()}"
    )

    print(
        "\nTraining target distribution:"
    )

    class_counts = (
        y.value_counts()
        .reindex(LABELS)
    )

    class_shares = (
        y.value_counts(
            normalize=True
        )
        .reindex(LABELS)
    )

    for label in LABELS:
        print(
            f"  {label:4s}: "
            f"{int(class_counts[label]):4d} "
            f"({class_shares[label] * 100:.2f}%)"
        )

    # ========================================================
    # Fit final production classifier
    # ========================================================

    print(
        "\nFitting final canonical Random Forest..."
    )

    model = RandomForestClassifier(
        **RF_PARAMETERS
    )

    model.fit(
        X,
        y,
    )

    if list(model.classes_) != LABELS:
        raise ValueError(
            "Classifier class order differs from "
            "the frozen M3 label order."
        )

    # ========================================================
    # Fit final production temperature
    # ========================================================

    print(
        "\nFitting production temperature "
        "from purged OOS probabilities..."
    )

    oos = pd.read_csv(
        OOS_PREDICTIONS_FILE,
        parse_dates=["Date"],
    )

    oos_required = [
        "Date",
        "Actual",
        *OOS_PROBABILITY_COLUMNS,
    ]

    missing_oos = [
        column
        for column in oos_required
        if column not in oos.columns
    ]

    if missing_oos:
        raise ValueError(
            f"Missing OOS calibration columns: "
            f"{missing_oos}"
        )

    if (
        oos[
            OOS_PROBABILITY_COLUMNS
        ]
        .isna()
        .any()
        .any()
    ):
        raise ValueError(
            "Purged OOS probabilities contain "
            "missing values."
        )

    probability_sums = (
        oos[
            OOS_PROBABILITY_COLUMNS
        ]
        .sum(axis=1)
    )

    max_probability_error = float(
        (
            probability_sums
            - 1.0
        )
        .abs()
        .max()
    )

    if max_probability_error > 1e-6:
        raise ValueError(
            "Purged OOS probabilities do not "
            "sum to one."
        )

    y_oos = (
        oos["Actual"]
        .to_numpy()
    )

    p_oos = (
        oos[
            OOS_PROBABILITY_COLUMNS
        ]
        .to_numpy(
            dtype=float
        )
    )

    temperature = fit_temperature(
        y_oos,
        p_oos,
    )

    p_calibrated = (
        temperature_scale(
            p_oos,
            temperature,
        )
    )

    raw_log_loss = log_loss(
        y_oos,
        p_oos,
        labels=LABELS,
    )

    calibrated_log_loss = log_loss(
        y_oos,
        p_calibrated,
        labels=LABELS,
    )

    raw_prediction = (
        np.array(LABELS)[
            p_oos.argmax(axis=1)
        ]
    )

    calibrated_prediction = (
        np.array(LABELS)[
            p_calibrated.argmax(axis=1)
        ]
    )

    predictions_unchanged = bool(
        np.array_equal(
            raw_prediction,
            calibrated_prediction,
        )
    )

    if not predictions_unchanged:
        raise ValueError(
            "Positive temperature scaling unexpectedly "
            "changed class predictions."
        )

    print(
        f"OOS calibration observations: "
        f"{len(oos):,}"
    )

    print(
        f"OOS calibration period: "
        f"{oos['Date'].min().date()} "
        f"to "
        f"{oos['Date'].max().date()}"
    )

    print(
        f"Final temperature: "
        f"{temperature:.6f}"
    )

    print(
        f"Raw OOS log loss: "
        f"{raw_log_loss:.6f}"
    )

    print(
        f"Temperature-fit log loss: "
        f"{calibrated_log_loss:.6f}"
    )

    print(
        "Class predictions unchanged: "
        f"{predictions_unchanged}"
    )

    # NOTE:
    # The calibrated log-loss value here is a fit diagnostic
    # because this same OOS probability set is used to estimate
    # the final production temperature. The prior chronological
    # calibration_analysis.py results remain the unbiased
    # validation evidence for selecting temperature scaling.

    # ========================================================
    # Feature importance
    # ========================================================

    importance = pd.DataFrame(
        {
            "Feature":
                FEATURES,

            "Importance":
                model.feature_importances_,
        }
    )

    importance = (
        importance
        .sort_values(
            "Importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    # ========================================================
    # Save model
    # ========================================================

    joblib.dump(
        model,
        MODEL_FILE,
    )

    importance.to_csv(
        IMPORTANCE_FILE,
        index=False,
    )

    # ========================================================
    # Save calibration artifact
    # ========================================================

    calibration_metadata = {
        "model_version":
            MODEL_VERSION,

        "method":
            "temperature_scaling",

        "temperature":
            temperature,

        "class_order":
            LABELS,

        "fit_source":
            "purged annual OOS Random Forest probabilities",

        "fit_observations":
            int(len(oos)),

        "fit_start":
            str(
                oos["Date"]
                .min()
                .date()
            ),

        "fit_end":
            str(
                oos["Date"]
                .max()
                .date()
            ),

        "raw_log_loss_fit_diagnostic":
            raw_log_loss,

        "calibrated_log_loss_fit_diagnostic":
            calibrated_log_loss,

        "class_predictions_unchanged":
            predictions_unchanged,

        "note":
            (
                "Fit diagnostics are not new unbiased "
                "validation metrics. Chronological expanding "
                "calibration validation is stored under "
                "outputs/milestone_3/validation."
            ),
    }

    with open(
        CALIBRATION_FILE,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            calibration_metadata,
            handle,
            indent=2,
        )

    # ========================================================
    # Save complete model metadata
    # ========================================================

    metadata = {
        "model_version":
            MODEL_VERSION,

        "milestone":
            3,

        "task":
            "USDTRY 5-trading-day direction classification",

        "target":
            TARGET,

        "target_definition": {
            "horizon_trading_days":
                5,

            "down_threshold":
                -0.005,

            "up_threshold":
                0.005,

            "classes":
                LABELS,
        },

        "feature_set":
            "BASE_8_FEATURES",

        "features":
            FEATURES,

        "classifier":
            "RandomForestClassifier",

        "random_forest_parameters":
            RF_PARAMETERS,

        "calibration":
            {
                "method":
                    "temperature_scaling",

                "temperature":
                    temperature,

                "artifact":
                    CALIBRATION_FILE.name,
            },

        "use_m1_features":
            False,

        "use_m2_features":
            False,

        "training": {
            "observations":
                int(len(df)),

            "start_date":
                str(
                    df["Date"]
                    .min()
                    .date()
                ),

            "end_date":
                str(
                    df["Date"]
                    .max()
                    .date()
                ),

            "class_counts": {
                label:
                    int(
                        class_counts[label]
                    )
                for label in LABELS
            },

            "class_shares": {
                label:
                    float(
                        class_shares[label]
                    )
                for label in LABELS
            },
        },

        "validation_reference": {
            "method":
                (
                    "purged expanding annual "
                    "walk-forward"
                ),

            "purge_observations":
                5,

            "purged_oos_balanced_accuracy":
                float(
                    selection[
                        "Purged_OOS_Balanced_Accuracy"
                    ]
                ),

            "purged_oos_macro_f1":
                float(
                    selection[
                        "Purged_OOS_Macro_F1"
                    ]
                ),

            "purged_oos_down_f1":
                float(
                    selection[
                        "Purged_OOS_DOWN_F1"
                    ]
                ),

            "cross_milestone_features":
                "rejected",

            "temperature_scaling":
                "selected",
        },

        "software": {
            "python":
                __import__(
                    "sys"
                ).version.split()[0],

            "scikit_learn":
                sklearn.__version__,

            "scipy":
                scipy.__version__,

            "pandas":
                pd.__version__,

            "numpy":
                np.__version__,

            "joblib":
                joblib.__version__,
        },
    }

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            metadata,
            handle,
            indent=2,
        )

    # ========================================================
    # Save concise training summary
    # ========================================================

    summary = pd.DataFrame(
        [
            {
                "Model_Version":
                    MODEL_VERSION,

                "Training_Observations":
                    len(df),

                "Training_Start":
                    df["Date"]
                    .min()
                    .date(),

                "Training_End":
                    df["Date"]
                    .max()
                    .date(),

                "DOWN_Count":
                    int(
                        class_counts["DOWN"]
                    ),

                "FLAT_Count":
                    int(
                        class_counts["FLAT"]
                    ),

                "UP_Count":
                    int(
                        class_counts["UP"]
                    ),

                "Final_Temperature":
                    temperature,

                "OOS_Calibration_Observations":
                    len(oos),

                "Raw_OOS_LogLoss_Fit_Diagnostic":
                    raw_log_loss,

                "Calibrated_LogLoss_Fit_Diagnostic":
                    calibrated_log_loss,

                "Model_File":
                    str(
                        MODEL_FILE.relative_to(
                            ROOT
                        )
                    ),

                "Calibration_File":
                    str(
                        CALIBRATION_FILE.relative_to(
                            ROOT
                        )
                    ),

                "Metadata_File":
                    str(
                        METADATA_FILE.relative_to(
                            ROOT
                        )
                    ),
            }
        ]
    )

    summary.to_csv(
        TRAINING_SUMMARY_FILE,
        index=False,
    )

    diagnostics = pd.DataFrame(
        [
            {
                "Temperature":
                    temperature,

                "OOS_Observations":
                    len(oos),

                "Raw_Log_Loss":
                    raw_log_loss,

                "Calibrated_Log_Loss":
                    calibrated_log_loss,

                "Delta_Log_Loss":
                    (
                        calibrated_log_loss
                        - raw_log_loss
                    ),

                "Class_Predictions_Unchanged":
                    predictions_unchanged,

                "Max_Input_Probability_Sum_Error":
                    max_probability_error,
            }
        ]
    )

    diagnostics.to_csv(
        CALIBRATION_DIAGNOSTICS_FILE,
        index=False,
    )

    # ========================================================
    # Reproducibility check
    # ========================================================

    reloaded = joblib.load(
        MODEL_FILE
    )

    original_probabilities = (
        model.predict_proba(
            X
        )
    )

    reloaded_probabilities = (
        reloaded.predict_proba(
            X
        )
    )

    max_reload_difference = float(
        np.max(
            np.abs(
                original_probabilities
                - reloaded_probabilities
            )
        )
    )

    if max_reload_difference > 1e-12:
        raise ValueError(
            "Serialized model failed "
            "reproducibility check."
        )

    # ========================================================
    # Final report
    # ========================================================

    print(
        "\n"
        "=========================================="
    )
    print(
        "FINAL M3 TRAINING COMPLETE"
    )
    print(
        "=========================================="
    )

    print(
        f"Model version:       "
        f"{MODEL_VERSION}"
    )

    print(
        f"Classifier:          "
        f"Canonical Random Forest"
    )

    print(
        f"Features:            "
        f"{len(FEATURES)}"
    )

    print(
        f"Training rows:       "
        f"{len(df):,}"
    )

    print(
        f"Final temperature:   "
        f"{temperature:.6f}"
    )

    print(
        f"Reload max diff:     "
        f"{max_reload_difference:.3e}"
    )

    print(
        "\n--- FINAL FEATURE IMPORTANCE ---"
    )

    print(
        importance
        .round(4)
        .to_string(index=False)
    )

    print(
        "\n--- SAVED ---"
    )

    for path in [
        MODEL_FILE,
        CALIBRATION_FILE,
        METADATA_FILE,
        IMPORTANCE_FILE,
        TRAINING_SUMMARY_FILE,
        CALIBRATION_DIAGNOSTICS_FILE,
    ]:
        print(path)


if __name__ == "__main__":
    main()