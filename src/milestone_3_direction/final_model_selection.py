from pathlib import Path
import json

import pandas as pd


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

VALIDATION_DIR = (
    ROOT
    / "outputs"
    / "milestone_3"
    / "validation"
)

RF_SUMMARY_FILE = (
    VALIDATION_DIR
    / "purged_rf_vs_momentum_summary.csv"
)

RF_PER_CLASS_FILE = (
    VALIDATION_DIR
    / "purged_rf_vs_momentum_per_class.csv"
)

SEED_FILE = (
    VALIDATION_DIR
    / "rf_seed_stability.csv"
)

AGREEMENT_FILE = (
    VALIDATION_DIR
    / "rf_seed_prediction_agreement.csv"
)

ABLATION_FILE = (
    VALIDATION_DIR
    / "purged_cross_milestone_summary.csv"
)

CALIBRATION_FILE = (
    VALIDATION_DIR
    / "calibration_summary.csv"
)

CALIBRATED_PREDICTIONS_FILE = (
    VALIDATION_DIR
    / "calibrated_rf_predictions.csv"
)

OUTPUT_CSV = (
    VALIDATION_DIR
    / "final_model_selection.csv"
)

OUTPUT_JSON = (
    VALIDATION_DIR
    / "final_model_selection.json"
)


# ============================================================
# Frozen M3 specification
# ============================================================

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

RF_PARAMETERS = {
    "n_estimators": 500,
    "max_depth": 6,
    "min_samples_leaf": 10,
    "max_features": "sqrt",
    "class_weight": None,
    "random_state": 42,
    "n_jobs": -1,
}


# ============================================================
# Helpers
# ============================================================

def require_file(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Required validation file missing: {path}"
        )


def get_row(
    df,
    column,
    value,
):

    row = df[
        df[column] == value
    ]

    if len(row) != 1:
        raise ValueError(
            f"Expected one row where "
            f"{column} == {value!r}; "
            f"found {len(row)}."
        )

    return row.iloc[0]


# ============================================================
# Main
# ============================================================

def main():

    for path in [
        RF_SUMMARY_FILE,
        RF_PER_CLASS_FILE,
        SEED_FILE,
        AGREEMENT_FILE,
        ABLATION_FILE,
        CALIBRATION_FILE,
        CALIBRATED_PREDICTIONS_FILE,
    ]:
        require_file(path)

    # --------------------------------------------------------
    # Load evidence
    # --------------------------------------------------------

    rf_summary = pd.read_csv(
        RF_SUMMARY_FILE
    )

    per_class = pd.read_csv(
        RF_PER_CLASS_FILE
    )

    seed = pd.read_csv(
        SEED_FILE
    )

    agreement = pd.read_csv(
        AGREEMENT_FILE
    )

    ablation = pd.read_csv(
        ABLATION_FILE
    )

    calibration = pd.read_csv(
        CALIBRATION_FILE
    )

    calibrated_predictions = pd.read_csv(
        CALIBRATED_PREDICTIONS_FILE
    )

    rf = get_row(
        rf_summary,
        "Model",
        "RandomForest",
    )

    momentum = get_row(
        rf_summary,
        "Model",
        "Momentum_5D",
    )

    rf_down = per_class[
        (per_class["Model"] == "RandomForest")
        & (per_class["Class"] == "DOWN")
    ].iloc[0]

    momentum_down = per_class[
        (per_class["Model"] == "Momentum_5D")
        & (per_class["Class"] == "DOWN")
    ].iloc[0]

    raw_cal = get_row(
        calibration,
        "Method",
        "Uncalibrated",
    )

    temp_cal = get_row(
        calibration,
        "Method",
        "Temperature",
    )

    # --------------------------------------------------------
    # RF robustness
    # --------------------------------------------------------

    all_seeds_ba = bool(
        (
            seed["Balanced_Accuracy"]
            > momentum["Balanced_Accuracy"]
        ).all()
    )

    all_seeds_f1 = bool(
        (
            seed["Macro_F1"]
            > momentum["Macro_F1"]
        ).all()
    )

    mean_seed_agreement = float(
        agreement[
            "Prediction_Agreement"
        ].mean()
    )

    # --------------------------------------------------------
    # Cross-milestone ablation
    # --------------------------------------------------------

    base_ablation = get_row(
        ablation,
        "Model",
        "Base_RF_Matched",
    )

    augmented = ablation[
        ablation["Model"]
        != "Base_RF_Matched"
    ]

    best_augmented_ba = float(
        augmented[
            "Balanced_Accuracy"
        ].max()
    )

    best_augmented_f1 = float(
        augmented[
            "Macro_F1"
        ].max()
    )

    base_remains_best = bool(
        (
            base_ablation[
                "Balanced_Accuracy"
            ]
            >= best_augmented_ba
        )
        and
        (
            base_ablation[
                "Macro_F1"
            ]
            >= best_augmented_f1
        )
    )

    # --------------------------------------------------------
    # Calibration
    # --------------------------------------------------------

    class_predictions_unchanged = bool(
        (
            calibrated_predictions[
                "Uncalibrated_Prediction"
            ]
            ==
            calibrated_predictions[
                "Temperature_Prediction"
            ]
        ).all()
    )

    calibration_improves_logloss = bool(
        temp_cal["Log_Loss"]
        < raw_cal["Log_Loss"]
    )

    calibration_improves_brier = bool(
        temp_cal["Brier_Score"]
        < raw_cal["Brier_Score"]
    )

    calibration_improves_ece = bool(
        temp_cal["Confidence_ECE"]
        < raw_cal["Confidence_ECE"]
    )

    # ========================================================
    # Final gates
    # ========================================================

    gates = {
        "RF_Beats_Momentum_Balanced_Accuracy":
            bool(
                rf["Balanced_Accuracy"]
                > momentum[
                    "Balanced_Accuracy"
                ]
            ),

        "RF_Beats_Momentum_Macro_F1":
            bool(
                rf["Macro_F1"]
                > momentum["Macro_F1"]
            ),

        "RF_Improves_DOWN_F1":
            bool(
                rf_down["F1"]
                > momentum_down["F1"]
            ),

        "All_RF_Seeds_Beat_Momentum_BA":
            all_seeds_ba,

        "All_RF_Seeds_Beat_Momentum_F1":
            all_seeds_f1,

        "RF_Seed_Agreement_At_Least_95pct":
            mean_seed_agreement >= 0.95,

        "Base_RF_Wins_Cross_Milestone_Ablation":
            base_remains_best,

        "Temperature_Improves_LogLoss":
            calibration_improves_logloss,

        "Temperature_Improves_Brier":
            calibration_improves_brier,

        "Temperature_Improves_ECE":
            calibration_improves_ece,

        "Temperature_Preserves_Class_Predictions":
            class_predictions_unchanged,
    }

    pass_count = sum(
        gates.values()
    )

    all_pass = (
        pass_count == len(gates)
    )

    # ========================================================
    # Console report
    # ========================================================

    print(
        "\n"
        "=========================================="
    )

    print(
        "MILESTONE 3 — FINAL MODEL SELECTION"
    )

    print(
        "=========================================="
    )

    print(
        "\n--- PURGED CLASSIFICATION PERFORMANCE ---"
    )

    print(
        f"Random Forest Balanced Accuracy: "
        f"{rf['Balanced_Accuracy'] * 100:.2f}%"
    )

    print(
        f"Momentum Balanced Accuracy:      "
        f"{momentum['Balanced_Accuracy'] * 100:.2f}%"
    )

    print(
        f"Random Forest Macro F1:          "
        f"{rf['Macro_F1'] * 100:.2f}%"
    )

    print(
        f"Momentum Macro F1:               "
        f"{momentum['Macro_F1'] * 100:.2f}%"
    )

    print(
        f"RF DOWN F1:                      "
        f"{rf_down['F1'] * 100:.2f}%"
    )

    print(
        f"Momentum DOWN F1:                "
        f"{momentum_down['F1'] * 100:.2f}%"
    )

    print(
        "\n--- ROBUSTNESS ---"
    )

    print(
        f"Mean seed prediction agreement: "
        f"{mean_seed_agreement * 100:.2f}%"
    )

    print(
        f"All seeds beat momentum BA: "
        f"{all_seeds_ba}"
    )

    print(
        f"All seeds beat momentum F1: "
        f"{all_seeds_f1}"
    )

    print(
        "\n--- CROSS-MILESTONE ABLATION ---"
    )

    print(
        "Selected feature architecture: "
        "BASE_8_FEATURES"
    )

    print(
        f"Base RF remains best: "
        f"{base_remains_best}"
    )

    print(
        "\n--- PROBABILITY CALIBRATION ---"
    )

    print(
        f"Raw log loss:         "
        f"{raw_cal['Log_Loss']:.5f}"
    )

    print(
        f"Temperature log loss: "
        f"{temp_cal['Log_Loss']:.5f}"
    )

    print(
        f"Raw ECE:              "
        f"{raw_cal['Confidence_ECE']:.5f}"
    )

    print(
        f"Temperature ECE:      "
        f"{temp_cal['Confidence_ECE']:.5f}"
    )

    print(
        "\n--- FINAL ACCEPTANCE GATES ---"
    )

    for gate, passed in gates.items():

        print(
            f"{gate}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

    print(
        f"\nGates passed: "
        f"{pass_count}/{len(gates)}"
    )

    # ========================================================
    # Architecture selection
    # ========================================================

    if all_pass:

        status = (
            "SELECTED_FOR_FINAL_TRAINING"
        )

        classifier = (
            "CANONICAL_RANDOM_FOREST"
        )

        calibration_method = (
            "TEMPERATURE_SCALING"
        )

        use_m1 = False
        use_m2 = False

    else:

        status = (
            "REVIEW_REQUIRED"
        )

        classifier = (
            "CANONICAL_RANDOM_FOREST"
        )

        calibration_method = (
            "TEMPERATURE_SCALING"
            if (
                calibration_improves_logloss
                and
                calibration_improves_ece
            )
            else
            "UNCALIBRATED"
        )

        use_m1 = False
        use_m2 = False

    print(
        "\n"
        "=========================================="
    )

    print(
        "FINAL M3 ARCHITECTURE"
    )

    print(
        "=========================================="
    )

    print(
        f"Status:       {status}"
    )

    print(
        f"Target:       {TARGET}"
    )

    print(
        f"Classifier:   {classifier}"
    )

    print(
        "Features:     BASE_8_FEATURES"
    )

    print(
        f"Calibration:  "
        f"{calibration_method}"
    )

    print(
        "Use M1:       NO"
    )

    print(
        "Use M2:       NO"
    )

    # ========================================================
    # Save
    # ========================================================

    selection_row = {
        "Status":
            status,

        "Target":
            TARGET,

        "Classifier":
            classifier,

        "Feature_Set":
            "BASE_8_FEATURES",

        "Number_of_Features":
            len(FEATURES),

        "Use_M1":
            use_m1,

        "Use_M2":
            use_m2,

        "Calibration":
            calibration_method,

        "Purged_OOS_Accuracy":
            float(
                rf["Accuracy"]
            ),

        "Purged_OOS_Balanced_Accuracy":
            float(
                rf[
                    "Balanced_Accuracy"
                ]
            ),

        "Purged_OOS_Macro_F1":
            float(
                rf["Macro_F1"]
            ),

        "Purged_OOS_DOWN_F1":
            float(
                rf_down["F1"]
            ),

        "Mean_RF_Seed_Agreement":
            mean_seed_agreement,

        "Calibration_Log_Loss":
            float(
                temp_cal[
                    "Log_Loss"
                ]
            ),

        "Calibration_Brier":
            float(
                temp_cal[
                    "Brier_Score"
                ]
            ),

        "Calibration_ECE":
            float(
                temp_cal[
                    "Confidence_ECE"
                ]
            ),

        "Acceptance_Gates_Passed":
            pass_count,

        "Acceptance_Gates_Total":
            len(gates),
    }

    pd.DataFrame(
        [selection_row]
    ).to_csv(
        OUTPUT_CSV,
        index=False,
    )

    metadata = {
        **selection_row,

        "features":
            FEATURES,

        "random_forest_parameters":
            RF_PARAMETERS,

        "acceptance_gates":
            gates,

        "target_definition": {
            "horizon_trading_days": 5,
            "down_threshold": -0.005,
            "up_threshold": 0.005,
            "classes": [
                "DOWN",
                "FLAT",
                "UP",
            ],
        },

        "validation": {
            "method":
                "expanding annual walk-forward",

            "purge_trading_observations":
                5,

            "oos_period":
                "2021-01-01 to 2026-07-31",

            "cross_milestone_ablation":
                "M1/M2 rejected",
        },
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            metadata,
            handle,
            indent=2,
        )

    print(
        "\n--- SAVED ---"
    )

    print(OUTPUT_CSV)
    print(OUTPUT_JSON)

    print(
        "\nFinal model-selection "
        "evaluation complete."
    )


if __name__ == "__main__":
    main()