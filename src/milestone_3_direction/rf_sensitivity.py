from pathlib import Path

import itertools

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "usdtry_direction_features.csv"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "milestone_3"
    / "validation"
)

RESULT_FILE = (
    OUTPUT_DIR
    / "rf_sensitivity_results.csv"
)

SEED_FILE = (
    OUTPUT_DIR
    / "rf_seed_stability.csv"
)

AGREEMENT_FILE = (
    OUTPUT_DIR
    / "rf_seed_prediction_agreement.csv"
)


# ============================================================
# Specification
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
]

TARGET = "Target_5D_0p5pct"

LABELS = [
    "DOWN",
    "FLAT",
    "UP",
]

TEST_YEARS = [
    2021,
    2022,
    2023,
    2024,
    2025,
    2026,
]

SEEDS = [
    7,
    21,
    42,
    77,
    101,
]


# Nearby configurations around the selected RF.
CONFIGS = {
    "Canonical": {
        "max_depth": 6,
        "min_samples_leaf": 10,
    },

    "Shallower": {
        "max_depth": 4,
        "min_samples_leaf": 10,
    },

    "Deeper": {
        "max_depth": 8,
        "min_samples_leaf": 10,
    },

    "Smaller_Leaf": {
        "max_depth": 6,
        "min_samples_leaf": 5,
    },

    "Larger_Leaf": {
        "max_depth": 6,
        "min_samples_leaf": 20,
    },
}


MOMENTUM_BA = 0.5006
MOMENTUM_F1 = 0.5003


# ============================================================
# Metrics
# ============================================================

def balanced_accuracy_present(
    y_true,
    y_pred,
):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    recalls = []

    for label in LABELS:

        mask = (
            y_true == label
        )

        if mask.sum() == 0:
            continue

        recalls.append(
            (
                y_pred[mask]
                == label
            ).mean()
        )

    return float(
        np.mean(recalls)
    )


def evaluate(
    y_true,
    y_pred,
):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    present_labels = [
        label
        for label in LABELS
        if np.any(
            y_true == label
        )
    ]

    return {
        "Accuracy":
            accuracy_score(
                y_true,
                y_pred,
            ),

        "Balanced_Accuracy":
            balanced_accuracy_present(
                y_true,
                y_pred,
            ),

        "Macro_F1":
            f1_score(
                y_true,
                y_pred,
                labels=present_labels,
                average="macro",
                zero_division=0,
            ),
    }


# ============================================================
# Walk-forward experiment
# ============================================================

def run_experiment(
    df,
    config_name,
    config,
    seed,
):

    prediction_frames = []

    for year in TEST_YEARS:

        train = (
            df[
                df["Year"] < year
            ]
        )

        test = (
            df[
                df["Year"] == year
            ]
        )

        if test.empty:
            continue

        model = (
            RandomForestClassifier(
                n_estimators=500,

                max_depth=config[
                    "max_depth"
                ],

                min_samples_leaf=config[
                    "min_samples_leaf"
                ],

                max_features="sqrt",

                class_weight=None,

                random_state=seed,

                n_jobs=-1,
            )
        )

        model.fit(
            train[FEATURES],
            train[TARGET],
        )

        prediction = (
            model.predict(
                test[FEATURES]
            )
        )

        frame = pd.DataFrame(
            {
                "Date":
                    test["Date"].values,

                "Test_Year":
                    year,

                "Actual":
                    test[TARGET].values,

                "Prediction":
                    prediction,
            }
        )

        prediction_frames.append(
            frame
        )

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    metrics = evaluate(
        predictions["Actual"],
        predictions["Prediction"],
    )

    return (
        predictions,
        metrics,
    )


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["Date"],
    )

    df = (
        df
        .sort_values("Date")
        .reset_index(drop=True)
    )

    df["Year"] = (
        df["Date"].dt.year
    )

    print(
        "\n"
        "=========================================="
    )

    print(
        "MILESTONE 3 — RF SENSITIVITY"
    )

    print(
        "=========================================="
    )

    print(
        f"Observations: {len(df):,}"
    )

    rows = []

    canonical_predictions = {}

    # ========================================================
    # Run all nearby configurations and seeds
    # ========================================================

    for (
        config_name,
        config,
    ) in CONFIGS.items():

        for seed in SEEDS:

            (
                predictions,
                metrics,
            ) = run_experiment(
                df,
                config_name,
                config,
                seed,
            )

            rows.append(
                {
                    "Configuration":
                        config_name,

                    "Seed":
                        seed,

                    "Max_Depth":
                        config[
                            "max_depth"
                        ],

                    "Min_Samples_Leaf":
                        config[
                            "min_samples_leaf"
                        ],

                    "Accuracy":
                        metrics[
                            "Accuracy"
                        ],

                    "Balanced_Accuracy":
                        metrics[
                            "Balanced_Accuracy"
                        ],

                    "Macro_F1":
                        metrics[
                            "Macro_F1"
                        ],

                    "Delta_BA_vs_Momentum":
                        (
                            metrics[
                                "Balanced_Accuracy"
                            ]
                            - MOMENTUM_BA
                        ),

                    "Delta_F1_vs_Momentum":
                        (
                            metrics[
                                "Macro_F1"
                            ]
                            - MOMENTUM_F1
                        ),
                }
            )

            if (
                config_name
                == "Canonical"
            ):

                canonical_predictions[
                    seed
                ] = (
                    predictions[
                        "Prediction"
                    ]
                    .reset_index(
                        drop=True
                    )
                )

            print(
                f"{config_name:15s} "
                f"seed={seed:3d} | "
                f"BA="
                f"{metrics['Balanced_Accuracy']:.4f} | "
                f"F1="
                f"{metrics['Macro_F1']:.4f}"
            )

    results = pd.DataFrame(
        rows
    )

    # ========================================================
    # Configuration summary
    # ========================================================

    summary = (
        results
        .groupby(
            "Configuration"
        )
        .agg(
            Mean_Accuracy=(
                "Accuracy",
                "mean",
            ),

            Std_Accuracy=(
                "Accuracy",
                "std",
            ),

            Mean_Balanced_Accuracy=(
                "Balanced_Accuracy",
                "mean",
            ),

            Std_Balanced_Accuracy=(
                "Balanced_Accuracy",
                "std",
            ),

            Min_Balanced_Accuracy=(
                "Balanced_Accuracy",
                "min",
            ),

            Mean_Macro_F1=(
                "Macro_F1",
                "mean",
            ),

            Std_Macro_F1=(
                "Macro_F1",
                "std",
            ),

            Min_Macro_F1=(
                "Macro_F1",
                "min",
            ),
        )
        .reset_index()
    )

    print(
        "\n"
        "=========================================="
    )

    print(
        "CONFIGURATION SENSITIVITY SUMMARY"
    )

    print(
        "=========================================="
    )

    display = (
        summary.copy()
    )

    metric_columns = [
        column
        for column in display.columns
        if column
        != "Configuration"
    ]

    for column in metric_columns:

        display[column] *= 100

    print(
        display
        .round(2)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # Canonical seed stability
    # ========================================================

    canonical = (
        results[
            results[
                "Configuration"
            ]
            == "Canonical"
        ]
        .copy()
    )

    print(
        "\n--- CANONICAL SEED STABILITY ---"
    )

    canonical_display = (
        canonical[
            [
                "Seed",
                "Accuracy",
                "Balanced_Accuracy",
                "Macro_F1",
                "Delta_BA_vs_Momentum",
                "Delta_F1_vs_Momentum",
            ]
        ]
        .copy()
    )

    for column in [
        "Accuracy",
        "Balanced_Accuracy",
        "Macro_F1",
        "Delta_BA_vs_Momentum",
        "Delta_F1_vs_Momentum",
    ]:

        canonical_display[
            column
        ] *= 100

    print(
        canonical_display
        .round(2)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # Pairwise prediction agreement across seeds
    # ========================================================

    agreement_rows = []

    for seed_a, seed_b in (
        itertools.combinations(
            SEEDS,
            2,
        )
    ):

        pred_a = (
            canonical_predictions[
                seed_a
            ]
        )

        pred_b = (
            canonical_predictions[
                seed_b
            ]
        )

        agreement = (
            pred_a.values
            == pred_b.values
        ).mean()

        agreement_rows.append(
            {
                "Seed_A":
                    seed_a,

                "Seed_B":
                    seed_b,

                "Prediction_Agreement":
                    agreement,
            }
        )

    agreements = pd.DataFrame(
        agreement_rows
    )

    print(
        "\n--- CANONICAL PAIRWISE AGREEMENT ---"
    )

    print(
        f"Mean agreement: "
        f"{agreements['Prediction_Agreement'].mean() * 100:.2f}%"
    )

    print(
        f"Minimum agreement: "
        f"{agreements['Prediction_Agreement'].min() * 100:.2f}%"
    )

    # ========================================================
    # Robustness gates
    # ========================================================

    canonical_all_ba = bool(
        (
            canonical[
                "Balanced_Accuracy"
            ]
            > MOMENTUM_BA
        ).all()
    )

    canonical_all_f1 = bool(
        (
            canonical[
                "Macro_F1"
            ]
            > MOMENTUM_F1
        ).all()
    )

    nearby_mean_ba = bool(
        (
            summary[
                "Mean_Balanced_Accuracy"
            ]
            > MOMENTUM_BA
        ).sum()
        >= 3
    )

    nearby_mean_f1 = bool(
        (
            summary[
                "Mean_Macro_F1"
            ]
            > MOMENTUM_F1
        ).sum()
        >= 3
    )

    agreement_gate = bool(
        agreements[
            "Prediction_Agreement"
        ].mean()
        >= 0.95
    )

    gates = {
        "All_Canonical_Seeds_Beat_Momentum_BA":
            canonical_all_ba,

        "All_Canonical_Seeds_Beat_Momentum_F1":
            canonical_all_f1,

        "At_Least_3_Configs_Beat_Momentum_BA":
            nearby_mean_ba,

        "At_Least_3_Configs_Beat_Momentum_F1":
            nearby_mean_f1,

        "Canonical_Seed_Agreement_At_Least_95pct":
            agreement_gate,
    }

    print(
        "\n--- RF ROBUSTNESS GATES ---"
    )

    for gate, passed in (
        gates.items()
    ):

        print(
            f"{gate}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

    print(
        f"\nGates passed: "
        f"{sum(gates.values())}"
        f"/{len(gates)}"
    )

    # ========================================================
    # Save
    # ========================================================

    results.to_csv(
        RESULT_FILE,
        index=False,
    )

    canonical.to_csv(
        SEED_FILE,
        index=False,
    )

    agreements.to_csv(
        AGREEMENT_FILE,
        index=False,
    )

    print(
        "\n--- SAVED ---"
    )

    print(
        RESULT_FILE
    )

    print(
        SEED_FILE
    )

    print(
        AGREEMENT_FILE
    )

    print(
        "\nRF sensitivity evaluation complete."
    )


if __name__ == "__main__":
    main()