from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)


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

MOMENTUM_FILE = (
    VALIDATION_DIR
    / "naive_baseline_predictions.csv"
)

TREE_FILE = (
    VALIDATION_DIR
    / "tree_model_predictions.csv"
)

YEARLY_OUTPUT = (
    VALIDATION_DIR
    / "rf_vs_momentum_yearly.csv"
)

PER_CLASS_OUTPUT = (
    VALIDATION_DIR
    / "rf_vs_momentum_per_class.csv"
)

PAIRED_OUTPUT = (
    VALIDATION_DIR
    / "rf_vs_momentum_paired_outcomes.csv"
)

CONFIDENCE_OUTPUT = (
    VALIDATION_DIR
    / "random_forest_confidence_diagnostics.csv"
)

SUMMARY_OUTPUT = (
    VALIDATION_DIR
    / "rf_vs_momentum_summary.csv"
)


# ============================================================
# Specification
# ============================================================

TARGET = "Target_5D_0p5pct"

LABELS = [
    "DOWN",
    "FLAT",
    "UP",
]

RF_PRED = "RandomForest_Prediction"

RF_CONF = "RandomForest_Confidence"

MOMENTUM_PRED = "Momentum_5D"


# ============================================================
# Metrics
# ============================================================

def present_class_balanced_accuracy(
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

    if not recalls:
        return np.nan

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
            present_class_balanced_accuracy(
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
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Load predictions
    # --------------------------------------------------------

    momentum = pd.read_csv(
        MOMENTUM_FILE,
        parse_dates=["Date"],
    )

    tree = pd.read_csv(
        TREE_FILE,
        parse_dates=["Date"],
    )

    required_momentum = {
        "Date",
        TARGET,
        "Test_Year",
        MOMENTUM_PRED,
    }

    required_tree = {
        "Date",
        TARGET,
        "Test_Year",
        RF_PRED,
        RF_CONF,
    }

    missing_momentum = (
        required_momentum
        - set(momentum.columns)
    )

    missing_tree = (
        required_tree
        - set(tree.columns)
    )

    if missing_momentum:

        raise ValueError(
            "Missing momentum columns: "
            f"{sorted(missing_momentum)}"
        )

    if missing_tree:

        raise ValueError(
            "Missing tree columns: "
            f"{sorted(missing_tree)}"
        )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    merged = pd.merge(
        momentum[
            [
                "Date",
                TARGET,
                "Test_Year",
                MOMENTUM_PRED,
            ]
        ],
        tree[
            [
                "Date",
                TARGET,
                "Test_Year",
                RF_PRED,
                RF_CONF,
            ]
        ],
        on=[
            "Date",
            TARGET,
            "Test_Year",
        ],
        how="inner",
        validate="one_to_one",
    )

    merged = (
        merged
        .sort_values("Date")
        .reset_index(drop=True)
    )

    print(
        "\n"
        "=========================================="
    )

    print(
        "MILESTONE 3 — MODEL STABILITY"
    )

    print(
        "=========================================="
    )

    print(
        f"Matched OOS observations: "
        f"{len(merged):,}"
    )

    print(
        f"Period: "
        f"{merged['Date'].min().date()} "
        f"to "
        f"{merged['Date'].max().date()}"
    )

    # ========================================================
    # Pooled comparison
    # ========================================================

    y = merged[TARGET]

    rf_metrics = evaluate(
        y,
        merged[RF_PRED],
    )

    momentum_metrics = evaluate(
        y,
        merged[MOMENTUM_PRED],
    )

    print(
        "\n--- POOLED COMPARISON ---"
    )

    print(
        "\nRandom Forest:"
    )

    for metric, value in (
        rf_metrics.items()
    ):

        print(
            f"  {metric}: "
            f"{value * 100:.2f}%"
        )

    print(
        "\nMomentum-5D:"
    )

    for metric, value in (
        momentum_metrics.items()
    ):

        print(
            f"  {metric}: "
            f"{value * 100:.2f}%"
        )

    print(
        "\nRandom Forest advantage:"
    )

    print(
        "  Balanced Accuracy: "
        f"{(
            rf_metrics['Balanced_Accuracy']
            - momentum_metrics['Balanced_Accuracy']
        ) * 100:+.2f} pp"
    )

    print(
        "  Macro F1: "
        f"{(
            rf_metrics['Macro_F1']
            - momentum_metrics['Macro_F1']
        ) * 100:+.2f} pp"
    )

    # ========================================================
    # Annual comparison
    # ========================================================

    yearly_rows = []

    for year, group in (
        merged.groupby(
            "Test_Year"
        )
    ):

        rf = evaluate(
            group[TARGET],
            group[RF_PRED],
        )

        momentum_eval = evaluate(
            group[TARGET],
            group[MOMENTUM_PRED],
        )

        yearly_rows.append(
            {
                "Test_Year":
                    int(year),

                "Observations":
                    len(group),

                "RF_Accuracy":
                    rf["Accuracy"],

                "Momentum_Accuracy":
                    momentum_eval[
                        "Accuracy"
                    ],

                "RF_Balanced_Accuracy":
                    rf[
                        "Balanced_Accuracy"
                    ],

                "Momentum_Balanced_Accuracy":
                    momentum_eval[
                        "Balanced_Accuracy"
                    ],

                "Delta_Balanced_Accuracy":
                    (
                        rf[
                            "Balanced_Accuracy"
                        ]
                        - momentum_eval[
                            "Balanced_Accuracy"
                        ]
                    ),

                "RF_Macro_F1":
                    rf[
                        "Macro_F1"
                    ],

                "Momentum_Macro_F1":
                    momentum_eval[
                        "Macro_F1"
                    ],

                "Delta_Macro_F1":
                    (
                        rf[
                            "Macro_F1"
                        ]
                        - momentum_eval[
                            "Macro_F1"
                        ]
                    ),
            }
        )

    yearly = pd.DataFrame(
        yearly_rows
    )

    print(
        "\n--- YEARLY COMPARISON ---"
    )

    display_yearly = (
        yearly.copy()
    )

    metric_columns = [
        "RF_Accuracy",
        "Momentum_Accuracy",
        "RF_Balanced_Accuracy",
        "Momentum_Balanced_Accuracy",
        "Delta_Balanced_Accuracy",
        "RF_Macro_F1",
        "Momentum_Macro_F1",
        "Delta_Macro_F1",
    ]

    for column in metric_columns:

        display_yearly[
            column
        ] *= 100

    print(
        display_yearly
        .round(2)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # Win counts
    # ========================================================

    ba_wins = int(
        (
            yearly[
                "Delta_Balanced_Accuracy"
            ] > 0
        ).sum()
    )

    f1_wins = int(
        (
            yearly[
                "Delta_Macro_F1"
            ] > 0
        ).sum()
    )

    n_years = len(
        yearly
    )

    print(
        "\n--- TEMPORAL WIN RATE ---"
    )

    print(
        f"Balanced Accuracy wins: "
        f"{ba_wins}/{n_years}"
    )

    print(
        f"Macro F1 wins: "
        f"{f1_wins}/{n_years}"
    )

    # ========================================================
    # Per-class comparison
    # ========================================================

    per_class_rows = []

    for model_name, prediction_col in [
        (
            "RandomForest",
            RF_PRED,
        ),
        (
            "Momentum_5D",
            MOMENTUM_PRED,
        ),
    ]:

        (
            precision,
            recall,
            f1,
            support,
        ) = (
            precision_recall_fscore_support(
                merged[TARGET],
                merged[
                    prediction_col
                ],
                labels=LABELS,
                zero_division=0,
            )
        )

        for (
            label,
            p,
            r,
            f,
            s,
        ) in zip(
            LABELS,
            precision,
            recall,
            f1,
            support,
        ):

            per_class_rows.append(
                {
                    "Model":
                        model_name,

                    "Class":
                        label,

                    "Support":
                        int(s),

                    "Precision":
                        p,

                    "Recall":
                        r,

                    "F1":
                        f,
                }
            )

    per_class = pd.DataFrame(
        per_class_rows
    )

    print(
        "\n--- POOLED PER-CLASS METRICS ---"
    )

    per_class_display = (
        per_class.copy()
    )

    for column in [
        "Precision",
        "Recall",
        "F1",
    ]:

        per_class_display[
            column
        ] *= 100

    print(
        per_class_display
        .round(2)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # Model agreement
    # ========================================================

    merged[
        "Models_Agree"
    ] = (
        merged[
            RF_PRED
        ]
        == merged[
            MOMENTUM_PRED
        ]
    )

    merged[
        "RF_Correct"
    ] = (
        merged[
            RF_PRED
        ]
        == merged[
            TARGET
        ]
    )

    merged[
        "Momentum_Correct"
    ] = (
        merged[
            MOMENTUM_PRED
        ]
        == merged[
            TARGET
        ]
    )

    agreement_rate = (
        merged[
            "Models_Agree"
        ].mean()
    )

    print(
        "\n--- MODEL AGREEMENT ---"
    )

    print(
        f"Prediction agreement: "
        f"{agreement_rate * 100:.2f}%"
    )

    # ========================================================
    # Paired outcomes
    # ========================================================

    both_correct = int(
        (
            merged[
                "RF_Correct"
            ]
            & merged[
                "Momentum_Correct"
            ]
        ).sum()
    )

    rf_only = int(
        (
            merged[
                "RF_Correct"
            ]
            & ~merged[
                "Momentum_Correct"
            ]
        ).sum()
    )

    momentum_only = int(
        (
            ~merged[
                "RF_Correct"
            ]
            & merged[
                "Momentum_Correct"
            ]
        ).sum()
    )

    both_wrong = int(
        (
            ~merged[
                "RF_Correct"
            ]
            & ~merged[
                "Momentum_Correct"
            ]
        ).sum()
    )

    paired = pd.DataFrame(
        [
            {
                "Outcome":
                    "Both_Correct",

                "Count":
                    both_correct,
            },
            {
                "Outcome":
                    "RF_Only_Correct",

                "Count":
                    rf_only,
            },
            {
                "Outcome":
                    "Momentum_Only_Correct",

                "Count":
                    momentum_only,
            },
            {
                "Outcome":
                    "Both_Wrong",

                "Count":
                    both_wrong,
            },
        ]
    )

    paired[
        "Share"
    ] = (
        paired["Count"]
        / len(merged)
    )

    print(
        "\n--- PAIRED PREDICTION OUTCOMES ---"
    )

    paired_display = (
        paired.copy()
    )

    paired_display[
        "Share"
    ] *= 100

    print(
        paired_display
        .round(2)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # RF confidence diagnostics
    # ========================================================

    bins = [
        0.0,
        0.50,
        0.60,
        0.70,
        0.80,
        1.01,
    ]

    labels = [
        "<50%",
        "50-60%",
        "60-70%",
        "70-80%",
        "80%+",
    ]

    merged[
        "RF_Confidence_Bin"
    ] = pd.cut(
        merged[
            RF_CONF
        ],
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
    )

    confidence = (
        merged
        .groupby(
            "RF_Confidence_Bin",
            observed=False,
        )
        .agg(
            Observations=(
                TARGET,
                "size",
            ),
            Mean_Confidence=(
                RF_CONF,
                "mean",
            ),
            Accuracy=(
                "RF_Correct",
                "mean",
            ),
        )
        .reset_index()
    )

    print(
        "\n--- RANDOM FOREST CONFIDENCE ---"
    )

    confidence_display = (
        confidence.copy()
    )

    confidence_display[
        "Mean_Confidence"
    ] *= 100

    confidence_display[
        "Accuracy"
    ] *= 100

    print(
        confidence_display
        .round(2)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # Selection gates
    # ========================================================

    pooled_ba_delta = (
        rf_metrics[
            "Balanced_Accuracy"
        ]
        - momentum_metrics[
            "Balanced_Accuracy"
        ]
    )

    pooled_f1_delta = (
        rf_metrics[
            "Macro_F1"
        ]
        - momentum_metrics[
            "Macro_F1"
        ]
    )

    gates = {
        "Pooled_BA_Improvement_1pp":
            pooled_ba_delta
            >= 0.01,

        "Pooled_F1_Improvement_1pp":
            pooled_f1_delta
            >= 0.01,

        "Annual_BA_Wins_At_Least_4":
            ba_wins >= 4,

        "Annual_F1_Wins_At_Least_3":
            f1_wins >= 3,
    }

    print(
        "\n--- PROVISIONAL RF SELECTION GATES ---"
    )

    for gate, passed in (
        gates.items()
    ):

        print(
            f"{gate}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

    pass_count = sum(
        gates.values()
    )

    print(
        f"\nGates passed: "
        f"{pass_count}/{len(gates)}"
    )

    # ========================================================
    # Summary
    # ========================================================

    summary = pd.DataFrame(
        [
            {
                "RF_Pooled_Accuracy":
                    rf_metrics[
                        "Accuracy"
                    ],

                "Momentum_Pooled_Accuracy":
                    momentum_metrics[
                        "Accuracy"
                    ],

                "RF_Pooled_Balanced_Accuracy":
                    rf_metrics[
                        "Balanced_Accuracy"
                    ],

                "Momentum_Pooled_Balanced_Accuracy":
                    momentum_metrics[
                        "Balanced_Accuracy"
                    ],

                "Delta_Balanced_Accuracy":
                    pooled_ba_delta,

                "RF_Pooled_Macro_F1":
                    rf_metrics[
                        "Macro_F1"
                    ],

                "Momentum_Pooled_Macro_F1":
                    momentum_metrics[
                        "Macro_F1"
                    ],

                "Delta_Macro_F1":
                    pooled_f1_delta,

                "BA_Annual_Wins":
                    ba_wins,

                "F1_Annual_Wins":
                    f1_wins,

                "Prediction_Agreement":
                    agreement_rate,

                "Selection_Gates_Passed":
                    pass_count,

                "Selection_Gates_Total":
                    len(gates),
            }
        ]
    )

    # ========================================================
    # Save
    # ========================================================

    yearly.to_csv(
        YEARLY_OUTPUT,
        index=False,
    )

    per_class.to_csv(
        PER_CLASS_OUTPUT,
        index=False,
    )

    paired.to_csv(
        PAIRED_OUTPUT,
        index=False,
    )

    confidence.to_csv(
        CONFIDENCE_OUTPUT,
        index=False,
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    print(
        "\n--- SAVED ---"
    )

    print(YEARLY_OUTPUT)
    print(PER_CLASS_OUTPUT)
    print(PAIRED_OUTPUT)
    print(CONFIDENCE_OUTPUT)
    print(SUMMARY_OUTPUT)

    print(
        "\nModel stability evaluation complete."
    )


if __name__ == "__main__":
    main()