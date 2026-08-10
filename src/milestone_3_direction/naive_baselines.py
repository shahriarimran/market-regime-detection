from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
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

FOLD_METRICS_FILE = (
    OUTPUT_DIR
    / "naive_baseline_fold_metrics.csv"
)

PER_CLASS_FILE = (
    OUTPUT_DIR
    / "naive_baseline_per_class.csv"
)

PREDICTIONS_FILE = (
    OUTPUT_DIR
    / "naive_baseline_predictions.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "naive_baseline_summary.csv"
)


# ============================================================
# M3 specification
# ============================================================

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

MOMENTUM_THRESHOLD = 0.005


# ============================================================
# Baselines
# ============================================================

def majority_baseline(
    y_train,
    n_test,
):

    majority_class = (
        y_train
        .value_counts()
        .idxmax()
    )

    predictions = np.full(
        n_test,
        majority_class,
        dtype=object,
    )

    return (
        predictions,
        majority_class,
    )


def prior_random_baseline(
    y_train,
    n_test,
    seed,
):

    counts = (
        y_train
        .value_counts()
        .reindex(
            LABELS,
            fill_value=0,
        )
        .astype(float)
    )

    probabilities = (
        counts / counts.sum()
    )

    rng = np.random.default_rng(
        seed
    )

    predictions = rng.choice(
        LABELS,
        size=n_test,
        p=probabilities.values,
    )

    return (
        predictions,
        probabilities,
    )


def momentum_baseline(
    return_5d,
):

    predictions = np.full(
        len(return_5d),
        "FLAT",
        dtype=object,
    )

    predictions[
        return_5d.values
        > MOMENTUM_THRESHOLD
    ] = "UP"

    predictions[
        return_5d.values
        < -MOMENTUM_THRESHOLD
    ] = "DOWN"

    return predictions


# ============================================================
# Metrics
# ============================================================

def evaluate_predictions(
    y_true,
    y_pred,
):

    # Classes actually observable in this test year.
    present_labels = [
        label
        for label in LABELS
        if (y_true == label).any()
    ]

    return {
        "Accuracy":
            accuracy_score(
                y_true,
                y_pred,
            ),

        "Balanced_Accuracy":
            balanced_accuracy_score(
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


def per_class_metrics(
    y_true,
    y_pred,
    year,
    model_name,
):

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=LABELS,
            zero_division=0,
        )
    )

    rows = []

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

        # If the class does not exist in the
        # test year, its recall/F1 are not
        # economically meaningful.
        if s == 0:

            recall_value = np.nan
            f1_value = np.nan

        else:

            recall_value = r
            f1_value = f

        rows.append(
            {
                "Test_Year":
                    year,

                "Baseline":
                    model_name,

                "Class":
                    label,

                "Support":
                    int(s),

                "Precision":
                    p,

                "Recall":
                    recall_value,

                "F1":
                    f1_value,
            }
        )

    return rows


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
        "MILESTONE 3 — NAIVE BASELINES"
    )

    print(
        "=========================================="
    )

    print(
        f"Observations: {len(df):,}"
    )

    print(
        f"Period: "
        f"{df['Date'].min().date()} "
        f"to "
        f"{df['Date'].max().date()}"
    )

    fold_rows = []
    class_rows = []
    prediction_frames = []

    # ========================================================
    # Expanding walk-forward folds
    # ========================================================

    for year in TEST_YEARS:

        train = (
            df[
                df["Year"] < year
            ]
            .copy()
        )

        test = (
            df[
                df["Year"] == year
            ]
            .copy()
        )

        if test.empty:
            continue

        y_train = train[TARGET]
        y_test = test[TARGET]

        print(
            "\n"
            "=========================================="
        )

        print(
            f"BASELINE FOLD: {year}"
        )

        print(
            "=========================================="
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

        print(
            "\nTest class shares:"
        )

        shares = (
            y_test
            .value_counts(
                normalize=True
            )
            .reindex(
                LABELS,
                fill_value=0,
            )
            * 100
        )

        print(
            shares
            .round(2)
            .to_string()
        )

        # ----------------------------------------------------
        # Baseline 1: majority class
        # ----------------------------------------------------

        (
            majority_pred,
            majority_class,
        ) = majority_baseline(
            y_train,
            len(test),
        )

        # ----------------------------------------------------
        # Baseline 2: training-prior random
        # ----------------------------------------------------

        (
            random_pred,
            training_priors,
        ) = prior_random_baseline(
            y_train,
            len(test),
            seed=42 + year,
        )

        # ----------------------------------------------------
        # Baseline 3: 5D momentum persistence
        # ----------------------------------------------------

        momentum_pred = (
            momentum_baseline(
                test["Return_5D"]
            )
        )

        predictions = {
            "Majority":
                majority_pred,

            "Prior_Random":
                random_pred,

            "Momentum_5D":
                momentum_pred,
        }

        # ----------------------------------------------------
        # Evaluate
        # ----------------------------------------------------

        fold_prediction_df = (
            test[
                [
                    "Date",
                    TARGET,
                ]
            ]
            .copy()
        )

        fold_prediction_df[
            "Test_Year"
        ] = year

        for (
            name,
            prediction,
        ) in predictions.items():

            metrics = (
                evaluate_predictions(
                    y_test,
                    prediction,
                )
            )

            fold_rows.append(
                {
                    "Test_Year":
                        year,

                    "Baseline":
                        name,

                    "Train_Observations":
                        len(train),

                    "Test_Observations":
                        len(test),

                    **metrics,
                }
            )

            class_rows.extend(
                per_class_metrics(
                    y_test,
                    prediction,
                    year,
                    name,
                )
            )

            fold_prediction_df[
                name
            ] = prediction

            print(
                f"\n{name}:"
            )

            print(
                f"  Accuracy: "
                f"{metrics['Accuracy']:.4f}"
            )

            print(
                f"  Balanced accuracy: "
                f"{metrics['Balanced_Accuracy']:.4f}"
            )

            print(
                f"  Macro F1: "
                f"{metrics['Macro_F1']:.4f}"
            )

        print(
            "\nTraining majority class:",
            majority_class,
        )

        print(
            "Training priors:"
        )

        print(
            (
                training_priors
                * 100
            )
            .round(2)
            .to_string()
        )

        prediction_frames.append(
            fold_prediction_df
        )

    # ========================================================
    # Save fold results
    # ========================================================

    fold_metrics = pd.DataFrame(
        fold_rows
    )

    per_class = pd.DataFrame(
        class_rows
    )

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    # ========================================================
    # Pooled out-of-sample evaluation
    #
    # This is especially important because some individual
    # years contain no DOWN observations.
    # ========================================================

    pooled_rows = []

    y_pooled = predictions[TARGET]

    for baseline in [
        "Majority",
        "Prior_Random",
        "Momentum_5D",
    ]:

        metrics = evaluate_predictions(
            y_pooled,
            predictions[baseline],
        )

        pooled_rows.append(
            {
                "Baseline":
                    baseline,

                "OOS_Observations":
                    len(predictions),

                **metrics,
            }
        )

    summary = pd.DataFrame(
        pooled_rows
    )

    # ========================================================
    # Print aggregate results
    # ========================================================

    print(
        "\n"
        "=========================================="
    )

    print(
        "POOLED OUT-OF-SAMPLE BASELINES"
    )

    print(
        "=========================================="
    )

    display = (
        summary.copy()
    )

    for column in [
        "Accuracy",
        "Balanced_Accuracy",
        "Macro_F1",
    ]:

        display[column] = (
            display[column]
            * 100
        )

    print(
        display
        .round(2)
        .to_string(index=False)
    )

    # Annual averages
    annual_summary = (
        fold_metrics
        .groupby("Baseline")[
            [
                "Accuracy",
                "Balanced_Accuracy",
                "Macro_F1",
            ]
        ]
        .mean()
    )

    print(
        "\n--- MEAN ANNUAL METRICS ---"
    )

    print(
        (
            annual_summary
            * 100
        )
        .round(2)
        .to_string()
    )

    # ========================================================
    # Save
    # ========================================================

    fold_metrics.to_csv(
        FOLD_METRICS_FILE,
        index=False,
    )

    per_class.to_csv(
        PER_CLASS_FILE,
        index=False,
    )

    predictions.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    print(
        "\n--- SAVED ---"
    )

    print(
        FOLD_METRICS_FILE
    )

    print(
        PER_CLASS_FILE
    )

    print(
        PREDICTIONS_FILE
    )

    print(
        SUMMARY_FILE
    )

    print(
        "\nNaive baseline evaluation complete."
    )


if __name__ == "__main__":
    main()