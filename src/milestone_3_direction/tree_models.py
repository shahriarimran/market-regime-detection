from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)

from sklearn.metrics import (
    accuracy_score,
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

FOLD_FILE = (
    OUTPUT_DIR
    / "tree_model_fold_metrics.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "tree_model_summary.csv"
)

PER_CLASS_FILE = (
    OUTPUT_DIR
    / "tree_model_per_class.csv"
)

PREDICTIONS_FILE = (
    OUTPUT_DIR
    / "tree_model_predictions.csv"
)

RF_IMPORTANCE_FILE = (
    OUTPUT_DIR
    / "random_forest_feature_importance.csv"
)


# ============================================================
# M3 specification
# ============================================================

FEATURE_COLUMNS = [
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


# ============================================================
# Models
# ============================================================

def build_models():

    return {
        "RandomForest":
            RandomForestClassifier(
                n_estimators=500,
                max_depth=6,
                min_samples_leaf=10,
                max_features="sqrt",
                class_weight=None,
                random_state=42,
                n_jobs=-1,
            ),

        "RandomForest_Balanced":
            RandomForestClassifier(
                n_estimators=500,
                max_depth=6,
                min_samples_leaf=10,
                max_features="sqrt",
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            ),

        "HistGradientBoosting":
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=200,
                max_leaf_nodes=15,
                min_samples_leaf=20,
                l2_regularization=1.0,
                early_stopping=False,
                class_weight=None,
                random_state=42,
            ),

        "HistGradientBoosting_Balanced":
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=200,
                max_leaf_nodes=15,
                min_samples_leaf=20,
                l2_regularization=1.0,
                early_stopping=False,
                class_weight="balanced",
                random_state=42,
            ),
    }


# ============================================================
# Metrics
# ============================================================

def present_class_balanced_accuracy(
    y_true,
    y_pred,
):

    recalls = []

    for label in LABELS:

        mask = (
            y_true == label
        )

        support = int(
            mask.sum()
        )

        if support == 0:
            continue

        recall = (
            (
                y_pred[mask]
                == label
            ).mean()
        )

        recalls.append(
            recall
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


def per_class_metrics(
    y_true,
    y_pred,
    year,
    model_name,
):

    (
        precision,
        recall,
        f1,
        support,
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABELS,
        zero_division=0,
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

        rows.append(
            {
                "Test_Year":
                    year,

                "Model":
                    model_name,

                "Class":
                    label,

                "Support":
                    int(s),

                "Precision":
                    p,

                "Recall":
                    (
                        r
                        if s > 0
                        else np.nan
                    ),

                "F1":
                    (
                        f
                        if s > 0
                        else np.nan
                    ),
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
        "MILESTONE 3 — TREE MODELS"
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
    importance_rows = []

    # ========================================================
    # Expanding walk-forward validation
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

        X_train = (
            train[
                FEATURE_COLUMNS
            ]
        )

        y_train = (
            train[
                TARGET
            ]
        )

        X_test = (
            test[
                FEATURE_COLUMNS
            ]
        )

        y_test = (
            test[
                TARGET
            ]
        )

        print(
            "\n"
            "=========================================="
        )

        print(
            f"TREE FOLD: {year}"
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

        fold_predictions = (
            test[
                [
                    "Date",
                    TARGET,
                ]
            ]
            .copy()
        )

        fold_predictions[
            "Test_Year"
        ] = year

        models = build_models()

        # ====================================================
        # Train each model
        # ====================================================

        for (
            model_name,
            model,
        ) in models.items():

            model.fit(
                X_train,
                y_train,
            )

            prediction = (
                model.predict(
                    X_test
                )
            )

            probabilities = (
                model.predict_proba(
                    X_test
                )
            )

            metrics = evaluate(
                y_test,
                prediction,
            )

            confidence = (
                probabilities.max(
                    axis=1
                )
            )

            fold_rows.append(
                {
                    "Test_Year":
                        year,

                    "Model":
                        model_name,

                    "Train_Observations":
                        len(train),

                    "Test_Observations":
                        len(test),

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

                    "Mean_Confidence":
                        confidence.mean(),
                }
            )

            class_rows.extend(
                per_class_metrics(
                    y_test,
                    prediction,
                    year,
                    model_name,
                )
            )

            # -----------------------------------------------
            # Save predictions/probabilities
            # -----------------------------------------------

            fold_predictions[
                f"{model_name}_Prediction"
            ] = prediction

            for (
                class_name,
                probability_column,
            ) in zip(
                model.classes_,
                probabilities.T,
            ):

                fold_predictions[
                    f"{model_name}_P_{class_name}"
                ] = probability_column

            fold_predictions[
                f"{model_name}_Confidence"
            ] = confidence

            # -----------------------------------------------
            # Random Forest feature importance
            # -----------------------------------------------

            if isinstance(
                model,
                RandomForestClassifier,
            ):

                for (
                    feature,
                    importance,
                ) in zip(
                    FEATURE_COLUMNS,
                    model.feature_importances_,
                ):

                    importance_rows.append(
                        {
                            "Test_Year":
                                year,

                            "Model":
                                model_name,

                            "Feature":
                                feature,

                            "Importance":
                                importance,
                        }
                    )

            print(
                f"\n{model_name}:"
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
                f"  Mean confidence: "
                f"{confidence.mean():.4f}"
            )

        prediction_frames.append(
            fold_predictions
        )

    # ========================================================
    # Combine
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

    importance = pd.DataFrame(
        importance_rows
    )

    # ========================================================
    # Pooled OOS results
    # ========================================================

    summary_rows = []

    y_pooled = (
        predictions[
            TARGET
        ]
    )

    for model_name in build_models():

        prediction_column = (
            f"{model_name}_Prediction"
        )

        metrics = evaluate(
            y_pooled,
            predictions[
                prediction_column
            ].values,
        )

        mean_confidence = (
            predictions[
                f"{model_name}_Confidence"
            ].mean()
        )

        summary_rows.append(
            {
                "Model":
                    model_name,

                "OOS_Observations":
                    len(predictions),

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

                "Mean_Confidence":
                    mean_confidence,
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    # ========================================================
    # Print pooled results
    # ========================================================

    print(
        "\n"
        "=========================================="
    )

    print(
        "POOLED OUT-OF-SAMPLE TREE RESULTS"
    )

    print(
        "=========================================="
    )

    display = summary.copy()

    for column in [
        "Accuracy",
        "Balanced_Accuracy",
        "Macro_F1",
        "Mean_Confidence",
    ]:

        display[column] *= 100

    print(
        display
        .round(2)
        .to_string(index=False)
    )

    # ========================================================
    # Annual results
    # ========================================================

    print(
        "\n--- MEAN ANNUAL METRICS ---"
    )

    annual = (
        fold_metrics
        .groupby("Model")[
            [
                "Accuracy",
                "Balanced_Accuracy",
                "Macro_F1",
            ]
        ]
        .mean()
        * 100
    )

    print(
        annual
        .round(2)
        .to_string()
    )

    # ========================================================
    # Momentum comparison
    # ========================================================

    print(
        "\n--- MOMENTUM BENCHMARK ---"
    )

    print(
        "Balanced Accuracy: 50.06%"
    )

    print(
        "Macro F1:          50.03%"
    )

    for _, row in (
        summary.iterrows()
    ):

        balanced_delta = (
            row[
                "Balanced_Accuracy"
            ]
            * 100
            - 50.06
        )

        f1_delta = (
            row[
                "Macro_F1"
            ]
            * 100
            - 50.03
        )

        print(
            f"\n{row['Model']}:"
        )

        print(
            f"  Δ Balanced Accuracy: "
            f"{balanced_delta:+.2f} pp"
        )

        print(
            f"  Δ Macro F1: "
            f"{f1_delta:+.2f} pp"
        )

    # ========================================================
    # Mean RF importance
    # ========================================================

    if not importance.empty:

        print(
            "\n--- RANDOM FOREST FEATURE IMPORTANCE ---"
        )

        mean_importance = (
            importance
            .groupby(
                [
                    "Model",
                    "Feature",
                ]
            )[
                "Importance"
            ]
            .mean()
            .reset_index()
        )

        mean_importance = (
            mean_importance
            .sort_values(
                [
                    "Model",
                    "Importance",
                ],
                ascending=[
                    True,
                    False,
                ],
            )
        )

        print(
            mean_importance
            .round(4)
            .to_string(
                index=False
            )
        )

    # ========================================================
    # Save
    # ========================================================

    fold_metrics.to_csv(
        FOLD_FILE,
        index=False,
    )

    summary.to_csv(
        SUMMARY_FILE,
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

    importance.to_csv(
        RF_IMPORTANCE_FILE,
        index=False,
    )

    print(
        "\n--- SAVED ---"
    )

    print(FOLD_FILE)
    print(SUMMARY_FILE)
    print(PER_CLASS_FILE)
    print(PREDICTIONS_FILE)
    print(RF_IMPORTANCE_FILE)

    print(
        "\nTree-model evaluation complete."
    )


if __name__ == "__main__":
    main()