from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DIRECTION_FILE = (
    ROOT
    / "data"
    / "processed"
    / "usdtry_direction_features.csv"
)

M1_FILE = (
    ROOT
    / "outputs"
    / "milestone_3"
    / "cross_milestone"
    / "m1_oos_features.csv"
)

M2_FILE = (
    ROOT
    / "outputs"
    / "milestone_3"
    / "cross_milestone"
    / "m2_oos_features.csv"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "milestone_3"
    / "validation"
)


# ============================================================
# Specification
# ============================================================

TARGET = "Target_5D_0p5pct"

HORIZON = 5

LABELS = [
    "DOWN",
    "FLAT",
    "UP",
]

BASE_FEATURES = [
    "Return_1D",
    "Return_5D",
    "Volatility_5D",
    "Volatility_20D",
    "Volatility_60D",
    "MA_Distance_20D",
    "MA_Slope_20D",
    "Drawdown_60D",
]

M1_FEATURES = [
    "M1_P_LOW",
    "M1_P_ELEVATED",
    "M1_P_STRESS",
]

M2_FEATURES = [
    "M2_Baseline_Score",
    "M2_IF_Score",
    "M2_IF_Training_Percentile",
    "M2_Baseline_Anomaly_Flag",
]

CANONICAL_TEST_YEARS = [
    2021,
    2022,
    2023,
    2024,
    2025,
    2026,
]

ABLATION_TEST_YEARS = [
    2022,
    2023,
    2024,
    2025,
    2026,
]


# ============================================================
# Model
# ============================================================

def build_rf():

    return RandomForestClassifier(
        n_estimators=500,
        max_depth=6,
        min_samples_leaf=10,
        max_features="sqrt",
        class_weight=None,
        random_state=42,
        n_jobs=-1,
    )


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


def purge_training_boundary(
    train,
):

    train = (
        train
        .sort_values("Date")
        .copy()
    )

    if len(train) <= HORIZON:
        raise ValueError(
            "Insufficient training observations "
            "after purge."
        )

    # The final HORIZON rows have targets whose
    # 5-day forward-return window reaches into
    # the next chronological fold.
    return (
        train
        .iloc[:-HORIZON]
        .copy()
    )


def momentum_prediction(
    return_5d,
):

    values = np.asarray(
        return_5d,
        dtype=float,
    )

    return np.where(
        values > 0.005,
        "UP",
        np.where(
            values < -0.005,
            "DOWN",
            "FLAT",
        ),
    )


# ============================================================
# Canonical RF validation
# ============================================================

def canonical_validation(
    df,
):

    fold_rows = []
    prediction_frames = []

    print(
        "\n"
        "=========================================="
    )

    print(
        "PURGED CANONICAL RF VALIDATION"
    )

    print(
        "=========================================="
    )

    for year in CANONICAL_TEST_YEARS:

        test = (
            df[
                df["Year"] == year
            ]
            .copy()
        )

        if test.empty:
            continue

        train = (
            df[
                df["Date"]
                < test["Date"].min()
            ]
            .copy()
        )

        before_purge = len(train)

        train = purge_training_boundary(
            train
        )

        model = build_rf()

        model.fit(
            train[BASE_FEATURES],
            train[TARGET],
        )

        rf_pred = model.predict(
            test[BASE_FEATURES]
        )

        rf_prob = model.predict_proba(
            test[BASE_FEATURES]
        )

        momentum_pred = (
            momentum_prediction(
                test["Return_5D"]
            )
        )

        rf_metrics = evaluate(
            test[TARGET],
            rf_pred,
        )

        momentum_metrics = evaluate(
            test[TARGET],
            momentum_pred,
        )

        print(
            f"\nFold {year}"
        )

        print(
            f"Train before purge: "
            f"{before_purge:,}"
        )

        print(
            f"Train after purge:  "
            f"{len(train):,}"
        )

        print(
            f"Purged:             "
            f"{before_purge - len(train)}"
        )

        print(
            f"Test:               "
            f"{len(test):,}"
        )

        print(
            "\nRandom Forest:"
        )

        print(
            f"  Accuracy: "
            f"{rf_metrics['Accuracy']:.4f}"
        )

        print(
            f"  Balanced Accuracy: "
            f"{rf_metrics['Balanced_Accuracy']:.4f}"
        )

        print(
            f"  Macro F1: "
            f"{rf_metrics['Macro_F1']:.4f}"
        )

        print(
            "\nMomentum:"
        )

        print(
            f"  Accuracy: "
            f"{momentum_metrics['Accuracy']:.4f}"
        )

        print(
            f"  Balanced Accuracy: "
            f"{momentum_metrics['Balanced_Accuracy']:.4f}"
        )

        print(
            f"  Macro F1: "
            f"{momentum_metrics['Macro_F1']:.4f}"
        )

        for model_name, metrics in [
            (
                "RandomForest",
                rf_metrics,
            ),
            (
                "Momentum_5D",
                momentum_metrics,
            ),
        ]:

            fold_rows.append(
                {
                    "Test_Year":
                        year,

                    "Model":
                        model_name,

                    "Train_Before_Purge":
                        before_purge,

                    "Train_After_Purge":
                        len(train),

                    "Purged_Observations":
                        before_purge
                        - len(train),

                    "Test_Observations":
                        len(test),

                    **metrics,
                }
            )

        prediction = pd.DataFrame(
            {
                "Date":
                    test["Date"].values,

                "Test_Year":
                    year,

                "Actual":
                    test[TARGET].values,

                "RandomForest_Prediction":
                    rf_pred,

                "Momentum_5D_Prediction":
                    momentum_pred,

                "RandomForest_Confidence":
                    rf_prob.max(
                        axis=1
                    ),
            }
        )

        for (
            class_name,
            probability,
        ) in zip(
            model.classes_,
            rf_prob.T,
        ):

            prediction[
                f"RandomForest_P_{class_name}"
            ] = probability

        prediction_frames.append(
            prediction
        )

    folds = pd.DataFrame(
        fold_rows
    )

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Pooled metrics
    # --------------------------------------------------------

    pooled_rows = []

    for (
        name,
        column,
    ) in [
        (
            "RandomForest",
            "RandomForest_Prediction",
        ),
        (
            "Momentum_5D",
            "Momentum_5D_Prediction",
        ),
    ]:

        metrics = evaluate(
            predictions["Actual"],
            predictions[column],
        )

        pooled_rows.append(
            {
                "Model":
                    name,

                "OOS_Observations":
                    len(predictions),

                **metrics,
            }
        )

    pooled = pd.DataFrame(
        pooled_rows
    )

    print(
        "\n"
        "=========================================="
    )

    print(
        "PURGED POOLED RF VS MOMENTUM"
    )

    print(
        "=========================================="
    )

    display = pooled.copy()

    for column in [
        "Accuracy",
        "Balanced_Accuracy",
        "Macro_F1",
    ]:

        display[column] *= 100

    print(
        display
        .round(2)
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Per-class pooled
    # --------------------------------------------------------

    per_class_rows = []

    for (
        name,
        column,
    ) in [
        (
            "RandomForest",
            "RandomForest_Prediction",
        ),
        (
            "Momentum_5D",
            "Momentum_5D_Prediction",
        ),
    ]:

        (
            precision,
            recall,
            f1,
            support,
        ) = precision_recall_fscore_support(
            predictions["Actual"],
            predictions[column],
            labels=LABELS,
            zero_division=0,
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
                        name,

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
        "\n--- PURGED POOLED PER-CLASS ---"
    )

    pc_display = (
        per_class.copy()
    )

    for column in [
        "Precision",
        "Recall",
        "F1",
    ]:

        pc_display[column] *= 100

    print(
        pc_display
        .round(2)
        .to_string(
            index=False
        )
    )

    return (
        folds,
        pooled,
        per_class,
        predictions,
    )


# ============================================================
# Purged cross-milestone ablation
# ============================================================

def cross_milestone_validation(
    df,
):

    if not M1_FILE.exists():
        raise FileNotFoundError(
            M1_FILE
        )

    if not M2_FILE.exists():
        raise FileNotFoundError(
            M2_FILE
        )

    m1 = pd.read_csv(
        M1_FILE,
        parse_dates=["Date"],
    )

    m2 = pd.read_csv(
        M2_FILE,
        parse_dates=["Date"],
    )

    merged = (
        df
        .merge(
            m1,
            on="Date",
            how="left",
        )
        .merge(
            m2,
            on="Date",
            how="left",
        )
    )

    merged[
        "M2_Baseline_Anomaly_Flag"
    ] = (
        merged[
            "M2_Baseline_Anomaly"
        ]
        .map(
            {
                "NORMAL": 0,
                "ANOMALOUS": 1,
            }
        )
    )

    required = (
        BASE_FEATURES
        + M1_FEATURES
        + M2_FEATURES
        + [TARGET]
    )

    matched = (
        merged
        .dropna(
            subset=required
        )
        .sort_values("Date")
        .reset_index(drop=True)
    )

    print(
        "\n"
        "=========================================="
    )

    print(
        "PURGED CROSS-MILESTONE ABLATION"
    )

    print(
        "=========================================="
    )

    print(
        f"Matched observations: "
        f"{len(matched):,}"
    )

    print(
        f"Matched period: "
        f"{matched['Date'].min().date()} "
        f"to "
        f"{matched['Date'].max().date()}"
    )

    model_features = {
        "Base_RF_Matched":
            BASE_FEATURES,

        "Base_RF_Plus_M1":
            BASE_FEATURES
            + M1_FEATURES,

        "Base_RF_Plus_M2":
            BASE_FEATURES
            + M2_FEATURES,

        "Base_RF_Plus_M1_M2":
            BASE_FEATURES
            + M1_FEATURES
            + M2_FEATURES,
    }

    fold_rows = []
    prediction_frames = []

    for year in ABLATION_TEST_YEARS:

        test = (
            matched[
                matched["Year"] == year
            ]
            .copy()
        )

        if test.empty:
            continue

        train = (
            matched[
                matched["Date"]
                < test["Date"].min()
            ]
            .copy()
        )

        before_purge = len(train)

        train = purge_training_boundary(
            train
        )

        print(
            f"\nAblation fold {year}: "
            f"train "
            f"{before_purge:,}"
            f" -> "
            f"{len(train):,} "
            f"after purge; "
            f"test {len(test):,}"
        )

        for (
            model_name,
            features,
        ) in model_features.items():

            model = build_rf()

            model.fit(
                train[features],
                train[TARGET],
            )

            pred = model.predict(
                test[features]
            )

            metrics = evaluate(
                test[TARGET],
                pred,
            )

            fold_rows.append(
                {
                    "Test_Year":
                        year,

                    "Model":
                        model_name,

                    "Train_Before_Purge":
                        before_purge,

                    "Train_After_Purge":
                        len(train),

                    "Test_Observations":
                        len(test),

                    **metrics,
                }
            )

            prediction_frames.append(
                pd.DataFrame(
                    {
                        "Date":
                            test["Date"].values,

                        "Test_Year":
                            year,

                        "Model":
                            model_name,

                        "Actual":
                            test[
                                TARGET
                            ].values,

                        "Predicted":
                            pred,
                    }
                )
            )

    folds = pd.DataFrame(
        fold_rows
    )

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    summary_rows = []

    for (
        model_name,
        group,
    ) in predictions.groupby(
        "Model",
        sort=False,
    ):

        metrics = evaluate(
            group["Actual"],
            group["Predicted"],
        )

        (
            precision,
            recall,
            f1,
            support,
        ) = precision_recall_fscore_support(
            group["Actual"],
            group["Predicted"],
            labels=LABELS,
            zero_division=0,
        )

        down_index = (
            LABELS.index(
                "DOWN"
            )
        )

        summary_rows.append(
            {
                "Model":
                    model_name,

                "OOS_Observations":
                    len(group),

                **metrics,

                "DOWN_Precision":
                    precision[
                        down_index
                    ],

                "DOWN_Recall":
                    recall[
                        down_index
                    ],

                "DOWN_F1":
                    f1[
                        down_index
                    ],

                "DOWN_Support":
                    int(
                        support[
                            down_index
                        ]
                    ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    base = (
        summary[
            summary["Model"]
            == "Base_RF_Matched"
        ]
        .iloc[0]
    )

    summary[
        "Delta_BA_vs_Base"
    ] = (
        summary[
            "Balanced_Accuracy"
        ]
        - base[
            "Balanced_Accuracy"
        ]
    )

    summary[
        "Delta_F1_vs_Base"
    ] = (
        summary[
            "Macro_F1"
        ]
        - base[
            "Macro_F1"
        ]
    )

    summary[
        "Delta_DOWN_Recall_vs_Base"
    ] = (
        summary[
            "DOWN_Recall"
        ]
        - base[
            "DOWN_Recall"
        ]
    )

    print(
        "\n--- PURGED CROSS-MILESTONE SUMMARY ---"
    )

    display = summary.copy()

    percentage_columns = [
        "Accuracy",
        "Balanced_Accuracy",
        "Macro_F1",
        "DOWN_Precision",
        "DOWN_Recall",
        "DOWN_F1",
        "Delta_BA_vs_Base",
        "Delta_F1_vs_Base",
        "Delta_DOWN_Recall_vs_Base",
    ]

    for column in percentage_columns:

        display[column] *= 100

    print(
        display
        .round(2)
        .to_string(
            index=False
        )
    )

    return (
        folds,
        summary,
        predictions,
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
        DIRECTION_FILE,
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
        "MILESTONE 3 — PURGED VALIDATION AUDIT"
    )

    print(
        "=========================================="
    )

    print(
        f"Target horizon: "
        f"{HORIZON} trading days"
    )

    print(
        f"Observations: "
        f"{len(df):,}"
    )

    # --------------------------------------------------------
    # Canonical RF
    # --------------------------------------------------------

    (
        canonical_folds,
        canonical_summary,
        canonical_per_class,
        canonical_predictions,
    ) = canonical_validation(
        df
    )

    canonical_folds.to_csv(
        OUTPUT_DIR
        / "purged_rf_vs_momentum_fold_metrics.csv",
        index=False,
    )

    canonical_summary.to_csv(
        OUTPUT_DIR
        / "purged_rf_vs_momentum_summary.csv",
        index=False,
    )

    canonical_per_class.to_csv(
        OUTPUT_DIR
        / "purged_rf_vs_momentum_per_class.csv",
        index=False,
    )

    canonical_predictions.to_csv(
        OUTPUT_DIR
        / "purged_rf_predictions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Cross-milestone ablation
    # --------------------------------------------------------

    (
        ablation_folds,
        ablation_summary,
        ablation_predictions,
    ) = cross_milestone_validation(
        df
    )

    ablation_folds.to_csv(
        OUTPUT_DIR
        / "purged_cross_milestone_fold_metrics.csv",
        index=False,
    )

    ablation_summary.to_csv(
        OUTPUT_DIR
        / "purged_cross_milestone_summary.csv",
        index=False,
    )

    ablation_predictions.to_csv(
        OUTPUT_DIR
        / "purged_cross_milestone_predictions.csv",
        index=False,
    )

    print(
        "\n--- SAVED ---"
    )

    for path in [
        OUTPUT_DIR
        / "purged_rf_vs_momentum_fold_metrics.csv",

        OUTPUT_DIR
        / "purged_rf_vs_momentum_summary.csv",

        OUTPUT_DIR
        / "purged_rf_vs_momentum_per_class.csv",

        OUTPUT_DIR
        / "purged_rf_predictions.csv",

        OUTPUT_DIR
        / "purged_cross_milestone_fold_metrics.csv",

        OUTPUT_DIR
        / "purged_cross_milestone_summary.csv",

        OUTPUT_DIR
        / "purged_cross_milestone_predictions.csv",
    ]:

        print(path)

    print(
        "\nPurged validation audit complete."
    )


if __name__ == "__main__":
    main()