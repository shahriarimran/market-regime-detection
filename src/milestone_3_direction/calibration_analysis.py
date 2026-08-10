from pathlib import Path

import numpy as np
import pandas as pd

from scipy.optimize import minimize_scalar

from sklearn.metrics import (
    accuracy_score,
    log_loss,
)


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT
    / "outputs"
    / "milestone_3"
    / "validation"
    / "purged_rf_predictions.csv"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "milestone_3"
    / "validation"
)

FOLD_FILE = (
    OUTPUT_DIR
    / "calibration_fold_metrics.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "calibration_summary.csv"
)

TEMPERATURE_FILE = (
    OUTPUT_DIR
    / "calibration_temperature_by_year.csv"
)

PREDICTION_FILE = (
    OUTPUT_DIR
    / "calibrated_rf_predictions.csv"
)

RELIABILITY_FILE = (
    OUTPUT_DIR
    / "calibration_reliability_bins.csv"
)


# ============================================================
# Specification
# ============================================================

LABELS = [
    "DOWN",
    "FLAT",
    "UP",
]

PROBABILITY_COLUMNS = [
    "RandomForest_P_DOWN",
    "RandomForest_P_FLAT",
    "RandomForest_P_UP",
]

CALIBRATION_TEST_YEARS = [
    2022,
    2023,
    2024,
    2025,
    2026,
]

N_BINS = 10

EPSILON = 1e-12


# ============================================================
# Helpers
# ============================================================

def softmax(logits):

    logits = np.asarray(
        logits,
        dtype=float,
    )

    shifted = (
        logits
        - np.max(
            logits,
            axis=1,
            keepdims=True,
        )
    )

    exp_values = np.exp(
        shifted
    )

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

        # Optimize log(T) so T is always positive.
        temperature = np.exp(
            log_temperature
        )

        calibrated = (
            temperature_scale(
                probabilities,
                temperature,
            )
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
            "xatol": 1e-6,
        },
    )

    if not result.success:
        raise RuntimeError(
            "Temperature optimization failed."
        )

    return float(
        np.exp(
            result.x
        )
    )


# ============================================================
# Probabilistic metrics
# ============================================================

def multiclass_brier_score(
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

    one_hot = np.zeros_like(
        probabilities
    )

    for index, label in enumerate(
        LABELS
    ):

        one_hot[
            :,
            index,
        ] = (
            y_true == label
        ).astype(float)

    # Multiclass Brier:
    # mean squared probability error summed
    # across all classes for each observation.
    return float(
        np.mean(
            np.sum(
                (
                    probabilities
                    - one_hot
                )
                ** 2,
                axis=1,
            )
        )
    )


def confidence_ece(
    y_true,
    probabilities,
    n_bins=N_BINS,
):

    y_true = np.asarray(
        y_true
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    predicted_index = (
        probabilities.argmax(
            axis=1
        )
    )

    predicted_labels = np.array(
        LABELS
    )[
        predicted_index
    ]

    confidence = (
        probabilities.max(
            axis=1
        )
    )

    correct = (
        predicted_labels
        == y_true
    )

    edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    ece = 0.0

    for i in range(
        n_bins
    ):

        lower = edges[i]
        upper = edges[i + 1]

        if i == (
            n_bins - 1
        ):

            mask = (
                (confidence >= lower)
                & (confidence <= upper)
            )

        else:

            mask = (
                (confidence >= lower)
                & (confidence < upper)
            )

        count = int(
            mask.sum()
        )

        if count == 0:
            continue

        bin_accuracy = (
            correct[mask].mean()
        )

        bin_confidence = (
            confidence[mask].mean()
        )

        ece += (
            count
            / len(y_true)
            * abs(
                bin_accuracy
                - bin_confidence
            )
        )

    return float(
        ece
    )


def classwise_ece(
    y_true,
    probabilities,
    n_bins=N_BINS,
):

    y_true = np.asarray(
        y_true
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    class_results = {}

    for class_index, label in enumerate(
        LABELS
    ):

        predicted_prob = (
            probabilities[
                :,
                class_index,
            ]
        )

        actual = (
            y_true == label
        ).astype(float)

        ece = 0.0

        for i in range(
            n_bins
        ):

            lower = edges[i]
            upper = edges[i + 1]

            if i == (
                n_bins - 1
            ):

                mask = (
                    (predicted_prob >= lower)
                    & (predicted_prob <= upper)
                )

            else:

                mask = (
                    (predicted_prob >= lower)
                    & (predicted_prob < upper)
                )

            count = int(
                mask.sum()
            )

            if count == 0:
                continue

            observed_rate = (
                actual[mask].mean()
            )

            mean_probability = (
                predicted_prob[
                    mask
                ].mean()
            )

            ece += (
                count
                / len(y_true)
                * abs(
                    observed_rate
                    - mean_probability
                )
            )

        class_results[
            label
        ] = float(
            ece
        )

    return class_results


def mean_entropy(
    probabilities,
):

    probabilities = np.clip(
        np.asarray(
            probabilities,
            dtype=float,
        ),
        EPSILON,
        1.0,
    )

    entropy = -np.sum(
        probabilities
        * np.log(
            probabilities
        ),
        axis=1,
    )

    return float(
        entropy.mean()
    )


def evaluate_probabilities(
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

    predictions = np.array(
        LABELS
    )[
        probabilities.argmax(
            axis=1
        )
    ]

    class_ece = (
        classwise_ece(
            y_true,
            probabilities,
        )
    )

    return {
        "Accuracy":
            accuracy_score(
                y_true,
                predictions,
            ),

        "Log_Loss":
            log_loss(
                y_true,
                probabilities,
                labels=LABELS,
            ),

        "Brier_Score":
            multiclass_brier_score(
                y_true,
                probabilities,
            ),

        "Confidence_ECE":
            confidence_ece(
                y_true,
                probabilities,
            ),

        "Mean_Classwise_ECE":
            np.mean(
                list(
                    class_ece.values()
                )
            ),

        "DOWN_ECE":
            class_ece[
                "DOWN"
            ],

        "FLAT_ECE":
            class_ece[
                "FLAT"
            ],

        "UP_ECE":
            class_ece[
                "UP"
            ],

        "Mean_Confidence":
            probabilities.max(
                axis=1
            ).mean(),

        "Mean_Entropy":
            mean_entropy(
                probabilities
            ),
    }


# ============================================================
# Reliability bins
# ============================================================

def reliability_bins(
    y_true,
    probabilities,
    method,
    test_year,
):

    y_true = np.asarray(
        y_true
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    predicted_index = (
        probabilities.argmax(
            axis=1
        )
    )

    predicted_labels = (
        np.array(
            LABELS
        )[
            predicted_index
        ]
    )

    confidence = (
        probabilities.max(
            axis=1
        )
    )

    correct = (
        predicted_labels
        == y_true
    )

    edges = np.linspace(
        0.0,
        1.0,
        N_BINS + 1,
    )

    rows = []

    for i in range(
        N_BINS
    ):

        lower = edges[i]
        upper = edges[i + 1]

        if i == (
            N_BINS - 1
        ):

            mask = (
                (confidence >= lower)
                & (confidence <= upper)
            )

        else:

            mask = (
                (confidence >= lower)
                & (confidence < upper)
            )

        count = int(
            mask.sum()
        )

        if count == 0:
            continue

        rows.append(
            {
                "Test_Year":
                    test_year,

                "Method":
                    method,

                "Bin_Lower":
                    lower,

                "Bin_Upper":
                    upper,

                "Observations":
                    count,

                "Mean_Confidence":
                    confidence[
                        mask
                    ].mean(),

                "Observed_Accuracy":
                    correct[
                        mask
                    ].mean(),

                "Calibration_Gap":
                    (
                        correct[
                            mask
                        ].mean()
                        - confidence[
                            mask
                        ].mean()
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

    required = [
        "Date",
        "Test_Year",
        "Actual",
        *PROBABILITY_COLUMNS,
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: "
            f"{missing}"
        )

    probability_sum = (
        df[
            PROBABILITY_COLUMNS
        ]
        .sum(
            axis=1
        )
    )

    max_sum_error = (
        probability_sum
        .sub(1.0)
        .abs()
        .max()
    )

    if max_sum_error > 1e-6:
        raise ValueError(
            "RF probabilities do not "
            "sum to 1."
        )

    print(
        "\n"
        "=========================================="
    )

    print(
        "MILESTONE 3 — PROBABILITY CALIBRATION"
    )

    print(
        "=========================================="
    )

    print(
        f"Purged OOS observations: "
        f"{len(df):,}"
    )

    print(
        f"Period: "
        f"{df['Date'].min().date()} "
        f"to "
        f"{df['Date'].max().date()}"
    )

    print(
        "Calibration method: "
        "chronological expanding "
        "temperature scaling"
    )

    fold_rows = []
    temperature_rows = []
    prediction_frames = []
    reliability_rows = []

    # ========================================================
    # Expanding chronological calibration
    # ========================================================

    for year in (
        CALIBRATION_TEST_YEARS
    ):

        calibration = (
            df[
                df["Test_Year"]
                < year
            ]
            .copy()
        )

        test = (
            df[
                df["Test_Year"]
                == year
            ]
            .copy()
        )

        if calibration.empty:
            continue

        if test.empty:
            continue

        y_cal = (
            calibration[
                "Actual"
            ]
            .to_numpy()
        )

        p_cal = (
            calibration[
                PROBABILITY_COLUMNS
            ]
            .to_numpy(
                dtype=float
            )
        )

        y_test = (
            test[
                "Actual"
            ]
            .to_numpy()
        )

        p_raw = (
            test[
                PROBABILITY_COLUMNS
            ]
            .to_numpy(
                dtype=float
            )
        )

        temperature = (
            fit_temperature(
                y_cal,
                p_cal,
            )
        )

        p_scaled = (
            temperature_scale(
                p_raw,
                temperature,
            )
        )

        raw_metrics = (
            evaluate_probabilities(
                y_test,
                p_raw,
            )
        )

        scaled_metrics = (
            evaluate_probabilities(
                y_test,
                p_scaled,
            )
        )

        print(
            "\n"
            "=========================================="
        )

        print(
            f"CALIBRATION FOLD: {year}"
        )

        print(
            "=========================================="
        )

        print(
            f"Calibration observations: "
            f"{len(calibration):,}"
        )

        print(
            f"Test observations: "
            f"{len(test):,}"
        )

        print(
            f"Temperature: "
            f"{temperature:.4f}"
        )

        print(
            "\nUncalibrated:"
        )

        print(
            f"  Log loss: "
            f"{raw_metrics['Log_Loss']:.4f}"
        )

        print(
            f"  Brier: "
            f"{raw_metrics['Brier_Score']:.4f}"
        )

        print(
            f"  Confidence ECE: "
            f"{raw_metrics['Confidence_ECE']:.4f}"
        )

        print(
            "\nTemperature scaled:"
        )

        print(
            f"  Log loss: "
            f"{scaled_metrics['Log_Loss']:.4f}"
        )

        print(
            f"  Brier: "
            f"{scaled_metrics['Brier_Score']:.4f}"
        )

        print(
            f"  Confidence ECE: "
            f"{scaled_metrics['Confidence_ECE']:.4f}"
        )

        for (
            method,
            metrics,
        ) in [
            (
                "Uncalibrated",
                raw_metrics,
            ),
            (
                "Temperature",
                scaled_metrics,
            ),
        ]:

            fold_rows.append(
                {
                    "Test_Year":
                        year,

                    "Method":
                        method,

                    "Calibration_Observations":
                        len(
                            calibration
                        ),

                    "Test_Observations":
                        len(
                            test
                        ),

                    "Temperature":
                        (
                            1.0
                            if method
                            == "Uncalibrated"
                            else temperature
                        ),

                    **metrics,
                }
            )

        temperature_rows.append(
            {
                "Test_Year":
                    year,

                "Calibration_Observations":
                    len(
                        calibration
                    ),

                "Temperature":
                    temperature,
            }
        )

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        frame = pd.DataFrame(
            {
                "Date":
                    test[
                        "Date"
                    ].values,

                "Test_Year":
                    year,

                "Actual":
                    y_test,
            }
        )

        raw_prediction = (
            np.array(
                LABELS
            )[
                p_raw.argmax(
                    axis=1
                )
            ]
        )

        scaled_prediction = (
            np.array(
                LABELS
            )[
                p_scaled.argmax(
                    axis=1
                )
            ]
        )

        frame[
            "Uncalibrated_Prediction"
        ] = raw_prediction

        frame[
            "Temperature_Prediction"
        ] = scaled_prediction

        for index, label in enumerate(
            LABELS
        ):

            frame[
                f"Uncalibrated_P_{label}"
            ] = p_raw[
                :,
                index,
            ]

            frame[
                f"Temperature_P_{label}"
            ] = p_scaled[
                :,
                index,
            ]

        frame[
            "Temperature"
        ] = temperature

        prediction_frames.append(
            frame
        )

        reliability_rows.extend(
            reliability_bins(
                y_test,
                p_raw,
                "Uncalibrated",
                year,
            )
        )

        reliability_rows.extend(
            reliability_bins(
                y_test,
                p_scaled,
                "Temperature",
                year,
            )
        )

    # ========================================================
    # Combine
    # ========================================================

    folds = pd.DataFrame(
        fold_rows
    )

    temperatures = pd.DataFrame(
        temperature_rows
    )

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    reliability = pd.DataFrame(
        reliability_rows
    )

    # ========================================================
    # Pooled matched comparison
    # ========================================================

    y_pooled = (
        predictions[
            "Actual"
        ]
        .to_numpy()
    )

    raw_pooled = (
        predictions[
            [
                "Uncalibrated_P_DOWN",
                "Uncalibrated_P_FLAT",
                "Uncalibrated_P_UP",
            ]
        ]
        .to_numpy(
            dtype=float
        )
    )

    temp_pooled = (
        predictions[
            [
                "Temperature_P_DOWN",
                "Temperature_P_FLAT",
                "Temperature_P_UP",
            ]
        ]
        .to_numpy(
            dtype=float
        )
    )

    raw_metrics = (
        evaluate_probabilities(
            y_pooled,
            raw_pooled,
        )
    )

    temp_metrics = (
        evaluate_probabilities(
            y_pooled,
            temp_pooled,
        )
    )

    summary = pd.DataFrame(
        [
            {
                "Method":
                    "Uncalibrated",

                "OOS_Observations":
                    len(
                        predictions
                    ),

                **raw_metrics,
            },

            {
                "Method":
                    "Temperature",

                "OOS_Observations":
                    len(
                        predictions
                    ),

                **temp_metrics,
            },
        ]
    )

    baseline = summary[
        summary["Method"]
        == "Uncalibrated"
    ].iloc[0]

    summary[
        "Delta_Log_Loss"
    ] = (
        summary[
            "Log_Loss"
        ]
        - baseline[
            "Log_Loss"
        ]
    )

    summary[
        "Delta_Brier"
    ] = (
        summary[
            "Brier_Score"
        ]
        - baseline[
            "Brier_Score"
        ]
    )

    summary[
        "Delta_Confidence_ECE"
    ] = (
        summary[
            "Confidence_ECE"
        ]
        - baseline[
            "Confidence_ECE"
        ]
    )

    # ========================================================
    # Print summary
    # ========================================================

    print(
        "\n"
        "=========================================="
    )

    print(
        "POOLED CALIBRATION RESULTS"
    )

    print(
        "=========================================="
    )

    print(
        summary
        .round(5)
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Annual win counts
    # --------------------------------------------------------

    pivot_log = (
        folds.pivot(
            index="Test_Year",
            columns="Method",
            values="Log_Loss",
        )
    )

    pivot_brier = (
        folds.pivot(
            index="Test_Year",
            columns="Method",
            values="Brier_Score",
        )
    )

    pivot_ece = (
        folds.pivot(
            index="Test_Year",
            columns="Method",
            values="Confidence_ECE",
        )
    )

    log_wins = int(
        (
            pivot_log[
                "Temperature"
            ]
            < pivot_log[
                "Uncalibrated"
            ]
        ).sum()
    )

    brier_wins = int(
        (
            pivot_brier[
                "Temperature"
            ]
            < pivot_brier[
                "Uncalibrated"
            ]
        ).sum()
    )

    ece_wins = int(
        (
            pivot_ece[
                "Temperature"
            ]
            < pivot_ece[
                "Uncalibrated"
            ]
        ).sum()
    )

    print(
        "\n--- TEMPORAL CALIBRATION WINS ---"
    )

    print(
        f"Log-loss wins: "
        f"{log_wins}/"
        f"{len(CALIBRATION_TEST_YEARS)}"
    )

    print(
        f"Brier wins:   "
        f"{brier_wins}/"
        f"{len(CALIBRATION_TEST_YEARS)}"
    )

    print(
        f"ECE wins:     "
        f"{ece_wins}/"
        f"{len(CALIBRATION_TEST_YEARS)}"
    )

    # ========================================================
    # Selection gates
    # ========================================================

    raw = raw_metrics
    scaled = temp_metrics

    # Temperature scaling should not alter
    # class argmax for positive T.
    labels_unchanged = bool(
        (
            predictions[
                "Uncalibrated_Prediction"
            ]
            == predictions[
                "Temperature_Prediction"
            ]
        ).all()
    )

    gates = {
        "Pooled_Log_Loss_Improves":
            scaled[
                "Log_Loss"
            ]
            < raw[
                "Log_Loss"
            ],

        "Pooled_Brier_Improves":
            scaled[
                "Brier_Score"
            ]
            < raw[
                "Brier_Score"
            ],

        "Pooled_ECE_Improves":
            scaled[
                "Confidence_ECE"
            ]
            < raw[
                "Confidence_ECE"
            ],

        "Annual_LogLoss_Wins_At_Least_3":
            log_wins >= 3,

        "Class_Predictions_Unchanged":
            labels_unchanged,
    }

    print(
        "\n--- CALIBRATION SELECTION GATES ---"
    )

    for (
        gate,
        passed,
    ) in gates.items():

        print(
            f"{gate}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

    pass_count = sum(
        gates.values()
    )

    print(
        f"\nGates passed: "
        f"{pass_count}/"
        f"{len(gates)}"
    )

    if pass_count == len(
        gates
    ):

        recommendation = (
            "TEMPERATURE_SCALING"
        )

    elif (
        gates[
            "Pooled_Log_Loss_Improves"
        ]
        and gates[
            "Pooled_ECE_Improves"
        ]
    ):

        recommendation = (
            "TEMPERATURE_SCALING_"
            "PROVISIONAL"
        )

    else:

        recommendation = (
            "KEEP_UNCALIBRATED_RF"
        )

    print(
        "\nRecommended probability architecture:"
    )

    print(
        recommendation
    )

    # ========================================================
    # Save
    # ========================================================

    folds.to_csv(
        FOLD_FILE,
        index=False,
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    temperatures.to_csv(
        TEMPERATURE_FILE,
        index=False,
    )

    predictions.to_csv(
        PREDICTION_FILE,
        index=False,
    )

    reliability.to_csv(
        RELIABILITY_FILE,
        index=False,
    )

    print(
        "\n--- SAVED ---"
    )

    print(FOLD_FILE)
    print(SUMMARY_FILE)
    print(TEMPERATURE_FILE)
    print(PREDICTION_FILE)
    print(RELIABILITY_FILE)

    print(
        "\nProbability calibration "
        "analysis complete."
    )


if __name__ == "__main__":
    main()