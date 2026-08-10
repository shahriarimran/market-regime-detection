from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest


# ============================================================
# Paths
# ============================================================

M2_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    M2_DIR
    / "data"
    / "usdtry_anomaly_features.csv"
)

OUTPUT_DIR = (
    M2_DIR
    / "outputs"
    / "stability_sensitivity"
)

SEED_FILE = (
    OUTPUT_DIR
    / "if_seed_stability.csv"
)

SEED_PAIRWISE_FILE = (
    OUTPUT_DIR
    / "if_seed_pairwise_jaccard.csv"
)

MAX_SAMPLES_FILE = (
    OUTPUT_DIR
    / "if_max_samples_sensitivity.csv"
)

THRESHOLD_FILE = (
    OUTPUT_DIR
    / "if_threshold_sensitivity.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "if_stability_summary.csv"
)


# ============================================================
# Features
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


# ============================================================
# Baseline thresholds
# ============================================================

RETURN_Z_THRESHOLD = 4.0
VOL_RATIO_THRESHOLD = 2.0
DRAWDOWN_CHANGE_THRESHOLD = 0.05
ABS_RETURN_THRESHOLD = 0.04


# ============================================================
# Walk-forward
# ============================================================

TEST_YEARS = [
    2021,
    2022,
    2023,
    2024,
    2025,
    2026,
]


# ============================================================
# Sensitivity settings
# ============================================================

SEEDS = [
    0,
    7,
    21,
    42,
    99,
]

MAX_SAMPLES_VALUES = [
    256,
    512,
    1024,
]

# Multipliers modify the training-derived target anomaly rate.
#
# 0.75 = stricter threshold
# 1.00 = current specification
# 1.25 = more permissive threshold
THRESHOLD_MULTIPLIERS = [
    0.75,
    1.00,
    1.25,
]

N_ESTIMATORS = 500


# ============================================================
# Load
# ============================================================

def load_data():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["Date"],
    )

    required = {
        "Date",
        "Absolute_Return_1D",
        *FEATURES,
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    df = (
        df
        .sort_values("Date")
        .reset_index(drop=True)
    )

    X = (
        df[FEATURES]
        .to_numpy(dtype=float)
    )

    if not np.isfinite(X).all():

        raise ValueError(
            "Feature matrix contains NaN or infinity."
        )

    return df


# ============================================================
# Baseline
# ============================================================

def baseline_flag(df):

    return (
        (
            df["Return_ZScore_20D"].abs()
            >= RETURN_Z_THRESHOLD
        )
        |
        (
            df["Volatility_Ratio_5D_60D"]
            >= VOL_RATIO_THRESHOLD
        )
        |
        (
            df["Drawdown_Change_5D"].abs()
            >= DRAWDOWN_CHANGE_THRESHOLD
        )
        |
        (
            df["Absolute_Return_1D"]
            >= ABS_RETURN_THRESHOLD
        )
    ).to_numpy(dtype=bool)


# ============================================================
# Jaccard
# ============================================================

def jaccard(a, b):

    a = np.asarray(
        a,
        dtype=bool,
    )

    b = np.asarray(
        b,
        dtype=bool,
    )

    union = (
        a | b
    ).sum()

    if union == 0:

        return 1.0

    return float(
        (
            a & b
        ).sum()
        / union
    )


# ============================================================
# Fit and score IF
# ============================================================

def fit_and_score(
    train,
    test,
    seed,
    max_samples,
    threshold_multiplier,
):

    X_train = (
        train[FEATURES]
        .to_numpy(dtype=float)
    )

    X_test = (
        test[FEATURES]
        .to_numpy(dtype=float)
    )

    train_baseline = (
        baseline_flag(train)
    )

    base_rate = float(
        train_baseline.mean()
    )

    target_rate = float(
        np.clip(
            base_rate
            * threshold_multiplier,
            0.005,
            0.25,
        )
    )

    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        max_samples=min(
            max_samples,
            len(train),
        ),
        contamination="auto",
        max_features=1.0,
        bootstrap=False,
        random_state=seed,
        n_jobs=-1,
    )

    model.fit(
        X_train
    )

    train_scores = (
        -model.score_samples(
            X_train
        )
    )

    threshold = float(
        np.quantile(
            train_scores,
            1.0 - target_rate,
        )
    )

    test_scores = (
        -model.score_samples(
            X_test
        )
    )

    test_flags = (
        test_scores
        >= threshold
    )

    return {
        "base_rate":
            base_rate,

        "target_rate":
            target_rate,

        "threshold":
            threshold,

        "scores":
            test_scores,

        "flags":
            test_flags,
    }


# ============================================================
# Prepare fold
# ============================================================

def get_fold(
    df,
    year,
):

    start = pd.Timestamp(
        f"{year}-01-01"
    )

    end = pd.Timestamp(
        f"{year}-12-31"
    )

    train = (
        df[
            df["Date"] < start
        ]
        .copy()
        .reset_index(drop=True)
    )

    test = (
        df[
            (
                df["Date"] >= start
            )
            &
            (
                df["Date"] <= end
            )
        ]
        .copy()
        .reset_index(drop=True)
    )

    return train, test


# ============================================================
# Seed stability
# ============================================================

def seed_stability(df):

    rows = []

    stored_flags = {}

    print(
        "\n--- RANDOM-SEED STABILITY ---"
    )

    for year in TEST_YEARS:

        train, test = (
            get_fold(
                df,
                year,
            )
        )

        baseline_rate = float(
            baseline_flag(test).mean()
        )

        for seed in SEEDS:

            result = fit_and_score(
                train=train,
                test=test,
                seed=seed,
                max_samples=512,
                threshold_multiplier=1.0,
            )

            key = (
                year,
                seed,
            )

            stored_flags[key] = (
                result["flags"]
            )

            rows.append(
                {
                    "Test_Year":
                        year,

                    "Seed":
                        seed,

                    "Train_N":
                        len(train),

                    "Test_N":
                        len(test),

                    "Training_Target_Rate":
                        result[
                            "target_rate"
                        ],

                    "Baseline_Test_Rate":
                        baseline_rate,

                    "IF_Test_Rate":
                        result[
                            "flags"
                        ].mean(),

                    "Threshold":
                        result[
                            "threshold"
                        ],
                }
            )

    results = pd.DataFrame(
        rows
    )

    pairwise_rows = []

    for year in TEST_YEARS:

        for seed_a, seed_b in combinations(
            SEEDS,
            2,
        ):

            flags_a = (
                stored_flags[
                    (
                        year,
                        seed_a,
                    )
                ]
            )

            flags_b = (
                stored_flags[
                    (
                        year,
                        seed_b,
                    )
                ]
            )

            pairwise_rows.append(
                {
                    "Test_Year":
                        year,

                    "Seed_A":
                        seed_a,

                    "Seed_B":
                        seed_b,

                    "Jaccard":
                        jaccard(
                            flags_a,
                            flags_b,
                        ),
                }
            )

    pairwise = pd.DataFrame(
        pairwise_rows
    )

    annual = (
        results
        .groupby(
            "Test_Year"
        )
        .agg(
            Mean_IF_Rate=(
                "IF_Test_Rate",
                "mean",
            ),
            SD_IF_Rate=(
                "IF_Test_Rate",
                "std",
            ),
            Min_IF_Rate=(
                "IF_Test_Rate",
                "min",
            ),
            Max_IF_Rate=(
                "IF_Test_Rate",
                "max",
            ),
        )
        .reset_index()
    )

    annual_jaccard = (
        pairwise
        .groupby(
            "Test_Year"
        )[
            "Jaccard"
        ]
        .mean()
        .reset_index(
            name="Mean_Pairwise_Jaccard"
        )
    )

    annual = annual.merge(
        annual_jaccard,
        on="Test_Year",
    )

    print(
        annual
        .round(4)
        .to_string(index=False)
    )

    return (
        results,
        pairwise,
        annual,
    )


# ============================================================
# Max-samples sensitivity
# ============================================================

def max_samples_sensitivity(
    df,
):

    rows = []

    print(
        "\n--- MAX-SAMPLES SENSITIVITY ---"
    )

    for max_samples in (
        MAX_SAMPLES_VALUES
    ):

        for year in TEST_YEARS:

            train, test = (
                get_fold(
                    df,
                    year,
                )
            )

            result = fit_and_score(
                train=train,
                test=test,
                seed=42,
                max_samples=max_samples,
                threshold_multiplier=1.0,
            )

            rows.append(
                {
                    "Max_Samples":
                        max_samples,

                    "Test_Year":
                        year,

                    "IF_Test_Rate":
                        result[
                            "flags"
                        ].mean(),

                    "Threshold":
                        result[
                            "threshold"
                        ],
                }
            )

    results = pd.DataFrame(
        rows
    )

    aggregate = (
        results
        .groupby(
            "Max_Samples"
        )
        .agg(
            Mean_Test_Rate=(
                "IF_Test_Rate",
                "mean",
            ),
            SD_Annual_Rate=(
                "IF_Test_Rate",
                "std",
            ),
            Min_Test_Rate=(
                "IF_Test_Rate",
                "min",
            ),
            Max_Test_Rate=(
                "IF_Test_Rate",
                "max",
            ),
        )
        .reset_index()
    )

    print(
        aggregate
        .round(4)
        .to_string(index=False)
    )

    return results


# ============================================================
# Threshold sensitivity
# ============================================================

def threshold_sensitivity(
    df,
):

    rows = []

    print(
        "\n--- THRESHOLD SENSITIVITY ---"
    )

    for multiplier in (
        THRESHOLD_MULTIPLIERS
    ):

        for year in TEST_YEARS:

            train, test = (
                get_fold(
                    df,
                    year,
                )
            )

            result = fit_and_score(
                train=train,
                test=test,
                seed=42,
                max_samples=512,
                threshold_multiplier=
                    multiplier,
            )

            rows.append(
                {
                    "Threshold_Multiplier":
                        multiplier,

                    "Test_Year":
                        year,

                    "Training_Target_Rate":
                        result[
                            "target_rate"
                        ],

                    "IF_Test_Rate":
                        result[
                            "flags"
                        ].mean(),

                    "Threshold":
                        result[
                            "threshold"
                        ],
                }
            )

    results = pd.DataFrame(
        rows
    )

    aggregate = (
        results
        .groupby(
            "Threshold_Multiplier"
        )
        .agg(
            Mean_Test_Rate=(
                "IF_Test_Rate",
                "mean",
            ),
            SD_Annual_Rate=(
                "IF_Test_Rate",
                "std",
            ),
            Min_Test_Rate=(
                "IF_Test_Rate",
                "min",
            ),
            Max_Test_Rate=(
                "IF_Test_Rate",
                "max",
            ),
        )
        .reset_index()
    )

    print(
        aggregate
        .round(4)
        .to_string(index=False)
    )

    return results


# ============================================================
# Final summary
# ============================================================

def build_summary(
    df,
    seed_results,
    pairwise,
    seed_annual,
    max_samples_results,
    threshold_results,
):

    baseline_annual = []

    for year in TEST_YEARS:

        _, test = (
            get_fold(
                df,
                year,
            )
        )

        baseline_annual.append(
            baseline_flag(
                test
            ).mean()
        )

    baseline_annual = np.array(
        baseline_annual
    )

    seed42 = (
        seed_results[
            seed_results[
                "Seed"
            ]
            == 42
        ]
        .sort_values(
            "Test_Year"
        )
    )

    if_rates = (
        seed42[
            "IF_Test_Rate"
        ]
        .to_numpy()
    )

    summary = pd.DataFrame(
        [
            {
                "Metric":
                    "Baseline annual rate SD",

                "Value":
                    np.std(
                        baseline_annual,
                        ddof=1,
                    ),
            },
            {
                "Metric":
                    "IF annual rate SD (seed 42)",

                "Value":
                    np.std(
                        if_rates,
                        ddof=1,
                    ),
            },
            {
                "Metric":
                    "Mean IF seed pairwise Jaccard",

                "Value":
                    pairwise[
                        "Jaccard"
                    ].mean(),
            },
            {
                "Metric":
                    "Worst annual mean seed Jaccard",

                "Value":
                    seed_annual[
                        "Mean_Pairwise_Jaccard"
                    ].min(),
            },
            {
                "Metric":
                    "Mean absolute IF-baseline annual rate gap",

                "Value":
                    np.mean(
                        np.abs(
                            if_rates
                            - baseline_annual
                        )
                    ),
            },
        ]
    )

    return summary


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
        "IF STABILITY AND SENSITIVITY ---"
    )

    df = load_data()

    (
        seed_results,
        pairwise,
        seed_annual,
    ) = seed_stability(
        df
    )

    max_samples_results = (
        max_samples_sensitivity(
            df
        )
    )

    threshold_results = (
        threshold_sensitivity(
            df
        )
    )

    summary = build_summary(
        df,
        seed_results,
        pairwise,
        seed_annual,
        max_samples_results,
        threshold_results,
    )

    print(
        "\n--- CORE STABILITY SUMMARY ---"
    )

    print(
        summary
        .round(4)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    seed_results.to_csv(
        SEED_FILE,
        index=False,
    )

    pairwise.to_csv(
        SEED_PAIRWISE_FILE,
        index=False,
    )

    max_samples_results.to_csv(
        MAX_SAMPLES_FILE,
        index=False,
    )

    threshold_results.to_csv(
        THRESHOLD_FILE,
        index=False,
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    print(
        "\n--- OUTPUT ---"
    )

    print(SEED_FILE)
    print(SEED_PAIRWISE_FILE)
    print(MAX_SAMPLES_FILE)
    print(THRESHOLD_FILE)
    print(SUMMARY_FILE)

    print(
        "\nIF stability and sensitivity "
        "analysis complete."
    )


if __name__ == "__main__":
    main()