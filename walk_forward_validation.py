from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from chronological_validation import (
    FEATURE_COLUMNS,
    N_STATES,
    causal_filter,
    fit_best_hmm,
    get_state_mapping,
)


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = Path(
    "data/processed/usdtry_features.csv"
)

OUTPUT_DIR = Path(
    "outputs/validation"
)

FIGURE_DIR = Path(
    "outputs/figures"
)

TEST_YEARS = [
    2021,
    2022,
    2023,
    2024,
    2025,
    2026,
]


# ============================================================
# Helpers
# ============================================================

def canonical_regime_order(state_names):
    """
    Return the numeric state IDs corresponding to the
    three canonical economic regime names.
    """

    reverse = {
        name: state
        for state, name
        in state_names.items()
    }

    return {
        "Low-Volatility Trend":
            reverse["Low-Volatility Trend"],

        "Elevated-Volatility Transition":
            reverse[
                "Elevated-Volatility Transition"
            ],

        "High-Volatility Stress":
            reverse["High-Volatility Stress"],
    }


def build_episodes(df):

    if len(df) == 0:
        return pd.DataFrame()

    temp = df.copy()

    temp["Changed"] = (
        temp["State"]
        != temp["State"].shift(1)
    )

    temp["Episode_ID"] = (
        temp["Changed"].cumsum()
    )

    return (
        temp.groupby("Episode_ID")
        .agg(
            State=("State", "first"),
            Regime_Name=(
                "Regime_Name",
                "first",
            ),
            Start_Date=(
                "Date",
                "first",
            ),
            End_Date=(
                "Date",
                "last",
            ),
            Duration=(
                "Date",
                "size",
            ),
        )
        .reset_index()
    )


def state_profile_table(
    df,
    fold_year,
    dataset_name,
):

    profiles = (
        df.groupby(
            [
                "State",
                "Regime_Name",
            ]
        )[FEATURE_COLUMNS]
        .mean()
        .reset_index()
    )

    profiles.insert(
        0,
        "Test_Year",
        fold_year,
    )

    profiles.insert(
        1,
        "Dataset",
        dataset_name,
    )

    return profiles


# ============================================================
# Single walk-forward fold
# ============================================================

def run_fold(
    df,
    test_year,
):

    test_start = pd.Timestamp(
        f"{test_year}-01-01"
    )

    test_end = pd.Timestamp(
        f"{test_year}-12-31"
    )

    train = (
        df[
            df["Date"]
            < test_start
        ]
        .copy()
        .reset_index(drop=True)
    )

    test = (
        df[
            (
                df["Date"] >= test_start
            )
            &
            (
                df["Date"] <= test_end
            )
        ]
        .copy()
        .reset_index(drop=True)
    )

    if len(train) == 0:
        raise ValueError(
            f"No training data for {test_year}"
        )

    if len(test) == 0:
        raise ValueError(
            f"No test data for {test_year}"
        )

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"WALK-FORWARD FOLD: {test_year}"
    )

    print(
        "=" * 70
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

    # --------------------------------------------------------
    # Train-only preprocessing
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        train[FEATURE_COLUMNS]
    )

    X_test = scaler.transform(
        test[FEATURE_COLUMNS]
    )

    # --------------------------------------------------------
    # Train-only model
    # --------------------------------------------------------

    (
        model,
        training_log_likelihood,
        best_seed,
    ) = fit_best_hmm(
        X_train
    )

    # --------------------------------------------------------
    # Causal filtering on TRAIN
    # --------------------------------------------------------

    (
        train_probabilities,
        causal_train_log_likelihood,
    ) = causal_filter(
        model,
        X_train,
    )

    # Important implementation sanity check.
    hmmlearn_score = model.score(
        X_train
    )

    likelihood_difference = abs(
        causal_train_log_likelihood
        - hmmlearn_score
    )

    print(
        "\nCausal-filter likelihood check:"
    )

    print(
        f"hmmlearn score: "
        f"{hmmlearn_score:.6f}"
    )

    print(
        f"causal filter: "
        f"{causal_train_log_likelihood:.6f}"
    )

    print(
        f"difference: "
        f"{likelihood_difference:.10f}"
    )

    if likelihood_difference > 1e-4:
        raise RuntimeError(
            "Causal filtering likelihood does not "
            "match hmmlearn model.score()."
        )

    # --------------------------------------------------------
    # State interpretation based ONLY on training model
    # --------------------------------------------------------

    (
        state_names,
        emission_means,
    ) = get_state_mapping(
        model,
        scaler,
    )

    canonical_states = (
        canonical_regime_order(
            state_names
        )
    )

    # --------------------------------------------------------
    # Training classifications
    # --------------------------------------------------------

    train["State"] = (
        train_probabilities.argmax(
            axis=1
        )
    )

    train["Confidence"] = (
        train_probabilities.max(
            axis=1
        )
    )

    train["Regime_Name"] = (
        train["State"].map(
            state_names
        )
    )

    # --------------------------------------------------------
    # Causal TEST inference
    # --------------------------------------------------------

    initial_test_posterior = (
        train_probabilities[-1]
    )

    (
        test_probabilities,
        test_log_likelihood,
    ) = causal_filter(
        model,
        X_test,
        initial_filtered=(
            initial_test_posterior
        ),
    )

    test["State"] = (
        test_probabilities.argmax(
            axis=1
        )
    )

    test["Confidence"] = (
        test_probabilities.max(
            axis=1
        )
    )

    test["Regime_Name"] = (
        test["State"].map(
            state_names
        )
    )

    # Preserve posterior probability for each
    # economically named state rather than only
    # arbitrary numeric state IDs.
    for regime_name, state in (
        canonical_states.items()
    ):

        safe_name = (
            regime_name
            .upper()
            .replace("-", "_")
            .replace(" ", "_")
        )

        test[
            f"P_{safe_name}"
        ] = (
            test_probabilities[
                :,
                state,
            ]
        )

    # --------------------------------------------------------
    # Episodes and switching
    # --------------------------------------------------------

    episodes = build_episodes(
        test
    )

    transitions = int(
        (
            test["State"]
            != test["State"].shift(1)
        )
        .iloc[1:]
        .sum()
    )

    switch_rate = (
        transitions
        / max(
            len(test) - 1,
            1,
        )
    )

    # --------------------------------------------------------
    # State balance
    # --------------------------------------------------------

    state_balance = (
        test.groupby(
            [
                "State",
                "Regime_Name",
            ]
        )
        .size()
        .rename(
            "Observations"
        )
        .reset_index()
    )

    state_balance[
        "Share"
    ] = (
        state_balance[
            "Observations"
        ]
        / len(test)
    )

    state_balance.insert(
        0,
        "Test_Year",
        test_year,
    )

    # --------------------------------------------------------
    # Persistence by regime
    # --------------------------------------------------------

    persistence = (
        episodes.groupby(
            [
                "State",
                "Regime_Name",
            ]
        )["Duration"]
        .agg(
            Episodes="count",
            Mean_Duration="mean",
            Median_Duration="median",
            Min_Duration="min",
            Max_Duration="max",
        )
        .reset_index()
    )

    persistence.insert(
        0,
        "Test_Year",
        test_year,
    )

    # --------------------------------------------------------
    # Train/test feature profiles
    # --------------------------------------------------------

    train_profiles = (
        state_profile_table(
            train,
            test_year,
            "TRAIN",
        )
    )

    test_profiles = (
        state_profile_table(
            test,
            test_year,
            "TEST",
        )
    )

    profiles = pd.concat(
        [
            train_profiles,
            test_profiles,
        ],
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Learned state-emission profiles
    # --------------------------------------------------------

    emission_means = (
        emission_means.copy()
    )

    emission_means[
        "Regime_Name"
    ] = (
        emission_means[
            "HMM_State"
        ]
        .map(
            state_names
        )
    )

    emission_means.insert(
        0,
        "Test_Year",
        test_year,
    )

    # --------------------------------------------------------
    # Per-fold summary
    # --------------------------------------------------------

    state_shares = {
        name: 0.0
        for name in (
            canonical_states.keys()
        )
    }

    for _, row in (
        state_balance.iterrows()
    ):

        state_shares[
            row["Regime_Name"]
        ] = row["Share"]

    summary = {
        "Test_Year":
            test_year,

        "Train_Start":
            train["Date"].min(),

        "Train_End":
            train["Date"].max(),

        "Test_Start":
            test["Date"].min(),

        "Test_End":
            test["Date"].max(),

        "Train_Observations":
            len(train),

        "Test_Observations":
            len(test),

        "Best_Seed":
            best_seed,

        "Train_LogL_Per_Obs":
            training_log_likelihood
            / len(train),

        "Test_LogL_Per_Obs":
            test_log_likelihood
            / len(test),

        "Mean_Confidence":
            test[
                "Confidence"
            ].mean(),

        "Median_Confidence":
            test[
                "Confidence"
            ].median(),

        "Share_Confidence_Below_60":
            (
                test[
                    "Confidence"
                ]
                < 0.60
            ).mean(),

        "Episodes":
            len(episodes),

        "Transitions":
            transitions,

        "Switch_Rate":
            switch_rate,

        "Mean_Episode_Duration":
            episodes[
                "Duration"
            ].mean(),

        "Median_Episode_Duration":
            episodes[
                "Duration"
            ].median(),

        "Low_Vol_Share":
            state_shares[
                "Low-Volatility Trend"
            ],

        "Elevated_Vol_Share":
            state_shares[
                "Elevated-Volatility Transition"
            ],

        "Stress_Share":
            state_shares[
                "High-Volatility Stress"
            ],
    }

    print(
        "\n--- FOLD SUMMARY ---"
    )

    print(
        f"Mean confidence: "
        f"{summary['Mean_Confidence']:.2%}"
    )

    print(
        f"Switch rate: "
        f"{switch_rate:.2%}"
    )

    print(
        f"Median episode: "
        f"{summary['Median_Episode_Duration']:.1f}"
    )

    print(
        "\nState shares:"
    )

    print(
        f"  Low:      "
        f"{summary['Low_Vol_Share']:.2%}"
    )

    print(
        f"  Elevated: "
        f"{summary['Elevated_Vol_Share']:.2%}"
    )

    print(
        f"  Stress:   "
        f"{summary['Stress_Share']:.2%}"
    )

    # Include test year on individual predictions.
    test.insert(
        0,
        "Test_Year",
        test_year,
    )

    return {
        "Summary":
            summary,

        "Predictions":
            test,

        "State_Balance":
            state_balance,

        "Persistence":
            persistence,

        "Profiles":
            profiles,

        "Emission_Means":
            emission_means,
    }


# ============================================================
# Visualization
# ============================================================

def plot_walk_forward_summary(
    summary,
):

    years = (
        summary["Test_Year"]
        .to_numpy()
    )

    # --------------------------------------------------------
    # State shares
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    ax.plot(
        years,
        summary[
            "Low_Vol_Share"
        ] * 100,
        marker="o",
        label="Low-Volatility Trend",
    )

    ax.plot(
        years,
        summary[
            "Elevated_Vol_Share"
        ] * 100,
        marker="o",
        label=(
            "Elevated-Volatility "
            "Transition"
        ),
    )

    ax.plot(
        years,
        summary[
            "Stress_Share"
        ] * 100,
        marker="o",
        label=(
            "High-Volatility Stress"
        ),
    )

    ax.set_title(
        "Walk-Forward Out-of-Sample "
        "State Shares"
    )

    ax.set_xlabel(
        "Test Year"
    )

    ax.set_ylabel(
        "Share of Test Observations (%)"
    )

    ax.legend()

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    output_file = (
        FIGURE_DIR
        / "10_walk_forward_state_shares.png"
    )

    fig.savefig(
        output_file,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {output_file}"
    )

    # --------------------------------------------------------
    # Persistence
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    ax.plot(
        years,
        summary[
            "Median_Episode_Duration"
        ],
        marker="o",
    )

    ax.set_title(
        "Walk-Forward Regime Persistence"
    )

    ax.set_xlabel(
        "Test Year"
    )

    ax.set_ylabel(
        "Median Episode Duration"
    )

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    output_file = (
        FIGURE_DIR
        / "11_walk_forward_persistence.png"
    )

    fig.savefig(
        output_file,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {output_file}"
    )

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    ax.plot(
        years,
        summary[
            "Mean_Confidence"
        ] * 100,
        marker="o",
    )

    ax.set_ylim(
        0,
        101,
    )

    ax.set_title(
        "Walk-Forward Mean "
        "Regime Confidence"
    )

    ax.set_xlabel(
        "Test Year"
    )

    ax.set_ylabel(
        "Mean Posterior Confidence (%)"
    )

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    output_file = (
        FIGURE_DIR
        / "12_walk_forward_confidence.png"
    )

    fig.savefig(
        output_file,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {output_file}"
    )


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["Date"],
    )

    df = (
        df.sort_values("Date")
        .reset_index(drop=True)
    )

    results = []

    for test_year in TEST_YEARS:

        fold = run_fold(
            df,
            test_year,
        )

        results.append(
            fold
        )

    # --------------------------------------------------------
    # Combine all folds
    # --------------------------------------------------------

    summary = pd.DataFrame(
        [
            result["Summary"]
            for result in results
        ]
    )

    predictions = pd.concat(
        [
            result["Predictions"]
            for result in results
        ],
        ignore_index=True,
    )

    balances = pd.concat(
        [
            result["State_Balance"]
            for result in results
        ],
        ignore_index=True,
    )

    persistence = pd.concat(
        [
            result["Persistence"]
            for result in results
        ],
        ignore_index=True,
    )

    profiles = pd.concat(
        [
            result["Profiles"]
            for result in results
        ],
        ignore_index=True,
    )

    emissions = pd.concat(
        [
            result["Emission_Means"]
            for result in results
        ],
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Print final summary
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "WALK-FORWARD VALIDATION SUMMARY"
    )

    print(
        "=" * 70
    )

    display_columns = [
        "Test_Year",
        "Test_Observations",
        "Test_LogL_Per_Obs",
        "Mean_Confidence",
        "Switch_Rate",
        "Median_Episode_Duration",
        "Low_Vol_Share",
        "Elevated_Vol_Share",
        "Stress_Share",
    ]

    print(
        summary[
            display_columns
        ]
        .round(4)
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Aggregate diagnostics
    # --------------------------------------------------------

    print(
        "\n--- AGGREGATE WALK-FORWARD METRICS ---"
    )

    print(
        f"Mean test confidence: "
        f"{summary['Mean_Confidence'].mean():.2%}"
    )

    print(
        f"Mean switch rate: "
        f"{summary['Switch_Rate'].mean():.2%}"
    )

    print(
        f"Median of annual median "
        f"episode durations: "
        f"{summary['Median_Episode_Duration'].median():.2f}"
    )

    years_with_stress = (
        summary.loc[
            summary[
                "Stress_Share"
            ] > 0,
            "Test_Year",
        ]
        .tolist()
    )

    print(
        "Years containing out-of-sample "
        f"stress state: {years_with_stress}"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    summary.to_csv(
        OUTPUT_DIR
        / "walk_forward_summary.csv",
        index=False,
    )

    predictions.to_csv(
        OUTPUT_DIR
        / "walk_forward_predictions.csv",
        index=False,
    )

    balances.to_csv(
        OUTPUT_DIR
        / "walk_forward_state_balance.csv",
        index=False,
    )

    persistence.to_csv(
        OUTPUT_DIR
        / "walk_forward_persistence.csv",
        index=False,
    )

    profiles.to_csv(
        OUTPUT_DIR
        / "walk_forward_state_profiles.csv",
        index=False,
    )

    emissions.to_csv(
        OUTPUT_DIR
        / "walk_forward_emission_means.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Figures
    # --------------------------------------------------------

    plot_walk_forward_summary(
        summary
    )

    print(
        "\nWalk-forward validation complete."
    )


if __name__ == "__main__":
    main()