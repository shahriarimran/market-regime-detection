from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from sklearn.preprocessing import StandardScaler


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

SPLIT_DATE = pd.Timestamp(
    "2024-01-01"
)

N_STATES = 3
N_STARTS = 20
MIN_STATE_SIZE = 20


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


# ============================================================
# Fit HMM
# ============================================================

def fit_best_hmm(X_train):

    best_model = None
    best_score = -np.inf
    best_seed = None

    print(
        "\n--- TRAINING HMM ---"
    )

    for seed in range(N_STARTS):

        model = GaussianHMM(
            n_components=N_STATES,
            covariance_type="diag",
            n_iter=500,
            tol=1e-4,
            random_state=seed,
        )

        try:

            model.fit(X_train)

            score = model.score(
                X_train
            )

            labels = model.predict(
                X_train
            )

            sizes = (
                pd.Series(labels)
                .value_counts()
            )

            if len(sizes) != N_STATES:
                continue

            if sizes.min() < MIN_STATE_SIZE:
                continue

            print(
                f"Seed {seed:2d} | "
                f"logL = {score:10.2f} | "
                f"sizes = "
                f"{sizes.sort_index().to_dict()} | "
                f"converged = "
                f"{model.monitor_.converged}"
            )

            if score > best_score:

                best_model = model
                best_score = score
                best_seed = seed

        except Exception as exc:

            print(
                f"Seed {seed:2d} failed: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

    if best_model is None:

        raise RuntimeError(
            "No valid HMM solution found."
        )

    return (
        best_model,
        best_score,
        best_seed,
    )


# ============================================================
# Gaussian emission probability
# ============================================================

def log_emission_probability(
    model,
    X,
):

    """
    Compute log p(x_t | state).

    This implementation is for the diagonal-covariance
    Gaussian HMM used in this project.
    """

    means = np.asarray(
        model.means_
    )

    covars = np.asarray(
        model.covars_
    )

    # hmmlearn may expose diagonal covariances
    # as full diagonal matrices.
    if covars.ndim == 3:

        variances = np.diagonal(
            covars,
            axis1=1,
            axis2=2,
        )

    else:

        variances = covars

    variances = np.maximum(
        variances,
        1e-12,
    )

    n_samples = X.shape[0]
    n_features = X.shape[1]

    output = np.empty(
        (
            n_samples,
            model.n_components,
        )
    )

    constant = (
        n_features
        * np.log(
            2.0 * np.pi
        )
    )

    for state in range(
        model.n_components
    ):

        diff = (
            X
            - means[state]
        )

        mahalanobis = np.sum(
            (
                diff ** 2
            )
            / variances[state],
            axis=1,
        )

        log_det = np.sum(
            np.log(
                variances[state]
            )
        )

        output[:, state] = (
            -0.5
            * (
                constant
                + log_det
                + mahalanobis
            )
        )

    return output


# ============================================================
# Causal HMM filtering
# ============================================================

def causal_filter(
    model,
    X,
    initial_filtered=None,
):

    """
    Forward filtering only.

    At observation t, the posterior uses information
    available through time t and never future observations.

    Returns:
        filtered probabilities
        total predictive log-likelihood
    """

    log_emissions = (
        log_emission_probability(
            model,
            X,
        )
    )

    n_samples = X.shape[0]
    n_states = model.n_components

    filtered = np.zeros(
        (
            n_samples,
            n_states,
        )
    )

    total_log_likelihood = 0.0

    previous = (
        None
        if initial_filtered is None
        else np.asarray(
            initial_filtered,
            dtype=float,
        )
    )

    for t in range(n_samples):

        # --------------------------------
        # Predict state before seeing x_t
        # --------------------------------

        if previous is None:

            prior = np.asarray(
                model.startprob_,
                dtype=float,
            )

        else:

            prior = (
                previous
                @ model.transmat_
            )

        prior = np.clip(
            prior,
            1e-300,
            None,
        )

        prior = (
            prior
            / prior.sum()
        )

        # --------------------------------
        # Observe x_t
        # --------------------------------

        log_joint = (
            np.log(prior)
            + log_emissions[t]
        )

        log_normalizer = (
            logsumexp(
                log_joint
            )
        )

        posterior = np.exp(
            log_joint
            - log_normalizer
        )

        filtered[t] = posterior

        total_log_likelihood += (
            log_normalizer
        )

        previous = posterior

    return (
        filtered,
        total_log_likelihood,
    )


# ============================================================
# Canonical state naming
# ============================================================

def get_state_mapping(
    model,
    scaler,
):

    """
    Convert HMM state means back to original feature units,
    then order states according to Volatility_20D.

    This avoids relying on arbitrary numeric HMM state IDs.
    """

    raw_means = scaler.inverse_transform(
        model.means_
    )

    means = pd.DataFrame(
        raw_means,
        columns=FEATURE_COLUMNS,
    )

    means["HMM_State"] = (
        range(N_STATES)
    )

    ordered_states = (
        means
        .sort_values(
            "Volatility_20D"
        )["HMM_State"]
        .tolist()
    )

    names = {
        ordered_states[0]:
            "Low-Volatility Trend",

        ordered_states[1]:
            "Elevated-Volatility Transition",

        ordered_states[2]:
            "High-Volatility Stress",
    }

    return (
        names,
        means,
    )


# ============================================================
# Episode analysis
# ============================================================

def build_episodes(
    df,
):

    temp = df.copy()

    temp["Changed"] = (
        temp["State"]
        != temp["State"].shift(1)
    )

    temp["Episode_ID"] = (
        temp["Changed"]
        .cumsum()
    )

    episodes = (
        temp.groupby(
            "Episode_ID"
        )
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

    return episodes


# ============================================================
# Transition matrix
# ============================================================

def empirical_transition_matrix(
    states,
):

    current = pd.Series(
        states[:-1],
        name="Current",
    )

    next_state = pd.Series(
        states[1:],
        name="Next",
    )

    return pd.crosstab(
        current,
        next_state,
        normalize="index",
    )


# ============================================================
# State profile comparison
# ============================================================

def state_profiles(
    df,
    name,
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

    profiles["Dataset"] = name

    return profiles


# ============================================================
# Visualization
# ============================================================

def plot_test_regimes(
    test,
):

    default_colors = (
        plt.rcParams[
            "axes.prop_cycle"
        ]
        .by_key()["color"]
    )

    states = sorted(
        test["State"].unique()
    )

    colors = {
        state:
            default_colors[i]
        for i, state in enumerate(
            states
        )
    }

    fig, ax = plt.subplots(
        figsize=(15, 7)
    )

    ax.plot(
        test["Date"],
        test["USDTRY"],
        linewidth=1,
        alpha=0.4,
        label="USD/TRY",
    )

    for state in states:

        subset = test[
            test["State"]
            == state
        ]

        regime_name = (
            subset[
                "Regime_Name"
            ]
            .iloc[0]
        )

        ax.scatter(
            subset["Date"],
            subset["USDTRY"],
            s=12,
            color=colors[state],
            label=regime_name,
        )

    ax.set_title(
        "Chronological Out-of-Sample "
        "HMM Regime Detection"
    )

    ax.set_xlabel(
        "Date"
    )

    ax.set_ylabel(
        "USD/TRY"
    )

    ax.legend()

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    output_file = (
        FIGURE_DIR
        / "09_chronological_validation.png"
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

    # --------------------------------
    # Load data
    # --------------------------------

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["Date"],
    )

    df = (
        df.sort_values("Date")
        .reset_index(drop=True)
    )

    train = (
        df[
            df["Date"]
            < SPLIT_DATE
        ]
        .copy()
        .reset_index(drop=True)
    )

    test = (
        df[
            df["Date"]
            >= SPLIT_DATE
        ]
        .copy()
        .reset_index(drop=True)
    )

    if len(train) == 0:
        raise ValueError(
            "Training dataset is empty."
        )

    if len(test) == 0:
        raise ValueError(
            "Test dataset is empty."
        )

    print(
        "\n--- CHRONOLOGICAL SPLIT ---"
    )

    print(
        f"Train observations: "
        f"{len(train):,}"
    )

    print(
        f"Train period: "
        f"{train['Date'].min().date()} "
        f"to "
        f"{train['Date'].max().date()}"
    )

    print(
        f"\nTest observations: "
        f"{len(test):,}"
    )

    print(
        f"Test period: "
        f"{test['Date'].min().date()} "
        f"to "
        f"{test['Date'].max().date()}"
    )

    # --------------------------------
    # Train-only scaling
    # --------------------------------

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        train[FEATURE_COLUMNS]
    )

    X_test = scaler.transform(
        test[FEATURE_COLUMNS]
    )

    # --------------------------------
    # Fit train-only HMM
    # --------------------------------

    (
        model,
        train_log_likelihood,
        best_seed,
    ) = fit_best_hmm(
        X_train
    )

    print(
        "\n--- BEST TRAINING MODEL ---"
    )

    print(
        f"Seed: {best_seed}"
    )

    print(
        f"Training log-likelihood: "
        f"{train_log_likelihood:.2f}"
    )

    print(
        f"Training log-likelihood / obs: "
        f"{train_log_likelihood / len(train):.4f}"
    )

    # --------------------------------
    # Canonical state names
    # --------------------------------

    (
        state_names,
        emission_means,
    ) = get_state_mapping(
        model,
        scaler,
    )

    print(
        "\n--- TRAINED STATE EMISSION MEANS ---"
    )

    print(
        emission_means
        .round(5)
        .to_string(
            index=False
        )
    )

    print(
        "\n--- STATE INTERPRETATION ---"
    )

    for state in sorted(
        state_names
    ):

        print(
            f"State {state}: "
            f"{state_names[state]}"
        )

    # --------------------------------
    # CAUSAL train filtering
    # --------------------------------

    (
        train_probs,
        train_filter_log_likelihood,
    ) = causal_filter(
        model,
        X_train,
    )

    train["State"] = (
        train_probs.argmax(
            axis=1
        )
    )

    train["Confidence"] = (
        train_probs.max(
            axis=1
        )
    )

    train["Regime_Name"] = (
        train["State"]
        .map(state_names)
    )

    # --------------------------------
    # CAUSAL out-of-sample filtering
    #
    # Carry the final filtered TRAIN
    # posterior into the TEST period.
    # --------------------------------

    initial_test_state = (
        train_probs[-1]
    )

    (
        test_probs,
        test_log_likelihood,
    ) = causal_filter(
        model,
        X_test,
        initial_filtered=(
            initial_test_state
        ),
    )

    test["State"] = (
        test_probs.argmax(
            axis=1
        )
    )

    test["Confidence"] = (
        test_probs.max(
            axis=1
        )
    )

    test["Regime_Name"] = (
        test["State"]
        .map(state_names)
    )

    for state in range(
        N_STATES
    ):

        test[
            f"P_State_{state}"
        ] = (
            test_probs[:, state]
        )

    # --------------------------------
    # OOS likelihood
    # --------------------------------

    avg_test_log_likelihood = (
        test_log_likelihood
        / len(test)
    )

    print(
        "\n--- OUT-OF-SAMPLE LIKELIHOOD ---"
    )

    print(
        f"Test log-likelihood: "
        f"{test_log_likelihood:.2f}"
    )

    print(
        f"Test log-likelihood / obs: "
        f"{avg_test_log_likelihood:.4f}"
    )

    # --------------------------------
    # Confidence
    # --------------------------------

    mean_confidence = (
        test["Confidence"].mean()
    )

    median_confidence = (
        test["Confidence"].median()
    )

    low_confidence_share = (
        (
            test["Confidence"]
            < 0.60
        )
        .mean()
    )

    print(
        "\n--- OUT-OF-SAMPLE CONFIDENCE ---"
    )

    print(
        f"Mean confidence: "
        f"{mean_confidence:.2%}"
    )

    print(
        f"Median confidence: "
        f"{median_confidence:.2%}"
    )

    print(
        f"Share below 60% confidence: "
        f"{low_confidence_share:.2%}"
    )

    # --------------------------------
    # State balance
    # --------------------------------

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

    state_balance["Share"] = (
        state_balance[
            "Observations"
        ]
        / len(test)
    )

    print(
        "\n--- OUT-OF-SAMPLE STATE BALANCE ---"
    )

    balance_print = (
        state_balance.copy()
    )

    balance_print["Share"] *= 100

    print(
        balance_print
        .round(2)
        .to_string(
            index=False
        )
    )

    # --------------------------------
    # Persistence
    # --------------------------------

    episodes = build_episodes(
        test
    )

    persistence = (
        episodes
        .groupby(
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

    print(
        "\n--- OUT-OF-SAMPLE PERSISTENCE ---"
    )

    print(
        persistence
        .round(2)
        .to_string(
            index=False
        )
    )

    # --------------------------------
    # Switching
    # --------------------------------

    states = (
        test["State"]
        .to_numpy()
    )

    transitions = int(
        np.sum(
            states[1:]
            != states[:-1]
        )
    )

    switch_rate = (
        transitions
        / (
            len(states) - 1
        )
    )

    print(
        "\n--- OUT-OF-SAMPLE SWITCHING ---"
    )

    print(
        f"Transitions: "
        f"{transitions}"
    )

    print(
        f"Switch rate: "
        f"{switch_rate:.2%}"
    )

    transition = (
        empirical_transition_matrix(
            states
        )
    )

    print(
        "\n--- OUT-OF-SAMPLE "
        "TRANSITION MATRIX ---"
    )

    print(
        transition
        .round(4)
        .to_string()
    )

    # --------------------------------
    # Train/test observed profiles
    # --------------------------------

    train_profiles = (
        state_profiles(
            train,
            "TRAIN",
        )
    )

    test_profiles = (
        state_profiles(
            test,
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

    print(
        "\n--- TRAIN / TEST "
        "OBSERVED STATE PROFILES ---"
    )

    print(
        profiles[
            [
                "Dataset",
                "State",
                "Regime_Name",
                "Return_5D",
                "Volatility_20D",
                "MA_Slope_20D",
                "Drawdown_60D",
            ]
        ]
        .round(5)
        .to_string(
            index=False
        )
    )

    # --------------------------------
    # Lowest-confidence OOS dates
    # --------------------------------

    print(
        "\n--- LOWEST-CONFIDENCE "
        "OUT-OF-SAMPLE DATES ---"
    )

    print(
        test[
            [
                "Date",
                "USDTRY",
                "State",
                "Regime_Name",
                "Confidence",
            ]
        ]
        .sort_values(
            "Confidence"
        )
        .head(20)
        .to_string(
            index=False
        )
    )

    # --------------------------------
    # Summary
    # --------------------------------

    summary = pd.DataFrame(
        [
            {
                "Train_Start":
                    train[
                        "Date"
                    ].min(),

                "Train_End":
                    train[
                        "Date"
                    ].max(),

                "Test_Start":
                    test[
                        "Date"
                    ].min(),

                "Test_End":
                    test[
                        "Date"
                    ].max(),

                "Train_Observations":
                    len(train),

                "Test_Observations":
                    len(test),

                "Best_Seed":
                    best_seed,

                "Train_LogL_Per_Obs":
                    (
                        train_log_likelihood
                        / len(train)
                    ),

                "Test_LogL_Per_Obs":
                    avg_test_log_likelihood,

                "Mean_Test_Confidence":
                    mean_confidence,

                "Median_Test_Confidence":
                    median_confidence,

                "Low_Confidence_Share":
                    low_confidence_share,

                "Test_Transitions":
                    transitions,

                "Test_Switch_Rate":
                    switch_rate,

                "Test_Episodes":
                    len(episodes),

                "Median_Test_Episode":
                    episodes[
                        "Duration"
                    ].median(),

                "Mean_Test_Episode":
                    episodes[
                        "Duration"
                    ].mean(),
            }
        ]
    )

    # --------------------------------
    # Save results
    # --------------------------------

    test.to_csv(
        OUTPUT_DIR
        / "chronological_test_predictions.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT_DIR
        / "chronological_summary.csv",
        index=False,
    )

    state_balance.to_csv(
        OUTPUT_DIR
        / "chronological_state_balance.csv",
        index=False,
    )

    persistence.to_csv(
        OUTPUT_DIR
        / "chronological_persistence.csv",
        index=False,
    )

    transition.to_csv(
        OUTPUT_DIR
        / "chronological_transition_matrix.csv"
    )

    profiles.to_csv(
        OUTPUT_DIR
        / "chronological_state_profiles.csv",
        index=False,
    )

    emission_means.to_csv(
        OUTPUT_DIR
        / "chronological_emission_means.csv",
        index=False,
    )

    # --------------------------------
    # Plot
    # --------------------------------

    plot_test_regimes(
        test
    )

    print(
        "\n--- VALIDATION NOTE ---"
    )

    print(
        "No supervised accuracy is reported "
        "because market regimes do not have "
        "ground-truth class labels."
    )

    print(
        "Validation is therefore based on "
        "out-of-sample likelihood, confidence, "
        "persistence, state balance, transition "
        "behavior, and state-profile stability."
    )

    print(
        "\nChronological validation complete."
    )


if __name__ == "__main__":
    main()