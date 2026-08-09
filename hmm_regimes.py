from pathlib import Path

import numpy as np
import pandas as pd

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


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

N_STATES = 3
N_STARTS = 20
MIN_STATE_SIZE = 20


def fit_best_hmm(X_scaled):
    """
    Fit the HMM multiple times with different random seeds.

    EM estimation can converge to different solutions depending
    on initialization, so retain the valid model with the
    highest log-likelihood.
    """

    best_model = None
    best_score = -np.inf
    best_seed = None
    best_sizes = None

    for seed in range(N_STARTS):

        model = GaussianHMM(
            n_components=N_STATES,
            covariance_type="diag",
            n_iter=500,
            tol=1e-4,
            random_state=seed,
        )

        try:
            model.fit(X_scaled)

            score = model.score(X_scaled)
            labels = model.predict(X_scaled)

            sizes = pd.Series(labels).value_counts()

            # Reject degenerate solutions with tiny states
            if len(sizes) != N_STATES:
                continue

            if sizes.min() < MIN_STATE_SIZE:
                continue

            print(
                f"Seed {seed:2d} | "
                f"logL = {score:10.2f} | "
                f"sizes = {sizes.sort_index().to_dict()} | "
                f"converged = {model.monitor_.converged}"
            )

            if score > best_score:
                best_model = model
                best_score = score
                best_seed = seed
                best_sizes = sizes.sort_index()

        except Exception as exc:
            print(
                f"Seed {seed:2d} failed: "
                f"{type(exc).__name__}: {exc}"
            )

    if best_model is None:
        raise RuntimeError(
            "No valid 3-state HMM was found."
        )

    return (
        best_model,
        best_score,
        best_seed,
        best_sizes,
    )


def assign_regime_names(df):
    """
    Assign economically interpretable names AFTER fitting.

    HMM state numbers themselves are arbitrary.
    """

    means = (
        df.groupby("HMM_State")[FEATURE_COLUMNS]
        .mean()
    )

    # Explicit interpretation for this fitted 3-state model
    name_map = {
        0: "Elevated-Volatility Transition",
        1: "Low-Volatility Trend",
        2: "High-Volatility Stress",
    }

    code_map = {
        0: "ELEVATED_VOL",
        1: "LOW_VOL",
        2: "HIGH_VOL_STRESS",
    }

    return means, name_map, code_map


def build_episode_table(df):
    """
    Convert daily HMM states into continuous regime episodes.
    """

    result = df.copy()

    result["State_Changed"] = (
        result["HMM_State"]
        != result["HMM_State"].shift(1)
    )

    result["Episode_ID"] = (
        result["State_Changed"].cumsum()
    )

    episodes = (
        result.groupby("Episode_ID")
        .agg(
            HMM_State=("HMM_State", "first"),
            Regime_Name=("Regime_Name", "first"),
            Start_Date=("Date", "first"),
            End_Date=("Date", "last"),
            Duration=("Date", "size"),
            Start_Price=("USDTRY", "first"),
            End_Price=("USDTRY", "last"),
            Mean_Confidence=("Regime_Confidence", "mean"),
        )
        .reset_index()
    )

    episodes["Price_Change"] = (
        episodes["End_Price"]
        / episodes["Start_Price"]
        - 1
    )

    return result, episodes


def main():

    input_file = Path(
        "data/processed/usdtry_features.csv"
    )

    output_dir = Path("outputs")
    processed_dir = Path("data/processed")

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    processed_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ---------------------------------
    # Load data
    # ---------------------------------

    df = pd.read_csv(
        input_file,
        parse_dates=["Date"],
    )

    df = (
        df.sort_values("Date")
        .reset_index(drop=True)
    )

    X = df[FEATURE_COLUMNS].copy()

    print("\n--- HMM INPUT ---")
    print(f"Observations: {len(X):,}")
    print(f"Features: {len(FEATURE_COLUMNS)}")
    print(f"States: {N_STATES}")

    # ---------------------------------
    # Standardization
    # ---------------------------------

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ---------------------------------
    # Multi-start HMM fitting
    # ---------------------------------

    print("\n--- FITTING HMM ---")

    (
        model,
        best_score,
        best_seed,
        best_sizes,
    ) = fit_best_hmm(X_scaled)

    print("\n--- BEST MODEL ---")
    print(f"Seed: {best_seed}")
    print(f"Log-likelihood: {best_score:.2f}")
    print("State sizes:")
    print(best_sizes)

    # ---------------------------------
    # Decode hidden states
    # ---------------------------------

    states = model.predict(X_scaled)

    posterior = model.predict_proba(
        X_scaled
    )

    df["HMM_State"] = states

    # Maximum posterior probability
    df["Regime_Confidence"] = (
        posterior.max(axis=1)
    )

    # Preserve individual probabilities
    for state in range(N_STATES):
        df[f"P_State_{state}"] = posterior[:, state]

    # ---------------------------------
    # Interpret states
    # ---------------------------------

    (
        state_means,
        name_map,
        code_map,
    ) = assign_regime_names(df)

    df["Regime_Name"] = (
        df["HMM_State"].map(name_map)
    )

    df["Regime_Code"] = (
        df["HMM_State"].map(code_map)
    )

    print("\n--- HMM STATE FEATURE MEANS ---")
    print(
        state_means
        .round(5)
        .to_string()
    )

    print("\n--- STATE INTERPRETATION ---")

    for state in sorted(name_map):
        print(
            f"State {state}: "
            f"{name_map[state]}"
        )

    # ---------------------------------
    # Learned transition matrix
    # ---------------------------------

    transition_matrix = pd.DataFrame(
        model.transmat_,
        index=[
            f"State_{i}"
            for i in range(N_STATES)
        ],
        columns=[
            f"State_{i}"
            for i in range(N_STATES)
        ],
    )

    print("\n--- HMM LEARNED TRANSITION MATRIX ---")
    print(
        transition_matrix
        .round(4)
        .to_string()
    )

    # ---------------------------------
    # Empirical decoded transitions
    # ---------------------------------

    df["Next_State"] = (
        df["HMM_State"].shift(-1)
    )

    empirical_transitions = pd.crosstab(
        df["HMM_State"],
        df["Next_State"],
        normalize="index",
    )

    print(
        "\n--- EMPIRICAL DECODED "
        "TRANSITION PROBABILITIES ---"
    )

    print(
        empirical_transitions
        .round(4)
        .to_string()
    )

    # ---------------------------------
    # Episodes / persistence
    # ---------------------------------

    df, episodes = build_episode_table(df)

    persistence = (
        episodes
        .groupby(
            ["HMM_State", "Regime_Name"]
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

    print("\n--- HMM REGIME PERSISTENCE ---")

    print(
        persistence
        .round(2)
        .to_string(index=False)
    )

    # ---------------------------------
    # Longest episodes
    # ---------------------------------

    print("\n--- LONGEST HMM EPISODES ---")

    longest = (
        episodes
        .sort_values(
            "Duration",
            ascending=False
        )
        .head(20)
    )

    print(
        longest[
            [
                "Regime_Name",
                "Start_Date",
                "End_Date",
                "Duration",
                "Price_Change",
                "Mean_Confidence",
            ]
        ]
        .round(
            {
                "Price_Change": 4,
                "Mean_Confidence": 4,
            }
        )
        .to_string(index=False)
    )

    # ---------------------------------
    # Current / latest regime
    # ---------------------------------

    latest = df.iloc[-1]

    print("\n--- LATEST REGIME ---")
    print(f"Date:       {latest['Date'].date()}")
    print(f"USDTRY:     {latest['USDTRY']:.4f}")
    print(f"State:      {latest['HMM_State']}")
    print(f"Regime:     {latest['Regime_Name']}")
    print(
        f"Confidence: "
        f"{latest['Regime_Confidence']:.2%}"
    )

    # ---------------------------------
    # Save outputs
    # ---------------------------------

    labelled_file = (
        processed_dir /
        "usdtry_regimes_hmm3.csv"
    )

    episode_file = (
        output_dir /
        "hmm_regime_episodes.csv"
    )

    persistence_file = (
        output_dir /
        "hmm_regime_persistence.csv"
    )

    transition_file = (
        output_dir /
        "hmm_transition_matrix.csv"
    )

    df.to_csv(
        labelled_file,
        index=False,
    )

    episodes.to_csv(
        episode_file,
        index=False,
    )

    persistence.to_csv(
        persistence_file,
        index=False,
    )

    transition_matrix.to_csv(
        transition_file,
    )

    print("\n--- SAVED OUTPUTS ---")
    print(labelled_file)
    print(episode_file)
    print(persistence_file)
    print(transition_file)


if __name__ == "__main__":
    main()