from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.optimize import linear_sum_assignment

from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.preprocessing import StandardScaler


# ============================================================
# Configuration
# ============================================================

FEATURE_FILE = Path(
    "data/processed/usdtry_features.csv"
)

KMEANS_FILE = Path(
    "data/processed/usdtry_regimes_k3.csv"
)

HMM_FILE = Path(
    "data/processed/usdtry_regimes_hmm3.csv"
)

OUTPUT_DIR = Path("outputs")
FIGURE_DIR = OUTPUT_DIR / "figures"


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


KMEANS_NAMES = {
    0: "High-Volatility Uptrend",
    1: "Baseline Trend",
    2: "High-Volatility Correction",
}


HMM_NAMES = {
    0: "Elevated-Volatility Transition",
    1: "Low-Volatility Trend",
    2: "High-Volatility Stress",
}


# ============================================================
# Data loading
# ============================================================

def load_data():

    features = pd.read_csv(
        FEATURE_FILE,
        parse_dates=["Date"],
    )

    kmeans = pd.read_csv(
        KMEANS_FILE,
        parse_dates=["Date"],
    )

    hmm = pd.read_csv(
        HMM_FILE,
        parse_dates=["Date"],
    )

    required_kmeans = {
        "Date",
        "Regime",
    }

    required_hmm = {
        "Date",
        "HMM_State",
    }

    if not required_kmeans.issubset(kmeans.columns):
        raise ValueError(
            "K-Means file must contain Date and Regime."
        )

    if not required_hmm.issubset(hmm.columns):
        raise ValueError(
            "HMM file must contain Date and HMM_State."
        )

    # Keep only columns needed from model outputs.
    kmeans_labels = kmeans[
        ["Date", "Regime"]
    ].rename(
        columns={
            "Regime": "KMeans_State"
        }
    )

    hmm_columns = [
        "Date",
        "HMM_State",
    ]

    if "Regime_Confidence" in hmm.columns:
        hmm_columns.append(
            "Regime_Confidence"
        )

    hmm_labels = hmm[
        hmm_columns
    ].copy()

    # Merge by date rather than assuming row ordering.
    df = (
        features
        .merge(
            kmeans_labels,
            on="Date",
            how="inner",
        )
        .merge(
            hmm_labels,
            on="Date",
            how="inner",
        )
        .sort_values("Date")
        .reset_index(drop=True)
    )

    if len(df) != len(features):
        print(
            "WARNING: merged model output does not "
            "contain every feature observation."
        )

    return df


# ============================================================
# Episode analysis
# ============================================================

def build_episodes(
    df,
    label_column,
):

    temp = df[
        ["Date", label_column]
    ].copy()

    temp["Changed"] = (
        temp[label_column]
        != temp[label_column].shift(1)
    )

    temp["Episode_ID"] = (
        temp["Changed"].cumsum()
    )

    episodes = (
        temp.groupby("Episode_ID")
        .agg(
            State=(label_column, "first"),
            Start_Date=("Date", "first"),
            End_Date=("Date", "last"),
            Duration=("Date", "size"),
        )
        .reset_index()
    )

    return episodes


def persistence_by_state(
    df,
    label_column,
    names,
):

    episodes = build_episodes(
        df,
        label_column,
    )

    result = (
        episodes
        .groupby("State")["Duration"]
        .agg(
            Episodes="count",
            Mean_Duration="mean",
            Median_Duration="median",
            Min_Duration="min",
            Max_Duration="max",
        )
        .reset_index()
    )

    result["Regime_Name"] = (
        result["State"].map(names)
    )

    result = result[
        [
            "State",
            "Regime_Name",
            "Episodes",
            "Mean_Duration",
            "Median_Duration",
            "Min_Duration",
            "Max_Duration",
        ]
    ]

    return result, episodes


# ============================================================
# Transition analysis
# ============================================================

def transition_matrix(
    labels,
):

    current = pd.Series(
        labels[:-1],
        name="Current",
    )

    next_state = pd.Series(
        labels[1:],
        name="Next",
    )

    matrix = pd.crosstab(
        current,
        next_state,
        normalize="index",
    )

    return matrix


def mean_self_transition(
    matrix,
):

    common_states = (
        matrix.index
        .intersection(
            matrix.columns
        )
    )

    values = [
        matrix.loc[state, state]
        for state in common_states
    ]

    return float(
        np.mean(values)
    )


# ============================================================
# Geometric clustering metrics
# ============================================================

def clustering_metrics(
    X_scaled,
    labels,
):

    return {
        "Silhouette": silhouette_score(
            X_scaled,
            labels,
        ),
        "Davies_Bouldin": (
            davies_bouldin_score(
                X_scaled,
                labels,
            )
        ),
        "Calinski_Harabasz": (
            calinski_harabasz_score(
                X_scaled,
                labels,
            )
        ),
    }


# ============================================================
# Overall temporal metrics
# ============================================================

def temporal_metrics(
    df,
    label_column,
):

    labels = (
        df[label_column]
        .to_numpy()
    )

    episodes = build_episodes(
        df,
        label_column,
    )

    transitions = int(
        np.sum(
            labels[1:]
            != labels[:-1]
        )
    )

    possible_transitions = (
        len(labels) - 1
    )

    switch_rate = (
        transitions
        / possible_transitions
    )

    matrix = transition_matrix(
        labels
    )

    return {
        "Observations": len(labels),
        "States": len(
            np.unique(labels)
        ),
        "Episodes": len(episodes),
        "Transitions": transitions,
        "Switch_Rate": switch_rate,
        "Mean_Episode_Duration": (
            episodes["Duration"].mean()
        ),
        "Median_Episode_Duration": (
            episodes["Duration"].median()
        ),
        "Max_Episode_Duration": (
            episodes["Duration"].max()
        ),
        "Mean_Self_Transition": (
            mean_self_transition(
                matrix
            )
        ),
    }


# ============================================================
# State balance
# ============================================================

def state_balance(
    df,
    label_column,
    names,
    model_name,
):

    counts = (
        df[label_column]
        .value_counts()
        .sort_index()
    )

    result = pd.DataFrame({
        "State": counts.index,
        "Observations": counts.values,
    })

    result["Share"] = (
        result["Observations"]
        / len(df)
    )

    result["Regime_Name"] = (
        result["State"].map(names)
    )

    result["Model"] = model_name

    return result


# ============================================================
# State feature profiles
# ============================================================

def feature_profiles(
    df,
    label_column,
    names,
    model_name,
):

    profile = (
        df.groupby(
            label_column
        )[FEATURE_COLUMNS]
        .mean()
        .reset_index()
        .rename(
            columns={
                label_column: "State"
            }
        )
    )

    profile["Regime_Name"] = (
        profile["State"].map(names)
    )

    profile["Model"] = model_name

    columns = [
        "Model",
        "State",
        "Regime_Name",
        *FEATURE_COLUMNS,
    ]

    return profile[columns]


# ============================================================
# K-Means vs HMM agreement
# ============================================================

def calculate_agreement(
    df,
):

    k = (
        df["KMeans_State"]
        .to_numpy()
    )

    h = (
        df["HMM_State"]
        .to_numpy()
    )

    ari = adjusted_rand_score(
        k,
        h,
    )

    nmi = normalized_mutual_info_score(
        k,
        h,
    )

    contingency = pd.crosstab(
        df["KMeans_State"],
        df["HMM_State"],
    )

    matrix = (
        contingency
        .to_numpy()
    )

    # Hungarian assignment finds the mapping
    # giving the maximum possible raw agreement.
    row_ind, col_ind = (
        linear_sum_assignment(
            -matrix
        )
    )

    mapping = {
        contingency.index[row]:
        contingency.columns[col]
        for row, col in zip(
            row_ind,
            col_ind,
        )
    }

    mapped_kmeans = (
        df["KMeans_State"]
        .map(mapping)
    )

    best_match_accuracy = (
        mapped_kmeans
        == df["HMM_State"]
    ).mean()

    return {
        "ARI": ari,
        "NMI": nmi,
        "Best_Match_Agreement": (
            best_match_accuracy
        ),
        "Mapping": mapping,
        "Contingency": contingency,
    }


# ============================================================
# Visualization 6:
# State shares
# ============================================================

def plot_state_shares(
    balance,
):

    plot_df = balance.copy()

    plot_df["Label"] = (
        plot_df["Model"]
        + "\n"
        + plot_df["Regime_Name"]
    )

    fig, ax = plt.subplots(
        figsize=(13, 7)
    )

    ax.bar(
        plot_df["Label"],
        plot_df["Share"] * 100,
    )

    ax.set_ylabel(
        "Share of Observations (%)"
    )

    ax.set_title(
        "State Distribution: K-Means vs HMM"
    )

    ax.tick_params(
        axis="x",
        rotation=20,
    )

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    fig.tight_layout()

    output_file = (
        FIGURE_DIR
        / "06_model_state_shares.png"
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
# Visualization 7:
# Episode duration
# ============================================================

def plot_episode_durations(
    k_persistence,
    h_persistence,
):

    k = (
        k_persistence.copy()
    )

    h = (
        h_persistence.copy()
    )

    k["Model"] = "K-Means"
    h["Model"] = "HMM"

    combined = pd.concat(
        [k, h],
        ignore_index=True,
    )

    combined["Label"] = (
        combined["Model"]
        + "\n"
        + combined["Regime_Name"]
    )

    fig, ax = plt.subplots(
        figsize=(13, 7)
    )

    ax.bar(
        combined["Label"],
        combined[
            "Median_Duration"
        ],
    )

    ax.set_ylabel(
        "Median Episode Duration "
        "(Observations)"
    )

    ax.set_title(
        "Regime Persistence: "
        "K-Means vs HMM"
    )

    ax.tick_params(
        axis="x",
        rotation=20,
    )

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    fig.tight_layout()

    output_file = (
        FIGURE_DIR
        / "07_model_episode_duration.png"
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
# Visualization 8:
# Cross-model contingency matrix
# ============================================================

def plot_agreement_matrix(
    contingency,
):

    matrix = (
        contingency
        .to_numpy(
            dtype=float
        )
    )

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    image = ax.imshow(
        matrix,
        aspect="auto",
    )

    ax.set_xticks(
        range(
            len(
                contingency.columns
            )
        )
    )

    ax.set_yticks(
        range(
            len(
                contingency.index
            )
        )
    )

    ax.set_xticklabels(
        [
            HMM_NAMES[state]
            for state
            in contingency.columns
        ],
        rotation=25,
        ha="right",
    )

    ax.set_yticklabels(
        [
            KMEANS_NAMES[state]
            for state
            in contingency.index
        ]
    )

    ax.set_xlabel(
        "HMM State"
    )

    ax.set_ylabel(
        "K-Means Cluster"
    )

    ax.set_title(
        "K-Means vs HMM "
        "Classification Agreement"
    )

    for i in range(
        matrix.shape[0]
    ):

        for j in range(
            matrix.shape[1]
        ):

            ax.text(
                j,
                i,
                f"{int(matrix[i, j])}",
                ha="center",
                va="center",
            )

    fig.colorbar(
        image,
        ax=ax,
        label="Observations",
    )

    fig.tight_layout()

    output_file = (
        FIGURE_DIR
        / "08_kmeans_hmm_agreement.png"
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
# Markdown report
# ============================================================

def write_report(
    comparison,
    agreement,
    k_persistence,
    h_persistence,
):

    k = comparison.loc[
        "K-Means"
    ]

    h = comparison.loc[
        "HMM"
    ]

    if (
        h["Switch_Rate"]
        < k["Switch_Rate"]
        and
        h["Median_Episode_Duration"]
        > k["Median_Episode_Duration"]
    ):
        temporal_conclusion = (
            "The HMM produces more temporally "
            "coherent regimes than K-Means, "
            "with fewer switches and longer "
            "typical regime episodes."
        )
    else:
        temporal_conclusion = (
            "The HMM does not provide an "
            "unambiguous improvement in "
            "temporal persistence over K-Means."
        )

    report = f"""
# K-Means vs Gaussian HMM Model Comparison

## Objective

The purpose of this comparison is to determine whether the
Gaussian Hidden Markov Model provides a more useful market-regime
representation than the K-Means baseline for daily USD/TRY data.

The models are compared using geometric separation, temporal
persistence, state balance, cross-model agreement, and economic
interpretability.

## Geometric Separation

| Metric | K-Means | HMM |
|---|---:|---:|
| Silhouette | {k['Silhouette']:.4f} | {h['Silhouette']:.4f} |
| Davies-Bouldin | {k['Davies_Bouldin']:.4f} | {h['Davies_Bouldin']:.4f} |
| Calinski-Harabasz | {k['Calinski_Harabasz']:.2f} | {h['Calinski_Harabasz']:.2f} |

Higher Silhouette and Calinski-Harabasz values indicate stronger
geometric separation, while a lower Davies-Bouldin value is
preferred.

These metrics must not be interpreted as a direct likelihood-based
test between K-Means and HMM. K-Means explicitly optimizes geometric
cluster structure, whereas the HMM additionally models temporal
state transitions.

## Temporal Persistence

| Metric | K-Means | HMM |
|---|---:|---:|
| Episodes | {int(k['Episodes'])} | {int(h['Episodes'])} |
| State transitions | {int(k['Transitions'])} | {int(h['Transitions'])} |
| Switch rate | {k['Switch_Rate']:.2%} | {h['Switch_Rate']:.2%} |
| Mean episode duration | {k['Mean_Episode_Duration']:.2f} | {h['Mean_Episode_Duration']:.2f} |
| Median episode duration | {k['Median_Episode_Duration']:.2f} | {h['Median_Episode_Duration']:.2f} |
| Maximum episode duration | {int(k['Max_Episode_Duration'])} | {int(h['Max_Episode_Duration'])} |
| Mean self-transition probability | {k['Mean_Self_Transition']:.2%} | {h['Mean_Self_Transition']:.2%} |

{temporal_conclusion}

## Cross-Model Agreement

Adjusted Rand Index:

**{agreement['ARI']:.4f}**

Normalized Mutual Information:

**{agreement['NMI']:.4f}**

Best possible label-matched agreement:

**{agreement['Best_Match_Agreement']:.2%}**

The best-match agreement is computed after relabeling the arbitrary
numeric cluster identifiers using an optimal assignment. It should
not be interpreted as evidence that the economic meaning of the
K-Means and HMM states is identical.

## Interpretation

K-Means serves as the static clustering baseline. Each observation is
classified according to its position in feature space without using
the previous market state.

The Gaussian HMM models both the observed feature distributions and
the probability of transitioning between latent states. Consequently,
it is designed to represent persistent market conditions rather than
independent daily clusters.

The HMM states obtained in this analysis are interpreted as:

1. Low-Volatility Trend
2. Elevated-Volatility Transition
3. High-Volatility Stress

The HMM transition structure additionally suggests that movement
between the low- and high-volatility states generally occurs through
the elevated-volatility state.

## Model Selection

For a market-regime monitoring system, temporal coherence is a core
requirement. Therefore, geometric clustering metrics alone should not
determine the preferred model.

If the HMM preserves economically interpretable state profiles while
reducing regime flickering and increasing episode persistence, it is
preferred over K-Means for operational regime detection.

K-Means remains useful as a transparent benchmark against which the
temporal HMM can be evaluated.

## Remaining Validation

This comparison is based on models fitted to the complete historical
dataset. It is therefore an in-sample model comparison rather than a
prospective performance test.

The next stage should use chronological out-of-sample and walk-forward
validation to determine whether the HMM state structure remains stable
when applied to future observations.
""".strip()

    output_file = (
        OUTPUT_DIR
        / "model_comparison_report.md"
    )

    output_file.write_text(
        report,
        encoding="utf-8",
    )

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

    df = load_data()

    print(
        "\n--- MODEL COMPARISON INPUT ---"
    )

    print(
        f"Observations: {len(df):,}"
    )

    print(
        f"Start: {df['Date'].min().date()}"
    )

    print(
        f"End:   {df['Date'].max().date()}"
    )

    # ---------------------------------
    # Standardize features
    # ---------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        df[FEATURE_COLUMNS]
    )

    # ---------------------------------
    # Geometric metrics
    # ---------------------------------

    k_geometry = clustering_metrics(
        X_scaled,
        df["KMeans_State"],
    )

    h_geometry = clustering_metrics(
        X_scaled,
        df["HMM_State"],
    )

    # ---------------------------------
    # Temporal metrics
    # ---------------------------------

    k_temporal = temporal_metrics(
        df,
        "KMeans_State",
    )

    h_temporal = temporal_metrics(
        df,
        "HMM_State",
    )

    comparison = pd.DataFrame(
        {
            "K-Means": {
                **k_geometry,
                **k_temporal,
            },
            "HMM": {
                **h_geometry,
                **h_temporal,
            },
        }
    ).T

    # ---------------------------------
    # Persistence by state
    # ---------------------------------

    (
        k_persistence,
        k_episodes,
    ) = persistence_by_state(
        df,
        "KMeans_State",
        KMEANS_NAMES,
    )

    (
        h_persistence,
        h_episodes,
    ) = persistence_by_state(
        df,
        "HMM_State",
        HMM_NAMES,
    )

    # ---------------------------------
    # State balance
    # ---------------------------------

    k_balance = state_balance(
        df,
        "KMeans_State",
        KMEANS_NAMES,
        "K-Means",
    )

    h_balance = state_balance(
        df,
        "HMM_State",
        HMM_NAMES,
        "HMM",
    )

    balance = pd.concat(
        [
            k_balance,
            h_balance,
        ],
        ignore_index=True,
    )

    # ---------------------------------
    # Feature profiles
    # ---------------------------------

    k_profiles = feature_profiles(
        df,
        "KMeans_State",
        KMEANS_NAMES,
        "K-Means",
    )

    h_profiles = feature_profiles(
        df,
        "HMM_State",
        HMM_NAMES,
        "HMM",
    )

    profiles = pd.concat(
        [
            k_profiles,
            h_profiles,
        ],
        ignore_index=True,
    )

    # ---------------------------------
    # Cross-model agreement
    # ---------------------------------

    agreement = calculate_agreement(
        df
    )

    # ---------------------------------
    # Print main comparison
    # ---------------------------------

    print(
        "\n--- OVERALL MODEL COMPARISON ---"
    )

    print(
        comparison
        .round(4)
        .to_string()
    )

    print(
        "\n--- K-MEANS PERSISTENCE ---"
    )

    print(
        k_persistence
        .round(2)
        .to_string(index=False)
    )

    print(
        "\n--- HMM PERSISTENCE ---"
    )

    print(
        h_persistence
        .round(2)
        .to_string(index=False)
    )

    print(
        "\n--- STATE BALANCE ---"
    )

    balance_print = (
        balance.copy()
    )

    balance_print["Share"] = (
        balance_print["Share"]
        * 100
    )

    print(
        balance_print
        .round(2)
        .to_string(index=False)
    )

    print(
        "\n--- CROSS-MODEL AGREEMENT ---"
    )

    print(
        f"Adjusted Rand Index: "
        f"{agreement['ARI']:.4f}"
    )

    print(
        f"Normalized Mutual Information: "
        f"{agreement['NMI']:.4f}"
    )

    print(
        f"Best-match agreement: "
        f"{agreement['Best_Match_Agreement']:.2%}"
    )

    print(
        "\nOptimal numeric-label mapping:"
    )

    for k_state, h_state in (
        agreement["Mapping"].items()
    ):
        print(
            f"K-Means {k_state} "
            f"-> HMM {h_state}"
        )

    print(
        "\nContingency table:"
    )

    print(
        agreement[
            "Contingency"
        ].to_string()
    )

    # ---------------------------------
    # Save CSV outputs
    # ---------------------------------

    comparison.to_csv(
        OUTPUT_DIR
        / "model_comparison_summary.csv"
    )

    k_persistence.to_csv(
        OUTPUT_DIR
        / "kmeans_persistence_comparison.csv",
        index=False,
    )

    h_persistence.to_csv(
        OUTPUT_DIR
        / "hmm_persistence_comparison.csv",
        index=False,
    )

    balance.to_csv(
        OUTPUT_DIR
        / "model_state_balance.csv",
        index=False,
    )

    profiles.to_csv(
        OUTPUT_DIR
        / "model_state_feature_profiles.csv",
        index=False,
    )

    agreement[
        "Contingency"
    ].to_csv(
        OUTPUT_DIR
        / "kmeans_hmm_contingency.csv"
    )

    # ---------------------------------
    # Figures
    # ---------------------------------

    plot_state_shares(
        balance
    )

    plot_episode_durations(
        k_persistence,
        h_persistence,
    )

    plot_agreement_matrix(
        agreement[
            "Contingency"
        ]
    )

    # ---------------------------------
    # Markdown report
    # ---------------------------------

    write_report(
        comparison,
        agreement,
        k_persistence,
        h_persistence,
    )

    print(
        "\nModel comparison complete."
    )


if __name__ == "__main__":
    main()