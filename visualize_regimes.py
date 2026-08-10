from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = Path(
    "data/processed/usdtry_regimes_hmm3.csv"
)

TRANSITION_FILE = Path(
    "outputs/hmm_transition_matrix.csv"
)

OUTPUT_DIR = Path(
    "outputs/figures"
)


# These labels correspond to the current HMM solution.
REGIME_NAMES = {
    0: "Elevated-Volatility Transition",
    1: "Low-Volatility Trend",
    2: "High-Volatility Stress",
}


FEATURES_TO_PLOT = [
    "Return_5D",
    "Volatility_20D",
    "MA_Slope_20D",
    "Drawdown_60D",
]


# ============================================================
# Utilities
# ============================================================

def load_data():
    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["Date"],
    )

    df = (
        df.sort_values("Date")
        .reset_index(drop=True)
    )

    # Override the earlier automatic naming
    df["Regime_Name"] = (
        df["HMM_State"]
        .map(REGIME_NAMES)
    )

    return df


def get_regime_colors():
    """
    Use Matplotlib's default color cycle rather than
    hardcoding a custom palette.
    """

    default_colors = (
        plt.rcParams[
            "axes.prop_cycle"
        ]
        .by_key()["color"]
    )

    return {
        state: default_colors[i]
        for i, state in enumerate(
            sorted(REGIME_NAMES)
        )
    }


def build_episodes(df):
    """
    Construct continuous HMM regime episodes.
    """

    temp = df.copy()

    temp["State_Changed"] = (
        temp["HMM_State"]
        != temp["HMM_State"].shift(1)
    )

    temp["Episode_ID"] = (
        temp["State_Changed"]
        .cumsum()
    )

    episodes = (
        temp.groupby("Episode_ID")
        .agg(
            HMM_State=("HMM_State", "first"),
            Start_Date=("Date", "first"),
            End_Date=("Date", "last"),
            Duration=("Date", "size"),
        )
        .reset_index()
    )

    return episodes


# ============================================================
# Figure 1
# USD/TRY price colored by HMM regime
# ============================================================

def plot_price_regimes(
    df,
    regime_colors,
):

    fig, ax = plt.subplots(
        figsize=(15, 7)
    )

    # Continuous underlying price
    ax.plot(
        df["Date"],
        df["USDTRY"],
        linewidth=1.1,
        alpha=0.45,
        label="USD/TRY",
    )

    # Regime-specific observations
    for state, name in REGIME_NAMES.items():

        subset = df[
            df["HMM_State"] == state
        ]

        ax.scatter(
            subset["Date"],
            subset["USDTRY"],
            s=10,
            color=regime_colors[state],
            label=name,
        )

    ax.set_title(
        "USD/TRY Market Regimes Detected by Gaussian HMM"
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("USD/TRY Exchange Rate")

    ax.legend(
        loc="upper left"
    )

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    output_file = (
        OUTPUT_DIR /
        "01_usdtry_hmm_regimes.png"
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
# Figure 2
# HMM regime timeline
# ============================================================

def plot_regime_timeline(
    df,
    regime_colors,
):

    fig, ax = plt.subplots(
        figsize=(15, 4)
    )

    for state, name in REGIME_NAMES.items():

        subset = df[
            df["HMM_State"] == state
        ]

        ax.scatter(
            subset["Date"],
            [state] * len(subset),
            s=10,
            color=regime_colors[state],
            label=name,
        )

    ax.set_yticks(
        sorted(REGIME_NAMES)
    )

    ax.set_yticklabels(
        [
            REGIME_NAMES[state]
            for state in sorted(REGIME_NAMES)
        ]
    )

    ax.set_xlabel("Date")

    ax.set_title(
        "HMM Market Regime Timeline"
    )

    ax.grid(
        axis="x",
        alpha=0.2
    )

    fig.tight_layout()

    output_file = (
        OUTPUT_DIR /
        "02_hmm_regime_timeline.png"
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
# Figure 3
# Feature distributions by regime
# ============================================================

def plot_feature_distributions(
    df,
):

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10),
    )

    axes = axes.flatten()

    states = sorted(
        REGIME_NAMES
    )

    labels = [
        REGIME_NAMES[state]
        for state in states
    ]

    for ax, feature in zip(
        axes,
        FEATURES_TO_PLOT,
    ):

        feature_data = [
            df.loc[
                df["HMM_State"] == state,
                feature,
            ].dropna()
            for state in states
        ]

        ax.boxplot(
            feature_data,
            tick_labels=labels,
            showfliers=False,
        )

        ax.set_title(
            feature
        )

        ax.set_ylabel(
            feature
        )

        ax.tick_params(
            axis="x",
            rotation=20
        )

        ax.grid(
            axis="y",
            alpha=0.2
        )

    fig.suptitle(
        "Feature Distributions Across HMM Market Regimes",
        fontsize=14,
    )

    fig.tight_layout()

    output_file = (
        OUTPUT_DIR /
        "03_hmm_feature_distributions.png"
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
# Figure 4
# HMM transition matrix
# ============================================================

def plot_transition_matrix():

    transition = pd.read_csv(
        TRANSITION_FILE,
        index_col=0,
    )

    matrix = (
        transition
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

    states = list(
        range(
            len(matrix)
        )
    )

    labels = [
        REGIME_NAMES[state]
        for state in states
    ]

    ax.set_xticks(
        states
    )

    ax.set_yticks(
        states
    )

    ax.set_xticklabels(
        labels,
        rotation=25,
        ha="right",
    )

    ax.set_yticklabels(
        labels
    )

    ax.set_xlabel(
        "Next Regime"
    )

    ax.set_ylabel(
        "Current Regime"
    )

    ax.set_title(
        "HMM Learned Transition Probabilities"
    )

    # Write numeric probabilities
    # directly into each matrix cell.
    for i in range(
        matrix.shape[0]
    ):

        for j in range(
            matrix.shape[1]
        ):

            ax.text(
                j,
                i,
                f"{matrix[i, j]:.3f}",
                ha="center",
                va="center",
            )

    fig.colorbar(
        image,
        ax=ax,
        label="Transition Probability",
    )

    fig.tight_layout()

    output_file = (
        OUTPUT_DIR /
        "04_hmm_transition_matrix.png"
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
# Figure 5
# HMM regime confidence
# ============================================================

def plot_regime_confidence(
    df,
    regime_colors,
):

    fig, ax = plt.subplots(
        figsize=(15, 6)
    )

    # Overall maximum posterior probability
    ax.plot(
        df["Date"],
        df["Regime_Confidence"],
        linewidth=1,
        alpha=0.8,
        label="Maximum posterior probability",
    )

    # Highlight observations according
    # to the decoded regime.
    for state, name in REGIME_NAMES.items():

        subset = df[
            df["HMM_State"] == state
        ]

        ax.scatter(
            subset["Date"],
            subset["Regime_Confidence"],
            s=8,
            color=regime_colors[state],
            alpha=0.5,
            label=name,
        )

    ax.set_ylim(
        0,
        1.02
    )

    ax.set_xlabel(
        "Date"
    )

    ax.set_ylabel(
        "Posterior Probability"
    )

    ax.set_title(
        "HMM Regime Classification Confidence"
    )

    ax.grid(
        alpha=0.2
    )

    ax.legend(
        loc="lower left"
    )

    fig.tight_layout()

    output_file = (
        OUTPUT_DIR /
        "05_hmm_regime_confidence.png"
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
# Extra diagnostic:
# lowest-confidence observations
# ============================================================

def print_low_confidence_dates(
    df,
):

    print(
        "\n--- LOWEST CONFIDENCE DATES ---"
    )

    columns = [
        "Date",
        "USDTRY",
        "HMM_State",
        "Regime_Name",
        "Regime_Confidence",
    ]

    lowest = (
        df[columns]
        .sort_values(
            "Regime_Confidence"
        )
        .head(20)
    )

    print(
        lowest.to_string(
            index=False
        )
    )


# ============================================================
# Extra diagnostic:
# regime episode summary
# ============================================================

def print_episode_summary(
    df,
):

    episodes = build_episodes(
        df
    )

    print(
        "\n--- REGIME EPISODE SUMMARY ---"
    )

    summary = (
        episodes
        .groupby(
            "HMM_State"
        )["Duration"]
        .agg(
            Episodes="count",
            Mean="mean",
            Median="median",
            Maximum="max",
        )
    )

    summary.index = [
        REGIME_NAMES[state]
        for state in summary.index
    ]

    print(
        summary.round(2)
        .to_string()
    )


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_data()

    regime_colors = (
        get_regime_colors()
    )

    print(
        "\n--- VISUALIZATION INPUT ---"
    )

    print(
        f"Observations: {len(df):,}"
    )

    print(
        f"Start date: "
        f"{df['Date'].min().date()}"
    )

    print(
        f"End date: "
        f"{df['Date'].max().date()}"
    )

    print(
        "\nCreating figures..."
    )

    plot_price_regimes(
        df,
        regime_colors,
    )

    plot_regime_timeline(
        df,
        regime_colors,
    )

    plot_feature_distributions(
        df,
    )

    plot_transition_matrix()

    plot_regime_confidence(
        df,
        regime_colors,
    )

    print_low_confidence_dates(
        df
    )

    print_episode_summary(
        df
    )

    print(
        "\nVisualization complete."
    )


if __name__ == "__main__":
    main()