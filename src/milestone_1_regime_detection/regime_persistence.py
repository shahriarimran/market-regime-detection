from pathlib import Path

import pandas as pd


REGIME_NAMES = {
    0: "High-Volatility Uptrend",
    1: "Baseline Trend",
    2: "High-Volatility Correction",
}


def main():
    input_file = Path(
        "data/processed/usdtry_regimes_k3.csv"
    )

    df = pd.read_csv(
        input_file,
        parse_dates=["Date"]
    )

    df = df.sort_values("Date").reset_index(drop=True)

    # ----------------------------------
    # 1. Detect regime changes
    # ----------------------------------

    df["Regime_Changed"] = (
        df["Regime"] != df["Regime"].shift(1)
    )

    # Each continuous block gets an ID
    df["Episode_ID"] = df["Regime_Changed"].cumsum()

    # ----------------------------------
    # 2. Build episode table
    # ----------------------------------

    episodes = (
        df.groupby("Episode_ID")
        .agg(
            Regime=("Regime", "first"),
            Start_Date=("Date", "first"),
            End_Date=("Date", "last"),
            Duration=("Date", "size"),
            Start_Price=("USDTRY", "first"),
            End_Price=("USDTRY", "last"),
        )
        .reset_index()
    )

    episodes["Regime_Name"] = (
        episodes["Regime"]
        .map(REGIME_NAMES)
    )

    episodes["Price_Change"] = (
        episodes["End_Price"]
        / episodes["Start_Price"]
        - 1
    )

    # ----------------------------------
    # 3. Regime persistence statistics
    # ----------------------------------

    persistence = (
        episodes.groupby(
            ["Regime", "Regime_Name"]
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

    print("\n--- REGIME PERSISTENCE ---")
    print(
        persistence
        .round(2)
        .to_string(index=False)
    )

    # ----------------------------------
    # 4. Daily transition counts
    # ----------------------------------

    df["Next_Regime"] = df["Regime"].shift(-1)

    transition_counts = pd.crosstab(
        df["Regime"],
        df["Next_Regime"]
    )

    print("\n--- TRANSITION COUNTS ---")
    print(transition_counts)

    # ----------------------------------
    # 5. Transition probabilities
    # ----------------------------------

    transition_probabilities = pd.crosstab(
        df["Regime"],
        df["Next_Regime"],
        normalize="index"
    )

    print("\n--- TRANSITION PROBABILITIES ---")
    print(
        transition_probabilities
        .round(4)
    )

    # ----------------------------------
    # 6. Same-regime probability
    # ----------------------------------

    print("\n--- DAILY REGIME PERSISTENCE ---")

    for regime in sorted(df["Regime"].unique()):

        subset = df[
            df["Regime"] == regime
        ]

        valid = subset[
            subset["Next_Regime"].notna()
        ]

        stay_probability = (
            valid["Next_Regime"]
            .eq(regime)
            .mean()
        )

        print(
            f"{regime} "
            f"({REGIME_NAMES[regime]}): "
            f"{stay_probability:.2%}"
        )

    # ----------------------------------
    # 7. Longest episodes
    # ----------------------------------

    print("\n--- LONGEST EPISODES ---")

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
            ]
        ]
        .to_string(index=False)
    )

    # ----------------------------------
    # 8. Save outputs
    # ----------------------------------

    output_dir = Path("outputs")
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    episodes.to_csv(
        output_dir /
        "regime_episodes.csv",
        index=False
    )

    persistence.to_csv(
        output_dir /
        "regime_persistence_summary.csv",
        index=False
    )

    transition_probabilities.to_csv(
        output_dir /
        "regime_transition_probabilities.csv"
    )

    print("\nSaved:")
    print(
        output_dir /
        "regime_episodes.csv"
    )
    print(
        output_dir /
        "regime_persistence_summary.csv"
    )
    print(
        output_dir /
        "regime_transition_probabilities.csv"
    )


if __name__ == "__main__":
    main()