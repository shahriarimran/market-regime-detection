from pathlib import Path

import pandas as pd

from sklearn.cluster import KMeans
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


def main():

    input_file = Path(
        "data/processed/usdtry_features.csv"
    )

    df = pd.read_csv(
        input_file,
        parse_dates=["Date"]
    )

    X = df[FEATURE_COLUMNS].copy()

    # -------------------------------
    # Standardize
    # -------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # -------------------------------
    # Fit K = 3
    # -------------------------------

    model = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=50
    )

    df["Regime"] = model.fit_predict(X_scaled)

    # -------------------------------
    # Cluster sizes
    # -------------------------------

    print("\n--- REGIME SIZES ---")

    regime_sizes = (
        df["Regime"]
        .value_counts()
        .sort_index()
    )

    print(regime_sizes)

    # -------------------------------
    # Mean feature characteristics
    # -------------------------------

    print("\n--- REGIME FEATURE MEANS ---")

    regime_means = (
        df.groupby("Regime")[FEATURE_COLUMNS]
        .mean()
    )

    print(
        regime_means
        .round(5)
        .to_string()
    )

    # -------------------------------
    # Median characteristics
    # -------------------------------

    print("\n--- REGIME FEATURE MEDIANS ---")

    regime_medians = (
        df.groupby("Regime")[FEATURE_COLUMNS]
        .median()
    )

    print(
        regime_medians
        .round(5)
        .to_string()
    )

    # -------------------------------
    # Price statistics
    # -------------------------------

    print("\n--- USDTRY BY REGIME ---")

    price_summary = (
        df.groupby("Regime")["USDTRY"]
        .agg(
            [
                "count",
                "mean",
                "median",
                "min",
                "max",
            ]
        )
    )

    print(
        price_summary
        .round(4)
        .to_string()
    )

    # -------------------------------
    # Dates belonging to each regime
    # -------------------------------

    for regime in sorted(df["Regime"].unique()):

        subset = df[df["Regime"] == regime]

        print(
            f"\n--- REGIME {regime}: "
            f"{len(subset)} OBSERVATIONS ---"
        )

        print("\nFirst 15 dates:")
        print(
            subset[
                ["Date", "USDTRY"]
            ]
            .head(15)
            .to_string(index=False)
        )

        print("\nLargest 1-day movements:")

        print(
            subset[
                [
                    "Date",
                    "USDTRY",
                    "Return_1D",
                    "Return_5D",
                    "Volatility_20D",
                    "Drawdown_60D",
                ]
            ]
            .sort_values(
                "Return_1D",
                key=abs,
                ascending=False
            )
            .head(10)
            .to_string(index=False)
        )

    # -------------------------------
    # Save regime-labelled dataset
    # -------------------------------

    output_file = Path(
        "data/processed/usdtry_regimes_k3.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nSaved labelled dataset to: "
        f"{output_file}"
    )


if __name__ == "__main__":
    main()