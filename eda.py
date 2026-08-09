from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

    csv_file = Path("data/processed/usdtry_features.csv")

    df = pd.read_csv(
        csv_file,
        parse_dates=["Date"]
    )

    print("\n--- DATASET ---")
    print(f"Rows: {len(df):,}")
    print(f"Start: {df['Date'].min().date()}")
    print(f"End:   {df['Date'].max().date()}")

    # ---------------------------------
    # 1. Missing / infinite values
    # ---------------------------------

    print("\n--- MISSING VALUES ---")
    print(df[FEATURE_COLUMNS].isna().sum())

    infinite_counts = np.isinf(
        df[FEATURE_COLUMNS]
    ).sum()

    print("\n--- INFINITE VALUES ---")
    print(pd.Series(
        infinite_counts,
        index=FEATURE_COLUMNS
    ))

    # ---------------------------------
    # 2. Correlation matrix
    # ---------------------------------

    correlation = df[FEATURE_COLUMNS].corr()

    print("\n--- CORRELATION MATRIX ---")
    print(correlation.round(3))

    # ---------------------------------
    # 3. Save correlation matrix
    # ---------------------------------

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    correlation.to_csv(
        output_dir / "feature_correlations.csv"
    )

    # ---------------------------------
    # 4. Plot feature distributions
    # ---------------------------------

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    for feature in FEATURE_COLUMNS:

        plt.figure(figsize=(8, 5))

        plt.hist(
            df[feature],
            bins=50
        )

        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.title(f"Distribution of {feature}")

        plt.tight_layout()

        plt.savefig(
            figure_dir / f"{feature}_distribution.png",
            dpi=150
        )

        plt.close()

    print("\nEDA complete.")
    print(f"Correlation table: {output_dir / 'feature_correlations.csv'}")
    print(f"Figures: {figure_dir}")


if __name__ == "__main__":
    main()
