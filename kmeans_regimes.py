from pathlib import Path

import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
)
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

    # ---------------------------------
    # Load engineered feature dataset
    # ---------------------------------

    input_file = Path(
        "data/processed/usdtry_features.csv"
    )

    df = pd.read_csv(
        input_file,
        parse_dates=["Date"]
    )

    X = df[FEATURE_COLUMNS].copy()

    print("\n--- INPUT ---")
    print(f"Observations: {len(X):,}")
    print(f"Features: {len(FEATURE_COLUMNS)}")

    # ---------------------------------
    # Standardize
    # ---------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    print("\n--- STANDARDIZATION ---")
    print("Feature means after scaling:")
    print(X_scaled.mean(axis=0).round(6))

    print("\nFeature standard deviations after scaling:")
    print(X_scaled.std(axis=0).round(6))

    # ---------------------------------
    # Test K = 2 to 6
    # ---------------------------------

    results = []

    for k in range(2, 7):

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=50
        )

        labels = model.fit_predict(X_scaled)

        silhouette = silhouette_score(
            X_scaled,
            labels
        )

        davies_bouldin = davies_bouldin_score(
            X_scaled,
            labels
        )

        cluster_sizes = pd.Series(labels).value_counts().sort_index()

        results.append(
            {
                "K": k,
                "Silhouette": silhouette,
                "Davies_Bouldin": davies_bouldin,
                "Smallest_Cluster": cluster_sizes.min(),
                "Largest_Cluster": cluster_sizes.max(),
            }
        )

        print(f"\n--- K = {k} ---")
        print(f"Silhouette:      {silhouette:.4f}")
        print(f"Davies-Bouldin:  {davies_bouldin:.4f}")

        print("Cluster sizes:")

        for cluster, size in cluster_sizes.items():
            print(
                f"  Regime {cluster}: "
                f"{size:,} observations"
            )

    # ---------------------------------
    # Save evaluation table
    # ---------------------------------

    results_df = pd.DataFrame(results)

    output_dir = Path("outputs")
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    results_file = (
        output_dir / "kmeans_model_comparison.csv"
    )

    results_df.to_csv(
        results_file,
        index=False
    )

    print("\n--- MODEL COMPARISON ---")
    print(
        results_df.to_string(
            index=False
        )
    )

    print(
        f"\nSaved comparison to: "
        f"{results_file}"
    )


if __name__ == "__main__":
    main()