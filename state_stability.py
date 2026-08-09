from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = Path(
    "outputs/validation/walk_forward_emission_means.csv"
)

OUTPUT_DIR = Path(
    "outputs/validation"
)

FIGURE_DIR = Path(
    "outputs/figures"
)


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


REGIME_ORDER = [
    "Low-Volatility Trend",
    "Elevated-Volatility Transition",
    "High-Volatility Stress",
]


# ============================================================
# Load
# ============================================================

def load_data():

    df = pd.read_csv(
        INPUT_FILE
    )

    required = {
        "Test_Year",
        "HMM_State",
        "Regime_Name",
        *FEATURE_COLUMNS,
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing columns: "
            f"{sorted(missing)}"
        )

    return (
        df.sort_values(
            [
                "Test_Year",
                "Regime_Name",
            ]
        )
        .reset_index(drop=True)
    )


# ============================================================
# 1. Ordering checks
# ============================================================

def calculate_ordering_checks(df):

    rows = []

    for year in sorted(
        df["Test_Year"].unique()
    ):

        fold = (
            df[
                df["Test_Year"] == year
            ]
            .set_index("Regime_Name")
        )

        if not all(
            regime in fold.index
            for regime in REGIME_ORDER
        ):
            continue

        low = fold.loc[
            REGIME_ORDER[0]
        ]

        elevated = fold.loc[
            REGIME_ORDER[1]
        ]

        stress = fold.loc[
            REGIME_ORDER[2]
        ]

        checks = {
            "Volatility_5D_Order":
                low["Volatility_5D"]
                < elevated["Volatility_5D"]
                < stress["Volatility_5D"],

            "Volatility_20D_Order":
                low["Volatility_20D"]
                < elevated["Volatility_20D"]
                < stress["Volatility_20D"],

            "Volatility_60D_Order":
                low["Volatility_60D"]
                < elevated["Volatility_60D"]
                < stress["Volatility_60D"],

            "MA_Slope_Order":
                low["MA_Slope_20D"]
                <= elevated["MA_Slope_20D"]
                < stress["MA_Slope_20D"],
        }

        rows.append(
            {
                "Test_Year": year,
                **checks,
                "Passed_Checks":
                    sum(checks.values()),
                "Total_Checks":
                    len(checks),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# 2. Standardize centroid profiles
# ============================================================

def standardized_profiles(df):

    result = df.copy()

    scaler = StandardScaler()

    scaled = scaler.fit_transform(
        result[FEATURE_COLUMNS]
    )

    scaled_columns = []

    for i, feature in enumerate(
        FEATURE_COLUMNS
    ):

        column = (
            f"Z_{feature}"
        )

        result[column] = (
            scaled[:, i]
        )

        scaled_columns.append(
            column
        )

    return (
        result,
        scaled_columns,
    )


# ============================================================
# 3. Canonical profiles
# ============================================================

def calculate_reference_profiles(
    df,
    scaled_columns,
):

    """
    Median centroid across folds.

    Median is preferred over mean here because
    it is less sensitive to one unusual fold.
    """

    reference = (
        df.groupby(
            "Regime_Name"
        )[scaled_columns]
        .median()
        .reindex(
            REGIME_ORDER
        )
    )

    return reference


# ============================================================
# 4. Distance from canonical state
# ============================================================

def calculate_reference_distances(
    df,
    scaled_columns,
    reference,
):

    rows = []

    for _, row in df.iterrows():

        regime = row[
            "Regime_Name"
        ]

        vector = (
            row[
                scaled_columns
            ]
            .to_numpy(
                dtype=float
            )
        )

        ref = (
            reference.loc[
                regime
            ]
            .to_numpy(
                dtype=float
            )
        )

        distance = np.linalg.norm(
            vector - ref
        )

        rows.append(
            {
                "Test_Year":
                    row["Test_Year"],

                "Regime_Name":
                    regime,

                "Reference_Distance":
                    distance,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# 5. Adjacent-fold drift
# ============================================================

def calculate_fold_drift(
    df,
    scaled_columns,
):

    rows = []

    for regime in REGIME_ORDER:

        subset = (
            df[
                df["Regime_Name"]
                == regime
            ]
            .sort_values(
                "Test_Year"
            )
        )

        previous = None
        previous_year = None

        for _, row in (
            subset.iterrows()
        ):

            current = (
                row[
                    scaled_columns
                ]
                .to_numpy(
                    dtype=float
                )
            )

            if previous is not None:

                drift = np.linalg.norm(
                    current
                    - previous
                )

                rows.append(
                    {
                        "Regime_Name":
                            regime,

                        "From_Year":
                            previous_year,

                        "To_Year":
                            row[
                                "Test_Year"
                            ],

                        "Centroid_Drift":
                            drift,
                    }
                )

            previous = current

            previous_year = (
                row["Test_Year"]
            )

    return pd.DataFrame(rows)


# ============================================================
# 6. Within-fold regime separation
# ============================================================

def calculate_separation(
    df,
    scaled_columns,
):

    rows = []

    for year in sorted(
        df["Test_Year"].unique()
    ):

        fold = (
            df[
                df["Test_Year"]
                == year
            ]
            .set_index(
                "Regime_Name"
            )
        )

        if not all(
            regime in fold.index
            for regime in REGIME_ORDER
        ):
            continue

        vectors = {
            regime:
                fold.loc[
                    regime,
                    scaled_columns,
                ]
                .to_numpy(
                    dtype=float
                )

            for regime
            in REGIME_ORDER
        }

        d_low_elevated = (
            np.linalg.norm(
                vectors[
                    REGIME_ORDER[0]
                ]
                -
                vectors[
                    REGIME_ORDER[1]
                ]
            )
        )

        d_elevated_stress = (
            np.linalg.norm(
                vectors[
                    REGIME_ORDER[1]
                ]
                -
                vectors[
                    REGIME_ORDER[2]
                ]
            )
        )

        d_low_stress = (
            np.linalg.norm(
                vectors[
                    REGIME_ORDER[0]
                ]
                -
                vectors[
                    REGIME_ORDER[2]
                ]
            )
        )

        rows.append(
            {
                "Test_Year":
                    year,

                "Low_to_Elevated":
                    d_low_elevated,

                "Elevated_to_Stress":
                    d_elevated_stress,

                "Low_to_Stress":
                    d_low_stress,

                "Minimum_Separation":
                    min(
                        d_low_elevated,
                        d_elevated_stress,
                        d_low_stress,
                    ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# 7. Coefficient of variation
# ============================================================

def calculate_feature_stability(
    df,
):

    rows = []

    for regime in REGIME_ORDER:

        subset = df[
            df["Regime_Name"]
            == regime
        ]

        for feature in FEATURE_COLUMNS:

            mean = (
                subset[
                    feature
                ].mean()
            )

            std = (
                subset[
                    feature
                ].std(
                    ddof=1
                )
            )

            if abs(mean) > 1e-12:

                cv = (
                    abs(std / mean)
                )

            else:

                cv = np.nan

            rows.append(
                {
                    "Regime_Name":
                        regime,

                    "Feature":
                        feature,

                    "Mean":
                        mean,

                    "Std":
                        std,

                    "Abs_CV":
                        cv,
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# Figures
# ============================================================

def plot_volatility_profiles(
    df,
):

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    for regime in REGIME_ORDER:

        subset = (
            df[
                df["Regime_Name"]
                == regime
            ]
            .sort_values(
                "Test_Year"
            )
        )

        ax.plot(
            subset[
                "Test_Year"
            ],
            subset[
                "Volatility_20D"
            ],
            marker="o",
            label=regime,
        )

    ax.set_title(
        "HMM State Stability Across "
        "Walk-Forward Retraining Folds"
    )

    ax.set_xlabel(
        "Out-of-Sample Test Year"
    )

    ax.set_ylabel(
        "Learned 20-Day Volatility Mean"
    )

    ax.legend()

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    output_file = (
        FIGURE_DIR
        / "13_state_volatility_stability.png"
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


def plot_reference_distance(
    distances,
):

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    for regime in REGIME_ORDER:

        subset = (
            distances[
                distances[
                    "Regime_Name"
                ]
                == regime
            ]
            .sort_values(
                "Test_Year"
            )
        )

        ax.plot(
            subset[
                "Test_Year"
            ],
            subset[
                "Reference_Distance"
            ],
            marker="o",
            label=regime,
        )

    ax.set_title(
        "State-Centroid Distance "
        "from Canonical Regime Profile"
    )

    ax.set_xlabel(
        "Out-of-Sample Test Year"
    )

    ax.set_ylabel(
        "Standardized Euclidean Distance"
    )

    ax.legend()

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    output_file = (
        FIGURE_DIR
        / "14_state_reference_distance.png"
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


def plot_regime_separation(
    separation,
):

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    ax.plot(
        separation[
            "Test_Year"
        ],
        separation[
            "Low_to_Elevated"
        ],
        marker="o",
        label=(
            "Low ↔ Elevated"
        ),
    )

    ax.plot(
        separation[
            "Test_Year"
        ],
        separation[
            "Elevated_to_Stress"
        ],
        marker="o",
        label=(
            "Elevated ↔ Stress"
        ),
    )

    ax.plot(
        separation[
            "Test_Year"
        ],
        separation[
            "Low_to_Stress"
        ],
        marker="o",
        label=(
            "Low ↔ Stress"
        ),
    )

    ax.set_title(
        "Latent-State Separation "
        "Across Retraining Folds"
    )

    ax.set_xlabel(
        "Out-of-Sample Test Year"
    )

    ax.set_ylabel(
        "Standardized Centroid Distance"
    )

    ax.legend()

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    output_file = (
        FIGURE_DIR
        / "15_state_separation.png"
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

    df = load_data()

    print(
        "\n--- STATE STABILITY INPUT ---"
    )

    print(
        f"Rows: {len(df)}"
    )

    print(
        "Years:",
        sorted(
            df[
                "Test_Year"
            ].unique()
        ),
    )

    # --------------------------------------------------------
    # Ordering
    # --------------------------------------------------------

    ordering = (
        calculate_ordering_checks(
            df
        )
    )

    print(
        "\n--- REGIME ORDERING CHECKS ---"
    )

    print(
        ordering.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Standardized profile space
    # --------------------------------------------------------

    (
        scaled_df,
        scaled_columns,
    ) = standardized_profiles(
        df
    )

    reference = (
        calculate_reference_profiles(
            scaled_df,
            scaled_columns,
        )
    )

    # --------------------------------------------------------
    # Reference distances
    # --------------------------------------------------------

    distances = (
        calculate_reference_distances(
            scaled_df,
            scaled_columns,
            reference,
        )
    )

    print(
        "\n--- DISTANCE FROM "
        "CANONICAL REGIME PROFILE ---"
    )

    print(
        distances
        .round(4)
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Fold-to-fold drift
    # --------------------------------------------------------

    drift = (
        calculate_fold_drift(
            scaled_df,
            scaled_columns,
        )
    )

    print(
        "\n--- ADJACENT-FOLD "
        "STATE DRIFT ---"
    )

    print(
        drift
        .round(4)
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # State separation
    # --------------------------------------------------------

    separation = (
        calculate_separation(
            scaled_df,
            scaled_columns,
        )
    )

    print(
        "\n--- WITHIN-FOLD "
        "STATE SEPARATION ---"
    )

    print(
        separation
        .round(4)
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Feature stability
    # --------------------------------------------------------

    feature_stability = (
        calculate_feature_stability(
            df
        )
    )

    print(
        "\n--- FEATURE STABILITY "
        "BY REGIME ---"
    )

    print(
        feature_stability[
            [
                "Regime_Name",
                "Feature",
                "Mean",
                "Std",
                "Abs_CV",
            ]
        ]
        .round(4)
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Aggregate regime stability
    # --------------------------------------------------------

    regime_summary = (
        distances
        .groupby(
            "Regime_Name"
        )[
            "Reference_Distance"
        ]
        .agg(
            Mean_Reference_Distance="mean",
            Max_Reference_Distance="max",
        )
        .reset_index()
    )

    drift_summary = (
        drift
        .groupby(
            "Regime_Name"
        )[
            "Centroid_Drift"
        ]
        .agg(
            Mean_Fold_Drift="mean",
            Max_Fold_Drift="max",
        )
        .reset_index()
    )

    regime_summary = (
        regime_summary
        .merge(
            drift_summary,
            on="Regime_Name",
            how="left",
        )
    )

    print(
        "\n--- REGIME STABILITY SUMMARY ---"
    )

    print(
        regime_summary
        .round(4)
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Overall checks
    # --------------------------------------------------------

    total_ordering_checks = (
        ordering[
            "Total_Checks"
        ].sum()
    )

    passed_ordering_checks = (
        ordering[
            "Passed_Checks"
        ].sum()
    )

    ordering_pass_rate = (
        passed_ordering_checks
        / total_ordering_checks
    )

    print(
        "\n--- OVERALL STABILITY CHECK ---"
    )

    print(
        f"Ordering pass rate: "
        f"{ordering_pass_rate:.2%}"
    )

    print(
        f"Mean minimum state separation: "
        f"{separation['Minimum_Separation'].mean():.4f}"
    )

    print(
        f"Minimum observed state separation: "
        f"{separation['Minimum_Separation'].min():.4f}"
    )

    # --------------------------------------------------------
    # Save tables
    # --------------------------------------------------------

    ordering.to_csv(
        OUTPUT_DIR
        / "state_ordering_checks.csv",
        index=False,
    )

    distances.to_csv(
        OUTPUT_DIR
        / "state_reference_distances.csv",
        index=False,
    )

    drift.to_csv(
        OUTPUT_DIR
        / "state_fold_drift.csv",
        index=False,
    )

    separation.to_csv(
        OUTPUT_DIR
        / "state_separation.csv",
        index=False,
    )

    feature_stability.to_csv(
        OUTPUT_DIR
        / "state_feature_stability.csv",
        index=False,
    )

    regime_summary.to_csv(
        OUTPUT_DIR
        / "state_stability_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Figures
    # --------------------------------------------------------

    plot_volatility_profiles(
        df
    )

    plot_reference_distance(
        distances
    )

    plot_regime_separation(
        separation
    )

    print(
        "\nState stability analysis complete."
    )


if __name__ == "__main__":
    main()