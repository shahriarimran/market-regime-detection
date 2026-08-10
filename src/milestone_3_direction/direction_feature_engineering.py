from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "usdtry_direction_targets.csv"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "usdtry_direction_features.csv"
)

DIAGNOSTIC_FILE = (
    ROOT
    / "outputs"
    / "milestone_3"
    / "validation"
    / "direction_feature_diagnostics.csv"
)


# ============================================================
# M3 model specification
# ============================================================

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

PRIMARY_TARGET = "Target_5D_0p5pct"

SECONDARY_TARGET = "Target_20D_2p0pct"

FUTURE_RETURN_PRIMARY = "Future_Return_5D"

LABEL_ORDER = [
    "DOWN",
    "FLAT",
    "UP",
]


# ============================================================
# Validation helpers
# ============================================================

def check_required_columns(df):

    required = {
        "Date",
        "USDTRY",
        PRIMARY_TARGET,
        SECONDARY_TARGET,
        FUTURE_RETURN_PRIMARY,
        *FEATURE_COLUMNS,
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing)}"
        )


def check_feature_quality(df):

    print(
        "\n--- FEATURE QUALITY ---"
    )

    missing = (
        df[FEATURE_COLUMNS]
        .isna()
        .sum()
    )

    print(
        "\nMissing values:"
    )

    print(
        missing.to_string()
    )

    numeric = (
        df[FEATURE_COLUMNS]
        .to_numpy(
            dtype=float
        )
    )

    infinite_counts = {}

    for i, feature in enumerate(
        FEATURE_COLUMNS
    ):

        infinite_counts[
            feature
        ] = int(
            np.isinf(
                numeric[:, i]
            ).sum()
        )

    infinite = pd.Series(
        infinite_counts
    )

    print(
        "\nInfinite values:"
    )

    print(
        infinite.to_string()
    )

    if missing.sum() > 0:

        raise ValueError(
            "Feature matrix contains "
            "missing values."
        )

    if infinite.sum() > 0:

        raise ValueError(
            "Feature matrix contains "
            "infinite values."
        )


def check_target_values(df):

    observed = set(
        df[
            PRIMARY_TARGET
        ]
        .dropna()
        .unique()
    )

    allowed = set(
        LABEL_ORDER
    )

    unexpected = (
        observed - allowed
    )

    if unexpected:

        raise ValueError(
            "Unexpected target labels: "
            f"{sorted(unexpected)}"
        )


# ============================================================
# Diagnostics
# ============================================================

def build_yearly_diagnostics(df):

    rows = []

    for year, group in (
        df.groupby("Year")
    ):

        counts = (
            group[
                PRIMARY_TARGET
            ]
            .value_counts()
            .reindex(
                LABEL_ORDER,
                fill_value=0,
            )
        )

        total = int(
            counts.sum()
        )

        for label in LABEL_ORDER:

            rows.append(
                {
                    "Year":
                        int(year),

                    "Class":
                        label,

                    "Count":
                        int(
                            counts[label]
                        ),

                    "Share":
                        (
                            counts[label]
                            / total
                            if total > 0
                            else np.nan
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Load target dataset
    # --------------------------------------------------------

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["Date"],
    )

    df = (
        df.sort_values("Date")
        .reset_index(drop=True)
    )

    check_required_columns(
        df
    )

    print(
        "\n"
        "=========================================="
    )

    print(
        "MILESTONE 3 — DIRECTION FEATURES"
    )

    print(
        "=========================================="
    )

    print(
        f"Input observations: "
        f"{len(df):,}"
    )

    print(
        f"Input period: "
        f"{df['Date'].min().date()} "
        f"to "
        f"{df['Date'].max().date()}"
    )

    # --------------------------------------------------------
    # Keep only rows with known future target
    #
    # Last 5 observations cannot have a valid
    # five-trading-day future return.
    # --------------------------------------------------------

    model_df = (
        df[
            df[
                PRIMARY_TARGET
            ].notna()
        ]
        .copy()
        .reset_index(drop=True)
    )

    print(
        f"\nRows with known 5D target: "
        f"{len(model_df):,}"
    )

    print(
        f"Rows excluded because future "
        f"5D return is unknown: "
        f"{len(df) - len(model_df):,}"
    )

    # --------------------------------------------------------
    # Add purely descriptive year column
    #
    # Year is NOT a model feature.
    # --------------------------------------------------------

    model_df["Year"] = (
        model_df[
            "Date"
        ].dt.year
    )

    # --------------------------------------------------------
    # Check feature quality
    # --------------------------------------------------------

    check_feature_quality(
        model_df
    )

    check_target_values(
        model_df
    )

    # --------------------------------------------------------
    # Construct clean modeling table
    # --------------------------------------------------------

    output_columns = [
        "Date",

        # Retained for interpretation / reporting,
        # but NOT included in the model feature list.
        "USDTRY",

        *FEATURE_COLUMNS,

        # Primary supervised target
        PRIMARY_TARGET,

        # Kept only for evaluating realized outcomes.
        # MUST NEVER be passed into model training.
        FUTURE_RETURN_PRIMARY,

        # Secondary research target
        SECONDARY_TARGET,
    ]

    final_df = (
        model_df[
            output_columns
        ]
        .copy()
    )

    # --------------------------------------------------------
    # Target balance
    # --------------------------------------------------------

    print(
        "\n--- PRIMARY TARGET ---"
    )

    print(
        "Horizon: 5 trading days"
    )

    print(
        "Threshold: ±0.5%"
    )

    counts = (
        final_df[
            PRIMARY_TARGET
        ]
        .value_counts()
        .reindex(
            LABEL_ORDER,
            fill_value=0,
        )
    )

    shares = (
        counts
        / counts.sum()
        * 100
    )

    target_summary = (
        pd.DataFrame(
            {
                "Count":
                    counts,

                "Share_pct":
                    shares,
            }
        )
    )

    print(
        target_summary
        .round(2)
        .to_string()
    )

    # --------------------------------------------------------
    # Feature summary
    # --------------------------------------------------------

    print(
        "\n--- MODEL FEATURES ---"
    )

    for i, feature in enumerate(
        FEATURE_COLUMNS,
        start=1,
    ):

        print(
            f"{i}. {feature}"
        )

    print(
        "\nFeature count:",
        len(FEATURE_COLUMNS),
    )

    # --------------------------------------------------------
    # Feature statistics by future class
    #
    # Descriptive only.
    # --------------------------------------------------------

    print(
        "\n--- FEATURE MEANS BY FUTURE CLASS ---"
    )

    class_means = (
        final_df
        .groupby(
            PRIMARY_TARGET
        )[FEATURE_COLUMNS]
        .mean()
        .reindex(
            LABEL_ORDER
        )
    )

    print(
        class_means
        .round(5)
        .to_string()
    )

    # --------------------------------------------------------
    # Annual target distribution
    # --------------------------------------------------------

    diagnostics = (
        build_yearly_diagnostics(
            model_df
        )
    )

    print(
        "\n--- YEARLY TARGET BALANCE ---"
    )

    yearly = (
        diagnostics
        .pivot(
            index="Year",
            columns="Class",
            values="Share",
        )
        .reindex(
            columns=LABEL_ORDER
        )
        * 100
    )

    print(
        yearly
        .round(2)
        .to_string()
    )

    # --------------------------------------------------------
    # Leakage contract
    # --------------------------------------------------------

    print(
        "\n--- LEAKAGE CONTRACT ---"
    )

    print(
        "Model inputs contain only "
        "information available at time t."
    )

    print(
        "Future_Return_5D is retained "
        "for evaluation only."
    )

    print(
        "Target columns must never be "
        "included in X."
    )

    print(
        "USDTRY level and Year are "
        "not model features."
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    DIAGNOSTIC_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    diagnostics.to_csv(
        DIAGNOSTIC_FILE,
        index=False,
    )

    print(
        "\n--- SAVED ---"
    )

    print(
        OUTPUT_FILE
    )

    print(
        DIAGNOSTIC_FILE
    )

    print(
        "\nDirection feature "
        "engineering complete."
    )


if __name__ == "__main__":
    main()