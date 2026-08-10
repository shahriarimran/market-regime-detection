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
    / "usdtry_features.csv"
)

OUTPUT_DATA = (
    ROOT
    / "data"
    / "processed"
    / "usdtry_direction_targets.csv"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "milestone_3"
    / "validation"
)

BALANCE_FILE = (
    OUTPUT_DIR
    / "target_balance.csv"
)

YEARLY_BALANCE_FILE = (
    OUTPUT_DIR
    / "target_balance_by_year.csv"
)


# ============================================================
# Candidate thresholds
# ============================================================

FIXED_THRESHOLDS = {
    5: [
        0.005,
        0.010,
        0.015,
        0.020,
    ],
    20: [
        0.010,
        0.020,
        0.030,
        0.050,
    ],
}


# ============================================================
# Label helpers
# ============================================================

def classify_direction(
    future_return,
    threshold,
):

    labels = np.full(
        len(future_return),
        None,
        dtype=object,
    )

    valid = future_return.notna()

    labels[
        valid
        & (future_return > threshold)
    ] = "UP"

    labels[
        valid
        & (future_return < -threshold)
    ] = "DOWN"

    labels[
        valid
        & (
            future_return.abs()
            <= threshold
        )
    ] = "FLAT"

    return labels


def classify_volatility_adjusted(
    future_return,
    volatility,
    horizon,
    sigma_threshold=1.0,
):

    expected_move = (
        volatility
        * np.sqrt(horizon)
    )

    normalized_move = (
        future_return
        / expected_move
    )

    labels = np.full(
        len(future_return),
        None,
        dtype=object,
    )

    valid = (
        future_return.notna()
        & expected_move.notna()
        & (expected_move > 0)
    )

    labels[
        valid
        & (
            normalized_move
            > sigma_threshold
        )
    ] = "UP"

    labels[
        valid
        & (
            normalized_move
            < -sigma_threshold
        )
    ] = "DOWN"

    labels[
        valid
        & (
            normalized_move.abs()
            <= sigma_threshold
        )
    ] = "FLAT"

    return (
        labels,
        normalized_move,
    )


# ============================================================
# Target diagnostics
# ============================================================

def summarize_target(
    df,
    target_column,
):

    valid = df[
        target_column
    ].dropna()

    counts = (
        valid
        .value_counts()
        .reindex(
            [
                "DOWN",
                "FLAT",
                "UP",
            ],
            fill_value=0,
        )
    )

    total = counts.sum()

    rows = []

    for label in [
        "DOWN",
        "FLAT",
        "UP",
    ]:

        count = int(
            counts[label]
        )

        share = (
            count / total
            if total
            else np.nan
        )

        rows.append(
            {
                "Target":
                    target_column,

                "Class":
                    label,

                "Count":
                    count,

                "Share":
                    share,
            }
        )

    return rows


def summarize_target_by_year(
    df,
    target_column,
):

    rows = []

    valid = df[
        df[
            target_column
        ].notna()
    ].copy()

    for year, group in (
        valid.groupby("Year")
    ):

        counts = (
            group[
                target_column
            ]
            .value_counts()
            .reindex(
                [
                    "DOWN",
                    "FLAT",
                    "UP",
                ],
                fill_value=0,
            )
        )

        total = counts.sum()

        for label in [
            "DOWN",
            "FLAT",
            "UP",
        ]:

            rows.append(
                {
                    "Target":
                        target_column,

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
                            if total
                            else np.nan
                        ),
                }
            )

    return rows


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load M1 feature dataset
    # --------------------------------------------------------

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["Date"],
    )

    df = (
        df
        .sort_values("Date")
        .reset_index(drop=True)
    )

    required = {
        "Date",
        "USDTRY",
        "Volatility_20D",
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

    df["Year"] = (
        df["Date"].dt.year
    )

    print(
        "\n"
        "=========================================="
    )

    print(
        "MILESTONE 3 — TARGET ENGINEERING"
    )

    print(
        "=========================================="
    )

    print(
        f"Observations: {len(df):,}"
    )

    print(
        f"Period: "
        f"{df['Date'].min().date()} "
        f"to "
        f"{df['Date'].max().date()}"
    )

    # --------------------------------------------------------
    # Forward returns
    # --------------------------------------------------------

    for horizon in [
        5,
        20,
    ]:

        column = (
            f"Future_Return_{horizon}D"
        )

        df[column] = (
            df["USDTRY"]
            .shift(-horizon)
            / df["USDTRY"]
            - 1
        )

    # --------------------------------------------------------
    # Fixed-threshold targets
    # --------------------------------------------------------

    target_columns = []

    for (
        horizon,
        thresholds,
    ) in FIXED_THRESHOLDS.items():

        return_column = (
            f"Future_Return_{horizon}D"
        )

        for threshold in thresholds:

            threshold_pct = int(
                round(
                    threshold * 1000
                )
            )

            # Example:
            # 0.010 -> 10 -> 1.0%
            label_string = (
                f"{threshold * 100:.1f}"
                .replace(".", "p")
            )

            target_column = (
                f"Target_{horizon}D_"
                f"{label_string}pct"
            )

            df[
                target_column
            ] = classify_direction(
                df[
                    return_column
                ],
                threshold,
            )

            target_columns.append(
                target_column
            )

    # --------------------------------------------------------
    # Volatility-adjusted targets
    # --------------------------------------------------------

    for horizon in [
        5,
        20,
    ]:

        return_column = (
            f"Future_Return_{horizon}D"
        )

        (
            labels,
            normalized_move,
        ) = classify_volatility_adjusted(
            future_return=df[
                return_column
            ],
            volatility=df[
                "Volatility_20D"
            ],
            horizon=horizon,
            sigma_threshold=1.0,
        )

        normalized_column = (
            f"Future_Move_Z_{horizon}D"
        )

        target_column = (
            f"Target_{horizon}D_1sigma"
        )

        df[
            normalized_column
        ] = normalized_move

        df[
            target_column
        ] = labels

        target_columns.append(
            target_column
        )

    # --------------------------------------------------------
    # Global class balance
    # --------------------------------------------------------

    balance_rows = []

    yearly_rows = []

    for target in target_columns:

        balance_rows.extend(
            summarize_target(
                df,
                target,
            )
        )

        yearly_rows.extend(
            summarize_target_by_year(
                df,
                target,
            )
        )

    balance = pd.DataFrame(
        balance_rows
    )

    yearly_balance = pd.DataFrame(
        yearly_rows
    )

    # --------------------------------------------------------
    # Print compact diagnostics
    # --------------------------------------------------------

    print(
        "\n--- TARGET CLASS BALANCE ---"
    )

    pivot = (
        balance
        .pivot(
            index="Target",
            columns="Class",
            values="Share",
        )
        .fillna(0)
    )

    pivot = (
        pivot[
            [
                "DOWN",
                "FLAT",
                "UP",
            ]
        ]
        * 100
    )

    print(
        pivot
        .round(2)
        .to_string()
    )

    # --------------------------------------------------------
    # Future-return diagnostics
    # --------------------------------------------------------

    print(
        "\n--- FUTURE RETURN SUMMARY ---"
    )

    summary = (
        df[
            [
                "Future_Return_5D",
                "Future_Return_20D",
            ]
        ]
        .describe(
            percentiles=[
                0.05,
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
            ]
        )
    )

    print(
        summary
        .round(5)
        .to_string()
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_DATA,
        index=False,
    )

    balance.to_csv(
        BALANCE_FILE,
        index=False,
    )

    yearly_balance.to_csv(
        YEARLY_BALANCE_FILE,
        index=False,
    )

    print(
        "\n--- SAVED ---"
    )

    print(
        OUTPUT_DATA
    )

    print(
        BALANCE_FILE
    )

    print(
        YEARLY_BALANCE_FILE
    )

    print(
        "\nTarget engineering complete."
    )


if __name__ == "__main__":
    main()