from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT
    / "outputs"
    / "milestone_3"
    / "validation"
    / "target_balance_by_year.csv"
)

OUTPUT_FILE = (
    ROOT
    / "outputs"
    / "milestone_3"
    / "validation"
    / "target_selection_summary.csv"
)


CANDIDATES = [
    "Target_5D_0p5pct",
    "Target_5D_1p0pct",
    "Target_20D_1p0pct",
    "Target_20D_2p0pct",
    "Target_20D_3p0pct",
    "Target_20D_1sigma",
]


def main():

    df = pd.read_csv(INPUT_FILE)

    df = df[
        df["Target"].isin(CANDIDATES)
    ].copy()

    # Ignore incomplete 2026 for strict annual comparison.
    full_years = df[
        (df["Year"] >= 2018)
        & (df["Year"] <= 2025)
    ].copy()

    rows = []

    for target in CANDIDATES:

        subset = full_years[
            full_years["Target"] == target
        ]

        pivot = (
            subset
            .pivot(
                index="Year",
                columns="Class",
                values="Share",
            )
            .fillna(0)
        )

        for cls in ["DOWN", "FLAT", "UP"]:
            if cls not in pivot.columns:
                pivot[cls] = 0.0

        minimum_class_share = (
            pivot[
                ["DOWN", "FLAT", "UP"]
            ]
            .min(axis=1)
        )

        rows.append(
            {
                "Target": target,

                "Min_DOWN_Annual":
                    pivot["DOWN"].min(),

                "Min_FLAT_Annual":
                    pivot["FLAT"].min(),

                "Min_UP_Annual":
                    pivot["UP"].min(),

                "Mean_DOWN_Annual":
                    pivot["DOWN"].mean(),

                "Mean_FLAT_Annual":
                    pivot["FLAT"].mean(),

                "Mean_UP_Annual":
                    pivot["UP"].mean(),

                "Worst_Annual_Class_Share":
                    minimum_class_share.min(),

                "Years_With_Class_Below_5pct":
                    int(
                        (
                            minimum_class_share
                            < 0.05
                        ).sum()
                    ),
            }
        )

    summary = pd.DataFrame(rows)

    summary = summary.sort_values(
        [
            "Years_With_Class_Below_5pct",
            "Worst_Annual_Class_Share",
        ],
        ascending=[
            True,
            False,
        ],
    )

    print(
        "\n"
        "=========================================="
    )

    print(
        "MILESTONE 3 — TARGET SELECTION"
    )

    print(
        "=========================================="
    )

    print(
        "\nAnnual class stability "
        "(2017–2025 full years):\n"
    )

    display = summary.copy()

    percentage_columns = [
        "Min_DOWN_Annual",
        "Min_FLAT_Annual",
        "Min_UP_Annual",
        "Mean_DOWN_Annual",
        "Mean_FLAT_Annual",
        "Mean_UP_Annual",
        "Worst_Annual_Class_Share",
    ]

    for column in percentage_columns:
        display[column] *= 100

    print(
        display
        .round(2)
        .to_string(index=False)
    )

    print(
        "\n--- YEARLY CLASS SHARES: 5D ±0.5% ---"
    )

    candidate = full_years[
        full_years["Target"]
        == "Target_5D_0p5pct"
    ]

    pivot = (
        candidate
        .pivot(
            index="Year",
            columns="Class",
            values="Share",
        )
        .fillna(0)
        .reindex(
            columns=[
                "DOWN",
                "FLAT",
                "UP",
            ],
            fill_value=0,
        )
        * 100
    )

    print(
        pivot
        .round(2)
        .to_string()
    )

    print(
        "\n--- YEARLY CLASS SHARES: 20D ±2.0% ---"
    )

    candidate = full_years[
        full_years["Target"]
        == "Target_20D_2p0pct"
    ]

    pivot = (
        candidate
        .pivot(
            index="Year",
            columns="Class",
            values="Share",
        )
        .fillna(0)
        .reindex(
            columns=[
                "DOWN",
                "FLAT",
                "UP",
            ],
            fill_value=0,
        )
        * 100
    )

    print(
        pivot
        .round(2)
        .to_string()
    )

    summary.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"\nSaved: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()