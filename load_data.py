from pathlib import Path

import pandas as pd

def load_usdtry_csv(csv_path: str | Path) -> pd.DataFrame:
    """
    Load and validate the cleaned USD/TRY historical dataset.

    Expected columns:
        Date
        USDTRY
    """

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Read CSV
    df = pd.read_csv(csv_path)

    required_columns = {"Date", "USDTRY"}

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # Keep only the columns we need
    df = df[["Date", "USDTRY"]].copy()

    df["Date"] = pd.to_datetime(
        df["Date"],
        format="%m/%d/%Y",
        errors="raise"
    )

    # Ensure exchange-rate values are numeric
    df["USDTRY"] = pd.to_numeric(
        df["USDTRY"],
        errors="raise"
    )

    # Sort chronologically
    df = df.sort_values("Date").reset_index(drop=True)

    if df["Date"].duplicated().any():
        duplicates = df.loc[
            df["Date"].duplicated(keep=False),
            "Date"
        ]

        raise ValueError(
            f"Duplicate dates detected:\n{duplicates}"
        )

    if df[["Date", "USDTRY"]].isna().any().any():
        raise ValueError("Missing Date or USDTRY values detected.")

    return df

if __name__ == "__main__":

    csv_file = "Financial_Trackers_AutoRun_ML_clean.csv"

    df = load_usdtry_csv(csv_file)

    print("\nDataset loaded successfully.")
    print(f"Rows: {len(df):,}")
    print(f"Start date: {df['Date'].min().date()}")
    print(f"End date:   {df['Date'].max().date()}")

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nLast 5 rows:")
    print(df.tail())

    print("\nData types:")
    print(df.dtypes)

    print("\nUSD/TRY summary:")
    print(df["USDTRY"].describe())