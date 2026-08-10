from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

M2_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    M2_DIR
    / "data"
    / "usdtry_anomaly_features.csv"
)

OUTPUT_DIR = (
    M2_DIR
    / "outputs"
    / "statistical_baseline"
)

SCORES_FILE = (
    OUTPUT_DIR
    / "statistical_baseline_scores.csv"
)

TOP_ANOMALIES_FILE = (
    OUTPUT_DIR
    / "top_statistical_anomalies.csv"
)


# ============================================================
# Baseline configuration
#
# These are deliberately transparent engineering thresholds.
# They are NOT claimed to be universal financial thresholds.
# ============================================================

RETURN_Z_THRESHOLD = 4.0

VOLATILITY_RATIO_THRESHOLD = 2.0

DRAWDOWN_CHANGE_THRESHOLD = 0.05

ABS_RETURN_THRESHOLD = 0.04


# ============================================================
# Load
# ============================================================

def load_data():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["Date"],
    )

    required = [
        "Date",
        "USDTRY",
        "Return_1D",
        "Absolute_Return_1D",
        "Return_ZScore_20D",
        "Volatility_Ratio_5D_60D",
        "Drawdown_Change_5D",
    ]

    missing = (
        set(required)
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    return (
        df
        .sort_values("Date")
        .reset_index(drop=True)
    )


# ============================================================
# Baseline
# ============================================================

def calculate_baseline(df):

    df = df.copy()

    # --------------------------------------------------------
    # Individual rule magnitudes
    #
    # Score = 1 means exactly at the rule threshold.
    # --------------------------------------------------------

    df["Score_Return_Z"] = (
        df["Return_ZScore_20D"].abs()
        / RETURN_Z_THRESHOLD
    )

    df["Score_Volatility_Ratio"] = (
        df["Volatility_Ratio_5D_60D"]
        / VOLATILITY_RATIO_THRESHOLD
    )

    df["Score_Drawdown_Change"] = (
        df["Drawdown_Change_5D"].abs()
        / DRAWDOWN_CHANGE_THRESHOLD
    )

    df["Score_Absolute_Return"] = (
        df["Absolute_Return_1D"]
        / ABS_RETURN_THRESHOLD
    )

    # --------------------------------------------------------
    # Overall score
    #
    # Max-rule baseline:
    # an observation is unusual if ANY transparent
    # market-shock indicator becomes extreme.
    # --------------------------------------------------------

    score_columns = [
        "Score_Return_Z",
        "Score_Volatility_Ratio",
        "Score_Drawdown_Change",
        "Score_Absolute_Return",
    ]

    df["Baseline_Anomaly_Score"] = (
        df[score_columns]
        .max(axis=1)
    )

    # --------------------------------------------------------
    # Number of rules breached
    # --------------------------------------------------------

    df["Rule_Return_Z"] = (
        df["Score_Return_Z"] >= 1.0
    )

    df["Rule_Volatility_Ratio"] = (
        df["Score_Volatility_Ratio"] >= 1.0
    )

    df["Rule_Drawdown_Change"] = (
        df["Score_Drawdown_Change"] >= 1.0
    )

    df["Rule_Absolute_Return"] = (
        df["Score_Absolute_Return"] >= 1.0
    )

    rule_columns = [
        "Rule_Return_Z",
        "Rule_Volatility_Ratio",
        "Rule_Drawdown_Change",
        "Rule_Absolute_Return",
    ]

    df["Rules_Breached"] = (
        df[rule_columns]
        .sum(axis=1)
    )

    # --------------------------------------------------------
    # Baseline classification
    # --------------------------------------------------------

    df["Baseline_Anomaly"] = (
        df["Rules_Breached"] >= 1
    )

    return df


# ============================================================
# Explain dominant reason
# ============================================================

def assign_primary_reason(df):

    score_map = {
        "Score_Return_Z":
            "LOCAL_RETURN_SHOCK",

        "Score_Volatility_Ratio":
            "VOLATILITY_ACCELERATION",

        "Score_Drawdown_Change":
            "DRAWDOWN_SHIFT",

        "Score_Absolute_Return":
            "LARGE_ABSOLUTE_RETURN",
    }

    score_columns = list(
        score_map.keys()
    )

    dominant = (
        df[score_columns]
        .idxmax(axis=1)
    )

    df["Primary_Reason"] = (
        dominant.map(score_map)
    )

    return df


# ============================================================
# Summary
# ============================================================

def print_summary(df):

    n = len(df)

    anomaly_count = int(
        df["Baseline_Anomaly"].sum()
    )

    anomaly_rate = (
        anomaly_count / n
    )

    print(
        "\n--- STATISTICAL ANOMALY BASELINE ---"
    )

    print(
        f"Observations: {n:,}"
    )

    print(
        f"Baseline anomalies: "
        f"{anomaly_count:,}"
    )

    print(
        f"Baseline anomaly rate: "
        f"{anomaly_rate:.2%}"
    )

    print(
        "\n--- RULE BREACH COUNTS ---"
    )

    rules = {
        "|Return Z20| >= 4":
            "Rule_Return_Z",

        "Volatility ratio >= 2":
            "Rule_Volatility_Ratio",

        "|5D drawdown change| >= 5%":
            "Rule_Drawdown_Change",

        "|1D return| >= 4%":
            "Rule_Absolute_Return",
    }

    for label, column in rules.items():

        count = int(
            df[column].sum()
        )

        print(
            f"{label:32s}: "
            f"{count:4d} "
            f"({count / n:.2%})"
        )

    print(
        "\n--- RULES BREACHED PER OBSERVATION ---"
    )

    print(
        df["Rules_Breached"]
        .value_counts()
        .sort_index()
        .to_string()
    )


# ============================================================
# Top anomalies
# ============================================================

def print_top_anomalies(df):

    columns = [
        "Date",
        "USDTRY",
        "Return_1D",
        "Return_ZScore_20D",
        "Volatility_Ratio_5D_60D",
        "Drawdown_Change_5D",
        "Baseline_Anomaly_Score",
        "Rules_Breached",
        "Primary_Reason",
    ]

    top = (
        df
        .sort_values(
            "Baseline_Anomaly_Score",
            ascending=False,
        )
        .head(25)
    )

    print(
        "\n--- TOP 25 STATISTICAL ANOMALIES ---"
    )

    print(
        top[columns]
        .round(4)
        .to_string(index=False)
    )

    return top


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_data()

    df = calculate_baseline(df)

    df = assign_primary_reason(df)

    print_summary(df)

    top = print_top_anomalies(df)

    df.to_csv(
        SCORES_FILE,
        index=False,
    )

    top.to_csv(
        TOP_ANOMALIES_FILE,
        index=False,
    )

    print(
        "\n--- OUTPUT ---"
    )

    print(SCORES_FILE)
    print(TOP_ANOMALIES_FILE)

    print(
        "\nStatistical baseline complete."
    )


if __name__ == "__main__":
    main()