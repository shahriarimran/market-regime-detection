from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from chronological_validation import (
    causal_filter,
)


# ============================================================
# Configuration
# ============================================================

MODEL_FILE = Path(
    "models/usdtry_hmm3.joblib"
)

INPUT_FILE = Path(
    "data/processed/usdtry_features.csv"
)

OUTPUT_DIR = Path(
    "outputs/operational"
)

LATEST_JSON = (
    OUTPUT_DIR /
    "latest_regime.json"
)

HISTORY_CSV = (
    OUTPUT_DIR /
    "regime_inference_history.csv"
)


# ============================================================
# Machine-readable codes
# ============================================================

REGIME_CODES = {
    "Low-Volatility Trend":
        "LOW_VOL",

    "Elevated-Volatility Transition":
        "ELEVATED_VOL",

    "High-Volatility Stress":
        "HIGH_VOL_STRESS",
}


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load frozen validated model
    # --------------------------------------------------------

    if not MODEL_FILE.exists():

        raise FileNotFoundError(
            f"Final model not found: "
            f"{MODEL_FILE}"
        )

    artifact = joblib.load(
        MODEL_FILE
    )

    model = artifact["model"]

    scaler = artifact["scaler"]

    feature_columns = artifact[
        "feature_columns"
    ]

    state_names = artifact[
        "state_names"
    ]

    training_end_date = (
        pd.Timestamp(
            artifact[
                "training_end_date"
            ]
        )
    )

    # --------------------------------------------------------
    # Load latest feature data
    # --------------------------------------------------------

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["Date"],
    )

    df = (
        df.sort_values("Date")
        .reset_index(drop=True)
    )

    missing = (
        set(feature_columns)
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            f"Missing required features: "
            f"{sorted(missing)}"
        )

    # --------------------------------------------------------
    # Fixed-scaler transform
    #
    # DO NOT refit scaler here.
    # --------------------------------------------------------

    X = scaler.transform(
        df[feature_columns]
    )

    # --------------------------------------------------------
    # Causal filtering
    #
    # Each date uses observations only
    # through that date.
    # --------------------------------------------------------

    probabilities, _ = (
        causal_filter(
            model,
            X,
        )
    )

    states = (
        probabilities.argmax(
            axis=1
        )
    )

    confidence = (
        probabilities.max(
            axis=1
        )
    )

    df["HMM_State"] = states

    df["Regime_Name"] = [
        state_names[
            int(state)
        ]
        for state in states
    ]

    df["Regime_Code"] = (
        df["Regime_Name"]
        .map(REGIME_CODES)
    )

    df["Regime_Confidence"] = (
        confidence
    )

    # --------------------------------------------------------
    # Add canonical probabilities
    # --------------------------------------------------------

    reverse_names = {
        name: state
        for state, name
        in state_names.items()
    }

    low_state = reverse_names[
        "Low-Volatility Trend"
    ]

    elevated_state = reverse_names[
        "Elevated-Volatility Transition"
    ]

    stress_state = reverse_names[
        "High-Volatility Stress"
    ]

    df["P_LOW_VOL"] = (
        probabilities[
            :,
            low_state,
        ]
    )

    df["P_ELEVATED_VOL"] = (
        probabilities[
            :,
            elevated_state,
        ]
    )

    df["P_HIGH_VOL_STRESS"] = (
        probabilities[
            :,
            stress_state,
        ]
    )

    # --------------------------------------------------------
    # Latest observation
    # --------------------------------------------------------

    latest = df.iloc[-1]

    latest_date = pd.Timestamp(
        latest["Date"]
    )

    latest_state = int(
        latest["HMM_State"]
    )

    # --------------------------------------------------------
    # Determine whether this observation
    # is genuinely post-training.
    # --------------------------------------------------------

    if latest_date > training_end_date:

        inference_mode = (
            "OUT_OF_SAMPLE_OPERATIONAL"
        )

    else:

        inference_mode = (
            "TRAINING_REFERENCE"
        )

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print(
        "\n--- OPERATIONAL REGIME INFERENCE ---"
    )

    print(
        f"Observation date: "
        f"{latest_date.date()}"
    )

    print(
        f"Model training cutoff: "
        f"{training_end_date.date()}"
    )

    print(
        f"Inference mode: "
        f"{inference_mode}"
    )

    print(
        f"\nUSD/TRY: "
        f"{latest['USDTRY']:.4f}"
    )

    print(
        f"State: "
        f"{latest_state}"
    )

    print(
        f"Regime: "
        f"{latest['Regime_Name']}"
    )

    print(
        f"Code: "
        f"{latest['Regime_Code']}"
    )

    print(
        f"Confidence: "
        f"{latest['Regime_Confidence']:.2%}"
    )

    print(
        "\nPosterior probabilities:"
    )

    print(
        f"LOW_VOL: "
        f"{latest['P_LOW_VOL']:.2%}"
    )

    print(
        f"ELEVATED_VOL: "
        f"{latest['P_ELEVATED_VOL']:.2%}"
    )

    print(
        f"HIGH_VOL_STRESS: "
        f"{latest['P_HIGH_VOL_STRESS']:.2%}"
    )

    # --------------------------------------------------------
    # Operational JSON
    # --------------------------------------------------------

    output = {
        "date":
            str(
                latest_date.date()
            ),

        "model_training_cutoff":
            str(
                training_end_date.date()
            ),

        "inference_mode":
            inference_mode,

        "usdtry":
            float(
                latest["USDTRY"]
            ),

        "state":
            latest_state,

        "regime_code":
            latest["Regime_Code"],

        "regime_name":
            latest["Regime_Name"],

        "confidence":
            float(
                latest[
                    "Regime_Confidence"
                ]
            ),

        "probabilities": {
            "LOW_VOL":
                float(
                    latest[
                        "P_LOW_VOL"
                    ]
                ),

            "ELEVATED_VOL":
                float(
                    latest[
                        "P_ELEVATED_VOL"
                    ]
                ),

            "HIGH_VOL_STRESS":
                float(
                    latest[
                        "P_HIGH_VOL_STRESS"
                    ]
                ),
        },
    }

    LATEST_JSON.write_text(
        json.dumps(
            output,
            indent=4,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Save inference history
    # --------------------------------------------------------

    history_columns = [
        "Date",
        "USDTRY",
        "HMM_State",
        "Regime_Code",
        "Regime_Name",
        "Regime_Confidence",
        "P_LOW_VOL",
        "P_ELEVATED_VOL",
        "P_HIGH_VOL_STRESS",
    ]

    df[
        history_columns
    ].to_csv(
        HISTORY_CSV,
        index=False,
    )

    print(
        "\n--- OUTPUT ---"
    )

    print(
        f"Latest JSON: "
        f"{LATEST_JSON}"
    )

    print(
        f"Inference history: "
        f"{HISTORY_CSV}"
    )

    print(
        "\nOperational inference complete."
    )


if __name__ == "__main__":
    main()