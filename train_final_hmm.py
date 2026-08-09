from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from chronological_validation import (
    FEATURE_COLUMNS,
    causal_filter,
    fit_best_hmm,
    get_state_mapping,
)


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = Path(
    "data/processed/usdtry_features.csv"
)

MODEL_DIR = Path("models")

MODEL_FILE = (
    MODEL_DIR /
    "usdtry_hmm3.joblib"
)

METADATA_FILE = (
    MODEL_DIR /
    "usdtry_hmm3_metadata.json"
)

PROFILE_FILE = (
    MODEL_DIR /
    "usdtry_hmm3_state_profiles.csv"
)


# ============================================================
# Main
# ============================================================

def main():

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load complete validated dataset
    # --------------------------------------------------------

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["Date"],
    )

    df = (
        df.sort_values("Date")
        .reset_index(drop=True)
    )

    print(
        "\n--- FINAL HMM TRAINING ---"
    )

    print(
        f"Observations: {len(df):,}"
    )

    print(
        f"Start date: "
        f"{df['Date'].min().date()}"
    )

    print(
        f"End date: "
        f"{df['Date'].max().date()}"
    )

    # --------------------------------------------------------
    # Fit scaler
    # --------------------------------------------------------

    scaler = StandardScaler()

    X = scaler.fit_transform(
        df[FEATURE_COLUMNS]
    )

    # --------------------------------------------------------
    # Fit best validated architecture
    # --------------------------------------------------------

    (
        model,
        log_likelihood,
        best_seed,
    ) = fit_best_hmm(
        X
    )

    print(
        "\n--- FINAL MODEL ---"
    )

    print(
        f"Best seed: {best_seed}"
    )

    print(
        f"Log-likelihood: "
        f"{log_likelihood:.4f}"
    )

    print(
        f"Log-likelihood / observation: "
        f"{log_likelihood / len(df):.4f}"
    )

    # --------------------------------------------------------
    # Verify causal filter
    # --------------------------------------------------------

    (
        filtered_probabilities,
        causal_log_likelihood,
    ) = causal_filter(
        model,
        X,
    )

    difference = abs(
        causal_log_likelihood
        - model.score(X)
    )

    print(
        "\n--- CAUSAL FILTER CHECK ---"
    )

    print(
        f"hmmlearn score: "
        f"{model.score(X):.8f}"
    )

    print(
        f"Causal score:   "
        f"{causal_log_likelihood:.8f}"
    )

    print(
        f"Difference:     "
        f"{difference:.12f}"
    )

    if difference > 1e-4:

        raise RuntimeError(
            "Causal filtering validation failed."
        )

    # --------------------------------------------------------
    # Canonical state interpretation
    # --------------------------------------------------------

    (
        state_names,
        emission_means,
    ) = get_state_mapping(
        model,
        scaler,
    )

    print(
        "\n--- FINAL STATE INTERPRETATION ---"
    )

    for state in sorted(
        state_names
    ):

        print(
            f"State {state}: "
            f"{state_names[state]}"
        )

    # --------------------------------------------------------
    # Store state profiles
    # --------------------------------------------------------

    profiles = (
        emission_means.copy()
    )

    profiles["Regime_Name"] = (
        profiles[
            "HMM_State"
        ]
        .map(state_names)
    )

    profiles.to_csv(
        PROFILE_FILE,
        index=False,
    )

    print(
        "\n--- FINAL STATE PROFILES ---"
    )

    print(
        profiles
        .round(5)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Final filtered state
    # --------------------------------------------------------

    final_probability = (
        filtered_probabilities[-1]
    )

    final_state = int(
        np.argmax(
            final_probability
        )
    )

    final_confidence = float(
        np.max(
            final_probability
        )
    )

    print(
        "\n--- TRAINING-CUTOFF STATE ---"
    )

    print(
        f"Date: "
        f"{df.iloc[-1]['Date'].date()}"
    )

    print(
        f"State: "
        f"{final_state}"
    )

    print(
        f"Regime: "
        f"{state_names[final_state]}"
    )

    print(
        f"Confidence: "
        f"{final_confidence:.2%}"
    )

    # --------------------------------------------------------
    # Save model package
    # --------------------------------------------------------

    artifact = {
        "model": model,
        "scaler": scaler,
        "feature_columns": FEATURE_COLUMNS,
        "state_names": state_names,

        # Required so future inference knows
        # exactly when this model stopped training.
        "training_start_date":
            df["Date"].min(),

        "training_end_date":
            df["Date"].max(),

        "training_observations":
            len(df),

        "best_seed":
            best_seed,

        "training_log_likelihood":
            log_likelihood,

        "final_filtered_probability":
            final_probability,
    }

    joblib.dump(
        artifact,
        MODEL_FILE,
    )

    # --------------------------------------------------------
    # Human-readable metadata
    # --------------------------------------------------------

    metadata = {
        "model_type":
            "GaussianHMM",

        "n_states":
            int(model.n_components),

        "covariance_type":
            model.covariance_type,

        "training_start_date":
            str(
                df[
                    "Date"
                ].min().date()
            ),

        "training_end_date":
            str(
                df[
                    "Date"
                ].max().date()
            ),

        "training_observations":
            int(len(df)),

        "best_seed":
            int(best_seed),

        "features":
            list(FEATURE_COLUMNS),

        "state_names": {
            str(key): value
            for key, value
            in state_names.items()
        },

        "validation_status":
            "PASSED",

        "acceptance_gates":
            "7/7",

        "intended_use":
            (
                "Low-frequency USD/TRY "
                "market-regime detection"
            ),

        "not_intended_for":
            (
                "Direct BUY/SELL forecasting"
            ),
    }

    METADATA_FILE.write_text(
        json.dumps(
            metadata,
            indent=4,
        ),
        encoding="utf-8",
    )

    print(
        "\n--- SAVED FINAL MODEL ---"
    )

    print(MODEL_FILE)
    print(METADATA_FILE)
    print(PROFILE_FILE)

    print(
        "\nFinal HMM training complete."
    )


if __name__ == "__main__":
    main()