from pathlib import Path
import argparse
import json

import joblib
import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "usdtry_direction_targets.csv"
)

MODEL_FILE = (
    ROOT
    / "models"
    / "milestone_3"
    / "usdtry_direction_rf.joblib"
)

TEMPERATURE_FILE = (
    ROOT
    / "models"
    / "milestone_3"
    / "usdtry_direction_temperature.json"
)

METADATA_FILE = (
    ROOT
    / "models"
    / "milestone_3"
    / "usdtry_direction_metadata.json"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "milestone_3"
    / "operational"
)

LATEST_FILE = (
    OUTPUT_DIR
    / "latest_direction.json"
)

HISTORY_FILE = (
    OUTPUT_DIR
    / "direction_inference_history.csv"
)


# ============================================================
# Frozen specification
# ============================================================

FEATURES = [
    "Return_1D",
    "Return_5D",
    "Volatility_5D",
    "Volatility_20D",
    "Volatility_60D",
    "MA_Distance_20D",
    "MA_Slope_20D",
    "Drawdown_60D",
]

LABELS = [
    "DOWN",
    "FLAT",
    "UP",
]

EPSILON = 1e-12


# ============================================================
# Calibration
# ============================================================

def softmax(logits):
    logits = np.asarray(
        logits,
        dtype=float,
    )

    logits = (
        logits
        - np.max(
            logits,
            axis=1,
            keepdims=True,
        )
    )

    exp_values = np.exp(
        logits
    )

    return (
        exp_values
        / exp_values.sum(
            axis=1,
            keepdims=True,
        )
    )


def temperature_scale(
    probabilities,
    temperature,
):
    if (
        not np.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError(
            "Temperature must be a positive "
            "finite number."
        )
    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    probabilities = np.clip(
        probabilities,
        EPSILON,
        1.0,
    )

    logits = np.log(
        probabilities
    )

    return softmax(
        logits / temperature
    )


# ============================================================
# Helpers
# ============================================================

def load_json(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required artifact missing: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def select_observation(
    df,
    requested_date=None,
):
    df = (
        df
        .sort_values("Date")
        .reset_index(drop=True)
    )

    if requested_date is None:
        return df.iloc[-1].copy()

    requested = pd.Timestamp(
        requested_date
    )

    selected = df[
        df["Date"] == requested
    ]

    if len(selected) == 0:
        raise ValueError(
            f"No observation found for "
            f"{requested.date()}."
        )

    if len(selected) > 1:
        raise ValueError(
            f"Duplicate observations found for "
            f"{requested.date()}."
        )

    return selected.iloc[0].copy()


def confidence_margin(
    probabilities,
):
    ordered = np.sort(
        probabilities
    )[::-1]

    return float(
        ordered[0] - ordered[1]
    )


def normalized_entropy(
    probabilities,
):
    probabilities = np.clip(
        np.asarray(
            probabilities,
            dtype=float,
        ),
        EPSILON,
        1.0,
    )

    entropy = -np.sum(
        probabilities
        * np.log(
            probabilities
        )
    )

    maximum_entropy = np.log(
        len(probabilities)
    )

    return float(
        entropy / maximum_entropy
    )


# ============================================================
# Inference
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Frozen M3 USD/TRY "
            "direction inference."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help=(
            "CSV containing Date and the "
            "eight frozen M3 features."
        ),
    )

    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help=(
            "Optional YYYY-MM-DD inference date. "
            "Defaults to latest available row."
        ),
    )

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Load frozen artifacts
    # ========================================================

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            MODEL_FILE
        )

    model = joblib.load(
        MODEL_FILE
    )

    calibration = load_json(
        TEMPERATURE_FILE
    )

    metadata = load_json(
        METADATA_FILE
    )

    temperature = float(
        calibration["temperature"]
    )

    if temperature <= 0:
        raise ValueError(
            "Temperature must be positive."
        )

    if list(model.classes_) != LABELS:
        raise ValueError(
            "Frozen model class order "
            "does not match M3 contract."
        )

    metadata_features = (
        metadata.get(
            "features",
            []
        )
    )

    if metadata_features != FEATURES:
        raise ValueError(
            "Frozen metadata feature order "
            "does not match inference contract."
        )

    model_version = metadata[
        "model_version"
    ]

    training_end = pd.Timestamp(
        metadata[
            "training"
        ][
            "end_date"
        ]
    )

    # ========================================================
    # Load inference data
    # ========================================================

    if not args.input.exists():
        raise FileNotFoundError(
            f"Inference input missing: "
            f"{args.input}"
        )

    df = pd.read_csv(
        args.input,
        parse_dates=["Date"],
    )

    required = [
        "Date",
        *FEATURES,
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing inference columns: "
            f"{missing}"
        )

    if df["Date"].duplicated().any():
        raise ValueError(
            "Inference dataset contains "
            "duplicate dates."
        )

    row = select_observation(
        df,
        requested_date=args.date,
    )

    observation_date = pd.Timestamp(
        row["Date"]
    )

    X = pd.DataFrame(
        [
            {
                feature:
                    row[feature]
                for feature in FEATURES
            }
        ],
        columns=FEATURES,
    )

    if X.isna().any().any():
        raise ValueError(
            "Selected observation contains "
            "missing M3 features."
        )

    if np.isinf(
        X.to_numpy(
            dtype=float
        )
    ).any():
        raise ValueError(
            "Selected observation contains "
            "infinite M3 features."
        )

    # ========================================================
    # Raw RF probabilities
    # ========================================================

    raw_probabilities = (
        model.predict_proba(
            X
        )[0]
    )

    if not np.isclose(
        raw_probabilities.sum(),
        1.0,
        atol=1e-10,
    ):
        raise ValueError(
            "Raw RF probabilities "
            "do not sum to one."
        )

    raw_index = int(
        np.argmax(
            raw_probabilities
        )
    )

    raw_direction = (
        LABELS[
            raw_index
        ]
    )

    # ========================================================
    # Temperature calibration
    # ========================================================

    calibrated_probabilities = (
        temperature_scale(
            raw_probabilities.reshape(
                1,
                -1,
            ),
            temperature,
        )[0]
    )

    calibrated_index = int(
        np.argmax(
            calibrated_probabilities
        )
    )

    direction = (
        LABELS[
            calibrated_index
        ]
    )

    if direction != raw_direction:
        raise ValueError(
            "Temperature scaling changed "
            "the predicted class."
        )

    confidence = float(
        calibrated_probabilities[
            calibrated_index
        ]
    )

    margin = confidence_margin(
        calibrated_probabilities
    )

    uncertainty = normalized_entropy(
        calibrated_probabilities
    )

    # ========================================================
    # Inference mode
    # ========================================================

    if observation_date > training_end:
        inference_mode = (
            "OUT_OF_SAMPLE_OPERATIONAL"
        )
    else:
        inference_mode = (
            "TRAINING_REFERENCE"
        )

    # ========================================================
    # Optional market level
    # ========================================================

    usdtry = None

    if "USDTRY" in row.index:
        if pd.notna(
            row["USDTRY"]
        ):
            usdtry = float(
                row["USDTRY"]
            )

    # ========================================================
    # Output object
    # ========================================================

    result = {
        "model_version":
            model_version,

        "observation_date":
            str(
                observation_date.date()
            ),

        "forecast_horizon":
            "NEXT_5_TRADING_DAYS",

        "target_definition": {
            "DOWN":
                "future 5D return < -0.5%",

            "FLAT":
                "-0.5% <= future 5D return <= +0.5%",

            "UP":
                "future 5D return > +0.5%",
        },

        "direction":
            direction,

        "confidence":
            confidence,

        "probability_margin":
            margin,

        "normalized_entropy":
            uncertainty,

        "calibrated_probabilities": {
            "DOWN":
                float(
                    calibrated_probabilities[0]
                ),

            "FLAT":
                float(
                    calibrated_probabilities[1]
                ),

            "UP":
                float(
                    calibrated_probabilities[2]
                ),
        },

        "raw_probabilities": {
            "DOWN":
                float(
                    raw_probabilities[0]
                ),

            "FLAT":
                float(
                    raw_probabilities[1]
                ),

            "UP":
                float(
                    raw_probabilities[2]
                ),
        },

        "temperature":
            temperature,

        "inference_mode":
            inference_mode,

        "training_cutoff":
            str(
                training_end.date()
            ),

        "feature_set":
            "BASE_8_FEATURES",

        "use_m1":
            False,

        "use_m2":
            False,
    }

    if usdtry is not None:
        result[
            "USDTRY"
        ] = usdtry

    # ========================================================
    # Save latest JSON
    # ========================================================

    with open(
        LATEST_FILE,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            result,
            handle,
            indent=2,
        )

    # ========================================================
    # Save/upsert history
    # ========================================================

    history_row = {
        "Date":
            observation_date.date(),

        "Model_Version":
            model_version,

        "Inference_Mode":
            inference_mode,

        "USDTRY":
            usdtry,

        "Direction":
            direction,

        "Confidence":
            confidence,

        "Probability_Margin":
            margin,

        "Normalized_Entropy":
            uncertainty,

        "P_DOWN":
            calibrated_probabilities[0],

        "P_FLAT":
            calibrated_probabilities[1],

        "P_UP":
            calibrated_probabilities[2],

        "Raw_P_DOWN":
            raw_probabilities[0],

        "Raw_P_FLAT":
            raw_probabilities[1],

        "Raw_P_UP":
            raw_probabilities[2],

        "Temperature":
            temperature,

        "Training_Cutoff":
            training_end.date(),
    }

    new_history = pd.DataFrame(
        [history_row]
    )

    if HISTORY_FILE.exists():

        history = pd.read_csv(
            HISTORY_FILE
        )

        # Upsert instead of creating duplicates
        # when inference is rerun for the same
        # date/model version.
        history = history[
            ~(
                (
                    history["Date"].astype(str)
                    ==
                    str(
                        observation_date.date()
                    )
                )
                &
                (
                    history[
                        "Model_Version"
                    ].astype(str)
                    ==
                    str(model_version)
                )
            )
        ]

        history = pd.concat(
            [
                history,
                new_history,
            ],
            ignore_index=True,
        )

    else:
        history = new_history

    history = (
        history
        .sort_values("Date")
        .reset_index(drop=True)
    )

    history.to_csv(
        HISTORY_FILE,
        index=False,
    )

    # ========================================================
    # Console report
    # ========================================================

    print(
        "\n"
        "=========================================="
    )

    print(
        "MILESTONE 3 — OPERATIONAL INFERENCE"
    )

    print(
        "=========================================="
    )

    print(
        f"Model version:     "
        f"{model_version}"
    )

    print(
        f"Observation date:  "
        f"{observation_date.date()}"
    )

    if usdtry is not None:
        print(
            f"USD/TRY:           "
            f"{usdtry:.6f}"
        )

    print(
        f"Inference mode:    "
        f"{inference_mode}"
    )

    print(
        f"Training cutoff:   "
        f"{training_end.date()}"
    )

    print(
        "\n--- DIRECTION ---"
    )

    print(
        f"Prediction:        "
        f"{direction}"
    )

    print(
        f"Confidence:        "
        f"{confidence * 100:.2f}%"
    )

    print(
        f"Probability margin:"
        f" {margin * 100:.2f} pp"
    )

    print(
        "\n--- CALIBRATED PROBABILITIES ---"
    )

    print(
        f"DOWN: "
        f"{calibrated_probabilities[0] * 100:.2f}%"
    )

    print(
        f"FLAT: "
        f"{calibrated_probabilities[1] * 100:.2f}%"
    )

    print(
        f"UP:   "
        f"{calibrated_probabilities[2] * 100:.2f}%"
    )

    print(
        "\n--- RAW RF PROBABILITIES ---"
    )

    print(
        f"DOWN: "
        f"{raw_probabilities[0] * 100:.2f}%"
    )

    print(
        f"FLAT: "
        f"{raw_probabilities[1] * 100:.2f}%"
    )

    print(
        f"UP:   "
        f"{raw_probabilities[2] * 100:.2f}%"
    )

    print(
        f"\nTemperature:       "
        f"{temperature:.6f}"
    )

    print(
        "\n--- SAVED ---"
    )

    print(LATEST_FILE)
    print(HISTORY_FILE)

    print(
        "\nOperational inference complete."
    )


if __name__ == "__main__":
    main()
