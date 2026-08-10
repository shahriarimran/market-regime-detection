from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from .prepare_anomaly_features import (
    M1_FEATURES,
    add_anomaly_features,
    validate_features,
)


# ============================================================
# Paths
# ============================================================

M2_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = M2_DIR.parents[1]

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "usdtry_features.csv"
)

MODEL_DIR = (
    M2_DIR
    / "models"
)

MODEL_FILE = (
    MODEL_DIR
    / "usdtry_isolation_forest.joblib"
)

METADATA_FILE = (
    MODEL_DIR
    / "usdtry_anomaly_metadata.json"
)

TRAINING_SCORES_FILE = (
    MODEL_DIR
    / "usdtry_if_training_scores.npy"
)

TRAINING_REFERENCE_FILE = (
    M2_DIR
    / "outputs"
    / "final_training"
    / "latest_training_reference.json"
)

OUTPUT_DIR = (
    M2_DIR
    / "outputs"
    / "operational"
)

LATEST_OUTPUT_FILE = (
    OUTPUT_DIR
    / "latest_anomaly.json"
)

HISTORY_OUTPUT_FILE = (
    OUTPUT_DIR
    / "anomaly_inference_history.csv"
)


# ============================================================
# Load frozen artifacts
# ============================================================

def load_artifacts():

    required = [
        FEATURE_FILE,
        MODEL_FILE,
        METADATA_FILE,
        TRAINING_SCORES_FILE,
    ]

    for path in required:

        if not path.exists():

            raise FileNotFoundError(
                f"Required artifact not found:\n{path}"
            )

    artifact = joblib.load(
        MODEL_FILE
    )

    if not isinstance(artifact, dict):
        raise ValueError(
            "M2 model artifact must include frozen preprocessing metadata."
        )

    required_artifact_keys = {
        "model",
        "feature_columns",
        "volatility_floor",
        "volatility_floor_quantile",
        "training_cutoff",
        "model_version",
    }
    missing_artifact_keys = required_artifact_keys - set(artifact)
    if missing_artifact_keys:
        raise ValueError(
            f"M2 artifact missing keys: {sorted(missing_artifact_keys)}"
        )

    model = artifact["model"]

    metadata = json.loads(
        METADATA_FILE.read_text(
            encoding="utf-8"
        )
    )

    if artifact["feature_columns"] != metadata["features"]:
        raise ValueError("Artifact/metadata feature order mismatch.")

    floor_metadata = metadata.get("volatility_floor", {})
    if not np.isclose(
        float(artifact["volatility_floor"]),
        float(floor_metadata.get("value", np.nan)),
    ):
        raise ValueError("Artifact/metadata volatility floor mismatch.")

    training_scores = np.load(
        TRAINING_SCORES_FILE
    )

    training_scores = np.asarray(
        training_scores,
        dtype=float,
    )

    if len(training_scores) == 0:

        raise ValueError(
            "Saved IF training-score "
            "distribution is empty."
        )

    if not np.isfinite(
        training_scores
    ).all():

        raise ValueError(
            "Saved IF training scores contain "
            "NaN or infinite values."
        )

    # They were saved sorted, but enforce it
    # defensively.
    training_scores = np.sort(
        training_scores
    )

    return (
        model,
        metadata,
        training_scores,
    )


# ============================================================
# Load features
# ============================================================

def load_features(metadata):

    raw = pd.read_csv(
        FEATURE_FILE,
        parse_dates=["Date"],
    )

    required = {
        "Date",
        "USDTRY",
        *M1_FEATURES,
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(
            f"Missing operational columns: {sorted(missing)}"
        )

    raw = raw.sort_values("Date").reset_index(drop=True)
    if raw["Date"].duplicated().any():
        raise ValueError("Duplicate dates detected.")

    floor_spec = metadata.get("volatility_floor", {})
    volatility_floor = float(floor_spec.get("value", np.nan))

    df = validate_features(
        add_anomaly_features(
            raw,
            volatility_floor=volatility_floor,
        )
    )

    features = metadata["features"]
    X = df[features].to_numpy(dtype=float)
    if not np.isfinite(X).all():
        raise ValueError(
            "Operational feature matrix contains NaN or infinite values."
        )

    return df

# ============================================================
# Statistical baseline
# ============================================================

def calculate_baseline(
    df,
    metadata,
):

    thresholds = metadata[
        "baseline_thresholds"
    ]

    return_z_threshold = float(
        thresholds[
            "absolute_return_zscore_20d"
        ]
    )

    volatility_ratio_threshold = float(
        thresholds[
            "volatility_ratio_5d_60d"
        ]
    )

    drawdown_change_threshold = float(
        thresholds[
            "absolute_drawdown_change_5d"
        ]
    )

    absolute_return_threshold = float(
        thresholds[
            "absolute_return_1d"
        ]
    )

    result = df.copy()

    # --------------------------------------------------------
    # Transparent component scores
    #
    # Component = 1.0 means its threshold is exactly met.
    # --------------------------------------------------------

    result["Score_Return_Z"] = (
        result[
            "Return_ZScore_20D"
        ].abs()
        / return_z_threshold
    )

    result[
        "Score_Volatility_Ratio"
    ] = (
        result[
            "Volatility_Ratio_5D_60D"
        ]
        / volatility_ratio_threshold
    )

    result[
        "Score_Drawdown_Change"
    ] = (
        result[
            "Drawdown_Change_5D"
        ].abs()
        / drawdown_change_threshold
    )

    result[
        "Score_Absolute_Return"
    ] = (
        result[
            "Absolute_Return_1D"
        ]
        / absolute_return_threshold
    )

    score_columns = [
        "Score_Return_Z",
        "Score_Volatility_Ratio",
        "Score_Drawdown_Change",
        "Score_Absolute_Return",
    ]

    result[
        "Baseline_Anomaly_Score"
    ] = (
        result[
            score_columns
        ]
        .max(axis=1)
    )

    # --------------------------------------------------------
    # Rule breaches
    # --------------------------------------------------------

    result[
        "Rule_Return_Z"
    ] = (
        result[
            "Score_Return_Z"
        ]
        >= 1.0
    )

    result[
        "Rule_Volatility_Ratio"
    ] = (
        result[
            "Score_Volatility_Ratio"
        ]
        >= 1.0
    )

    result[
        "Rule_Drawdown_Change"
    ] = (
        result[
            "Score_Drawdown_Change"
        ]
        >= 1.0
    )

    result[
        "Rule_Absolute_Return"
    ] = (
        result[
            "Score_Absolute_Return"
        ]
        >= 1.0
    )

    rule_columns = [
        "Rule_Return_Z",
        "Rule_Volatility_Ratio",
        "Rule_Drawdown_Change",
        "Rule_Absolute_Return",
    ]

    result[
        "Rules_Breached"
    ] = (
        result[
            rule_columns
        ]
        .sum(axis=1)
    )

    result[
        "Baseline_Anomaly"
    ] = (
        result[
            "Rules_Breached"
        ]
        >= 1
    )

    # --------------------------------------------------------
    # Dominant component
    #
    # This is the largest component even when no rule is
    # breached. Therefore Primary_Reason is descriptive,
    # not itself an anomaly classification.
    # --------------------------------------------------------

    score_reason_map = {
        "Score_Return_Z":
            "LOCAL_RETURN_SHOCK",

        "Score_Volatility_Ratio":
            "VOLATILITY_ACCELERATION",

        "Score_Drawdown_Change":
            "DRAWDOWN_SHIFT",

        "Score_Absolute_Return":
            "LARGE_ABSOLUTE_RETURN",
    }

    dominant = (
        result[
            list(
                score_reason_map.keys()
            )
        ]
        .idxmax(axis=1)
    )

    result[
        "Primary_Reason"
    ] = (
        dominant.map(
            score_reason_map
        )
    )

    result[
        "Anomaly_State"
    ] = np.where(
        result[
            "Baseline_Anomaly"
        ],
        "ANOMALOUS",
        "NORMAL",
    )

    return result


# ============================================================
# Empirical IF percentile
# ============================================================

def empirical_percentile(
    score,
    sorted_reference_scores,
):

    position = np.searchsorted(
        sorted_reference_scores,
        score,
        side="right",
    )

    return float(
        position
        / len(
            sorted_reference_scores
        )
    )


# ============================================================
# Score frozen Isolation Forest
# ============================================================

def calculate_if_scores(
    df,
    model,
    metadata,
    training_scores,
):

    features = metadata[
        "features"
    ]

    X = (
        df[features]
        .to_numpy(dtype=float)
    )

    # sklearn score_samples:
    # lower = more abnormal
    #
    # Our convention:
    # higher = more abnormal
    if_scores = (
        -model.score_samples(X)
    )

    df = df.copy()

    df[
        "IF_Anomaly_Score"
    ] = if_scores

    df[
        "IF_Training_Percentile"
    ] = [
        empirical_percentile(
            score,
            training_scores,
        )
        for score
        in if_scores
    ]

    return df


# ============================================================
# Inference mode
# ============================================================

def assign_inference_mode(
    df,
    metadata,
):

    cutoff = pd.Timestamp(
        metadata[
            "training_cutoff"
        ]
    )

    df = df.copy()

    df[
        "Inference_Mode"
    ] = np.where(
        df["Date"] <= cutoff,
        "TRAINING_REFERENCE",
        "OUT_OF_SAMPLE_OPERATIONAL",
    )

    return df


# ============================================================
# Training-reference consistency check
# ============================================================

def check_training_reference(
    latest,
    metadata,
):

    cutoff = pd.Timestamp(
        metadata[
            "training_cutoff"
        ]
    )

    if (
        latest["Date"]
        != cutoff
    ):

        # New data exist beyond training cutoff.
        # No exact final-training equality is expected.
        return

    if not TRAINING_REFERENCE_FILE.exists():

        print(
            "\nTraining reference JSON not found; "
            "skipping consistency check."
        )

        return

    reference = json.loads(
        TRAINING_REFERENCE_FILE.read_text(
            encoding="utf-8"
        )
    )

    tolerance = 1e-8

    checks = {
        "baseline anomaly score":
            (
                float(
                    latest[
                        "Baseline_Anomaly_Score"
                    ]
                ),
                float(
                    reference[
                        "baseline_anomaly_score"
                    ]
                ),
            ),

        "IF anomaly score":
            (
                float(
                    latest[
                        "IF_Anomaly_Score"
                    ]
                ),
                float(
                    reference[
                        "if_anomaly_score"
                    ]
                ),
            ),

        "IF training percentile":
            (
                float(
                    latest[
                        "IF_Training_Percentile"
                    ]
                ),
                float(
                    reference[
                        "if_training_percentile"
                    ]
                ),
            ),
    }

    print(
        "\n--- TRAINING REFERENCE CHECK ---"
    )

    for name, (
        operational_value,
        reference_value,
    ) in checks.items():

        difference = abs(
            operational_value
            - reference_value
        )

        passed = (
            difference
            <= tolerance
        )

        print(
            f"{name:25s}: "
            f"diff={difference:.12g} | "
            f"{'PASS' if passed else 'FAIL'}"
        )

        if not passed:

            raise RuntimeError(
                f"Training-reference mismatch: "
                f"{name}"
            )


# ============================================================
# Latest JSON
# ============================================================

def build_latest_output(
    latest,
    metadata,
):

    return {
        "model_version":
            metadata[
                "model_version"
            ],

        "date":
            str(
                latest[
                    "Date"
                ].date()
            ),

        "usdtry":
            float(
                latest[
                    "USDTRY"
                ]
            ),

        "anomaly_state":
            str(
                latest[
                    "Anomaly_State"
                ]
            ),

        "baseline_anomaly":
            bool(
                latest[
                    "Baseline_Anomaly"
                ]
            ),

        "baseline_anomaly_score":
            float(
                latest[
                    "Baseline_Anomaly_Score"
                ]
            ),

        "rules_breached":
            int(
                latest[
                    "Rules_Breached"
                ]
            ),

        "primary_reason":
            str(
                latest[
                    "Primary_Reason"
                ]
            ),

        "rule_components": {
            "return_z":
                float(
                    latest[
                        "Score_Return_Z"
                    ]
                ),

            "volatility_ratio":
                float(
                    latest[
                        "Score_Volatility_Ratio"
                    ]
                ),

            "drawdown_change":
                float(
                    latest[
                        "Score_Drawdown_Change"
                    ]
                ),

            "absolute_return":
                float(
                    latest[
                        "Score_Absolute_Return"
                    ]
                ),
        },

        "if_anomaly_score":
            float(
                latest[
                    "IF_Anomaly_Score"
                ]
            ),

        "if_training_percentile":
            float(
                latest[
                    "IF_Training_Percentile"
                ]
            ),

        "if_interpretation":
            (
                "Continuous secondary "
                "multivariate anomaly score; "
                "not an anomaly probability."
            ),

        "primary_detector":
            metadata[
                "primary_operational_detector"
            ],

        "secondary_ml_signal":
            metadata[
                "secondary_ml_signal"
            ],

        "volatility_floor":
            float(metadata["volatility_floor"]["value"]),

        "volatility_floor_quantile":
            float(metadata["volatility_floor"]["quantile"]),

        "training_cutoff":
            metadata[
                "training_cutoff"
            ],

        "inference_mode":
            str(
                latest[
                    "Inference_Mode"
                ]
            ),
    }


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\n--- MILESTONE 2: "
        "OPERATIONAL ANOMALY INFERENCE ---"
    )

    (
        model,
        metadata,
        training_scores,
    ) = load_artifacts()

    print(
        f"Model version: "
        f"{metadata['model_version']}"
    )

    print(
        f"Training cutoff: "
        f"{metadata['training_cutoff']}"
    )

    print(
        "Primary detector: "
        f"{metadata['primary_operational_detector']}"
    )

    print(
        "Secondary signal: "
        f"{metadata['secondary_ml_signal']}"
    )

    df = load_features(
        metadata
    )

    print(
        f"\nAvailable observations: "
        f"{len(df):,}"
    )

    print(
        f"Latest feature date: "
        f"{df['Date'].max().date()}"
    )

    # --------------------------------------------------------
    # Frozen operational scoring
    # --------------------------------------------------------

    scored = calculate_baseline(
        df,
        metadata,
    )

    scored = calculate_if_scores(
        scored,
        model,
        metadata,
        training_scores,
    )

    scored = assign_inference_mode(
        scored,
        metadata,
    )

    latest = scored.iloc[-1]

    # --------------------------------------------------------
    # Verify exact reproducibility at training cutoff
    # --------------------------------------------------------

    check_training_reference(
        latest,
        metadata,
    )

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print(
        "\n--- LATEST ANOMALY INFERENCE ---"
    )

    print(
        f"Date: "
        f"{latest['Date'].date()}"
    )

    print(
        f"USDTRY: "
        f"{latest['USDTRY']:.4f}"
    )

    print(
        f"Inference mode: "
        f"{latest['Inference_Mode']}"
    )

    print(
        f"\nAnomaly state: "
        f"{latest['Anomaly_State']}"
    )

    print(
        f"Baseline anomaly score: "
        f"{latest['Baseline_Anomaly_Score']:.4f}"
    )

    print(
        f"Rules breached: "
        f"{int(latest['Rules_Breached'])}"
    )

    print(
        f"Dominant component: "
        f"{latest['Primary_Reason']}"
    )

    print(
        "\n--- BASELINE COMPONENTS ---"
    )

    print(
        f"Return z-score component: "
        f"{latest['Score_Return_Z']:.4f}"
    )

    print(
        f"Volatility-ratio component: "
        f"{latest['Score_Volatility_Ratio']:.4f}"
    )

    print(
        f"Drawdown-change component: "
        f"{latest['Score_Drawdown_Change']:.4f}"
    )

    print(
        f"Absolute-return component: "
        f"{latest['Score_Absolute_Return']:.4f}"
    )

    print(
        "\n--- SECONDARY ML SIGNAL ---"
    )

    print(
        f"IF anomaly score: "
        f"{latest['IF_Anomaly_Score']:.6f}"
    )

    print(
        f"IF training percentile: "
        f"{latest['IF_Training_Percentile']:.2%}"
    )

    print(
        "Interpretation: percentile is "
        "historical rank, not probability."
    )

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    history_columns = [
        "Date",
        "USDTRY",
        "Anomaly_State",
        "Baseline_Anomaly",
        "Baseline_Anomaly_Score",
        "Rules_Breached",
        "Primary_Reason",
        "Score_Return_Z",
        "Score_Volatility_Ratio",
        "Score_Drawdown_Change",
        "Score_Absolute_Return",
        "IF_Anomaly_Score",
        "IF_Training_Percentile",
        "Inference_Mode",
    ]

    scored[
        history_columns
    ].to_csv(
        HISTORY_OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Save latest JSON
    # --------------------------------------------------------

    latest_output = (
        build_latest_output(
            latest,
            metadata,
        )
    )

    LATEST_OUTPUT_FILE.write_text(
        json.dumps(
            latest_output,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n--- OUTPUT ---"
    )

    print(
        LATEST_OUTPUT_FILE
    )

    print(
        HISTORY_OUTPUT_FILE
    )

    print(
        "\nOperational anomaly inference "
        "complete."
    )


if __name__ == "__main__":
    main()
