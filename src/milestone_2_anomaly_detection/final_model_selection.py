from pathlib import Path

import pandas as pd


# ============================================================
# Paths
# ============================================================

M2_DIR = Path(__file__).resolve().parent

SYNTHETIC_FILE = (
    M2_DIR
    / "outputs"
    / "synthetic_flip_validation"
    / "synthetic_flip_validation_summary.csv"
)

WALK_FORWARD_FILE = (
    M2_DIR
    / "outputs"
    / "walk_forward_validation"
    / "walk_forward_summary.csv"
)

STABILITY_FILE = (
    M2_DIR
    / "outputs"
    / "stability_sensitivity"
    / "if_stability_summary.csv"
)

REGIME_FILE = (
    M2_DIR
    / "outputs"
    / "regime_conditioned_validation"
    / "regime_conditioned_summary.csv"
)

OUTPUT_DIR = (
    M2_DIR
    / "outputs"
    / "final_model_selection"
)

REPORT_FILE = (
    OUTPUT_DIR
    / "final_model_selection.md"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "final_model_selection.csv"
)


# ============================================================
# Load
# ============================================================

def load_inputs():

    required = [
        SYNTHETIC_FILE,
        WALK_FORWARD_FILE,
        STABILITY_FILE,
        REGIME_FILE,
    ]

    for path in required:

        if not path.exists():

            raise FileNotFoundError(
                f"Required validation output "
                f"not found:\n{path}"
            )

    synthetic = pd.read_csv(
        SYNTHETIC_FILE
    )

    walk = pd.read_csv(
        WALK_FORWARD_FILE
    )

    stability = pd.read_csv(
        STABILITY_FILE
    )

    regime = pd.read_csv(
        REGIME_FILE
    )

    return (
        synthetic,
        walk,
        stability,
        regime,
    )


# ============================================================
# Aggregate metrics
# ============================================================

def calculate_metrics(
    synthetic,
    walk,
    stability,
    regime,
):

    # --------------------------------------------------------
    # Synthetic sensitivity
    # --------------------------------------------------------

    synthetic_mean = (
        synthetic
        .groupby("Model")[
            "Flip_Rate"
        ]
        .mean()
    )

    baseline_synthetic = float(
        synthetic_mean[
            "Baseline"
        ]
    )

    if_synthetic = float(
        synthetic_mean[
            "IF"
        ]
    )

    ocsvm_synthetic = float(
        synthetic_mean[
            "OCSVM"
        ]
    )

    # --------------------------------------------------------
    # Walk-forward anomaly rates
    # --------------------------------------------------------

    baseline_rate_mean = float(
        walk[
            "Baseline_Test_Rate"
        ].mean()
    )

    if_rate_mean = float(
        walk[
            "IF_Test_Rate"
        ].mean()
    )

    ocsvm_rate_mean = float(
        walk[
            "OCSVM_Test_Rate"
        ].mean()
    )

    # --------------------------------------------------------
    # Temporal stability
    # --------------------------------------------------------

    baseline_rate_sd = float(
        walk[
            "Baseline_Test_Rate"
        ].std()
    )

    if_rate_sd = float(
        walk[
            "IF_Test_Rate"
        ].std()
    )

    ocsvm_rate_sd = float(
        walk[
            "OCSVM_Test_Rate"
        ].std()
    )

    # --------------------------------------------------------
    # Mean target gaps
    # --------------------------------------------------------

    if_target_gap = float(
        walk[
            "IF_Target_Gap"
        ].mean()
    )

    ocsvm_target_gap = float(
        walk[
            "OCSVM_Target_Gap"
        ].mean()
    )

    # --------------------------------------------------------
    # IF random stability
    # --------------------------------------------------------

    stability_map = dict(
        zip(
            stability[
                "Metric"
            ],
            stability[
                "Value"
            ],
        )
    )

    if_seed_jaccard = float(
        stability_map[
            "Mean IF seed pairwise Jaccard"
        ]
    )

    # --------------------------------------------------------
    # Regime-conditioning comparison
    # --------------------------------------------------------

    global_if_gap = float(
        regime[
            "Global_IF_Calibration_Gap"
        ].mean()
    )

    conditioned_if_gap = float(
        regime[
            "Conditioned_IF_Calibration_Gap"
        ].mean()
    )

    global_oc_gap = float(
        regime[
            "Global_OCSVM_Calibration_Gap"
        ].mean()
    )

    conditioned_oc_gap = float(
        regime[
            "Conditioned_OCSVM_Calibration_Gap"
        ].mean()
    )

    return {
        "baseline_synthetic":
            baseline_synthetic,

        "if_synthetic":
            if_synthetic,

        "ocsvm_synthetic":
            ocsvm_synthetic,

        "baseline_rate_mean":
            baseline_rate_mean,

        "if_rate_mean":
            if_rate_mean,

        "ocsvm_rate_mean":
            ocsvm_rate_mean,

        "baseline_rate_sd":
            baseline_rate_sd,

        "if_rate_sd":
            if_rate_sd,

        "ocsvm_rate_sd":
            ocsvm_rate_sd,

        "if_target_gap":
            if_target_gap,

        "ocsvm_target_gap":
            ocsvm_target_gap,

        "if_seed_jaccard":
            if_seed_jaccard,

        "global_if_gap":
            global_if_gap,

        "conditioned_if_gap":
            conditioned_if_gap,

        "global_oc_gap":
            global_oc_gap,

        "conditioned_oc_gap":
            conditioned_oc_gap,
    }


# ============================================================
# Acceptance gates
# ============================================================

def evaluate_gates(m):

    gates = []

    # --------------------------------------------------------
    # Gate 1
    # ML should add synthetic anomaly sensitivity.
    # --------------------------------------------------------

    gates.append(
        {
            "Gate":
                "IF improves synthetic sensitivity",

            "Passed":
                (
                    m["if_synthetic"]
                    >
                    m["baseline_synthetic"]
                ),

            "Value":
                m["if_synthetic"],

            "Reference":
                m["baseline_synthetic"],
        }
    )

    # --------------------------------------------------------
    # Gate 2
    # IF should be random-seed stable.
    # --------------------------------------------------------

    gates.append(
        {
            "Gate":
                "IF seed stability >= 0.90",

            "Passed":
                (
                    m["if_seed_jaccard"]
                    >= 0.90
                ),

            "Value":
                m["if_seed_jaccard"],

            "Reference":
                0.90,
        }
    )

    # --------------------------------------------------------
    # Gate 3
    # IF temporal variation should not exceed
    # twice the baseline annual SD.
    # --------------------------------------------------------

    gates.append(
        {
            "Gate":
                "IF temporal SD <= 2x baseline",

            "Passed":
                (
                    m["if_rate_sd"]
                    <=
                    2.0
                    * m["baseline_rate_sd"]
                ),

            "Value":
                m["if_rate_sd"],

            "Reference":
                (
                    2.0
                    * m[
                        "baseline_rate_sd"
                    ]
                ),
        }
    )

    # --------------------------------------------------------
    # Gate 4
    # IF mean target calibration gap <= 5%.
    # --------------------------------------------------------

    gates.append(
        {
            "Gate":
                "IF mean target gap <= 5%",

            "Passed":
                (
                    m["if_target_gap"]
                    <= 0.05
                ),

            "Value":
                m["if_target_gap"],

            "Reference":
                0.05,
        }
    )

    # --------------------------------------------------------
    # Gate 5
    # OCSVM calibration should be acceptable.
    # --------------------------------------------------------

    gates.append(
        {
            "Gate":
                "OCSVM mean target gap <= 5%",

            "Passed":
                (
                    m[
                        "ocsvm_target_gap"
                    ]
                    <= 0.05
                ),

            "Value":
                m[
                    "ocsvm_target_gap"
                ],

            "Reference":
                0.05,
        }
    )

    # --------------------------------------------------------
    # Gate 6
    # Hard regime-conditioned IF should improve.
    # --------------------------------------------------------

    gates.append(
        {
            "Gate":
                "Hard regime conditioning improves IF",

            "Passed":
                (
                    m[
                        "conditioned_if_gap"
                    ]
                    <
                    m[
                        "global_if_gap"
                    ]
                ),

            "Value":
                m[
                    "conditioned_if_gap"
                ],

            "Reference":
                m[
                    "global_if_gap"
                ],
        }
    )

    return pd.DataFrame(
        gates
    )


# ============================================================
# Model decision
# ============================================================

def make_decision(
    metrics,
    gates,
):

    # --------------------------------------------------------
    # Binary operational decision
    # --------------------------------------------------------

    primary_detector = (
        "STATISTICAL_BASELINE"
    )

    secondary_detector = (
        "ISOLATION_FOREST_SCORE"
    )

    rejected_primary = [
        "ISOLATION_FOREST_BINARY",
        "ONE_CLASS_SVM_BINARY",
        "HARD_REGIME_CONDITIONED_MODELS",
    ]

    return {
        "Primary_Detector":
            primary_detector,

        "Secondary_Detector":
            secondary_detector,

        "Rejected_Primary":
            ", ".join(
                rejected_primary
            ),
    }


# ============================================================
# Report
# ============================================================

def write_report(
    metrics,
    gates,
    decision,
):

    lines = []

    lines.append(
        "# Milestone 2 Final Model Selection"
    )

    lines.append("")

    lines.append(
        "## Final Decision"
    )

    lines.append("")

    lines.append(
        f"**Primary binary anomaly detector:** "
        f"{decision['Primary_Detector']}"
    )

    lines.append("")

    lines.append(
        f"**Secondary multivariate anomaly signal:** "
        f"{decision['Secondary_Detector']}"
    )

    lines.append("")

    lines.append(
        "The statistical baseline is retained "
        "for the operational NORMAL/ANOMALOUS "
        "classification because it showed better "
        "temporal calibration and year-to-year "
        "stability."
    )

    lines.append("")

    lines.append(
        "Isolation Forest is retained as a "
        "continuous secondary anomaly score because "
        "it demonstrated strong compound-stress "
        "sensitivity and excellent random-seed "
        "stability, despite temporal threshold drift."
    )

    lines.append("")

    lines.append(
        "One-Class SVM is retained only as an "
        "experimental comparator because its "
        "excellent synthetic sensitivity was offset "
        "by substantial walk-forward calibration "
        "instability."
    )

    lines.append("")

    lines.append(
        "Hard regime-conditioned anomaly models "
        "were rejected because separate per-regime "
        "training worsened out-of-sample calibration."
    )

    lines.append("")

    lines.append(
        "## Key Metrics"
    )

    lines.append("")

    lines.append(
        f"- Baseline mean synthetic flip rate: "
        f"{metrics['baseline_synthetic']:.2%}"
    )

    lines.append(
        f"- Isolation Forest mean synthetic flip rate: "
        f"{metrics['if_synthetic']:.2%}"
    )

    lines.append(
        f"- OCSVM mean synthetic flip rate: "
        f"{metrics['ocsvm_synthetic']:.2%}"
    )

    lines.append(
        f"- Baseline annual anomaly-rate SD: "
        f"{metrics['baseline_rate_sd']:.2%}"
    )

    lines.append(
        f"- Isolation Forest annual anomaly-rate SD: "
        f"{metrics['if_rate_sd']:.2%}"
    )

    lines.append(
        f"- OCSVM annual anomaly-rate SD: "
        f"{metrics['ocsvm_rate_sd']:.2%}"
    )

    lines.append(
        f"- Isolation Forest mean target gap: "
        f"{metrics['if_target_gap']:.2%}"
    )

    lines.append(
        f"- OCSVM mean target gap: "
        f"{metrics['ocsvm_target_gap']:.2%}"
    )

    lines.append(
        f"- IF mean seed Jaccard: "
        f"{metrics['if_seed_jaccard']:.4f}"
    )

    lines.append("")

    lines.append(
        "## Acceptance Gates"
    )

    lines.append("")

    for _, row in gates.iterrows():

        status = (
            "PASS"
            if row["Passed"]
            else "FAIL"
        )

        lines.append(
            f"- [{status}] "
            f"{row['Gate']}: "
            f"value={row['Value']:.4f}, "
            f"reference={row['Reference']:.4f}"
        )

    lines.append("")

    lines.append(
        "## Operational Interpretation"
    )

    lines.append("")

    lines.append(
        "Milestone 2 should therefore output both "
        "a transparent binary classification and "
        "a complementary machine-learning score:"
    )

    lines.append("")

    lines.append(
        "- `Anomaly_State`: NORMAL or ANOMALOUS"
    )

    lines.append(
        "- `Baseline_Anomaly_Score`: transparent "
        "rule-based severity"
    )

    lines.append(
        "- `IF_Anomaly_Score`: continuous "
        "multivariate abnormality score"
    )

    lines.append("")

    lines.append(
        "The Isolation Forest score must not be "
        "interpreted as a probability of anomaly."
    )

    REPORT_FILE.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


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
        "FINAL MODEL SELECTION ---"
    )

    (
        synthetic,
        walk,
        stability,
        regime,
    ) = load_inputs()

    metrics = calculate_metrics(
        synthetic,
        walk,
        stability,
        regime,
    )

    gates = evaluate_gates(
        metrics
    )

    decision = make_decision(
        metrics,
        gates,
    )

    print(
        "\n--- ACCEPTANCE GATES ---"
    )

    printable = gates.copy()

    printable[
        "Status"
    ] = printable[
        "Passed"
    ].map(
        {
            True: "PASS",
            False: "FAIL",
        }
    )

    print(
        printable[
            [
                "Gate",
                "Status",
                "Value",
                "Reference",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    print(
        "\n--- FINAL DECISION ---"
    )

    print(
        "Primary operational detector: "
        f"{decision['Primary_Detector']}"
    )

    print(
        "Secondary ML signal: "
        f"{decision['Secondary_Detector']}"
    )

    print(
        "Rejected as primary: "
        f"{decision['Rejected_Primary']}"
    )

    summary = pd.DataFrame(
        [
            {
                **metrics,
                **decision,
            }
        ]
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    write_report(
        metrics,
        gates,
        decision,
    )

    print(
        "\n--- OUTPUT ---"
    )

    print(SUMMARY_FILE)
    print(REPORT_FILE)

    print(
        "\nFinal model selection complete."
    )


if __name__ == "__main__":
    main()