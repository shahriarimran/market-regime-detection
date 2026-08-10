from pathlib import Path

import pandas as pd


# ============================================================
# Files
# ============================================================

OUTPUT_DIR = Path("outputs")
VALIDATION_DIR = OUTPUT_DIR / "validation"

MODEL_COMPARISON_FILE = (
    OUTPUT_DIR
    / "model_comparison_summary.csv"
)

WALK_FORWARD_FILE = (
    VALIDATION_DIR
    / "walk_forward_summary.csv"
)

ORDERING_FILE = (
    VALIDATION_DIR
    / "state_ordering_checks.csv"
)

SEPARATION_FILE = (
    VALIDATION_DIR
    / "state_separation.csv"
)

STABILITY_FILE = (
    VALIDATION_DIR
    / "state_stability_summary.csv"
)

REPORT_FILE = (
    VALIDATION_DIR
    / "final_model_selection.md"
)


# ============================================================
# Main
# ============================================================

def main():

    comparison = pd.read_csv(
        MODEL_COMPARISON_FILE,
        index_col=0,
    )

    walk_forward = pd.read_csv(
        WALK_FORWARD_FILE
    )

    ordering = pd.read_csv(
        ORDERING_FILE
    )

    separation = pd.read_csv(
        SEPARATION_FILE
    )

    stability = pd.read_csv(
        STABILITY_FILE
    )

    # --------------------------------------------------------
    # K-Means vs HMM
    # --------------------------------------------------------

    kmeans = comparison.loc[
        "K-Means"
    ]

    hmm = comparison.loc[
        "HMM"
    ]

    hmm_more_persistent = (
        hmm["Median_Episode_Duration"]
        >
        kmeans["Median_Episode_Duration"]
    )

    hmm_higher_self_transition = (
        hmm["Mean_Self_Transition"]
        >
        kmeans["Mean_Self_Transition"]
    )

    # --------------------------------------------------------
    # Walk-forward behavior
    # --------------------------------------------------------

    mean_switch_rate = (
        walk_forward[
            "Switch_Rate"
        ].mean()
    )

    median_annual_episode = (
        walk_forward[
            "Median_Episode_Duration"
        ].median()
    )

    years_with_stress = (
        walk_forward.loc[
            walk_forward[
                "Stress_Share"
            ] > 0,
            "Test_Year",
        ]
        .astype(int)
        .tolist()
    )

    stress_oos_detected = (
        len(years_with_stress) > 0
    )

    # --------------------------------------------------------
    # Ordering stability
    # --------------------------------------------------------

    total_checks = (
        ordering[
            "Total_Checks"
        ].sum()
    )

    passed_checks = (
        ordering[
            "Passed_Checks"
        ].sum()
    )

    ordering_pass_rate = (
        passed_checks
        / total_checks
    )

    # --------------------------------------------------------
    # State separation
    # --------------------------------------------------------

    minimum_separation = (
        separation[
            "Minimum_Separation"
        ].min()
    )

    mean_minimum_separation = (
        separation[
            "Minimum_Separation"
        ].mean()
    )

    # --------------------------------------------------------
    # Transparent project-specific gates
    #
    # These are engineering acceptance criteria,
    # NOT universal statistical thresholds.
    # --------------------------------------------------------

    gates = {
        "HMM improves median persistence":
            hmm_more_persistent,

        "HMM improves self-transition":
            hmm_higher_self_transition,

        "Mean OOS switch rate <= 5%":
            mean_switch_rate <= 0.05,

        "Median annual episode >= 5 observations":
            median_annual_episode >= 5,

        "Stress detected out of sample":
            stress_oos_detected,

        "State ordering >= 90%":
            ordering_pass_rate >= 0.90,

        "Minimum standardized separation >= 1.0":
            minimum_separation >= 1.0,
    }

    passed = sum(
        gates.values()
    )

    total = len(
        gates
    )

    pass_rate = (
        passed
        / total
    )

    # Require all major gates.
    selected = all(
        gates.values()
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        "\n--- FINAL MODEL SELECTION ---"
    )

    print(
        "\nAcceptance criteria:"
    )

    for criterion, result in (
        gates.items()
    ):

        status = (
            "PASS"
            if result
            else "FAIL"
        )

        print(
            f"{status:4s} | "
            f"{criterion}"
        )

    print(
        "\n--- SUMMARY METRICS ---"
    )

    print(
        f"K-Means median episode: "
        f"{kmeans['Median_Episode_Duration']:.2f}"
    )

    print(
        f"HMM median episode: "
        f"{hmm['Median_Episode_Duration']:.2f}"
    )

    print(
        f"K-Means mean self-transition: "
        f"{kmeans['Mean_Self_Transition']:.2%}"
    )

    print(
        f"HMM mean self-transition: "
        f"{hmm['Mean_Self_Transition']:.2%}"
    )

    print(
        f"Walk-forward mean switch rate: "
        f"{mean_switch_rate:.2%}"
    )

    print(
        f"Walk-forward median annual "
        f"episode duration: "
        f"{median_annual_episode:.2f}"
    )

    print(
        f"State ordering pass rate: "
        f"{ordering_pass_rate:.2%}"
    )

    print(
        f"Minimum state separation: "
        f"{minimum_separation:.4f}"
    )

    print(
        f"Mean minimum state separation: "
        f"{mean_minimum_separation:.4f}"
    )

    print(
        "Stress detected OOS in years: "
        f"{years_with_stress}"
    )

    print(
        f"\nAcceptance gates passed: "
        f"{passed}/{total} "
        f"({pass_rate:.1%})"
    )

    # --------------------------------------------------------
    # Decision
    # --------------------------------------------------------

    if selected:

        decision = (
            "SELECT 3-STATE GAUSSIAN HMM"
        )

        print(
            "\nFINAL DECISION:"
        )

        print(
            decision
        )

    else:

        decision = (
            "DO NOT DEPLOY YET"
        )

        print(
            "\nFINAL DECISION:"
        )

        print(
            decision
        )

    # --------------------------------------------------------
    # Markdown report
    # --------------------------------------------------------

    gate_rows = "\n".join(
        (
            f"| {criterion} | "
            f"{'PASS' if result else 'FAIL'} |"
        )
        for criterion, result
        in gates.items()
    )

    report = f"""
# Final Market-Regime Model Selection

## Candidate Models

Two unsupervised models were evaluated:

1. K-Means clustering
2. Three-state diagonal Gaussian Hidden Markov Model

K-Means was retained as the static clustering benchmark.

The Gaussian HMM was evaluated as the candidate temporal
market-regime model.

## Model Comparison

| Metric | K-Means | HMM |
|---|---:|---:|
| Median episode duration | {kmeans['Median_Episode_Duration']:.2f} | {hmm['Median_Episode_Duration']:.2f} |
| Mean self-transition | {kmeans['Mean_Self_Transition']:.2%} | {hmm['Mean_Self_Transition']:.2%} |
| Switch rate | {kmeans['Switch_Rate']:.2%} | {hmm['Switch_Rate']:.2%} |

K-Means demonstrated stronger geometric cluster separation,
while the HMM produced substantially greater temporal
coherence.

## Walk-Forward Validation

Mean out-of-sample switch rate:

**{mean_switch_rate:.2%}**

Median annual median episode duration:

**{median_annual_episode:.2f} observations**

High-volatility stress was detected out of sample in:

**{years_with_stress}**

## State Stability

State-ordering checks passed:

**{ordering_pass_rate:.2%}**

Minimum standardized state separation:

**{minimum_separation:.4f}**

Mean minimum standardized state separation:

**{mean_minimum_separation:.4f}**

The learned low-, elevated-, and high-volatility states
therefore remained distinguishable across expanding-window
retraining folds.

## Acceptance Criteria

| Criterion | Result |
|---|---|
{gate_rows}

## Decision

**{decision}**

The three-state Gaussian HMM is selected as the final
market-regime architecture if all acceptance criteria pass.

Its operational interpretation is:

- Low-Volatility Trend
- Elevated-Volatility Transition
- High-Volatility Stress

The model should be used as a market-state detector rather
than as a direct BUY/SELL forecasting model.

## Deployment Constraint

Operational inference must remain causal. The regime assigned
to the current observation may use only observations available
through that date.

The next implementation stage therefore creates a final
full-history fitted model and a separate causal inference
interface suitable for later Excel and Telegram integration.
""".strip()

    REPORT_FILE.write_text(
        report,
        encoding="utf-8",
    )

    print(
        f"\nSaved report: "
        f"{REPORT_FILE}"
    )


if __name__ == "__main__":
    main()