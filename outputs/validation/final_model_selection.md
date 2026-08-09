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
| Median episode duration | 3.00 | 14.00 |
| Mean self-transition | 90.70% | 96.74% |
| Switch rate | 2.95% | 2.82% |

K-Means demonstrated stronger geometric cluster separation,
while the HMM produced substantially greater temporal
coherence.

## Walk-Forward Validation

Mean out-of-sample switch rate:

**3.27%**

Median annual median episode duration:

**19.00 observations**

High-volatility stress was detected out of sample in:

**[2021, 2022, 2023]**

## State Stability

State-ordering checks passed:

**95.83%**

Minimum standardized state separation:

**1.0175**

Mean minimum standardized state separation:

**1.8574**

The learned low-, elevated-, and high-volatility states
therefore remained distinguishable across expanding-window
retraining folds.

## Acceptance Criteria

| Criterion | Result |
|---|---|
| HMM improves median persistence | PASS |
| HMM improves self-transition | PASS |
| Mean OOS switch rate <= 5% | PASS |
| Median annual episode >= 5 observations | PASS |
| Stress detected out of sample | PASS |
| State ordering >= 90% | PASS |
| Minimum standardized separation >= 1.0 | PASS |

## Decision

**SELECT 3-STATE GAUSSIAN HMM**

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