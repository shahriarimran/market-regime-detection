# Project 1–3 Test Audit

Audit date: 2026-08-10  
Remediation: M2.1 temporal-leakage correction

## Decision summary

The M2 volatility-floor leakage has been remediated. Every temporal validation fold now estimates the return-z-score volatility floor from its training sample only and freezes it for both train and test transformations. Synthetic tests freeze the floor before injection. Final M2 training freezes the value estimated through 2026-08-07 in both the joblib artifact and metadata, and operational inference reuses it without estimation.

All 52 Project 1–3 tests pass. Frozen P1 and P3 behavior remains within prior regression tolerances. M2 has been regenerated, revalidated, versioned as `M2-v0.2.0`, and protected by updated regression fixtures.

## Environment

| Component | Version |
|---|---:|
| OS | Windows 11 Home 64-bit, 10.0.26200 |
| Python | 3.14.5 |
| pytest | 9.1.1 |
| NumPy | 2.5.1 |
| pandas | 3.0.3 |
| scikit-learn | 1.9.0 |
| SciPy | 1.18.0 |
| joblib | 1.5.3 |
| hmmlearn | 0.3.3 |

## M2.1 implementation

The implementation in `prepare_anomaly_features.py` now has two distinct responsibilities:

- `estimate_volatility_floor(training_df, quantile=0.05)` estimates the floor from a supplied training dataframe only.
- `add_anomaly_features(df, volatility_floor)` requires an explicit positive finite floor and never estimates or recomputes it.

The floor distribution is the rolling standard deviation of `Return_1D.shift(1)` over the prior 20 observations. Its 5th percentile is used as a numerical denominator floor. The z-score numerator and denominator are both trailing and exclude the current return from their reference window.

For each 2021–2026 walk-forward fold:

1. the floor is estimated from rows strictly before the test year;
2. the transformation includes trailing historical context through the test period;
3. the same scalar is applied to fold training and test rows;
4. no test/future row contributes to floor estimation.

The same policy is applied to stability/sensitivity folds. The fixed 2022-12-30 chronological split used by synthetic and regime-conditioned validation estimates one training-only floor. Synthetic injection occurs only after the leakage-safe feature transformation, so the floor cannot react to injected values.

## Final volatility floor

| Property | Value |
|---|---:|
| Quantile | 0.05 |
| Window | Prior 20 observations |
| Estimation cutoff | 2026-08-07 |
| Final frozen floor | 0.0007028294836738612 |
| Provenance | Training-only |
| Artifact version | M2-v0.2.0 |

Fold floors are persisted in `walk_forward_summary.csv`: 0.003022 (2021), 0.003139 (2022), 0.001960 (2023), 0.000771 (2024), 0.000829 (2025), and 0.000778 (2026), shown rounded here.

## M2 old versus new metrics

| Metric | Old M2-v0.1.0 | New M2-v0.2.0 |
|---|---:|---:|
| Training observations | 2,425 | 2,425 |
| Baseline anomaly count | 162 | 162 |
| Baseline anomaly rate | 0.0668041 | 0.0668041 |
| IF score median | 0.3658916 | 0.3662563 |
| IF score P95 | 0.5346728 | 0.5363940 |
| IF score P99 | 0.6781642 | 0.6791762 |
| Latest baseline score | 0.4484812 | 0.3962690 |
| Latest IF score | 0.3485976 | 0.3472305 |
| Latest IF percentile | 0.2618557 | 0.2338144 |
| Mean walk-forward baseline rate | 0.0649452 | 0.0630221 |
| Mean walk-forward IF rate | 0.0620788 | 0.0582351 |
| Mean walk-forward IF target gap | 0.0688137 | 0.0655564 |
| Mean IF seed Jaccard | 0.9702373 | 0.9669513 |

The old final floor was implicit and estimated retrospectively from the complete input. The numerically equivalent 5th-percentile full-training value is now explicitly persisted with its cutoff and provenance. Fold floors differ because they are now estimated from each fold''s training history only.

## M2 model-selection result

The final architecture remains:

- Primary binary detector: `STATISTICAL_BASELINE`
- Binary states: `NORMAL` / `ANOMALOUS`
- Secondary signal: `ISOLATION_FOREST_SCORE` plus frozen-training empirical percentile
- Rejected as primary: Isolation Forest binary classification, One-Class SVM binary classification, and hard regime-conditioned models

The leakage-safe validation did not justify promoting an ML binary detector. Synthetic response and seed-stability gates pass; temporal-calibration gates continue to reject IF/OCSVM binary use. This supports retaining the simpler statistical baseline architecture.

## Regenerated M2 outputs

The following workflows were rerun successfully:

- anomaly feature generation and diagnostics;
- statistical baseline;
- Isolation Forest and One-Class SVM analyses;
- annual walk-forward validation;
- synthetic anomaly and fair-flip validation;
- regime-conditioned validation;
- seed/max-sample/threshold stability and sensitivity;
- final model selection;
- final frozen-model training;
- operational inference.

Synthetic flip monotonicity remains 100%. Operational training-reference checks for baseline score, IF score, and IF percentile all pass exactly.

## Frozen artifact audit

Updated artifacts:

- `src/milestone_2_anomaly_detection/models/usdtry_isolation_forest.joblib`
- `src/milestone_2_anomaly_detection/models/usdtry_anomaly_metadata.json`
- `src/milestone_2_anomaly_detection/models/usdtry_if_training_scores.npy`
- `src/milestone_2_anomaly_detection/outputs/final_training/final_training_summary.csv`
- `src/milestone_2_anomaly_detection/outputs/final_training/latest_training_reference.json`
- `src/milestone_2_anomaly_detection/outputs/operational/latest_anomaly.json`
- `src/milestone_2_anomaly_detection/outputs/operational/anomaly_inference_history.csv`

The joblib artifact is now a dictionary containing the Isolation Forest, exact feature order, model version, training cutoff, frozen floor, and floor quantile. Operational inference validates artifact/metadata consistency before scoring.

## Test inventory and results

| Command | Result |
|---|---|
| `python -m pytest -q` | 52 passed |
| `python -m pytest -m unit -q` | 30 passed, 22 deselected |
| `python -m pytest -m integration -q` | 13 passed, 39 deselected |
| `python -m pytest -m regression -q` | 9 passed, 43 deselected |
| `python -m pytest -m slow -q` | 13 passed, 39 deselected |

No tests were skipped, xfailed, weakened, or disabled.

Warnings are from joblib loading historical NumPy pickle representations under NumPy 2.5 (17,035 repeated deprecation warnings in the full artifact run) plus a pytest cache warning caused by an existing Windows cache-path artifact. They do not affect predictions or pass/fail results.

## Leakage audit

| Contract | Result |
|---|---|
| Shared M1 rolling features causal | PASS |
| M1 HMM forward filtering causal | PASS |
| M2 floor training-only and explicit | PASS |
| M2 features invariant to future-row changes | PASS |
| M2 synthetic floor frozen before injection | PASS |
| M2 walk-forward floor frozen per fold | PASS |
| M2 operational floor loaded from frozen artifact/metadata | PASS |
| M3 predictors causal | PASS |
| M3 five-observation validation purge | PASS |

The original failing M2 causality test now passes because the implementation is fixed. A second regression test perturbs real post-cutoff observations and verifies all M2 feature rows at or before the cutoff remain exactly unchanged.

## Regression status

- Project 1: PASS. No P1 source or artifact behavior was changed.
- Project 2: PASS against the new M2-v0.2.0 artifact and leakage-safe reference values.
- Project 3: PASS. No P3 model behavior or artifact was changed by M2.1.
- Cross-project contracts: PASS. P3 still excludes P1/P2 outputs from its final predictor set.

## Files changed

Source/API:

- `src/milestone_2_anomaly_detection/prepare_anomaly_features.py`
- `walk_forward_validation.py`
- `stability_sensitivity.py`
- `regime_conditioned_validation.py`
- `synthetic_anomaly_validation.py`
- `synthetic_flip_validation.py`
- `train_final_anomaly_model.py`
- `operational_anomaly_inference.py`

Tests:

- `tests/unit/test_milestone_2.py`
- `tests/unit/test_extended_contracts.py`
- `tests/integration/test_frozen_artifacts.py`
- `tests/regression/test_frozen_contracts.py`

Documentation:

- `README.md`
- `docs/PROJECT_1_TO_3_TEST_AUDIT.md`

M2 datasets, validation outputs, selection outputs, model artifacts, and operational outputs were regenerated. No Project 1 or Project 3 source/model artifact was regenerated or modified as part of M2.1.

READY FOR PROJECT 4

