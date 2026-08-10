# Milestone 2 Final Model Selection

## Final Decision

**Primary binary anomaly detector:** STATISTICAL_BASELINE

**Secondary multivariate anomaly signal:** ISOLATION_FOREST_SCORE

The statistical baseline is retained for the operational NORMAL/ANOMALOUS classification because it showed better temporal calibration and year-to-year stability.

Isolation Forest is retained as a continuous secondary anomaly score because it demonstrated strong compound-stress sensitivity and excellent random-seed stability, despite temporal threshold drift.

One-Class SVM is retained only as an experimental comparator because its excellent synthetic sensitivity was offset by substantial walk-forward calibration instability.

Hard regime-conditioned anomaly models were rejected because separate per-regime training worsened out-of-sample calibration.

## Key Metrics

- Baseline mean synthetic flip rate: 57.82%
- Isolation Forest mean synthetic flip rate: 60.00%
- OCSVM mean synthetic flip rate: 80.06%
- Baseline annual anomaly-rate SD: 3.28%
- Isolation Forest annual anomaly-rate SD: 7.19%
- OCSVM annual anomaly-rate SD: 17.75%
- Isolation Forest mean target gap: 6.56%
- OCSVM mean target gap: 13.75%
- IF mean seed Jaccard: 0.9670

## Acceptance Gates

- [PASS] IF improves synthetic sensitivity: value=0.6000, reference=0.5782
- [PASS] IF seed stability >= 0.90: value=0.9670, reference=0.9000
- [FAIL] IF temporal SD <= 2x baseline: value=0.0719, reference=0.0657
- [FAIL] IF mean target gap <= 5%: value=0.0656, reference=0.0500
- [FAIL] OCSVM mean target gap <= 5%: value=0.1375, reference=0.0500
- [FAIL] Hard regime conditioning improves IF: value=0.0384, reference=0.0299

## Operational Interpretation

Milestone 2 should therefore output both a transparent binary classification and a complementary machine-learning score:

- `Anomaly_State`: NORMAL or ANOMALOUS
- `Baseline_Anomaly_Score`: transparent rule-based severity
- `IF_Anomaly_Score`: continuous multivariate abnormality score

The Isolation Forest score must not be interpreted as a probability of anomaly.