# Project 3 — USD/TRY Direction Classification

## 1. Objective

Project 3 extends the USD/TRY market-regime and anomaly-detection work into supervised direction classification. The objective is to classify the **next 5 trading days** of USD/TRY movement using only information available at observation time `t`.

The final target is:

- **DOWN:** future 5-trading-day return < -0.5%
- **FLAT:** -0.5% <= future 5-trading-day return <= +0.5%
- **UP:** future 5-trading-day return > +0.5%

The final model provides a directional probability distribution rather than directly issuing a BUY/HOLD/SELL instruction.

## 2. Dataset and Target Engineering

The supervised dataset contains **2,440 labeled observations** spanning **2017-03-27 to 2026-07-31**.

Candidate fixed-threshold and volatility-adjusted targets were tested. The selected primary target was **5 trading days at ±0.5%**.

| Class | Share |
|---|---:|
| DOWN | 17.79% |
| FLAT | 43.32% |
| UP | 38.89% |

Annual class shares show pronounced temporal drift, with DOWN disappearing from the 2025 and 2026 labeled samples. This motivated expanding chronological validation and class-aware metrics.

## 3. Predictor Set

Eight causal predictors were retained:

1. `Return_1D`
2. `Return_5D`
3. `Volatility_5D`
4. `Volatility_20D`
5. `Volatility_60D`
6. `MA_Distance_20D`
7. `MA_Slope_20D`
8. `Drawdown_60D`

Raw `USDTRY`, year, future-return columns, and target columns were excluded from the model inputs.

## 4. Naive Baselines

Three expanding-window baselines were evaluated over 2021–2026.

| Baseline | Accuracy | Balanced Accuracy | Macro F1 |
|---|---:|---:|---:|
| Majority | 35.30% | 33.33% | 17.39% |
| Prior Random | 35.99% | 35.52% | 31.91% |
| **Momentum-5D** | **63.19%** | **50.06%** | **50.03%** |

Momentum-5D became the principal hurdle for subsequent supervised models.

## 5. Logistic Regression

| Model | Accuracy | Balanced Accuracy | Macro F1 |
|---|---:|---:|---:|
| Logistic | 58.72% | 48.01% | 48.03% |
| Logistic Balanced | 59.41% | 46.46% | 43.54% |

Neither linear model exceeded the momentum benchmark.

## 6. Tree Models

| Model | Accuracy | Balanced Accuracy | Macro F1 |
|---|---:|---:|---:|
| **Random Forest** | **64.70%** | **53.56%** | **53.60%** |
| Random Forest Balanced | 60.92% | 51.40% | 49.71% |
| HistGradientBoosting | 50.62% | 43.85% | 42.64% |
| HistGradientBoosting Balanced | 53.37% | 45.87% | 44.42% |

The unweighted Random Forest was the first supervised model to beat Momentum-5D on both primary class-aware metrics.

## 7. Random Forest Stability and Robustness

The unpurged pooled RF result improved balanced accuracy by **+3.50 pp** and macro F1 by **+3.57 pp** relative to momentum.

Pooled class performance:

| Model | Class | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| Random Forest | DOWN | 29.73% | 26.40% | 27.97% |
| Random Forest | FLAT | 78.98% | 72.22% | 75.45% |
| Random Forest | UP | 53.34% | 62.06% | 57.37% |
| Momentum-5D | DOWN | 19.53% | 20.00% | 19.76% |
| Momentum-5D | FLAT | 74.82% | 74.54% | 74.68% |
| Momentum-5D | UP | 55.64% | 55.64% | 55.64% |

RF won balanced accuracy in **4/6 years** and macro F1 in **3/6 years**.

Confidence was monotonic with observed accuracy:

| Confidence | Accuracy |
|---|---:|
| <50% | 50.72% |
| 50–60% | 59.21% |
| 60–70% | 66.53% |
| 70–80% | 76.67% |
| 80%+ | 95.08% |

Five canonical seeds all beat momentum. Mean pairwise seed agreement was **97.80%** and minimum agreement was **96.91%**. All **5/5 robustness gates passed**.

The frozen classifier specification is:

```python
RandomForestClassifier(
    n_estimators=500,
    max_depth=6,
    min_samples_leaf=10,
    max_features="sqrt",
    class_weight=None,
    random_state=42,
    n_jobs=-1,
)
```

## 8. Cross-Milestone Ablation

Leakage-safe M1 regime and M2 anomaly features were tested on a matched 2022–2026 sample.

| Architecture | Balanced Accuracy | Macro F1 |
|---|---:|---:|
| **Base RF** | **46.46%** | **46.25%** |
| Base RF + M1 | 45.98% | 45.63% |
| Base RF + M2 | 44.67% | 44.04% |
| Base RF + M1 + M2 | 44.85% | 43.95% |

M1 slightly increased DOWN recall on the restricted matched sample, but all augmented architectures reduced overall performance.

**Decision:** M1 and M2 remain contextual modules and are not inputs to M3.

## 9. Purged Chronological Validation

Because each label uses the next five trading days, the final five training observations before each annual test fold were purged to prevent target-window overlap.

Final pooled results:

| Model | Accuracy | Balanced Accuracy | Macro F1 |
|---|---:|---:|---:|
| **Random Forest** | **64.56%** | **53.41%** | **53.49%** |
| Momentum-5D | 63.19% | 50.06% | 50.03% |

Final purged RF advantage:

- balanced accuracy: **+3.35 pp**
- macro F1: **+3.46 pp**

Final purged per-class results:

| Model | Class | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| Random Forest | DOWN | 30.00% | 26.40% | **28.09%** |
| Random Forest | FLAT | 78.70% | 72.34% | **75.38%** |
| Random Forest | UP | 53.11% | 61.48% | **56.99%** |
| Momentum-5D | DOWN | 19.53% | 20.00% | 19.76% |
| Momentum-5D | FLAT | 74.82% | 74.54% | 74.68% |
| Momentum-5D | UP | 55.64% | 55.64% | 55.64% |

The purge changed the RF result only marginally, confirming that the performance advantage was not driven by label-window boundary leakage.

## 10. Probability Calibration

Chronological expanding temperature scaling was evaluated for 2022–2026.

Temperatures were:

- 2022: 0.9974
- 2023: 0.9796
- 2024: 0.9781
- 2025: 0.9648
- 2026: 0.8880

Pooled calibration:

| Metric | Uncalibrated | Temperature scaled |
|---|---:|---:|
| Log loss | 0.74239 | **0.73858** |
| Brier score | 0.44734 | **0.44581** |
| Confidence ECE | 0.06190 | **0.05697** |
| Mean classwise ECE | 0.07873 | **0.07605** |

Temporal wins:

- log loss: **5/5**
- Brier: **3/5**
- ECE: **4/5**
- selection gates: **5/5**

Temperature scaling preserved all class predictions.

## 11. Final Model Selection

All **11/11 final acceptance gates passed**.

Final selected architecture:

- Target: `Target_5D_0p5pct`
- Features: `BASE_8_FEATURES`
- Classifier: canonical Random Forest
- M1 features: rejected
- M2 features: rejected
- Calibration: temperature scaling
- Status: `SELECTED_FOR_FINAL_TRAINING`

## 12. Final Frozen Training

The final production RF was fitted on all **2,440 labeled observations**.

Training distribution:

| Class | Count | Share |
|---|---:|---:|
| DOWN | 434 | 17.79% |
| FLAT | 1,057 | 43.32% |
| UP | 949 | 38.89% |

The production temperature was fitted using all **1,456 purged OOS RF probability observations**:

**T = 0.841805**

Fit diagnostics:

- raw OOS log loss: **0.790844**
- temperature-fit log loss: **0.786317**
- class predictions unchanged: **True**
- serialized model reload maximum difference: **2.220e-16**

Final feature importance:

| Feature | Importance |
|---|---:|
| Volatility_20D | 21.37% |
| Volatility_5D | 20.32% |
| Volatility_60D | 17.83% |
| MA_Slope_20D | 9.44% |
| Drawdown_60D | 9.28% |
| Return_5D | 9.25% |
| MA_Distance_20D | 9.04% |
| Return_1D | 3.46% |

The volatility environment dominates the final model importance ranking, showing that the RF is not merely reproducing the 5-day momentum rule.

## 13. Operational Inference

First latest-row operational inference:

- observation date: **2026-08-07**
- USD/TRY: **47.7111**
- training cutoff: **2026-07-31**
- inference mode: `OUT_OF_SAMPLE_OPERATIONAL`

Prediction:

**FLAT**

| Direction | Calibrated probability |
|---|---:|
| DOWN | 0.27% |
| **FLAT** | **87.94%** |
| UP | 11.79% |

Diagnostics:

- confidence: **87.94%**
- probability margin: **76.15 pp**
- raw RF: DOWN 0.65%, FLAT 83.89%, UP 15.46%
- final temperature: **0.841805**

The operational result predicts that the next 5-trading-day USD/TRY return will most likely remain inside the predefined **-0.5% to +0.5% FLAT band**.

## 14. Conclusion

Project 3 demonstrates a modest but repeatable improvement over a strong momentum heuristic. The final purged Random Forest achieved **64.56% accuracy, 53.41% balanced accuracy, and 53.49% macro F1**, while improving DOWN-class F1 from **19.76% to 28.09%** relative to Momentum-5D.

The advantage survived seed variation, nearby hyperparameter variants, chronological purge validation, and direct temporal stability checks. M1 regime and M2 anomaly features did not add incremental direction-prediction value and were excluded from the final classifier. Temperature scaling provided a small but consistent improvement in probability quality without altering class predictions.

The final M3 production architecture is therefore a **temperature-calibrated, 8-feature Random Forest for 5-trading-day USD/TRY direction classification**.

## 15. Production Artifacts

```text
models/milestone_3/usdtry_direction_rf.joblib
models/milestone_3/usdtry_direction_temperature.json
models/milestone_3/usdtry_direction_metadata.json
models/milestone_3/usdtry_direction_feature_importance.csv

outputs/milestone_3/operational/latest_direction.json
outputs/milestone_3/operational/direction_inference_history.csv
```

Model version: **M3-v0.1.0**
