# USD/TRY Machine-Learning Decision-Support System

A three-module machine-learning framework for analyzing the USD/TRY foreign-exchange market through:

1. **Market Regime Detection**
2. **FX Anomaly Detection**
3. **5-Day Direction Classification**

The project is designed for **low-frequency financial monitoring**, not high-frequency trading. All final models use chronological validation, causal features, leakage controls, and frozen operational inference.

---

## Project Overview

```text
USD/TRY historical data
        ↓
Causal financial features
        ↓
┌─────────────────────────────────────────────┐
│ P1 — Market Regime Detection               │
│ 3-state Gaussian HMM                       │
│ LOW_VOL / ELEVATED_VOL / HIGH_VOL_STRESS  │
└─────────────────────────────────────────────┘
        +
┌─────────────────────────────────────────────┐
│ P2 — FX Anomaly Detection                  │
│ Statistical baseline + Isolation Forest    │
│ NORMAL / ANOMALOUS + anomaly score         │
└─────────────────────────────────────────────┘
        +
┌─────────────────────────────────────────────┐
│ P3 — Direction Classification              │
│ Calibrated Random Forest                   │
│ DOWN / FLAT / UP over next 5 trading days │
└─────────────────────────────────────────────┘
```

The three modules intentionally remain separate. P1 and P2 outputs were tested as additional P3 predictors but did not improve out-of-sample directional classification.

---

# Project 1 — Market Regime Detection

P1 identifies the prevailing statistical environment of USD/TRY rather than predicting its future price.

### Dataset

* Feature-complete observations: **2,445**
* Period: **2017-03-27 → 2026-08-07**
* Weekday observations only

### Features

Eight causal financial features are used:

* `Return_1D`
* `Return_5D`
* `Volatility_5D`
* `Volatility_20D`
* `Volatility_60D`
* `MA_Distance_20D`
* `MA_Slope_20D`
* `Drawdown_60D`

### Model comparison

K-Means was used as the static clustering baseline. Although it achieved stronger geometric separation, its states were less temporally persistent. The final model is a **3-state diagonal Gaussian Hidden Markov Model**.

| Regime Code       | Interpretation                               |
| ----------------- | -------------------------------------------- |
| `LOW_VOL`         | Orderly low-volatility trend                 |
| `ELEVATED_VOL`    | Transitional elevated-volatility environment |
| `HIGH_VOL_STRESS` | High-volatility market stress                |

### Validation

* HMM median episode: **14 observations**
* K-Means median episode: **3**
* HMM mean self-transition: **96.74%**
* Walk-forward mean switch rate: **3.27%**
* State-ordering stability: **95.83%**
* Minimum standardized state separation: **1.0175**
* Final model-selection gates: **7/7 PASS**

Expanding walk-forward validation covered 2021–2026, with causal forward filtering used for unseen observations.

### Final model

```text
8 causal features
      ↓
StandardScaler
      ↓
3-state Gaussian HMM
      ↓
Causal filtering
      ↓
LOW_VOL / ELEVATED_VOL / HIGH_VOL_STRESS
```

---

# Project 2 — FX Anomaly Detection

P2 identifies observations that are unusual relative to historical USD/TRY behavior.

The final architecture deliberately separates the operational decision from the ML score:

```text
Primary:
Statistical baseline
→ NORMAL / ANOMALOUS

Secondary:
Isolation Forest
→ multivariate anomaly score
→ historical percentile
```

### Anomaly features

P2 extends the P1 feature set with:

* absolute 1-day return;
* causal 20-day return z-score;
* 5D/60D volatility ratio;
* 5-day drawdown change.

The return z-score uses a **training-only volatility floor**. Each chronological validation fold estimates the floor from its training observations and freezes it for test transformation. The final operational model uses a floor of:

```text
0.0007028294836738612
```

estimated from data through `2026-08-07`.

### Statistical baseline

An observation becomes anomalous when at least one normalized rule exceeds its threshold:

```text
|20D return z-score|        ≥ 4
5D / 60D volatility ratio   ≥ 2
|5D drawdown change|        ≥ 5%
|1D return|                 ≥ 4%
```

Historical baseline anomaly frequency:

**6.68%**

### Alternative models

Evaluated:

* Isolation Forest
* One-Class SVM
* regime-conditioned anomaly models

Isolation Forest was highly stable across random seeds with mean pairwise Jaccard ≈ **0.97**, but its binary threshold was less stable chronologically. One-Class SVM showed strong synthetic sensitivity but poor temporal calibration. Hard regime conditioning also worsened out-of-sample calibration.

Therefore:

* **Statistical baseline** = operational binary detector
* **Isolation Forest** = secondary continuous signal

---

# Project 3 — USD/TRY Direction Classification

P3 predicts the direction of USD/TRY over the **next five trading days**.

### Target

```text
DOWN  → Future 5D return < -0.5%
FLAT  → -0.5% ≤ Future 5D return ≤ +0.5%
UP    → Future 5D return > +0.5%
```

The supervised dataset contains **2,440 observations** from `2017-03-27` through `2026-07-31`.

### Final predictor set

The same eight causal P1 market features are used.

Raw exchange-rate level, year, future returns, target variables, P1 regime outputs, and P2 anomaly outputs are excluded from the final predictor matrix.

### Baseline

The strongest heuristic was **Momentum-5D**:

| Metric            | Momentum-5D |
| ----------------- | ----------: |
| Accuracy          |      63.19% |
| Balanced Accuracy |      50.06% |
| Macro F1          |      50.03% |

### Final Random Forest

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

Chronological expanding validation uses a **5-trading-day purge** at train/test boundaries to prevent future target windows from overlapping the test period.

### Final out-of-sample results

| Metric            | Random Forest | Momentum-5D |
| ----------------- | ------------: | ----------: |
| Accuracy          |    **64.56%** |      63.19% |
| Balanced Accuracy |    **53.41%** |      50.06% |
| Macro F1          |    **53.49%** |      50.03% |
| DOWN F1           |    **28.09%** |      19.76% |

The Random Forest therefore improved balanced accuracy by **3.35 percentage points** and macro F1 by **3.46 percentage points** after leakage-controlled validation.

### Robustness

Five canonical random seeds all remained above Momentum-5D.

* Mean pairwise prediction agreement: **97.80%**
* Minimum agreement: **96.91%**
* Robustness gates: **5/5 PASS**

### Cross-project ablation

Adding P1 regime features or P2 anomaly features did **not** improve the matched-sample directional classifier.

The final P3 model therefore retains only the eight base financial predictors.

### Probability calibration

Chronological temperature scaling improved probability quality without changing predicted classes.

Final production temperature:

```text
T = 0.841805
```

### Model selection

**11/11 acceptance gates passed.**

Final architecture:

```text
8 causal financial features
        ↓
Random Forest
        ↓
Temperature scaling
        ↓
DOWN / FLAT / UP probabilities
```

---

# Latest Operational Reference

For the observation on **2026-08-07**:

```text
USD/TRY:       47.7111
P1 Regime:     LOW_VOL
P3 Direction:  FLAT

P(DOWN):        0.27%
P(FLAT):       87.94%
P(UP):         11.79%
```

The P3 observation is genuinely out-of-sample because its final supervised training cutoff is `2026-07-31`.

---

# Validation Philosophy

The project avoids conventional random train/test splitting because financial observations are temporally dependent.

Validation includes:

* chronological holdouts;
* expanding walk-forward validation;
* training-only preprocessing;
* target-window purging;
* causal HMM filtering;
* temporal leakage tests;
* synthetic anomaly injection;
* random-seed stability;
* hyperparameter sensitivity;
* cross-module ablation;
* probability calibration;
* serialized-model reproducibility;
* unit, integration, and regression testing.

The objective is not to maximize an isolated historical metric, but to determine whether model behavior survives realistic chronological deployment.

---

# Repository Structure

```text
market_regime_detection/
│
├── data/
├── docs/
├── models/
│   └── milestone_3/
├── outputs/
│   └── milestone_3/
│
├── src/
│   ├── milestone_1_regime_detection/
│   ├── milestone_2_anomaly_detection/
│   └── milestone_3_direction/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── regression/
│
├── pytest.ini
├── requirements-dev.txt
└── README.md
```

---

# Operational Role

The ML system is intended to complement—not replace—the existing low-frequency financial tracker.

```text
Existing financial rules
BUY / HOLD / SELL
        +
P1 Market Regime
        +
P2 Anomaly State
        +
P3 Direction Probabilities
        ↓
Decision-support output
        ↓
Excel / Telegram / dashboard
```

No individual ML output should be interpreted as a guaranteed trading signal.

---

# Project Status

| Module                        | Status     |
| ----------------------------- | ---------- |
| P1 — Market Regime Detection  | ✅ Complete |
| P2 — FX Anomaly Detection     | ✅ Complete |
| P3 — Direction Classification | ✅ Complete |

**Project 4 forecasting was deliberately not pursued.** Standard forecasting is already available in the existing Excel workflow, while the incremental value of another ML regression layer was judged insufficient relative to the additional complexity.

Future updates will include integration with my Telegram bot and its spreadsheet in the other repository, as well as expansion into other precious metals (gold, silver, etc) and stocks. Other applications may include forecasting.
