# K-Means vs Gaussian HMM Model Comparison

## Objective

The purpose of this comparison is to determine whether the
Gaussian Hidden Markov Model provides a more useful market-regime
representation than the K-Means baseline for daily USD/TRY data.

The models are compared using geometric separation, temporal
persistence, state balance, cross-model agreement, and economic
interpretability.

## Geometric Separation

| Metric | K-Means | HMM |
|---|---:|---:|
| Silhouette | 0.5661 | 0.1672 |
| Davies-Bouldin | 1.1536 | 1.7235 |
| Calinski-Harabasz | 805.77 | 425.14 |

Higher Silhouette and Calinski-Harabasz values indicate stronger
geometric separation, while a lower Davies-Bouldin value is
preferred.

These metrics must not be interpreted as a direct likelihood-based
test between K-Means and HMM. K-Means explicitly optimizes geometric
cluster structure, whereas the HMM additionally models temporal
state transitions.

## Temporal Persistence

| Metric | K-Means | HMM |
|---|---:|---:|
| Episodes | 73 | 70 |
| State transitions | 72 | 69 |
| Switch rate | 2.95% | 2.82% |
| Mean episode duration | 33.49 | 34.93 |
| Median episode duration | 3.00 | 14.00 |
| Maximum episode duration | 768 | 296 |
| Mean self-transition probability | 90.70% | 96.74% |

The HMM produces more temporally coherent regimes than K-Means, with fewer switches and longer typical regime episodes.

## Cross-Model Agreement

Adjusted Rand Index:

**0.1654**

Normalized Mutual Information:

**0.2393**

Best possible label-matched agreement:

**55.17%**

The best-match agreement is computed after relabeling the arbitrary
numeric cluster identifiers using an optimal assignment. It should
not be interpreted as evidence that the economic meaning of the
K-Means and HMM states is identical.

## Interpretation

K-Means serves as the static clustering baseline. Each observation is
classified according to its position in feature space without using
the previous market state.

The Gaussian HMM models both the observed feature distributions and
the probability of transitioning between latent states. Consequently,
it is designed to represent persistent market conditions rather than
independent daily clusters.

The HMM states obtained in this analysis are interpreted as:

1. Low-Volatility Trend
2. Elevated-Volatility Transition
3. High-Volatility Stress

The HMM transition structure additionally suggests that movement
between the low- and high-volatility states generally occurs through
the elevated-volatility state.

## Model Selection

For a market-regime monitoring system, temporal coherence is a core
requirement. Therefore, geometric clustering metrics alone should not
determine the preferred model.

If the HMM preserves economically interpretable state profiles while
reducing regime flickering and increasing episode persistence, it is
preferred over K-Means for operational regime detection.

K-Means remains useful as a transparent benchmark against which the
temporal HMM can be evaluated.

## Remaining Validation

This comparison is based on models fitted to the complete historical
dataset. It is therefore an in-sample model comparison rather than a
prospective performance test.

The next stage should use chronological out-of-sample and walk-forward
validation to determine whether the HMM state structure remains stable
when applied to future observations.