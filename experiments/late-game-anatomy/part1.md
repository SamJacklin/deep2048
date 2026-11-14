# Stage 7, Part 1: does the entry position matter?

Generated 2026-11-14 16:30 from `p1-greedy` and `p1-depth3`.

## 6.1 Position-dependent conversion

| Policy | Positions × continuations | Makes 32768 | rho (32768) [95%] | SD of true conversion (95% upper bound) | Mean score added | Between-position share of score variance [95%] |
|---|---:|---:|---:|---:|---:|---:|
| greedy | 5,000 × 64 | 11.24% | 0.000 [0.000, 0.001] | 0.54% (≤ 0.94%) | 161,330 | 0.1% [0.0%, 0.2%] |
| depth 3 | 1,000 × 8 | 18.74% | 0.000 [0.000, 0.008] | 0.04% (≤ 3.57%) | 201,365 | 0.0% [0.0%, 1.1%] |

rho and the SD of true conversion come from a beta-binomial fit; the upper bound uses the top of rho's bootstrap interval. A single beta distribution describes the bulk of positions, so a small subgroup of unusual positions could move it very little.

On the depth-3 subset's 1,000 positions, greedy makes 32768 in 11.3% and depth 3 in 18.7%; the per-position rates correlate at r = -0.02 [95%: -0.07, 0.04, resampling positions]. Depth 3 has only 8 continuations per position, and with so little between-position variance both per-position rates are mostly binomial noise: even if the true rates were perfectly correlated, the expected observed r would be at most about 0.06 (using the upper bounds of rho). So a near-zero r isn't evidence that the two are unrelated.

### Binomial overdispersion check

Independent of the beta-binomial fit: if every position had the same conversion (the pooled rate), each position's count of continuations making 32768 would be binomial. The observed variance of the counts is compared with the binomial variance; the simulated range is the 95% range of that ratio over 10,000 simulations with no position effect, and p is the share of simulations whose variance is at least the observed one. rho by moments is (ratio − 1) / (continuations − 1).

| Policy | Pooled rate | Observed variance | Binomial variance | Ratio [95% bootstrap] | Simulated 95% range, no effect | p | rho by moments (95% upper bound) |
|---|---:|---:|---:|---:|---:|---:|---:|
| greedy | 11.24% | 6.487 | 6.387 | 1.016 [0.976, 1.057] | 0.961–1.040 | 0.212 | 0.0002 (≤ 0.0009) |
| depth 3 | 18.74% | 1.191 | 1.218 | 0.978 [0.891, 1.065] | 0.912–1.091 | 0.679 | -0.0032 (≤ 0.0093) |

Distribution of positions by continuations making 32768, greedy (64 each):

| Continuations making 32768 | Positions observed | Expected if binomial |
|---:|---:|---:|
| 0 | 4 | 2.4 |
| 1 | 19 | 19.6 |
| 2 | 82 | 78.2 |
| 3 | 197 | 204.8 |
| 4 | 424 | 395.8 |
| 5 | 602 | 601.7 |
| 6 | 725 | 749.6 |
| 7 | 765 | 786.9 |
| 8 | 699 | 710.3 |
| 9 | 600 | 560.0 |
| 10 | 389 | 390.2 |
| 11 | 233 | 242.7 |
| 12 | 132 | 135.8 |
| 13 | 76 | 68.8 |
| 14 | 29 | 31.8 |
| 15 | 16 | 13.4 |
| 16 | 2 | 5.2 |
| 17+ | 6 | 2.7 |

Distribution of positions by continuations making 32768, depth 3 (8 each):

| Continuations making 32768 | Positions observed | Expected if binomial |
|---:|---:|---:|
| 0 | 183 | 190.2 |
| 1 | 360 | 350.8 |
| 2 | 286 | 283.1 |
| 3 | 124 | 130.5 |
| 4 | 41 | 37.6 |
| 5 | 5 | 6.9 |
| 6+ | 1 | 0.9 |

**greedy:** the beta-binomial interval includes zero and the overdispersion check does not detect extra variance (p = 0.212); upper bounds on rho: 0.0009 (beta-binomial) and 0.0009 (moments).

**depth 3:** the beta-binomial interval includes zero and the overdispersion check does not detect extra variance (p = 0.679); upper bounds on rho: 0.0085 (beta-binomial) and 0.0093 (moments).

## 6.2 Predictive board features

Out-of-fold predictions from L2-penalised logistic regression on all features, 5 folds grouped by source game, against a constant predictor fitted on each training fold. Scores are per continuation; lower is better.

| Model | Penalty | Brier | Log loss | Brier skill vs constant | SD of predictions |
|---|---:|---:|---:|---:|---:|
| Constant | — | 0.09980 | 0.35161 | 0 | 0 |
| All features | 0.1 | 0.09981 | 0.35162 | -0.00% | 0.26% |
| All features | 1 | 0.09981 | 0.35162 | -0.00% | 0.26% |
| All features | 10 | 0.09981 | 0.35162 | -0.00% | 0.26% |
| All features | 100 | 0.09981 | 0.35162 | -0.00% | 0.25% |

Out of fold, no penalty improves on the constant predictor: the best Brier score is 0.09981 (penalty 100) against 0.09980, and the best log loss 0.35162 against 0.35161. So these features don't predict which positions convert better than giving every position the pooled rate.

Calibration (penalty 1), by tenths of predicted conversion:

| Tenth | Predicted | Observed |
|---:|---:|---:|
| 1 | 10.9% | 11.3% |
| 2 | 11.1% | 11.3% |
| 3 | 11.2% | 11.6% |
| 4 | 11.2% | 11.4% |
| 5 | 11.2% | 11.3% |
| 6 | 11.3% | 11.2% |
| 7 | 11.3% | 11.3% |
| 8 | 11.3% | 10.9% |
| 9 | 11.4% | 11.0% |
| 10 | 11.5% | 11.1% |

Each feature on its own: conversion by the feature's level, over all continuations.

| Feature | Level | Positions | Makes 32768 | Mean score added |
|---|---|---:|---:|---:|
| Top tile | corner | 4,529 | 11.3% | 161,400 |
| Top tile | edge | 366 | 11.1% | 160,323 |
| Top tile | interior | 105 | 11.6% | 161,813 |
| empty cells | 1–6 | 1,159 | 11.2% | 160,810 |
| empty cells | 7–7 | 1,427 | 11.1% | 161,144 |
| empty cells | 8–11 | 2,414 | 11.3% | 161,690 |
| 2nd largest (exponent) | 3–3 | 1,537 | 11.3% | 161,501 |
| 2nd largest (exponent) | 4–12 | 3,463 | 11.2% | 161,254 |
| 2nd largest: distance from top | 1–1 | 1,195 | 11.3% | 161,631 |
| 2nd largest: distance from top | 2–5 | 3,805 | 11.2% | 161,236 |
| 3rd largest (exponent) | 2–2 | 1,250 | 11.2% | 161,461 |
| 3rd largest (exponent) | 3–9 | 3,750 | 11.3% | 161,286 |
| 3rd largest: distance from top | 1–1 | 1,548 | 11.2% | 161,191 |
| 3rd largest: distance from top | 2–6 | 3,452 | 11.3% | 161,392 |
| monotonicity | 0.666667–0.75 | 876 | 11.2% | 160,734 |
| monotonicity | 0.791667–0.791667 | 1,250 | 11.3% | 161,497 |
| monotonicity | 0.833333–1 | 2,874 | 11.3% | 161,439 |
| legal moves | 2–3 | 1,307 | 11.2% | 161,159 |
| legal moves | 4–4 | 3,693 | 11.3% | 161,391 |
| adjacent equal pairs | 0–0 | 1,334 | 11.2% | 161,226 |
| adjacent equal pairs | 1–1 | 1,894 | 11.3% | 161,226 |
| adjacent equal pairs | 2–5 | 1,772 | 11.2% | 161,520 |
| chain from top tile | 1–1 | 5,000 | 11.2% | 161,330 |
| tiles of 256+ besides the top | 0–2 | 5,000 | 11.2% | 161,330 |
| other tiles' sum / 1,000 | 0.026–0.032 | 1,458 | 11.2% | 161,470 |
| other tiles' sum / 1,000 | 0.034–0.038 | 1,545 | 11.4% | 161,743 |
| other tiles' sum / 1,000 | 0.04–4.73 | 1,997 | 11.2% | 160,908 |

## 6.3 Predicted best against predicted worst tenth (out of fold, penalty 1)

| Group | Positions | Makes 32768 | Mean score added |
|---|---:|---:|---:|
| Predicted best 10% | 500 | 11.1% | 161,231 |
| Predicted worst 10% | 500 | 11.3% | 161,039 |

Difference in conversion, best minus worst: -0.3 percentage points [95%: -0.7, +0.2].

| Feature | Best 10% | Worst 10% | All |
|---|---:|---:|---:|
| top tile in a corner | 0.79 | 0.69 | 0.91 |
| top tile on an edge | 0.03 | 0.30 | 0.07 |
| empty cells | 7.85 | 6.39 | 7.37 |
| 2nd largest (exponent) | 3.82 | 4.51 | 3.90 |
| 2nd largest: distance from top | 1.27 | 2.47 | 2.01 |
| 3rd largest (exponent) | 2.89 | 3.27 | 2.88 |
| 3rd largest: distance from top | 2.16 | 2.17 | 2.09 |
| monotonicity | 0.87 | 0.79 | 0.83 |
| legal moves | 3.86 | 3.39 | 3.72 |
| adjacent equal pairs | 1.67 | 0.80 | 1.22 |
| chain from top tile | 1.00 | 1.00 | 1.00 |
| tiles of 256+ besides the top | 0.01 | 0.02 | 0.00 |
| other tiles' sum / 1,000 | 0.04 | 0.10 | 0.05 |

This table shows which boards the fitted model ranks highest and lowest. Since the model doesn't beat the constant predictor out of fold, these differences aren't evidence that any feature affects conversion.

## 6.4 Summary

- **Greedy, 5,000 positions × 64:** conversion is 11.24%. There is little detectable variation in conversion among these entry positions, all genuine first-16384 boards from greedy play of the 2M network (stage 6's dev pool): rho = 0.0003 [0.0000, 0.0009], so the SD of true conversion across positions is at most about 0.94% (95% upper bound) around 11.24%. The overdispersion check agrees: variance ratio 1.016 [0.976, 1.057], p = 0.212.
- **What this doesn't show:** that entry quality can never matter. It holds for positions this policy produces, which may be similar to one another; other policies or deliberately varied positions could differ. And a small subgroup of unusual positions with very different conversion would barely change rho or the variance ratio.
- **Depth 3, 1,000 positions × 8:** conversion is 18.74%, and rho = 0.0000 [0.0000, 0.0085] (SD of true conversion ≤ 3.57%). With 8 continuations per position this is far less precise than greedy's 64, so it's consistent with no position effect but doesn't establish it; likewise the greedy/depth-3 correlation of r = -0.02 [-0.07, 0.04] can't show the two are unrelated, since even a perfect true correlation would show as r ≤ about 0.06 here.
- **Features:** none of the feature models improves on the constant predictor out of fold, and the predicted best tenth converts -0.3 percentage points [95%: -0.7, +0.2] relative to the predicted worst tenth. So the current features don't identify reliably better entry positions.
