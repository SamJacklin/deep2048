# Stage 7 on the Ryzen: results

Run on the Ryzen under WSL2 Ubuntu, 2026-11-14, following the `ryzen-stage7-instructions` README. All times are UTC.

## 1. The machine (B3)

```
CPU(s):                               16
Model name:                           AMD Ryzen 7 9850X3D 8-Core Processor
Thread(s) per core:                   2
               total        used        free      shared  buff/cache   available
Mem:              27           0          21           0           5          26
/dev/sdd       1007G   60G  897G   7% /
```

## 2. The commit (B1)

`feat/late-game-anatomy` at `5fd1b15828a174ce0be5cb6777e716374a05ac3f` ("style: wrap two long report lines in stage 7's script"). `git status --short` beforehand listed only three untracked logs from the earlier long runs (`ryzen-optimism-ext.log`, `ryzen-optimism.log`, `ryzen-recipe.log`). `uv sync` was clean.

```
129 passed in 91.23s (0:01:31)
```

**Later code change (at the user's request, see section 6):** `scripts/late_game_anatomy.py` was modified during the run, and `tests/test_part1_stats.py` was added, in commit `7c6593a` on `results/late-game-anatomy-ryzen`. `manifest.json` records commit 5fd1b15, since it was written at the start. The steps after 16:30 (Part 1's regenerated report, the pilot report, p2-plan, the expansion and the final report) ran with the modified script. The change was to Part 1's report only: no rollout, seed, pool or Part 2 code changed.

## 3. The network copy (B2)

The user copied the network into WSL themselves before this session reached B2 (directory created 14:19:17, file mtime kept from 2026-11-12 14:59:23).

```
2026-11-14T14:53:31+00:00
total 1572880
drwxr-xr-x 2 sam11 sam11       4096 2026-11-14 14:19:17.087678780 +0000 .
drwxr-xr-x 6 sam11 sam11       4096 2026-11-14 14:17:26.709869070 +0000 ..
-rw-r--r-- 1 sam11 sam11 1610615644 2026-11-12 14:59:23.000000000 +0000 games_2000000.npz
8f117f2b449ddba0c48e8a1d71b1afe4720b7b8d5c168b88ac9fbfe8b0dde513  checkpoints/ntuple/decision-8x6-tc-reset/games_2000000.npz
```

The sha256 matches the expected `8f117f2b449ddba0c48e8a1d71b1afe4720b7b8d5c168b88ac9fbfe8b0dde513`.

## 4. The smoke test (B4)

14:53:40 to 14:56:58. Exit 0, last line `stage 7 finished`, no errors or warnings in the log. The smoke folder had `part1.md`, `part2-pilot.md`, `p2-plan.json`, `part2-final.md`, `costs.jsonl` and `manifest.json`, among others. Its outputs were deleted from the repo afterwards as instructed. The end of what it printed, with the replay check and the folder listing:

```
2026-11-14T14:53:40+00:00
exit: 0
| 501–1,500 | 8 | +9.4% [+6.2%, +12.5%] | +35,434 [+22,766, +48,103] |
| 1,501–3,000 | 12 | -2.1% [-8.3%, +4.2%] | -1,439 [-27,729, +24,851] |
| over 3,000 | 38 | +2.0% [+0.0%, +3.9%] | +2,395 [+129, +4,660] |
2026-11-14 14:56:58 finished p2-report --phase final
2026-11-14 14:56:58 stage 7 finished
{
  "positions": 60,
  "continuations 0 and 1 identical to stage 6's control": true
}
costs.jsonl
depth3-subset.json
manifest.json
p2-plan.json
p2-trajectories-expansion.json
p2-trajectories-pilot.json
part1.json
part1.md
part2-final.md
part2-pilot.md
stage6-replay-check.json
2026-11-14T14:56:58+00:00
```

The full run's replay check (`stage6-replay-check.json`):

```
{
  "positions": 5000,
  "continuations 0 and 1 identical to stage 6's control": true
}

```

## 5. Windows Update

Not paused. It was found unpaused at 14:51, and at 15:03 the user chose to leave it ("I'll do it later"). It stayed unpaused for the whole run, and caused no restart or interruption.

## 6. Timings

The run was started at 14:57:10 with the exact C1 command, and finished at 21:52:48 (`stage 7 finished`). There were no interruptions, restarts or errors.

| Step | Started | Finished |
|---|---|---|
| manifest | 14:57:10 | 14:57:11 |
| p1-greedy | 14:57:11 | 15:01:45 |
| p1-depth3 | 15:01:45 | 15:46:39 |
| p1-report | 15:46:39 | 15:46:50 |
| p2-trajectories --phase pilot | 15:46:50 | 15:46:55 |
| p2-run --phase pilot | 15:46:55 | 16:33:25 |
| p2-report --phase pilot | 16:33:25 | 16:33:28 |
| p2-plan | 16:33:28 | 16:33:29 |
| p2-trajectories --phase expansion | 16:33:29 | 16:33:34 |
| p2-run --phase expansion | 16:33:34 | 21:52:44 |
| p2-report --phase final | 21:52:44 | 21:52:48 |

**Problem and code change:**
- **15:46:** in the original `part1.md`, the "Makes 32768" column of the 6.1 table showed 719.7% (greedy) and 149.9% (depth 3). That is 100 × the mean number of continuations per position making 32768, not a rate.
- **16:30:47:** the user asked for this to be fixed, overriding the instruction not to change code, along with other changes to Part 1's report:
  - a binomial overdispersion check;
  - an upper bound on the SD of true conversion;
  - a bootstrap interval and an attenuation ceiling for the greedy/depth-3 correlation;
  - reporting the out-of-fold Brier and log-loss comparison in place of the explained-variance figure;
  - more cautious wording, and a 6.4 summary.
- **How it was done:** `part1.md` was regenerated from the saved Part 1 data. No rollouts were rerun, and the Part 2 pilot wasn't interrupted. The change was developed in a separate worktree and moved into place while the pilot ran.
- **What changed in `part1.json`:** only "makes 32768" (7.1966 → 0.1124 and 1.499 → 0.1874), plus the new fields. All other statistics and the rollout chunks' sha256 were unchanged.
- **The pilot report:** `part2-pilot.md` (16:33:28) was generated with the fixed script, since its "natural conversion" line reads that field.
- **C3 interim push (16:34):** commits `7c6593a` (the code change) and `2e5770c` (Part 1 and the pilot report) to `results/late-game-anatomy-ryzen`.

## 7. Compute

| Step | Wall hours | CPU hours | Continuations | Moves |
|---|---:|---:|---:|---:|
| p1-greedy | 0.08 | 1.18 | 320,000 | 2,069,485,870 |
| p1-depth3 | 0.75 | 11.90 | 8,000 | 63,182,103 |
| p2-trajectories pilot | 0.00 | 0.00 | 0 | 0 |
| p2-run pilot policy A | 0.00 | 0.02 | 9,400 | 35,380,510 |
| p2-run pilot policy B | 0.49 | 7.54 | 9,400 | 44,929,736 |
| p2-run pilot policy C | 0.02 | 0.36 | 9,400 | 35,902,398 |
| p2-run pilot policy D | 0.26 | 4.04 | 9,400 | 36,347,504 |
| p2-trajectories expansion | 0.00 | 0.00 | 0 | 0 |
| p2-run expansion policy A | 0.01 | 0.14 | 65,640 | 247,297,195 |
| p2-run expansion policy B | 3.34 | 52.91 | 65,640 | 310,991,731 |
| p2-run expansion policy C | 0.16 | 2.56 | 65,640 | 250,272,200 |
| p2-run expansion policy D | 1.81 | 28.73 | 65,640 | 250,488,181 |

## 8. Results

### part1.md

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

### part2-pilot.md

# Stage 7, Part 2: when can failing games be rescued? (pilot)

Generated 2026-11-14 16:33.

Policies: **A** greedy; **B** depth 3 from here; **C** depth 3 for 200 moves, then greedy; **D** pruned depth 5 (cutoff 0.01) for 200 moves, then greedy.

Each cell is the mean paired difference from greedy (A) over points, then continuations, with a 95% interval from resampling whole trajectories. Moves remaining are retrospective (the agent can't know them).

Trajectories: 20 that made 32768 (sampled up to it) and 20 that didn't. Points: 1,175. Continuations: 37,600.


## B: depth 3 from here


**By moves since the first 16384, failed trajectories**

| Group | Points | Trajectories | Greedy makes 32768 | Δ makes 32768 [95%] | Δ score added [95%] | CPU s per continuation | Chance nodes per continuation |
|---|---:|---:|---:|---:|---:|---:|---:|
| at the first 16384 | 20 | 20 | 13.1% | +4.4% [-3.8%, +12.5%] | +26,773 [+7,984, +45,324] | 3.02 | 440,083 |
| 1–500 moves after | 60 | 20 | 12.9% | +7.3% [+2.9%, +11.9%] | +37,111 [+28,755, +45,205] | 3.02 | 435,466 |
| 501–1,500 | 80 | 20 | 13.3% | +5.6% [+1.4%, +9.4%] | +32,050 [+21,822, +41,840] | 3.02 | 379,586 |
| 1,501–3,000 | 120 | 20 | 9.5% | +8.4% [+5.4%, +11.1%] | +40,867 [+32,890, +48,473] | 3.02 | 304,407 |
| over 3,000 | 275 | 19 | 15.8% | +6.0% [+3.7%, +8.5%] | +21,879 [+16,965, +27,948] | 2.83 | 169,913 |

**By moves since the first 16384, successful trajectories**

| Group | Points | Trajectories | Greedy makes 32768 | Δ makes 32768 [95%] | Δ score added [95%] | CPU s per continuation | Chance nodes per continuation |
|---|---:|---:|---:|---:|---:|---:|---:|
| at the first 16384 | 20 | 20 | 11.2% | +6.9% [-0.6%, +14.4%] | +39,477 [+22,799, +57,522] | 2.85 | 452,459 |
| 1–500 moves after | 60 | 20 | 11.7% | +10.6% [+6.0%, +15.4%] | +45,596 [+34,369, +56,472] | 2.85 | 444,264 |
| 501–1,500 | 80 | 20 | 10.9% | +9.4% [+4.2%, +14.7%] | +43,780 [+31,494, +55,885] | 2.85 | 388,856 |
| 1,501–3,000 | 120 | 20 | 12.3% | +6.6% [+3.9%, +10.0%] | +33,157 [+26,977, +41,027] | 2.85 | 307,960 |
| over 3,000 | 340 | 20 | 19.2% | +5.0% [+2.9%, +6.9%] | +19,901 [+15,316, +24,341] | 2.85 | 176,424 |

**By moves remaining, failed trajectories**

| Group | Points | Trajectories | Greedy makes 32768 | Δ makes 32768 [95%] | Δ score added [95%] | CPU s per continuation | Chance nodes per continuation |
|---|---:|---:|---:|---:|---:|---:|---:|
| over 2,000 moves left | 356 | 20 | 12.3% | +6.2% [+4.5%, +7.8%] | +30,556 [+26,292, +34,673] | 2.87 | 322,182 |
| 501–2,000 left | 120 | 20 | 14.6% | +7.9% [+4.6%, +11.4%] | +28,789 [+20,282, +37,727] | 3.02 | 195,656 |
| 201–500 left | 25 | 20 | 16.5% | +9.5% [+1.6%, +16.7%] | +34,112 [+18,756, +47,030] | 3.09 | 153,010 |
| 200 or fewer left | 54 | 20 | 19.7% | +4.4% [-0.7%, +9.5%] | +19,658 [+8,142, +31,084] | 3.00 | 121,303 |

**By empty cells, failed trajectories**

| Group | Points | Trajectories | Greedy makes 32768 | Δ makes 32768 [95%] | Δ score added [95%] | CPU s per continuation | Chance nodes per continuation |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0–1 empty | 109 | 20 | 14.0% | +7.5% [+3.5%, +11.8%] | +31,363 [+22,652, +40,331] | 2.93 | 259,535 |
| 2–3 empty | 293 | 20 | 14.0% | +7.3% [+5.2%, +9.8%] | +30,790 [+25,060, +37,414] | 2.90 | 256,059 |
| 4–6 empty | 133 | 20 | 13.3% | +4.0% [+1.4%, +6.9%] | +23,976 [+18,296, +30,142] | 2.99 | 281,223 |
| 7+ empty | 20 | 15 | 9.4% | +6.9% [+2.0%, +12.5%] | +30,906 [+16,147, +47,321] | 2.97 | 391,703 |

At the first 16384, weighting successes and failures by greedy's natural 11.2% conversion, B changes conversion by +4.7%.

## C: depth 3 for 200 moves, then greedy


**By moves since the first 16384, failed trajectories**

| Group | Points | Trajectories | Greedy makes 32768 | Δ makes 32768 [95%] | Δ score added [95%] | CPU s per continuation | Chance nodes per continuation |
|---|---:|---:|---:|---:|---:|---:|---:|
| at the first 16384 | 20 | 20 | 13.1% | -1.9% [-8.1%, +4.4%] | -8,042 [-25,032, +9,289] | 0.14 | 15,329 |
| 1–500 moves after | 60 | 20 | 12.9% | -0.4% [-3.5%, +2.9%] | +2,086 [-3,497, +8,127] | 0.14 | 13,254 |
| 501–1,500 | 80 | 20 | 13.3% | -0.3% [-3.4%, +2.8%] | -1,049 [-7,932, +6,141] | 0.14 | 12,169 |
| 1,501–3,000 | 120 | 20 | 9.5% | +3.4% [+0.9%, +5.9%] | +10,123 [+3,957, +16,492] | 0.14 | 11,861 |
| over 3,000 | 275 | 19 | 15.8% | +1.2% [-0.6%, +2.7%] | +3,330 [+38, +6,146] | 0.14 | 10,359 |

**By moves since the first 16384, successful trajectories**

| Group | Points | Trajectories | Greedy makes 32768 | Δ makes 32768 [95%] | Δ score added [95%] | CPU s per continuation | Chance nodes per continuation |
|---|---:|---:|---:|---:|---:|---:|---:|
| at the first 16384 | 20 | 20 | 11.2% | -1.9% [-6.9%, +3.1%] | -997 [-11,554, +9,858] | 0.14 | 15,304 |
| 1–500 moves after | 60 | 20 | 11.7% | -1.0% [-5.4%, +3.3%] | -4,667 [-13,840, +4,648] | 0.14 | 13,156 |
| 501–1,500 | 80 | 20 | 10.9% | +1.2% [-2.3%, +4.8%] | +5,384 [-2,349, +12,870] | 0.14 | 12,148 |
| 1,501–3,000 | 120 | 20 | 12.3% | -0.2% [-3.6%, +3.1%] | +738 [-6,303, +7,143] | 0.14 | 11,919 |
| over 3,000 | 340 | 20 | 19.2% | -1.7% [-3.2%, -0.4%] | -1,509 [-4,432, +1,290] | 0.14 | 10,693 |

**By moves remaining, failed trajectories**

| Group | Points | Trajectories | Greedy makes 32768 | Δ makes 32768 [95%] | Δ score added [95%] | CPU s per continuation | Chance nodes per continuation |
|---|---:|---:|---:|---:|---:|---:|---:|
| over 2,000 moves left | 356 | 20 | 12.3% | +1.5% [+0.1%, +2.9%] | +4,243 [+680, +7,635] | 0.14 | 12,203 |
| 501–2,000 left | 120 | 20 | 14.6% | +0.9% [-1.6%, +3.2%] | +2,989 [-1,395, +7,114] | 0.14 | 11,137 |
| 201–500 left | 25 | 20 | 16.5% | -1.0% [-9.9%, +7.1%] | +2,271 [-14,494, +17,605] | 0.14 | 9,822 |
| 200 or fewer left | 54 | 20 | 19.7% | +0.5% [-4.3%, +5.1%] | +1,571 [-7,491, +10,335] | 0.14 | 7,799 |

**By empty cells, failed trajectories**

| Group | Points | Trajectories | Greedy makes 32768 | Δ makes 32768 [95%] | Δ score added [95%] | CPU s per continuation | Chance nodes per continuation |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0–1 empty | 109 | 20 | 14.0% | +0.9% [-3.0%, +4.5%] | +4,341 [-3,262, +11,473] | 0.14 | 10,832 |
| 2–3 empty | 293 | 20 | 14.0% | +1.9% [-0.1%, +3.8%] | +4,828 [+401, +8,992] | 0.14 | 11,236 |
| 4–6 empty | 133 | 20 | 13.3% | -0.2% [-1.5%, +1.2%] | +1,198 [-2,165, +4,458] | 0.14 | 11,930 |
| 7+ empty | 20 | 15 | 9.4% | +0.6% [-4.4%, +5.1%] | -1,807 [-17,743, +12,469] | 0.14 | 14,394 |

At the first 16384, weighting successes and failures by greedy's natural 11.2% conversion, C changes conversion by -1.9%.

## D: pruned depth 5 (cutoff 0.01) for 200 moves, then greedy


**By moves since the first 16384, failed trajectories**

| Group | Points | Trajectories | Greedy makes 32768 | Δ makes 32768 [95%] | Δ score added [95%] | CPU s per continuation | Chance nodes per continuation |
|---|---:|---:|---:|---:|---:|---:|---:|
| at the first 16384 | 20 | 20 | 13.1% | -1.2% [-8.8%, +6.2%] | +990 [-18,521, +20,813] | 1.56 | 150,086 |
| 1–500 moves after | 60 | 20 | 12.9% | +0.8% [-2.5%, +4.4%] | +2,328 [-5,871, +10,968] | 1.56 | 178,150 |
| 501–1,500 | 80 | 20 | 13.3% | +0.2% [-2.5%, +2.8%] | +1,867 [-3,822, +8,235] | 1.56 | 187,027 |
| 1,501–3,000 | 120 | 20 | 9.5% | +1.7% [-0.6%, +4.1%] | +7,558 [+1,241, +13,387] | 1.56 | 189,298 |
| over 3,000 | 275 | 19 | 15.8% | +2.6% [+0.3%, +4.8%] | +5,843 [+1,129, +10,181] | 1.54 | 183,228 |

**By moves since the first 16384, successful trajectories**

| Group | Points | Trajectories | Greedy makes 32768 | Δ makes 32768 [95%] | Δ score added [95%] | CPU s per continuation | Chance nodes per continuation |
|---|---:|---:|---:|---:|---:|---:|---:|
| at the first 16384 | 20 | 20 | 11.2% | +1.9% [-3.1%, +6.9%] | +5,020 [-7,333, +17,893] | 1.55 | 150,399 |
| 1–500 moves after | 60 | 20 | 11.7% | +0.4% [-4.4%, +5.4%] | +683 [-9,696, +11,571] | 1.55 | 178,405 |
| 501–1,500 | 80 | 20 | 10.9% | -1.2% [-4.2%, +1.9%] | +482 [-7,238, +8,152] | 1.55 | 188,471 |
| 1,501–3,000 | 120 | 20 | 12.3% | +0.6% [-2.8%, +4.5%] | +1,642 [-5,558, +8,705] | 1.55 | 189,773 |
| over 3,000 | 340 | 20 | 19.2% | -0.1% [-1.9%, +1.4%] | +1,316 [-2,293, +4,528] | 1.55 | 188,350 |

**By moves remaining, failed trajectories**

| Group | Points | Trajectories | Greedy makes 32768 | Δ makes 32768 [95%] | Δ score added [95%] | CPU s per continuation | Chance nodes per continuation |
|---|---:|---:|---:|---:|---:|---:|---:|
| over 2,000 moves left | 356 | 20 | 12.3% | +2.2% [+0.4%, +3.9%] | +5,716 [+1,004, +9,931] | 1.55 | 186,455 |
| 501–2,000 left | 120 | 20 | 14.6% | +0.9% [-2.3%, +4.3%] | +3,238 [-2,472, +9,343] | 1.56 | 191,415 |
| 201–500 left | 25 | 20 | 16.5% | -5.0% [-12.0%, +1.6%] | -3,048 [-16,313, +9,772] | 1.57 | 181,633 |
| 200 or fewer left | 54 | 20 | 19.7% | +3.5% [-2.6%, +9.2%] | +8,803 [-2,736, +18,762] | 1.56 | 145,695 |

**By empty cells, failed trajectories**

| Group | Points | Trajectories | Greedy makes 32768 | Δ makes 32768 [95%] | Δ score added [95%] | CPU s per continuation | Chance nodes per continuation |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0–1 empty | 109 | 20 | 14.0% | +3.3% [-0.6%, +7.6%] | +10,606 [+2,135, +19,146] | 1.55 | 174,261 |
| 2–3 empty | 293 | 20 | 14.0% | +2.0% [-0.1%, +3.9%] | +4,703 [+60, +9,069] | 1.55 | 186,568 |
| 4–6 empty | 133 | 20 | 13.3% | -0.3% [-3.3%, +2.6%] | +1,677 [-2,617, +5,617] | 1.56 | 186,627 |
| 7+ empty | 20 | 15 | 9.4% | +1.9% [-3.8%, +8.1%] | +3,275 [-14,636, +22,946] | 1.56 | 163,793 |

At the first 16384, weighting successes and failures by greedy's natural 11.2% conversion, D changes conversion by -0.9%.

## D against C: deeper search over the same 200 moves

| Group (failed trajectories) | Points | Δ makes 32768, D − C [95%] | Δ score, D − C [95%] |
|---|---:|---:|---:|
| at the first 16384 | 20 | +0.6% [-5.6%, +7.5%] | +9,032 [-4,972, +25,540] |
| 1–500 moves after | 60 | +1.2% [-1.0%, +3.3%] | +242 [-6,467, +6,269] |
| 501–1,500 | 80 | +0.5% [-2.7%, +3.9%] | +2,915 [-2,975, +9,177] |
| 1,501–3,000 | 120 | -1.8% [-4.2%, +0.7%] | -2,565 [-7,787, +2,491] |
| over 3,000 | 275 | +1.5% [-0.6%, +3.3%] | +2,513 [-1,809, +6,296] |

### p2-plan.json

```json
{
  "groups": [
    [
      "failure",
      0
    ],
    [
      "failure",
      1
    ],
    [
      "failure",
      2
    ],
    [
      "failure",
      3
    ],
    [
      "failure",
      4
    ],
    [
      "success",
      0
    ],
    [
      "success",
      1
    ],
    [
      "success",
      2
    ],
    [
      "success",
      3
    ],
    [
      "success",
      4
    ]
  ],
  "policies": [
    "A",
    "B",
    "C",
    "D"
  ],
  "continuations": 32,
  "estimated hours": 5.22,
  "budget hours": 10.0,
  "reasons": [
    "failure, at the first 16384: B \u0394 score +26,773 [+7,984, +45,324], \u0394 conversion +4.4% [-3.8%, +12.5%]",
    "failure, 1\u2013500 moves after: B \u0394 score +37,111 [+28,755, +45,205], \u0394 conversion +7.3% [+2.9%, +11.9%]",
    "failure, 501\u20131,500: B \u0394 score +32,050 [+21,822, +41,840], \u0394 conversion +5.6% [+1.4%, +9.4%]",
    "failure, 1,501\u20133,000: B \u0394 score +40,867 [+32,890, +48,473], \u0394 conversion +8.4% [+5.4%, +11.1%]",
    "failure, over 3,000: B \u0394 score +21,879 [+16,965, +27,948], \u0394 conversion +6.0% [+3.7%, +8.5%]",
    "success, at the first 16384: B \u0394 score +39,477 [+22,799, +57,522], \u0394 conversion +6.9% [-0.6%, +14.4%]",
    "success, 1\u2013500 moves after: B \u0394 score +45,596 [+34,369, +56,472], \u0394 conversion +10.6% [+6.0%, +15.4%]",
    "success, 501\u20131,500: B \u0394 score +43,780 [+31,494, +55,885], \u0394 conversion +9.4% [+4.2%, +14.7%]",
    "success, 1,501\u20133,000: B \u0394 score +33,157 [+26,977, +41,027], \u0394 conversion +6.6% [+3.9%, +10.0%]",
    "success, over 3,000: B \u0394 score +19,901 [+15,316, +24,341], \u0394 conversion +5.0% [+2.9%, +6.9%]",
    "D joins: its edge over C could reach +5,000 in ['failure', 0]"
  ]
}

```

### part2-final.md

# Stage 7, Part 2: when can failing games be rescued? (final)

Generated 2026-11-14 21:52.

Policies: **A** greedy; **B** depth 3 from here; **C** depth 3 for 200 moves, then greedy; **D** pruned depth 5 (cutoff 0.01) for 200 moves, then greedy.

Each cell is the mean paired difference from greedy (A) over points, then continuations, with a 95% interval from resampling whole trajectories. Moves remaining are retrospective (the agent can't know them).

Trajectories: 40 that made 32768 (sampled up to it) and 40 that didn't. Points: 2,345. Continuations: 300,160.


## B: depth 3 from here


**By moves since the first 16384, failed trajectories**

| Group | Points | Trajectories | Greedy makes 32768 | Δ makes 32768 [95%] | Δ score added [95%] | CPU s per continuation | Chance nodes per continuation |
|---|---:|---:|---:|---:|---:|---:|---:|
| at the first 16384 | 40 | 40 | 12.0% | +7.6% [+5.0%, +10.1%] | +38,787 [+32,800, +44,828] | 3.04 | 456,443 |
| 1–500 moves after | 120 | 40 | 11.3% | +8.6% [+7.0%, +10.3%] | +41,509 [+38,158, +44,892] | 3.04 | 438,747 |
| 501–1,500 | 160 | 40 | 11.6% | +7.1% [+5.6%, +8.5%] | +38,640 [+34,339, +42,850] | 3.04 | 380,801 |
| 1,501–3,000 | 241 | 40 | 11.4% | +7.9% [+6.9%, +8.8%] | +38,290 [+35,470, +41,204] | 3.05 | 306,757 |
| over 3,000 | 544 | 38 | 15.4% | +4.7% [+3.8%, +5.6%] | +18,909 [+16,829, +21,290] | 2.87 | 166,483 |

**By moves since the first 16384, successful trajectories**

| Group | Points | Trajectories | Greedy makes 32768 | Δ makes 32768 [95%] | Δ score added [95%] | CPU s per continuation | Chance nodes per continuation |
|---|---:|---:|---:|---:|---:|---:|---:|
| at the first 16384 | 40 | 40 | 10.4% | +9.4% [+6.7%, +11.9%] | +46,525 [+40,411, +52,438] | 2.85 | 462,844 |
| 1–500 moves after | 120 | 40 | 12.1% | +6.8% [+5.2%, +8.5%] | +37,459 [+33,994, +41,082] | 2.85 | 436,523 |
| 501–1,500 | 160 | 40 | 11.3% | +7.2% [+5.8%, +8.7%] | +38,006 [+35,189, +40,878] | 2.85 | 380,561 |
| 1,501–3,000 | 240 | 40 | 12.1% | +6.8% [+5.7%, +8.0%] | +33,923 [+31,254, +36,633] | 2.85 | 308,394 |
| over 3,000 | 680 | 40 | 18.7% | +3.6% [+3.0%, +4.2%] | +17,038 [+15,785, +18,348] | 2.85 | 170,858 |

**By moves remaining, failed trajectories**

| Group | Points | Trajectories | Greedy makes 32768 | Δ makes 32768 [95%] | Δ score added [95%] | CPU s per continuation | Chance nodes per continuation |
|---|---:|---:|---:|---:|---:|---:|---:|
| over 2,000 moves left | 706 | 40 | 12.4% | +6.9% [+6.1%, +7.6%] | +33,505 [+31,355, +35,838] | 2.91 | 328,521 |
| 501–2,000 left | 240 | 40 | 14.5% | +6.2% [+4.8%, +7.8%] | +24,110 [+19,833, +28,788] | 3.04 | 187,975 |
| 201–500 left | 52 | 40 | 15.9% | +4.6% [+1.0%, +8.0%] | +20,341 [+12,046, +28,190] | 3.10 | 139,545 |
| 200 or fewer left | 107 | 40 | 16.3% | +3.3% [+1.1%, +5.4%] | +16,170 [+10,593, +21,567] | 3.02 | 112,378 |

**By empty cells, failed trajectories**

| Group | Points | Trajectories | Greedy makes 32768 | Δ makes 32768 [95%] | Δ score added [95%] | CPU s per continuation | Chance nodes per continuation |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0–1 empty | 230 | 40 | 13.5% | +6.4% [+5.3%, +7.7%] | +28,581 [+25,976, +31,651] | 2.98 | 259,373 |
| 2–3 empty | 566 | 40 | 14.2% | +6.0% [+4.9%, +7.0%] | +27,968 [+25,229, +31,216] | 2.93 | 251,896 |
| 4–6 empty | 269 | 40 | 11.9% | +6.6% [+5.7%, +7.5%] | +31,230 [+28,729, +33,935] | 3.00 | 288,410 |
| 7+ empty | 40 | 31 | 11.7% | +7.3% [+5.4%, +9.4%] | +35,622 [+30,491, +41,135] | 2.98 | 412,994 |

At the first 16384, weighting successes and failures by greedy's natural 11.2% conversion, B changes conversion by +7.8%.

## C: depth 3 for 200 moves, then greedy


**By moves since the first 16384, failed trajectories**

| Group | Points | Trajectories | Greedy makes 32768 | Δ makes 32768 [95%] | Δ score added [95%] | CPU s per continuation | Chance nodes per continuation |
|---|---:|---:|---:|---:|---:|---:|---:|
| at the first 16384 | 40 | 40 | 12.0% | -2.0% [-4.5%, +0.2%] | -2,469 [-8,999, +3,720] | 0.14 | 15,354 |
| 1–500 moves after | 120 | 40 | 11.3% | +0.2% [-1.0%, +1.4%] | -514 [-3,789, +3,058] | 0.14 | 13,278 |
| 501–1,500 | 160 | 40 | 11.6% | -0.8% [-1.8%, +0.2%] | -657 [-3,614, +2,272] | 0.14 | 12,151 |
| 1,501–3,000 | 241 | 40 | 11.4% | +0.7% [-0.1%, +1.5%] | +4,057 [+2,085, +6,062] | 0.14 | 11,847 |
| over 3,000 | 544 | 38 | 15.4% | +0.6% [+0.1%, +1.2%] | +1,994 [+835, +3,228] | 0.14 | 10,476 |

**By moves since the first 16384, successful trajectories**

| Group | Points | Trajectories | Greedy makes 32768 | Δ makes 32768 [95%] | Δ score added [95%] | CPU s per continuation | Chance nodes per continuation |
|---|---:|---:|---:|---:|---:|---:|---:|
| at the first 16384 | 40 | 40 | 10.4% | -0.1% [-2.2%, +2.0%] | +322 [-4,497, +5,099] | 0.14 | 15,254 |
| 1–500 moves after | 120 | 40 | 12.1% | -0.9% [-2.0%, +0.2%] | -2,894 [-5,264, -416] | 0.14 | 13,237 |
| 501–1,500 | 160 | 40 | 11.3% | +0.9% [-0.3%, +2.2%] | +1,673 [-1,263, +4,660] | 0.14 | 12,166 |
| 1,501–3,000 | 240 | 40 | 12.1% | +0.2% [-0.7%, +1.2%] | +1,921 [-320, +4,200] | 0.14 | 11,910 |
| over 3,000 | 680 | 40 | 18.7% | +0.4% [-0.2%, +1.0%] | +1,388 [+247, +2,597] | 0.14 | 10,692 |

**By moves remaining, failed trajectories**

| Group | Points | Trajectories | Greedy makes 32768 | Δ makes 32768 [95%] | Δ score added [95%] | CPU s per continuation | Chance nodes per continuation |
|---|---:|---:|---:|---:|---:|---:|---:|
| over 2,000 moves left | 706 | 40 | 12.4% | +0.1% [-0.4%, +0.5%] | +1,176 [-74, +2,322] | 0.14 | 12,210 |
| 501–2,000 left | 240 | 40 | 14.5% | +1.0% [+0.1%, +1.9%] | +2,766 [+917, +4,631] | 0.14 | 11,146 |
| 201–500 left | 52 | 40 | 15.9% | +1.0% [-2.3%, +4.1%] | +3,595 [-2,980, +9,778] | 0.14 | 9,941 |
| 200 or fewer left | 107 | 40 | 16.3% | +0.0% [-1.5%, +1.5%] | +1,083 [-2,104, +4,124] | 0.14 | 8,352 |

**By empty cells, failed trajectories**

| Group | Points | Trajectories | Greedy makes 32768 | Δ makes 32768 [95%] | Δ score added [95%] | CPU s per continuation | Chance nodes per continuation |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0–1 empty | 230 | 40 | 13.5% | -0.2% [-1.0%, +0.7%] | +410 [-1,519, +2,349] | 0.14 | 11,068 |
| 2–3 empty | 566 | 40 | 14.2% | +0.4% [-0.3%, +1.0%] | +1,921 [+450, +3,333] | 0.14 | 11,237 |
| 4–6 empty | 269 | 40 | 11.9% | +0.8% [-0.0%, +1.6%] | +2,480 [+442, +4,471] | 0.14 | 11,975 |
| 7+ empty | 40 | 31 | 11.7% | -1.2% [-3.5%, +1.3%] | -1,300 [-7,447, +5,848] | 0.14 | 14,467 |

At the first 16384, weighting successes and failures by greedy's natural 11.2% conversion, C changes conversion by -1.8%.

## D: pruned depth 5 (cutoff 0.01) for 200 moves, then greedy


**By moves since the first 16384, failed trajectories**

| Group | Points | Trajectories | Greedy makes 32768 | Δ makes 32768 [95%] | Δ score added [95%] | CPU s per continuation | Chance nodes per continuation |
|---|---:|---:|---:|---:|---:|---:|---:|
| at the first 16384 | 40 | 40 | 12.0% | -1.8% [-4.5%, +0.9%] | -3,988 [-10,191, +2,293] | 1.59 | 148,724 |
| 1–500 moves after | 120 | 40 | 11.3% | +0.2% [-1.1%, +1.6%] | +626 [-2,699, +3,990] | 1.59 | 177,951 |
| 501–1,500 | 160 | 40 | 11.6% | +0.4% [-0.8%, +1.5%] | +1,623 [-1,274, +4,453] | 1.59 | 187,356 |
| 1,501–3,000 | 241 | 40 | 11.4% | +0.6% [-0.4%, +1.6%] | +3,606 [+1,355, +5,856] | 1.59 | 189,720 |
| over 3,000 | 544 | 38 | 15.4% | +1.1% [+0.3%, +1.9%] | +2,609 [+967, +4,174] | 1.57 | 184,552 |

**By moves since the first 16384, successful trajectories**

| Group | Points | Trajectories | Greedy makes 32768 | Δ makes 32768 [95%] | Δ score added [95%] | CPU s per continuation | Chance nodes per continuation |
|---|---:|---:|---:|---:|---:|---:|---:|
| at the first 16384 | 40 | 40 | 10.4% | +1.8% [-0.5%, +4.2%] | +5,693 [-405, +11,820] | 1.57 | 150,525 |
| 1–500 moves after | 120 | 40 | 12.1% | +0.0% [-1.3%, +1.3%] | -979 [-4,616, +2,540] | 1.57 | 177,669 |
| 501–1,500 | 160 | 40 | 11.3% | +0.2% [-1.0%, +1.5%] | +107 [-3,215, +3,371] | 1.57 | 187,237 |
| 1,501–3,000 | 240 | 40 | 12.1% | -0.7% [-1.6%, +0.2%] | +61 [-2,244, +2,385] | 1.57 | 189,886 |
| over 3,000 | 680 | 40 | 18.7% | +0.6% [-0.1%, +1.3%] | +1,879 [+653, +3,107] | 1.57 | 188,412 |

**By moves remaining, failed trajectories**

| Group | Points | Trajectories | Greedy makes 32768 | Δ makes 32768 [95%] | Δ score added [95%] | CPU s per continuation | Chance nodes per continuation |
|---|---:|---:|---:|---:|---:|---:|---:|
| over 2,000 moves left | 706 | 40 | 12.4% | +0.1% [-0.5%, +0.6%] | +1,063 [-362, +2,358] | 1.57 | 186,101 |
| 501–2,000 left | 240 | 40 | 14.5% | +0.9% [-0.2%, +2.0%] | +2,543 [+347, +4,740] | 1.59 | 191,290 |
| 201–500 left | 52 | 40 | 15.9% | +0.2% [-2.6%, +3.1%] | +2,510 [-3,297, +8,124] | 1.59 | 184,343 |
| 200 or fewer left | 107 | 40 | 16.3% | +4.3% [+1.9%, +6.7%] | +9,087 [+4,812, +13,376] | 1.59 | 154,351 |

**By empty cells, failed trajectories**

| Group | Points | Trajectories | Greedy makes 32768 | Δ makes 32768 [95%] | Δ score added [95%] | CPU s per continuation | Chance nodes per continuation |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0–1 empty | 230 | 40 | 13.5% | +0.4% [-0.9%, +1.7%] | +2,207 [-306, +4,856] | 1.58 | 177,569 |
| 2–3 empty | 566 | 40 | 14.2% | +0.9% [+0.3%, +1.5%] | +2,381 [+975, +3,804] | 1.57 | 187,241 |
| 4–6 empty | 269 | 40 | 11.9% | +0.7% [-0.2%, +1.6%] | +2,545 [+777, +4,391] | 1.58 | 186,488 |
| 7+ empty | 40 | 31 | 11.7% | -1.4% [-4.1%, +1.5%] | -1,916 [-7,886, +5,288] | 1.58 | 160,352 |

At the first 16384, weighting successes and failures by greedy's natural 11.2% conversion, D changes conversion by -1.4%.

## D against C: deeper search over the same 200 moves

| Group (failed trajectories) | Points | Δ makes 32768, D − C [95%] | Δ score, D − C [95%] |
|---|---:|---:|---:|
| at the first 16384 | 40 | +0.2% [-2.0%, +2.5%] | -1,519 [-6,839, +4,034] |
| 1–500 moves after | 120 | +0.0% [-1.3%, +1.3%] | +1,140 [-1,735, +4,105] |
| 501–1,500 | 160 | +1.2% [-0.0%, +2.4%] | +2,281 [-1,005, +5,540] |
| 1,501–3,000 | 241 | -0.1% [-1.1%, +0.9%] | -451 [-2,649, +1,856] |
| over 3,000 | 544 | +0.4% [-0.4%, +1.2%] | +615 [-984, +2,181] |
