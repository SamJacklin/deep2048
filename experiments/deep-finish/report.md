# Deep finish

The last stretch is searched to depth 8, pruning spawn branches less likely than 0.02. Intervals are 95%.

## From well-placed arrival positions

| Policy | Positions | Makes 32768 | vs its ordinary player (paired) | Moves searched deep, per continuation | Seconds |
|---|---:|---:|---:|---:|---:|
| greedy | 1,000 | 23.9% | — | 0 | 9 |
| greedy, deep in the last 64 | 1,000 | 39.2% | +15.3% ± 1.5% | 23 | 84 |
| greedy, deep in the last 256 | 1,000 | 39.3% | +15.4% ± 1.7% | 79 | 378 |
| depth 3 | 300 | 27.6% | — | 0 | 76 |
| depth 3, deep in the last 64 | 300 | 36.8% | +9.2% ± 2.3% | 21 | 98 |

## Whole games (seeds 10,000-10,999)

| Policy | Average score | vs its ordinary player (paired) | ≥16384 | ≥32768 | vs its ordinary player | Moves searched deep, per game | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| greedy | 316,674 | — | 76.5% | 7.8% | — | 0.0 | 12 |
| greedy, deep in the last 64 | 328,495 | +11,821 ± 3,233 | 76.5% | 15.2% | +7.4% ± 1.8% | 8.6 | 15 |
| greedy, deep in the last 256 | 326,865 | +10,190 ± 3,633 | 76.5% | 13.8% | +6.0% ± 2.0% | 28.1 | 44 |

The ordinary players' games against stage 6's stored control (experiments/late-network/evaluations/control.npz): greedy: identical.

