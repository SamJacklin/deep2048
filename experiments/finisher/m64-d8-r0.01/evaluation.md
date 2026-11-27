# Finisher m64-d8-r0.01: evaluation at 200,000 attempts

Whole games on seeds 10,000 to 10,999, the base and the finisher on the same seeds. Differences are paired game by game, ± 95%.

| Play | Network | Mean score | vs base | ≥16384 | ≥32768 | vs base |
|---|---|---:|---:|---:|---:|---:|
| Greedy | Base | 316,674 | | 76.5% | 7.8% | |
| Greedy | Finisher | 323,881 | +7,207 ± 3,404 | 76.5% | 12.1% | +4.3% ± 1.9% |

The held-out test set, which no setting was chosen on: 1,020 window boards and 500 well-placed arrival positions, 4 continuations each.

| From | Base greedy | Finisher greedy | Difference | Base's deep search |
|---|---:|---:|---:|---:|
| The window's edge | 42.0% | 56.4% | +14.3% ± 2.0% | 69.2% |
| Arrival | 23.2% | 30.8% | +7.5% ± 1.9% | |
