# Finisher m64-d8-r0.01-natural: evaluation at 100,000 attempts

Whole games on seeds 10,000 to 10,999, the base and the finisher on the same seeds. Differences are paired game by game, ± 95%.

| Play | Network | Mean score | vs base | ≥16384 | ≥32768 | vs base |
|---|---|---:|---:|---:|---:|---:|
| Greedy | Base | 338,648 | | 80.0% | 13.8% | |
| Greedy | Finisher | 342,930 | +4,282 ± 3,708 | 80.0% | 16.5% | +2.7% ± 1.9% |

The held-out test set, which no setting was chosen on: 1,086 window boards and 500 well-placed arrival positions, 4 continuations each.

| From | Base greedy | Finisher greedy | Difference | Base's deep search |
|---|---:|---:|---:|---:|
| The window's edge | 48.8% | 57.1% | +8.3% ± 1.8% | 67.3% |
| Arrival | 30.3% | 34.8% | +4.5% ± 2.1% | |
