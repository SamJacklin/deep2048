# Finisher m64-d8-r0.01-natural: evaluation at 200,000 attempts

Whole games on seeds 10,000 to 10,999, the base and the finisher on the same seeds. Differences are paired game by game, ± 95%.

| Play | Network | Mean score | vs base | ≥16384 | ≥32768 | vs base |
|---|---|---:|---:|---:|---:|---:|
| Greedy | Base | 330,480 | | 78.9% | 13.0% | |
| Greedy | Finisher | 330,837 | +357 ± 3,230 | 78.9% | 12.7% | -0.3% ± 1.8% |

The held-out test set, which no setting was chosen on: 1,076 window boards and 500 well-placed arrival positions, 4 continuations each.

| From | Base greedy | Finisher greedy | Difference | Base's deep search |
|---|---:|---:|---:|---:|
| The window's edge | 48.0% | 55.8% | +7.8% ± 1.9% | 67.3% |
| Arrival | 30.4% | 35.0% | +4.6% ± 2.0% | |
