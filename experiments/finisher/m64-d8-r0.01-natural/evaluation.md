# Finisher m64-d8-r0.01-natural: evaluation at 200,000 attempts

Whole games on seeds 10,000 to 10,999, the base and the finisher on the same seeds. Differences are paired game by game, ± 95%.

| Play | Network | Mean score | vs base | ≥16384 | ≥32768 | vs base |
|---|---|---:|---:|---:|---:|---:|
| Greedy | Base | 316,674 | | 76.5% | 7.8% | |
| Greedy | Finisher | 323,930 | +7,256 ± 3,201 | 76.5% | 12.4% | +4.6% ± 1.8% |

The held-out test set, which no setting was chosen on: 1,020 window boards and 500 well-placed arrival positions, 4 continuations each.

| From | Base greedy | Finisher greedy | Difference | Base's deep search |
|---|---:|---:|---:|---:|
| The window's edge | 42.0% | 58.2% | +16.1% ± 2.0% | 69.2% |
| Arrival | 23.2% | 31.9% | +8.7% ± 1.9% | |
