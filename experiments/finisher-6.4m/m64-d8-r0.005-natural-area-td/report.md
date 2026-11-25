# Finisher m64-d8-r0.005-natural-area-td

Window: the last 64 of tile sum before a 32768. Teacher: depth 8, cutoff 0.02. Dev set: 1,083 window boards and 500 well-placed arrival positions, 4 continuations each. Differences are paired with the base, ± 95%.

The base alone, from the window: greedy play makes the 32768 in 50.4%, depth 3 in 43.8%, the teacher's search in 68.1%. From arrival, greedy play makes it in 29.8%.

| Attempts | Window, greedy | vs base greedy | Share of the teacher's gain | Window, depth 3 | vs base depth 3 | Arrival, greedy | vs base | Picks the teacher's move | Regret | Value error | Ranking error | Patch weights | In training: teacher / finisher |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 50.4% | | | 43.8% | | 29.8% | | 71.0% | 2,718 | 33,925 | 27,005 | 0 | |
| 500,000 | 41.2% | -9.2% ± 1.9% | -52% | 39.3% | -4.5% ± 1.9% | 24.1% | -5.7% ± 2.1% | 70.7% | 2,552 | 34,933 | 25,940 | 154,270 | 0.0% / 43.2% |
| 1,000,000 | 41.0% | -9.4% ± 2.0% | -53% | 41.0% | -2.8% ± 1.9% | 24.9% | -4.9% ± 2.2% | 71.0% | 2,670 | 36,051 | 25,974 | 194,731 | 0.0% / 41.0% |
| 1,500,000 | 40.1% | -10.3% ± 1.9% | -58% | 39.7% | -4.1% ± 2.0% | 23.8% | -6.0% ± 2.1% | 70.7% | 2,790 | 36,707 | 25,958 | 226,973 | 0.0% / 40.6% |
| 2,000,000 | 40.1% | -10.4% ± 2.0% | -58% | 40.3% | -3.5% ± 2.0% | 25.1% | -4.8% ± 2.2% | 70.6% | 2,845 | 37,106 | 25,955 | 253,163 | 0.0% / 40.2% |

Picks the teacher's move: how often the finisher's one-move choice is the teacher's, along the teacher's play from dev boards. Regret: how much more the teacher values its own move than the finisher's, on average. Value error: the mean difference between the two's values of a move. Ranking error: the same once each choice's mean difference is taken off.
