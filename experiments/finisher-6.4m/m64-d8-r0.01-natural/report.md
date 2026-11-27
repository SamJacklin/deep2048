# Finisher m64-d8-r0.01-natural

Window: the last 64 of tile sum before a 32768. Teacher: depth 8, cutoff 0.02. Dev set: 1,083 window boards and 500 well-placed arrival positions, 4 continuations each. Differences are paired with the base, ± 95%.

The base alone, from the window: greedy play makes the 32768 in 50.4%, depth 3 in 43.8%, the teacher's search in 68.1%. From arrival, greedy play makes it in 29.8%.

| Attempts | Window, greedy | vs base greedy | Share of the teacher's gain | Window, depth 3 | vs base depth 3 | Arrival, greedy | vs base | Picks the teacher's move | Regret | Value error | Ranking error | Patch weights | In training: teacher / finisher |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 50.4% | | | 43.8% | | 29.8% | | 71.0% | 2,718 | 33,925 | 27,005 | 0 | |
| 20,000 | 55.8% | +5.4% ± 1.9% | 31% | 42.2% | -1.7% ± 1.9% | 33.7% | +3.9% ± 2.0% | 79.3% | 1,055 | 8,284 | 6,979 | 1,467,607 | 68.9% / 54.2% |
| 40,000 | 55.7% | +5.3% ± 1.9% | 30% | 44.4% | +0.6% ± 2.0% | 33.6% | +3.8% ± 2.0% | 80.4% | 956 | 7,744 | 6,556 | 1,862,383 | 68.8% / 56.4% |
| 60,000 | 57.2% | +6.7% ± 1.9% | 38% | 40.9% | -2.9% ± 2.0% | 34.5% | +4.7% ± 2.1% | 80.9% | 950 | 7,439 | 6,334 | 2,104,313 | 68.8% / 56.8% |
| 80,000 | 57.4% | +7.0% ± 1.9% | 39% | 41.1% | -2.7% ± 2.0% | 34.1% | +4.2% ± 2.1% | 81.9% | 906 | 7,278 | 6,218 | 2,296,422 | 68.5% / 57.4% |
| 100,000 | 57.7% | +7.3% ± 1.9% | 41% | 41.9% | -1.9% ± 2.0% | 34.2% | +4.5% ± 2.1% | 82.3% | 889 | 7,179 | 6,142 | 2,440,443 | 67.6% / 58.1% |

Picks the teacher's move: how often the finisher's one-move choice is the teacher's, along the teacher's play from dev boards. Regret: how much more the teacher values its own move than the finisher's, on average. Value error: the mean difference between the two's values of a move. Ranking error: the same once each choice's mean difference is taken off.
