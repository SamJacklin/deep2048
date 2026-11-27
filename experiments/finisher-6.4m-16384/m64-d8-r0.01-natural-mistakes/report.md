# Finisher m64-d8-r0.01-natural-mistakes

Window: the last 64 of tile sum before a 32768. Teacher: depth 8, cutoff 0.02. Dev set: 1,766 window boards and 500 well-placed arrival positions, 4 continuations each. Differences are paired with the base, ± 95%.

The base alone, from the window: greedy play makes the 32768 in 95.4%, depth 3 in 96.9%, the teacher's search in 97.4%. From arrival, greedy play makes it in 83.5%.

| Attempts | Window, greedy | vs base greedy | Share of the teacher's gain | Window, depth 3 | vs base depth 3 | Arrival, greedy | vs base | Picks the teacher's move | Regret | Value error | Ranking error | Patch weights | In training: teacher / finisher |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 95.4% | | | 96.9% | | 83.5% | | 80.8% | 676 | 27,966 | 23,968 | 0 | |
| 20,000 | 95.5% | +0.1% ± 0.6% | 7% | 95.0% | -1.9% ± 0.6% | 83.5% | -0.1% ± 1.1% | 90.6% | 194 | 34,807 | 16,762 | 479,634 | 97.4% / 95.1% |
| 40,000 | 96.0% | +0.6% ± 0.6% | 31% | 95.3% | -1.7% ± 0.6% | 85.0% | +1.5% ± 1.0% | 92.6% | 161 | 35,474 | 16,522 | 684,481 | 97.6% / 95.7% |

Picks the teacher's move: how often the finisher's one-move choice is the teacher's, along the teacher's play from dev boards. Regret: how much more the teacher values its own move than the finisher's, on average. Value error: the mean difference between the two's values of a move. Ranking error: the same once each choice's mean difference is taken off.
