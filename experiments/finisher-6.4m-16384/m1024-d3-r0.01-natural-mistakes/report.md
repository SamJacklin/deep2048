# Finisher m1024-d3-r0.01-natural-mistakes

Window: the last 1024 of tile sum before a 32768. Teacher: depth 3, cutoff 0. Dev set: 1,885 window boards and 500 well-placed arrival positions, 4 continuations each. Differences are paired with the base, ± 95%.

The base alone, from the window: greedy play makes the 32768 in 88.6%, depth 3 in 94.4%, the teacher's search in 94.4%. From arrival, greedy play makes it in 83.5%.

| Attempts | Window, greedy | vs base greedy | Share of the teacher's gain | Window, depth 3 | vs base depth 3 | Arrival, greedy | vs base | Picks the teacher's move | Regret | Value error | Ranking error | Patch weights | In training: teacher / finisher |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 88.6% | | | 94.4% | | 83.5% | | 72.8% | 107 | 9,006 | 10,386 | 0 | |
| 5,000 | 84.6% | -4.1% ± 1.0% | -72% | 88.2% | -6.2% ± 0.9% | 80.8% | -2.8% ± 1.8% | 75.6% | 96 | 10,226 | 8,506 | 1,113,776 | 94.2% / 83.0% |
| 10,000 | 85.4% | -3.2% ± 1.0% | -56% | 89.4% | -5.0% ± 0.8% | 79.5% | -4.0% ± 1.9% | 76.9% | 87 | 10,364 | 8,367 | 1,525,026 | 94.9% / 86.2% |
| 15,000 | 86.9% | -1.7% ± 1.0% | -30% | 89.0% | -5.3% ± 0.8% | 81.8% | -1.7% ± 1.8% | 78.1% | 83 | 10,494 | 8,295 | 1,919,199 | 95.0% / 85.8% |
| 20,000 | 87.1% | -1.5% ± 1.0% | -27% | 89.4% | -4.9% ± 0.8% | 82.6% | -0.9% ± 1.8% | 78.8% | 78 | 10,606 | 8,293 | 2,248,139 | 94.0% / 87.0% |

Picks the teacher's move: how often the finisher's one-move choice is the teacher's, along the teacher's play from dev boards. Regret: how much more the teacher values its own move than the finisher's, on average. Value error: the mean difference between the two's values of a move. Ranking error: the same once each choice's mean difference is taken off.
