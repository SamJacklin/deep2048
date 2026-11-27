# Finisher m64-d8-r0.01-natural-mistakes

Window: the last 64 of tile sum before a 32768. Teacher: depth 8, cutoff 0.02. Dev set: 1,083 window boards and 500 well-placed arrival positions, 4 continuations each. Differences are paired with the base, ± 95%.

The base alone, from the window: greedy play makes the 32768 in 50.4%, depth 3 in 43.8%, the teacher's search in 68.1%. From arrival, greedy play makes it in 29.8%.

| Attempts | Window, greedy | vs base greedy | Share of the teacher's gain | Window, depth 3 | vs base depth 3 | Arrival, greedy | vs base | Picks the teacher's move | Regret | Value error | Ranking error | Patch weights | In training: teacher / finisher |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 50.4% | | | 43.8% | | 29.8% | | 71.0% | 2,718 | 33,925 | 27,005 | 0 | |
| 20,000 | 58.3% | +7.9% ± 1.8% | 44% | 52.5% | +8.7% ± 2.0% | 35.0% | +5.2% ± 2.1% | 86.0% | 622 | 44,381 | 21,067 | 664,837 | 68.6% / 55.2% |
| 40,000 | 60.2% | +9.8% ± 1.8% | 55% | 53.2% | +9.4% ± 2.0% | 36.8% | +7.0% ± 2.1% | 87.3% | 588 | 45,848 | 20,997 | 858,201 | 68.2% / 56.9% |
| 60,000 | 58.7% | +8.3% ± 2.0% | 47% | 53.0% | +9.2% ± 2.0% | 34.4% | +4.6% ± 2.1% | 87.8% | 556 | 46,575 | 21,009 | 996,215 | 68.1% / 58.0% |
| 80,000 | 59.4% | +9.0% ± 1.9% | 51% | 52.2% | +8.4% ± 2.0% | 35.4% | +5.5% ± 2.1% | 87.8% | 569 | 47,205 | 20,967 | 1,124,779 | 67.7% / 59.1% |
| 100,000 | 62.0% | +11.6% ± 1.8% | 65% | 52.4% | +8.5% ± 1.9% | 36.8% | +7.0% ± 2.1% | 88.6% | 488 | 47,677 | 20,949 | 1,211,899 | 69.3% / 59.0% |

Picks the teacher's move: how often the finisher's one-move choice is the teacher's, along the teacher's play from dev boards. Regret: how much more the teacher values its own move than the finisher's, on average. Value error: the mean difference between the two's values of a move. Ranking error: the same once each choice's mean difference is taken off.
