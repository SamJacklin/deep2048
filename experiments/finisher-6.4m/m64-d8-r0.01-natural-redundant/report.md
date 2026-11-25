# Finisher m64-d8-r0.01-natural-redundant

Window: the last 64 of tile sum before a 32768. Teacher: depth 8, cutoff 0.02. Dev set: 1,083 window boards and 500 well-placed arrival positions, 4 continuations each. Differences are paired with the base, ± 95%.

The base alone, from the window: greedy play makes the 32768 in 50.4%, depth 3 in 43.8%, the teacher's search in 68.1%. From arrival, greedy play makes it in 29.8%.

| Attempts | Window, greedy | vs base greedy | Share of the teacher's gain | Window, depth 3 | vs base depth 3 | Arrival, greedy | vs base | Picks the teacher's move | Regret | Value error | Ranking error | Patch weights | In training: teacher / finisher |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 50.4% | | | 43.8% | | 29.8% | | 71.0% | 2,718 | 33,925 | 27,005 | 0 | |
| 20,000 | 56.0% | +5.5% ± 1.9% | 31% | 40.0% | -3.8% ± 2.0% | 33.0% | +3.1% ± 2.0% | 79.3% | 1,041 | 8,204 | 6,902 | 1,538,571 | 68.2% / 54.4% |
| 40,000 | 57.1% | +6.7% ± 1.9% | 38% | 42.8% | -1.0% ± 2.0% | 34.2% | +4.5% ± 2.0% | 79.9% | 1,004 | 7,667 | 6,516 | 1,897,118 | 68.3% / 56.3% |

Picks the teacher's move: how often the finisher's one-move choice is the teacher's, along the teacher's play from dev boards. Regret: how much more the teacher values its own move than the finisher's, on average. Value error: the mean difference between the two's values of a move. Ranking error: the same once each choice's mean difference is taken off.
