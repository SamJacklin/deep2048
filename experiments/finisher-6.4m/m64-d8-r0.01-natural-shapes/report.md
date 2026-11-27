# Finisher m64-d8-r0.01-natural-shapes

Window: the last 64 of tile sum before a 32768. Teacher: depth 8, cutoff 0.02. Dev set: 1,083 window boards and 500 well-placed arrival positions, 4 continuations each. Differences are paired with the base, ± 95%.

The base alone, from the window: greedy play makes the 32768 in 50.4%, depth 3 in 43.8%, the teacher's search in 68.1%. From arrival, greedy play makes it in 29.8%.

| Attempts | Window, greedy | vs base greedy | Share of the teacher's gain | Window, depth 3 | vs base depth 3 | Arrival, greedy | vs base | Picks the teacher's move | Regret | Value error | Ranking error | Patch weights | In training: teacher / finisher |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 50.4% | | | 43.8% | | 29.8% | | 71.0% | 2,718 | 33,925 | 27,005 | 0 | |
| 20,000 | 56.9% | +6.5% ± 1.9% | 36% | 40.8% | -3.0% ± 1.9% | 34.5% | +4.7% ± 2.0% | 79.9% | 1,047 | 8,089 | 6,849 | 2,213,730 | 68.8% / 55.2% |
| 40,000 | 57.2% | +6.8% ± 1.9% | 39% | 44.2% | +0.3% ± 1.9% | 33.6% | +3.8% ± 2.0% | 80.8% | 954 | 7,572 | 6,441 | 2,750,337 | 69.1% / 57.3% |

Picks the teacher's move: how often the finisher's one-move choice is the teacher's, along the teacher's play from dev boards. Regret: how much more the teacher values its own move than the finisher's, on average. Value error: the mean difference between the two's values of a move. Ranking error: the same once each choice's mean difference is taken off.
