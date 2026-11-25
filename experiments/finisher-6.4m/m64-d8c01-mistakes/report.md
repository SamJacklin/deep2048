# Finisher m64-d8c01-mistakes

Window: the last 64 of tile sum before a 32768. Teacher: depth 8, cutoff 0.01. Dev set: 1,083 window boards and 500 well-placed arrival positions, 4 continuations each. Differences are paired with the base, ± 95%.

The base alone, from the window: greedy play makes the 32768 in 50.4%, depth 3 in 43.8%, the teacher's search in 70.7%. From arrival, greedy play makes it in 29.8%.

| Attempts | Window, greedy | vs base greedy | Share of the teacher's gain | Window, depth 3 | vs base depth 3 | Arrival, greedy | vs base | Picks the teacher's move | Regret | Value error | Ranking error | Patch weights | In training: teacher / finisher |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 50.4% | | | 43.8% | | 29.8% | | 69.0% | 2,617 | 35,625 | 26,472 | 0 | |
| 20,000 | 56.6% | +6.1% ± 1.9% | 30% | 54.9% | +11.1% ± 2.0% | 34.2% | +4.3% ± 2.0% | 83.3% | 711 | 46,532 | 20,919 | 688,364 | 71.6% / 57.0% |
| 40,000 | 60.4% | +9.9% ± 1.8% | 49% | 53.4% | +9.6% ± 2.0% | 36.6% | +6.9% ± 2.0% | 85.6% | 542 | 47,982 | 20,949 | 923,793 | 70.8% / 58.3% |

Picks the teacher's move: how often the finisher's one-move choice is the teacher's, along the teacher's play from dev boards. Regret: how much more the teacher values its own move than the finisher's, on average. Value error: the mean difference between the two's values of a move. Ranking error: the same once each choice's mean difference is taken off.
