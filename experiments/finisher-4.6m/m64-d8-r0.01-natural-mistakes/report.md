# Finisher m64-d8-r0.01-natural-mistakes

Window: the last 64 of tile sum before a 32768. Teacher: depth 8, cutoff 0.02. Dev set: 1,040 window boards and 500 well-placed arrival positions, 4 continuations each. Differences are paired with the base, ± 95%.

The base alone, from the window: greedy play makes the 32768 in 49.7%, depth 3 in 46.4%, the teacher's search in 68.3%. From arrival, greedy play makes it in 29.2%.

| Attempts | Window, greedy | vs base greedy | Share of the teacher's gain | Window, depth 3 | vs base depth 3 | Arrival, greedy | vs base | Picks the teacher's move | Regret | Value error | Ranking error | Patch weights | In training: teacher / finisher |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 49.7% | | | 46.4% | | 29.2% | | 74.4% | 2,777 | 34,275 | 27,007 | 0 | |
| 20,000 | 57.5% | +7.8% ± 1.9% | 42% | 54.9% | +8.4% ± 1.9% | 33.1% | +3.9% ± 2.2% | 86.9% | 754 | 45,473 | 22,523 | 530,063 | 69.1% / 57.5% |
| 40,000 | 56.0% | +6.3% ± 2.0% | 34% | 56.7% | +10.2% ± 2.0% | 32.5% | +3.2% ± 2.1% | 88.5% | 677 | 46,589 | 22,317 | 744,696 | 69.1% / 58.5% |

Picks the teacher's move: how often the finisher's one-move choice is the teacher's, along the teacher's play from dev boards. Regret: how much more the teacher values its own move than the finisher's, on average. Value error: the mean difference between the two's values of a move. Ranking error: the same once each choice's mean difference is taken off.
