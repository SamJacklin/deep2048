# Finisher m64-d8-r0.01-natural-mistakes-area

Window: the last 64 of tile sum before a 32768. Teacher: depth 8, cutoff 0.02. Dev set: 1,070 window boards and 500 well-placed arrival positions, 4 continuations each. Differences are paired with the base, ± 95%.

The base alone, from the window: greedy play makes the 32768 in 47.6%, depth 3 in 43.8%, the teacher's search in 67.3%. From arrival, greedy play makes it in 29.0%.

| Attempts | Window, greedy | vs base greedy | Share of the teacher's gain | Window, depth 3 | vs base depth 3 | Arrival, greedy | vs base | Picks the teacher's move | Regret | Value error | Ranking error | Patch weights | In training: teacher / finisher |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 47.6% | | | 43.8% | | 29.0% | | 71.0% | 2,889 | 33,311 | 26,313 | 0 | |
| 10,000 | 56.9% | +9.3% ± 1.8% | 47% | 53.6% | +9.8% ± 1.9% | 34.4% | +5.4% ± 2.0% | 85.9% | 828 | 43,021 | 20,439 | 516,546 | 69.3% / 59.7% |
| 20,000 | 62.0% | +14.4% ± 1.8% | 73% | 51.0% | +7.2% ± 2.0% | 37.8% | +8.7% ± 1.9% | 87.6% | 622 | 44,160 | 20,143 | 692,973 | 68.6% / 62.2% |
| 30,000 | 62.1% | +14.5% ± 1.8% | 74% | 50.1% | +6.3% ± 1.9% | 38.7% | +9.7% ± 1.9% | 88.3% | 582 | 44,935 | 19,950 | 828,225 | 69.1% / 63.4% |
| 40,000 | 63.1% | +15.4% ± 1.8% | 79% | 50.0% | +6.2% ± 1.9% | 39.4% | +10.3% ± 2.0% | 88.8% | 497 | 45,506 | 19,894 | 911,001 | 69.1% / 64.1% |

Picks the teacher's move: how often the finisher's one-move choice is the teacher's, along the teacher's play from dev boards. Regret: how much more the teacher values its own move than the finisher's, on average. Value error: the mean difference between the two's values of a move. Ranking error: the same once each choice's mean difference is taken off.
