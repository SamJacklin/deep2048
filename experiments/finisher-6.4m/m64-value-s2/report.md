# Finisher m64-value-s2

Window: the last 64 of tile sum before a 32768. Teacher: depth 8, cutoff 0.02. Dev set: 1,083 window boards and 500 well-placed arrival positions, 4 continuations each. Differences are paired with the base, ± 95%.

The base alone, from the window: greedy play makes the 32768 in 50.4%, depth 3 in 43.8%, the teacher's search in 68.1%. From arrival, greedy play makes it in 29.8%.

| Attempts | Window, greedy | vs base greedy | Share of the teacher's gain | Window, depth 3 | vs base depth 3 | Arrival, greedy | vs base | Picks the teacher's move | Regret | Value error | Ranking error | Patch weights | In training: teacher / finisher |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 50.4% | | | 43.8% | | 29.8% | | 71.0% | 2,718 | 33,925 | 27,005 | 0 | |
| 10,000 | 53.1% | +2.7% ± 1.9% | 15% | 43.8% | +0.0% ± 1.9% | 33.0% | +3.2% ± 2.1% | 78.2% | 1,125 | 9,006 | 7,527 | 1,177,642 | 68.6% / 53.1% |
| 20,000 | 55.7% | +5.3% ± 1.9% | 30% | 40.7% | -3.1% ± 1.9% | 33.0% | +3.1% ± 2.1% | 79.5% | 1,094 | 8,273 | 6,985 | 1,506,170 | 67.6% / 55.4% |
| 30,000 | 56.0% | +5.6% ± 1.9% | 32% | 38.7% | -5.1% ± 1.9% | 32.8% | +2.9% ± 2.0% | 80.9% | 1,018 | 7,946 | 6,742 | 1,701,035 | 67.5% / 56.5% |
| 40,000 | 57.1% | +6.6% ± 1.9% | 37% | 39.5% | -4.4% ± 1.9% | 34.1% | +4.3% ± 2.0% | 81.0% | 986 | 7,751 | 6,574 | 1,845,923 | 68.6% / 58.1% |

Picks the teacher's move: how often the finisher's one-move choice is the teacher's, along the teacher's play from dev boards. Regret: how much more the teacher values its own move than the finisher's, on average. Value error: the mean difference between the two's values of a move. Ranking error: the same once each choice's mean difference is taken off.
