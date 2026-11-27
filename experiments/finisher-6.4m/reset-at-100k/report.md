# Finisher reset-at-100k

Window: the last 64 of tile sum before a 32768. Teacher: depth 8, cutoff 0.02. Dev set: 1,083 window boards and 500 well-placed arrival positions, 4 continuations each. Differences are paired with the base, ± 95%.

The base alone, from the window: greedy play makes the 32768 in 50.4%, depth 3 in 43.8%, the teacher's search in 68.1%. From arrival, greedy play makes it in 29.8%.

| Attempts | Window, greedy | vs base greedy | Share of the teacher's gain | Window, depth 3 | vs base depth 3 | Arrival, greedy | vs base | Picks the teacher's move | Regret | Value error | Ranking error | Patch weights | In training: teacher / finisher |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 120,000 | 57.2% | +6.8% ± 1.9% | 39% | 43.4% | -0.4% ± 2.0% | 34.5% | +4.7% ± 2.0% | 81.7% | 867 | 7,200 | 6,122 | 2,570,087 | 68.9% / 57.2% |
| 140,000 | 57.8% | +7.4% ± 1.9% | 42% | 42.1% | -1.8% ± 2.0% | 34.9% | +5.1% ± 2.1% | 82.0% | 865 | 7,080 | 6,050 | 2,692,319 | 68.5% / 56.9% |

Picks the teacher's move: how often the finisher's one-move choice is the teacher's, along the teacher's play from dev boards. Regret: how much more the teacher values its own move than the finisher's, on average. Value error: the mean difference between the two's values of a move. Ranking error: the same once each choice's mean difference is taken off.
