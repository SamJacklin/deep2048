# Finisher m64-mistakes-onpolicy

Window: the last 64 of tile sum before a 32768. Teacher: depth 8, cutoff 0.02. Dev set: 1,083 window boards and 500 well-placed arrival positions, 4 continuations each. Differences are paired with the base, ± 95%.

The base alone, from the window: greedy play makes the 32768 in 50.4%, depth 3 in 43.8%, the teacher's search in 68.1%. From arrival, greedy play makes it in 29.8%.

| Attempts | Window, greedy | vs base greedy | Share of the teacher's gain | Window, depth 3 | vs base depth 3 | Arrival, greedy | vs base | Picks the teacher's move | Regret | Value error | Ranking error | Patch weights | In training: teacher / finisher |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 50.4% | | | 43.8% | | 29.8% | | 71.0% | 2,718 | 33,925 | 27,005 | 0 | |
| 20,000 | 55.4% | +5.0% ± 1.9% | 28% | 54.4% | +10.6% ± 2.0% | 32.9% | +3.0% ± 2.1% | 84.9% | 821 | 41,959 | 20,782 | 650,105 | 0.0% / 56.7% |
| 40,000 | 57.0% | +6.6% ± 1.9% | 37% | 56.6% | +12.7% ± 1.9% | 33.9% | +4.0% ± 2.1% | 85.7% | 750 | 43,592 | 20,774 | 877,863 | 0.0% / 58.5% |

Picks the teacher's move: how often the finisher's one-move choice is the teacher's, along the teacher's play from dev boards. Regret: how much more the teacher values its own move than the finisher's, on average. Value error: the mean difference between the two's values of a move. Ranking error: the same once each choice's mean difference is taken off.
