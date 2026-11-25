# Finisher m64-d10-r0.01-natural-mistakes

Window: the last 64 of tile sum before a 32768. Teacher: depth 10, cutoff 0.02. Dev set: 1,083 window boards and 500 well-placed arrival positions, 4 continuations each. Differences are paired with the base, ± 95%.

The base alone, from the window: greedy play makes the 32768 in 50.4%, depth 3 in 43.8%, the teacher's search in 70.5%. From arrival, greedy play makes it in 29.8%.

| Attempts | Window, greedy | vs base greedy | Share of the teacher's gain | Window, depth 3 | vs base depth 3 | Arrival, greedy | vs base | Picks the teacher's move | Regret | Value error | Ranking error | Patch weights | In training: teacher / finisher |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 50.4% | | | 43.8% | | 29.8% | | 72.0% | 2,662 | 34,095 | 26,443 | 0 | |
| 20,000 | 55.9% | +5.5% ± 1.9% | 28% | 52.4% | +8.6% ± 2.0% | 33.3% | +3.5% ± 2.1% | 85.3% | 732 | 44,935 | 20,565 | 654,865 | 69.6% / 56.7% |
| 40,000 | 58.0% | +7.5% ± 1.9% | 38% | 54.1% | +10.3% ± 2.0% | 33.0% | +3.2% ± 2.1% | 86.3% | 652 | 46,104 | 20,776 | 850,553 | 68.9% / 58.1% |

Picks the teacher's move: how often the finisher's one-move choice is the teacher's, along the teacher's play from dev boards. Regret: how much more the teacher values its own move than the finisher's, on average. Value error: the mean difference between the two's values of a move. Ranking error: the same once each choice's mean difference is taken off.
