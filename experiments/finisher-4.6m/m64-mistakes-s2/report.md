# Finisher m64-mistakes-s2

Window: the last 64 of tile sum before a 32768. Teacher: depth 8, cutoff 0.02. Dev set: 1,040 window boards and 500 well-placed arrival positions, 4 continuations each. Differences are paired with the base, ± 95%.

The base alone, from the window: greedy play makes the 32768 in 49.7%, depth 3 in 46.4%, the teacher's search in 68.3%. From arrival, greedy play makes it in 29.2%.

| Attempts | Window, greedy | vs base greedy | Share of the teacher's gain | Window, depth 3 | vs base depth 3 | Arrival, greedy | vs base | Picks the teacher's move | Regret | Value error | Ranking error | Patch weights | In training: teacher / finisher |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 49.7% | | | 46.4% | | 29.2% | | 74.4% | 2,777 | 34,275 | 27,007 | 0 | |
| 10,000 | 56.4% | +6.7% ± 1.9% | 36% | 57.8% | +11.4% ± 1.9% | 33.4% | +4.1% ± 2.0% | 86.8% | 796 | 43,644 | 22,210 | 446,607 | 68.8% / 56.0% |
| 20,000 | 59.1% | +9.4% ± 1.9% | 50% | 55.7% | +9.3% ± 1.9% | 35.4% | +6.2% ± 1.9% | 87.0% | 788 | 44,860 | 22,206 | 604,517 | 69.7% / 58.0% |
| 30,000 | 59.5% | +9.8% ± 1.9% | 53% | 54.7% | +8.3% ± 2.0% | 34.8% | +5.6% ± 2.1% | 87.9% | 660 | 45,669 | 22,221 | 710,937 | 68.5% / 57.6% |
| 40,000 | 58.8% | +9.1% ± 1.9% | 49% | 56.0% | +9.5% ± 2.0% | 34.2% | +4.9% ± 2.1% | 88.2% | 571 | 46,199 | 22,195 | 786,355 | 69.4% / 57.9% |

Picks the teacher's move: how often the finisher's one-move choice is the teacher's, along the teacher's play from dev boards. Regret: how much more the teacher values its own move than the finisher's, on average. Value error: the mean difference between the two's values of a move. Ranking error: the same once each choice's mean difference is taken off.
