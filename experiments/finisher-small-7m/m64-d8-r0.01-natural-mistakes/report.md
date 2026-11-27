# Finisher m64-d8-r0.01-natural-mistakes

Window: the last 64 of tile sum before a 32768. Teacher: depth 8, cutoff 0.02. Dev set: 1,200 window boards and 500 well-placed arrival positions, 4 continuations each. Differences are paired with the base, ± 95%.

The base alone, from the window: greedy play makes the 32768 in 65.4%, depth 3 in 66.9%, the teacher's search in 73.3%. From arrival, greedy play makes it in 43.4%.

| Attempts | Window, greedy | vs base greedy | Share of the teacher's gain | Window, depth 3 | vs base depth 3 | Arrival, greedy | vs base | Picks the teacher's move | Regret | Value error | Ranking error | Patch weights | In training: teacher / finisher |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 65.4% | | | 66.9% | | 43.4% | | 70.3% | 2,200 | 34,059 | 27,449 | 0 | |
| 10,000 | 67.8% | +2.4% ± 1.5% | 30% | 69.0% | +2.1% ± 1.5% | 44.4% | +1.1% ± 1.9% | 87.0% | 663 | 39,050 | 20,141 | 656,621 | 73.6% / 65.9% |
| 20,000 | 69.4% | +4.0% ± 1.5% | 51% | 68.8% | +1.9% ± 1.5% | 46.6% | +3.2% ± 1.9% | 87.9% | 548 | 39,756 | 19,639 | 871,247 | 73.3% / 68.7% |
| 30,000 | 69.7% | +4.3% ± 1.5% | 55% | 67.5% | +0.7% ± 1.5% | 46.2% | +2.9% ± 1.9% | 88.6% | 482 | 40,290 | 19,285 | 1,020,997 | 73.0% / 70.0% |
| 40,000 | 69.4% | +4.1% ± 1.6% | 51% | 67.8% | +1.0% ± 1.5% | 46.8% | +3.4% ± 1.9% | 89.0% | 448 | 40,755 | 18,964 | 1,176,967 | 74.1% / 69.1% |

Picks the teacher's move: how often the finisher's one-move choice is the teacher's, along the teacher's play from dev boards. Regret: how much more the teacher values its own move than the finisher's, on average. Value error: the mean difference between the two's values of a move. Ranking error: the same once each choice's mean difference is taken off.
