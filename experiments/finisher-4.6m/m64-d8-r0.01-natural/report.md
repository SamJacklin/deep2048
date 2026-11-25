# Finisher m64-d8-r0.01-natural

Window: the last 64 of tile sum before a 32768. Teacher: depth 8, cutoff 0.02. Dev set: 1,040 window boards and 500 well-placed arrival positions, 4 continuations each. Differences are paired with the base, ± 95%.

The base alone, from the window: greedy play makes the 32768 in 49.7%, depth 3 in 46.4%, the teacher's search in 68.3%. From arrival, greedy play makes it in 29.2%.

| Attempts | Window, greedy | vs base greedy | Share of the teacher's gain | Window, depth 3 | vs base depth 3 | Arrival, greedy | vs base | Picks the teacher's move | Regret | Value error | Ranking error | Patch weights | In training: teacher / finisher |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 49.7% | | | 46.4% | | 29.2% | | 74.4% | 2,777 | 34,275 | 27,007 | 0 | |
| 20,000 | 58.2% | +8.5% ± 1.8% | 46% | 42.8% | -3.6% ± 1.9% | 33.8% | +4.5% ± 2.1% | 81.4% | 1,015 | 8,732 | 7,378 | 1,334,935 | 69.5% / 54.4% |
| 40,000 | 59.5% | +9.8% ± 1.8% | 53% | 40.3% | -6.1% ± 1.9% | 34.1% | +4.8% ± 2.1% | 82.6% | 923 | 8,264 | 7,018 | 1,691,067 | 69.8% / 57.4% |
| 60,000 | 59.6% | +10.0% ± 1.8% | 53% | 41.3% | -5.1% ± 1.9% | 34.2% | +4.9% ± 2.0% | 83.0% | 881 | 7,999 | 6,839 | 1,940,874 | 69.0% / 59.7% |
| 80,000 | 58.6% | +8.9% ± 1.8% | 48% | 42.3% | -4.2% ± 1.9% | 34.0% | +4.7% ± 2.1% | 83.2% | 853 | 7,827 | 6,713 | 2,127,479 | 69.4% / 58.1% |
| 100,000 | 59.4% | +9.7% ± 1.8% | 52% | 41.1% | -5.3% ± 1.9% | 34.5% | +5.2% ± 2.0% | 83.4% | 879 | 7,710 | 6,615 | 2,285,852 | 68.6% / 57.8% |
| 120,000 | 59.1% | +9.4% ± 1.8% | 51% | 41.0% | -5.5% ± 1.9% | 33.7% | +4.5% ± 2.1% | 84.0% | 856 | 7,623 | 6,563 | 2,412,409 | 70.0% / 58.7% |
| 140,000 | 58.0% | +8.3% ± 1.8% | 44% | 39.5% | -6.9% ± 1.9% | 33.7% | +4.4% ± 2.1% | 84.1% | 822 | 7,574 | 6,516 | 2,536,442 | 69.2% / 58.5% |
| 160,000 | 59.3% | +9.6% ± 1.8% | 51% | 42.0% | -4.5% ± 1.9% | 34.4% | +5.1% ± 2.1% | 84.3% | 843 | 7,528 | 6,474 | 2,640,805 | 68.8% / 59.5% |
| 180,000 | 58.2% | +8.5% ± 1.8% | 46% | 42.0% | -4.4% ± 1.9% | 33.8% | +4.5% ± 2.1% | 84.2% | 832 | 7,445 | 6,401 | 2,732,093 | 69.7% / 58.4% |
| 200,000 | 58.9% | +9.2% ± 1.8% | 49% | 41.3% | -5.2% ± 1.9% | 33.6% | +4.3% ± 2.1% | 83.9% | 824 | 7,455 | 6,415 | 2,820,338 | 69.9% / 58.8% |

Picks the teacher's move: how often the finisher's one-move choice is the teacher's, along the teacher's play from dev boards. Regret: how much more the teacher values its own move than the finisher's, on average. Value error: the mean difference between the two's values of a move. Ranking error: the same once each choice's mean difference is taken off.
