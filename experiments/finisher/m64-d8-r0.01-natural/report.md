# Finisher m64-d8-r0.01-natural

Window: the last 64 of tile sum before a 32768. Teacher: depth 8, cutoff 0.02. Dev set: 1,020 window boards and 500 well-placed arrival positions, 4 continuations each. Differences are paired with the base, ± 95%.

The base alone, from the window: greedy play makes the 32768 in 42.4%, depth 3 in 52.1%, the teacher's search in 71.1%. From arrival, greedy play makes it in 24.9%.

| Attempts | Window, greedy | vs base greedy | Share of the teacher's gain | Window, depth 3 | vs base depth 3 | Arrival, greedy | vs base | Picks the teacher's move | Regret | Value error | Ranking error | Patch weights | In training: teacher / finisher |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 42.4% | | | 52.1% | | 24.9% | | 76.2% | 2,541 | 32,487 | 23,780 | 0 | |
| 20,000 | 56.6% | +14.2% ± 1.9% | 49% | 45.7% | -6.3% ± 2.0% | 32.2% | +7.3% ± 2.1% | 82.0% | 1,064 | 8,672 | 6,951 | 1,239,972 | 68.9% / 54.2% |
| 40,000 | 56.8% | +14.4% ± 1.9% | 50% | 44.9% | -7.2% ± 2.0% | 33.2% | +8.3% ± 2.1% | 82.8% | 961 | 8,120 | 6,570 | 1,563,262 | 69.1% / 55.9% |
| 60,000 | 57.2% | +14.8% ± 1.9% | 52% | 42.5% | -9.6% ± 2.0% | 34.4% | +9.5% ± 2.1% | 83.6% | 919 | 7,826 | 6,371 | 1,790,493 | 69.7% / 57.4% |
| 80,000 | 58.8% | +16.4% ± 1.9% | 57% | 45.3% | -6.8% ± 2.1% | 33.1% | +8.2% ± 2.1% | 83.8% | 928 | 7,704 | 6,287 | 1,979,781 | 69.5% / 57.0% |
| 100,000 | 58.7% | +16.2% ± 1.9% | 57% | 43.4% | -8.7% ± 2.0% | 32.9% | +8.0% ± 2.1% | 83.7% | 896 | 7,574 | 6,186 | 2,136,706 | 68.3% / 56.4% |
| 120,000 | 59.0% | +16.6% ± 1.9% | 58% | 49.1% | -3.0% ± 2.0% | 33.8% | +8.9% ± 2.2% | 83.7% | 889 | 7,529 | 6,127 | 2,273,750 | 69.5% / 56.6% |
| 140,000 | 57.1% | +14.7% ± 2.0% | 51% | 43.2% | -8.9% ± 2.0% | 33.8% | +8.8% ± 2.2% | 84.2% | 890 | 7,473 | 6,112 | 2,419,885 | 69.0% / 57.3% |
| 160,000 | 58.0% | +15.6% ± 1.9% | 54% | 44.3% | -7.7% ± 2.1% | 32.7% | +7.8% ± 2.1% | 84.9% | 890 | 7,479 | 6,093 | 2,530,040 | 69.8% / 56.6% |
| 180,000 | 59.1% | +16.7% ± 1.9% | 58% | 44.4% | -7.7% ± 2.0% | 33.4% | +8.5% ± 2.1% | 83.9% | 905 | 7,369 | 6,028 | 2,621,528 | 70.0% / 57.5% |
| 200,000 | 58.9% | +16.5% ± 1.9% | 58% | 42.0% | -10.1% ± 2.1% | 33.0% | +8.1% ± 2.1% | 84.2% | 865 | 7,300 | 5,986 | 2,717,756 | 69.6% / 58.5% |

Picks the teacher's move: how often the finisher's one-move choice is the teacher's, along the teacher's play from dev boards. Regret: how much more the teacher values its own move than the finisher's, on average. Value error: the mean difference between the two's values of a move. Ranking error: the same once each choice's mean difference is taken off.
