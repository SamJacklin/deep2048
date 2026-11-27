# Finisher m1024-d3-r0.01-natural-mistakes-rollout10

Window: the last 1024 of tile sum before a 32768. Teacher: depth 3, cutoff 0. Dev set: 1,930 window boards and 500 well-placed arrival positions, 4 continuations each. Differences are paired with the base, ± 95%.

The base alone, from the window: greedy play makes the 32768 in 92.1%, depth 3 in 95.2%, the teacher's search in 95.2%. From arrival, greedy play makes it in 87.4%.

| Attempts | Window, greedy | vs base greedy | Share of the teacher's gain | Window, depth 3 | vs base depth 3 | Arrival, greedy | vs base | Picks the teacher's move | Regret | Value error | Ranking error | Patch weights | In training: teacher / finisher |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 92.1% | | | 95.2% | | 87.4% | | 73.0% | 79 | 9,655 | 11,207 | 0 | |
| 5,000 | 90.4% | -1.6% ± 0.8% | -53% | 94.1% | -1.0% ± 0.7% | 86.6% | -0.8% ± 1.7% | 71.0% | 96 | 10,282 | 9,852 | 931,479 | 93.3% / 87.7% |
| 10,000 | 90.9% | -1.1% ± 0.8% | -36% | 93.5% | -1.6% ± 0.7% | 87.8% | +0.4% ± 1.6% | 71.6% | 91 | 10,404 | 9,786 | 1,300,012 | 93.5% / 90.1% |
| 15,000 | 91.2% | -0.9% ± 0.8% | -28% | 94.2% | -0.9% ± 0.6% | 87.9% | +0.5% ± 1.6% | 71.8% | 93 | 10,332 | 9,570 | 1,590,247 | 93.5% / 89.8% |
| 20,000 | 91.5% | -0.5% ± 0.8% | -17% | 93.8% | -1.3% ± 0.7% | 87.4% | +0.1% ± 1.6% | 71.8% | 90 | 10,400 | 9,588 | 1,837,014 | 93.8% / 90.6% |
| 25,000 | 91.5% | -0.5% ± 0.8% | -18% | 94.1% | -1.1% ± 0.7% | 87.1% | -0.3% ± 1.6% | 71.8% | 93 | 10,416 | 9,507 | 2,040,434 | 93.0% / 91.4% |
| 30,000 | 91.9% | -0.2% ± 0.8% | -5% | 93.8% | -1.4% ± 0.7% | 88.9% | +1.6% ± 1.5% | 72.2% | 88 | 10,461 | 9,480 | 2,233,084 | 92.8% / 90.7% |
| 35,000 | 92.2% | +0.1% ± 0.8% | 3% | 93.8% | -1.3% ± 0.7% | 88.4% | +1.1% ± 1.5% | 71.8% | 92 | 10,433 | 9,375 | 2,417,001 | 93.8% / 91.7% |
| 40,000 | 91.5% | -0.6% ± 0.8% | -18% | 93.7% | -1.5% ± 0.7% | 88.2% | +0.9% ± 1.5% | 71.9% | 91 | 10,471 | 9,379 | 2,595,912 | 92.4% / 91.3% |

Picks the teacher's move: how often the finisher's one-move choice is the teacher's, along the teacher's play from dev boards. Regret: how much more the teacher values its own move than the finisher's, on average. Value error: the mean difference between the two's values of a move. Ranking error: the same once each choice's mean difference is taken off.
