# Finisher m1024-d3-r0.01-natural-mistakes-rollout10

Window: the last 1024 of tile sum before a 32768. Teacher: depth 3, cutoff 0. Dev set: 1,885 window boards and 500 well-placed arrival positions, 4 continuations each. Differences are paired with the base, ± 95%.

The base alone, from the window: greedy play makes the 32768 in 88.6%, depth 3 in 94.4%, the teacher's search in 94.4%. From arrival, greedy play makes it in 83.5%.

| Attempts | Window, greedy | vs base greedy | Share of the teacher's gain | Window, depth 3 | vs base depth 3 | Arrival, greedy | vs base | Picks the teacher's move | Regret | Value error | Ranking error | Patch weights | In training: teacher / finisher |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 88.6% | | | 94.4% | | 83.5% | | 72.8% | 107 | 9,006 | 10,386 | 0 | |
| 5,000 | 87.2% | -1.5% ± 1.0% | -26% | 91.5% | -2.8% ± 0.7% | 82.0% | -1.6% ± 1.8% | 70.7% | 123 | 10,081 | 9,008 | 797,379 | 91.7% / 87.6% |
| 10,000 | 88.3% | -0.3% ± 1.0% | -5% | 91.7% | -2.7% ± 0.8% | 83.9% | +0.4% ± 1.8% | 71.3% | 124 | 10,051 | 8,727 | 1,167,027 | 92.3% / 87.5% |
| 15,000 | 88.3% | -0.4% ± 0.9% | -7% | 91.8% | -2.5% ± 0.8% | 83.2% | -0.4% ± 1.8% | 71.4% | 126 | 10,139 | 8,699 | 1,440,752 | 92.4% / 89.2% |
| 20,000 | 89.0% | +0.3% ± 0.9% | 6% | 91.4% | -2.9% ± 0.8% | 84.8% | +1.2% ± 1.8% | 71.5% | 126 | 10,163 | 8,616 | 1,710,772 | 92.6% / 89.8% |
| 25,000 | 89.0% | +0.3% ± 0.9% | 6% | 91.7% | -2.7% ± 0.7% | 83.5% | -0.1% ± 1.7% | 71.5% | 126 | 10,132 | 8,515 | 1,930,790 | 92.7% / 90.0% |
| 30,000 | 89.7% | +1.0% ± 0.9% | 18% | 92.3% | -2.1% ± 0.7% | 84.8% | +1.2% ± 1.7% | 71.7% | 124 | 10,138 | 8,418 | 2,166,580 | 91.8% / 89.5% |
| 35,000 | 89.4% | +0.8% ± 0.9% | 14% | 92.0% | -2.4% ± 0.7% | 85.0% | +1.5% ± 1.7% | 72.0% | 122 | 10,166 | 8,403 | 2,362,191 | 91.9% / 89.3% |
| 40,000 | 89.7% | +1.1% ± 0.9% | 19% | 91.9% | -2.4% ± 0.7% | 84.5% | +0.9% ± 1.7% | 72.0% | 126 | 10,198 | 8,379 | 2,551,934 | 91.8% / 88.9% |

Picks the teacher's move: how often the finisher's one-move choice is the teacher's, along the teacher's play from dev boards. Regret: how much more the teacher values its own move than the finisher's, on average. Value error: the mean difference between the two's values of a move. Ranking error: the same once each choice's mean difference is taken off.
