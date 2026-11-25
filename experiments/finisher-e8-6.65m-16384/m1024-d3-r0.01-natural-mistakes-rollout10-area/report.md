# Finisher m1024-d3-r0.01-natural-mistakes-rollout10-area

Window: the last 1024 of tile sum before a 32768. Teacher: depth 3, cutoff 0. Dev set: 1,885 window boards and 500 well-placed arrival positions, 4 continuations each. Differences are paired with the base, ± 95%.

The base alone, from the window: greedy play makes the 32768 in 89.4%, depth 3 in 95.2%, the teacher's search in 95.2%. From arrival, greedy play makes it in 83.8%.

| Attempts | Window, greedy | vs base greedy | Share of the teacher's gain | Window, depth 3 | vs base depth 3 | Arrival, greedy | vs base | Picks the teacher's move | Regret | Value error | Ranking error | Patch weights | In training: teacher / finisher |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 89.4% | | | 95.2% | | 83.8% | | 73.6% | 89 | 9,047 | 10,506 | 0 | |
| 5,000 | 88.7% | -0.7% ± 0.9% | -11% | 92.4% | -2.8% ± 0.7% | 84.6% | +0.8% ± 1.7% | 71.5% | 103 | 9,823 | 9,296 | 836,434 | 90.7% / 86.3% |
| 10,000 | 88.8% | -0.6% ± 0.9% | -11% | 92.9% | -2.3% ± 0.7% | 84.0% | +0.1% ± 1.8% | 72.0% | 104 | 9,896 | 9,107 | 1,191,122 | 92.8% / 88.8% |
| 15,000 | 89.8% | +0.4% ± 0.9% | 8% | 92.3% | -2.9% ± 0.7% | 84.9% | +1.1% ± 1.7% | 72.4% | 101 | 9,901 | 8,974 | 1,471,645 | 91.9% / 89.2% |
| 20,000 | 90.3% | +0.9% ± 0.9% | 15% | 92.3% | -2.9% ± 0.7% | 85.0% | +1.2% ± 1.8% | 72.6% | 99 | 9,878 | 8,850 | 1,764,167 | 93.4% / 89.5% |
| 25,000 | 90.2% | +0.8% ± 0.9% | 14% | 92.1% | -3.1% ± 0.7% | 84.9% | +1.1% ± 1.8% | 72.7% | 100 | 9,854 | 8,738 | 2,002,658 | 91.8% / 88.7% |
| 30,000 | 90.6% | +1.2% ± 0.9% | 21% | 92.6% | -2.6% ± 0.7% | 84.0% | +0.2% ± 1.7% | 72.6% | 103 | 9,863 | 8,668 | 2,210,236 | 93.7% / 90.3% |
| 35,000 | 90.5% | +1.1% ± 0.9% | 19% | 92.6% | -2.6% ± 0.7% | 84.4% | +0.6% ± 1.8% | 72.8% | 101 | 9,868 | 8,596 | 2,393,703 | 92.0% / 91.1% |
| 40,000 | 90.2% | +0.8% ± 0.9% | 14% | 91.9% | -3.2% ± 0.8% | 84.9% | +1.1% ± 1.7% | 72.9% | 99 | 9,852 | 8,522 | 2,590,620 | 92.5% / 89.6% |

Picks the teacher's move: how often the finisher's one-move choice is the teacher's, along the teacher's play from dev boards. Regret: how much more the teacher values its own move than the finisher's, on average. Value error: the mean difference between the two's values of a move. Ranking error: the same once each choice's mean difference is taken off.
