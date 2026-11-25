# Finisher m64-d8-r0.02-natural-area-td

Window: the last 64 of tile sum before a 32768. Teacher: depth 8, cutoff 0.02. Dev set: 1,083 window boards and 500 well-placed arrival positions, 4 continuations each. Differences are paired with the base, ± 95%.

The base alone, from the window: greedy play makes the 32768 in 50.4%, depth 3 in 43.8%, the teacher's search in 68.1%. From arrival, greedy play makes it in 29.8%.

| Attempts | Window, greedy | vs base greedy | Share of the teacher's gain | Window, depth 3 | vs base depth 3 | Arrival, greedy | vs base | Picks the teacher's move | Regret | Value error | Ranking error | Patch weights | In training: teacher / finisher |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 50.4% | | | 43.8% | | 29.8% | | 71.0% | 2,718 | 33,925 | 27,005 | 0 | |
| 500,000 | 40.2% | -10.2% ± 1.9% | -58% | 39.5% | -4.4% ± 2.0% | 24.1% | -5.7% ± 2.3% | 71.0% | 2,844 | 37,041 | 25,959 | 166,420 | 0.0% / 40.7% |
| 1,000,000 | 39.0% | -11.4% ± 1.9% | -64% | 36.5% | -7.3% ± 1.9% | 22.9% | -7.0% ± 2.3% | 71.8% | 2,948 | 38,176 | 25,908 | 214,684 | 0.0% / 39.5% |
| 1,500,000 | 38.9% | -11.5% ± 2.0% | -65% | 34.1% | -9.7% ± 1.9% | 23.4% | -6.4% ± 2.2% | 71.0% | 3,004 | 38,818 | 25,908 | 249,115 | 0.0% / 38.8% |
| 2,000,000 | 38.4% | -12.0% ± 2.0% | -68% | 34.6% | -9.2% ± 2.0% | 23.1% | -6.7% ± 2.3% | 70.8% | 3,039 | 39,324 | 25,948 | 277,569 | 0.0% / 38.4% |

Picks the teacher's move: how often the finisher's one-move choice is the teacher's, along the teacher's play from dev boards. Regret: how much more the teacher values its own move than the finisher's, on average. Value error: the mean difference between the two's values of a move. Ranking error: the same once each choice's mean difference is taken off.
