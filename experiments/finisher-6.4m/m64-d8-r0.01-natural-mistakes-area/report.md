# Finisher m64-d8-r0.01-natural-mistakes-area

Window: the last 64 of tile sum before a 32768. Teacher: depth 8, cutoff 0.02. Dev set: 1,083 window boards and 500 well-placed arrival positions, 4 continuations each. Differences are paired with the base, ± 95%.

The base alone, from the window: greedy play makes the 32768 in 50.4%, depth 3 in 43.8%, the teacher's search in 68.1%. From arrival, greedy play makes it in 29.8%.

| Attempts | Window, greedy | vs base greedy | Share of the teacher's gain | Window, depth 3 | vs base depth 3 | Arrival, greedy | vs base | Picks the teacher's move | Regret | Value error | Ranking error | Patch weights | In training: teacher / finisher |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 50.4% | | | 43.8% | | 29.8% | | 71.0% | 2,718 | 33,925 | 27,005 | 0 | |
| 10,000 | 63.0% | +12.5% ± 1.8% | 71% | 52.6% | +8.7% ± 1.9% | 37.2% | +7.4% ± 1.9% | 87.2% | 615 | 42,482 | 20,962 | 540,154 | 68.9% / 57.8% |
| 20,000 | 63.6% | +13.2% ± 1.8% | 74% | 53.1% | +9.3% ± 2.1% | 37.0% | +7.1% ± 2.0% | 89.1% | 450 | 43,522 | 20,626 | 702,691 | 68.0% / 63.3% |
| 30,000 | 65.0% | +14.5% ± 1.8% | 82% | 53.1% | +9.3% ± 2.0% | 37.6% | +7.8% ± 2.0% | 90.1% | 425 | 44,344 | 20,500 | 816,644 | 68.2% / 63.4% |
| 40,000 | 64.8% | +14.3% ± 1.9% | 81% | 53.8% | +10.0% ± 2.0% | 37.4% | +7.6% ± 2.0% | 90.5% | 372 | 44,722 | 20,430 | 918,723 | 67.7% / 64.8% |

Picks the teacher's move: how often the finisher's one-move choice is the teacher's, along the teacher's play from dev boards. Regret: how much more the teacher values its own move than the finisher's, on average. Value error: the mean difference between the two's values of a move. Ranking error: the same once each choice's mean difference is taken off.
