# Finisher m64-d8-r0.01

Window: the last 64 of tile sum before a 32768. Teacher: depth 8, cutoff 0.02. Dev set: 1,020 window boards and 500 well-placed arrival positions, 4 continuations each. Differences are paired with the base, ± 95%.

The base alone, from the window: greedy play makes the 32768 in 42.4%, depth 3 in 52.1%, the teacher's search in 71.1%. From arrival, greedy play makes it in 24.9%.

| Attempts | Window, greedy | vs base greedy | Share of the teacher's gain | Window, depth 3 | vs base depth 3 | Arrival, greedy | vs base | Picks the teacher's move | Regret | Value error | Ranking error | Patch weights | In training: teacher / finisher |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 42.4% | | | 52.1% | | 24.9% | | 76.2% | 2,541 | 32,487 | 23,780 | 0 | |
| 20,000 | 54.0% | +11.6% ± 2.0% | 41% | 49.2% | -2.9% ± 2.1% | 31.2% | +6.3% ± 2.1% | 79.4% | 1,244 | 10,107 | 8,209 | 2,313,432 | 43.9% / 25.2% |
| 40,000 | 54.7% | +12.3% ± 2.0% | 43% | 48.8% | -3.3% ± 2.1% | 32.2% | +7.3% ± 2.1% | 80.7% | 1,091 | 9,462 | 7,716 | 3,108,583 | 41.7% / 27.3% |
| 60,000 | 55.8% | +13.4% ± 2.0% | 47% | 51.7% | -0.4% ± 2.0% | 33.4% | +8.5% ± 2.1% | 80.9% | 1,035 | 9,292 | 7,577 | 3,664,274 | 42.6% / 26.6% |
| 80,000 | 55.5% | +13.1% ± 1.9% | 46% | 49.2% | -2.9% ± 2.0% | 32.9% | +8.0% ± 2.1% | 81.2% | 993 | 9,042 | 7,376 | 4,044,940 | 42.4% / 26.7% |
| 100,000 | 56.1% | +13.7% ± 2.0% | 48% | 51.1% | -1.0% ± 2.0% | 33.7% | +8.8% ± 2.1% | 81.8% | 966 | 8,886 | 7,225 | 4,370,886 | 43.7% / 28.2% |
| 120,000 | 56.5% | +14.1% ± 1.9% | 49% | 49.5% | -2.5% ± 2.0% | 33.6% | +8.7% ± 2.2% | 81.8% | 941 | 8,830 | 7,230 | 4,661,028 | 41.9% / 28.1% |
| 140,000 | 56.5% | +14.1% ± 1.9% | 49% | 52.3% | +0.3% ± 1.9% | 33.2% | +8.3% ± 2.2% | 82.1% | 948 | 8,782 | 7,178 | 4,892,058 | 43.0% / 28.4% |
| 160,000 | 56.7% | +14.3% ± 2.0% | 50% | 52.4% | +0.3% ± 1.9% | 34.0% | +9.1% ± 2.2% | 82.1% | 914 | 8,747 | 7,138 | 5,098,977 | 41.9% / 27.6% |
| 180,000 | 56.8% | +14.4% ± 1.9% | 50% | 52.8% | +0.7% ± 2.0% | 33.8% | +8.9% ± 2.1% | 82.0% | 904 | 8,614 | 7,039 | 5,277,451 | 41.8% / 27.4% |
| 200,000 | 57.6% | +15.2% ± 1.9% | 53% | 52.5% | +0.5% ± 1.9% | 34.4% | +9.5% ± 2.1% | 82.3% | 916 | 8,657 | 7,059 | 5,440,848 | 42.5% / 27.8% |

Picks the teacher's move: how often the finisher's one-move choice is the teacher's, along the teacher's play from dev boards. Regret: how much more the teacher values its own move than the finisher's, on average. Value error: the mean difference between the two's values of a move. Ranking error: the same once each choice's mean difference is taken off.
