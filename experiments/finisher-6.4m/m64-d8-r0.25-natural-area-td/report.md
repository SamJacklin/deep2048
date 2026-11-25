# Finisher m64-d8-r0.25-natural-area-td

Window: the last 64 of tile sum before a 32768. Teacher: depth 8, cutoff 0.02. Dev set: 1,083 window boards and 500 well-placed arrival positions, 4 continuations each. Differences are paired with the base, ± 95%.

The base alone, from the window: greedy play makes the 32768 in 50.4%, depth 3 in 43.8%, the teacher's search in 68.1%. From arrival, greedy play makes it in 29.8%.

| Attempts | Window, greedy | vs base greedy | Share of the teacher's gain | Window, depth 3 | vs base depth 3 | Arrival, greedy | vs base | Picks the teacher's move | Regret | Value error | Ranking error | Patch weights | In training: teacher / finisher |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 50.4% | | | 43.8% | | 29.8% | | 71.0% | 2,718 | 33,925 | 27,005 | 0 | |
| 50,000 | 34.6% | -15.8% ± 2.0% | -89% | 36.0% | -7.8% ± 1.9% | 20.5% | -9.2% ± 2.2% | 70.7% | 2,975 | 38,258 | 25,772 | 87,089 | 0.0% / 35.3% |
| 100,000 | 35.9% | -14.5% ± 1.9% | -82% | 35.1% | -8.7% ± 1.9% | 20.5% | -9.2% ± 2.2% | 71.0% | 3,063 | 39,320 | 25,770 | 112,693 | 0.0% / 36.5% |
| 150,000 | 37.5% | -12.9% ± 1.9% | -73% | 35.2% | -8.6% ± 1.9% | 21.6% | -8.2% ± 2.2% | 71.2% | 3,083 | 39,835 | 25,902 | 130,523 | 0.0% / 36.3% |
| 200,000 | 35.2% | -15.2% ± 2.0% | -86% | 34.9% | -8.9% ± 1.9% | 21.6% | -8.2% ± 2.3% | 69.6% | 3,216 | 40,660 | 26,092 | 145,392 | 0.0% / 35.5% |
| 250,000 | 32.1% | -18.3% ± 2.1% | -103% | 33.3% | -10.5% ± 1.9% | 20.1% | -9.8% ± 2.2% | 69.6% | 3,189 | 41,335 | 26,214 | 157,935 | 0.0% / 33.9% |
| 300,000 | 31.0% | -19.5% ± 2.1% | -110% | 30.9% | -12.9% ± 2.0% | 18.2% | -11.6% ± 2.3% | 69.6% | 3,384 | 41,650 | 26,147 | 169,338 | 0.0% / 30.9% |
| 350,000 | 31.6% | -18.9% ± 2.1% | -106% | 31.4% | -12.4% ± 2.0% | 18.6% | -11.2% ± 2.3% | 70.4% | 3,342 | 42,029 | 26,159 | 179,567 | 0.0% / 30.5% |
| 400,000 | 31.4% | -19.0% ± 2.1% | -107% | 31.0% | -12.8% ± 2.0% | 19.1% | -10.8% ± 2.2% | 69.8% | 3,359 | 42,276 | 26,220 | 188,168 | 0.0% / 30.5% |

Picks the teacher's move: how often the finisher's one-move choice is the teacher's, along the teacher's play from dev boards. Regret: how much more the teacher values its own move than the finisher's, on average. Value error: the mean difference between the two's values of a move. Ranking error: the same once each choice's mean difference is taken off.
