2,000 greedy games of checkpoints/ntuple/methods-8x6-scratch-stall-small/stall_7000000.npz, seeds from 62,000,000; 27,625,715 decisions

| Largest tile | Decisions | Afterstates with an exact read | with a one-over read | one-over but no exact | Exact reads per afterstate | One-over reads per afterstate | Decisions where a move has a one-over read | where moves differ in one-over reads | where moves differ in exact reads |
|---|---|---|---|---|---|---|---|---|---|
| ≤4096 | 7,379,566 | 90.1% | 81.0% | 9.0% | 5.65 | 3.20 | 86.7% | 86.6% | 99.9% |
| 8192 | 7,027,526 | 86.5% | 69.4% | 10.7% | 4.66 | 2.36 | 82.3% | 82.1% | 99.9% |
| 16384 | 11,024,827 | 84.3% | 63.1% | 11.6% | 4.06 | 2.12 | 78.4% | 78.2% | 99.8% |
| 32768 | 2,193,796 | 85.7% | 70.6% | 11.4% | 4.21 | 2.74 | 83.2% | 83.0% | 99.4% |
| All | 27,625,715 | 86.5% | 70.1% | 10.6% | 4.65 | 2.52 | 82.0% | 81.8% | 99.8% |

Largest tile 8192, by how far the chain runs below it

| Chain runs down to | Decisions | Afterstates with an exact read | with a one-over read | one-over but no exact | Exact reads per afterstate | One-over reads per afterstate | Decisions where a move has a one-over read | where moves differ in one-over reads | where moves differ in exact reads |
|---|---|---|---|---|---|---|---|---|---|
| 8,192 | 3,561,256 | 89.6% | 78.4% | 9.0% | 5.49 | 2.77 | 87.3% | 87.0% | 100.0% |
| 4,096 | 1,757,068 | 86.1% | 67.1% | 11.5% | 4.53 | 2.17 | 81.0% | 80.8% | 100.0% |
| 2,048 | 871,241 | 82.4% | 56.2% | 13.8% | 3.79 | 1.66 | 75.7% | 75.6% | 99.8% |
| 1,024 | 424,464 | 80.8% | 49.0% | 13.4% | 3.06 | 1.72 | 73.3% | 73.1% | 99.7% |
| 512 | 211,096 | 77.2% | 84.6% | 14.5% | 1.54 | 3.10 | 99.5% | 99.2% | 99.7% |
| 256 | 113,877 | 76.7% | 10.7% | 10.7% | 1.53 | 0.21 | 36.7% | 36.7% | 99.6% |
| 128 | 38,813 | 72.7% | 7.1% | 7.1% | 1.45 | 0.14 | 24.2% | 24.2% | 99.6% |
| 64+ | 49,711 | 73.7% | 10.7% | 10.7% | 1.47 | 0.21 | 34.8% | 34.8% | 99.2% |

Largest tile 16384, by how far the chain runs below it

| Chain runs down to | Decisions | Afterstates with an exact read | with a one-over read | one-over but no exact | Exact reads per afterstate | One-over reads per afterstate | Decisions where a move has a one-over read | where moves differ in one-over reads | where moves differ in exact reads |
|---|---|---|---|---|---|---|---|---|---|
| 16,384 | 6,212,484 | 86.6% | 69.5% | 10.7% | 4.67 | 2.37 | 82.3% | 82.1% | 99.9% |
| 8,192 | 2,684,233 | 83.4% | 60.1% | 12.4% | 3.81 | 1.94 | 77.2% | 77.0% | 99.8% |
| 4,096 | 1,195,712 | 80.4% | 52.9% | 13.5% | 3.05 | 1.72 | 73.3% | 73.2% | 99.6% |
| 2,048 | 518,320 | 78.1% | 49.2% | 13.4% | 2.28 | 1.73 | 70.7% | 70.5% | 99.1% |
| 1,024 | 229,070 | 75.4% | 48.9% | 13.0% | 1.51 | 1.70 | 67.5% | 67.4% | 98.7% |
| 512 | 102,372 | 74.6% | 10.4% | 10.4% | 1.49 | 0.21 | 34.1% | 34.1% | 98.8% |
| 256 | 47,047 | 71.3% | 9.1% | 9.1% | 1.43 | 0.18 | 28.5% | 28.5% | 97.8% |
| 128+ | 35,589 | 70.3% | 12.6% | 12.6% | 1.41 | 0.25 | 36.2% | 36.0% | 95.3% |

Largest tile 32768, by how far the chain runs below it

| Chain runs down to | Decisions | Afterstates with an exact read | with a one-over read | one-over but no exact | Exact reads per afterstate | One-over reads per afterstate | Decisions where a move has a one-over read | where moves differ in one-over reads | where moves differ in exact reads |
|---|---|---|---|---|---|---|---|---|---|
| 32,768 | 2,193,796 | 85.7% | 70.6% | 11.4% | 4.21 | 2.74 | 83.2% | 83.0% | 99.4% |

| Largest tile | Move | Strip | 0 above 128 | 1 | 2 | 3+ |
|---|---|---|---|---|---|---|
| ≤4096 | chosen | top two rows | 34.3% | 21.7% | 27.5% | 16.4% |
| ≤4096 | chosen | middle two rows | 50.4% | 12.6% | 36.6% | 0.3% |
| ≤4096 | rejected | top two rows | 30.7% | 26.2% | 28.8% | 14.3% |
| ≤4096 | rejected | middle two rows | 34.5% | 16.1% | 43.5% | 5.9% |
| 8192 | chosen | top two rows | 30.1% | 9.9% | 35.4% | 24.6% |
| 8192 | chosen | middle two rows | 42.6% | 16.0% | 39.4% | 2.1% |
| 8192 | rejected | top two rows | 25.0% | 14.8% | 37.8% | 22.5% |
| 8192 | rejected | middle two rows | 27.6% | 16.2% | 44.8% | 11.4% |
| 16384 | chosen | top two rows | 28.3% | 7.6% | 34.5% | 29.6% |
| 16384 | chosen | middle two rows | 35.6% | 16.8% | 42.2% | 5.4% |
| 16384 | rejected | top two rows | 22.7% | 12.4% | 37.2% | 27.7% |
| 16384 | rejected | middle two rows | 22.9% | 14.9% | 46.0% | 16.3% |
| 32768 | chosen | top two rows | 28.7% | 10.6% | 33.2% | 27.5% |
| 32768 | chosen | middle two rows | 36.7% | 21.3% | 37.5% | 4.5% |
| 32768 | rejected | top two rows | 23.6% | 15.3% | 35.6% | 25.5% |
| 32768 | rejected | middle two rows | 24.0% | 19.8% | 42.4% | 13.8% |

| Largest tile | Lone tile is 256 | 512 | 1024 | 2048+ |
|---|---|---|---|---|
| ≤4096 | 38.6% | 23.2% | 10.5% | 27.8% |
| 8192 | 37.0% | 15.0% | 10.6% | 37.4% |
| 16384 | 37.4% | 16.4% | 10.3% | 35.8% |
| 32768 | 33.5% | 15.1% | 7.5% | 43.9% |
