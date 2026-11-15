# Normalisation events replayed from checkpoints/ntuple/stage8-tidbd/arm-F/games_1000000.npz (1,000,000 games)

| Replay | Games | Moves | Normalisations | per 10k games | per 1M moves | Shift at start | Shift at end | Σ log M of recorded events |
|---|---|---|---|---|---|---|---|---|
| 8 threads | 40,000 | 368M | 77 | 19.2 | 0.21 | -1.4213 | -1.9364 | 0.5151 of 0.5151 |
| 1 thread | 4,000 | 37M | 6 | 15.0 | 0.16 | -1.4213 | -1.4678 | 0.0465 of 0.0465 |

Events per 10k games through the 8-thread replay: 11, 20, 15, 31

M over the 77 recorded events (8 threads): p10 1.0010, p50 1.0054, p90 1.0140, p99 1.0254, max 1.0265; mean log M 0.0067

Events from games started at a synthetic late position: 22.1% (20% of games are). Median move number within the game: 12,943.
Weights with a non-zero contribution per event: median 48, largest 48. The largest single contribution is 92% of the positive part of the sum at the median event, and exceeds 1 by itself in 0.0% of events.

60 distinct weights appear among the top 6 positive contributors. Their share of the summed log M (0.5151), each event's split among its contributors by size:

| Weight | Reads | Share of Σ log M | Cumulative | Events | Largest in | Median step size | Median contribution | x → x' (median) | z (median) | Read more than once | Updates |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 25799629 | tuple 1 cells [4, 5, 6, 7, 8, 9]: 8192 4096 2048 1024 512 256 | 90.9% | 90.9% | 77 | 77 | 0.924 | 0.924 | 1 → 0 | 1 | 0% | 29,364 |
| 94996321 | tuple 5 cells [3, 4, 5, 6, 7, 8]: 2 64 128 256 512 1024 | 0.3% | 91.2% | 26 | 0 | 0.00681 | 0.00681 | 1 → 0 | 1 | 0% | 4,027 |
| 29005942 | tuple 1 cells [4, 5, 6, 7, 8, 9]: 64 128 256 512 1024 2048 | 0.3% | 91.5% | 30 | 0 | 0.00519 | 0.00519 | 1 → 0 | 1 | 0% | 6,340 |
| 126615712 | tuple 7 cells [0, 1, 4, 8, 9, 10]: · 1024 · · 4096 256 | 0.3% | 91.8% | 53 | 0 | 0.00485 | 0.00485 | 1 → 0 | 1 | 0% | 21,687 |
| 29005941 | tuple 1 cells [4, 5, 6, 7, 8, 9]: 32 128 256 512 1024 2048 | 0.3% | 92.1% | 30 | 0 | 0.0108 | 0.0108 | 1 → 0 | 1 | 0% | 7,704 |
| 62439465 | tuple 3 cells [2, 3, 4, 5, 6, 9]: 512 4 · 4096 256 2048 | 0.3% | 92.4% | 26 | 0 | 0.00822 | 0.00822 | 1 → 0 | 1 | 0% | 9,866 |
| 84052993 | tuple 5 cells [3, 4, 5, 6, 7, 8]: 2 · 4096 256 4 · | 0.3% | 92.7% | 31 | 0 | 0.00688 | 0.00688 | 1 → 0 | 1 | 0% | 13,812 |
| 97941649 | tuple 5 cells [3, 4, 5, 6, 7, 8]: 2 512 256 128 64 8192 | 0.2% | 92.9% | 18 | 0 | 0.0049 | 0.0049 | 1 → 0 | 1 | 0% | 2,585 |
| 84114177 | tuple 5 cells [3, 4, 5, 6, 7, 8]: 2 · 2048 128 8 · | 0.2% | 93.0% | 18 | 0 | 0.00656 | 0.00656 | 1 → 0 | 1 | 0% | 29,018 |
| 107448594 | tuple 6 cells [1, 3, 4, 5, 6, 7]: 4 2 512 256 128 64 | 0.1% | 93.1% | 8 | 0 | 0.00436 | 0.00436 | 1 → 0 | 1 | 0% | 5,703 |
| 25299828 | tuple 1 cells [4, 5, 6, 7, 8, 9]: 16 128 2048 · 4 256 | 0.1% | 93.2% | 9 | 0 | 0.00473 | 0.00473 | 1 → 0 | 1 | 0% | 2,618 |
| 25365364 | tuple 1 cells [4, 5, 6, 7, 8, 9]: 16 128 2048 · 8 256 | 0.1% | 93.2% | 4 | 0 | 0.00722 | 0.00722 | 1 → 0 | 1 | 0% | 2,258 |
| 84048641 | tuple 5 cells [3, 4, 5, 6, 7, 8]: 2 · 2048 128 4 · | 0.1% | 93.3% | 6 | 0 | 0.00684 | 0.00684 | 1 → 0 | 1 | 0% | 60,940 |
| 107448627 | tuple 6 cells [1, 3, 4, 5, 6, 7]: 8 8 512 256 128 64 | 0.0% | 93.3% | 3 | 0 | 0.00717 | 0.00717 | 1 → 0 | 1 | 0% | 1,288 |
| 83983106 | tuple 5 cells [3, 4, 5, 6, 7, 8]: 4 · 2048 128 2 · | 0.0% | 93.4% | 3 | 0 | 0.017 | 0.017 | 1 → 0 | 1 | 0% | 37,663 |

Weights needed to account for a share of the summed log M: 50%: 1, 80%: 1, 90%: 1, 99%: 61

## Sample events

- Batch from game 2,000, move 14,414: M 1.0046, shift -1.4213 → -1.4259, positive 1.0046, negative 0.0000
  - afterstate updated: 4 64 2048 · / 32 128 4096 · / 4 256 8192 · / 2 512 16384 2
  - successor afterstate: 4 64 2048 2 / · 32 128 4096 / · 4 256 8192 / 2 512 16384 2
  - +0.8857 from tuple 1 cells [4, 5, 6, 7, 8, 9]: 8192 4096 2048 1024 512 256 (step size 0.886, x 1 → x' 0, z 1)
  - +0.0090 from tuple 5 cells [3, 4, 5, 6, 7, 8]: 2 64 128 256 512 1024 (step size 0.00895, x 1 → x' 0, z 1)
  - +0.0077 from tuple 5 cells [3, 4, 5, 6, 7, 8]: 2 · 4096 256 4 · (step size 0.00772, x 1 → x' 0, z 1)
- Batch from game 11,000, move 13,890: M 1.0027, shift -1.5424 → -1.5452, positive 1.0027, negative 0.0000
  - afterstate updated: 2 16384 512 4 / · 8192 256 2 / · 4096 64 4 / · 1024 16 2
  - successor afterstate: 2 16384 512 4 / 8192 256 2 · / 4096 64 4 · / 2 1024 16 2
  - +0.9075 from tuple 1 cells [4, 5, 6, 7, 8, 9]: 8192 4096 2048 1024 512 256 (step size 0.907, x 1 → x' 0, z 1)
  - +0.0112 from tuple 5 cells [3, 4, 5, 6, 7, 8]: 2 · 2048 64 4 · (step size 0.0112, x 1 → x' 0, z 1)
  - +0.0062 from tuple 3 cells [2, 3, 4, 5, 6, 9]: 512 4 · 4096 256 2048 (step size 0.00625, x 1 → x' 0, z 1)
- Batch from game 19,000, move 13,885: M 1.0119, shift -1.6507 → -1.6626, positive 1.0119, negative 0.0000
  - afterstate updated: 4 16 4 2 / 32 128 256 512 / 1024 4096 8192 16384 / · · · 2
  - successor afterstate: · · 4 2 / 4 16 256 512 / 32 128 8192 16384 / 1024 4096 2 2
  - +0.9277 from tuple 1 cells [4, 5, 6, 7, 8, 9]: 8192 4096 2048 1024 512 256 (step size 0.928, x 1 → x' 0, z 1)
  - +0.0110 from tuple 1 cells [4, 5, 6, 7, 8, 9]: 32 128 256 512 1024 2048 (step size 0.011, x 1 → x' 0, z 1)
  - +0.0081 from tuple 5 cells [3, 4, 5, 6, 7, 8]: 4 · 2048 128 16 · (step size 0.0081, x 1 → x' 0, z 1)
- Batch from game 29,000, move 6,924: M 1.0058, shift -1.7612 → -1.7669, positive 1.0058, negative 0.0000
  - afterstate updated: 2 32 2048 · / 4 64 4096 · / 2 256 8192 · / 4 512 16384 4
  - successor afterstate: 2 32 2048 2 / · 4 64 4096 / · 2 256 8192 / 4 512 16384 4
  - +0.9338 from tuple 1 cells [4, 5, 6, 7, 8, 9]: 8192 4096 2048 1024 512 256 (step size 0.934, x 1 → x' 0, z 1)
  - +0.0087 from tuple 3 cells [2, 3, 4, 5, 6, 9]: 512 4 · 4096 256 2048 (step size 0.00871, x 1 → x' 0, z 1)
  - +0.0079 from tuple 5 cells [3, 4, 5, 6, 7, 8]: 2 · 2048 64 4 · (step size 0.00791, x 1 → x' 0, z 1)
- Batch from game 32,000, move 13,008: M 1.0068, shift -1.8517 → -1.8584, positive 1.0068, negative 0.0000
  - afterstate updated: 4 16384 512 2 / · 8192 256 4 / · 2048 128 8 / · 1024 64 2
  - successor afterstate: 4 16384 512 2 / 8192 256 4 · / 2 2048 128 8 / 1024 64 2 ·
  - +0.9268 from tuple 1 cells [4, 5, 6, 7, 8, 9]: 8192 4096 2048 1024 512 256 (step size 0.927, x 1 → x' 0, z 1)
  - +0.0065 from tuple 5 cells [3, 4, 5, 6, 7, 8]: 2 · 2048 128 8 · (step size 0.00652, x 1 → x' 0, z 1)
  - +0.0055 from tuple 5 cells [3, 4, 5, 6, 7, 8]: 2 64 128 256 512 1024 (step size 0.0055, x 1 → x' 0, z 1)
- Batch from game 39,000, move 13,961: M 1.0059, shift -1.9305 → -1.9364, positive 1.0059, negative 0.0000
  - afterstate updated: · 1024 32 2 / · 4096 128 8 / · 8192 256 4 / 2 16384 512 2
  - successor afterstate: 2 1024 32 2 / 4096 128 8 · / 8192 256 4 · / 2 16384 512 2
  - +0.9399 from tuple 1 cells [4, 5, 6, 7, 8, 9]: 8192 4096 2048 1024 512 256 (step size 0.94, x 1 → x' 0, z 1)
  - +0.0095 from tuple 1 cells [4, 5, 6, 7, 8, 9]: 32 128 256 512 1024 2048 (step size 0.00948, x 1 → x' 0, z 1)
  - +0.0060 from tuple 5 cells [3, 4, 5, 6, 7, 8]: 2 · 2048 128 8 · (step size 0.00604, x 1 → x' 0, z 1)
