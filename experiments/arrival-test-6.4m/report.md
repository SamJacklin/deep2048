# Arrival test

Source: checkpoints/ntuple/methods-8x6-second-reset/stall_6400000.npz. Arrival: the first board holding the 4 tiles from 16,384 down, with nothing larger. Intervals are 95%, from the spread between positions.

## Continued by greedy play (32 continuations per position)

**Placement at arrival.**

| Arrived from | Placement | Positions | Share | Makes 32768 |
|---|---|---:|---:|---:|
| greedy games | all | 2,000 | 100% | 26.6% ± 0.5% |
| greedy games | in order along one edge | 1,792 | 90% | 29.4% ± 0.4% |
| greedy games | largest in a corner, chain out of order | 132 | 7% | 2.7% ± 1.0% |
| greedy games | largest off its corner | 76 | 4% | 1.7% ± 1.0% |

**Do well-placed positions differ in their chances?** Positions in order along one edge, ranked by conversion on their even-numbered continuations and measured on the odd-numbered ones.

| Fifth of positions, by the ranking half | Ranking half | Measured half |
|---|---:|---:|
| lowest | 14.2% | 30.7% |
| second | 23.3% | 29.3% |
| middle | 28.8% | 29.3% |
| fourth | 34.7% | 29.4% |
| highest | 46.1% | 28.1% |

Spread of positions' true chances, after removing the luck of 32 continuations: standard deviation 0.0% around 29.4%.

| Empty cells at arrival | Positions | Makes 32768 |
|---:|---:|---:|
| 2 | 4 | 32.8% ± 9.5% |
| 3 | 77 | 28.5% ± 1.9% |
| 4 | 305 | 29.2% ± 0.9% |
| 5 | 695 | 29.4% ± 0.6% |
| 6 | 520 | 29.4% ± 0.7% |
| 7 | 167 | 29.9% ± 1.2% |
| 8 | 23 | 29.5% ± 3.3% |

**How far continuations from well-placed positions got** (57,344 continuations).

| Chain ran to at least | Share |
|---|---:|
| 1024 | 91.1% |
| 512 | 78.7% |
| 256 | 69.3% |
| 128 | 60.6% |
| a 32768 | 29.4% |

**How they failed** (40,495 failed continuations from well-placed positions).

| | Share of failures |
|---|---:|
| The formation broke before the end | 64% |
| ...by a forced move (no legal move kept it) | 7% |
| ...by a chosen move (another legal move kept it) | 57% |
| Ended with the formation never broken | 36% |

Of continuations whose formation broke, 35% still made the 32768; of those where it never broke, 16% did. So a break doesn't by itself decide the outcome: games that convert break formation too, on the way into their last cascade.

**Was a chosen break a mistake?** Not necessarily. In 45% of chosen breaks in failures, even the safest move that kept the formation would more likely than not have ended the game on the spot (no legal move after the spawn); in 54% it couldn't have.

At the break, in failures (medians): 1 empty cells on the board the move was made from; 26 moves from the break to the end; 801 moves from arrival to the break; 0 decisions in a row before it with at most one safe move.

| Chain ran to, at the break | Share of breaks in failures | Forced |
|---|---:|---:|
| 2048 | 23% | 21% |
| 1024 | 20% | 14% |
| 512 | 14% | 10% |
| 256 | 20% | 9% |
| 128 or beyond | 23% | 3% |

Five boards a breaking move was made from (first continuations of the first positions):

- `    0     4     2     2 /     4     8    16    32 /    64  1024   512   256 /  2048  4096  8192 16384`
- `    2     2    16     0 /     8     8    32     2 /  1024   512   256    32 /  2048  4096  8192 16384`
- `    4     4     2     2 /     4     8    16    32 /   128  1024   512   256 /  2048  4096  8192 16384`
- `    2     4     2     4 /     2     8    16    32 /  1024   512   256    64 /  2048  4096  8192 16384`
- `    2     4     4     0 /     8    16    64    32 /  1024   512   128    64 /  2048  4096  8192 16384`

