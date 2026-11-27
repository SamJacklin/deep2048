# Arrival test

Source: checkpoints/ntuple/methods-8x6-scratch-stall-small/stall_7000000.npz. Arrival: the first board holding the 4 tiles from 16,384 down, with nothing larger. Intervals are 95%, from the spread between positions.

## Continued by greedy play (32 continuations per position)

**Placement at arrival.**

| Arrived from | Placement | Positions | Share | Makes 32768 |
|---|---|---:|---:|---:|
| greedy games | all | 2,000 | 100% | 39.1% ± 0.6% |
| greedy games | in order along one edge | 1,811 | 91% | 42.7% ± 0.4% |
| greedy games | largest in a corner, chain out of order | 111 | 6% | 5.5% ± 1.8% |
| greedy games | largest off its corner | 78 | 4% | 3.0% ± 1.2% |

**Do well-placed positions differ in their chances?** Positions in order along one edge, ranked by conversion on their even-numbered continuations and measured on the odd-numbered ones.

| Fifth of positions, by the ranking half | Ranking half | Measured half |
|---|---:|---:|
| lowest | 26.0% | 43.5% |
| second | 36.0% | 42.9% |
| middle | 42.6% | 41.1% |
| fourth | 49.6% | 43.1% |
| highest | 60.9% | 41.8% |

Spread of positions' true chances, after removing the luck of 32 continuations: standard deviation 0.0% around 42.7%.

| Empty cells at arrival | Positions | Makes 32768 |
|---:|---:|---:|
| 2 | 6 | 42.2% ± 7.9% |
| 3 | 59 | 41.8% ± 2.5% |
| 4 | 366 | 43.2% ± 0.9% |
| 5 | 679 | 42.9% ± 0.6% |
| 6 | 488 | 42.7% ± 0.7% |
| 7 | 170 | 42.5% ± 1.4% |
| 8 | 43 | 39.8% ± 2.9% |

**How far continuations from well-placed positions got** (57,952 continuations).

| Chain ran to at least | Share |
|---|---:|
| 1024 | 92.9% |
| 512 | 82.8% |
| 256 | 74.9% |
| 128 | 67.1% |
| a 32768 | 42.7% |

**How they failed** (33,183 failed continuations from well-placed positions).

| | Share of failures |
|---|---:|
| The formation broke before the end | 70% |
| ...by a forced move (no legal move kept it) | 7% |
| ...by a chosen move (another legal move kept it) | 62% |
| Ended with the formation never broken | 30% |

Of continuations whose formation broke, 42% still made the 32768; of those where it never broke, 45% did. So a break doesn't by itself decide the outcome: games that convert break formation too, on the way into their last cascade.

**Was a chosen break a mistake?** Not necessarily. In 50% of chosen breaks in failures, even the safest move that kept the formation would more likely than not have ended the game on the spot (no legal move after the spawn); in 48% it couldn't have.

At the break, in failures (medians): 1 empty cells on the board the move was made from; 23 moves from the break to the end; 802 moves from arrival to the break; 0 decisions in a row before it with at most one safe move.

| Chain ran to, at the break | Share of breaks in failures | Forced |
|---|---:|---:|
| 2048 | 23% | 18% |
| 1024 | 20% | 12% |
| 512 | 15% | 8% |
| 256 | 20% | 11% |
| 128 or beyond | 22% | 4% |

Five boards a breaking move was made from (first continuations of the first positions):

- ` 2048    64     4     4 /  4096  1024     8     4 /  8192   512    16     2 / 16384   128    32     2`
- ` 2048    64     2     0 /  4096   512     8     2 /  8192   256    16     8 / 16384   128    32     2`
- ` 2048   512     2     0 /  4096   256     4     0 /  8192    64     8     0 / 16384    32     2     0`
- ` 2048  1024     4     0 /  4096   128     8     0 /  8192    32     4     0 / 16384     8     2     0`
- ` 2048   512     4     0 /  4096   256     8     0 /  8192    32     4     0 / 16384     4     2     0`

