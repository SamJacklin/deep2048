# Arrival test

Source: checkpoints/ntuple/decision-8x6-tc-reset/games_2000000.npz. Arrival: the first board holding the 4 tiles from 16,384 down, with nothing larger. Intervals are 95%, from the spread between positions.

## Continued by greedy play (32 continuations per position)

**Placement at arrival.**

| Arrived from | Placement | Positions | Share | Makes 32768 |
|---|---|---:|---:|---:|
| greedy games | all | 2,000 | 100% | 21.4% ± 0.5% |
| greedy games | in order along one edge | 1,731 | 87% | 24.4% ± 0.4% |
| greedy games | largest in a corner, chain out of order | 172 | 9% | 2.9% ± 0.9% |
| greedy games | largest off its corner | 97 | 5% | 1.5% ± 0.5% |
| depth3 games | all | 500 | 100% | 19.5% ± 1.0% |
| depth3 games | in order along one edge | 376 | 75% | 24.1% ± 0.8% |
| depth3 games | largest in a corner, chain out of order | 86 | 17% | 7.1% ± 1.7% |
| depth3 games | largest off its corner | 38 | 8% | 2.4% ± 1.4% |

**Do well-placed positions differ in their chances?** Positions in order along one edge, ranked by conversion on their even-numbered continuations and measured on the odd-numbered ones.

| Fifth of positions, by the ranking half | Ranking half | Measured half |
|---|---:|---:|
| lowest | 9.6% | 24.5% |
| second | 18.1% | 23.8% |
| middle | 24.0% | 24.6% |
| fourth | 29.7% | 24.9% |
| highest | 39.6% | 24.9% |

Spread of positions' true chances, after removing the luck of 32 continuations: standard deviation 1.7% around 24.4%.

| Empty cells at arrival | Positions | Makes 32768 |
|---:|---:|---:|
| 1 | 2 | 23.4% ± 9.2% |
| 2 | 9 | 20.5% ± 4.7% |
| 3 | 57 | 25.3% ± 2.2% |
| 4 | 351 | 24.6% ± 0.8% |
| 5 | 748 | 24.2% ± 0.6% |
| 6 | 665 | 24.7% ± 0.6% |
| 7 | 234 | 23.7% ± 0.9% |
| 8 | 38 | 23.2% ± 2.6% |
| 9 | 3 | 18.8% ± 9.4% |

**How far continuations from well-placed positions got** (67,424 continuations).

| Chain ran to at least | Share |
|---|---:|
| 1024 | 90.7% |
| 512 | 77.5% |
| 256 | 66.8% |
| 128 | 58.6% |
| a 32768 | 24.4% |

**How they failed** (51,004 failed continuations from well-placed positions).

| | Share of failures |
|---|---:|
| The formation broke before the end | 64% |
| ...by a forced move (no legal move kept it) | 8% |
| ...by a chosen move (another legal move kept it) | 56% |
| Ended with the formation never broken | 36% |

Of continuations whose formation broke, 25% still made the 32768; of those where it never broke, 23% did. So a break doesn't by itself decide the outcome: games that convert break formation too, on the way into their last cascade.

**Was a chosen break a mistake?** Not necessarily. In 46% of chosen breaks in failures, even the safest move that kept the formation would more likely than not have ended the game on the spot (no legal move after the spawn); in 53% it couldn't have.

At the break, in failures (medians): 1 empty cells on the board the move was made from; 26 moves from the break to the end; 799 moves from arrival to the break; 0 decisions in a row before it with at most one safe move.

| Chain ran to, at the break | Share of breaks in failures | Forced |
|---|---:|---:|
| 2048 | 23% | 23% |
| 1024 | 21% | 15% |
| 512 | 14% | 12% |
| 256 | 18% | 10% |
| 128 or beyond | 24% | 3% |

Five boards a breaking move was made from (first continuations of the first positions):

- `    0     2     4     8 /     2     8    32     4 /    32   128   256   512 / 16384  8192  4096  2048`
- `    2     4     2     4 /     2     8    32    16 /    16    64  1024   128 / 16384  8192  4096  2048`
- `    2     2     4     2 /    32    16     8     2 /    64   128   256  1024 / 16384  8192  4096  2048`
- `    2     2     4     0 /    32    16     8     8 /   256   512  1024    64 / 16384  8192  4096  2048`
- `    4     2     4     2 /    64    16     8     2 /   128   256   512  1024 / 16384  8192  4096  2048`

## Continued by depth-3 search (8 continuations per position)

**Placement at arrival.**

| Arrived from | Placement | Positions | Share | Makes 32768 |
|---|---|---:|---:|---:|
| greedy games | all | 500 | 100% | 24.1% ± 1.4% |
| greedy games | in order along one edge | 428 | 86% | 26.3% ± 1.4% |
| greedy games | largest in a corner, chain out of order | 47 | 9% | 11.7% ± 4.7% |
| greedy games | largest off its corner | 25 | 5% | 10.5% ± 4.4% |
| depth3 games | all | 500 | 100% | 23.5% ± 1.5% |
| depth3 games | in order along one edge | 376 | 75% | 26.8% ± 1.6% |
| depth3 games | largest in a corner, chain out of order | 86 | 17% | 16.4% ± 4.2% |
| depth3 games | largest off its corner | 38 | 8% | 7.2% ± 3.5% |

**Do well-placed positions differ in their chances?** Positions in order along one edge, ranked by conversion on their even-numbered continuations and measured on the odd-numbered ones.

| Fifth of positions, by the ranking half | Ranking half | Measured half |
|---|---:|---:|
| lowest | 0.0% | 26.1% |
| second | 13.5% | 26.9% |
| middle | 25.0% | 25.5% |
| fourth | 38.8% | 24.2% |
| highest | 58.3% | 27.3% |

Spread of positions' true chances, after removing the luck of 8 continuations: standard deviation 0.0% around 26.5%.

| Empty cells at arrival | Positions | Makes 32768 |
|---:|---:|---:|
| 1 | 2 | 43.8% ± 12.2% |
| 2 | 2 | 18.8% ± 12.2% |
| 3 | 29 | 25.4% ± 4.9% |
| 4 | 126 | 26.5% ± 2.8% |
| 5 | 252 | 27.8% ± 2.0% |
| 6 | 266 | 26.1% ± 1.8% |
| 7 | 111 | 24.7% ± 2.7% |
| 8 | 16 | 27.3% ± 6.8% |

**How far continuations from well-placed positions got** (6,432 continuations).

| Chain ran to at least | Share |
|---|---:|
| 1024 | 94.0% |
| 512 | 81.1% |
| 256 | 64.3% |
| 128 | 54.4% |
| a 32768 | 26.5% |

**How they failed** (4,725 failed continuations from well-placed positions).

| | Share of failures |
|---|---:|
| The formation broke before the end | 70% |
| ...by a forced move (no legal move kept it) | 2% |
| ...by a chosen move (another legal move kept it) | 68% |
| Ended with the formation never broken | 30% |

Of continuations whose formation broke, 30% still made the 32768; of those where it never broke, 16% did. So a break doesn't by itself decide the outcome: games that convert break formation too, on the way into their last cascade.

**Was a chosen break a mistake?** Not necessarily. In 19% of chosen breaks in failures, even the safest move that kept the formation would more likely than not have ended the game on the spot (no legal move after the spawn); in 80% it couldn't have.

At the break, in failures (medians): 1 empty cells on the board the move was made from; 35 moves from the break to the end; 795 moves from arrival to the break; 0 decisions in a row before it with at most one safe move.

| Chain ran to, at the break | Share of breaks in failures | Forced |
|---|---:|---:|
| 2048 | 23% | 5% |
| 1024 | 22% | 3% |
| 512 | 22% | 2% |
| 256 | 20% | 3% |
| 128 or beyond | 12% | 1% |

Five boards a breaking move was made from (first continuations of the first positions):

- `    4     2     2     0 /    32    16     8     4 /   128   256  1024    64 / 16384  8192  4096  2048`
- `    0     2     4     4 /    32    64    16     8 /    64   128   256  1024 / 16384  8192  4096  2048`
- `    4     4     2     0 /    32    16     8     2 /    64   256   512  1024 / 16384  8192  4096  2048`
- `    2     2     4     4 /    32    16     8     4 /   128   256   512    64 / 16384  8192  4096  2048`
- `    2     4     8     2 /     2     8    16    64 /    32   256   512  1024 / 16384  8192  4096  2048`

## How late was the outcome settled? (2,000 greedy games from well-placed positions, 32 fresh continuations per board)

The chance of making the 32768 from the board a game decided from, some moves before its stage ended. 23.8% of these games made it.

| Moves before the end | Games that failed | Games that made the 32768 |
|---:|---:|---:|
| 1 | 4.9% (1,524) | 100.0% (476) |
| 5 | 17.2% (1,524) | 100.0% (476) |
| 10 | 18.9% (1,524) | 100.0% (476) |
| 25 | 22.7% (1,524) | 64.8% (476) |
| 50 | 26.5% (1,523) | 49.7% (476) |
| 100 | 28.4% (1,521) | 42.2% (476) |
| 200 | 28.5% (1,513) | 35.9% (476) |

Board counts are in brackets: stages shorter than the offset have no board there.

## Can looking further ahead finish the last stretch? (1,000 boards, 8 continuations each)

Boards are the first whose tiles sum to within 64 of 32768, from greedy continuations of well-placed positions; 57.3% of those continuations got that far. Every policy plays on from the same boards with the same spawns.

| Played on by | Makes 32768 | vs greedy play (paired) | Seconds for all continuations |
|---|---:|---:|---:|
| greedy | 43.6% ± 1.5% | — | 2 |
| depth 3 | 52.4% ± 1.7% | +8.8% ± 1.4% | 16 |
| depth 5, cutoff 0.02 | 65.8% ± 1.7% | +22.2% ± 1.4% | 94 |
| depth 8, cutoff 0.02 | 71.3% ± 1.7% | +27.7% ± 1.4% | 277 |

