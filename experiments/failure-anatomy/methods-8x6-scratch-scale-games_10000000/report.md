# Failure anatomy: checkpoints/ntuple/methods-8x6-scratch-scale/games_10000000.npz

10,000,000 games of training; 5,000 greedy games from seed 70,000,000.

## 1. Outcomes

| | Games | Share |
|---|---:|---:|
| Ended before 8192 | 230 | 4.6% |
| Reached 8192 | 4,770 | 95.4% |
| Converted to 16384 | 4,036 | 80.7% (84.6% of those reaching 8192) |
| Ended in the 8192 stage | 734 | 14.7% |

## 2. On the way to 16384

Board features at five points of the 8192 stage (the first board holding the 8192 and the chain named), means for the games that go on to convert and those that don't (converts / fails). Games that never reach a point are left out of its column.

| Feature | first 8192 | +4096 | +4096+2048 | +4096+2048+1024 | +4096+2048+1024+512 |
|---|---:|---:|---:|---:|---:|
| corner | 0.73 / 0.75 | 0.97 / 0.96 | 0.98 / 0.95 | 0.99 / 0.89 | 0.99 / 0.89 |
| chain | 2.50 / 2.55 | 3.01 / 3.03 | 3.48 / 3.43 | 6.02 / 5.27 | 6.28 / 5.34 |
| empty | 7.29 / 7.31 | 6.93 / 6.92 | 6.48 / 6.33 | 5.36 / 5.23 | 4.33 / 4.34 |
| big | 1.00 / 1.00 | 2.00 / 2.00 | 3.00 / 3.00 | 4.00 / 4.02 | 5.00 / 5.00 |
| stranded | 0.27 / 0.25 | 0.70 / 0.73 | 0.77 / 0.84 | 0.04 / 0.55 | 0.09 / 0.81 |
| progress | 50.05 / 50.04 | 75.04 / 75.05 | 88.01 / 88.01 | 94.02 / 94.07 | 97.01 / 97.02 |

| Point | Games reaching it | Of those, convert |
|---|---:|---:|
| first 8192 | 4,770 | 84.6% |
| +4096 | 4,604 | 87.7% |
| +4096+2048 | 4,513 | 89.4% |
| +4096+2048+1024 | 4,420 | 91.3% |
| +4096+2048+1024+512 | 4,350 | 92.8% |

Conversion by feature band, at the first 8192 with a 4096, a 2048 and a 1024:

| corner | games reaching 8192 | convert to 16384 |
|---|---:|---:|
| largest tile in a corner | 4,354 | 92.1% |
| not in a corner | 66 | 36.4% |

| chain | games reaching 8192 | convert to 16384 |
|---|---:|---:|
| 1-2 | 67 | 37.3% |
| 3-4 | 49 | 57.1% |
| 5-6 | 2,855 | 92.5% |
| 7-8 | 1,381 | 92.3% |
| 9+ | 68 | 97.1% |

| empty | games reaching 8192 | convert to 16384 |
|---|---:|---:|
| 0-2 | 34 | 73.5% |
| 3-4 | 975 | 90.1% |
| 5-6 | 2,664 | 92.2% |
| 7+ | 747 | 90.6% |

| stranded | games reaching 8192 | convert to 16384 |
|---|---:|---:|
| 0 | 4,270 | 92.7% |
| 1 | 73 | 64.4% |
| 2 | 3 | 33.3% |
| 3+ | 74 | 37.8% |

## 3. How the 8192-stage failures end

Whether the largest tile left its corner at any point after the first 8192, and if so whether the move was forced:

| | Failures | Share of failures | For comparison: converting games |
|---|---:|---:|---:|
| Corner lost, forced | 120 | 16.3% | 9.7% |
| Corner lost, chosen | 518 | 70.6% | 71.8% |
| Corner kept | 96 | 13.1% | 18.5% |

Where the failures ended (means):

| | Games | Moves after the first 8192 | Corner | Chain | Empty | Big | Stranded | Progress % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Corner lost, forced | 120 | 2813 | 0.32 | 1.61 | 0.00 | 4.79 | 4.02 | 88 |
| Corner lost, chosen | 518 | 2974 | 0.40 | 2.37 | 0.00 | 4.87 | 3.63 | 90 |
| Corner kept | 96 | 2652 | 1.00 | 5.90 | 0.00 | 4.74 | 1.45 | 86 |

Corner lost a median 26 moves after the first 8192, and the game ended a median 2526 moves after that.

## 4. Example failures

Each: the board 100 moves before the end, then the end.

**Corner lost, forced**, game 101 (3716 moves after its first 8192):

```
  8192   4096   2048   1024
    32     64    128    512
    16      8      8      4
     2      4      2      0

     2   4096   2048      2
  8192    128    256      4
    64     16    512   1024
     4      2      8      4
```

**Corner lost, forced**, game 131 (3501 moves after its first 8192):

```
  8192   4096   2048   1024
    64    256     16      8
    16      4      0      0
     8      0      0      2

  8192   4096   2048   1024
    64    256    128     64
     8     32     16      8
     4      2      4      2
```

**Corner lost, forced**, game 159 (3631 moves after its first 8192):

```
     4   4096   2048      2
  8192     64    512   1024
     2     32     32      8
     0      2      4      4

     2   4096   2048      2
     4    256    512      4
  8192     64   1024      8
     2      4     16      4
```

**Corner lost, chosen**, game 9 (3660 moves after its first 8192):

```
  8192   4096   2048   1024
    16     64    256    512
     0      4      4     16
     0      0      2      4

  8192   4096   2048   1024
    64    128    256    512
     8     16     32     64
     4      2      4      2
```

**Corner lost, chosen**, game 15 (3301 moves after its first 8192):

```
     8      2      0      0
    32      4      0      2
    32    128    256    512
  8192   4096   2048      2

     2      4      2      4
     4     32     16      2
    32     64   1024      4
  8192   4096   2048      2
```

**Corner lost, chosen**, game 22 (2413 moves after its first 8192):

```
   256    512      2      4
    64    128   4096   8192
     2      8     32      8
     2      8      2      2

  8192   1024   4096      2
     2      4    128     16
    16      2     32      4
     2      8      4      2
```

**Corner kept**, game 17 (3721 moves after its first 8192):

```
     2      4      2      0
     4      2      8      8
    16    512    256      8
  1024   2048   4096   8192

     4      2      4      2
     8     16     32     64
    16    512    256    128
  1024   2048   4096   8192
```

**Corner kept**, game 52 (2798 moves after its first 8192):

```
   512    256      8      2
  1024     32      8      4
  4096      8      4      0
  8192      2      8      2

   512    256      8      4
  1024    128     32      2
     2   4096     64     16
  8192     32      8      4
```

**Corner kept**, game 66 (1507 moves after its first 8192):

```
  8192   2048    512    256
     8     32     64    128
     2     16     16      4
     0      0      2      4

  8192   2048   1024     64
     8     32     64     16
    32     16      8      4
     4      2      4      2
```

