# Failure anatomy: checkpoints/ntuple/methods-8x6-best-long-tc0.005/games_10000000.npz

10,000,000 games of training; 5,000 greedy games from seed 70,000,000.

## 1. Outcomes

| | Games | Share |
|---|---:|---:|
| Ended before 8192 | 197 | 3.9% |
| Reached 8192 | 4,803 | 96.1% |
| Converted to 16384 | 3,996 | 79.9% (83.2% of those reaching 8192) |
| Ended in the 8192 stage | 807 | 16.1% |

## 2. On the way to 16384

Board features at five points of the 8192 stage (the first board holding the 8192 and the chain named), means for the games that go on to convert and those that don't (converts / fails). Games that never reach a point are left out of its column.

| Feature | first 8192 | +4096 | +4096+2048 | +4096+2048+1024 | +4096+2048+1024+512 |
|---|---:|---:|---:|---:|---:|
| corner | 0.94 / 0.94 | 0.97 / 0.95 | 0.99 / 0.95 | 1.00 / 0.92 | 1.00 / 0.92 |
| chain | 3.20 / 3.17 | 3.23 / 3.09 | 3.52 / 3.37 | 6.03 / 5.41 | 6.27 / 5.70 |
| empty | 7.52 / 7.49 | 6.73 / 6.63 | 6.19 / 6.09 | 5.38 / 5.33 | 4.15 / 4.12 |
| big | 1.00 / 1.00 | 2.00 / 2.00 | 3.00 / 3.01 | 4.01 / 4.01 | 5.00 / 5.00 |
| stranded | 0.06 / 0.07 | 0.68 / 0.71 | 0.90 / 0.93 | 0.04 / 0.47 | 0.08 / 0.54 |
| progress | 50.03 / 50.04 | 75.05 / 75.05 | 88.01 / 88.04 | 94.02 / 94.06 | 97.01 / 97.01 |

| Point | Games reaching it | Of those, convert |
|---|---:|---:|
| first 8192 | 4,803 | 83.2% |
| +4096 | 4,649 | 86.0% |
| +4096+2048 | 4,560 | 87.6% |
| +4096+2048+1024 | 4,466 | 89.5% |
| +4096+2048+1024+512 | 4,374 | 91.4% |

Conversion by feature band, at the first 8192 with a 4096, a 2048 and a 1024:

| corner | games reaching 8192 | convert to 16384 |
|---|---:|---:|
| largest tile in a corner | 4,418 | 90.2% |
| not in a corner | 48 | 18.8% |

| chain | games reaching 8192 | convert to 16384 |
|---|---:|---:|
| 1-2 | 70 | 35.7% |
| 3-4 | 68 | 69.1% |
| 5-6 | 2,821 | 90.5% |
| 7-8 | 1,431 | 90.8% |
| 9+ | 76 | 94.7% |

| empty | games reaching 8192 | convert to 16384 |
|---|---:|---:|
| 0-2 | 9 | 100.0% |
| 3-4 | 945 | 88.6% |
| 5-6 | 2,839 | 89.9% |
| 7+ | 673 | 88.9% |

| stranded | games reaching 8192 | convert to 16384 |
|---|---:|---:|
| 0 | 4,325 | 90.8% |
| 1 | 33 | 72.7% |
| 2 | 21 | 52.4% |
| 3+ | 87 | 41.4% |

## 3. How the 8192-stage failures end

Whether the largest tile left its corner at any point after the first 8192, and if so whether the move was forced:

| | Failures | Share of failures | For comparison: converting games |
|---|---:|---:|---:|
| Corner lost, forced | 229 | 28.4% | 12.9% |
| Corner lost, chosen | 363 | 45.0% | 52.1% |
| Corner kept | 215 | 26.6% | 35.0% |

Where the failures ended (means):

| | Games | Moves after the first 8192 | Corner | Chain | Empty | Big | Stranded | Progress % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Corner lost, forced | 229 | 2880 | 0.41 | 2.07 | 0.00 | 4.68 | 3.58 | 89 |
| Corner lost, chosen | 363 | 3155 | 0.50 | 3.43 | 0.00 | 5.10 | 3.11 | 93 |
| Corner kept | 215 | 2852 | 1.00 | 7.09 | 0.00 | 4.99 | 0.75 | 89 |

Corner lost a median 1395 moves after the first 8192, and the game ended a median 1392 moves after that.

## 4. Example failures

Each: the board 100 moves before the end, then the end.

**Corner lost, forced**, game 24 (3198 moves after its first 8192):

```
     0      4      4      8
     2     32      4      8
  8192    128     32    512
     4   4096   2048      4

     4   8192      4      2
     8      4    128      8
  2048    256    512     16
     2   4096      2      8
```

**Corner lost, forced**, game 90 (3390 moves after its first 8192):

```
     2   4096      2     16
     4      2      8   1024
     0      0     16   2048
     0      2   8192      2

     2   2048      4      8
     4   1024     16     64
     8     32   4096    128
     2      4   8192      4
```

**Corner lost, forced**, game 215 (3602 moves after its first 8192):

```
  8192   4096   2048   1024
     4      8     32    512
     0      0      4     16
     2      0      2      0

  8192   4096   2048   1024
    16     64    128    512
     8     32     16      8
     4      2      4      2
```

**Corner lost, chosen**, game 10 (1452 moves after its first 8192):

```
  8192    256   2048      2
     2     16    128    512
     2      4     32      0
     8      2      0      0

  8192   1024   2048      8
    64     32     16      2
    16      8      2      4
     2      4      8      2
```

**Corner lost, chosen**, game 46 (3286 moves after its first 8192):

```
   512   2048   4096   8192
     4     32    256     64
     2      4     16      4
     0      2      0      2

     2      4   2048      2
     4   1024   8192   4096
     2      4     16      2
    32      2      8      4
```

**Corner lost, chosen**, game 53 (3695 moves after its first 8192):

```
  1024   2048   4096   8192
   512    256     16      8
     0      2     16      2
     2      0      2      4

  1024   2048   4096      2
   512    256    128   8192
    16     32     16     64
     8      4      2      8
```

**Corner kept**, game 1 (3656 moves after its first 8192):

```
  1024    512      2      0
  2048    128      2      0
  4096     32      8      0
  8192      4      2      2

  1024    512      2      4
  2048    256     32     16
  4096     64      8      4
  8192     16      4      2
```

**Corner kept**, game 3 (3387 moves after its first 8192):

```
     0      0      0   8192
     0      0      2   4096
     2      8      8   2048
     2      8     32   1024

     2      4      8   8192
     8     16     32   4096
     2      4     64   2048
     4     16    128   1024
```

**Corner kept**, game 8 (3672 moves after its first 8192):

```
     2      0      4      2
     2      4      8      2
   512    128     32      8
  1024   2048   4096   8192

     4      2      4      2
     8     16     32      8
   512    256     64     32
  1024   2048   4096   8192
```

