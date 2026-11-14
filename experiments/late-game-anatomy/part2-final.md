# Stage 7, Part 2: when can failing games be rescued? (final)

Generated 2026-11-14 21:52.

Policies: **A** greedy; **B** depth 3 from here; **C** depth 3 for 200 moves, then greedy; **D** pruned depth 5 (cutoff 0.01) for 200 moves, then greedy.

Each cell is the mean paired difference from greedy (A) over points, then continuations, with a 95% interval from resampling whole trajectories. Moves remaining are retrospective (the agent can't know them).

Trajectories: 40 that made 32768 (sampled up to it) and 40 that didn't. Points: 2,345. Continuations: 300,160.


## B: depth 3 from here


**By moves since the first 16384, failed trajectories**

| Group | Points | Trajectories | Greedy makes 32768 | Δ makes 32768 [95%] | Δ score added [95%] | CPU s per continuation | Chance nodes per continuation |
|---|---:|---:|---:|---:|---:|---:|---:|
| at the first 16384 | 40 | 40 | 12.0% | +7.6% [+5.0%, +10.1%] | +38,787 [+32,800, +44,828] | 3.04 | 456,443 |
| 1–500 moves after | 120 | 40 | 11.3% | +8.6% [+7.0%, +10.3%] | +41,509 [+38,158, +44,892] | 3.04 | 438,747 |
| 501–1,500 | 160 | 40 | 11.6% | +7.1% [+5.6%, +8.5%] | +38,640 [+34,339, +42,850] | 3.04 | 380,801 |
| 1,501–3,000 | 241 | 40 | 11.4% | +7.9% [+6.9%, +8.8%] | +38,290 [+35,470, +41,204] | 3.05 | 306,757 |
| over 3,000 | 544 | 38 | 15.4% | +4.7% [+3.8%, +5.6%] | +18,909 [+16,829, +21,290] | 2.87 | 166,483 |

**By moves since the first 16384, successful trajectories**

| Group | Points | Trajectories | Greedy makes 32768 | Δ makes 32768 [95%] | Δ score added [95%] | CPU s per continuation | Chance nodes per continuation |
|---|---:|---:|---:|---:|---:|---:|---:|
| at the first 16384 | 40 | 40 | 10.4% | +9.4% [+6.7%, +11.9%] | +46,525 [+40,411, +52,438] | 2.85 | 462,844 |
| 1–500 moves after | 120 | 40 | 12.1% | +6.8% [+5.2%, +8.5%] | +37,459 [+33,994, +41,082] | 2.85 | 436,523 |
| 501–1,500 | 160 | 40 | 11.3% | +7.2% [+5.8%, +8.7%] | +38,006 [+35,189, +40,878] | 2.85 | 380,561 |
| 1,501–3,000 | 240 | 40 | 12.1% | +6.8% [+5.7%, +8.0%] | +33,923 [+31,254, +36,633] | 2.85 | 308,394 |
| over 3,000 | 680 | 40 | 18.7% | +3.6% [+3.0%, +4.2%] | +17,038 [+15,785, +18,348] | 2.85 | 170,858 |

**By moves remaining, failed trajectories**

| Group | Points | Trajectories | Greedy makes 32768 | Δ makes 32768 [95%] | Δ score added [95%] | CPU s per continuation | Chance nodes per continuation |
|---|---:|---:|---:|---:|---:|---:|---:|
| over 2,000 moves left | 706 | 40 | 12.4% | +6.9% [+6.1%, +7.6%] | +33,505 [+31,355, +35,838] | 2.91 | 328,521 |
| 501–2,000 left | 240 | 40 | 14.5% | +6.2% [+4.8%, +7.8%] | +24,110 [+19,833, +28,788] | 3.04 | 187,975 |
| 201–500 left | 52 | 40 | 15.9% | +4.6% [+1.0%, +8.0%] | +20,341 [+12,046, +28,190] | 3.10 | 139,545 |
| 200 or fewer left | 107 | 40 | 16.3% | +3.3% [+1.1%, +5.4%] | +16,170 [+10,593, +21,567] | 3.02 | 112,378 |

**By empty cells, failed trajectories**

| Group | Points | Trajectories | Greedy makes 32768 | Δ makes 32768 [95%] | Δ score added [95%] | CPU s per continuation | Chance nodes per continuation |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0–1 empty | 230 | 40 | 13.5% | +6.4% [+5.3%, +7.7%] | +28,581 [+25,976, +31,651] | 2.98 | 259,373 |
| 2–3 empty | 566 | 40 | 14.2% | +6.0% [+4.9%, +7.0%] | +27,968 [+25,229, +31,216] | 2.93 | 251,896 |
| 4–6 empty | 269 | 40 | 11.9% | +6.6% [+5.7%, +7.5%] | +31,230 [+28,729, +33,935] | 3.00 | 288,410 |
| 7+ empty | 40 | 31 | 11.7% | +7.3% [+5.4%, +9.4%] | +35,622 [+30,491, +41,135] | 2.98 | 412,994 |

At the first 16384, weighting successes and failures by greedy's natural 11.2% conversion, B changes conversion by +7.8%.

## C: depth 3 for 200 moves, then greedy


**By moves since the first 16384, failed trajectories**

| Group | Points | Trajectories | Greedy makes 32768 | Δ makes 32768 [95%] | Δ score added [95%] | CPU s per continuation | Chance nodes per continuation |
|---|---:|---:|---:|---:|---:|---:|---:|
| at the first 16384 | 40 | 40 | 12.0% | -2.0% [-4.5%, +0.2%] | -2,469 [-8,999, +3,720] | 0.14 | 15,354 |
| 1–500 moves after | 120 | 40 | 11.3% | +0.2% [-1.0%, +1.4%] | -514 [-3,789, +3,058] | 0.14 | 13,278 |
| 501–1,500 | 160 | 40 | 11.6% | -0.8% [-1.8%, +0.2%] | -657 [-3,614, +2,272] | 0.14 | 12,151 |
| 1,501–3,000 | 241 | 40 | 11.4% | +0.7% [-0.1%, +1.5%] | +4,057 [+2,085, +6,062] | 0.14 | 11,847 |
| over 3,000 | 544 | 38 | 15.4% | +0.6% [+0.1%, +1.2%] | +1,994 [+835, +3,228] | 0.14 | 10,476 |

**By moves since the first 16384, successful trajectories**

| Group | Points | Trajectories | Greedy makes 32768 | Δ makes 32768 [95%] | Δ score added [95%] | CPU s per continuation | Chance nodes per continuation |
|---|---:|---:|---:|---:|---:|---:|---:|
| at the first 16384 | 40 | 40 | 10.4% | -0.1% [-2.2%, +2.0%] | +322 [-4,497, +5,099] | 0.14 | 15,254 |
| 1–500 moves after | 120 | 40 | 12.1% | -0.9% [-2.0%, +0.2%] | -2,894 [-5,264, -416] | 0.14 | 13,237 |
| 501–1,500 | 160 | 40 | 11.3% | +0.9% [-0.3%, +2.2%] | +1,673 [-1,263, +4,660] | 0.14 | 12,166 |
| 1,501–3,000 | 240 | 40 | 12.1% | +0.2% [-0.7%, +1.2%] | +1,921 [-320, +4,200] | 0.14 | 11,910 |
| over 3,000 | 680 | 40 | 18.7% | +0.4% [-0.2%, +1.0%] | +1,388 [+247, +2,597] | 0.14 | 10,692 |

**By moves remaining, failed trajectories**

| Group | Points | Trajectories | Greedy makes 32768 | Δ makes 32768 [95%] | Δ score added [95%] | CPU s per continuation | Chance nodes per continuation |
|---|---:|---:|---:|---:|---:|---:|---:|
| over 2,000 moves left | 706 | 40 | 12.4% | +0.1% [-0.4%, +0.5%] | +1,176 [-74, +2,322] | 0.14 | 12,210 |
| 501–2,000 left | 240 | 40 | 14.5% | +1.0% [+0.1%, +1.9%] | +2,766 [+917, +4,631] | 0.14 | 11,146 |
| 201–500 left | 52 | 40 | 15.9% | +1.0% [-2.3%, +4.1%] | +3,595 [-2,980, +9,778] | 0.14 | 9,941 |
| 200 or fewer left | 107 | 40 | 16.3% | +0.0% [-1.5%, +1.5%] | +1,083 [-2,104, +4,124] | 0.14 | 8,352 |

**By empty cells, failed trajectories**

| Group | Points | Trajectories | Greedy makes 32768 | Δ makes 32768 [95%] | Δ score added [95%] | CPU s per continuation | Chance nodes per continuation |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0–1 empty | 230 | 40 | 13.5% | -0.2% [-1.0%, +0.7%] | +410 [-1,519, +2,349] | 0.14 | 11,068 |
| 2–3 empty | 566 | 40 | 14.2% | +0.4% [-0.3%, +1.0%] | +1,921 [+450, +3,333] | 0.14 | 11,237 |
| 4–6 empty | 269 | 40 | 11.9% | +0.8% [-0.0%, +1.6%] | +2,480 [+442, +4,471] | 0.14 | 11,975 |
| 7+ empty | 40 | 31 | 11.7% | -1.2% [-3.5%, +1.3%] | -1,300 [-7,447, +5,848] | 0.14 | 14,467 |

At the first 16384, weighting successes and failures by greedy's natural 11.2% conversion, C changes conversion by -1.8%.

## D: pruned depth 5 (cutoff 0.01) for 200 moves, then greedy


**By moves since the first 16384, failed trajectories**

| Group | Points | Trajectories | Greedy makes 32768 | Δ makes 32768 [95%] | Δ score added [95%] | CPU s per continuation | Chance nodes per continuation |
|---|---:|---:|---:|---:|---:|---:|---:|
| at the first 16384 | 40 | 40 | 12.0% | -1.8% [-4.5%, +0.9%] | -3,988 [-10,191, +2,293] | 1.59 | 148,724 |
| 1–500 moves after | 120 | 40 | 11.3% | +0.2% [-1.1%, +1.6%] | +626 [-2,699, +3,990] | 1.59 | 177,951 |
| 501–1,500 | 160 | 40 | 11.6% | +0.4% [-0.8%, +1.5%] | +1,623 [-1,274, +4,453] | 1.59 | 187,356 |
| 1,501–3,000 | 241 | 40 | 11.4% | +0.6% [-0.4%, +1.6%] | +3,606 [+1,355, +5,856] | 1.59 | 189,720 |
| over 3,000 | 544 | 38 | 15.4% | +1.1% [+0.3%, +1.9%] | +2,609 [+967, +4,174] | 1.57 | 184,552 |

**By moves since the first 16384, successful trajectories**

| Group | Points | Trajectories | Greedy makes 32768 | Δ makes 32768 [95%] | Δ score added [95%] | CPU s per continuation | Chance nodes per continuation |
|---|---:|---:|---:|---:|---:|---:|---:|
| at the first 16384 | 40 | 40 | 10.4% | +1.8% [-0.5%, +4.2%] | +5,693 [-405, +11,820] | 1.57 | 150,525 |
| 1–500 moves after | 120 | 40 | 12.1% | +0.0% [-1.3%, +1.3%] | -979 [-4,616, +2,540] | 1.57 | 177,669 |
| 501–1,500 | 160 | 40 | 11.3% | +0.2% [-1.0%, +1.5%] | +107 [-3,215, +3,371] | 1.57 | 187,237 |
| 1,501–3,000 | 240 | 40 | 12.1% | -0.7% [-1.6%, +0.2%] | +61 [-2,244, +2,385] | 1.57 | 189,886 |
| over 3,000 | 680 | 40 | 18.7% | +0.6% [-0.1%, +1.3%] | +1,879 [+653, +3,107] | 1.57 | 188,412 |

**By moves remaining, failed trajectories**

| Group | Points | Trajectories | Greedy makes 32768 | Δ makes 32768 [95%] | Δ score added [95%] | CPU s per continuation | Chance nodes per continuation |
|---|---:|---:|---:|---:|---:|---:|---:|
| over 2,000 moves left | 706 | 40 | 12.4% | +0.1% [-0.5%, +0.6%] | +1,063 [-362, +2,358] | 1.57 | 186,101 |
| 501–2,000 left | 240 | 40 | 14.5% | +0.9% [-0.2%, +2.0%] | +2,543 [+347, +4,740] | 1.59 | 191,290 |
| 201–500 left | 52 | 40 | 15.9% | +0.2% [-2.6%, +3.1%] | +2,510 [-3,297, +8,124] | 1.59 | 184,343 |
| 200 or fewer left | 107 | 40 | 16.3% | +4.3% [+1.9%, +6.7%] | +9,087 [+4,812, +13,376] | 1.59 | 154,351 |

**By empty cells, failed trajectories**

| Group | Points | Trajectories | Greedy makes 32768 | Δ makes 32768 [95%] | Δ score added [95%] | CPU s per continuation | Chance nodes per continuation |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0–1 empty | 230 | 40 | 13.5% | +0.4% [-0.9%, +1.7%] | +2,207 [-306, +4,856] | 1.58 | 177,569 |
| 2–3 empty | 566 | 40 | 14.2% | +0.9% [+0.3%, +1.5%] | +2,381 [+975, +3,804] | 1.57 | 187,241 |
| 4–6 empty | 269 | 40 | 11.9% | +0.7% [-0.2%, +1.6%] | +2,545 [+777, +4,391] | 1.58 | 186,488 |
| 7+ empty | 40 | 31 | 11.7% | -1.4% [-4.1%, +1.5%] | -1,916 [-7,886, +5,288] | 1.58 | 160,352 |

At the first 16384, weighting successes and failures by greedy's natural 11.2% conversion, D changes conversion by -1.4%.

## D against C: deeper search over the same 200 moves

| Group (failed trajectories) | Points | Δ makes 32768, D − C [95%] | Δ score, D − C [95%] |
|---|---:|---:|---:|
| at the first 16384 | 40 | +0.2% [-2.0%, +2.5%] | -1,519 [-6,839, +4,034] |
| 1–500 moves after | 120 | +0.0% [-1.3%, +1.3%] | +1,140 [-1,735, +4,105] |
| 501–1,500 | 160 | +1.2% [-0.0%, +2.4%] | +2,281 [-1,005, +5,540] |
| 1,501–3,000 | 241 | -0.1% [-1.1%, +0.9%] | -451 [-2,649, +1,856] |
| over 3,000 | 544 | +0.4% [-0.4%, +1.2%] | +615 [-984, +2,181] |
