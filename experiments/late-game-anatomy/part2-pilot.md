# Stage 7, Part 2: when can failing games be rescued? (pilot)

Generated 2026-11-14 16:33.

Policies: **A** greedy; **B** depth 3 from here; **C** depth 3 for 200 moves, then greedy; **D** pruned depth 5 (cutoff 0.01) for 200 moves, then greedy.

Each cell is the mean paired difference from greedy (A) over points, then continuations, with a 95% interval from resampling whole trajectories. Moves remaining are retrospective (the agent can't know them).

Trajectories: 20 that made 32768 (sampled up to it) and 20 that didn't. Points: 1,175. Continuations: 37,600.


## B: depth 3 from here


**By moves since the first 16384, failed trajectories**

| Group | Points | Trajectories | Greedy makes 32768 | Δ makes 32768 [95%] | Δ score added [95%] | CPU s per continuation | Chance nodes per continuation |
|---|---:|---:|---:|---:|---:|---:|---:|
| at the first 16384 | 20 | 20 | 13.1% | +4.4% [-3.8%, +12.5%] | +26,773 [+7,984, +45,324] | 3.02 | 440,083 |
| 1–500 moves after | 60 | 20 | 12.9% | +7.3% [+2.9%, +11.9%] | +37,111 [+28,755, +45,205] | 3.02 | 435,466 |
| 501–1,500 | 80 | 20 | 13.3% | +5.6% [+1.4%, +9.4%] | +32,050 [+21,822, +41,840] | 3.02 | 379,586 |
| 1,501–3,000 | 120 | 20 | 9.5% | +8.4% [+5.4%, +11.1%] | +40,867 [+32,890, +48,473] | 3.02 | 304,407 |
| over 3,000 | 275 | 19 | 15.8% | +6.0% [+3.7%, +8.5%] | +21,879 [+16,965, +27,948] | 2.83 | 169,913 |

**By moves since the first 16384, successful trajectories**

| Group | Points | Trajectories | Greedy makes 32768 | Δ makes 32768 [95%] | Δ score added [95%] | CPU s per continuation | Chance nodes per continuation |
|---|---:|---:|---:|---:|---:|---:|---:|
| at the first 16384 | 20 | 20 | 11.2% | +6.9% [-0.6%, +14.4%] | +39,477 [+22,799, +57,522] | 2.85 | 452,459 |
| 1–500 moves after | 60 | 20 | 11.7% | +10.6% [+6.0%, +15.4%] | +45,596 [+34,369, +56,472] | 2.85 | 444,264 |
| 501–1,500 | 80 | 20 | 10.9% | +9.4% [+4.2%, +14.7%] | +43,780 [+31,494, +55,885] | 2.85 | 388,856 |
| 1,501–3,000 | 120 | 20 | 12.3% | +6.6% [+3.9%, +10.0%] | +33,157 [+26,977, +41,027] | 2.85 | 307,960 |
| over 3,000 | 340 | 20 | 19.2% | +5.0% [+2.9%, +6.9%] | +19,901 [+15,316, +24,341] | 2.85 | 176,424 |

**By moves remaining, failed trajectories**

| Group | Points | Trajectories | Greedy makes 32768 | Δ makes 32768 [95%] | Δ score added [95%] | CPU s per continuation | Chance nodes per continuation |
|---|---:|---:|---:|---:|---:|---:|---:|
| over 2,000 moves left | 356 | 20 | 12.3% | +6.2% [+4.5%, +7.8%] | +30,556 [+26,292, +34,673] | 2.87 | 322,182 |
| 501–2,000 left | 120 | 20 | 14.6% | +7.9% [+4.6%, +11.4%] | +28,789 [+20,282, +37,727] | 3.02 | 195,656 |
| 201–500 left | 25 | 20 | 16.5% | +9.5% [+1.6%, +16.7%] | +34,112 [+18,756, +47,030] | 3.09 | 153,010 |
| 200 or fewer left | 54 | 20 | 19.7% | +4.4% [-0.7%, +9.5%] | +19,658 [+8,142, +31,084] | 3.00 | 121,303 |

**By empty cells, failed trajectories**

| Group | Points | Trajectories | Greedy makes 32768 | Δ makes 32768 [95%] | Δ score added [95%] | CPU s per continuation | Chance nodes per continuation |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0–1 empty | 109 | 20 | 14.0% | +7.5% [+3.5%, +11.8%] | +31,363 [+22,652, +40,331] | 2.93 | 259,535 |
| 2–3 empty | 293 | 20 | 14.0% | +7.3% [+5.2%, +9.8%] | +30,790 [+25,060, +37,414] | 2.90 | 256,059 |
| 4–6 empty | 133 | 20 | 13.3% | +4.0% [+1.4%, +6.9%] | +23,976 [+18,296, +30,142] | 2.99 | 281,223 |
| 7+ empty | 20 | 15 | 9.4% | +6.9% [+2.0%, +12.5%] | +30,906 [+16,147, +47,321] | 2.97 | 391,703 |

At the first 16384, weighting successes and failures by greedy's natural 11.2% conversion, B changes conversion by +4.7%.

## C: depth 3 for 200 moves, then greedy


**By moves since the first 16384, failed trajectories**

| Group | Points | Trajectories | Greedy makes 32768 | Δ makes 32768 [95%] | Δ score added [95%] | CPU s per continuation | Chance nodes per continuation |
|---|---:|---:|---:|---:|---:|---:|---:|
| at the first 16384 | 20 | 20 | 13.1% | -1.9% [-8.1%, +4.4%] | -8,042 [-25,032, +9,289] | 0.14 | 15,329 |
| 1–500 moves after | 60 | 20 | 12.9% | -0.4% [-3.5%, +2.9%] | +2,086 [-3,497, +8,127] | 0.14 | 13,254 |
| 501–1,500 | 80 | 20 | 13.3% | -0.3% [-3.4%, +2.8%] | -1,049 [-7,932, +6,141] | 0.14 | 12,169 |
| 1,501–3,000 | 120 | 20 | 9.5% | +3.4% [+0.9%, +5.9%] | +10,123 [+3,957, +16,492] | 0.14 | 11,861 |
| over 3,000 | 275 | 19 | 15.8% | +1.2% [-0.6%, +2.7%] | +3,330 [+38, +6,146] | 0.14 | 10,359 |

**By moves since the first 16384, successful trajectories**

| Group | Points | Trajectories | Greedy makes 32768 | Δ makes 32768 [95%] | Δ score added [95%] | CPU s per continuation | Chance nodes per continuation |
|---|---:|---:|---:|---:|---:|---:|---:|
| at the first 16384 | 20 | 20 | 11.2% | -1.9% [-6.9%, +3.1%] | -997 [-11,554, +9,858] | 0.14 | 15,304 |
| 1–500 moves after | 60 | 20 | 11.7% | -1.0% [-5.4%, +3.3%] | -4,667 [-13,840, +4,648] | 0.14 | 13,156 |
| 501–1,500 | 80 | 20 | 10.9% | +1.2% [-2.3%, +4.8%] | +5,384 [-2,349, +12,870] | 0.14 | 12,148 |
| 1,501–3,000 | 120 | 20 | 12.3% | -0.2% [-3.6%, +3.1%] | +738 [-6,303, +7,143] | 0.14 | 11,919 |
| over 3,000 | 340 | 20 | 19.2% | -1.7% [-3.2%, -0.4%] | -1,509 [-4,432, +1,290] | 0.14 | 10,693 |

**By moves remaining, failed trajectories**

| Group | Points | Trajectories | Greedy makes 32768 | Δ makes 32768 [95%] | Δ score added [95%] | CPU s per continuation | Chance nodes per continuation |
|---|---:|---:|---:|---:|---:|---:|---:|
| over 2,000 moves left | 356 | 20 | 12.3% | +1.5% [+0.1%, +2.9%] | +4,243 [+680, +7,635] | 0.14 | 12,203 |
| 501–2,000 left | 120 | 20 | 14.6% | +0.9% [-1.6%, +3.2%] | +2,989 [-1,395, +7,114] | 0.14 | 11,137 |
| 201–500 left | 25 | 20 | 16.5% | -1.0% [-9.9%, +7.1%] | +2,271 [-14,494, +17,605] | 0.14 | 9,822 |
| 200 or fewer left | 54 | 20 | 19.7% | +0.5% [-4.3%, +5.1%] | +1,571 [-7,491, +10,335] | 0.14 | 7,799 |

**By empty cells, failed trajectories**

| Group | Points | Trajectories | Greedy makes 32768 | Δ makes 32768 [95%] | Δ score added [95%] | CPU s per continuation | Chance nodes per continuation |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0–1 empty | 109 | 20 | 14.0% | +0.9% [-3.0%, +4.5%] | +4,341 [-3,262, +11,473] | 0.14 | 10,832 |
| 2–3 empty | 293 | 20 | 14.0% | +1.9% [-0.1%, +3.8%] | +4,828 [+401, +8,992] | 0.14 | 11,236 |
| 4–6 empty | 133 | 20 | 13.3% | -0.2% [-1.5%, +1.2%] | +1,198 [-2,165, +4,458] | 0.14 | 11,930 |
| 7+ empty | 20 | 15 | 9.4% | +0.6% [-4.4%, +5.1%] | -1,807 [-17,743, +12,469] | 0.14 | 14,394 |

At the first 16384, weighting successes and failures by greedy's natural 11.2% conversion, C changes conversion by -1.9%.

## D: pruned depth 5 (cutoff 0.01) for 200 moves, then greedy


**By moves since the first 16384, failed trajectories**

| Group | Points | Trajectories | Greedy makes 32768 | Δ makes 32768 [95%] | Δ score added [95%] | CPU s per continuation | Chance nodes per continuation |
|---|---:|---:|---:|---:|---:|---:|---:|
| at the first 16384 | 20 | 20 | 13.1% | -1.2% [-8.8%, +6.2%] | +990 [-18,521, +20,813] | 1.56 | 150,086 |
| 1–500 moves after | 60 | 20 | 12.9% | +0.8% [-2.5%, +4.4%] | +2,328 [-5,871, +10,968] | 1.56 | 178,150 |
| 501–1,500 | 80 | 20 | 13.3% | +0.2% [-2.5%, +2.8%] | +1,867 [-3,822, +8,235] | 1.56 | 187,027 |
| 1,501–3,000 | 120 | 20 | 9.5% | +1.7% [-0.6%, +4.1%] | +7,558 [+1,241, +13,387] | 1.56 | 189,298 |
| over 3,000 | 275 | 19 | 15.8% | +2.6% [+0.3%, +4.8%] | +5,843 [+1,129, +10,181] | 1.54 | 183,228 |

**By moves since the first 16384, successful trajectories**

| Group | Points | Trajectories | Greedy makes 32768 | Δ makes 32768 [95%] | Δ score added [95%] | CPU s per continuation | Chance nodes per continuation |
|---|---:|---:|---:|---:|---:|---:|---:|
| at the first 16384 | 20 | 20 | 11.2% | +1.9% [-3.1%, +6.9%] | +5,020 [-7,333, +17,893] | 1.55 | 150,399 |
| 1–500 moves after | 60 | 20 | 11.7% | +0.4% [-4.4%, +5.4%] | +683 [-9,696, +11,571] | 1.55 | 178,405 |
| 501–1,500 | 80 | 20 | 10.9% | -1.2% [-4.2%, +1.9%] | +482 [-7,238, +8,152] | 1.55 | 188,471 |
| 1,501–3,000 | 120 | 20 | 12.3% | +0.6% [-2.8%, +4.5%] | +1,642 [-5,558, +8,705] | 1.55 | 189,773 |
| over 3,000 | 340 | 20 | 19.2% | -0.1% [-1.9%, +1.4%] | +1,316 [-2,293, +4,528] | 1.55 | 188,350 |

**By moves remaining, failed trajectories**

| Group | Points | Trajectories | Greedy makes 32768 | Δ makes 32768 [95%] | Δ score added [95%] | CPU s per continuation | Chance nodes per continuation |
|---|---:|---:|---:|---:|---:|---:|---:|
| over 2,000 moves left | 356 | 20 | 12.3% | +2.2% [+0.4%, +3.9%] | +5,716 [+1,004, +9,931] | 1.55 | 186,455 |
| 501–2,000 left | 120 | 20 | 14.6% | +0.9% [-2.3%, +4.3%] | +3,238 [-2,472, +9,343] | 1.56 | 191,415 |
| 201–500 left | 25 | 20 | 16.5% | -5.0% [-12.0%, +1.6%] | -3,048 [-16,313, +9,772] | 1.57 | 181,633 |
| 200 or fewer left | 54 | 20 | 19.7% | +3.5% [-2.6%, +9.2%] | +8,803 [-2,736, +18,762] | 1.56 | 145,695 |

**By empty cells, failed trajectories**

| Group | Points | Trajectories | Greedy makes 32768 | Δ makes 32768 [95%] | Δ score added [95%] | CPU s per continuation | Chance nodes per continuation |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0–1 empty | 109 | 20 | 14.0% | +3.3% [-0.6%, +7.6%] | +10,606 [+2,135, +19,146] | 1.55 | 174,261 |
| 2–3 empty | 293 | 20 | 14.0% | +2.0% [-0.1%, +3.9%] | +4,703 [+60, +9,069] | 1.55 | 186,568 |
| 4–6 empty | 133 | 20 | 13.3% | -0.3% [-3.3%, +2.6%] | +1,677 [-2,617, +5,617] | 1.56 | 186,627 |
| 7+ empty | 20 | 15 | 9.4% | +1.9% [-3.8%, +8.1%] | +3,275 [-14,636, +22,946] | 1.56 | 163,793 |

At the first 16384, weighting successes and failures by greedy's natural 11.2% conversion, D changes conversion by -0.9%.

## D against C: deeper search over the same 200 moves

| Group (failed trajectories) | Points | Δ makes 32768, D − C [95%] | Δ score, D − C [95%] |
|---|---:|---:|---:|
| at the first 16384 | 20 | +0.6% [-5.6%, +7.5%] | +9,032 [-4,972, +25,540] |
| 1–500 moves after | 60 | +1.2% [-1.0%, +3.3%] | +242 [-6,467, +6,269] |
| 501–1,500 | 80 | +0.5% [-2.7%, +3.9%] | +2,915 [-2,975, +9,177] |
| 1,501–3,000 | 120 | -1.8% [-4.2%, +0.7%] | -2,565 [-7,787, +2,491] |
| over 3,000 | 275 | +1.5% [-0.6%, +3.3%] | +2,513 [-1,809, +6,296] |
