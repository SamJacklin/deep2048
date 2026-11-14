# Ryzen long runs: results

Two 10M-game runs from scratch, `recipe` (run `ryzen-recipe`) and `optimism` (run `ryzen-optimism`), following the
`ryzen-instructions` README, plus an extension of `optimism` from 10M to 14M games added at the user's request (section 8).

**Status of this file:** `optimism` and its extension are complete. `recipe` finished training to 10M games, and its
depth-3 evaluations are still running; its tables and charts will be added in a later commit on this branch.

## 1. The machine

- **CPU:** AMD Ryzen 7 9850X3D, 8 cores, 16 threads
- **RAM:** 32 GB (31.1 GiB visible to Windows); WSL2 limited to 28 GB, 16 processors and 8 GB swap by `.wslconfig` (`free -g` total 27)
- **OS:** Windows 11 Home, WSL2 2.5.10.0, Ubuntu 24.04.3 LTS with systemd

## 2. The commit

`aa522aab0c2393e241dd846a11738862725eab92` (`aa522aa`, the top of `feat/fast-tc` when cloned on 2026-11-13).
`uv run pytest -q`: 85 passed in 55.45 s.

## 3. The smoke test

Ran from 12:05:18 to 12:12:15 UTC on 2026-11-13. **Passed:** both arms printed their greedy lines, three depth-3
lines and `finished`; the recipe log contained the coherence-reset line, and the optimism log contained the
optimistic-initialisation line. The smoke runs were deleted afterwards. Output:

```text
=== recipe ===
N-tuple TD (10,000 games, TC lr 0.02, 8x6 network, downgraded from 16384)       66001       3763   100.0%   100.0%    95.0%    90.0%    70.0%    10.0%     0.0%  (20 games)
N-tuple TD (20,000 games, TC lr 0.02, 8x6 network, downgraded from 16384)       77058       3994   100.0%   100.0%   100.0%    90.0%    70.0%    15.0%     0.0%  (20 games)
N-tuple TD (30,000 games, TC lr 0.02, 8x6 network, downgraded from 16384)       87006       4403   100.0%   100.0%   100.0%   100.0%    75.0%    20.0%     0.0%  (20 games)
N-tuple TD (40,000 games, TC lr 0.02, 8x6 network, downgraded from 16384)      102011       5325   100.0%   100.0%   100.0%   100.0%    80.0%    40.0%     0.0%  (20 games)
N-tuple TD (6,000 games, TC lr 0.02, 8x6 network, downgraded from 16384) + search depth=3      117710       6656   100.0%   100.0%   100.0%   100.0%    95.0%    65.0%     0.0%  (20 games)
N-tuple TD (20,000 games, TC lr 0.02, 8x6 network, downgraded from 16384) + search depth=3      165763       8909   100.0%   100.0%   100.0%   100.0%    95.0%    80.0%    20.0%  (20 games)
N-tuple TD (40,000 games, TC lr 0.02, 8x6 network, downgraded from 16384) + search depth=3      205309      10854   100.0%   100.0%   100.0%   100.0%   100.0%    95.0%    35.0%  (20 games)
smoke-ryzen-recipe finished
=== optimism ===
N-tuple TD (20,000 games, lr 0.0016, 8x6 network, downgraded from 16384)        8376        640   100.0%    80.0%    35.0%     0.0%     0.0%     0.0%     0.0%  (20 games)
N-tuple TD (30,000 games, lr 0.0016, 8x6 network, downgraded from 16384)        9964        730   100.0%    85.0%    50.0%     0.0%     0.0%     0.0%     0.0%  (20 games)
Training smoke-ryzen-optimism from 36000 to 40000 games (NTUPLE_TEMPORAL_COHERENCE=true NTUPLE_LEARNING_RATE=0.02 )
N-tuple TD (40,000 games, TC lr 0.02, 8x6 network, downgraded from 16384)       10788        768   100.0%   100.0%    50.0%     0.0%     0.0%     0.0%     0.0%  (20 games)
N-tuple TD (36,000 games, lr 0.0016, 8x6 network, downgraded from 16384) + search depth=3       22781       1638   100.0%   100.0%   100.0%    60.0%     0.0%     0.0%     0.0%  (20 games)
N-tuple TD (20,000 games, lr 0.0016, 8x6 network, downgraded from 16384) + search depth=3       19939       1408   100.0%   100.0%    95.0%    40.0%     0.0%     0.0%     0.0%  (20 games)
N-tuple TD (40,000 games, TC lr 0.02, 8x6 network, downgraded from 16384) + search depth=3       19795       1331   100.0%   100.0%   100.0%    30.0%     0.0%     0.0%     0.0%  (20 games)
smoke-ryzen-optimism finished
=== checks ===
Temporal coherence sums reset: every weight starts back at the full rate
134,217,728 never-updated weights (100.0%) set to 5000
```

## 4. The benchmark

Ran from 12:12:29 to 12:17:27 UTC with nothing else running.

| Benchmark | Ryzen 7 9850X3D | Apple M4 (development machine) |
|---|---:|---:|
| Training, 1 thread, 20,000 games | 0.37M moves/s | 0.53M moves/s (0.49M on a second run) |
| Training, 8 threads, 40,000 games | 2.46M moves/s | 2.33M moves/s (2.37M on a second run) |
| Training, 10 threads (M4 maximum), 40,000 games | | 2.58M moves/s |
| Training, 16 threads, 40,000 games | 3.05M moves/s | |
| Greedy evaluation, 1,000 games, all CPUs | 6 s, average score **71,753** | 4 s, average score **71,753** |
| Depth-3 evaluation, 100 games, all CPUs | 56 s, average score **155,098** | 57 s, average score **155,098** |

**Both average scores matched the M4 exactly.** Raw output:

```text
Machine: Linux x86_64, 16 logical CPUs
Training, 1 thread(s), 20000 games from scratch: 0.37M moves/s over 15,000 games
Training, 8 thread(s), 40000 games from scratch: 2.46M moves/s over 35,000 games
Training, 16 thread(s), 40000 games from scratch: 3.05M moves/s over 35,000 games
Evaluation (--agent ntuple -n 1000, all CPUs): average score 71753 in 6 s
Evaluation (--agent ntuple_search --depth 3 -n 100, all CPUs): average score 155098 in 56 s
```

## 5. Timings

All times are UTC. Both runs used `THREADS=8` and the exact C1 commands. A hidden Windows process,
`wsl.exe -d Ubuntu -- sleep infinity`, kept WSL from idling the distro out.

| Run | Started | Training finished | Evaluations finished |
|---|---|---|---|
| `ryzen-optimism` | 2026-11-13 12:17:49 | 19:42 on 11-13 (9M at 18:14) | 20:28 on 11-13 |
| `ryzen-recipe` | 2026-11-13 12:17:49 | 03:52 on 11-14 | still running (see below) |
| `ryzen-optimism-ext` (section 8) | 2026-11-13 20:28:25 | 03:05 on 11-14 | 03:24 on 11-14 |

**Interruptions and problems:**
- **Windows crashed and rebooted at 03:55 UTC on 2026-11-14.** `ryzen-optimism` and `ryzen-optimism-ext` had
  already finished. `ryzen-recipe` had saved its 10M snapshot at 03:52, but its 10M greedy evaluation and its three
  depth-3 evaluations hadn't finished. At 08:24 UTC the keepalive was restarted, the 10M greedy evaluation was run
  by hand with the script's own `evaluate.py` command (rerunning the script skips greedy evaluations once the final
  snapshot exists), and then `THREADS=8 scripts/run_ryzen_arms.sh recipe >> ryzen-recipe.log` was rerun for the
  depth-3 evaluations. Both are appended to `ryzen-recipe/run.log`.
- **Not done by the agent:** pausing Windows Update, which needs a human. `gh auth login` was done by the user
  on 2026-11-14.

## 6. Results

Tables from `uv run python scripts/summarise_runs.py ryzen-optimism ryzen-optimism-ext`, pasted unchanged. Greedy
evaluations are on 1,000 seeded games every 250,000 games; depth-3 evaluations are on the same 1,000 games.
`ryzen-recipe`'s tables will be added when its evaluations finish.

### ryzen-optimism

| Games | Search | Average ± 95% | Median | ≥16384 | ≥32768 |
|---:|---|---:|---:|---:|---:|
| 250,000 | greedy | 50,216 ± 873 | 54,326 | 0.0% | 0.0% |
| 500,000 | greedy | 67,698 ± 1,015 | 70,926 | 0.0% | 0.0% |
| 750,000 | greedy | 113,116 ± 2,673 | 123,958 | 0.0% | 0.0% |
| 1,000,000 | greedy | 119,790 ± 2,535 | 131,238 | 0.1% | 0.0% |
| 1,250,000 | greedy | 122,984 ± 2,413 | 132,364 | 0.0% | 0.0% |
| 1,500,000 | greedy | 126,616 ± 2,395 | 136,130 | 0.0% | 0.0% |
| 1,750,000 | greedy | 125,424 ± 2,485 | 133,268 | 0.1% | 0.0% |
| 2,000,000 | greedy | 128,345 ± 2,462 | 144,214 | 0.2% | 0.0% |
| 2,250,000 | greedy | 129,554 ± 2,500 | 145,918 | 0.1% | 0.0% |
| 2,500,000 | greedy | 134,638 ± 2,516 | 148,146 | 0.8% | 0.0% |
| 2,750,000 | greedy | 134,567 ± 2,532 | 148,476 | 0.7% | 0.0% |
| 3,000,000 | greedy | 135,407 ± 2,988 | 147,818 | 3.8% | 0.0% |
| 3,250,000 | greedy | 185,097 ± 5,701 | 170,210 | 31.9% | 0.1% |
| 3,500,000 | greedy | 193,402 ± 5,919 | 172,234 | 35.9% | 0.4% |
| 3,750,000 | greedy | 197,993 ± 6,130 | 174,442 | 39.5% | 0.1% |
| 4,000,000 | greedy | 210,222 ± 6,113 | 176,278 | 42.0% | 0.5% |
| 4,250,000 | greedy | 201,553 ± 6,326 | 174,882 | 40.6% | 0.1% |
| 4,500,000 | greedy | 206,853 ± 6,406 | 175,406 | 41.6% | 0.5% |
| 4,750,000 | greedy | 203,182 ± 6,411 | 174,252 | 39.3% | 0.2% |
| 5,000,000 | depth 3 | 312,491 ± 5,841 | 355,736 | 78.2% | 3.4% |
| 5,000,000 | greedy | 200,297 ± 6,263 | 173,118 | 38.9% | 0.3% |
| 5,250,000 | greedy | 211,031 ± 6,367 | 176,620 | 45.0% | 0.2% |
| 5,500,000 | greedy | 210,295 ± 6,283 | 175,468 | 42.2% | 0.5% |
| 5,750,000 | greedy | 215,996 ± 6,640 | 176,684 | 45.7% | 0.7% |
| 6,000,000 | greedy | 218,575 ± 6,563 | 176,876 | 46.6% | 0.6% |
| 6,250,000 | greedy | 220,523 ± 6,636 | 177,266 | 48.1% | 0.5% |
| 6,500,000 | greedy | 222,300 ± 6,583 | 177,234 | 47.8% | 0.5% |
| 6,750,000 | greedy | 215,155 ± 6,412 | 176,802 | 45.9% | 0.7% |
| 7,000,000 | greedy | 220,608 ± 6,750 | 177,202 | 47.6% | 0.3% |
| 7,250,000 | greedy | 228,210 ± 6,818 | 222,156 | 50.3% | 0.7% |
| 7,500,000 | greedy | 224,771 ± 6,678 | 177,808 | 49.5% | 0.8% |
| 7,750,000 | greedy | 231,744 ± 6,777 | 236,570 | 51.5% | 0.6% |
| 8,000,000 | greedy | 223,677 ± 6,550 | 178,112 | 49.2% | 0.5% |
| 8,250,000 | greedy | 229,517 ± 6,660 | 219,984 | 50.3% | 0.7% |
| 8,500,000 | greedy | 225,735 ± 6,659 | 178,340 | 49.7% | 0.4% |
| 8,750,000 | greedy | 231,946 ± 6,640 | 239,750 | 52.0% | 1.3% |
| 9,000,000 | depth 3 | 330,406 ± 5,654 | 372,174 | 83.7% | 5.2% |
| 9,000,000 | greedy | 234,785 ± 6,573 | 242,196 | 53.6% | 1.2% |
| 9,250,000 | greedy | 291,323 ± 7,093 | 342,694 | 70.9% | 3.1% |
| 9,500,000 | greedy | 300,084 ± 6,879 | 342,678 | 73.7% | 4.6% |
| 9,750,000 | greedy | 305,809 ± 6,713 | 343,010 | 76.6% | 4.5% |
| 10,000,000 | depth 3 | 379,675 ± 5,602 | 382,918 | 92.6% | 16.8% |
| 10,000,000 | greedy | 300,537 ± 6,879 | 342,774 | 74.1% | 4.3% |

### ryzen-optimism-ext

| Games | Search | Average ± 95% | Median | ≥16384 | ≥32768 |
|---:|---|---:|---:|---:|---:|
| 10,250,000 | greedy | 305,132 ± 6,712 | 343,044 | 75.1% | 4.0% |
| 10,500,000 | greedy | 312,405 ± 6,679 | 364,368 | 77.9% | 4.9% |
| 10,750,000 | greedy | 310,252 ± 6,628 | 362,220 | 76.2% | 4.3% |
| 11,000,000 | greedy | 306,887 ± 6,653 | 363,176 | 75.1% | 2.7% |
| 11,250,000 | greedy | 318,408 ± 6,555 | 365,666 | 79.5% | 5.5% |
| 11,500,000 | greedy | 312,460 ± 6,690 | 359,740 | 78.7% | 4.6% |
| 11,750,000 | greedy | 315,379 ± 6,733 | 366,654 | 78.5% | 5.1% |
| 12,000,000 | depth 3 | 380,930 ± 5,644 | 382,542 | 93.4% | 17.1% |
| 12,000,000 | greedy | 314,550 ± 6,867 | 358,256 | 78.2% | 5.6% |
| 12,250,000 | greedy | 313,112 ± 6,760 | 366,042 | 77.8% | 4.4% |
| 12,500,000 | greedy | 318,942 ± 6,500 | 362,642 | 80.7% | 5.3% |
| 12,750,000 | greedy | 313,252 ± 6,748 | 364,852 | 78.1% | 5.1% |
| 13,000,000 | greedy | 310,501 ± 6,777 | 358,486 | 77.5% | 4.6% |
| 13,250,000 | greedy | 315,958 ± 6,621 | 366,046 | 79.2% | 5.1% |
| 13,500,000 | greedy | 315,746 ± 6,707 | 366,300 | 79.2% | 5.1% |
| 13,750,000 | greedy | 317,857 ± 6,749 | 363,626 | 79.8% | 5.9% |
| 14,000,000 | depth 3 | 384,536 ± 5,811 | 380,548 | 94.1% | 16.8% |
| 14,000,000 | greedy | 314,211 ± 6,860 | 365,838 | 77.3% | 5.2% |

## 7. Charts

- [ryzen-optimism/training_progress.png](ryzen-optimism/training_progress.png)
- [ryzen-optimism-ext/training_progress.png](ryzen-optimism-ext/training_progress.png)
- `figures/ryzen-arms.png` and `ryzen-recipe/training_progress.png` will be added with `recipe`'s results.

## 8. Extension: `optimism` from 10M to 14M games (added at the user's request)

Not part of the README plan. The user asked to continue `optimism` past 10M games if it hadn't plateaued, first to
12M and then to 14M. No code was changed: a script outside the repo, `~/ryzen-notes/run_optimism_ext.sh`, resumed
`checkpoints/ntuple/ryzen-optimism/games_10000000.npz` (weights and temporal coherence sums, no reset) into a separate run,
`ryzen-optimism-ext`, with the same settings as `optimism`'s last phase: TC at 0.02, 8 threads, the 8x6 network,
downgrading from 16384 and late-game starts from 8192. As in the main runs, there was a greedy evaluation on 1,000
seeded games every 250,000 games, and depth-3 evaluations on the same 1,000 games at 12M and 14M.

- **To 12M:** from 20:28:25 to 23:55 UTC on 2026-11-13 (`TOTAL=12000000`).
- **To 14M:** restarted at 23:55:27 from the 12M snapshot, with the log appended. Training finished at 03:05 on
  2026-11-14, and the depth-3 evaluation at 03:24.
- **Kept checkpoints** in `checkpoints/ntuple/ryzen-optimism-ext/`: 11M, 12M, 13M and 14M.
- **Run log:** `ryzen-optimism-ext/run.log`. Its full table is in section 6.

Greedy averages per million games, from the table in section 6:

| Games | Mean of the four greedy averages |
|---|---:|
| 9.25M–10M (`ryzen-optimism`, after TC came on at 9M) | 299,438 |
| 10.25M–11M | 308,669 |
| 11.25M–12M | 315,199 |
| 12.25M–13M | 313,952 |
| 13.25M–14M | 315,943 |

Depth-3 at the end of each stage:

| Games | Average ± 95% | ≥16384 | ≥32768 |
|---:|---:|---:|---:|
| 10M | 379,675 ± 5,602 | 92.6% | 16.8% |
| 12M | 380,930 ± 5,644 | 93.4% | 17.1% |
| 14M | 384,536 ± 5,811 | 94.1% | 16.8% |
