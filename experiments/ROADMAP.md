# Roadmap to 400k greedy

Written 2026-11-26, while the 12-tuple run was training. The goal is greedy play (no search at play time) above 400,000, inside 20M training games. Results are in [README.md](README.md); this file says what to do next and in what order.

## Where we are

| Player (50,000 paired games) | Score | ≥16384 | ≥32768 | Training games |
|---|---:|---:|---:|---:|
| 8×6 at 6.4M | 336,910 | 79.8% | 12.9% | 6.4M |
| 8×6 at 6.65M with expected backups | 341,704 | 80.9% | 13.3% | 6.65M |
| The same with both finishers | 351,390 | 81.8% | 17.9% | 6.65M |
| The same, halving repeatedly after a 32768 (A1) | 361,332 | 81.8% | 17.9% | 6.65M |
| **The same, with the finisher also working after a 32768 (A2)** | **362,514** | 81.8% | 17.9% | 6.65M |

The last two rows were measured from scratch scripts on 2026-11-26. Neither is in the repo's code yet.

Guei's networks, for comparison (thesis tables 7 to 13, checked against the PDF):

| Guei, 8×6 | Score | ≥16384 | ≥32768 | Training games |
|---|---:|---:|---:|---:|
| One stage | 370,194 | 85.3% | 22.2% | 100M |
| Two stages | 404,288 | 85.4% | 30.2% | 200M |
| Two stages, downgrading at play | 412,785 | 85.4% | 30.2% | 200M |
| Three stages | 412,492 | 85.3% | 33.6% | not stated |
| One stage, reset every 1M, snapshots averaged | 415,945 | 85.0% | 30.4% | 400M |

- **We're far ahead at equal games.** His curves sit at 224–285k between 10M and 20M games. We're at 362k after 6.65M.
- **His 16384 rate never moves.** It's 85.3–85.4% in every configuration. Everything he gains past 370k comes after 16384.
- **400k in 20M games would be a new result,** not a reproduction: his cheapest route to it took 200M.

## What 400k needs

Average score by the largest tile reached, for the best player:

| Largest tile | Share of games | Average score |
|---|---:|---:|
| Under 8192 | 4.1% | 55,042 |
| 8192 | 14.1% | 162,802 |
| 16384 | 63.9% | 352,574 |
| 32768 | 17.9% | 626,524 |

- **One more point of games reaching 16384 is worth about 1.9k.**
- **One more point reaching 32768 is worth about 2.7k.**

What different rates would score, with these averages:

| Reaching 16384 | Converting to 32768 | Reaching 32768 | Score |
|---:|---:|---:|---:|
| 81.8% (now) | 21.9% (now) | 17.9% | 362.5k |
| 85% | 21.9% | 18.6% | 370.7k |
| 85% | 30% | 25.5% | 389.5k |
| **85%** | **35%** | **29.8%** | **401.2k** |
| 88% | 30% | 26.4% | 398.9k |

So 400k is Guei's two rates. The reach to 16384 is worth about 8k and the conversion about 30k.

## What works, and what doesn't

**What works**

| Change | Gain | Evidence |
|---|---:|---|
| 8 tuples instead of 4 | +33% at 200k games | Network screen |
| Downgrading with late starts | +21.6k ± 4.5k | 3 seeds, 4×6 |
| Resetting the TC sums at a stall | +17.8k ± 10.3k | 1,000 games |
| Moving on at each stall | 335k at 4.6M against 327k at 8M | 5,000 games |
| Halving repeatedly after a 32768 | +9.9k ± 0.2k | 50,000 paired |
| The 32768 finisher with a work-area table | +7.9k ± 0.3k | 40,000 paired |
| Expected backups on crowded boards | about +2.5k | 50,000 paired |
| The 16384-window finisher | +1.6k ± 0.7k | 50,000 paired |
| The finisher after a 32768 | +1.2k ± 0.1k | 50,000 paired |

**What doesn't work well**

- **The 16384 stage leaks everywhere.**
  - From a first 16384, about 60% get the chain down to 2048 (derived, not measured directly).
  - Each of the next four steps keeps 86–91%.
  - The last stretch keeps 49%, or 64% with the finisher.
  - The same pattern now shows after a 32768: most of those games die with the chain rebuilt to the top.
- **Search only teaches in the last 64 moves.** Depth 3 on the current network is worse than greedy at the window's edge (43.8% against 49.4%), and worse over the whole stage (−9.5k). Depth 8 helps only at the very end.
- **The finisher is near its teacher.** 63.9% against the teacher's 68.1%, and stronger teachers added nothing.
- **More of anything on the same network is flat:** more games, a dedicated network for the 16384 stage, more late starts.
- **Sharing between stages doesn't transfer for free.** Halving from 8192 at play, without training for it, costs 78k (3,000 games). It worked after a 32768 only because that stretch plays exactly like the 16384 stage.

Put together, the leak is ordinary play quality on a crowded board, and neither more data nor a teacher fixes it. That leaves what the network can represent.

## What we had overlooked

1. **Positions the network reads with weights it never trained.** Play after a 32768 was one, worth 11k in total. Checked for others: boards with no gap are too rare to matter, and nothing else was found.
2. **Evaluation resolution.** Most early decisions were made on 1,000 games, which resolves about ±10k. Gains of 2–5k were invisible, so some "no gain" verdicts aren't safe. The cheap ones are retested in track A.
3. **Selection on the reporting seeds.** The 6.65M base was picked as the best of several snapshots on the same 50,000 seeds we report on, and snapshots wander by about ±2k. The headline needs seeds that were never used to choose anything.
4. **Large tuples over small tiles.** The work-area table is an 8-cell tuple that only sees tiles up to 128. It's the size of one 6-tuple, it gave the finisher most of its extra gain, and it has never been tried in the base network. Published 7-tuples are worth +12–13% at greedy but cost 16 times the memory. This is the cheap version of the same idea (experiment B2).
5. **One seed.** Every network descends from seed 42. A 2026 abstract from Matsuzaki's group reports that seeds settle into different styles, and that the difference shows in the 16384 to 32768 stretch.
6. **Late starts were never rechecked.** They were worth +21.6k at 300k games. Published results say starting games late costs greedy play.

## The plan

Rules for every experiment:

1. One change against a matched control.
2. 50,000 paired games before any claim under 5k.
3. The prediction and the pass mark are written down before it runs.
4. No edits to `deep_2048/` while a training run is live, because each 100k session re-imports the code.
5. Headline numbers are reported on seeds 100,000–149,999, which are used for nothing else.

### Track A: quick wins and one-hour tests

No training:

| | What | Status | Gain |
|---|---|---|---:|
| A1 | Halve repeatedly after a 32768 | In the repo: `downgrade_repeat` on a layout (`NTUPLE_DOWNGRADE_REPEAT` to train with it, `with_repeated_halving` to play a network trained without it), with tests. It replays the scratch script's 7,139 checked games exactly. | +9.9k ± 0.2k |
| A2 | Choose the finisher's window from the board halved once, so it opens while a second 16384 is built | In the repo under the same setting. A second 16384 is built in 21.3% of those games, up from 15.7%. The base with both finishers replays the scratch script's 9,437 checked games exactly (362,514 over the 50,000). | +1.2k ± 0.1k |

One hour of training each. All branch from the 6.4M network for 500k games at 0.005 with the sums kept, against the existing run with those settings (`expected-6.4m-e0`), on 50,000 paired games:

| | Question | Change | Pass mark | Prediction |
|---|---|---|---|---|
| A3 | Does training with the repeated halving help the 16384 stage? | Repeated halving in training | +2k, or +1 point of conversion | 0 to +3k. It adds about a sixth more 16384-stage games, and it gives the move that makes a 32768 a well-trained value to learn towards. |
| A4 | Do late starts still pay once 16384 is common? | No late starts | 2k either way decides it | Uncertain |
| A5 | Does TD(λ) help at a low rate? It lost at 0.08, the only rate tried, and both Jaśkowski and Stochastic MuZero use λ = 0.5. | λ = 0.5 over 3 moves | +2k | No gain. Our late values are over-scaled, but rescaling them didn't change greedy play. |

Results (2026-11-26), on 50,000 paired games against the control at the same games, with the average over each run's own 500k training games beside them. A3 is compared with the control playing with repeated halving, so the difference is the training alone.

| | Paired at 6.65M | Paired at 6.9M | Training games, 6.4M to 6.9M | Verdict |
|---|---|---|---|---|
| Control | 336,911 (344,243 with repeated halving at play) | 338,291 (345,838) | 337,827, 16384 in 80.07%, 32768 in 12.88% | — |
| A3 | 340,252, −3,991 ± 892 | 341,149, −4,689 ± 907, with 16384 −0.8 ± 0.3 points and 32768 −0.8 ± 0.2 | 16384 −0.53 points, 32768 −0.31 | **Fails.** Repeated halving is for play only. |
| A4 | 338,781, +1,871 ± 808 | 341,223, +2,931 ± 806, with 16384 +0.9 ± 0.3 points and 32768 −0.1 ± 0.2 | 339,772, +1,946, 16384 +0.42 points | **Passes at the mark.** No late starts in the last phase. |
| A5 | 338,822, +1,911 ± 808 | 339,303, +1,012 ± 810, with 16384 +0.3 ± 0.3 points and 32768 +0.5 ± 0.2 | 338,775, +948, 32768 +0.57 points (+0.86 in the second half) | **Under the mark.** A small, steady gain in conversion (16.9% against 16.3%). It can't be combined with expected backups, which are worth more. |

Snapshots of one run differ by more than the paired error suggests. Expected backups (e8) against the same control are +4,794 ± 805 at 6.65M and +356 ± 806 at 6.9M, while their training average is steady at +2.6k and +2.8k over the two halves. So a test of this size is read from both snapshots and the training average, not from one snapshot.

A4 goes into the final base's last phase, with expected backups. A3 and A5 don't.

### Track B: the base network

| | Question | Design | Cost | Pass mark | Prediction |
|---|---|---|---|---|---|
| B1 | Does a larger network have a higher ceiling? | The 12-tuple run on the stall schedule (running). Compared with the 8-tuple run at 2.4M and at each matching stall. | Overnight | +5k adopts it. Within ±2k means size isn't the lever. | **Passed at 2.4M: 317,336 against 305,681, +11,655 on 50,000 paired games** (± 870 on the 40,000). 16384 reached in 76.9% against 73.3%; the conversion to 32768 is unchanged (10.1% against 9.9%). **The lead was gone by the second stall:** 12 tuples at 3.7M against 8 tuples at 3.6M is 323,692 against 324,374, −682 on 50,000 paired games (± 863 on the 40,000), with 16384 reached in 79.2% against 77.3% but 32768 in 7.0% against 10.6%. The 8-tuple network's 32768 rate rose by 3 points in the reset phase and this one's didn't. Third stall (4.7M against 4.6M): 331,396 against 334,115, −2,720, with 16384 in 80.8% against 79.1% and 32768 in 8.2% against 12.5%. Fourth and last stall (5.7M against 5.6M): 328,917 against 330,357, −1,440, with 16384 in 80.4% against 78.6% and 32768 in 7.7% against 11.7%. So the 12-tuple network trained from scratch is not adopted. **Why:** the loss is all in play after a 16384 (the 8-tuple network playing on from the 12-tuple network's first 16384 makes a 32768 as often as it does alone, 10.8% against 10.9%). The two networks build the chain's second row differently. The 8-tuple network snakes back under the 2048 in 87% of games. The 12-tuple network starts the second row under the 16384 in 56% and misplaces the 1024 in another 20%, which converts at about 6%. The habit is learned one stage earlier: under an 8192 the 12-tuple network already starts the second row under the top tile in 96% of games, and there it converts as well as the snake (93.4% against 92.6%). It only costs under a 16384, which reads the same weights through halving. It was there at 2.5M and hasn't moved by 4.7M. Three 8-tuple networks trained from scratch on different schedules (the fixed path, scratch-scale, scratch-stall) all snake in 93–95% of games, so the layout looks like a consequence of the four extra shapes and not the luck of a run. A second 12-tuple run from scratch (seed 43, stopped at 500k) also goes under the top tile under an 8192 (81%), though at 500k it still snakes under a 16384 (82%), so how the 16384-stage habit forms is open. **Grown instead of from scratch:** the 8-tuple network at its 2.4M stall with the four shapes added at zero, then the rest of the schedule. It stalled at 3.4M. Against the 8 tuples at their 3.6M stall on 50,000 paired games: 324,933 against 324,374, +559 ± 794, with 16384 in 78.0% against 77.3% (+0.7 ± 0.3) and 32768 in 9.6% against 10.6% (−1.1 ± 0.2). It kept its parent's snake (96% under 8192, 90% under 16384) and still converts less often (12.2% against 13.8%), so the layout isn't the whole reason extra tuples cost conversions. Its training average sat at about 330k for the 500k games after the rate drop, level with the 8 tuples. Not adopted either: four more 6-tuples aren't the lever, from scratch or grown. |
| B2 | Do large tuples over small tiles help the base? | The 8 tuples plus two 8-cell tables for tiles up to 128: an edge strip and the middle strip. Read wherever every cell qualifies, and trained with everything else from the first game. | The code is in (`NTUPLE_SMALL_TABLES`, tested; training is about 18% slower with it). Then overnight: `methods-8x6-scratch-stall-small`, matched to the 8-tuple run | +5k over the same network without them at the same stall, read at the second stall as well as the first (B1's lead at the first stall didn't last) | Uncertain, and the most interesting test here. The same table failed when it learned alone beside a frozen network, which is a different setting. |
| B3 | Do full-range 7-tuples help? | The winner so far plus two 7-tuples (cells 0–6 and 4–10, the pair in Guei's code). About 8 GB in memory with the TC sums. | Overnight | +5k at the same stall | Smaller than B2. These tables fill slowly, and one abstract reports unstable learning with larger tuples. |
| B4 | Does a longer first phase lift where the later phases end? | Branch the 8-tuple run at its 2.4M stall, hold the first phase to 3.6M, then the same reset and cut. Compare with 4.6M (335k) and 6.4M (338.5k). | 3.4M games, 5 hours | +3k over the 4.6M network | No gain. The earlier jumps came from a new tile, and there isn't one left. |
| B5 | The final base | The winning network on the stall schedule, a second reset, expected backups in the last phase, and whatever passed in track A. | About 7–10M games | Beats 341.7k by the margins measured above | — |
| B6 | Is the result the seed? | B5's first phase again from another seed, compared at its first stall. | About 2.5M games | Within 3k means the seed doesn't matter | Unknown. It has never been measured at the plateau. |

**B2 at its first stall (27 November).** The run stalled at 3.6M games, against 2.4M for the 8 tuples, and carried on into the reset phase. Over 50,000 paired games, greedy, no finishers:

| Player | Avg score | ≥16384 | ≥32768 | 16384→32768 | Small-tile first stall minus it (± 1 se) |
|---|---|---|---|---|---|
| **8 tuples + small-tile tables, first stall (3.6M)** | **356,756** | 81.5% | 20.1% | 24.6% | — |
| **The same, halving repeatedly at play** | **370,221** | 81.5% | 20.1% | 24.6% | −13,465 ± 211 |
| 8 tuples, first stall (2.4M) | 305,681 | 73.3% | 7.3% | 10.0% | +51,075 ± 834 |
| 8 tuples at the same games (3.6M, second stall) | 324,374 | 77.3% | 10.6% | 13.8% | +32,383 ± 851 |
| 8 tuples, finished control base (6.65M) | 336,911 | 79.9% | 12.7% | 15.9% | +19,846 ± 849 |
| 8 tuples, finished base with expected backups (6.65M) | 341,704 | 80.9% | 13.3% | 16.4% | +15,052 ± 847 |
| That base with both finishers | 351,390 | 81.8% | 17.9% | 21.9% | +5,366 ± 870 |

- **It passes at the first stall** by ten times the +5k mark, stall against stall, and by six times at matched games. The second stall is still to come.
- **With repeated halving at play it scores 370,221**, above the previous best player's 362,514 (the 6.65M base with both finishers and A1 and A2; different players' games, not paired here). No finisher is involved.
- **The gain is conversion:** 24.6% against 10.0% at the first stalls, with 16384 reached in 8.3 points more games.
- An earlier copy of the run at 2.47M scored 347,469 over 10,000 games (+41,436 ± 1,846 on the 8 tuples' first stall).

**B2 at its second stall (27 November).** The reset phase ran from 3.6M to a stall at 4.8M (the 8 tuples: 2.4M to 3.6M), and the run went on to the 0.005 phase. Over the same 50,000 paired games:

| Player | Avg score | ≥16384 | ≥32768 | 16384→32768 | Small-tile second stall minus it (± 1 se) |
|---|---|---|---|---|---|
| **8 tuples + small-tile tables, second stall (4.8M)** | **362,729** | 82.8% | 21.1% | 25.4% | — |
| **The same, halving repeatedly at play** | **377,252** | 82.8% | 21.1% | 25.4% | −14,523 |
| Its own first stall (3.6M) | 356,756 | 81.5% | 20.1% | 24.6% | +5,973 ± 882 |
| 8 tuples, second stall (3.6M) | 324,374 | 77.3% | 10.6% | 13.8% | +38,355 ± 838 |
| 8 tuples, nearest in games (4.6M, third stall) | 334,115 | 79.2% | 12.5% | 15.7% | +28,614 ± 842 |
| 8 tuples, finished base with expected backups (6.65M) | 341,704 | 80.9% | 13.3% | 16.4% | +21,024 ± 840 |
| That base with both finishers | 351,390 | 81.8% | 17.9% | 21.9% | +11,338 ± 861 |

- **B2 passes at the second stall too:** +38k stall against stall and +29k at the nearest game count, against a +5k mark. B1's lead had gone by this point; this one has not.
- **The lead is smaller than at the first stall** (+51k stall against stall there). The reset phase added 6.0k to this network and 18.7k to the 8 tuples, whose first phase was 1.2M games shorter.
- **377,252 with repeated halving at play** is the best greedy player so far, with no finisher: +7,030 ± 1,037 on its own first stall played the same way.

**B2 at its third stall (27 November).** The 0.005 phase ran from 4.8M to a stall at 5.8M (the 8 tuples: 3.6M to 4.6M), and the run went on to its last phase at 0.00125. Over the same 50,000 paired games:

| Player | Avg score | ≥16384 | ≥32768 | 16384→32768 | Small-tile third stall minus it (± 1 se) |
|---|---|---|---|---|---|
| **8 tuples + small-tile tables, third stall (5.8M)** | **370,992** | 84.5% | 22.3% | 26.5% | — |
| **The same, halving repeatedly at play** | **386,505** | 84.5% | 22.3% | 26.5% | −15,513 |
| Its own second stall (4.8M) | 362,729 | 82.8% | 21.1% | 25.4% | +8,264 ± 873 |
| 8 tuples, third stall (4.6M) | 334,115 | 79.2% | 12.5% | 15.7% | +36,878 ± 841 |
| 8 tuples, nearest in games (5.6M, its fourth stall) | 330,357 | 78.6% | 11.7% | 14.9% | +40,635 ± 840 |
| 8 tuples, finished base with expected backups (6.65M) | 341,704 | 80.9% | 13.3% | 16.4% | +29,288 ± 838 |
| That base with both finishers | 351,390 | 81.8% | 17.9% | 21.9% | +19,602 ± 861 |

- **The lead holds through the lower rate:** +37k stall against stall, the same as at the second stall (+38k). The 0.005 phase added 8.3k to this network and 9.7k to the 8 tuples.
- **386,505 with repeated halving at play** is the best greedy player so far, with no finisher: +9,253 ± 1,036 on its own second stall played the same way.
- **Against 400k:** 16384 in 84.5% of games is at the roughly 85% needed; conversion is 26.5% against roughly 35%. Games that reach 32768 average 639,620 and games that stop at 16384 average 357,026.

**B2 finished (27 November).** The 0.00125 phase ran from 5.8M to a stall at 7.0M and the run ended there, 3M games under its budget (the 8 tuples ended at 5.6M). Over the same 50,000 paired games:

| Player | Avg score | ≥16384 | ≥32768 | 16384→32768 | Small-tile final minus it (± 1 se) |
|---|---|---|---|---|---|
| **8 tuples + small-tile tables, final (7.0M)** | **370,749** | 84.0% | 22.8% | 27.1% | — |
| **The same, halving repeatedly at play** | **386,652** | 84.0% | 22.8% | 27.1% | −15,903 |
| Its own third stall (5.8M) | 370,992 | 84.5% | 22.3% | 26.5% | −244 ± 871 |
| Its own first stall (3.6M) | 356,756 | 81.5% | 20.1% | 24.6% | +13,993 ± 887 |
| 8 tuples, final stall (5.6M) | 330,357 | 78.6% | 11.7% | 14.9% | +40,392 ± 848 |
| 8 tuples, best stall (4.6M) | 334,115 | 79.2% | 12.5% | 15.7% | +36,634 ± 846 |
| 8 tuples, after the second reset (6.4M) | 336,910 | 79.8% | 12.9% | 16.1% | +33,839 ± 843 |
| 8 tuples, finished base with expected backups (6.65M) | 341,704 | 80.9% | 13.3% | 16.4% | +29,044 ± 842 |
| That base with both finishers | 351,390 | 81.8% | 17.9% | 21.9% | +19,358 ± 862 |

- **B2 passes:** +40k at the end of the schedule and +37k against the 8 tuples' best stall, with no finisher and no expected backups.
- **The last phase added nothing:** −244 ± 871 on the third stall, and +147 ± 1,040 with repeated halving (386,652 against 386,505). The 8 tuples lost 3.8k in the same phase. The base to carry forward is either checkpoint; 5.8M costs 1.2M fewer games.
- **Against 400k:** 16384 in 84.0% of games and conversion 27.1%, against roughly 85% and 35%. With repeated halving, games that reach 32768 average 639,845 and games that stop at 16384 average 356,619.
- **Still one seed**, and the finishers and expected backups have not been tried on this base: the finishers need a code change to accept the small-tile tables.

**Finishers on B2 (27 November).** Both finishers were fitted on the 7.0M network with the 8 tuples' settings (40,000 attempts each), without the work-area table: each window now has its own copy of the small-tile tables, and the first of those reads the same 8 cells. Over the same 50,000 paired games:

| Player | Avg score | ≥16384 | ≥32768 | 16384→32768 | Minus the base played the same way (± 1 se) |
|---|---|---|---|---|---|
| Small-tile 7.0M | 370,749 | 84.0% | 22.8% | 27.1% | — |
| + 32768 finisher | 373,422 | 84.0% | 24.3% | 29.0% | +2,673 ± 285 |
| + 16384 finisher | 371,952 | 84.6% | 22.9% | 27.0% | +1,203 ± 694 |
| + both | 374,998 | 84.6% | 24.4% | 28.9% | +4,249 ± 705 |
| Small-tile 7.0M, repeated halving | 386,652 | 84.0% | 22.8% | 27.1% | — |
| + 32768 finisher, repeated halving | 390,878 | 84.0% | 24.3% | 29.0% | +4,226 ± 429 |
| **+ both, repeated halving** | **392,364** | 84.6% | 24.4% | 28.9% | +5,712 ± 892 |

- **The finishers add about half what they added to the 8 tuples** (+5.7k here, +13.3k there with repeated halving). From the 32768 window's edge this base already makes the tile in 65.4% of attempts against its depth-8 search's 73.3%; the 8 tuples' gap was 49.4% against 68.1%. The finisher closes about half of the smaller gap (69.4%).
- **392,364 is the best greedy player so far**, 7.6k short of 400k. With repeated halving the 16384 finisher adds +1,486 ± 897 to the 32768 finisher.
- Patches: `checkpoints/ntuple/finisher-small-7m/m64-d8-r0.01-natural-mistakes.npz` and `checkpoints/ntuple/finisher-small-7m-16384/m1024-d3-r0.01-natural-mistakes-rollout10.npz`.

B2 and B3 run whether or not B1 passes: tuple count, tuple size and small-tile tables are separate questions. If B1 to B4 all fail, the base is at its ceiling for this design.

### Track C: the late game on the final base

| | Question | Design | Cost | Pass mark | Prediction |
|---|---|---|---|---|---|
| C1 | Refit both finishers | The existing recipes on the new base, with A2's windows. | 3 hours | +9k over the base | It carries over. It did from 6.4M to 6.65M. |
| C2 | Can a rollout teacher help in the middle of the 16384 stage? | No training first. From 2,000 boards with the chain at 2048, measure the rollout teacher's reach to the last 64 against greedy. | 1 hour | +3 points, then train a finisher there | Uncertain. Depth 8 adds nothing there, but the rollout teacher did work before 16384 where depth 8 failed to teach. |
| C3 | Does the work-area table matter in the 16384 window? | The matched run without the table. | 80 minutes | A difference of 1k | Small |

### Track D: the spare budget

| | Question | Design | Cost | Pass mark |
|---|---|---|---|---|
| D1 | Do further reset cycles pay once the finishers are refitted? | A third reset on the final base, with the finishers refitted on both. | 1.8M games | +3k with finishers on both. The second reset's gain disappeared under that test. |

Guei's reset-and-average result used 400 cycles. We can afford perhaps seven, so D1 is a small bet.

### Order

1. **Done:** B1 (not adopted, from scratch or grown), and A1 and A2 in the repo.
2. **Done:** A3 fails, A4 passes, A5 is under the mark (`track-a-a3-repeat`, `track-a-a4-no-late-starts`, `track-a-a5-lambda`).
3. **Done:** the code for B2.
4. **Next, overnight:** B2.
5. **The night after:** B3 or B4, whichever B2's result makes more useful.
6. **Two nights:** B5, the final base. B6 alongside if the machine allows.
7. **After B5:** C1, then C2 and C3.
8. **Last:** D1 if games remain, then the headline on the held-out seeds.

### Where the gains could come from

| Step | Score after it | Basis |
|---|---:|---|
| Today, with A1 and A2 | 362.5k | Measured |
| Track A's training tests | about 364k | A4 measured at +2k to +3k on the base; untested with expected backups and the finishers |
| A better base, with 16384 reached in 85% | about 375k | Needs B1 or B2 to pass. B1 led by 11.7k at 2.4M, all of it from reaching 16384 more often, but was level by the second stall, and the same shapes grown onto the 8-tuple network were level too (+559 ± 794). |
| Conversion from 22% to 30% | about 390k | Needs the base to lift the whole 16384 stage |
| Conversion to 35% | about 401k | Guei's two-stage level |

Only the first row is measured. About 375k looks reachable with what we know. The last 25k needs the conversion to rise by half, and nothing we have tested does that. B2 is the experiment aimed at it; B1 may contribute.

## Closed: don't revisit

Each of these has been tested and lost. The README has the numbers.

- **Learning rules:** plain TD from scratch, TD then a TC finish, TC forgetting, AutoStep TIDBD, a constant rate after TC, optimistic weights in every form.
- **Second stages trained by TD:** about a dozen designs, on the 1.5M, 2M and 10M networks. All matched their base or lost. Guei's own second stage needs 9–15M of its own games before it first makes a 32768, and 100M in all.
- **Real tiles after 16384:** never learn to make a 32768.
- **Search as a teacher outside the last 64 moves:** stage 5, the offline fit, teaching depth 3 into the base's weights.
- **Averaging:** two independent runs, one lineage's snapshots, a constant-rate tail.
- **Small tuples:** added to a trained network or from scratch.
- **More of the same:** more games at a fixed rate, a larger share of late starts, a further rate cut, a dedicated network for the 16384 stage.
- **Halving from 8192 at play:** −78k without training for it.
- **Ruled out by choice:** search at play time as the headline, solved sub-games, optimistic initial weights.

One caution on this list. Most of these were judged on 1,000 games. A gain of 2–5k would have been invisible, so "lost" means "no large gain".

## What the literature adds

Checked on 2026-11-26 from the papers themselves, except where marked. Sources are in [RELATED_WORK.md](RELATED_WORK.md) or named below.

- **Nobody has clearly beaten Guei at greedy play.**
  - Stochastic MuZero's policy alone is level with him: about 415k at 1 simulation a move, read from its Fig. 2b. That took 10¹⁰ frames at 100 simulations each and a ResNet.
  - The best value network without search is a CNN at 364,083 (Wang, PhD thesis, Kochi University of Technology, 2025), the best of many checkpoints on about 300 games each. Our 362k on 50,000 games is level with it.
- **More tuples gives diminishing returns.**
  - Guei, 100M games: 6, 7 and 8 six-tuples score 338k, 360k and 371k.
  - Oka and Matsuzaki (2017), 5M games of plain TD: 5 tuples 214.5k, 6 tuples 212.4k, 8 tuples 230.5k (read from page images of a reprint; the 5- and 8-tuple figures were checked against its text).
  - Nothing is published for more than 8 six-tuples chosen with care, so B1 is new ground.
- **Larger tuples help at greedy.** Jaśkowski's five 7-tuples score 284,657 against 251,033 for five 6-tuples at 4×10¹⁰ actions. Matsuzaki's 8×7 scores 255,198 against 226,958 for his 8×6 (budgets not stated). This is B2 and B3.
- **Starting games from late positions costs greedy play elsewhere.** Matsuzaki's restart is about −10% at greedy and +30% at depth 3 (read from his figures). Jaśkowski's carousel shaping is "consistently slightly negative" at greedy. This is A4.
- **Multi-stage networks pay late and need help.** Matsuzaki's stages beat one stage at greedy only from 2×10¹⁰ actions. Jaśkowski's 16 stages lose to one stage without weight promotion (226k against 251k). Yeh's three stages lowered greedy play at equal total games. This agrees with our closed list.
- **Search-chosen moves in training hurt greedy play.** Matsuzaki (2018): 238k with greedy training, 210k with a quarter of moves from 2-ply search, 141k with three quarters. Guei's MCTS targets do the same. This agrees with our stage 5, and with the finisher working only as a narrow correction.
- **Averaging over every spawn is the one published target that helped a greedy player:** 248,575 to 337,167 on Wang's CNN. Ours gave about +2.5k on a network that was already at 337k.
- **Play after a 32768 in Guei's work.**
  - His tables stop at the 32768 rate. Nothing is published on how often greedy games build a 16384 beside a 32768.
  - His downgrading halves once, from the move that makes the 32768. His second stage trained on real 16384 boards, so one halving is enough for him. Ours trains on halved boards, which is why we halve again.
  - His gain is +8.5k over all games, about +28k a game that reaches 32768 (derived from his table 9). Ours is +11.1k with A2, about +62k a game, from a lower starting point.

## Budget

| Use | Games |
|---|---:|
| Best lineage so far | 6.65M |
| Cap | 20M |
| B5 as planned | 7–10M |
| B6, if it counts | 2.5M |

Test branches (track A, B1 to B4) don't count unless their network is adopted.

## Still unknown

- **Tuples beyond Matsuzaki's first eight.** His ranked list past eight and his 7-tuple ranking aren't available: the papers are paywalled, and his open papers reprint only the eight we already use. The four extra tuples in B1 were chosen by shape.
- **Whether small-tile tables can learn by TD in the base.** B2 is the test.
- **How much of any result is the seed.** B6 is the test.
