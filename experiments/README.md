# Experiments

One folder per run, committed so every result can be traced to the code and settings that produced it.

| File | Contents |
|---|---|
| `manifest.json` | Each training session: command, effective settings, git commit, package versions, machine, start and end |
| `training_history.csv` | Training progress, one row per logged batch |
| `evaluations.json` | Per-game scores and max tiles for each evaluated checkpoint, with the commit and environment |
| `NOTES.md` | What was run, how to reproduce it, and what was observed |

Checkpoints aren't committed. Training is deterministic, so the commit and command rebuild them exactly.

Thematic notes for the write-up: [related work](RELATED_WORK.md) and [learning rules: TD, temporal coherence and TD(λ)](learning-rules.md).

```bash
make train-ntuple RUN=my-run ARGS="--games 100000"     # records to experiments/my-run
make watch-ntuple RUN=my-run                           # live chart of that run
uv run python scripts/evaluate.py checkpoints/ntuple/my-run/latest.npz --agent ntuple -n 1000 --experiment experiments/my-run
```

## Runs

| Run | What |
|---|---|
| [`ntuple-td-1m`](ntuple-td-1m/NOTES.md) | N-tuple TD from scratch to 1M games, constant learning rate |
| [`dqn-1m`](dqn-1m/NOTES.md) | The DQN's 1M-step training run |
| [`heuristic-depth2`](heuristic-depth2/NOTES.md) | Hand-crafted heuristic with depth-2 expectimax (no training) |
| [`ntuple-1m-cont-lr0.0025`](ntuple-1m-cont-lr0.0025/NOTES.md) | 1M snapshot trained to 1.5M at the original learning rate (control) |
| [`ntuple-1m-cont-lr0.001`](ntuple-1m-cont-lr0.001/NOTES.md) | 1M snapshot trained to 1.5M at learning rate 0.001 |
| [`ntuple-1m-cont-lr0.0004`](ntuple-1m-cont-lr0.0004/NOTES.md) | 1M snapshot trained to 1.5M at learning rate 0.0004 |
| [`ntuple-1m-cont-lr-decay`](ntuple-1m-cont-lr-decay/NOTES.md) | 1M snapshot trained to 1.5M with the rate decaying from 0.0025 to 0.0001 |
| [`ntuple-1m-cont-tc`](ntuple-1m-cont-tc/NOTES.md) | 1M snapshot trained to 1.5M with temporal coherence learning |
| [`ntuple-tc-1m`](ntuple-tc-1m/NOTES.md) | Temporal coherence from scratch to 1M games, base rate 0.0025 |
| [`ntuple-tc-1m-lr0.01`](ntuple-tc-1m-lr0.01/NOTES.md) | Temporal coherence from scratch to 1M games, base rate 0.01 |
| `screen-*` (9 runs) | [Network screen](#network-screen): layout × temporal coherence rate, from scratch to 200k games |
| `sharing-4x6-*` (4 runs) | [Scale sharing](#scale-sharing): control, late-game starts (C), downgrading (A), downgrading + scale term (B) |
| `threads-4x6-t8-*` (2 runs) | [Multi-threaded training](#multi-threaded-training): lock-free 8-thread training vs single-threaded |
| `seed-4x6-tc0.08-*` (2 runs) | [Seed-to-seed variation](#seed-to-seed-variation): the screen's 4×6 at 0.08 configuration with seeds 7 and 123 |
| `ingredients-4x6-*` (15 runs) | [Ingredient test across seeds](#ingredient-test-across-seeds): control, A, C, A + C and B′, seeds 1–3 |
| `lambda-4x6-*` (9 runs) | [TD(λ) across seeds](#tdλ-across-seeds): Jaśkowski's λ = 0.5, a longer horizon and an annealed λ, on A + C, seeds 1–3 |
| [`long-8x6`](long-8x6/NOTES.md) | [The long run](#the-long-run): 8×6 at 0.02 with A + C, on 8 threads to 1.5M games, then search |
| [`search-long-8x6`](search-long-8x6/NOTES.md) | [Faster search](#faster-search): exact and pruned search at depths 3–6 on the long run's 1.5M network |
| `decision-8x6-*` (4 runs) | [Decision arms](#decision-arms): control, TC reset, constant rate and a larger late-game share, each 1.5M → 2M games from the long run |
| `decision2-8x6-*` (3 runs) | [Decision arms, round 2](#round-2-do-resets-keep-paying): continue, reset again and a gentle reset, each 2M → 2.5M games from round 1's TC reset |
| `decision3-8x6-*` (2 runs) | [Decision arms, round 3](#round-3-optimism-for-never-updated-weights): optimism at 1,000 and 5,000 for never-updated weights, each 2M → 2.5M games from round 1's TC reset |
| [`long-8x6-dg32768`](long-8x6-dg32768/NOTES.md) | [Downgrading from 32768](#downgrading-from-32768-instead-of-16384): the long run from scratch, sharing only the 32768 stage, with evaluations every 250k games |
| [`decision5-8x6-two-stage`](decision5-8x6-two-stage/NOTES.md) | [Decision arms, round 5](#round-5-a-second-stage): a second stage from 16384, copied from the long run's 1.5M network, 1.5M → 2M games |
| `decision6-8x6-*` (2 runs) | [Decision arms, round 6](#round-6-temporal-coherence-with-forgetting): temporal coherence forgetting at 0.999 and 0.9999, each 2M → 2.5M games from round 1's TC reset |
| [`decision8-8x6-two-stage-by-state`](decision8-8x6-two-stage-by-state/NOTES.md), [`decision9-8x6-two-stage-by-state`](decision9-8x6-two-stage-by-state/NOTES.md) | [Rounds 8 and 9](#rounds-8-and-9-a-second-stage-chosen-by-the-board-a-move-is-made-from): the second stage chosen by the board a move is made from, 1.5M → 2M → 2.5M games |
| [`decision10-8x6-two-stage-kept-sums`](decision10-8x6-two-stage-kept-sums/NOTES.md) | [Round 10](#round-10-a-second-stage-without-the-reset): a second stage with the coherence sums kept, 1.5M → 2M games |

## Learning at equal experience

1,000 seeded games each, 95% confidence intervals. Experience is the number of moves played during training.

| Agent | Moves trained | Avg score |
|---|---:|---:|
| DQN step_100083 | 0.10M | 2,228 ± 74 |
| N-tuple, 1k games | 0.26M | 4,331 ± 107 |
| DQN step_350068 | 0.35M | 2,631 ± 87 |
| DQN step_625066 | 0.63M | 3,534 ± 108 |
| DQN best | 0.93M | 5,689 ± 170 |
| N-tuple, 3k games | 1.1M | 6,953 ± 168 |
| Heuristic depth 2 | none | 19,981 ± 597 |
| N-tuple, 100k games | 158M | 53,748 ± 1,265 |
| N-tuple, 500k games | 1.55B | 102,693 ± 2,598 |

The n-tuple network learns more from the same experience. At 0.26M moves it beats the DQN at 0.35M by 65% and at 0.63M by 23%, and at 1.1M moves it beats the DQN's best (0.93M) by 22%. It also gathers that experience in about a second on one CPU core, where the DQN's took hours on a GPU, so it can go on to over 1,000 times as much.

## Learning rate after the plateau

`ntuple-td-1m` stops improving after about 500k games at learning rate 0.0025. Three runs resumed its 1M snapshot for another 500k games, identical except for the learning rate. 1,000 seeded games each, 95% confidence intervals:

| Learning rate | 1.25M games | 1.5M games | ≥8192 at 1.5M | ≥16384 at 1.5M |
|---|---:|---:|---:|---:|
| 0.0025 (control) | 103,098 ± 2,647 | 102,220 ± 2,751 | 55.5% | 0% |
| 0.001 | 111,344 ± 2,643 | 113,692 ± 2,684 | 66.5% | 0% |
| 0.0004 | 120,945 ± 2,650 | 118,983 ± 2,664 | 68.9% | 0.4% |

![Learning-rate comparison](figures/lr-comparison.png)

- Lowering the rate to 0.0004 lifts the score by about 17%, at no extra cost.
- The gain appears within a few thousand games of the change, then each rate settles on its own plateau: 1.25M and 1.5M are level within noise for all three.
- That pattern suggests the original rate kept the weights noisy rather than stopped them learning. A schedule that keeps decaying, or a per-weight adaptive rate, is the natural next test.

## Learning-rate methods

Two more continuations from the same 1M snapshot and seed: a rate that keeps decaying (0.0025 to 0.0001, geometrically) and temporal coherence (per-weight rates, base 0.0025). 1,000 seeded games each, 95% confidence intervals:

| Method | 1.25M games | 1.5M games | ≥8192 at 1.5M | ≥16384 at 1.5M |
|---|---:|---:|---:|---:|
| Constant 0.0025 (control) | 103,098 ± 2,647 | 102,220 ± 2,751 | 55.5% | 0% |
| Constant 0.0004 | 120,945 ± 2,650 | 118,983 ± 2,664 | 68.9% | 0.4% |
| Decay 0.0025 → 0.0001 | 122,626 ± 2,626 | 123,723 ± 2,551 | 73.4% | 0.4% |
| Temporal coherence | 126,803 ± 2,613 | 128,208 ± 2,640 | 74.2% | 0.7% |

![Learning-rate methods](figures/lr-methods-comparison.png)

- Temporal coherence is best at both snapshots: 25% above the control and about 8% above the best constant rate. It needs no rate to be tuned, gets there within a few thousand games, and was still creeping up at 1.5M.
- The decaying rate catches up as it falls, finishing about 4% above the best constant rate. Going below 0.0004 helps a little, not a lot.
- Temporal coherence costs about a third of training speed and triples the checkpoint size.

## Temporal coherence from scratch

Two runs trained from scratch with temporal coherence, using the same seed and snapshots as `ntuple-td-1m` (constant rate 0.0025). 1,000 seeded games per snapshot, 95% confidence intervals, with the moves each had trained on:

| Games | Constant 0.0025 | TC, base 0.0025 | TC, base 0.01 |
|---:|---:|---:|---:|
| 1k | 4,331 ± 107 (0.3M moves) | 4,625 ± 114 (0.3M) | 7,381 ± 178 (0.4M) |
| 3k | 6,953 ± 168 (1.1M) | 6,884 ± 165 (1.1M) | 11,218 ± 272 (1.6M) |
| 10k | 10,648 ± 254 (5.0M) | 10,790 ± 262 (5.2M) | 18,504 ± 465 (7.6M) |
| 100k | 53,748 ± 1,265 (158M) | 32,300 ± 815 (114M) | 77,593 ± 1,760 (236M) |
| 500k | 102,693 ± 2,598 (1.55B) | 84,955 ± 2,028 (1.35B) | 120,918 ± 2,633 (2.09B) |
| 1M | 101,376 ± 2,582 (3.69B) | 112,398 ± 2,633 (3.47B) | 125,498 ± 2,498 (4.71B) |

![Temporal coherence from scratch](figures/tc-from-scratch.png)

- With the same base rate, temporal coherence learns more slowly through the middle of training (32k vs 54k at 100k games), because it can only lower each weight's rate and many weights' errors disagree early on. It overtakes by 1M games (112k vs 101k) and was still rising, where the constant rate had plateaued.
- With a 4x higher base rate, temporal coherence is ahead at every snapshot: it reaches the constant run's final level in about 250k games and finishes at 125k, on par with the best continuation runs, in a single from-scratch run with no rate schedule.
- It also wins on experience, not just games: 7,381 after 0.4M moves, above the constant run at 1.1M (6,953) and the DQN's best at 0.93M (5,689 ± 170).
- Taken with the continuation runs, temporal coherence works best as a safety brake on an aggressive rate rather than on an already careful one.

## Expectimax search on the n-tuple network

`ntuple_search` plays by expectimax: each move is worth its reward plus the expected value over every possible tile spawn, down to a fixed depth, where the network values the afterstate. Depth 1 is the greedy agent. No retraining is involved. Evaluated on the two strongest networks, 1,000 seeded games each, 95% confidence intervals:

| Network | Greedy | Depth 2 | Depth 3 | ≥8192 at depth 3 | ≥16384 at depth 3 |
|---|---:|---:|---:|---:|---:|
| [`ntuple-tc-1m-lr0.01`](ntuple-tc-1m-lr0.01/NOTES.md), 1M games | 125,498 ± 2,498 | 153,946 ± 1,631 | 162,469 ± 1,497 | 97.4% | 2.4% |
| [`ntuple-1m-cont-tc`](ntuple-1m-cont-tc/NOTES.md), 1.5M games | 128,208 ± 2,640 | 167,654 ± 2,657 | 202,291 ± 3,457 | 98.9% | 42.6% |

- Search adds 30-58% at depth 3. Nearly every game reaches 8192, against about 72% greedy.
- The networks are level when greedy but split apart under search. With depth 3, the one trained for 1.5M games reaches 16384 in 43% of games, against 2% for the 1M-game network. Greedy play rarely survives long enough for the 16384-stage weights to matter, so greedy evaluation understates the value of longer training.
- Search is cheap: a depth-3 game takes about 3 seconds alone, or 6 per core with all cores busy, mostly in the Python game engine. Depth 4 takes about 70 seconds alone.

## Network screen

Which network layout and learning rate learn most from a small budget? Nine runs from scratch to 200k games, all with temporal coherence and the same seed. The 8×6 layout reads 64 weights per board against 32 for the others, so a given per-weight rate takes twice the total step on it. 1,000 seeded greedy games per snapshot, 95% confidence intervals:

| Run | 50k games | 100k games | 200k games | ≥8192 | ≥16384 | Moves to 200k |
|---|---:|---:|---:|---:|---:|---:|
| [ours (2×4 + 2×6), 0.01](screen-2x4-2x6-tc0.01/NOTES.md) | 52,734 ± 1,210 | 77,593 ± 1,760 | 98,747 ± 2,402 | 43.2% | 0% | 626M |
| [4×6, 0.005](screen-4x6-tc0.005/NOTES.md) | 44,827 ± 1,251 | 58,771 ± 1,717 | 77,182 ± 2,218 | 20.9% | 0% | 522M |
| [4×6, 0.01](screen-4x6-tc0.01/NOTES.md) | 58,469 ± 1,733 | 76,492 ± 2,212 | 106,255 ± 2,886 | 49.1% | 1.1% | 671M |
| [4×6, 0.02](screen-4x6-tc0.02/NOTES.md) | 73,743 ± 2,177 | 100,020 ± 2,903 | 129,929 ± 3,319 | 66.6% | 2.8% | 817M |
| [4×6, 0.04](screen-4x6-tc0.04/NOTES.md) | 91,223 ± 2,522 | 119,353 ± 3,135 | 149,304 ± 4,042 | 74.1% | 13.4% | 949M |
| [4×6, 0.08](screen-4x6-tc0.08/NOTES.md) | 109,077 ± 2,904 | 132,332 ± 3,317 | 154,758 ± 4,066 | 77.4% | 14.9% | 1,030M |
| [8×6, 0.005](screen-8x6-tc0.005/NOTES.md) | 70,572 ± 2,032 | 97,973 ± 2,949 | 131,314 ± 3,651 | 63.2% | 5.0% | 809M |
| [8×6, 0.01](screen-8x6-tc0.01/NOTES.md) | 92,877 ± 2,817 | 128,754 ± 3,721 | 172,244 ± 5,167 | 78.2% | 26.8% | 1,017M |
| [8×6, 0.02](screen-8x6-tc0.02/NOTES.md) | 121,046 ± 3,376 | 165,418 ± 4,568 | **205,447 ± 5,807** | **86.7%** | **41.2%** | 1,244M |

The ≥8192 and ≥16384 columns are at 200k games.

![4×6 rate sweep](figures/network-screen-4x6.png)
![8×6 rate sweep](figures/network-screen-8x6.png)

- **Bigger layouts learn more per game.** At the same total step (8×6 at 0.01 vs 4×6 at 0.02), 8×6 scores 33% more at 200k games.
- **Higher rates win at this budget.** Each doubling helped up to 0.04–0.08 on 4×6, where gains flatten. Temporal coherence keeps the high rates stable.
- **The 8×6 at 0.02 beats Yeh et al.'s 159,565 greedy (15M games) by 29% after 200k games,** and matches this project's earlier best agent *with* depth-3 search. Playing greedily, it reaches 16384 in 41% of games. It was still climbing steeply at 200k.
- **Cost:** 8×6 trains at roughly a third of 4×6's speed. With up to seven runs sharing the machine, these runs' games/s aren't representative of solo speed.

## Scale sharing

Published comparisons are in [RELATED_WORK.md](RELATED_WORK.md). The weakness they point to is the late game: n-tuple weights are indexed by exact tile values, so a 16384-8192-4096 chain and an 8192-4096-2048 chain share nothing, and the rare late-game positions leave their weights undertrained.

**The property all three designs rely on.** Suppose a board is missing some tile value below its largest tile (a gap). Doubling every tile above the gap, or halving it, gives a board with the same legal moves, merges and slides. Equal big tiles stay equal, and the new values land on the missing one, so they can't match a small tile. Only the rewards for merging big tiles change, and only until a small-tile merge fills the gap. Guei uses halving ("tile-downgrading") at play time only.

| Design | Idea | Main risk |
|---|---|---|
| A. Downgraded learning | Past a threshold, the network reads and trains on the halved board, reusing well-trained smaller-scale weights | Positions with different true returns share one value |
| B. A + scale term | As A, plus a small table indexed by how many tiles of each large size the real board has (Yeh's large-tile feature), so patterns are shared and the scale term learns the value offset | More to tune |
| C. Synthetic late-game starts | Double the tiles above a gap in positions from ordinary play to make valid larger-scale positions, and start some training games from them. Learning itself stays exact. | Synthetic positions may not match those reached in play |

The order is C, then B: C is the safest and answers "does more late-game experience help", and B is the most novel (it's Guei's proposed future work). A is B without the scale term, so it serves as B's ablation. Each is tested at 200–500k games against the best configuration from the network screen.

### Results

The 4×6 network with temporal coherence at base rate 0.08, four runs from scratch to 300k games with the same seed, differing only in the feature tested. 1,000 seeded greedy games per snapshot. The difference from the control is a paired per-game comparison on the same seeds, with a 95% confidence interval:

| Run | 100k | 200k | 300k | ≥16384 at 300k | vs control at 300k | Moves to 300k |
|---|---:|---:|---:|---:|---:|---:|
| [Control](sharing-4x6-control/NOTES.md) | 132,332 | 154,758 | 172,077 ± 4,519 | 24.4% | — | 1.69B |
| [**A: downgrading**](sharing-4x6-downgrade/NOTES.md) | 135,726 | 155,669 | **181,790 ± 5,015** | 27.5% | **+9,714 ± 6,729** | 1.72B |
| [C: late-game starts](sharing-4x6-late-starts/NOTES.md) | 133,603 | 157,741 | 177,324 ± 5,140 | **28.5%** | +5,247 ± 6,830 | **1.55B** |
| [B: downgrading + scale term](sharing-4x6-downgrade-scale/NOTES.md) | 113,764 | 144,132 | 163,073 ± 5,116 | 23.2% | −9,004 ± 6,886 | 1.55B |

![Scale sharing](figures/scale-sharing.png)

- **Downgrading (A) helps: +5.6% at 300k, significant at 95%.** It also raises the 16384 rate, and produced the only 32768 tile in these evaluations. It tracks the control until 16384 tiles appear, then pulls ahead. This is Guei's proposed future work (downgrading during training), and the first evidence that it pays off.
- **Late-game starts (C) look positive but aren't conclusive:** +3.0%, with the interval still including zero. They also used 9% fewer moves than the control for the same number of games, since synthetic games start mid-game, so per unit of experience C is somewhat stronger than its score suggests. It has the highest 16384 rate.
- **The scale term (B) hurts: −5.2%, significant.** B falls behind from the very first thousand games, before any large tile exists. That rules out downgrading as the cause, and points to the scale table's "no large tiles" entry acting as one shared bias weight updated at the full rate. A version that skips that entry, or gives the table its own lower rate, is untested.
- Training is deterministic: an earlier run of the same script on an identical machine reached exactly the same 300k training averages for all four runs. That attempt's records were discarded, because a smoke test with the same run names overlapped it. The script now refuses to start a run that's already training, and smoke tests use separate `smoke-*` names.

### Running it

```bash
uv sync
scripts/run_scale_sharing.sh                              # about 1.5–2 hours on a 10-core M4
GAMES=3000 EVAL_GAMES=20 scripts/run_scale_sharing.sh     # smoke test as separate smoke-* runs, about a minute
make watch-ntuple ARGS="--run sharing-4x6-control sharing-4x6-late-starts sharing-4x6-downgrade-scale sharing-4x6-downgrade"
```

The runs need about 1 GB of RAM each and about 13 GB of disk for checkpoints (`checkpoints/` is gitignored).

## Multi-threaded training

`NTUPLE_THREADS=n` plays games on n threads that update the shared weights without locks, as in Jaśkowski (2018). The first `NTUPLE_THREADS_AFTER` games (default 10,000) stay single-threaded, since a fresh network collides often. Threaded runs aren't bit-reproducible.

**Speed.** Measured from a 200k-game snapshot of the 4×6 network with temporal coherence:

| Threads | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| Moves/s | 1.06M | 2.17M | 3.86M | 5.04M |
| Speed-up | 1× | 2.0× | 3.6× | 4.8× |

Scaling is near-linear up to the M4's 4 performance cores, and threads beyond that land on efficiency cores. Each 200k-game threaded run took about 3.5 minutes.

These runs used numba's parallel loops (commits up to `7f28a17`). Those can't compile layouts with scale sharing, so threading now runs the GIL-free game loop on plain threads, handing out small shares of each batch as threads free up. On the same 200k-game network and 1,000-game batch, both versions reach the same speed on 8 threads: 3.99M moves/s for the new one, 3.93–3.98M for the old. The table above comes from a different measurement, timing training batches, so it isn't directly comparable.

**Quality.** Two 8-thread runs, A and B, of the 4×6 at rate 0.08 to 200k games, against the single-threaded [`screen-4x6-tc0.08`](screen-4x6-tc0.08/NOTES.md). 1,000 seeded greedy games each. The difference is paired per game on the same seeds:

| Run | 100k games | 200k games | ≥16384 at 200k | vs single-threaded at 200k |
|---|---:|---:|---:|---:|
| Single-threaded | 132,332 ± 3,317 | 154,758 ± 4,066 | 14.9% | — |
| [8 threads, A](threads-4x6-t8-a/NOTES.md) | 134,380 ± 3,376 | 154,931 ± 3,847 | 13.4% | +172 ± 5,649 |
| [8 threads, B](threads-4x6-t8-b/NOTES.md) | 131,040 ± 3,450 | 146,648 ± 3,594 | 8.9% | −8,110 ± 5,328 |

![Threaded vs single-threaded training](figures/threads.png)

- Run A matches single-threaded training. Run B is about 5% lower at 200k, beyond evaluation noise.
- A and B differ only by chance, in the order threads touched the weights and in each thread's random games, yet they differ from each other by as much as B does from the single-threaded run. So B's deficit may be ordinary run-to-run variation rather than a cost of threading.
- **Resolved by the [seed check](#seed-to-seed-variation): threading costs nothing measurable.** Single-threaded runs with different seeds spread just as widely (148.2k–154.8k at 200k). The average of the two threaded runs is within 900 points of the average of the three single-threaded ones, and B sits just below the lowest seed.
- Smoothed training curves agree. All five runs stay within about 2% of each other up to 100k games, and spread only in the late game, where single-threaded runs also vary. So a longer single-threaded warm-up wouldn't help: the default 10,000 games is enough.

## Seed-to-seed variation

Every other comparison in this archive uses seed 42. To measure how much that matters, the screen's 4×6 at 0.08 configuration was trained single-threaded with seeds 7 and 123 as well. 1,000 seeded greedy games each:

| Run | 100k games | 200k games | ≥16384 at 200k |
|---|---:|---:|---:|
| Seed 42 ([`screen-4x6-tc0.08`](screen-4x6-tc0.08/NOTES.md)) | 132,332 ± 3,317 | 154,758 ± 4,066 | 14.9% |
| [Seed 7](seed-4x6-tc0.08-s7/NOTES.md) | 130,461 ± 3,435 | 152,045 ± 3,961 | 13.2% |
| [Seed 123](seed-4x6-tc0.08-s123/NOTES.md) | 132,397 ± 3,461 | 148,188 ± 3,895 | 11.9% |
| Mean ± standard deviation of seeds | 131,730 ± 1,100 | 151,664 ± 3,301 | |
| 8 threads, A / B | 134,380 / 131,040 | 154,931 / 146,648 | 13.4% / 8.9% |

![Seeds and threads](figures/seeds-and-threads.png)

Runs vary by about ±2% between seeds at 200k games (one standard deviation, 3.3k). They agree to within about 1% at 100k and spread as the late game arrives. For the single-seed comparisons elsewhere in this archive:
- **Robust,** well beyond seed variation: bigger layouts and higher rates in the network screen (8×6 is 33% above 4×6 at the same total step), temporal coherence over constant rates, and downgrading (A, +9.7k at 300k, about 3 standard deviations).
- **Not conclusive:** late-game starts (C, +5.2k), 4×6 at 0.08 vs 0.04 (+3.6%), and the earlier single-run learning-rate comparisons where differences are within a few percent.
- Effects of a few percent need several seeds per arm, which multi-threaded training now makes affordable.

## Ingredient test across seeds

The single-seed scale-sharing test left late-game starts (C) inconclusive, and the [seed check](#seed-to-seed-variation) showed single runs vary by about ±2%. So each ingredient was retested with three seeds:
- **The arms:** control, downgrading (A), late-game starts (C), both together (A + C), and B′.
- **B′** is the scale term applied only to downgraded boards, a redesign of B intended to avoid its shared-bias problem.
- **The setup:** the 4×6 at temporal coherence rate 0.08, trained from scratch to 300k games on 8 threads, with seeds 1–3, and evaluated on 1,000 seeded greedy games.

| Arm | Seed 1 | Seed 2 | Seed 3 | Mean of seeds ± sd | vs control (paired by seed) | ≥16384 | Moves |
|---|---:|---:|---:|---:|---:|---:|---:|
| Control | 162,538 | 162,231 | 171,064 | 165,278 ± 5,013 | — | 20.6% | 1.67B |
| A: downgrading | 170,248 | 158,596 | 179,014 | 169,286 ± 10,243 | +4,008 ± 6,621 | 19.8% | 1.67B |
| C: late-game starts | 175,163 | 173,399 | 168,682 | 172,415 ± 3,351 | +7,137 ± 8,276 | 24.9% | 1.52B |
| **A + C** | **186,413** | **186,809** | **187,498** | **186,906 ± 549** | **+21,629 ± 4,513** | **31.2%** | 1.59B |
| B′: downgrading + scale term on downgraded boards | 151,519 | 175,887 | 180,914 | 169,440 ± 15,722 | +4,163 ± 13,284 | 20.4% | 1.69B |

The ≥16384 and moves columns are averages over the three seeds.

![Ingredient test, seed 2](figures/ingredients-seed2.png)

- **A + C is a robust +13%.** All three seeds gain 16–25k, and the mean gain is about 8 standard errors from zero. It also raises the 16384 rate by half, uses fewer moves than the control, and is strikingly consistent: its seeds differ by 1.1k, against 9k for the control.
- **The two ingredients are much stronger together than apart.** Alone, each gives a small gain that isn't consistent across seeds (A +4.0k, C +7.1k). Together they give +21.6k, about twice the sum. A likely mechanism: late-game starts supply the rare late-game experience, and downgrading lets it train weights shared across tile scales, so each synthetic game teaches every scale at once.
- **B′ is erratic** (−11k, +14k, +10k). Restricting the scale term to downgraded boards removed B's early damage, but the few heavily shared scale weights still make late-game learning noisy. It's left out.
- The earlier single-seed results fit this picture. A's +9.7k with seed 42 was above anything seen across these three seeds (−3.6k to +8.0k), so that test happened to land on a favourable seed. C's +5.2k sits within its range.

## TD(λ) across seeds

Jaśkowski (2018) trained with delayed TD(λ): each afterstate's update waits a few moves, then moves towards a blend of the real rewards over the next h moves and the value estimate after them. It's meant to carry information back faster than TD(0)'s one-move lookahead, which suits this project's sample-efficiency goal. This test asks whether it helps here, and whether two variants improve on his setting.

- **The arms:**
  - Close to Jaśkowski's setting: λ = 0.5 with horizon 3. His update sums h + 1 errors, so his h = 3 is horizon 4 here, and this arm is one step shorter.
  - A longer horizon: λ = 0.75 with horizon 6, to carry information further back.
  - Annealed: λ = 0.75 with horizon 6, falling linearly to TD(0) over the session. The idea was to get the fast early learning from long returns and the low noise of TD(0) at the end.
- **The setup:** the A + C recipe from the [ingredient test](#ingredient-test-across-seeds) (the 4×6 at temporal coherence rate 0.08, downgrading and late-game starts), trained from scratch to 300k games on 8 threads with seeds 1–3, and evaluated on 1,000 seeded greedy games. The TD(0) control is that test's A + C arm, which has exactly these settings, so it wasn't retrained.
- **The implementation** keeps the last h afterstates in a ring buffer and updates the oldest once the buffer is full. Whatever is left in the buffer is updated at game end. With horizon 1 it gives bit-identical weights to TD(0), which a test checks.

| Arm | Seed 1 | Seed 2 | Seed 3 | Mean of seeds ± sd | vs TD(0) (paired by seed) | ≥16384 | Moves |
|---|---:|---:|---:|---:|---:|---:|---:|
| **TD(0)** | **186,413** | **186,809** | **187,498** | **186,906 ± 549** | — | **31.2%** | 1.59B |
| Jaśkowski: λ = 0.5, horizon 3 | 154,002 | 155,942 | 157,229 | 155,724 ± 1,624 | −31,182 ± 1,105 | 12.4% | 1.47B |
| Longer horizon: λ = 0.75, horizon 6 | 127,291 | 126,810 | 124,120 | 126,074 ± 1,709 | −60,832 ± 2,247 | 0.9% | 1.17B |
| Annealed: λ = 0.75 → 0, horizon 6 | 153,274 | 158,236 | 153,674 | 155,061 ± 2,757 | −31,845 ± 2,855 | 11.6% | 1.33B |

The ≥16384 and moves columns are averages over the three seeds.

![TD(λ) test, seed 2](figures/td-lambda-seed2.png)

The training average score at points during training is shown below: the mean of the three seeds over the 10,000 games up to each point. Training games include the late-game starts, so these aren't comparable to the evaluations, only to each other.

| Games | TD(0) | Jaśkowski | Longer horizon | Annealed (λ at that point) |
|---:|---:|---:|---:|---:|
| 10k | 36,509 | **40,566** | 32,024 | 32,836 (0.73) |
| 30k | 78,973 | **84,464** | 60,792 | 63,041 (0.68) |
| 50k | 104,078 | **107,025** | 66,747 | 70,119 (0.63) |
| 100k | **131,725** | 127,686 | 94,325 | 102,703 (0.50) |
| 200k | **162,232** | 146,829 | 122,094 | 140,895 (0.25) |
| 300k | **187,380** | 156,207 | 126,411 | 154,851 (0) |

- **TD(λ) hurts at this learning rate and budget, in every variant and on every seed.** The Jaśkowski arm costs 17%, and its three seeds land within 2.2k of each other (−30.3k to −32.4k). The gap isn't explained by the fewer moves played: shorter games are a symptom of weaker play, and TD(0) was already ahead at that point.
- **The Jaśkowski arm learns faster at first, then falls behind.** On all three seeds it's 3.7–7.6k ahead at 10k and 30k games, about level at 70k, and behind from 100k onwards, with the gap still widening at 300k. The early lead is what TD(λ) is meant to give: while the values are poor, real rewards are a better target than the value estimate. Later the estimates are good, and each extra step of real play mostly adds the randomness of another tile spawn. Two likely ways that noise hurts:
  - At base rate 0.08 the noise goes straight into the weights.
  - It makes errors cancel more, so temporal coherence shrinks the rates sooner.
- **A longer horizon is worse at every stage.** Both horizon-6 arms stall at 60–70k between about 20k and 70k games, before 8192 is reached, while TD(0) climbs 25k over the same stretch. The longer-horizon arm almost never reaches 16384 (0.9%).
- **Annealing recovers half the gap, but stops there.** It ends level with the Jaśkowski arm. Even over the last 100k games, with λ down from 0.25 to 0, it gains 14.0k against TD(0)'s 25.1k, so the damage outlasts λ itself. A likely cause: temporal coherence's error sums accumulate over the whole run, so a long noisy phase leaves the rates low after the noise has gone.
- **How this compares with Jaśkowski.** There's no published result to contradict: he chose λ = 0.5 from "informal preliminary experiments" and reports no λ = 0 comparison. The likeliest reason λ fails here is the learning rate. The training budget is less likely, and the horizon doesn't explain it.
  - **Learning rate.** His rate is divided by the number of weights read, and he calls 1.0 "the maximum sensible value", the rate that cancels the whole error in one step. On the 4×6 that's 1/32 ≈ 0.031 per weight. This project's 0.08 can move a value by up to 2.56× its error before temporal coherence slows it, so a noisier target gets amplified.
  - **Training budget.** He trained for 10¹⁰ moves, about 6× this test. But the gap here widened with training, from −4k at 100k games to −31k at 300k, so a longer run would more likely widen it further.
  - **Horizon.** His exact setting, one step longer than this arm, would probably do no better, since the longer horizon did worse here.
- **For the [final training run](#final-training-run), TD(λ) is dropped.** Its only gain is a head start in the first ~70k games, which is minutes of a multi-day run. TD(λ) at lower base rates (0.02–0.04) would test the rate explanation, but it's a low priority.

## The long run

The screen's best layout and rate (8×6, temporal coherence at 0.02) with the ingredient test's best pair, downgrading and late-game starts (A + C). Trained from scratch to 1.5M games on 8 threads with seed 42, in 2 h 02 min on the M4. 1,000 seeded games per evaluation, 95% confidence intervals:

| Games | Moves | Greedy | Median | ≥8192 | ≥16384 | ≥32768 |
|---:|---:|---:|---:|---:|---:|---:|
| 100k | 0.44B | 167,785 ± 5,097 | 167,712 | 76.2% | 20.6% | 0.2% |
| 250k | 1.59B | 244,076 ± 7,054 | 256,096 | 87.8% | 55.3% | 1.9% |
| 500k | 3.87B | 267,705 ± 7,346 | 286,228 | 92.1% | 61.5% | 4.2% |
| 1M | 8.76B | 285,232 ± 7,405 | 314,318 | 92.3% | 68.0% | 4.4% |
| 1.5M | 13.86B | **299,894 ± 7,668** | 339,306 | 92.1% | 73.2% | 6.4% |

With search on the 1.5M network. The gain is paired per game against greedy on the same seeds:

| Agent | Score | Median | vs greedy | ≥16384 | ≥32768 | Time for 1,000 games |
|---|---:|---:|---:|---:|---:|---:|
| Greedy | 299,894 ± 7,668 | 339,306 | — | 73.2% | 6.4% | 5 min |
| Depth 2 | 343,160 ± 6,327 | 373,222 | +43,267 ± 9,629 | 85.8% | 7.2% | 7 min |
| Depth 3 | **365,404 ± 5,769** | 377,960 | +65,510 ± 9,458 | 90.7% | 8.5% | 31 min |

![The long run](long-8x6/training_progress.png)

Against the published agents in [RELATED_WORK.md](RELATED_WORK.md):

| Agent | Greedy | 3-ply | Training |
|---|---:|---:|---|
| Guei et al. 2022 (state of the art) | 412,785 | 563,316 | 200M episodes, 66 h on 20 threads |
| Jaśkowski 2018, full method | 324,710 | 511,759 | 4×10¹⁰ moves, 5.5 days on 24 cores |
| **This run** | **299,894** | **365,404** | 1.4×10¹⁰ moves, 2 h on a 10-core laptop |
| Yeh et al. 2016, plain TD | 159,565 | 321,496 | 15M games |
| Jaśkowski 2018, TC only | 248,770 | 333,010 | 10¹⁰ moves, 1.16 days on 24 cores |

- **This is the project's best agent by far: 2.3× the previous best greedy (128k) and 1.8× at depth 3 (202k).**
  - Greedy, it beats every published n-tuple result except Jaśkowski's full method (92% of it) and Guei's (73%).
  - At 3-ply it beats Yeh and Jaśkowski's TC-only network, and reaches 65% of the state of the art.
- **Sample efficiency is where it leads.**
  - At 1M games (8.8×10⁹ moves) it scores 285k greedy, against 249k for Jaśkowski's TC-only network after 10¹⁰ moves.
  - Guei reports that n-tuple networks "could not even achieve 240000 points" after 10¹⁰ moves, and his 4×6 needed 10M episodes to reach 248k.
  - Guei's 8×6 reaches 360–371k greedy after 100M games, so this run reaches 81–83% of that with 1.5% of the games.
- **Still improving at 1.5M, with no sign of temporal coherence stalling.**
  - The paired gains are +17,527 ± 10,259 from 500k to 1M games and +14,662 ± 10,713 from 1M to 1.5M, about +20–25k per doubling of games.
  - The training averages flattened over the last 100k games (+0.7k), but the evaluations don't confirm a plateau, so that was noise.
- **Search helps less than on weaker networks: +22% at depth 3, against +58% for the [earlier 1.5M-game network](#expectimax-search-on-the-n-tuple-network).** Almost every searched game reaches 16384, and depth 3 adds only 4.7k to depth 2's median.
- **The remaining gap is 32768.**
  - Building a 16384 tile with its 8192 and 4096 chain is worth about 380k, which is where the median sits at depths 2 and 3. Making a 32768 tile alone is worth about 460k.
  - Even at depth 3, only 8.5% of games make a 32768 tile. The state of the art's 3-ply average of 563k needs it in most games.
  - So the gap to the state of the art is about late-game training, more than search.

## Faster search

Two changes make search on the 8×6 cheap enough to use routinely, and a third makes deeper search affordable:

- **Whole games in numba.** Evaluations used to play on the Python engine, calling the agent once per move, and that overhead cost about 0.2 ms per move, which is more than the greedy search itself. The n-tuple agent now plays each seeded game entirely in numba, drawing tile spawns from the same `np.random.default_rng(seed)` stream in the same order. So it plays exactly the same games as before: every stored evaluation of the long run replays game for game. The games run on threads that share one copy of the network, where each worker process used to load its own 660 MB.
- **A transposition table.** Each search caches the values of the afterstates it has expanded, keyed by board and remaining depth, since different move orders often reach the same board. Exact search gives the same values as before.
- **A probability cutoff** (`--cutoff`). A spawn branch less likely than the cutoff, counted from the root, stops at the network's value instead of being expanded. The search then goes deep only on crowded boards, where each spawn is likely, so the depth adapts to the position, much like nneonneo's expectimax.
  - It's in the agent's label, so cached results for exact and pruned search can't mix.

**Speed.** Per move on one thread, over 125 positions from greedy games of the 1.5M long-run network. Times for 1,000 games are on 10 threads of the M4:

| Search | Per move | 1,000 games, before | 1,000 games, now |
|---|---:|---:|---:|
| Greedy | 0.003 ms | 5 min | 4.5 s |
| Depth 2 | 0.034 ms | 7 min | 53 s |
| Depth 3 | 0.53 → 0.35 ms | 31 min | 13 min |
| Depth 4 | about 4 ms | | |
| Depth 3, cutoff 0.05 | 0.21 ms | | 8 min |
| Depth 4, cutoff 0.1 | 0.27 ms | | 10 min |
| Depth 5, cutoff 0.1 | 0.31 ms | | 12 min |
| Depth 6, cutoff 0.1 | 0.38 ms | | 15 min |

**Strength.** The 1.5M long-run network, 1,000 seeded games each ([`search-long-8x6`](search-long-8x6/NOTES.md)), compared game for game with exact depth 3:

| Search | Score | vs exact depth 3 | ≥16384 | ≥32768 |
|---|---:|---:|---:|---:|
| Depth 3 | 365,404 ± 5,769 | — | 90.7% | 8.5% |
| Depth 3, cutoff 0.05 | 364,295 ± 5,884 | −1,108 ± 8,201 | 89.8% | 8.3% |
| Depth 4, cutoff 0.1 | 369,289 ± 5,827 | +3,885 ± 8,457 | 91.2% | 8.9% |
| Depth 4, cutoff 0.05 | 371,047 ± 5,582 | +5,643 ± 7,978 | 92.3% | 8.5% |
| Depth 5, cutoff 0.1 | 371,048 ± 5,789 | +5,644 ± 8,107 | 92.0% | 9.2% |
| **Depth 6, cutoff 0.1** | **376,221 ± 6,143** | **+10,817 ± 8,534** | 92.1% | **11.7%** |

- **Exact depth 3 is 2.3× faster, and greedy evaluation is about 70× faster.** Evaluating snapshots during experiments is now cheap.
- **Pruned deeper search gains a little, and only depth 6 is clear of noise.** Depth 6 at cutoff 0.1 adds 3% and raises the 32768 rate from 8.5% to 11.7%. It costs about what exact depth 3 does now, and half what it used to.
- **That's much less than the published agents gain from search.** Guei goes from 563k at 3-ply to 625k at 6-ply (+11%), and Jaśkowski from 512k to 609k at 1 s per move (+19%). Their searches spend hundreds to thousands of times more per move, but the main limit is the network. All these settings reach 16384 in 90–92% of games and 32768 in under 12%, and search can't make up for poorly learned 32768-stage values.
- **Pruned results aren't N-ply results.** At cutoff 0.1, most branches stop well short of the nominal depth. So "depth 6" here isn't comparable with Guei's 6-ply, and pruned results should be reported with their time per move. For comparison with papers, use exact depths: 3-ply takes 13 minutes per 1,000 games, and 4-ply should take 2–3 hours.
- **The bitboard caps tiles at 32768.** Two 32768s don't merge there, though they do on the Python engine. Whole-game play stops with an error if a board ever holds two, so a result can't silently differ from the Python engine. None of these games did.

## Decision arms

The [long run](#the-long-run) was still improving at 1.5M games, but the [final training run](#final-training-run) will last tens of millions. These arms test the decisions that only matter at that scale, by branching from the long run's 1.5M-game network rather than training from scratch.

- **The setup:**
  - Each arm resumes the 1.5M-game network and trains a further 500k games with the long run's settings (8×6, temporal coherence at 0.02, downgrading and late-game starts, 8 threads), changing one thing.
  - Each is evaluated on the usual 1,000 seeded games, greedily at 1.75M and 2M games and at exact depth 3 at 2M, and compared game for game with the control.
  - `scripts/run_decision_arms.sh` runs them, about 4.5 hours on the M4.
- **The arms:**
  - **Control:** unchanged.
  - **TC reset:** the temporal coherence sums start from zero (`--reset-coherence`), so every weight learns at the full rate again until its errors start cancelling. That's the closest a branch can get to the temporal coherence phase that 2048-rl ran after its constant-rate stage, which took its training average from about 276k to about 370k.
  - **Constant rate:** temporal coherence off, at 0.0016 per weight. That's 2048-rl's and Guei's constant rate of 0.1 per board, spread over 64 weights.
  - **Late-game share:** half of all games start from late-game positions instead of a fifth.
    - The plan was to start games from doubled first-16384 positions instead, to give the 32768 stage experience.
    - That's impossible here. The doubled boards hold a 32768, so a game could need two 32768s merged, which the bitboard can't do, and `upgrade()` leaves such boards alone.
    - The smoke test caught it: the arm made no late-game starts at all. Training now refuses that setting.

| Arm | Greedy at 2M | vs control | Exact depth 3 at 2M | vs control | ≥16384 at depth 3 | ≥32768 at depth 3 |
|---|---:|---:|---:|---:|---:|---:|
| Control | 302,227 ± 7,318 | — | 368,795 | — | 90.9% | 9.6% |
| **TC reset** | **320,042** | **+17,814 ± 10,300** | **402,208 ± 6,763** | **+33,414 ± 9,224** | **94.6%** | **20.1%** |
| Constant rate | 236,003 | −66,225 ± 10,383 | 356,872 | −11,923 ± 8,876 | 88.2% | 10.1% |
| Late-game share | 297,475 | −4,752 ± 10,569 | 356,813 | −11,981 ± 8,304 | 88.7% | 7.0% |

At 1.75M games the greedy differences were +10,704 ± 10,813 (reset), −67,100 ± 10,193 (constant) and −4,561 ± 10,557 (late share). Against the 1.5M network it started from, the control gained +2,334 ± 10,462 greedy and +3,391 ± 8,401 at depth 3.

![Decision arms](figures/decision-arms.png)

- **Temporal coherence has stalled by 1.5M games, and resetting it breaks the stall.**
  - The control gained nothing measurable in 500k games, against +14.7k from 1M to 1.5M.
  - The reset arm dropped about 25k for the first 50k games, as every weight took full-size steps again, was back level by 150k, and then climbed past the control.
  - It's the best agent yet: 320k greedy, and 402k at 3-ply. That's the first 3-ply result over 400k here, level with Matsuzaki's 407k (a CNN after 240 GPU-hours).
  - It doubled the 32768 rate at depth 3, to 20.1%, which is the weakness the [faster search](#faster-search) results pointed to.
  - Its training average was levelling off by 2M games, about +1k per 50k, so the gain may need repeating. Round 2, below, tests that.
- **Switching a trained network to a constant rate is harmful.**
  - The training average fell to 194k within 50k games and recovered only to about 233k, while the control held 300k.
  - Temporal coherence had shrunk most weights' steps far below 0.0016, and a constant rate never shrinks back.
  - Search hides most of the damage: −12k at depth 3 against −66k greedy, with the same 32768 rate. That suggests the early and middle game suffered most.
  - Guei's schedule uses a constant rate from the start, with optimistic initialisation. A branch can't test that.
- **More late-game experience doesn't help while the rates are settled.** The late-share arm stayed about 10k behind the control in training, ended within noise greedy and 12k behind at depth 3, with a lower 32768 rate. The bottleneck was the learning rates, not a shortage of late-game games.
- **Two quick checks alongside:**
  - **Snapshot averaging.** Averaging the long run's 1M and 1.5M weights gave −1.5k ± 10.6k greedy and +6.5k ± 8.3k at depth 3. That's within noise, and close to 2048-rl's +7k at depth 2.
  - **Temporal coherence's cost.** On the 1.5M network, training runs at 1.8–1.9M moves/s with temporal coherence and 3.4M without, a 1.8× cost, as in 2048-rl. Storing each weight next to its two sums, as 2048-rl does, would cut the memory accesses per update from three to one.

### Round 2: do resets keep paying?

Branching from the TC-reset arm at 2M games, for another 500k each (`ROUND=2 scripts/run_decision_arms.sh`):
- **Continue:** no reset, so the first reset's cycle carries on.
- **Reset again:** a second full reset.
- **Gentle reset:** the sums scaled by 0.1 (`--coherence-scale 0.1`).
  - A weight's rate is the ratio of its two sums, so scaling both keeps every rate as it was. New errors then count ten times as much, so rates rise again wherever errors start agreeing.
  - It's meant to keep a reset's gain without the dip. Halving the sums would barely change anything after 500k games of history.

| Arm | Greedy at 2.5M | vs continue | Exact depth 3 at 2.5M | vs continue | ≥32768 at depth 3 |
|---|---:|---:|---:|---:|---:|
| Continue | 314,202 | — | 397,229 | — | 18.1% |
| Reset again | 317,291 | +3,089 ± 10,879 | 384,451 | **−12,779 ± 9,286** | 14.5% |
| Gentle reset | 315,449 | +1,247 ± 10,819 | 399,341 | +2,112 ± 9,401 | 19.6% |

At 2.25M games the greedy differences were −15,974 ± 10,649 (reset again) and −7,241 ± 10,939 (gentle). Against the 2M network all three started from, continue scored −5,840 ± 11,022 greedy and −4,979 ± 9,348 at depth 3.

![Decision arms, round 2](figures/decision-arms-round2.png)

- **The first reset's gain is spent by 2M games.** Continuing for 500k games changed nothing measurable, and its training average stayed at about 315k throughout.
- **A second full reset doesn't repeat the first.**
  - It dipped further (33k against 25k) and recovered more slowly, about 350k games against 150k.
  - It ended level greedy and significantly worse at depth 3, with a lower 32768 rate.
  - A likely reason: a reset helps the weights temporal coherence had frozen, and the late-game weights are both the ones it helps most and the slowest to re-settle, since they're updated rarely. The first reset freed late-game weights frozen since early training. The second hit weights that had only just been re-learned, so it mostly added noise.
- **A gentle reset avoids the dip but gains nothing.** It stayed within about 5k of continue in training and ended level on both measures.
- **So resetting learning rates is a one-off fix, not a schedule.** It released one stall, but at about 315k it no longer restarts progress, while Guei's and 2048-rl's single 8×6 networks reach about 360–370k greedy. The next limit isn't the rates.

### Round 3: optimism for never-updated weights

Guei's and 2048-rl's single 8×6 networks reach about 360–370k greedy, against about 315k here, and both use optimistic initialisation: every weight starts high, so greedy play is drawn to positions it hasn't learned yet. 65% of this network's weights had never been updated by 2M games, which looked like a lack of exploration. Round 3 sets every never-updated weight (still exactly 0) to an optimistic value, from round 1's TC reset at 2M games, for another 500k games (`ROUND=3 scripts/run_decision_arms.sh`, `--optimistic-unvisited VALUE`):
- **1,000 per weight.**
- **5,000 per weight:** 2048-rl's 320,000 per board, spread over 64 weights.

The control is round 2's continue, which started from the same network and changed nothing.

| Arm | Greedy at 2.5M | vs continue | Exact depth 3 at 2.5M | vs continue | ≥32768 at depth 3 |
|---|---:|---:|---:|---:|---:|
| Continue | 314,202 | — | 397,229 | — | 18.1% |
| Optimism at 1,000 | 310,682 | −3,519 ± 10,738 | 396,528 | −701 ± 9,713 | 18.6% |
| Optimism at 5,000 | 317,967 | +3,766 ± 10,824 | 389,712 | −7,518 ± 9,360 | 16.6% |

At 2.25M games the greedy differences were −4,339 ± 10,900 (1,000) and −9,940 ± 10,943 (5,000).

![Decision arms, round 3](figures/decision-arms-round3.png)

- **Optimism changed nothing.** Both arms tracked continue within a few thousand points in training and ended within noise on both measures.
- **Most never-updated weights can't be reached at all.** Broken down by the largest tile in each weight's pattern:

  | Patterns that include… | Share of all weights | Never updated |
  |---|---:|---:|
  | a 32768 | 32% | 100% |
  | a 16384 or 32768 | 55% | 86% |
  | only tiles up to 8192 | 45% | 40% |

  - With downgrading from 16384, the tuples read every board holding a 16384 with its large tiles halved. So they almost never see a 16384 and never a 32768.
  - About 73% of the never-updated weights are patterns that can't occur, and the rest are mostly rare combinations.
  - So the 65% measured downgrading's footprint, not a lack of exploration, and optimism had little real territory to work on.
- **What this points to instead:** downgrading from 16384 leaves more than half of the 134M weights permanently unused, while one set of shared weights must fit both the 8192 and 16384 stages, whose true values differ.
  - Sharing paid off early, when 16384 was rare (it was most of A + C's +13%).
  - With 75% of games now reaching 16384, it may be the ceiling. The [32768 downgrading test](#next-experiments) checks this by giving the 16384 stage its own weights.

### Downgrading from 32768 instead of 16384

Round 3 found that downgrading from 16384 leaves more than half the network unused, and that one set of weights serves both the 8192 and 16384 stages, whose true values differ. This run tests whether giving the 16384 stage its own weights lifts the ceiling: the long run's setup from scratch, downgrading only boards that hold a 32768 ([`long-8x6-dg32768`](long-8x6-dg32768/NOTES.md), `DOWNGRADE_TILE=32768 EVAL_EVERY=250000 scripts/run_long_run.sh`).

| Games | Greedy | vs the long run (paired) | ≥16384 (long run) | ≥32768 (long run) |
|---:|---:|---:|---:|---:|
| 100k | 153,246 | −14,538 ± 6,587 | 12.9% (20.6%) | 0.0% (0.2%) |
| 250k | 212,947 | −31,129 ± 9,274 | 44.4% (55.3%) | 0.0% (1.9%) |
| 500k | 250,784 | −16,920 ± 9,662 | 62.1% (61.5%) | 0.0% (4.2%) |
| 750k | 255,880 | | 65.8% | 0.0% |
| 1M | 274,755 | −10,476 ± 9,250 | **73.7% (68.0%)** | 0.0% (4.4%) |
| 1.25M | 270,381 | | 71.2% | 0.0% |
| 1.5M | 273,643 | −26,250 ± 9,475 | 70.3% (73.2%) | 0.0% (6.4%) |
| 1.5M, depth 2 | 322,962 | −20,198 ± 7,995 | 87.0% (85.8%) | 0.2% (7.2%) |
| 1.5M, depth 3 | 346,617 | −18,786 ± 6,752 | 93.1% (90.7%) | 0.7% (8.5%) |

![Downgrading from 32768 against the long run](figures/downgrade-32768.png)

- **Its own weights make the 16384 stage better.** After trailing early without the long run's head start, it reached 16384 in more games from 500k on, 5.7 points more at 1M.
- **But it never learns to make a 32768.** Not one greedy game in any evaluation, and none in training, where the long run already reached 32768 in 4% of games by 500k. It plateaus at about 270–275k from 1M games.
- **Why:** making a 32768 means building a second 16384 next to the first.
  - Downgrading from 16384 reads that board like building a second 8192 next to a 16384, which the network has practised millions of times.
  - With the 16384 stage separate, the step starts from nothing. In a crowded late game, nothing doesn't get far enough to learn.
- **Moving the threshold on a trained network doesn't work either.**
  - `scripts/split_downgrade.py` moves the long run's 1.5M network to downgrading from 32768, and gives each untrained 16384 pattern the weight of the same pattern halved above its own gap.
  - Played greedily, it scored 258,967, −40,927 ± 4,581 against the original. It reached 16384 just as often (73.2%) but 32768 never.
  - Downgrading depends on the whole board: when any cell holds a 16384, every pattern reads its big tiles halved, including those that don't contain the 16384. No per-pattern copy can reproduce that.
  - It wasn't trained further (round 4 in the arms script is left unused).
- **So the threshold is the wrong lever.** Inside one network, it's the only signal that a board is in the 16384 stage, so it can't separate the stages without losing what sharing provides.
  - A second stage chosen by the whole board, starting as an exact copy and still reading boards downgraded from 16384, can separate them without that loss.
  - [Round 5](#next-experiments) tests that.

### Round 5: a second stage

A second stage gives boards with a 16384 or more their own copy of the weights, so the late game can learn values of its own instead of sharing the 8192 stage's. Unlike moving the downgrading threshold, it keeps reading boards downgraded from 16384, so the borrowing that makes 32768s possible survives.
- **The start:** `scripts/add_stage.py` copies the long run's 1.5M network into both stages, and the copy plays all 1,000 evaluation games identically to the original.
- **The test:** trained with reset sums to 2M games (`ROUND=5 scripts/run_decision_arms.sh`), against round 1's TC reset, which made the same reset from the same network with one stage ([`decision5-8x6-two-stage`](decision5-8x6-two-stage/NOTES.md)).

| Arm | Greedy at 1.75M | Greedy at 2M | vs TC reset at 2M | Exact depth 3 at 2M | vs TC reset | ≥16384 at depth 3 | ≥32768 at depth 3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| TC reset (one stage) | 311,045 | 320,042 | — | 402,208 | — | 94.6% | 20.1% |
| Two stages | 247,666 | 257,579 | −62,463 ± 10,573 | 327,421 | −74,787 ± 9,116 | 75.5% | 5.1% |

![Round 5 against round 1's TC reset](figures/decision-arms-round5.png)

- **It got much worse, and the reason is visible in the 16384 rate.**
  - Greedy, it fell from 73% at the split to 52% by 2M games, while the reset arm held 75–77%.
  - Stage 2 only plays once a 16384 exists, so a falling 16384 rate means stage 1 had become *less willing to make one*.
- **The cause is a flaw in how the stage was chosen.**
  - Each afterstate was read in the stage of its own largest tile. On a board whose largest tile is 8192, the move that merges two 8192s makes a 16384, so it was valued by stage 2 while the board's other moves were valued by stage 1.
  - The stages start identical, but once they learn apart, especially at full rate after the reset, the move into the next stage is judged on a different scale from its alternatives.
  - 2048-rl's player switches networks only once the board holds a 16384, which avoids this.
- **The fix:** choose the stage by the board a move is made from, so every move of a decision is valued, and learned, in one stage (`feat/stage-by-state`, round 8 in the arms script).

### Round 6: temporal coherence with forgetting

Temporal coherence sums each weight's errors over its whole history, so once they've cancelled out over millions of updates its rate stays near zero for good, even if it later has something new to learn. A reset undoes that once (round 1) but doesn't repeat (round 2). Forgetting is the continuous version: each update first shrinks the weight's two sums by a factor just below 1 (`NTUPLE_TC_DECAY`), so its rate follows its recent errors instead of all of them.
- **The arms:** decays of 0.999 (memory of about 1,000 updates) and 0.9999 (about 10,000), from round 1's TC reset at 2M games, for another 500k games (`ROUND=6 scripts/run_decision_arms.sh`).
- **The control:** round 2's continue, the same start without forgetting.

| Arm | Greedy at 2.5M | vs continue | Exact depth 3 at 2.5M | vs continue | ≥16384 at depth 3 | ≥32768 at depth 3 |
|---|---:|---:|---:|---:|---:|---:|
| Continue | 314,202 | — | 397,229 | — | 93.5% | 18.1% |
| Forgetting at 0.999 | 238,499 | −75,702 ± 10,294 | 356,838 | −40,392 ± 9,456 | 87.2% | 11.6% |
| Forgetting at 0.9999 | 274,051 | −40,151 ± 10,704 | 380,698 | −16,531 ± 9,583 | 92.1% | 14.0% |

At 2.25M games the greedy differences were −80,571 ± 10,417 (0.999) and −50,058 ± 11,055 (0.9999).

![Round 6 against continue](figures/decision-arms-round6.png)

- **Forgetting hurts, and faster forgetting hurts more.** Both arms dropped within the first 50k games and then stayed flat, the 0.999 arm about 75k below continue in training and the 0.9999 arm about 35k.
- **Why:** after millions of updates, a well-trained weight's errors are mostly noise, so its rate had shrunk to nearly nothing. Measured over its last N errors instead, noise looks like a rate of about 1/√N: about 3% of the base rate at 0.999 and 1% at 0.9999, far more than it had. Every settled weight starts taking noisy steps, like the constant-rate arm in round 1, and unlike a reset the rates never settle back.
- **So the rate a trained network needs is lower than any short memory gives.** A slower decay (0.99999 or beyond) would come closer to plain temporal coherence, but there's no sign here of a gain to find on the way. Forgetting is dropped.

### Rounds 8 and 9: a second stage, chosen by the board a move is made from

Round 5's second stage valued each afterstate in the stage of its own largest tile, so the move that made the first 16384 was judged by stage 2 against stage 1's values for the board's other moves, and the agent learned to avoid making it. Here every move of a decision is valued, and learned, in the stage of the board it's made from, as 2048-rl's player switches networks only once the board holds a 16384.
- **Round 8:** the same start and reset as round 5, the long run's 1.5M network given a second stage, trained to 2M games against round 1's TC reset (`ROUND=8 scripts/run_decision_arms.sh`).
- **Round 9:** round 8 carried on to 2.5M, against round 2's continue, the reset arm carried on over the same games (`ROUND=9`).

| Arm | Games | Greedy | vs one stage | Exact depth 3 | vs one stage | ≥16384 at depth 3 | ≥32768 at depth 3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| One stage (TC reset) | 2M | 320,042 | — | 402,208 | — | 94.6% | 20.1% |
| Round 5's two stages | 2M | 257,579 | −62,463 ± 10,573 | 327,421 | −74,787 ± 9,116 | 75.5% | 5.1% |
| [Round 8](decision8-8x6-two-stage-by-state/NOTES.md)'s two stages | 2M | 287,985 | −32,057 ± 10,423 | 367,599 | −34,609 ± 8,644 | 93.1% | 8.2% |
| One stage (continue) | 2.5M | 314,202 | — | 397,229 | — | 93.5% | 18.1% |
| [Round 9](decision9-8x6-two-stage-by-state/NOTES.md)'s two stages | 2.5M | 302,656 | −11,546 ± 10,755 | 374,308 | −22,921 ± 8,699 | 93.0% | 9.7% |

At 1.75M games round 8 was −59,191 ± 9,927 greedy against the reset arm, and at 2.25M round 9 was −27,483 ± 10,902 against continue.

![Round 8 against the TC reset](figures/decision-arms-round8.png)
![Round 9 against continue](figures/decision-arms-round9.png)

- **The fix works where it should.**
  - The 16384 rate no longer falls: it recovered from 58% at 1.55M to 72% by 2.5M in training, and to 93% at depth 3, level with one stage.
  - Against round 5, the fix gained 30k greedy and 40k at depth 3.
- **Two stages are still behind one after 1M games.**
  - Greedy, the gap closed from −59k at 1.75M to −11.5k at 2.5M, now near the edge of noise. The training average levelled off at about 300k by 2.4M, while one stage held about 315k.
  - At depth 3 it's still 23k behind, and it makes a 32768 in half as many games (9.7% against 18.1%).
- **The likely cost is the reset, doubled.**
  - Both stages restarted at full rate, and each weight now gets only the updates for its own stage, so the noisy phase lasts longer. The late-game stage, which is updated least, is noisiest of all.
  - That fits a slow recovery that hits the 32768 step hardest.
- **So a second stage hasn't paid for itself within 1M games of a full reset.**
  - The fair next test splits without resetting: the long run's 1.5M network given a second stage with its sums kept (`scripts/add_stage.py --keep-coherence`), against round 1's control.
  - 2048-rl's second stage needed tens of millions of games, so a split may also need far more training than a branch test gives.

### Round 10: a second stage without the reset

Rounds 5, 8 and 9 all reset the temporal coherence sums at the split, so the reset's shock could explain their deficit. Here the long run's 1.5M network gets a second stage with its sums kept (`scripts/add_stage.py --keep-coherence`), stage chosen by the board a move is made from, trained to 2M games against round 1's control: the same start, one stage, no reset (`ROUND=10 scripts/run_decision_arms.sh`, [`decision10-8x6-two-stage-kept-sums`](decision10-8x6-two-stage-kept-sums/NOTES.md)).

| Arm | Greedy at 1.75M | Greedy at 2M | vs control at 2M | Exact depth 3 at 2M | vs control | ≥16384 at depth 3 | ≥32768 at depth 3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Control (one stage) | 300,341 | 302,227 | — | 368,795 | — | 90.9% | 9.6% |
| Two stages, sums kept | 282,349 | 286,492 | −15,735 ± 10,098 | 350,274 | −18,521 ± 8,044 | 91.8% | 6.8% |

![Round 10 against the control](figures/decision-arms-round10.png)

- **A second stage loses even without the reset.**
  - It started level (the copy plays identically).
  - It drifted down for the first 200k games, its training average falling from about 291k to 279k while the control held about 301k, then stayed about 20k behind.
  - It ends significantly worse both greedy and at depth 3, with a lower 32768 rate.
- **The likely reason is specific to downgrading during training: sharing works in both directions.**
  - With one table, every update from a 16384-stage board also trains the weights the 8192 stage reads, since the downgraded board looks like an 8192 board. In long games that's a large share of all moves.
  - Splitting cuts that link, so stage 1 learns only from boards below 16384, and its play towards the first 16384 slowly loses what the late game had been teaching it.
  - Guei's, Yeh's and 2048-rl's stages don't share across scales in training, so splitting costs them nothing.
- **So with downgrading in training, a second stage needs far more training than these tests give, or a design that keeps feeding late-game experience back into stage 1.** It's left out of the final run for now.

## Final training run

The end goal is a single multi-day training run that produces the final model. The short tests above exist to choose its setup.

- **The machine:**
  - AMD Ryzen 7 9850X3D (8 cores, 16 threads), 32 GB of DDR5-6000, Windows.
  - Expect roughly the speed of the Macs used here. The 8×6 did 1.5M games in about 2 hours on 8 threads, so 3 days is in the region of 50M games: tens of millions, not Guei's 200M.
  - It may well be faster. Training is limited by memory latency, and the M4's 8 threads include 4 efficiency cores, where the Ryzen has 8 full cores, SMT and a 96 MB L3 cache. Decision 5 below measures it.
  - Memory isn't a constraint. The 8×6 with temporal coherence sums takes a few GB, and a 2-stage version would still fit.
- **What the short tests settle, and what they don't:**
  - **These carry over:**
    - training throughput and correctness (threading, determinism);
    - the evaluation method, including that single seeds vary by about ±2%;
    - choices that held up at 1.5M games: the 8×6 network, an aggressive temporal coherence rate early on, and downgrading with late-game starts. The 1.5M-game run built on them reached about 300k greedy.
  - **These don't:**
    - learning-rate tuning, since 0.08 won at 200k games but a long run will need lower rates or a schedule later;
    - anything whose benefit is early speed, which is why TD(λ) is dropped;
    - how much late-game starts help once natural games reach 16384 far more often.
- **Where the final model's gap is.** The [long run](#the-long-run) and [faster search](#faster-search) show it's the late game: 16384 is reached in 90% of depth-3 games, but 32768 in under 12%, and the state of the art needs it in most. Deeper search added only 3%, so the decisions below target late-game training.
- **Decisions that only matter at scale**, to be tested by branching from a mature checkpoint (the 1.5M-game 8×6) for about 500k games per arm, not by training from scratch:
  1. **The learning-rate schedule. Settled by the [decision arms](#decision-arms): temporal coherence with its sums reset.**
     - Temporal coherence stalls by 1.5M games, and one reset gave +18k greedy and +33k at depth 3, and doubled the 32768 rate.
     - Switching to a constant rate cost 22% greedy. Guei's schedule, with a constant rate from the start and optimism, would need its own run from scratch.
     - [Round 2](#round-2-do-resets-keep-paying) found the reset is a one-off: a second full reset 500k games later hurt at depth 3, and a gentle reset was neutral. So the final run resets once, at the first stall.
     - The next plateau, at about 315k, needs something other than learning rates. [Round 3](#round-3-optimism-for-never-updated-weights) found that optimism for the never-updated weights doesn't help either: most of them are patterns downgrading makes unreachable.
  2. **More late-game experience. Answered: it doesn't help.** Starts from 32768 positions turned out to be impossible in the bitboard, and a larger share of the starts that are possible cost 12k at depth 3. Late-game starts stay at a fifth of games, from 8192.
  3. **Separate late-game weights. Tested three ways, none better yet.**
     - [Downgrading from 32768](#downgrading-from-32768-instead-of-16384) gave the 16384 stage its own weights inside one network. That stage played better, but the network never learned to make a 32768.
     - A second stage chosen by the afterstate ([round 5](#round-5-a-second-stage)) misjudged the move into it.
     - Chosen by the board a move is made from ([rounds 8 and 9](#rounds-8-and-9-a-second-stage-chosen-by-the-board-a-move-is-made-from)), it closed to −11.5k greedy at 2.5M games but stayed 23k behind one stage at depth 3.
     - Still open: a split without the reset (sums kept), and whether a second stage needs far more training than a branch test gives.
  4. **Optimistic initialisation,** which only paid off after 12M+ games for Guei. Adding it to a trained network did nothing (round 3), so it's worth trying only from scratch, and only once the downgrading threshold is settled.
  5. **Thread scaling on the target machine:** 8 against 16 threads, before committing days to it.
- **Training speed:** temporal coherence costs 1.8× in moves per second. Since it stays for the whole run, storing each weight next to its sums is worth building first: it cuts the memory accesses per update from three to one.
- **Rejected for the final run:** a constant rate after temporal coherence, a larger late-game share, optimism added to a trained network, downgrading from 32768, and [temporal coherence forgetting](#round-6-temporal-coherence-with-forgetting), which hurt at both decays tried.
- **Evaluating the final model:** 1,000 seeded games at exact 3-ply and 4-ply, for comparison with the papers, and with pruned search (depth 6 at cutoff 0.1, or deeper) as the strongest agent, reported with its time per move.
- **Running on Windows:**
  - Use WSL2, since the scripts are bash and use `pgrep`, and keep the repo on the Linux filesystem, not `/mnt/c`.
  - Raise WSL's memory cap in `.wslconfig`.
  - Disable sleep and pause Windows Update restarts for the run. Checkpoints every 10k games and resume support limit the cost of an interruption to minutes.

## Next experiments

Where things stand, as of these runs:
- **The [long run](#the-long-run)** reached 300k greedy and 365k at 3-ply after 1.5M games. That's more sample-efficient than any n-tuple result in [RELATED_WORK.md](RELATED_WORK.md).
- **[Faster search](#faster-search)** made evaluation cheap, but deeper search added only 3%.
- **[TD(λ)](#tdλ-across-seeds)** hurts at this rate and budget in every variant, and is dropped.
- **The [decision arms](#decision-arms)** found that temporal coherence stalls by 1.5M games, and that resetting its sums breaks the stall: 320k greedy, 402k at 3-ply, and 32768 in 20% of depth-3 games, after 2M games. A constant rate and more late-game starts both hurt.
- **[Round 2](#round-2-do-resets-keep-paying)** found the reset works once: neither a second full reset nor a gentle one restarted progress from about 315k.
- **[Round 3](#round-3-optimism-for-never-updated-weights)** found that optimism for never-updated weights doesn't help, and that most of those weights are unreachable because of downgrading from 16384.
- **[Downgrading from 32768](#downgrading-from-32768-instead-of-16384)** improved the 16384 stage, but without sharing the network never learned to make a 32768.
- **[Round 6](#round-6-temporal-coherence-with-forgetting)** found that temporal coherence forgetting hurts.
- **[Rounds 5, 8 and 9](#rounds-8-and-9-a-second-stage-chosen-by-the-board-a-move-is-made-from)** found that a second stage needs choosing by the board a move is made from. Even then, after a full reset, it was still behind one stage at 2.5M games.

The best agent so far is still round 1's [TC reset](#decision-arms): 320k greedy and 402k at 3-ply after 2M games. The [final training run](#final-training-run) sets the decisions. In order:

1. **A lower learning rate** (round 11): the base rate cut from 0.02 to 0.005 on the plateaued network. The rate was chosen on a 200k-game screen for early speed, and is above what Jaśkowski calls the maximum sensible step per board.
2. **Faster temporal coherence:** store each weight next to its two sums, then measure the speed-up on the M4.
3. **Engine support for 65536,** before any run whose agent reaches 32768 often. The bitboard can't merge two 32768s, and evaluations stop with an error when two meet.
4. **On the Ryzen:** check 8 against 16 threads and the real speed.
5. **The final training run:** temporal coherence from scratch, downgrading from 16384, one reset at the first stall, and a second stage if item 1 shows it pays, evaluated as its section describes.

Left out for now:
- **TD(λ) at lower base rates** (0.02–0.04). It would test the rate explanation in [TD(λ) across seeds](#tdλ-across-seeds), but it can't help the final run.
- **Time-limited search.** The cutoff already gives most of what iterative deepening would. Add it only for a like-for-like comparison with Jaśkowski's 1 s per move.
