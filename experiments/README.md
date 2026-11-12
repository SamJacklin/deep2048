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
| [`long-8x6`](long-8x6/NOTES.md) | [The long run](#the-long-run): 8×6 at 0.02 with A + C, on 8 threads to 1.5M games, then search |
| [`search-long-8x6`](search-long-8x6/NOTES.md) | [Faster search](#faster-search): exact and pruned search at depths 3–6 on the long run's 1.5M network |

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

## Next experiments

Where things stand, as of these runs:
- **The [long run](#the-long-run)** reached 300k greedy and 365k at 3-ply after 1.5M games. That's more sample-efficient than any n-tuple result in [RELATED_WORK.md](RELATED_WORK.md), and it was still improving at about +20–25k per doubling of games.
- **[Faster search](#faster-search)** made evaluation cheap, but deeper search added only 3%.
- **The TD(λ) test** (on `feat/ntuple-td-lambda`) found that every variant hurts at this rate and budget. Jaśkowski's λ = 0.5 cost 17% on all three seeds.

So the gap to the state of the art is the network's late game: 32768 is reached in under 12% of games, and the state of the art needs it in most. The aim now is to settle what goes into a much longer run. In order:

1. **The learning-rate schedule for long training.** This is the decision the long run depends on most.
   - Guei found that temporal coherence settles too early on the 8×6 over 100M games, and his best recipe was a constant rate with temporal coherence only at the end.
   - The TD(λ) test adds a second warning: the annealed arm kept learning at half speed after λ had gone, consistent with temporal coherence's sums holding rates low long after a noisy phase.
   - **The test:** continue the 1.5M snapshot to 3M games three ways: temporal coherence as is (the control), temporal coherence with its sums reset every 500k games, and a constant rate of about 0.002 (Guei's 0.1 spread over the 8×6's 64 weights).
   - It's about 2.5 hours of training per arm on the M4. Evaluate every 250k games greedily and at depth 3, now that both are cheap.
2. **More late-game experience.** Raise the late-game start tile from 8192 to 16384.
   - Late-game starts now donate the position where a game first makes 8192 and double it, so synthetic starts hold a 16384. Normal games now reach 16384 in 73% of greedy games, so starting from 32768 positions is the natural next step.
   - It gives the 32768 stage the experience it's missing, which is exactly the gap above.
   - **The test:** continue the 1.5M snapshot with `NTUPLE_LATE_START_TILE=16384`, against the same continuation at 8192. It can share its control with item 1.
3. **Two-stage network.** A second weight set from the first 16384 tile, warm-started from the first.
   - It works against downgrading's sharing across scales, so it needs testing, not assuming.
   - It's also more costly than item 2 (twice the memory, about 3 GB with temporal coherence). So it's worth trying only if item 2 doesn't close much of the 32768 gap.
   - **The test:** continue the 1.5M snapshot with and without the split.
4. **The run at scale.** 10M+ games with whatever survives items 1–3.
   - At the long run's speed, that's about 1.2×10¹¹ moves, or 15–20 hours on the M4. A desktop with more fast cores, or a rented server, would cut that.
   - Optimistic initialisation, Guei's main ingredient, slows early learning and pays off only after about 10M episodes. So whether to include it is decided here, not in short tests.
   - Evaluate the result at exact 3-ply and 4-ply for comparison with the papers, and at depth 6 with cutoff 0.1 (or deeper) as the strongest agent, with its time per move.

Left out for now:
- **TD(λ)** follow-ups, such as λ early in training only, or at lower rates. Its only gain was in the first 70k games, which doesn't matter at 10M+ games.
- **Time-limited search.** The cutoff already gives most of what iterative deepening would. Add it only for a like-for-like comparison with Jaśkowski's 1 s per move.
