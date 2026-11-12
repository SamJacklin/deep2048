# Related work

Published 2048 agents, for placing this project's results. All use the standard game: 2 spawns 90% / 4 spawns 10%, score = sum of merged tile values. "N-ply" counts player moves, with 1-ply being greedy, which matches this project's search depth. Numbers were checked against the primary sources linked below. Rows marked *second-hand* could only be found quoted in a later paper. Rows marked *self-reported* come from a project's README or training log, not a paper.

## N-tuple networks and search

| Agent | Greedy | 3-ply | Best | Training | Evaluation |
|---|---:|---:|---:|---|---|
| Guei et al. 2022: 2-stage 8×6, optimistic TD + TC, tile-downgrading search | 412,785 | 563,316 | 625,377 ± 40,936 at 6-ply | 200M episodes, 66 h on 20 threads | 1M games greedy, 10k at 3-ply, **100** at 6-ply |
| 2048-rl 2026 *self-reported*: 2-stage, Guei's recipe (optimistic TD then TC), tile-downgrading search | 396,821 | 565,703 | 589,335 ± 12,170 at 4-ply | 107 h, about 1.6×10¹² moves, on a 16-thread laptop and an 8-vCPU Colab machine | 1,000 games (560,276 over 10k at 3-ply) |
| 2048-rl 2026 *self-reported*: single-stage 8×6, TD + TC | 276,699 | 342,382 | 351,671 at 4-ply | 4M games (3.6×10¹⁰ moves), 4 h on 12 threads | **100** games |
| Jaśkowski 2018: 4×6 ("42-33"), delayed TC(0.5), 24 stages with weight promotion, redundant encoding, carousel shaping | 324,710 | 511,759 | 609,104 at 1 s/move | 4×10¹⁰ moves, 5.5 days on 24 cores | 1,000 games (100 for the time-limited result) |
| Jaśkowski 2018: same network, TC(0.5) only | 248,770 | 333,010 | | 10¹⁰ moves, 1.16 days on 24 cores | 1,000 / 300 games |
| Yeh et al. 2016: 3-stage MS-TD, 4×6 + large-tile feature | 143,958 | 350,394 | | 5M games per stage | 10,000 games |
| Yeh et al. 2016: plain TD, 4×6 + large-tile feature | 159,565 | 321,496 | | 15M games, α = 0.0025 | 10,000 games |
| Szubert & Jaśkowski 2014: TD(0), this project's original network | 100,178 | | | 1M games | 1,000 games |
| Matsuzaki 2021: CNN value network trained by TD | 228,100 | 406,927 | | 240 h on one RTX 2080 Ti | best of 10 snapshots |
| nneonneo/xificurk: hand-tuned heuristic expectimax, adaptive depth 4–8 | | | mean 432,557 | none | 2,000 games (Yeh's rerun) |
| macroxue/2048-ai *self-reported*: heuristic expectimax + exhaustively computed endgame tables | | | 711,769 at depth 8 (625,898 at depth 4) | none; the tables need 27 GB of RAM | 1,000 games |
| 2048EndgameTablebase *self-reported*: search at adaptive depth 1–9 + precomputed endgame tablebases | | | 772,353 | none; the largest tables need 32–128 GB of RAM and 1 TB+ of disk to build | about 1,200 games |
| Stochastic MuZero, Antonoglou et al. 2022 | | | ≈510,000 *second-hand* | about 10¹¹ states | |
| Kao et al. 2022: Gumbel + Stochastic MuZero, 3 simulations per move in training | | | 394,645 *second-hand* | | |

**Where this project stands** (1,000 games each): 320,042 greedy and 402,208 at 3-ply, from the [TC-reset decision arm](README.md#decision-arms). It's the [long run](README.md#the-long-run)'s 8×6 network trained on to 2M games (1.9×10¹⁰ moves), with its temporal coherence sums reset at 1.5M games, in under 3 hours of training on 8 threads.
- **Greedy:** above every result here except Jaśkowski's full method (99% of it), 2048-rl's two-stage player (81%) and Guei's (78%).
- **3-ply:** above Yeh, Jaśkowski's TC-only network and 2048-rl's single-stage network, level with Matsuzaki's CNN (407k, after 240 GPU-hours), and 71% of the state of the art.
- **Best search:** 376,221 with [pruned search](README.md#faster-search) at nominal depth 6 on the long run's 1.5M network, at about 0.4 ms per move on one thread. Pruned search isn't N-ply, so its fair comparison is by time per move: Jaśkowski's 609k used 1 s per move, about 2,500× as much.
- **Sample efficiency:** at 1M games (8.8×10⁹ moves) it scores 285k greedy, where Guei reports n-tuple networks under 240k after 10¹⁰ moves, Jaśkowski's TC-only network scores 249k, and 2048-rl's single-stage 8×6 network scores 277k after 3.6×10¹⁰ moves.
- **The weakness is still 32768,** though the reset doubled its rate. At 3-ply, 16384 is reached in 95% of games, but 32768 in 20%.

## Guei et al.: what the state of the art leaves open

Guei's thesis and the OTD paper describe the strongest published agent. The gaps relevant to this project:

- **Sample efficiency.** Everything is tuned for 100M+ episodes. After about 10¹⁰ moves (about 1M games), Guei notes the n-tuple networks "could not even achieve 240000 points". The early figures he reports:

  | Network | Training | Greedy |
  |---|---|---|
  | 4×6, TC | 1M episodes | 208k |
  | 4×6, TC | 10M episodes | 248k |

- **Optimistic initialisation slows early learning.** It stays below plain TC until about 12–13M episodes, and doesn't help the 4×6 network with plain TD at all. Its benefit is late-game exploration, after tens of millions of episodes.
- **Late-game experience is scarce.** Stage 2 starts at the first 16384 tile and is trained from scratch. How its starting positions were produced isn't stated. Restarts from saved positions and carousel shaping are left open. 2048-rl answers part of this (see below).
- **Knowledge doesn't transfer across tile scales.** Tile-downgrading is used only at play time. Using it during training is listed as future work.
- **Training and play don't match.** Training is greedy but play uses 5–6-ply search. Only MCTS-based training targets were tried, and only on a small network.
- **Weak evaluation where the headline is.**
  - The 6-ply headline is averaged over 100 games, with ±12% on the 32768 rate.
  - It's ambiguous whether results come from the best of several networks.
  - Training uses lock-free parallelism, so runs aren't reproducible.

## Since Guei

No peer-reviewed learning result has beaten Guei since 2022. The frontier has moved to open-source projects.

- **2048-rl (Oct 2026) reproduces Guei on a laptop.** It scores 589,335 at 4-ply, with 32768 in 62.5% of games, after 107 hours. It fills in details Guei leaves open:
  - Stage 2 trains on about a million positions saved at the first 16384 tile of a finished stage 1, rather than on games played from the start.
  - Each stage uses plain TD at a fixed rate for 90% of games, then TC for the last 10%. The optimistic initial value is 320,000.
  - Not kept: a third stage, a CNN, and expected-value (search) training targets, which matched the plain player within noise.
  - Its record game (1,762,868) is the best of 2.2M games. It switches to hand-written search and endgame tables once the board holds a 32768 and a 16384, so it isn't a learning result.
  - Its training log records the cost and outcome of every run.
- **Endgame tables decide the top of the table.** macroxue reaches 32768 in 80.5% of games and 65536 in 3.5% at depth 8, and 2048EndgameTablebase 86.1% and 8.4%. No learned agent passes 72% and 0.02%. Their edge is late-game play, where this project is weakest.
- **Small boards have been solved.** Kaneko & Yamashita strongly solved 3×3 (2022) and 4×3 (2025, optimal expected score about 50,724). The value gap between the best and next-best move peaks near the 2048 and 4096 tiles, which they read as why splitting training into stages works. Matsuzaki's group uses the solved 3×3 game to measure n-tuple players against optimal play (Terauchi et al., TAAI 2023) and to test a further optimistic-TD variant (Matsuzaki & Terauchi, CoG 2024).
- **MuZero-style agents.** Kao et al. combine Gumbel and Stochastic MuZero. Agents trained with 3 simulations per move beat ones trained with 16 or 50.
- **Not yet read:** Matsuzaki & Terauchi 2024, Wang's PhD thesis (Double DQN on afterstates, reported as competitive with the best n-tuple agents), and Wang & Matsuzaki 2022 (global embedding for DNN players).

## Sources

- Szubert & Jaśkowski 2014: https://mszubert.github.io/papers/Szubert_2014_CIG.pdf
- Yeh et al. 2016: https://arxiv.org/abs/1606.07374
- Jaśkowski 2018: https://arxiv.org/abs/1604.05085
- Guei et al. 2022 (OTD): https://arxiv.org/abs/2111.11090
- Guei, PhD thesis: https://arxiv.org/abs/2212.11087
- TDL2048+: https://github.com/moporgic/TDL2048
- Matsuzaki 2021: https://www.jstage.jst.go.jp/article/ipsjjip/29/0/29_336/_article
- nneonneo/2048-ai: https://github.com/nneonneo/2048-ai
- Stochastic MuZero: https://openreview.net/forum?id=X6D9bAHhBQ1
- 2048-rl: https://github.com/LoganBradley787/2048-rl (training log: https://github.com/LoganBradley787/2048-rl/blob/main/docs/TRAINING_LOG.md)
- macroxue/2048-ai: https://github.com/macroxue/2048-ai
- 2048EndgameTablebase: https://github.com/game-difficulty/2048EndgameTablebase
- Kaneko & Yamashita 2025: https://arxiv.org/abs/2510.04580
- Kao et al. 2022: https://ieeexplore.ieee.org/document/10056566
- Matsuzaki & Terauchi 2024: https://ieeexplore.ieee.org/document/10645595
- Wang & Matsuzaki 2022: https://doi.org/10.1109/COG51982.2022.9893654
- Wang, PhD thesis: https://kutarr.kochi-tech.ac.jp/record/2000372/files/1258010.pdf
