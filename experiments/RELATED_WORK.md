# Related work

Published 2048 agents, for placing this project's results. All use the standard game: 2 spawns 90% / 4 spawns 10%, score = sum of merged tile values. "N-ply" counts player moves, with 1-ply being greedy, which matches this project's search depth. Numbers were checked against the primary sources linked below. Rows marked *second-hand* could only be found quoted in a later paper.

## N-tuple networks and search

| Agent | Greedy | 3-ply | Best | Training | Evaluation |
|---|---:|---:|---:|---|---|
| Guei et al. 2022: 2-stage 8×6, optimistic TD + TC, tile-downgrading search | 412,785 | 563,316 | 625,377 ± 40,936 at 6-ply | 200M episodes, 66 h on 20 threads | 1M games greedy, 10k at 3-ply, **100** at 6-ply |
| Jaśkowski 2018: 4×6 ("42-33"), delayed TC(0.5), 24 stages with weight promotion, redundant encoding, carousel shaping | 324,710 | 511,759 | 609,104 at 1 s/move | 4×10¹⁰ moves, 5.5 days on 24 cores | 1,000 games (100 for the time-limited result) |
| Jaśkowski 2018: same network, TC(0.5) only | 248,770 | 333,010 | | 10¹⁰ moves, 1.16 days on 24 cores | 1,000 / 300 games |
| Yeh et al. 2016: 3-stage MS-TD, 4×6 + large-tile feature | 143,958 | 350,394 | | 5M games per stage | 10,000 games |
| Yeh et al. 2016: plain TD, 4×6 + large-tile feature | 159,565 | 321,496 | | 15M games, α = 0.0025 | 10,000 games |
| Szubert & Jaśkowski 2014: TD(0), this project's original network | 100,178 | | | 1M games | 1,000 games |
| Matsuzaki 2021: CNN value network trained by TD | 228,100 | 406,927 | | 240 h on one RTX 2080 Ti | best of 10 snapshots |
| nneonneo/xificurk: hand-tuned heuristic expectimax, adaptive depth 4–8 | | | mean 432,557 | none | 2,000 games (Yeh's rerun) |
| Stochastic MuZero, Antonoglou et al. 2022 | | | ≈510,000 *second-hand* | about 10¹¹ states | |

**Where this project stands** (1,000 games each): 299,894 greedy and 365,404 at 3-ply, from the [long run](README.md#the-long-run). It's an 8×6 network trained for 1.5M games (1.4×10¹⁰ moves) in 2 hours on 8 threads.
- **Greedy:** above every result here except Jaśkowski's full method (92% of it) and Guei's (73%).
- **3-ply:** above Yeh and Jaśkowski's TC-only network, and 65% of the state of the art.
- **Sample efficiency:** at 1M games (8.8×10⁹ moves) it scores 285k greedy, where Guei reports n-tuple networks under 240k after 10¹⁰ moves, and Jaśkowski's TC-only network scores 249k.
- **The weakness is now 32768.** At 3-ply, 16384 is reached in 91% of games, level with Yeh, but 32768 in only 8.5%.

## Guei et al.: what the state of the art leaves open

Guei's thesis and the OTD paper describe the strongest published agent. The gaps relevant to this project:

- **Sample efficiency.** Everything is tuned for 100M+ episodes. After about 10¹⁰ moves (about 1M games), Guei notes the n-tuple networks "could not even achieve 240000 points". The early figures he reports:

  | Network | Training | Greedy |
  |---|---|---|
  | 4×6, TC | 1M episodes | 208k |
  | 4×6, TC | 10M episodes | 248k |

- **Optimistic initialisation slows early learning.** It stays below plain TC until about 12–13M episodes, and doesn't help the 4×6 network with plain TD at all. Its benefit is late-game exploration, after tens of millions of episodes.
- **Late-game experience is scarce.** Stage 2 starts at the first 16384 tile and is trained from scratch. How its starting positions were produced isn't stated. Restarts from saved positions and carousel shaping are left open.
- **Knowledge doesn't transfer across tile scales.** Tile-downgrading is used only at play time. Using it during training is listed as future work.
- **Training and play don't match.** Training is greedy but play uses 5–6-ply search. Only MCTS-based training targets were tried, and only on a small network.
- **Weak evaluation where the headline is.**
  - The 6-ply headline is averaged over 100 games, with ±12% on the 32768 rate.
  - It's ambiguous whether results come from the best of several networks.
  - Training uses lock-free parallelism, so runs aren't reproducible.

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
