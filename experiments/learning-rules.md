# Learning rules: TD, temporal coherence and TD(λ)

Notes for the write-up on how the learning rule affects n-tuple training for 2048, using this project's runs and the published ones.

## How the three relate

They are not alternatives to one another. They control different parts of the same update.

- **TD learning** is the algorithm every run here uses: after each move, the value of the previous afterstate moves towards the reward plus the value of the next afterstate. This project uses TD(0) on afterstates, as in Szubert & Jaśkowski (2014).
- **Temporal coherence (TC)** (Beal & Smith, 1999) is a *learning-rate rule* for TD. Each weight's step is the base rate times |sum of its recent errors| / sum of their sizes. That's near 1 while a weight's errors keep pointing the same way, and near 0 once they cancel out. So "TD vs TC" means TD with a constant rate vs TD with per-weight adaptive rates.
- **TD(λ)** changes the *target* each update moves towards. It blends several steps of real rewards with the value estimate instead of looking one move ahead, which can carry information back faster. It combines with TC: Jaśkowski (2018) used TC with λ = 0.5. This project tested it with TC in a multi-seed test (below).

## Evidence: TC vs a constant rate

| Comparison | Training | Constant rate | TC | Source |
|---|---|---:|---:|---|
| From scratch, same base rate 0.0025 | 1M games | 101,376 | 112,398 | [`ntuple-td-1m`](ntuple-td-1m/NOTES.md), [`ntuple-tc-1m`](ntuple-tc-1m/NOTES.md) |
| From scratch, TC with a 4× base rate (0.01) | 1M games | 101,376 | 125,498 | [`ntuple-tc-1m-lr0.01`](ntuple-tc-1m-lr0.01/NOTES.md) |
| Continuing a plateaued 1M network | to 1.5M games | 102,220 (same rate); 118,983 (best constant, 0.0004) | 128,208 | [Learning-rate methods](README.md#learning-rate-methods) |
| Jaśkowski 2018, TD(0.5) vs TC(0.5), 4×6 network | 10¹⁰ moves | 140,397 | 248,770 | Jaśkowski 2018, Table II |

Averages are greedy over 1,000 games, except Jaśkowski's, which are his own evaluations.

- **At the budgets tested here, TC wins every time.** It learns fast while weights are badly wrong, and settles them once their errors stop agreeing, which a single constant rate can't do. A constant rate is either too slow early or too noisy late.
- **TC does slow learning in the middle of training at a careful base rate.** From scratch at 0.0025 it trailed the constant rate from about 50k to 900k games (32,300 vs 53,748 at 100k; 84,955 vs 102,693 at 500k), then overtook. With a 4× higher base rate it led throughout. TC works best as a brake on an aggressive rate.
- **TC lets the rate be much higher.** In the network screen, every doubling of the TC base rate helped up to 0.04–0.08 on the 4×6, which is 16–32× the constant rate used by Szubert & Jaśkowski and Yeh et al.

## Evidence: TD(λ) with TC

The full results are in [TD(λ) across seeds](README.md#tdλ-across-seeds). The setup: delayed TD(λ) on the 4×6 with TC at base rate 0.08, downgrading and late-game starts, trained for 300k games with seeds 1–3, against TD(0) with the same settings. Values are greedy averages over 1,000 games.

| Target | Mean of seeds | vs TD(0), paired by seed |
|---|---:|---:|
| TD(0) | 186,906 | — |
| λ = 0.5, horizon 3 (Jaśkowski's setting) | 155,724 | −31,182 ± 1,105 |
| λ = 0.75, horizon 6 | 126,074 | −60,832 ± 2,247 |
| λ = 0.75 → 0 over the session, horizon 6 | 155,061 | −31,845 ± 2,855 |

- **At this rate and budget, TD(λ) hurts, and more λ hurts more.** Every variant lost on every seed.
- **λ = 0.5 helps early.** It led TD(0) on all three seeds for the first ~70k games, then fell behind, and the gap kept widening to the end. Multi-step returns help while the value estimates are poor. Once the estimates are good, each extra step mostly adds a tile spawn's randomness.
- **The two likely ways that noise hurts both involve the rate:**
  - At base rate 0.08 the noise goes straight into the weights.
  - Noisier errors cancel more, so TC shrinks each weight's rate sooner.
- **The damage outlasts λ.** Annealing λ to 0 left the run learning about half as fast as TD(0) over the last 100k games, when λ was already below 0.25. TC's error sums accumulate over the whole run, so a noisy phase can leave the rates permanently low.

## Over very long training

Guei (2022) trained for 100M games per network, with optimistic initialisation (thesis Table 5). Values are greedy averages:

| Network | Constant rate (OTD) | TC throughout (OTC) | Constant, then TC for the last 10% (OTD+TC) |
|---|---:|---:|---:|
| 4×6 | 239,075 | **280,281** | 261,433 |
| 8×6 | 361,471 | 360,228 | **370,907** |

- On the smaller network TC is clearly best. On the larger one, the best recipe is a high constant rate for most of training, with TC only to fine-tune at the end.
- Guei's explanation (Appendix A): TC "converges very fast" and "cannot find a solution to 32768-tiles before the convergence". TC's rates shrink as errors cancel, so over a very long run it can settle before discovering rare late-game strategies, while a constant high rate keeps exploring.
- In his run that averaged snapshots, Guei reset the TC sums every 1M games, which restores each weight's full rate.

## Open questions for this project

1. **Does TC stall over long training here?** This project's TC runs were still improving at 1.5M games. The [long run](README.md#the-long-run) (1.5M games on the 8×6) doesn't stall either: its greedy score gained +14.7k ± 10.7k from 1M to 1.5M games, close to the +17.5k from 500k to 1M. Guei's stall came after tens of millions of games, so periodic TC resets are still worth testing before a run that long.
2. **Constant rate, then TC:** worth testing only for much longer training (tens of millions of games), where Guei found it best on the 8×6.
3. **When can TD(λ) help?** At base rate 0.08 it only helps for the first ~70k games. Three cheap tests would show whether its early gain can be kept:
   - λ for the early phase only, then TD(0).
   - TD(λ) at lower base rates (0.02–0.04), where the extra noise should matter less.
   - Resetting the TC sums when λ is switched off.

   Jaśkowski's λ = 0.5 was chosen for 10¹⁰ moves of training, so it may also pay off only over much longer runs.
4. **How TC interacts with the scale-sharing ingredients.** All the ingredient tests used TC. Late-game starts in particular change which weights see coherent errors.

## Sources

- Beal & Smith 1999, temporal coherence: D. F. Beal and M. C. Smith, "Temporal coherence and prediction decay in TD learning", IJCAI 1999.
- Szubert & Jaśkowski 2014: https://mszubert.github.io/papers/Szubert_2014_CIG.pdf
- Jaśkowski 2018: https://arxiv.org/abs/1604.05085
- Guei, PhD thesis 2022: https://arxiv.org/abs/2212.11087
