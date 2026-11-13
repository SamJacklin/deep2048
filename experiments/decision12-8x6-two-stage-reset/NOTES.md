# Decision arms, round 12: the two-stage network given a reset

Part of the two-stage tests in [experiments/README.md](../README.md#round-12-a-reset-for-the-two-stage-network). Round 10's two-stage network (split with its sums kept) at 2M games, with its temporal coherence sums reset (`--reset-coherence`), trained a further 500k games with the long run's settings (temporal coherence at base rate 0.02, downgrading from 16384, a second stage from 16384 chosen by the board a move is made from, late-game starts from 8192 in a fifth of games, 8 threads, seed 42). Compared game for game with round 2's [continue](../decision2-8x6-continue/NOTES.md), the one-stage network over the same games. Evaluated greedily at 2.25M and 2.5M games and at exact depth 3 at 2.5M, on the usual 1,000 seeded games.

## Reproduce

```bash
git checkout $(git log --diff-filter=A -n 1 --format=%h -- experiments/decision12-8x6-two-stage-reset/NOTES.md)    # the commit that added this folder
ROUND=12 caffeinate -i scripts/run_decision_arms.sh    # about 1.5 hours on a 10-core M4; needs round 10's checkpoint at 2M games
```

- **Reproducibility:** threaded training isn't bit-reproducible, so a rerun gives slightly different numbers.
- **A pause:** training was paused for about 10 minutes at 2.065M games while a speed benchmark ran on the machine.
