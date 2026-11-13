# Decision arms, round 11: a learning-rate cut on the plateaued network

Part of the decision arms in [experiments/README.md](../README.md#round-11-a-learning-rate-cut). Round 1's [TC reset](../decision-8x6-tc-reset/NOTES.md) at 2M games, trained a further 500k games with the long run's settings (8×6, temporal coherence with its sums kept, downgrading from 16384, late-game starts from 8192 in a fifth of games, 8 threads, seed 42), but at a base rate of 0.005 instead of 0.02. Compared game for game with round 2's [continue](../decision2-8x6-continue/NOTES.md), the same start at 0.02. Evaluated greedily at 2.25M and 2.5M games and at exact depth 3 at 2.5M, on the usual 1,000 seeded games.

## Reproduce

```bash
git checkout $(git log --diff-filter=A -n 1 --format=%h -- experiments/decision11-8x6-rate-0.005/NOTES.md)    # the commit that added this folder
ROUND=11 caffeinate -i scripts/run_decision_arms.sh    # about 1.5 hours on a 10-core M4; needs round 1's tc-reset checkpoint at 2M games
```

Threaded training isn't bit-reproducible, so a rerun gives slightly different numbers.
