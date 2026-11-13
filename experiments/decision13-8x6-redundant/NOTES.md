# Decision arms, round 13: redundant small tuples added to a trained network

Part of the decision arms in [experiments/README.md](../README.md#round-13-redundant-small-tuples). The TC-reset network at 2M games with Jaśkowski's redundant tuples (two rows and three squares as 4-tuples, and two 3-tuples, `8x6-redundant`) added at zero by `scripts/add_tuples.py`, trained a further 500k games with the long run's settings (temporal coherence at base rate 0.0107, so a board's total step over its 120 lookups matches the 8x6 network's 64 at 0.02, downgrading from 16384, late-game starts from 8192 in a fifth of games, 8 threads, seed 42). Compared game for game with round 2's [continue](../decision2-8x6-continue/NOTES.md). Evaluated greedily at 2.25M and 2.5M games and at exact depth 3 at 2.5M, on the usual 1,000 seeded games.

## Reproduce

```bash
git checkout $(git log --diff-filter=A -n 1 --format=%h -- experiments/decision13-8x6-redundant/NOTES.md)    # the commit that added this folder
ROUND=13 caffeinate -i scripts/run_decision_arms.sh    # about 1.5 hours on a 10-core M4; needs the TC-reset network at 2M games
```

- **Reproducibility:** threaded training isn't bit-reproducible, so a rerun gives slightly different numbers.
