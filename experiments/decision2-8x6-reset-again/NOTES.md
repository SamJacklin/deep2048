# Decision arms, round 2: reset again

Part of [round 2 of the decision arms](../README.md#round-2-do-resets-keep-paying) in experiments/README.md. Round 1's [TC-reset arm](../decision-8x6-tc-reset/NOTES.md) at 2M games, trained a further 500k games with the long run's settings (8×6, temporal coherence at base rate 0.02, downgrading from 16384, late-game starts from 8192 in a fifth of games, 8 threads, seed 42), with the temporal coherence sums reset a second time at the branch point. Evaluated greedily at 2.25M and 2.5M games and at exact depth 3 at 2.5M, on the usual 1,000 seeded games.

## Reproduce

```bash
git checkout $(git log --diff-filter=A -n 1 --format=%h -- experiments/decision2-8x6-reset-again/NOTES.md)    # the commit that added this folder
ROUND=2 caffeinate -i scripts/run_decision_arms.sh    # all three arms, about 4 hours on a 10-core M4; needs round 1's tc-reset checkpoint at 2M games
```

Or this arm alone:

```bash
NTUPLE_NETWORK=8x6 NTUPLE_LEARNING_RATE=0.02 NTUPLE_TEMPORAL_COHERENCE=true NTUPLE_THREADS=8 NTUPLE_DOWNGRADE_TILE=16384 \
NTUPLE_LATE_START_FRACTION=0.2 NTUPLE_LATE_START_TILE=8192 \
uv run python scripts/train_ntuple.py --run decision2-8x6-reset-again --resume checkpoints/ntuple/decision-8x6-tc-reset/games_2000000.npz --reset-coherence --games 2500000 --snapshot-at 2250000 2500000
```

- **Reproducibility:** threaded training isn't bit-reproducible, so a rerun gives slightly different numbers.
- **The machine was shared** with other work for part of the run, so its games/s are lower than round 1's.
