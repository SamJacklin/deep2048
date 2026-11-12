# Decision arms, round 3: optimism at 1,000

Part of [round 3 of the decision arms](../README.md#round-3-optimism-for-never-updated-weights) in experiments/README.md. Round 1's [TC-reset arm](../decision-8x6-tc-reset/NOTES.md) at 2M games, with every weight that training had never updated (still exactly 0, 87,695,250 of 134,217,728) set to 1,000, then trained a further 500k games with the long run's settings (8×6, temporal coherence at base rate 0.02, downgrading from 16384, late-game starts from 8192 in a fifth of games, 8 threads, seed 42). Compared game for game with round 2's [continue](../decision2-8x6-continue/NOTES.md), which started from the same network and changed nothing. Evaluated greedily at 2.25M and 2.5M games and at exact depth 3 at 2.5M, on the usual 1,000 seeded games.

## Reproduce

```bash
git checkout $(git log --diff-filter=A -n 1 --format=%h -- experiments/decision3-8x6-optimism-1000/NOTES.md)    # the commit that added this folder
ROUND=3 caffeinate -i scripts/run_decision_arms.sh    # both arms, about 2.5 hours on a 10-core M4; needs round 1's tc-reset checkpoint at 2M games
```

Or this arm alone:

```bash
NTUPLE_NETWORK=8x6 NTUPLE_LEARNING_RATE=0.02 NTUPLE_TEMPORAL_COHERENCE=true NTUPLE_THREADS=8 NTUPLE_DOWNGRADE_TILE=16384 \
NTUPLE_LATE_START_FRACTION=0.2 NTUPLE_LATE_START_TILE=8192 \
uv run python scripts/train_ntuple.py --run decision3-8x6-optimism-1000 --resume checkpoints/ntuple/decision-8x6-tc-reset/games_2000000.npz --optimistic-unvisited 1000 --games 2500000 --snapshot-at 2250000 2500000
```

Threaded training isn't bit-reproducible, so a rerun gives slightly different numbers.
