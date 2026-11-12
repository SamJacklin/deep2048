# Decision arms: late-game share

Part of the [decision arms](../README.md#decision-arms) in experiments/README.md. The long run's 1.5M-game 8×6 network, trained a further 500k games with the long run's settings (temporal coherence at base rate 0.02, downgrading from 16384, late-game starts from 8192 in a fifth of games, 8 threads, seed 42), with half of all games starting from late-game positions instead of a fifth (`NTUPLE_LATE_START_FRACTION=0.5`). Evaluated greedily at 1.75M and 2M games and at exact depth 3 at 2M, on the usual 1,000 seeded games.

## Reproduce

```bash
git checkout $(git log --diff-filter=A -n 1 --format=%h -- experiments/decision-8x6-late-share/NOTES.md)    # the commit that added this folder
caffeinate -i scripts/run_decision_arms.sh    # all four arms, about 4.5 hours on a 10-core M4; needs checkpoints/ntuple/long-8x6/games_1500000.npz
```

Or this arm alone:

```bash
NTUPLE_NETWORK=8x6 NTUPLE_LEARNING_RATE=0.02 NTUPLE_TEMPORAL_COHERENCE=true NTUPLE_THREADS=8 NTUPLE_DOWNGRADE_TILE=16384 \
NTUPLE_LATE_START_FRACTION=0.2 NTUPLE_LATE_START_TILE=8192 NTUPLE_LATE_START_FRACTION=0.5 \
uv run python scripts/train_ntuple.py --run decision-8x6-late-share --resume checkpoints/ntuple/long-8x6/games_1500000.npz --games 2000000 --snapshot-at 1750000 2000000
```

- **Reproducibility:** threaded training isn't bit-reproducible, so a rerun gives slightly different numbers.
- **The late-game pool isn't checkpointed,** so like any resumed run, the arm made no late-game starts until its first games had donated positions.
- **The recorded commit is `cf2e599`,** the first commit of the arms script. Later commits generalised the reset into a scale, without changing what this arm does.
