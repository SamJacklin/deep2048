# Redundant small tuples trained from scratch (stopped at 264k games)

Part of the redundant-encoding tests in [experiments/README.md](../README.md#redundant-tuples-from-scratch). The `8x6-redundant` network (the 8x6 network plus Jaśkowski's two rows and three squares as 4-tuples and two 3-tuples) trained from zero with the long run's recipe: temporal coherence at base rate 0.0107 (so a board's total step over its 120 lookups matches the 8x6 network's 64 at 0.02), downgrading from 16384, late-game starts from 8192 in a fifth of games, 8 threads, seed 42. Compared window for window and on the same 1,000 seeded greedy games with the [long run](../long-8x6/NOTES.md). Stopped by hand at 264k games, after the 250k evaluation, because it had stopped reaching 16384.

## Reproduce

```bash
git checkout $(git log --diff-filter=A -n 1 --format=%h -- experiments/long-8x6-redundant/NOTES.md)    # the commit that added this folder
RUN=long-8x6-redundant NETWORK=8x6-redundant LEARNING_RATE=0.0107 caffeinate -i scripts/run_long_run.sh    # stop it after the 250k evaluation, about 20 minutes on a 10-core M4
```

- **Reproducibility:** threaded training isn't bit-reproducible, so a rerun gives slightly different numbers.
- **The weight sizes** in the README come from the 250k checkpoint, which has since been deleted.
