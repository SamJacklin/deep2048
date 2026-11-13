# The long run, downgrading from 32768 instead of 16384

Results and discussion are in [experiments/README.md](../README.md#downgrading-from-32768-instead-of-16384). The [long run](../long-8x6/NOTES.md)'s setup, trained from scratch to 1,500,000 games, except that only boards holding a 32768 are downgraded, so boards with a 16384 are read as they are and the 16384 stage gets weights of its own. Seed 42, 8 threads (the first 10,000 games single-threaded), temporal coherence at base rate 0.02, late-game starts from 8192 in a fifth of games. A snapshot every 250k games was evaluated greedily on the usual 1,000 seeded games while training went on, and the 1.5M network also at depths 2 and 3.

## Reproduce

```bash
git checkout $(git log --diff-filter=A -n 1 --format=%h -- experiments/long-8x6-dg32768/NOTES.md)    # the commit that added this folder
RUN=long-8x6-dg32768 DOWNGRADE_TILE=32768 EVAL_EVERY=250000 caffeinate -i scripts/run_long_run.sh    # about 2.5 hours on a 10-core M4
```

- **Snapshots:** the 750k and 1.25M snapshots were deleted once evaluated, and the others once the run was archived, to save disk space.
- **Reproducibility:** threaded training isn't bit-reproducible, so a rerun gives slightly different numbers.
- **Training history:** the first run with the `rate_32768` column.
