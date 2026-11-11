# Seed check: seed 123

Part of the seed check in [experiments/README.md](../README.md#seed-to-seed-variation). It's the 4×6 network with temporal coherence at base rate 0.08, single-threaded, trained from scratch to 200,000 games with seed 123 instead of the usual 42. Otherwise it's identical to [`screen-4x6-tc0.08`](../screen-4x6-tc0.08/NOTES.md). Evaluated greedily on the usual 1,000 seeded games at 100k and 200k.

## Reproduce

```bash
git checkout 7f28a17
scripts/run_seed_check.sh    # trains seeds 7 and 123 in parallel, then evaluates them
```
