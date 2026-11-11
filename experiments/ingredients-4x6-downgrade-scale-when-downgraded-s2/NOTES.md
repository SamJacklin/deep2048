# Ingredient test: downgrade-scale-when-downgraded, seed 2

Part of the multi-seed ingredient test in [experiments/README.md](../README.md#ingredient-test-across-seeds). It's the 4×6 network with temporal coherence at base rate 0.08, trained from scratch to 300,000 games on 8 threads (the first 10,000 single-threaded) with seed 2, and downgrading plus the scale term on downgraded boards only (B′): `NTUPLE_DOWNGRADE_TILE=16384 NTUPLE_SCALE_FEATURE=true NTUPLE_SCALE_WHEN_DOWNGRADED=true`. Evaluated greedily on the usual 1,000 seeded games at 300k.

## Reproduce

```bash
git checkout 6b35224
scripts/run_ingredient_test.sh    # all five arms × seeds 1, 2 and 3, then a summary
```

Threaded training isn't bit-reproducible, so a rerun gives slightly different numbers. The runs recorded the commit current when each started (`ac9b768`, `6b35224` or `6558b73`), and only documentation and the long-run script differ between those commits.
