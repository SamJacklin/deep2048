# TD(λ) test: longer horizon, seed 3

Part of the multi-seed TD(λ) test in [experiments/README.md](../README.md#tdλ-across-seeds). It's the 4×6 network with temporal coherence at base rate 0.08, downgrading and late-game starts (A + C), trained from scratch to 300,000 games on 8 threads (the first 10,000 single-threaded) with seed 3, using delayed TD(λ): λ = 0.75 with horizon 6. Evaluated greedily on the usual 1,000 seeded games at 300k.

**Result:** 124,120 average, with 16384 reached in 0.0% of games. That's −63,378 against the TD(0) control on the same seed ([`ingredients-4x6-downgrade-late-starts-s3`](../ingredients-4x6-downgrade-late-starts-s3/NOTES.md)).

## Reproduce

```bash
git checkout d93c4f8
scripts/run_lambda_test.sh    # all three arms × seeds 1, 2 and 3, then a summary against TD(0)
```

Threaded training isn't bit-reproducible, so a rerun gives slightly different numbers.
