# TD(λ) test: Jaśkowski's setting, seed 2

Part of the multi-seed TD(λ) test in [experiments/README.md](../README.md#tdλ-across-seeds). It's the 4×6 network with temporal coherence at base rate 0.08, downgrading and late-game starts (A + C), trained from scratch to 300,000 games on 8 threads (the first 10,000 single-threaded) with seed 2, using delayed TD(λ): λ = 0.5 with horizon 3, close to Jaśkowski (2018). His h = 3 sums four errors, which is horizon 4 here. Evaluated greedily on the usual 1,000 seeded games at 300k.

**Result:** 155,942 average, with 16384 reached in 12.3% of games. That's −30,866 against the TD(0) control on the same seed ([`ingredients-4x6-downgrade-late-starts-s2`](../ingredients-4x6-downgrade-late-starts-s2/NOTES.md)).

## Reproduce

```bash
git checkout 0e0470c
scripts/run_lambda_test.sh    # all three arms × seeds 1, 2 and 3, then a summary against TD(0)
```

Threaded training isn't bit-reproducible, so a rerun gives slightly different numbers.
