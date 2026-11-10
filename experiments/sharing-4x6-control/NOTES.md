# Scale-sharing test: control

Part of the scale-sharing test in [experiments/README.md](../README.md#scale-sharing). It's the 4×6 network with temporal coherence at base rate 0.08, trained from scratch to 300,000 games with the same seed as the other three runs, and evaluated greedily on 1,000 seeded games at 100k, 200k and 300k.

## Reproduce

```bash
git checkout 484ef2b
scripts/run_scale_sharing.sh    # trains and evaluates all four runs
```

No scale-sharing features. The 100k and 200k snapshots are byte-identical to `screen-4x6-tc0.08`'s, so their evaluations came from the result cache.

The four runs trained in parallel on one machine.
