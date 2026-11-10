# Scale-sharing test: downgrading only (A)

Part of the scale-sharing test in [experiments/README.md](../README.md#scale-sharing). It's the 4×6 network with temporal coherence at base rate 0.08, trained from scratch to 300,000 games with the same seed as the other three runs, and evaluated greedily on 1,000 seeded games at 100k, 200k and 300k.

## Reproduce

```bash
git checkout 484ef2b
scripts/run_scale_sharing.sh    # trains and evaluates all four runs
```

`NTUPLE_DOWNGRADE_TILE=16384`. Boards with a 16384 tile or larger and a gap are read by the tuples with the tiles above the gap halved. Identical to the control until the first 16384 appears.

The four runs trained in parallel on one machine.
