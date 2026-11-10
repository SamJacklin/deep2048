# Scale-sharing test: synthetic late-game starts (C)

Part of the scale-sharing test in [experiments/README.md](../README.md#scale-sharing). It's the 4×6 network with temporal coherence at base rate 0.08, trained from scratch to 300,000 games with the same seed as the other three runs, and evaluated greedily on 1,000 seeded games at 100k, 200k and 300k.

## Reproduce

```bash
git checkout 484ef2b
scripts/run_scale_sharing.sh    # trains and evaluates all four runs
```

`NTUPLE_LATE_START_FRACTION=0.2 NTUPLE_LATE_START_TILE=8192`. Normal games donate the position where they first make an 8192 tile, its tiles above the gap are doubled, and 20% of games start from those positions once the pool has some. Over the run, 58,508 of the 300,000 games were synthetic. Scores and tile rates in the history cover normal games only.

The four runs trained in parallel on one machine.
