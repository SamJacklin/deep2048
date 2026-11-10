# Scale-sharing test: downgrading and scale term (B)

Part of the scale-sharing test in [experiments/README.md](../README.md#scale-sharing). It's the 4×6 network with temporal coherence at base rate 0.08, trained from scratch to 300,000 games with the same seed as the other three runs, and evaluated greedily on 1,000 seeded games at 100k, 200k and 300k.

## Reproduce

```bash
git checkout 484ef2b
scripts/run_scale_sharing.sh    # trains and evaluates all four runs
```

`NTUPLE_DOWNGRADE_TILE=16384 NTUPLE_SCALE_FEATURE=true`. As downgrading only, plus one weight from a 1,024-entry table indexed by the real board's counts of 2048–32768 tiles. Its "no large tiles" entry is read on nearly every early board, so it acts as a shared bias weight updated at the full rate. The run falls behind from the first thousand games, before any large tile exists, which points to this entry rather than downgrading.

The four runs trained in parallel on one machine.
