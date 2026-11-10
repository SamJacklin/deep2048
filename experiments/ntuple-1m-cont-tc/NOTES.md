# N-tuple TD, 1M to 1.5M games with temporal coherence

Part of the learning-rate comparison in [experiments/README.md](../README.md#learning-rate-methods). Resumed from `ntuple-td-1m`'s 1M snapshot with the same seed as the other continuation runs, and trained a further 500k games with temporal coherence (Beal & Smith, 1999). Each weight's step is the base rate 0.0025 times |sum of its errors| / sum of their sizes, so TC can only lower a weight's rate from the original. The coherence traces started at zero.

## Reproduce

```bash
git checkout 7e44781
NTUPLE_LEARNING_RATE=0.0025 NTUPLE_TEMPORAL_COHERENCE=true uv run python scripts/train_ntuple.py --run ntuple-1m-cont-tc --resume checkpoints/ntuple/games_1000000.npz --games 1500000 --snapshot-at 1250000 1500000
```

## Costs

- Trains at about 1.5M moves/s against about 2.2M for the decay run on the same machine, since every update also reads and writes the two trace arrays.
- Checkpoints are 404 MB instead of 135 MB, because they carry the traces so training can resume. The agent only loads the weights.
