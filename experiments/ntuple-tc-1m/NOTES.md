# N-tuple TD with temporal coherence from scratch, base rate 0.0025

Part of the from-scratch comparison in [experiments/README.md](../README.md#temporal-coherence-from-scratch). Trained from scratch to 1M games with temporal coherence and base rate 0.0025, using the same seed and snapshot points as `ntuple-td-1m`. It ran in parallel with the other TC run and took 38 minutes.

## Reproduce

```bash
git checkout 29d404d
NTUPLE_LEARNING_RATE=0.0025 NTUPLE_TEMPORAL_COHERENCE=true uv run python scripts/train_ntuple.py --run ntuple-tc-1m --games 1000000 --snapshot-at 1000 3000 10000 100000 500000 1000000
```
