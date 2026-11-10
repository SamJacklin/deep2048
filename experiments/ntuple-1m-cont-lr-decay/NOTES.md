# N-tuple TD, 1M to 1.5M games with a decaying learning rate

Part of the learning-rate comparison in [experiments/README.md](../README.md#learning-rate-methods). Resumed from `ntuple-td-1m`'s 1M snapshot with the same seed as the other continuation runs, and trained a further 500k games while the rate decayed geometrically from 0.0025 to 0.0001. It passed 0.0004 at about 1.29M games.

## Reproduce

```bash
git checkout 7e44781
NTUPLE_LEARNING_RATE=0.0025 NTUPLE_LEARNING_RATE_END=0.0001 uv run python scripts/train_ntuple.py --run ntuple-1m-cont-lr-decay --resume checkpoints/ntuple/games_1000000.npz --games 1500000 --snapshot-at 1250000 1500000
```

Trained alongside `ntuple-1m-cont-tc`. Its snapshots were evaluated with 6 workers while that run was still training, which affects timings only. Agent labels show the rate at the snapshot with full precision, since they were recorded before labels were rounded.
