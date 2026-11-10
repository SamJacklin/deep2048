# N-tuple TD, 1M to 1.5M games at learning rate 0.001

Part of the learning-rate comparison in [experiments/README.md](../README.md#learning-rate-after-the-plateau). Resumed from `ntuple-td-1m`'s 1M snapshot and trained a further 500k games at learning rate 0.001, with the same seed as the other two runs so they see the same opening tile spawns.

## Reproduce

```bash
git checkout cb5060c
NTUPLE_LEARNING_RATE=0.001 uv run python scripts/train_ntuple.py --run ntuple-1m-cont-lr0.001 --resume checkpoints/ntuple/games_1000000.npz --games 1500000 --snapshot-at 1250000 1500000
```

The three runs trained in parallel on one machine, so their games/s figures affect each other. The 1.25M snapshots were evaluated while training was still running, before agent labels included the learning rate.
