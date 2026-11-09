# N-tuple TD, 1M games

A greedy n-tuple TD(0) afterstate agent trained from scratch to 1,000,000 games with the default settings: four tuples (two straight 4-tuples, two 2x3 rectangles) over 8 symmetries, constant learning rate 0.0025, seed 42.

## Reproduce

```bash
git checkout 667d873
uv run python scripts/train_ntuple.py --games 1000000 --snapshot-at 1000 3000 10000 100000 500000 1000000
```

Training is deterministic: retraining to 100,000 games reproduced byte-identical weights.

## Record

- `manifest.json` was rebuilt after the run, since manifests were added in the following commit. It uses the launch command, the log's created and modified times, and the final checkpoint's metadata.
- 500,000 wasn't in the original `--snapshot-at`. `games_500000.npz` is a copy of `latest.npz` taken when it was saved at exactly 500,000 games, and its metadata confirms 500,000 games and 1,548,206,304 moves. The command above includes 500000 so a rerun saves it directly.
- 21.6 minutes on one core of an Apple M4 (3.69B moves, about 2.85M moves/s including learning).
- `train.log` and `training_history.csv` hold one row per 1,000 training games. `training_progress.png` is the live chart at the end of the run.

## Evaluation

1,000 seeded games per snapshot (`evaluations.json`), mean score with a 95% confidence interval:

| Snapshot | Moves trained | Avg score | ≥1024 | ≥2048 | ≥4096 | ≥8192 |
|---|---:|---:|---:|---:|---:|---:|
| 1k games | 0.26M | 4,331 ± 107 | 0.8% | 0% | 0% | 0% |
| 3k games | 1.1M | 6,953 ± 168 | 16.3% | 0% | 0% | 0% |
| 10k games | 5.0M | 10,648 ± 254 | 55.9% | 1.8% | 0% | 0% |
| 100k games | 158M | 53,748 ± 1,265 | 98.9% | 93.2% | 61.8% | 1.7% |
| 500k games | 1.55B | 102,693 ± 2,598 | 98.8% | 97.0% | 89.7% | 54.5% |
| 1M games | 3.69B | 101,376 ± 2,582 | 98.7% | 97.0% | 88.7% | 54.1% |

## Observations

- Learning arrives in steps, one per new tile. The 2048 rate climbs between about 10k and 50k games, 4096 between 30k and 150k, and 8192 from about 100k, with a sharp jump between 320k and 335k games (training average 72k to 85k).
- It plateaus after about 500k games: the 500k and 1M snapshots are level. The constant learning rate is the main suspect.
- 16384 appeared in about 0.1% of training games from 393k games onwards, and in 1 of the 1,000 evaluation games at 500k.
- 100-game evaluations were misleading at this variance (standard deviation about 40k at 1M). The 100k snapshot scored 49,308 on the first 100 seeds and 53,748 on 1,000.
