# The long run: 8×6, temporal coherence 0.02, downgrading and late-game starts, to 1.5M games

Results and discussion are in [experiments/README.md](../README.md#the-long-run). The network screen's best layout and rate (8×6, temporal coherence at base rate 0.02) is combined with the ingredient test's best pair: downgrading from 16384 (A) and late-game starts (C: 20% of games from synthetic positions with an 8192 tile). It was trained from scratch to 1,500,000 games on 8 threads, with the first 10,000 games single-threaded, and seed 42. Each snapshot (100k, 250k, 500k, 1M, 1.5M) is evaluated greedily on the usual 1,000 seeded games, and the 1.5M network is also evaluated with depth-2 and depth-3 search.

## Reproduce

```bash
git checkout a30d8af
caffeinate -i scripts/run_long_run.sh    # trains, then runs every evaluation; rerunning resumes an interrupted run
```

- **Time:** training took 2 h 02 min on a 10-core M4 (13.86B moves, about 1.9M moves/s). The greedy evaluations took about 5 minutes each, depth 2 took 7 minutes and depth 3 took 31 minutes.
- **Resources:** the run needs about 1.5 GB of RAM for training, about 660 MB per evaluation worker, and 9 GB of disk for its six checkpoints.
- **Reproducibility:** threaded training isn't bit-reproducible, so a rerun gives slightly different numbers.
- **Training speed:** the dips in training speed around 900k games came from other work on the machine.
