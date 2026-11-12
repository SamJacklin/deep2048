# Search on the long run's 1.5M network: exact and pruned, depths 3–6

Results and discussion are in [experiments/README.md](../README.md#faster-search). The 1.5M-game snapshot of [`long-8x6`](../long-8x6/NOTES.md) was evaluated with expectimax at six settings, on the usual 1,000 seeded games each: exact depth 3, depth 3 at cutoff 0.05, depth 4 at cutoffs 0.1 and 0.05, and depths 5 and 6 at cutoff 0.1. No training was involved.

## Reproduce

```bash
git checkout $(git log --diff-filter=A -n 1 --format=%h -- experiments/search-long-8x6/NOTES.md)    # the commit that added this folder
scripts/run_long_run.sh    # if checkpoints/ntuple/long-8x6/games_1500000.npz isn't there yet (about 2 hours)
uv run python scripts/evaluate.py checkpoints/ntuple/long-8x6/games_1500000.npz --agent ntuple_search --depth 6 --cutoff 0.1 -n 1000 --experiment experiments/search-long-8x6
```

Repeat the last line for each setting: `--depth 3`, then `--depth 3 --cutoff 0.05`, `--depth 4 --cutoff 0.1`, `--depth 4 --cutoff 0.05` and `--depth 5 --cutoff 0.1`.

- **How the games were played:** with the search code of the commit that added this folder, from a short script calling `run_games` rather than `evaluate.py`. The records were written afterwards, so `recorded_at` is the archiving time.
- **The guard:** the check that stops a game holding two 32768 tiles was added after these games were played. None of them can have held two. A 32768 is worth at least 13 × 32768 = 425,984 points even if every spawn were a 4, so two need at least 852k, and the highest score in these runs is 741k.
- **Reproducibility:** search is deterministic, so a rerun gives the same games. Exact depth 3 reproduces the long run's stored depth-3 evaluation game for game.
- **Times:** each setting took 8–15 minutes on 10 threads of the M4. The exception is depth 4 at cutoff 0.05, whose run spanned the machine sleeping, so its recorded 45 minutes isn't meaningful. The per-move benchmark puts it at about 20 minutes.
