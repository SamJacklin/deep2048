# Decision arms, round 8: a second stage, chosen by the board a move is made from

Part of the two-stage tests in [experiments/README.md](../README.md). The same start as [round 5](../decision5-8x6-two-stage/NOTES.md) on `feat/two-stage`: the long run's 1.5M-game 8×6 network given a second copy of its weights for boards with a 16384 or more (`scripts/add_stage.py`), trained a further 500k games with the long run's settings (temporal coherence at base rate 0.02 with its sums starting from zero, downgrading from 16384, late-game starts from 8192 in a fifth of games, 8 threads, seed 42). Compared game for game with round 1's [TC reset](../decision-8x6-tc-reset/NOTES.md), the same start and reset with one stage.

Unlike round 5, every move of a decision is valued, and learned, in the stage of the board it's made from, so the move that makes the first 16384 is judged in the same stage as the board's other moves.

## Reproduce

```bash
git checkout $(git log --diff-filter=A -n 1 --format=%h -- experiments/decision8-8x6-two-stage-by-state/NOTES.md)    # the commit that added this folder
uv run python scripts/add_stage.py checkpoints/ntuple/long-8x6/games_1500000.npz checkpoints/ntuple/two-stage-8x6/games_1500000.npz
ROUND=8 caffeinate -i scripts/run_decision_arms.sh    # about 1.5 hours on a 10-core M4
```

- **Reproducibility:** threaded training isn't bit-reproducible, so a rerun gives slightly different numbers.
- **The run's own summary was lost:** round 9 was added to the arms script while this run's bash was still reading it, so it stopped with a syntax error at its summary step, after all three evaluations had been recorded.
