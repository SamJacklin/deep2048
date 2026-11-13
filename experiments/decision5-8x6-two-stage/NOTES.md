# Decision arms, round 5: a second stage, chosen by the afterstate

Part of [round 5 of the decision arms](../README.md#round-5-a-second-stage) in experiments/README.md. The long run's 1.5M-game 8×6 network given a second copy of its weights for boards with a 16384 or more (`scripts/add_stage.py`, which plays all 1,000 evaluation games identically to the original), then trained a further 500k games with the long run's settings (temporal coherence at base rate 0.02 with its sums starting from zero, downgrading from 16384, late-game starts from 8192 in a fifth of games, 8 threads, seed 42). Compared game for game with round 1's [TC reset](../decision-8x6-tc-reset/NOTES.md), the same start and reset with one stage.

In this version each afterstate is read in the stage of its own largest tile, so a move that makes the first 16384 is valued by stage 2 and the board's other moves by stage 1. That flaw is what the run shows; `feat/stage-by-state` chooses the stage by the board the move is made from instead.

## Reproduce

```bash
git checkout $(git log --diff-filter=A -n 1 --format=%h -- experiments/decision5-8x6-two-stage/NOTES.md)    # the commit that added this folder
uv run python scripts/add_stage.py checkpoints/ntuple/long-8x6/games_1500000.npz checkpoints/ntuple/two-stage-8x6/games_1500000.npz
ROUND=5 caffeinate -i scripts/run_decision_arms.sh    # about 1.5 hours on a 10-core M4
```

Threaded training isn't bit-reproducible, so a rerun gives slightly different numbers.
