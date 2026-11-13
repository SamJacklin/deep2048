# Decision arms, round 10: a second stage without the reset

Part of the two-stage tests in [experiments/README.md](../README.md#round-10-a-second-stage-without-the-reset). The long run's 1.5M-game 8×6 network given a second stage for boards with a 16384 or more, chosen by the board a move is made from, with its temporal coherence sums kept and copied into the second stage (`scripts/add_stage.py --keep-coherence`), then trained a further 500k games with the long run's settings (temporal coherence at base rate 0.02, downgrading from 16384, late-game starts from 8192 in a fifth of games, 8 threads, seed 42). Compared game for game with round 1's [control](../decision-8x6-control/NOTES.md), the same start with one stage and no reset.

## Reproduce

```bash
git checkout $(git log --diff-filter=A -n 1 --format=%h -- experiments/decision10-8x6-two-stage-kept-sums/NOTES.md)    # the commit that added this folder
ROUND=10 caffeinate -i scripts/run_decision_arms.sh    # about 1.5 hours on a 10-core M4; makes the two-stage start from the long run's 1.5M checkpoint
```

Threaded training isn't bit-reproducible, so a rerun gives slightly different numbers.
