# Decision arms, round 9: the corrected second stage carried on to 2.5M games

Part of the two-stage tests in [experiments/README.md](../README.md#rounds-8-and-9-a-second-stage-chosen-by-the-board-a-move-is-made-from). [Round 8](../decision8-8x6-two-stage-by-state/NOTES.md)'s two-stage network at 2M games, trained a further 500k games with the same settings (8×6, temporal coherence at base rate 0.02, downgrading from 16384, a second stage from 16384 chosen by the board a move is made from, late-game starts from 8192 in a fifth of games, 8 threads, seed 42). Compared game for game with round 2's [continue](../decision2-8x6-continue/NOTES.md), the one-stage TC reset carried on over the same games. Evaluated greedily at 2.25M and 2.5M games and at exact depth 3 at 2.5M, on the usual 1,000 seeded games.

## Reproduce

```bash
git checkout $(git log --diff-filter=A -n 1 --format=%h -- experiments/decision9-8x6-two-stage-by-state/NOTES.md)    # the commit that added this folder
ROUND=9 caffeinate -i scripts/run_decision_arms.sh    # about 1.5 hours on a 10-core M4; needs round 8's checkpoint at 2M games
```

Threaded training isn't bit-reproducible, so a rerun gives slightly different numbers.
