# Decision arms, round 12B: round 12's two-stage network carried on to 2.75M games

Part of the two-stage tests in [experiments/README.md](../README.md#round-12b-round-12-carried-on-to-find-its-plateau). Round 12's two-stage network at 2.5M games, trained a further 250k games with nothing changed and no further reset (temporal coherence at base rate 0.02, downgrading from 16384, a second stage from 16384 chosen by the board a move is made from, late-game starts from 8192 in a fifth of games, 8 threads, seed 42), to find where its climb stops. Evaluated greedily at 2.625M and 2.75M games and at exact depth 3 at 2.75M, on the usual 1,000 seeded games, and compared with [round 12](../decision12-8x6-two-stage-reset/NOTES.md) at 2.5M.

## Reproduce

```bash
git checkout $(git log --diff-filter=A -n 1 --format=%h -- experiments/decision12b-8x6-two-stage-continued/NOTES.md)    # the commit that added this folder
ROUND=12B caffeinate -i scripts/run_decision_arms.sh    # about 50 minutes on a 10-core M4; needs round 12's checkpoint at 2.5M games
```

- **Reproducibility:** threaded training isn't bit-reproducible, so a rerun gives slightly different numbers.
- **A pause:** a short smoke test shared the machine at about 2.53M games, slowing training briefly.
- **The runner's summary** pairs the 2.625M snapshot with round 12's 2.25M one, so its +44,760 there isn't a like-for-like comparison.
