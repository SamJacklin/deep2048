# Decision arms, round 6: temporal coherence forgetting at 0.9999

Part of [round 6 of the decision arms](../README.md#round-6-temporal-coherence-with-forgetting) in experiments/README.md. Round 1's [TC reset](../decision-8x6-tc-reset/NOTES.md) at 2M games, trained a further 500k games with the long run's settings (8×6, temporal coherence at base rate 0.02, downgrading from 16384, late-game starts from 8192 in a fifth of games, 8 threads, seed 42), except that each weight's two temporal coherence sums shrink by a factor of 0.9999 before every update, so its rate follows roughly its last 10,000 errors. Compared game for game with round 2's [continue](../decision2-8x6-continue/NOTES.md), the same start without forgetting. Evaluated greedily at 2.25M and 2.5M games and at exact depth 3 at 2.5M, on the usual 1,000 seeded games.

## Reproduce

```bash
git checkout $(git log --diff-filter=A -n 1 --format=%h -- experiments/decision6-8x6-forget-0.9999/NOTES.md)    # the commit that added this folder
ROUND=6 caffeinate -i scripts/run_decision_arms.sh    # both arms, about 2.5 hours on a 10-core M4; needs round 1's tc-reset checkpoint at 2M games
```

Threaded training isn't bit-reproducible, so a rerun gives slightly different numbers.
