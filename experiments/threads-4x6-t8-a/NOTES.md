# Multi-threaded training check: run A

Part of the multi-threaded training check in [experiments/README.md](../README.md#multi-threaded-training). It's the 4×6 network with temporal coherence at base rate 0.08, trained from scratch to 200,000 games, the same configuration as the single-threaded [`screen-4x6-tc0.08`](../screen-4x6-tc0.08/NOTES.md). The first 10,000 games ran on one thread, and the rest on 8 threads updating the shared weights without locks. The two runs, A and B, differ only by chance in the order the threads updated the weights.

## Reproduce

```bash
git checkout f99a1c4
scripts/run_thread_check.sh    # trains and evaluates both runs, and measures speed
```

Threaded training isn't bit-reproducible, so a rerun gives slightly different numbers. The two runs together show how large that variation is.
