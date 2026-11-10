# Network screen: 8x6 network, temporal coherence at base rate 0.01

Part of the network screen in [experiments/README.md](../README.md#network-screen). Trained from scratch to 200,000 games with temporal coherence, using the same seed as the other screen runs, and evaluated greedily on 1,000 seeded games at 50k, 100k and 200k.

## Reproduce

```bash
git checkout 01fa340
NTUPLE_NETWORK=8x6 NTUPLE_LEARNING_RATE=0.01 NTUPLE_TEMPORAL_COHERENCE=true uv run python scripts/train_ntuple.py --run screen-8x6-tc0.01 --games 200000 --snapshot-at 50000 100000 200000
```

Up to seven screen runs trained at once on the same machine, so the games/s in the history reflect a shared CPU and memory bandwidth, not this network's speed alone.
