"""Move a network's downgrading threshold up a tile, warm-starting the weights that become separate.

With downgrading from 16384, every board holding a 16384 is read with its tiles above the largest gap
halved, so the weights for patterns containing a 16384 barely train. Downgrading from 32768 reads those
boards as they are, which gives the 16384 stage weights of its own. Starting them at zero leaves steps
like merging two 16384s with nothing to go on, so each untrained one starts from the weight the old
network would have read instead: the same pattern with its tiles above the pattern's own largest gap
halved. Weights the old network did train keep their values. The checkpoint is saved without temporal
coherence sums, so training resumes with every weight at the full rate.

Usage:
    uv run python scripts/split_downgrade.py checkpoints/ntuple/long-8x6/games_1500000.npz checkpoints/ntuple/split-8x6/games_1500000.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from deep_2048.agents.ntuple import NTupleNetwork


def split_weights(weights: np.ndarray, tuple_sizes: list[int], old_exponent: int, new_exponent: int) -> int:
    """Warm-start, in place, every untrained weight whose largest tile is from old_exponent up to below new_exponent.

    Returns how many weights were filled.
    """
    filled = 0
    start = 0
    for size in tuple_sizes:
        count = 16**size
        table = weights[start : start + count]
        local = np.arange(count, dtype=np.uint32)
        digits = [((local >> np.uint32(4 * i)) & np.uint32(0xF)).astype(np.int64) for i in range(size)]
        top = np.max(digits, axis=0)
        chosen = (top >= old_exponent) & (top < new_exponent) & (table == 0)
        digits = [d[chosen] for d in digits]
        top = top[chosen]

        # The pattern's largest gap: the largest exponent below its top tile that none of its cells hold
        present = np.zeros(len(top), dtype=np.int64)
        for d in digits:
            present |= np.left_shift(1, d)
        gap = np.zeros(len(top), dtype=np.int64)
        for exponent in range(new_exponent - 1, 0, -1):
            missing = (gap == 0) & (exponent < top) & ((present >> exponent) & 1 == 0)
            gap[missing] = exponent

        source = np.zeros(len(top), dtype=np.int64)
        for i, d in enumerate(digits):
            source |= np.left_shift(d - (d > gap), 4 * i)
        table[np.flatnonzero(chosen)] = table[source]
        filled += int(chosen.sum())
        start += count
    return filled


def main() -> None:
    parser = argparse.ArgumentParser(description="Move an n-tuple network's downgrading threshold up, warm-starting the newly separate weights")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--to-tile", type=int, default=32768, help="The new downgrading threshold (default: 32768)")
    args = parser.parse_args()

    network, metadata = NTupleNetwork.load(args.checkpoint)
    if network.downgrade_tile is None or args.to_tile <= network.downgrade_tile:
        raise SystemExit(f"{args.checkpoint} downgrades from {network.downgrade_tile}; --to-tile must be above it")
    weights = network.weights.copy()
    filled = split_weights(weights, [len(t) for t in network.tuples], network.downgrade_tile.bit_length() - 1, args.to_tile.bit_length() - 1)
    split = NTupleNetwork(network.tuples, weights, downgrade_tile=args.to_tile)
    split.save(args.output, metadata)
    print(f"Warm-started {filled:,} weights ({filled / weights.size:.1%}); saved {args.output}, downgrading from {args.to_tile}")


if __name__ == "__main__":
    main()
