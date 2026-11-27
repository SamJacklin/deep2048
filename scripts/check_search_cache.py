"""Measure what the search's transposition table changes when unlikely spawns are pruned.

Usage:
    uv run python scripts/check_search_cache.py checkpoints/ntuple/decision-8x6-tc-reset/games_2000000.npz

The table is keyed by board and depth. Without pruning that's exact, but with a cutoff, how much of a subtree is
searched depends on the probability of the path into it, so a value cached from a likelier path is reused for less
likely ones. For each search setting, this searches the same positions with the table and without it, and reports
how often the chosen move changes, how far the chosen move's value moves, and the time per position each way.
Positions are boards greedy play with the network decides from, half of them with an 8192 or larger tile.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from deep_2048.agents.ntuple import NTupleNetwork
from deep_2048.agents.ntuple.search import action_values
from deep_2048.game import bitboard as bb
from scripts.update_gain import greedy_afterstates

SETTINGS = ((3, 0.0), (4, 0.05), (5, 0.1), (6, 0.1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure what the transposition table changes in pruned search")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--positions", type=int, default=200, help="Positions to search (default: 200)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    network, _ = NTupleNetwork.load(args.checkpoint)
    _, states = greedy_afterstates(network.weights, network.layout, 4, args.seed)
    rng = np.random.default_rng(args.seed)
    late = np.array([bb.max_exponent(s) >= 13 for s in states])
    half = args.positions // 2
    positions = np.concatenate([rng.choice(states[late], half, replace=False), rng.choice(states[~late], args.positions - half, replace=False)])

    action_values(network.weights, network.layout, positions[0], 2, 0.0, False)  # compile before timing
    print("| Depth | Cutoff | Positions | Chosen move changes | Chosen value change: median | max | ms per position, table | without |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for depth, cutoff in SETTINGS:
        changes, value_changes, seconds = 0, [], [0.0, 0.0]
        for board in positions:
            values = []
            for k, reuse in enumerate((True, False)):
                start = time.perf_counter()
                values.append(action_values(network.weights, network.layout, board, depth, cutoff, reuse))
                seconds[k] += time.perf_counter() - start
            reused, exact = values
            changes += int(np.nanargmax(reused) != np.nanargmax(exact))
            chosen = np.nanargmax(exact)
            value_changes.append(abs(reused[chosen] - exact[chosen]) / abs(exact[chosen]))
        per = [1000 * s / len(positions) for s in seconds]
        print(
            f"| {depth} | {cutoff:g} | {len(positions)} | {changes} ({changes / len(positions):.1%}) | "
            f"{np.median(value_changes):.2e} | {np.max(value_changes):.2e} | {per[0]:.1f} | {per[1]:.1f} |"
        )


if __name__ == "__main__":
    main()
