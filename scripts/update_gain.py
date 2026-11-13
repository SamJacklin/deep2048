"""Measure how far one temporal coherence update really moves a board's value, by stage of the game.

Usage:
    uv run python scripts/update_gain.py checkpoints/ntuple/decision-8x6-tc-reset/games_2000000.npz

The configured rate is per weight, but a board reads 64 weights (8 tuples × 8 symmetries), and a weight read m
times by one board is also updated m times. So an update with TD error δ changes the board's value by
δ × lr × Σ c_i m_i² over its distinct weights i, where c_i is weight i's coherence (its rate as a share of lr).
That sum, the effective gain ΔV/δ, is reported against the nominal lr × 64 that every weight at full rate with no
repeats would give. Boards are the afterstates of greedy games played with the network itself, the same moves
training makes, grouped by their largest tile and by whether downgrading changed what the tuples read.

A gain near zero means the network has stopped learning there, whatever the error. A gain far above 1 means a
single update overshoots the error it corrects.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from numba import njit

sys.path.insert(0, str(Path(__file__).parent.parent))

from deep_2048.agents.ntuple import NTupleNetwork
from deep_2048.agents.ntuple.network import TupleLayout, evaluate, load_traces, weight_indices
from deep_2048.game import bitboard as bb


def update_gain(indices: np.ndarray, errors: np.ndarray, abs_errors: np.ndarray, learning_rate: float) -> tuple[float, float, float]:
    """Return a board's effective gain ΔV/δ, its repeat factor Σ m_i² / len(indices), and the mean coherence of its reads."""
    distinct, counts = np.unique(indices, return_counts=True)
    sums = abs_errors[distinct]
    coherence = np.where(sums > 0, np.abs(errors[distinct]) / np.where(sums > 0, sums, 1), 1.0)
    gain = learning_rate * float(np.sum(coherence * counts**2))
    return gain, float(np.sum(counts**2)) / len(indices), float(np.sum(coherence * counts)) / len(indices)


@njit
def greedy_afterstates(weights: np.ndarray, layout: TupleLayout, games: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Play greedy games as training does and return every afterstate with the board it was moved from."""
    np.random.seed(seed)
    afterstates, states = [], []
    for _ in range(games):
        board = bb.initial_board()
        while True:
            best, best_value = board, -np.inf
            for action in range(bb.NUM_ACTIONS):
                afterstate, reward = bb.move(board, action)
                if afterstate != board:
                    value = reward + evaluate(weights, layout, afterstate, board)
                    if value > best_value:
                        best, best_value = afterstate, value
            if best == board:
                break
            afterstates.append(best)
            states.append(board)
            board = bb.spawn_tile(best)
    return np.array(afterstates, dtype=np.uint64), np.array(states, dtype=np.uint64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure the effective gain of temporal coherence updates by stage")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--games", type=int, default=20, help="Greedy games to sample boards from (default: 20)")
    parser.add_argument("--samples", type=int, default=2000, help="Boards per group at most (default: 2,000)")
    parser.add_argument("--learning-rate", type=float, default=0.02, help="The base rate (default: 0.02)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    network, metadata = NTupleNetwork.load(args.checkpoint)
    traces = load_traces(args.checkpoint)
    if traces is None:
        raise SystemExit(f"{args.checkpoint} has no temporal coherence sums")
    layout = network.layout
    afterstates, states = greedy_afterstates(network.weights, layout, args.games, args.seed)
    print(f"{args.checkpoint} ({metadata.games_played:,} games): {len(afterstates):,} afterstates from {args.games} greedy games")

    rng = np.random.default_rng(args.seed)
    downgrade = 0 if layout.sharing is None else layout.sharing.downgrade_exponent
    groups: dict[tuple[int, bool], list[int]] = {}
    for k, afterstate in enumerate(afterstates):
        downgraded = downgrade > 0 and bb.downgrade(afterstate, downgrade) != afterstate
        groups.setdefault((bb.max_exponent(afterstate), downgraded), []).append(k)

    nominal = args.learning_rate * len(weight_indices(layout, afterstates[0], states[0]))
    print(f"Nominal gain, every read at the full rate and no repeats: {nominal:.2f}\n")
    print("| Largest tile | Downgraded | Boards | Repeat factor | Mean coherence | Reads below 1% coherence | Gain ΔV/δ: p10 | median | p90 | Median / nominal |")
    print("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for (exponent, downgraded), members in sorted(groups.items()):
        chosen = rng.choice(members, size=min(args.samples, len(members)), replace=False)
        rows, frozen = [], []
        for k in chosen:
            indices = weight_indices(layout, afterstates[k], states[k])
            rows.append(update_gain(indices, traces.errors, traces.abs_errors, args.learning_rate))
            sums = traces.abs_errors[indices]
            frozen.append(np.mean((sums > 0) & (np.abs(traces.errors[indices]) < 0.01 * np.where(sums > 0, sums, 1))))
        gains, repeats, coherence = np.array(rows).T
        p10, median, p90 = np.percentile(gains, [10, 50, 90])
        print(
            f"| {1 << exponent:,} | {'yes' if downgraded else 'no'} | {len(chosen):,} | {repeats.mean():.2f} | {coherence.mean():.3f} | "
            f"{np.mean(frozen):.1%} | {p10:.3f} | {median:.3f} | {p90:.3f} | {median / nominal:.1%} |"
        )


if __name__ == "__main__":
    main()
