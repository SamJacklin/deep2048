"""How much optimism is left in a network trained from optimistic weights, and where in the game it is.

Usage:
    uv run python scripts/optimism_left.py checkpoints/ntuple/methods-8x6-td-opt-1250/before_cut_2600000.npz --value 1250

A weight still exactly at its starting value has never been updated. The network's own greedy games (seeded from
80,000,000) are recorded at every point from the 2048 stage to the 32768 stage, and from each recorded board every
legal move's afterstate is read as the network reads it. The table gives, by stage, the share of those reads that land
on a never-updated weight and the share of moves touching at least one: where those are high, choosing a move still
compares optimistic values with learned ones, so optimism is still steering play there; where they are near zero,
optimism has worn off and training on at the opening rate is plain TD.
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from numba import njit

sys.path.insert(0, str(Path(__file__).parent.parent))

from deep_2048.agents.ntuple import NTupleNetwork  # noqa: E402
from deep_2048.agents.ntuple import points as pt  # noqa: E402
from deep_2048.agents.ntuple.network import TupleLayout, weight_indices  # noqa: E402
from deep_2048.game import bitboard as bb  # noqa: E402

SEED_OFFSET = 80_000_000


@njit(nogil=True)
def untouched_reads(weights: np.ndarray, layout: TupleLayout, board: np.uint64, value: float) -> tuple[int, int, int]:
    """Over the board's legal moves: (reads of never-updated weights, all reads, moves touching at least one)."""
    untouched = reads = moves_touching = 0
    for action in range(bb.NUM_ACTIONS):
        after, _ = bb.move(board, action)
        if after == board:
            continue
        indices = weight_indices(layout, after, board)
        hits = 0
        for i in indices:
            if weights[i] == value:
                hits += 1
        untouched += hits
        reads += indices.size
        moves_touching += hits > 0
    return untouched, reads, moves_touching


def main() -> None:
    parser = argparse.ArgumentParser(description="How much optimism is left in a network, by stage of the game")
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--value", type=float, required=True, help="the value every weight started at (the run's --optimism)")
    parser.add_argument("--games", type=int, default=300, help="greedy games to record boards from (default 300)")
    parser.add_argument("--threads", type=int, default=os.cpu_count())
    args = parser.parse_args()
    value = np.float32(args.value)
    for checkpoint in args.checkpoints:
        network, _ = NTupleNetwork.load(checkpoint)
        weights, layout = network.weights, network.layout
        never = float((weights == value).mean())
        with ThreadPoolExecutor(max_workers=args.threads) as pool:
            games = list(pool.map(lambda s: pt.point_boards(weights, layout, 11, 15, 3, np.random.default_rng(int(s))), range(SEED_OFFSET, SEED_OFFSET + args.games)))
        by_stage: dict[int, list[int]] = {}
        moves_by_stage: dict[int, list[int]] = {}
        for keys, boards, _, _, n, _, _ in games:
            for k in range(n):
                stage = int(keys[k]) // pt.KEY_STRIDE
                untouched, reads, touching = untouched_reads(weights, layout, np.uint64(boards[k]), value)
                totals = by_stage.setdefault(stage, [0, 0, 0, 0])
                totals[0] += untouched
                totals[1] += reads
                totals[2] += touching
                totals[3] += 1
        finals = np.array([g[5] for g in games])
        print(f"\n{checkpoint}: {never:.1%} of all {weights.size:,} weights never updated; greedy mean over {args.games} games {finals.mean():,.0f}")
        print(f"{'stage':>7} {'boards':>7} {'reads of never-updated weights':>31} {'moves touching one, per board':>30}")
        for stage in sorted(by_stage):
            untouched, reads, touching, boards = by_stage[stage]
            print(f"{1 << stage:>7} {boards:>7} {untouched / reads:>31.2%} {touching / boards:>30.2f}")

if __name__ == "__main__":
    main()
