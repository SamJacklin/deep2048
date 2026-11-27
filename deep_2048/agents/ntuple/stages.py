"""A late-game second stage built on a trained network: stage 1 plays the game up to a tile, stage 2 from it on.

stage_two() copies stage 1's weights into a second stage for boards whose real largest tile is `stage_tile` or more.
The second stage can downgrade from a later tile than the first (`stage_downgrade_tile`), so it reads real 16384s
where stage 1 read them as 8192s; the tables for those patterns were never used by stage 1, so they start at 0. Every
second-stage weight still at exactly 0 can then be set to an optimistic value, so the late game explores the patterns
stage 1 never learned while keeping everything it did. The first stage is left exactly as it was.

first_boards() plays stage 1's own greedy games and keeps the first board of each that holds `stage_tile`: the pool
that every second-stage training game starts from, so the second stage trains only on the late game and never
touches the first stage's weights (once a game holds the tile, every later board does too).
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from deep_2048.agents.ntuple import points as pt
from deep_2048.agents.ntuple.network import NTupleNetwork


def stage_two(network: NTupleNetwork, stage_tile: int, stage_downgrade_tile: int | None, optimism: float | None) -> tuple[NTupleNetwork, int]:
    """Return (the network with a second stage copied from its first, how many second-stage weights were set optimistic)."""
    if network.stage_tile is not None:
        raise ValueError(f"The network already has a second stage from {network.stage_tile}")
    tables = sum(16 ** len(t) for t in network.tuples)
    weights = np.concatenate([network.weights[:tables], network.weights[:tables], network.weights[tables:]])
    staged = NTupleNetwork(
        network.tuples,
        weights,
        downgrade_tile=network.downgrade_tile,
        scale_feature=network.scale_feature,
        scale_when_downgraded=network.scale_when_downgraded,
        stage_tile=stage_tile,
        stage_downgrade_tile=stage_downgrade_tile,
    )
    if optimism is None:
        return staged, 0
    second = staged.weights[tables : 2 * tables]  # a view: the second stage's tables
    untouched = second == 0
    second[untouched] = optimism
    return staged, int(untouched.sum())


def first_boards(network: NTupleNetwork, stage_tile: int, games: int, seed_offset: int, threads: int) -> tuple[np.ndarray, np.ndarray]:
    """Play `games` seeded greedy games with the network and return (the first board of each holding `stage_tile`, all games' final scores)."""
    exponent = stage_tile.bit_length() - 1
    weights, layout = network.weights, network.layout
    with ThreadPoolExecutor(max_workers=threads) as pool:
        results = list(pool.map(lambda s: pt.point_boards(weights, layout, exponent, exponent, 0, np.random.default_rng(int(s))), range(seed_offset, seed_offset + games)))
    boards = np.array([r[1][0] for r in results if r[4] > 0], dtype=np.uint64)
    return boards, np.array([r[5] for r in results])
