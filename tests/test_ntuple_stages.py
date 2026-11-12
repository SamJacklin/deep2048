import numpy as np

from deep_2048.agents.ntuple import NTupleNetwork
from deep_2048.game import bitboard as bb
from scripts.add_stage import add_stage

from tests.helpers import SMALL_TUPLES, random_boards


def test_boards_with_the_stage_tile_read_the_second_stage() -> None:
    network = NTupleNetwork(SMALL_TUPLES, downgrade_tile=1024, stage_tile=256)
    tables = sum(16 ** len(t) for t in SMALL_TUPLES)
    assert network.weights.size == 2 * tables
    network.weights[tables:] = 1.0  # only the second stage is non-zero

    small = bb.from_array(np.array([[128, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]))
    large = bb.from_array(np.array([[256, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]))
    assert network.value(small) == 0.0
    assert network.value(large) == len(SMALL_TUPLES) * bb.NUM_SYMMETRIES

    # Learning on a late board only touches the second stage
    network.weights[:] = 0.0
    network.update(large, 1.0)
    assert not network.weights[:tables].any() and network.weights[tables:].any()


def test_an_added_stage_starts_as_an_exact_copy() -> None:
    single = NTupleNetwork(SMALL_TUPLES, downgrade_tile=1024)
    single.weights[:] = np.random.default_rng(0).normal(size=single.weights.size)
    staged = add_stage(single, 256)
    for board in random_boards(200, seed=1):
        packed = bb.from_array(board)
        assert staged.value(packed) == single.value(packed)
