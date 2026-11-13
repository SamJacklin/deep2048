import numpy as np

from deep_2048.agents.ntuple import NTupleNetwork
from deep_2048.agents.ntuple.network import CoherenceTraces
from deep_2048.game import bitboard as bb
from scripts.add_stage import add_stage, add_stage_to_traces

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


def test_kept_coherence_sums_are_copied_into_the_second_stage() -> None:
    single = NTupleNetwork(SMALL_TUPLES, downgrade_tile=1024)
    traces = CoherenceTraces(*np.random.default_rng(2).normal(size=(2, single.weights.size)).astype(np.float32))
    staged = add_stage_to_traces(traces, single)
    tables = single.weights.size
    assert staged.errors.size == 2 * tables
    assert (staged.errors[:tables] == traces.errors).all() and (staged.abs_errors[tables:] == traces.abs_errors).all()


def test_a_move_into_the_next_stage_is_valued_in_the_current_one() -> None:
    from deep_2048.agents.ntuple.network import evaluate

    network = NTupleNetwork(SMALL_TUPLES, downgrade_tile=1024, stage_tile=256)
    tables = sum(16 ** len(t) for t in SMALL_TUPLES)
    network.weights[tables:] = 1.0  # only the second stage is non-zero
    state = bb.from_array(np.array([[128, 128, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]))
    afterstate = np.uint64(bb.move(state, 2)[0])  # left: the 128s merge into a 256
    assert bb.max_exponent(afterstate) == 8
    assert evaluate(network.weights, network.layout, afterstate, state) == 0.0  # read in the 128 board's stage
    assert evaluate(network.weights, network.layout, afterstate, afterstate) == len(SMALL_TUPLES) * bb.NUM_SYMMETRIES
