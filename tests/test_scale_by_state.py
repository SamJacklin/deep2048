"""The large-tile count term read from the board a move is made from (scale_by_state) instead of the afterstate."""

import numpy as np
import pytest

from deep_2048.agents.ntuple import NTupleNetwork
from deep_2048.agents.ntuple.models import CheckpointMetadata
from deep_2048.agents.ntuple.network import CoherenceTraces, evaluate, weight_indices
from deep_2048.agents.ntuple.train import play_game
from deep_2048.game import bitboard as bb

from tests.helpers import SMALL_TUPLES

# One move merges the two 8192s into a 16384, the others don't: the case where an afterstate-read table puts a cliff
BEFORE_MERGE = np.uint64(bb.from_array(np.array([[8192, 8192, 4, 2], [2048, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])))


def network(by_state: bool) -> NTupleNetwork:
    net = NTupleNetwork(SMALL_TUPLES, downgrade_tile=16384, scale_feature=True, scale_skip_empty=True, scale_rate=1 / 64, scale_by_state=by_state)
    net.weights[net.weights.size - 1024 :] = np.arange(1024, dtype=np.float32) * 1000  # every table entry different and large
    return net


def afterstates(board: np.uint64) -> list[np.uint64]:
    return [np.uint64(after) for after, _ in (bb.move(board, a) for a in range(bb.NUM_ACTIONS)) if after != board]


def scale_index(net: NTupleNetwork, board: np.uint64, state: np.uint64) -> int | None:
    extra = weight_indices(net.layout, board, state)[net.n_features :]
    return int(extra[0]) if extra.size else None


def test_by_state_every_move_of_a_decision_reads_the_same_entry() -> None:
    net = network(True)
    moves = afterstates(BEFORE_MERGE)
    assert any(bb.max_exponent(a) == 14 for a in moves) and any(bb.max_exponent(a) == 13 for a in moves)
    entries = {scale_index(net, a, BEFORE_MERGE) for a in moves}
    assert len(entries) == 1 and entries != {None}
    # So the table adds the same amount to every move's value: values with and without it differ by one constant
    bare = NTupleNetwork(SMALL_TUPLES, net.weights.copy(), downgrade_tile=16384, scale_feature=True, scale_skip_empty=True, scale_rate=1 / 64, scale_by_state=True)
    bare.weights[bare.weights.size - 1024 :] = 0
    shifts = {round(net.value(a, BEFORE_MERGE) - bare.value(a, BEFORE_MERGE), 3) for a in moves}
    assert len(shifts) == 1


def test_off_it_reads_the_afterstate_as_before() -> None:
    net = network(False)
    moves = afterstates(BEFORE_MERGE)
    assert len({scale_index(net, a, BEFORE_MERGE) for a in moves}) > 1  # the merge reads another entry
    for a in moves:
        assert scale_index(net, a, BEFORE_MERGE) == scale_index(net, a, a)


def test_the_cut_off_applies_to_the_board_the_move_is_made_from() -> None:
    net = NTupleNetwork(SMALL_TUPLES, downgrade_tile=16384, scale_feature=True, scale_skip_empty=True, scale_below_tile=16384, scale_by_state=True)
    merge = next(a for a in afterstates(BEFORE_MERGE) if bb.max_exponent(a) == 14)
    assert scale_index(net, merge, BEFORE_MERGE) is not None  # made from a board short of 16384: still on the table
    assert scale_index(net, merge, merge) is None


def test_it_needs_the_scale_feature_and_survives_a_save(tmp_path) -> None:
    with pytest.raises(ValueError):
        NTupleNetwork(SMALL_TUPLES, scale_by_state=True)
    assert "scale_by_state" not in NTupleNetwork(SMALL_TUPLES, scale_feature=True).features
    net = network(True)
    net.save(tmp_path / "n.npz", CheckpointMetadata(games_played=0, moves_played=0, learning_rate=0.1))
    loaded, _ = NTupleNetwork.load(tmp_path / "n.npz")
    assert loaded.scale_by_state and loaded.features == net.features
    assert evaluate(loaded.weights, loaded.layout, BEFORE_MERGE, BEFORE_MERGE) == pytest.approx(net.value(BEFORE_MERGE))


def test_a_training_game_runs() -> None:
    net = NTupleNetwork(SMALL_TUPLES, downgrade_tile=2048, scale_feature=True, scale_rate=1 / 64, scale_by_state=True)  # small games: no 2048 to skip for
    traces = CoherenceTraces.zeros(net.weights.size)
    bb.seed(3)
    for _ in range(20):
        play_game(net.weights, traces, net.layout, 0.1, True, np.uint64(0), 0)
    assert np.any(net.weights[net.weights.size - 1024 :] != 0)  # the table learned
