"""The large-tile count (scale) term's own rate and its skipped empty entry."""

import numpy as np
import pytest

from deep_2048.agents.ntuple import NTupleNetwork
from deep_2048.agents.ntuple.models import CheckpointMetadata
from deep_2048.agents.ntuple.network import CoherenceTraces, update, update_coherent, weight_indices
from deep_2048.game import bitboard as bb

from tests.helpers import SMALL_TUPLES

SMALL_BOARD = np.uint64(bb.from_array(np.array([[2, 4, 8, 16], [0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])))
LARGE_BOARD = np.uint64(bb.from_array(np.array([[2048, 4, 8, 16], [0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])))


def scale_index(network: NTupleNetwork, board: np.uint64) -> int | None:
    indices = weight_indices(network.layout, board, board)
    extra = indices[network.n_features :]
    return int(extra[0]) if extra.size else None


def test_the_empty_entry_is_skipped_only_when_asked() -> None:
    plain = NTupleNetwork(SMALL_TUPLES, scale_feature=True)
    skipping = NTupleNetwork(SMALL_TUPLES, scale_feature=True, scale_skip_empty=True)
    assert scale_index(plain, SMALL_BOARD) is not None
    assert scale_index(skipping, SMALL_BOARD) is None
    assert scale_index(skipping, LARGE_BOARD) == scale_index(plain, LARGE_BOARD)


def test_update_scales_only_the_scale_weight() -> None:
    network = NTupleNetwork(SMALL_TUPLES, scale_feature=True, scale_skip_empty=True, scale_rate=0.25)
    update(network.weights, network.layout, LARGE_BOARD, LARGE_BOARD, 4.0)
    index = scale_index(network, LARGE_BOARD)
    assert network.weights[index] == pytest.approx(1.0)
    tuple_weights = network.weights[weight_indices(network.layout, LARGE_BOARD, LARGE_BOARD)[: network.n_features]]
    assert np.all(tuple_weights > 0) and np.all(tuple_weights % 4 == 0)  # each tuple read took the full step (repeats add up)


def test_coherent_update_scales_only_the_scale_weight() -> None:
    network = NTupleNetwork(SMALL_TUPLES, scale_feature=True, scale_skip_empty=True, scale_rate=1 / 64)
    traces = CoherenceTraces.zeros(network.weights.size)
    update_coherent(network.weights, traces, network.layout, LARGE_BOARD, LARGE_BOARD, 0.5, 128.0)
    assert network.weights[scale_index(network, LARGE_BOARD)] == pytest.approx(0.5 / 64 * 128.0)
    first_tuple = weight_indices(network.layout, LARGE_BOARD, LARGE_BOARD)[0]
    assert network.weights[first_tuple] >= 0.5 * 128.0


def test_a_scale_table_at_zero_leaves_values_unchanged() -> None:
    base = NTupleNetwork(SMALL_TUPLES, downgrade_tile=2048)
    base.weights[:] = np.random.default_rng(0).normal(size=base.weights.size).astype(np.float32)
    extended = NTupleNetwork(SMALL_TUPLES, np.concatenate([base.weights, np.zeros(extended_size(base) - base.weights.size, dtype=np.float32)]),
                             downgrade_tile=2048, scale_feature=True, scale_skip_empty=True, scale_rate=1 / 64)
    for board in (SMALL_BOARD, LARGE_BOARD):
        assert extended.value(board) == pytest.approx(base.value(board), rel=1e-6)


def extended_size(base: NTupleNetwork) -> int:
    return NTupleNetwork(base.tuples, downgrade_tile=base.downgrade_tile, scale_feature=True).weights.size


def test_the_options_survive_a_save_and_old_checkpoints_load_as_before(tmp_path) -> None:
    network = NTupleNetwork(SMALL_TUPLES, scale_feature=True, scale_skip_empty=True, scale_rate=1 / 64)
    network.save(tmp_path / "scaled.npz", CheckpointMetadata(games_played=0, moves_played=0, learning_rate=0.1))
    loaded, _ = NTupleNetwork.load(tmp_path / "scaled.npz")
    assert loaded.scale_rate == pytest.approx(1 / 64) and loaded.scale_skip_empty
    assert scale_index(loaded, SMALL_BOARD) is None
    plain = NTupleNetwork(SMALL_TUPLES, scale_feature=True)
    assert "scale_rate" not in plain.features and "scale_skip_empty" not in plain.features


def test_the_options_need_the_scale_feature() -> None:
    with pytest.raises(ValueError):
        NTupleNetwork(SMALL_TUPLES, scale_rate=0.5)
    with pytest.raises(ValueError):
        NTupleNetwork(SMALL_TUPLES, scale_skip_empty=True)


def test_boards_at_or_above_the_cut_off_read_no_scale_weight() -> None:
    network = NTupleNetwork(SMALL_TUPLES, downgrade_tile=16384, scale_feature=True, scale_skip_empty=True, scale_below_tile=16384)
    two_8192s = np.uint64(bb.from_array(np.array([[8192, 8192, 4, 2], [2048, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])))
    merged = np.uint64(bb.from_array(np.array([[16384, 4, 2, 0], [2048, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])))
    assert scale_index(network, two_8192s) is not None
    assert scale_index(network, merged) is None
    with pytest.raises(ValueError):
        NTupleNetwork(SMALL_TUPLES, scale_feature=True, scale_below_tile=10000)
