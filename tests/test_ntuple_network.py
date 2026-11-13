import json

import numpy as np

from deep_2048.agents.ntuple.models import CheckpointMetadata
from deep_2048.agents.ntuple.network import NETWORKS, SCALE_WEIGHTS, TUPLES, CoherenceTraces, NTupleNetwork, load_traces, network_name, update_coherent
from deep_2048.game import bitboard as bb

from tests.helpers import SMALL_TUPLES, random_boards, symmetry_grids


def random_network(seed: int) -> NTupleNetwork:
    network = NTupleNetwork(tuples=SMALL_TUPLES)
    network.weights[:] = np.random.default_rng(seed).normal(size=network.weights.shape)
    return network


def reference_indices(network: NTupleNetwork, board: np.ndarray) -> list[int]:
    """List the weights the network reads for a board, in read order, the slow, obvious way."""
    exponents = np.where(board == 0, 0, np.log2(np.maximum(board, 1))).astype(int).reshape(-1)
    indices = []
    offset = 0
    for cells in network.tuples:
        for symmetry in symmetry_grids():
            transformed = exponents[symmetry.reshape(-1)]
            indices.append(offset + sum(int(transformed[c]) << (4 * k) for k, c in enumerate(cells)))
        offset += 16 ** len(cells)
    return indices


def reference_value(network: NTupleNetwork, board: np.ndarray) -> float:
    return sum(float(network.weights[i]) for i in reference_indices(network, board))


def test_full_network_size() -> None:
    network = NTupleNetwork()
    assert network.tuples == TUPLES
    assert network.weights.size == 2 * 16**4 + 2 * 16**6
    assert network.n_features == 32


def test_named_networks_read_the_reference_weights() -> None:
    for name, tuples in NETWORKS.items():
        network = NTupleNetwork(tuples=tuples)
        assert network_name(network.tuples) == name
        for board in random_boards(20, seed=10):
            network.update(bb.from_array(board), 1.0)
            indices = reference_indices(network, board)
            touched = np.flatnonzero(network.weights)
            np.testing.assert_array_equal(touched, np.unique(indices))
            np.testing.assert_array_equal(network.weights[touched], np.bincount(indices)[touched])
            network.weights[touched] = 0
    assert network_name(SMALL_TUPLES) is None


def test_symmetries_match_numpy_rotations() -> None:
    grids = symmetry_grids()
    assert len({g.tobytes() for g in grids}) == 8
    for board in random_boards(500, seed=8):
        symmetric_boards = bb.symmetries(bb.from_array(board))
        for symmetric_board, grid in zip(symmetric_boards, grids):
            expected = board.reshape(-1)[grid.reshape(-1)].reshape(4, 4)
            np.testing.assert_array_equal(bb.to_array(np.uint64(symmetric_board)), expected)


def test_value_matches_reference() -> None:
    network = random_network(0)
    for board in random_boards(500, seed=1):
        assert np.isclose(network.value(bb.from_array(board)), reference_value(network, board))


def test_value_is_symmetric() -> None:
    network = random_network(2)
    for board in random_boards(200, seed=3):
        expected = network.value(bb.from_array(board))
        for symmetry in symmetry_grids():
            transformed = board.reshape(-1)[symmetry.reshape(-1)].reshape(4, 4)
            assert np.isclose(network.value(bb.from_array(transformed)), expected)


def test_update() -> None:
    network = NTupleNetwork(tuples=SMALL_TUPLES)
    board = bb.from_array(random_boards(1, seed=4)[0])

    network.update(board, 1.0)
    # A weight read by several pairs gets +1 from each, so the value is a sum of squared hit counts
    hit = network.weights[network.weights != 0]
    assert hit.sum() == network.n_features
    assert network.value(board) == float((hit**2).sum())


def test_coherent_update_matches_reference() -> None:
    network = random_network(5)
    traces = CoherenceTraces.zeros(network.weights.size)
    weights, errors, abs_errors = network.weights.astype(np.float64), np.zeros(network.weights.size), np.zeros(network.weights.size)
    rng = np.random.default_rng(6)

    for board in random_boards(100, seed=6):
        error = float(rng.normal())
        update_coherent(network.weights, traces, network.layout, bb.from_array(board), bb.from_array(board), 0.1, error)
        for i in reference_indices(network, board):
            coherence = abs(errors[i]) / abs_errors[i] if abs_errors[i] > 0 else 1.0
            weights[i] += 0.1 * coherence * error
            errors[i] += error
            abs_errors[i] += abs(error)

    np.testing.assert_allclose(network.weights, weights, atol=1e-4)
    np.testing.assert_allclose(traces.errors, errors, atol=1e-4)
    np.testing.assert_allclose(traces.abs_errors, abs_errors, atol=1e-4)


def test_save_and_load_round_trip(tmp_path) -> None:
    network = random_network(7)
    path = tmp_path / "net.npz"
    metadata = CheckpointMetadata(games_played=12, moves_played=3_456, learning_rate=0.01)
    network.save(path, metadata)

    loaded, loaded_metadata = NTupleNetwork.load(path)
    assert loaded.tuples == SMALL_TUPLES
    np.testing.assert_array_equal(loaded.weights, network.weights)
    assert loaded_metadata == metadata
    assert load_traces(path) is None
    assert not (tmp_path / "net.npz.tmp").exists()


def test_scale_sharing_features_match_reference() -> None:
    network = NTupleNetwork(tuples=SMALL_TUPLES, downgrade_tile=1024, scale_feature=True)
    network.weights[:] = np.random.default_rng(11).normal(size=network.weights.shape)
    scale_offset = network.weights.size - SCALE_WEIGHTS
    checked = 0
    for board in random_boards(500, seed=12):
        packed = bb.from_array(board)
        read = bb.to_array(np.uint64(bb.downgrade(packed, 10)))
        checked += not np.array_equal(read, board)
        expected = reference_value(network, read) + float(network.weights[scale_offset + bb.large_tile_counts(packed)])
        assert np.isclose(network.value(packed), expected)
    assert checked > 100


def test_scale_term_when_downgraded_only_reads_downgraded_boards() -> None:
    network = NTupleNetwork(tuples=SMALL_TUPLES, downgrade_tile=1024, scale_feature=True, scale_when_downgraded=True)
    network.weights[:] = np.random.default_rng(14).normal(size=network.weights.shape)
    scale_offset = network.weights.size - SCALE_WEIGHTS
    downgraded = 0
    for board in random_boards(500, seed=15):
        packed = bb.from_array(board)
        read = bb.to_array(np.uint64(bb.downgrade(packed, 10)))
        expected = reference_value(network, read)
        if not np.array_equal(read, board):
            expected += float(network.weights[scale_offset + bb.large_tile_counts(packed)])
            downgraded += 1
        assert np.isclose(network.value(packed), expected)
    assert 100 < downgraded < 500


def test_scale_sharing_features_survive_save_and_load(tmp_path) -> None:
    network = NTupleNetwork(tuples=SMALL_TUPLES, downgrade_tile=1024, scale_feature=True, scale_when_downgraded=True)
    network.save(tmp_path / "net.npz", CheckpointMetadata(games_played=1, moves_played=1, learning_rate=0.01))
    loaded, _ = NTupleNetwork.load(tmp_path / "net.npz")
    assert loaded.features == network.features
    assert loaded.layout.sharing == network.layout.sharing
    assert loaded.layout.sharing is not None and loaded.layout.sharing.downgrade_exponent == 10


def test_checkpoints_from_before_scale_sharing_still_load(tmp_path) -> None:
    network = random_network(13)
    metadata = CheckpointMetadata(games_played=1, moves_played=1, learning_rate=0.01)
    np.savez(tmp_path / "old.npz", weights=network.weights, tuples=json.dumps(network.tuples), metadata=metadata.model_dump_json())
    loaded, _ = NTupleNetwork.load(tmp_path / "old.npz")
    assert loaded.downgrade_tile is None and not loaded.scale_feature
    np.testing.assert_array_equal(loaded.weights, network.weights)


def test_save_and_load_coherence_traces(tmp_path) -> None:
    network = random_network(8)
    traces = CoherenceTraces(*np.random.default_rng(9).normal(size=(2, network.weights.size)).astype(np.float32))
    path = tmp_path / "net.npz"
    network.save(path, CheckpointMetadata(games_played=1, moves_played=1, learning_rate=0.01, temporal_coherence=True), traces)

    loaded = load_traces(path)
    assert loaded is not None
    np.testing.assert_array_equal(loaded.errors, traces.errors)
    np.testing.assert_array_equal(loaded.abs_errors, traces.abs_errors)


def test_coherence_sums_decay_before_each_update() -> None:
    from deep_2048.agents.ntuple.network import _update_coherent_weight

    def sums(decay: float, errors: list[float]) -> tuple[float, float]:
        weights = np.zeros(1, dtype=np.float32)
        traces = CoherenceTraces(np.zeros(1, dtype=np.float32), np.zeros(1, dtype=np.float32), decay)
        for error in errors:
            _update_coherent_weight(weights, traces, 0, 0.1, error)
        return float(traces.errors[0]), float(traces.abs_errors[0])

    assert sums(1.0, [1.0, -1.0]) == (0.0, 2.0)  # no decay: the errors cancel for good
    assert sums(0.5, [1.0, -1.0]) == (-0.5, 1.5)
    assert sums(0.5, [1.0, 1.0, 1.0]) == (1.75, 1.75)


def test_coherence_sums_keep_registering_small_errors_once_large() -> None:
    from deep_2048.agents.ntuple.network import _update_coherent_weight

    weights = np.zeros(1, dtype=np.float32)
    traces = CoherenceTraces.from_arrays(np.zeros(1), np.full(1, 3e11))  # as large as the long run's biggest sum
    assert traces.abs_errors.dtype == np.float64
    _update_coherent_weight(weights, traces, 0, 0.0, 111.0)  # the long run's median error size
    assert traces.abs_errors[0] == 3e11 + 111.0
    assert np.float32(3e11) + np.float32(111.0) == np.float32(3e11)  # what float32 sums lost
