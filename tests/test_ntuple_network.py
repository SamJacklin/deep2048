import numpy as np

from deep_2048.agents.ntuple.models import CheckpointMetadata
from deep_2048.agents.ntuple.network import TUPLES, NTupleNetwork
from deep_2048.game import bitboard as bb

from tests.helpers import SMALL_TUPLES, random_boards, symmetry_grids


def random_network(seed: int) -> NTupleNetwork:
    network = NTupleNetwork(tuples=SMALL_TUPLES)
    network.weights[:] = np.random.default_rng(seed).normal(size=network.weights.shape)
    return network


def reference_value(network: NTupleNetwork, board: np.ndarray) -> float:
    """Compute the network value the slow, obvious way."""
    exponents = np.where(board == 0, 0, np.log2(np.maximum(board, 1))).astype(int).reshape(-1)
    total = 0.0
    offset = 0
    for cells in network.tuples:
        for symmetry in symmetry_grids():
            transformed = exponents[symmetry.reshape(-1)]
            index = sum(int(transformed[c]) << (4 * k) for k, c in enumerate(cells))
            total += float(network.weights[offset + index])
        offset += 16 ** len(cells)
    return total


def test_full_network_size() -> None:
    network = NTupleNetwork()
    assert network.tuples == TUPLES
    assert network.weights.size == 2 * 16**4 + 2 * 16**6
    assert network.n_features == 32


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


def test_save_and_load_round_trip(tmp_path) -> None:
    network = random_network(7)
    path = tmp_path / "net.npz"
    metadata = CheckpointMetadata(games_played=12, learning_rate=0.01)
    network.save(path, metadata)

    loaded, loaded_metadata = NTupleNetwork.load(path)
    assert loaded.tuples == SMALL_TUPLES
    np.testing.assert_array_equal(loaded.weights, network.weights)
    assert loaded_metadata == metadata
    assert not (tmp_path / "net.npz.tmp").exists()
