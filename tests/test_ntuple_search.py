import numpy as np

from deep_2048.agents import load_agent
from deep_2048.agents.ntuple import NTupleAgent, NTupleNetwork
from deep_2048.agents.ntuple.models import CheckpointMetadata
from deep_2048.agents.ntuple.search import action_values
from deep_2048.game import bitboard as bb
from deep_2048.game.constants import Action
from deep_2048.game.moves import apply_move

from tests.helpers import SMALL_TUPLES, random_boards

SPAWNS = ((2, 0.9), (4, 0.1))


def random_network(seed: int) -> NTupleNetwork:
    network = NTupleNetwork(tuples=SMALL_TUPLES)
    network.weights[:] = np.random.default_rng(seed).normal(scale=100, size=network.weights.shape)
    return network


def reference_state_value(network: NTupleNetwork, board: np.ndarray, depth: int) -> float:
    """Expectimax on the reference engine, the slow, obvious way."""
    values = [value for action in Action if (value := reference_action_value(network, board, action, depth)) is not None]
    return max(values) if values else 0.0


def reference_action_value(network: NTupleNetwork, board: np.ndarray, action: Action, depth: int) -> float | None:
    result = apply_move(board, action)
    if not result.changed:
        return None
    if depth == 1:
        return result.reward + network.value(bb.from_array(result.board))

    empty = list(zip(*np.nonzero(result.board == 0)))
    expected = 0.0
    for r, c in empty:
        for tile, probability in SPAWNS:
            spawned = result.board.copy()
            spawned[r, c] = tile
            expected += probability / len(empty) * reference_state_value(network, spawned, depth - 1)
    return result.reward + expected


def sparse_boards(n: int, seed: int) -> list[np.ndarray]:
    """Boards with plenty of empty cells, like real play, so search has spawns to average over."""
    rng = np.random.default_rng(seed)
    boards = []
    for board in random_boards(n, seed):
        board[rng.random(board.shape) < 0.6] = 0
        boards.append(board)
    return boards


def test_depth_one_is_reward_plus_afterstate_value() -> None:
    network = random_network(0)
    for board in random_boards(300, seed=1):
        values = action_values(network.weights, network.layout, bb.from_array(board), 1)
        for action in Action:
            expected = reference_action_value(network, board, action, 1)
            assert np.isnan(values[action]) if expected is None else values[action] == expected


def test_depth_two_matches_reference_expectimax() -> None:
    network = random_network(2)
    for board in sparse_boards(20, seed=3):
        values = action_values(network.weights, network.layout, bb.from_array(board), 2)
        for action in Action:
            expected = reference_action_value(network, board, action, 2)
            assert np.isnan(values[action]) if expected is None else np.isclose(values[action], expected, rtol=1e-9)


def test_exact_search_is_unchanged_by_the_transposition_table() -> None:
    network = random_network(5)
    for board in sparse_boards(20, seed=6):
        packed = bb.from_array(board)
        reused = action_values(network.weights, network.layout, packed, 3)
        assert np.array_equal(reused, action_values(network.weights, network.layout, packed, 3, reuse=False), equal_nan=True)


def test_finished_game_has_no_moves() -> None:
    network = random_network(4)
    stuck = np.array([[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]])
    assert np.isnan(action_values(network.weights, network.layout, bb.from_array(stuck), 2)).all()


def test_load_agent_builds_searching_ntuple(tmp_path) -> None:
    path = tmp_path / "net.npz"
    random_network(5).save(path, CheckpointMetadata(games_played=7, moves_played=700, learning_rate=0.01))

    agent = load_agent("ntuple_search", checkpoint=str(path))
    assert isinstance(agent, NTupleAgent) and agent.depth == 2
    assert agent.label == "N-tuple TD (7 games, lr 0.01) + search depth=2"
    assert load_agent("ntuple", checkpoint=str(path)).label == "N-tuple TD (7 games, lr 0.01)"
