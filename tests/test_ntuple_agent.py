import numpy as np
import pytest

from deep_2048.agents import load_agent
from deep_2048.agents.ntuple import NTupleAgent, NTupleNetwork
from deep_2048.agents.ntuple import bitboard as bb
from deep_2048.agents.ntuple.models import CheckpointMetadata
from deep_2048.agents.ntuple.train import play_games
from deep_2048.bench import run_games
from deep_2048.game.game import Game2048
from deep_2048.game.moves import apply_move

from tests.helpers import SMALL_TUPLES


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> str:
    network = NTupleNetwork(tuples=SMALL_TUPLES)
    bb.seed(0)
    play_games(500, network.weights, network.layout, 0.01, True)
    path = tmp_path_factory.mktemp("ntuple") / "net.npz"
    network.save(path, CheckpointMetadata(games_played=500, learning_rate=0.01))
    return str(path)


def test_load_agent_builds_ntuple(checkpoint: str) -> None:
    agent = load_agent("ntuple", checkpoint=checkpoint)
    assert isinstance(agent, NTupleAgent)
    assert agent.label == "N-tuple TD (500 games)"


def test_picks_best_reward_plus_afterstate_value(checkpoint: str) -> None:
    agent = NTupleAgent(checkpoint)
    for seed in range(5):
        game = Game2048(seed=seed)
        for _ in range(50):
            if game.is_game_over():
                break
            board, valid_actions = game.get_board(), game.get_valid_actions()
            decision = agent.act(board, valid_actions)

            expected = {}
            for action in valid_actions:
                result = apply_move(board, action)
                expected[action.name] = result.reward + agent.network.value(bb.from_array(result.board))

            assert decision.scores.keys() == expected.keys()
            assert all(np.isclose(decision.scores[name], value) for name, value in expected.items())
            assert decision.scores[decision.action.name] == max(decision.scores.values())
            game.step(decision.action)


def test_seeded_games_are_reproducible(checkpoint: str) -> None:
    first = run_games(NTupleAgent(checkpoint), n=3)
    second = run_games(NTupleAgent(checkpoint), n=3)
    assert first.scores == second.scores
    assert all(score > 0 for score in first.scores)
