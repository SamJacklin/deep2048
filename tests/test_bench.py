import numpy as np

from deep_2048.agents.base import Agent, Decision
from deep_2048.bench import ResultCache, RunKey, run_games
from deep_2048.game.constants import Action


class FirstValidMove(Agent):
    label = "First valid move"

    def act(self, board: np.ndarray, valid_actions: list[Action]) -> Decision:
        return Decision(action=valid_actions[0])


def test_parallel_games_match_serial() -> None:
    agent = FirstValidMove()
    assert run_games(agent, n=6, workers=3) == run_games(agent, n=6, workers=1)


def test_cache_returns_stored_result(tmp_path) -> None:
    cache = ResultCache(tmp_path)
    key = RunKey.for_agent(FirstValidMove(), n_games=3)
    assert cache.get(key) is None

    result = run_games(FirstValidMove(), n=3)
    cache.put(key, result)
    assert cache.get(key) == result


def test_cache_key_follows_checkpoint_contents(tmp_path) -> None:
    agent = FirstValidMove()
    agent.checkpoint = tmp_path / "weights.bin"

    agent.checkpoint.write_bytes(b"before")
    before = RunKey.for_agent(agent, n_games=3)
    agent.checkpoint.write_bytes(b"after")
    after = RunKey.for_agent(agent, n_games=3)

    assert before.digest() != after.digest()


def test_unreadable_cache_entry_is_a_miss(tmp_path) -> None:
    cache = ResultCache(tmp_path)
    key = RunKey.for_agent(FirstValidMove(), n_games=3)
    cache.put(key, run_games(FirstValidMove(), n=3))

    next(tmp_path.glob("*.json")).write_text("{not json")
    assert cache.get(key) is None
