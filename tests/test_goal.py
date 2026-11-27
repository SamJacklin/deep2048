"""A tailored first stage: training games end on the move that makes the goal tile, worth a fixed bonus."""

import numpy as np
import pytest

from deep_2048.agents.ntuple import NTupleNetwork
from deep_2048.agents.ntuple.config import NTupleSettings
from deep_2048.agents.ntuple.models import Goal
from deep_2048.agents.ntuple.network import evaluate
from deep_2048.agents.ntuple.train import _best_move_goal, play_games
from deep_2048.game import bitboard as bb

from tests.helpers import SMALL_TUPLES


def board(rows: list[list[int]]) -> np.uint64:
    return np.uint64(bb.from_array(np.array(rows)))


def test_every_game_ends_on_the_move_that_makes_the_goal() -> None:
    network = NTupleNetwork(tuples=SMALL_TUPLES)
    bb.seed(1)
    results = play_games(200, network.weights, None, network.layout, 0.01, True, None, None, None, None, Goal(7, 1000.0))
    reached = results.max_exponents >= 7
    assert reached.mean() > 0.9  # a 128 is easy, so nearly every game makes it
    assert (results.max_exponents[reached] == 7).all()  # and stops there: none goes on to 256
    # Games short of the goal are ordinary game overs
    assert (results.max_exponents[~reached] < 7).all()


def test_the_goal_move_is_valued_at_the_bonus_exactly() -> None:
    network = NTupleNetwork(tuples=SMALL_TUPLES)  # every weight 0: only the bonus can make a move look good
    start = board([[64, 64, 0, 0], [2, 4, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    afterstate, reward, value = _best_move_goal(network.weights, network.layout, start, Goal(7, 5000.0))
    assert bb.max_exponent(afterstate) == 7
    assert reward == 128 and value == 128 + 5000.0


def test_the_goal_afterstates_learn_the_bonus() -> None:
    """Afterstates holding the goal tile learn towards the bonus: a bigger bonus, a higher value. (Not exactly the bonus,
    since tuples that miss the goal tile's cell share their weights with boards still short of it.)"""
    goal_board = board([[128, 8, 4, 2], [16, 4, 2, 0], [2, 0, 0, 0], [0, 0, 0, 0]])
    values = []
    for bonus in (0.0, 5000.0):
        network = NTupleNetwork(tuples=SMALL_TUPLES)
        bb.seed(2)
        for _ in range(10):
            play_games(200, network.weights, None, network.layout, 0.01, True, None, None, None, None, Goal(7, bonus))
        values.append(evaluate(network.weights, network.layout, goal_board, goal_board))
    assert values[1] - values[0] > 2500


def test_without_a_goal_games_play_on() -> None:
    network = NTupleNetwork(tuples=SMALL_TUPLES)
    bb.seed(3)
    play_games(300, network.weights, None, network.layout, 0.01, True)
    results = play_games(100, network.weights, None, network.layout, 0.0, False)
    assert (results.max_exponents > 7).any()


@pytest.mark.parametrize(
    "settings",
    [
        {"GOAL_TILE": 100},
        {"GOAL_TILE": 16384, "LATE_START_FRACTION": 0.2},
        {"GOAL_TILE": 16384, "DOWNGRADE_TILE": 16384},
        {"GOAL_TILE": 16384, "LAMBDA": 0.5},
        {"GOAL_TILE": 16384, "STAGE_TILE": 16384},
    ],
)
def test_a_goal_rejects_what_it_cannot_work_with(settings: dict) -> None:
    with pytest.raises(ValueError):
        NTupleSettings(**settings)


def test_a_goal_allows_downgrading_above_it() -> None:
    assert NTupleSettings(GOAL_TILE=16384, DOWNGRADE_TILE=32768).GOAL_TILE == 16384
