import numpy as np
import pytest

from deep_2048.agents.heuristic import board_heuristic, heuristic_select
from deep_2048.game.constants import Action
from deep_2048.game.moves import get_valid_actions

from tests.helpers import random_boards

# Recorded from the original pure-Python search. The first two boards have exact ties, which go to the earlier action.
RECORDED_CHOICES = [
    (
        [[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        Action.DOWN,
        {"UP": 0.0, "DOWN": 135.4943, "LEFT": 0.0, "RIGHT": 135.4943},
    ),
    (
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [2, 2, 0, 0]],
        Action.LEFT,
        {"UP": 132.6035, "DOWN": 0.0, "LEFT": 133.2507, "RIGHT": 133.2507},
    ),
    (
        [[2, 4, 256, 4], [0, 4, 64, 8], [2, 0, 16, 16], [0, 0, 0, 512]],
        Action.UP,
        {"UP": 67.9443, "DOWN": 60.1804, "LEFT": 42.737, "RIGHT": 64.304},
    ),
    (
        [[64, 0, 2, 0], [128, 0, 0, 0], [32, 0, 2, 0], [4, 4, 8, 0]],
        Action.LEFT,
        {"UP": 92.2293, "DOWN": 92.7631, "LEFT": 95.0207, "RIGHT": 79.1581},
    ),
    (
        [[128, 16, 4, 4], [512, 256, 32, 128], [4, 1024, 64, 2], [2, 8, 4, 2]],
        Action.DOWN,
        {"UP": -13.585, "DOWN": -0.475, "LEFT": -4.26, "RIGHT": -22.71},
    ),
]


def reference_heuristic(board: np.ndarray) -> float:
    """Score a board with plain numpy, reading rows and then columns."""
    exponents = np.log2(np.maximum(board, 1))
    disorder = merges = 0.0
    for lines in (exponents, exponents.T):
        steps = np.diff(lines, axis=1)
        disorder += np.minimum(np.maximum(steps, 0).sum(axis=1), np.maximum(-steps, 0).sum(axis=1)).sum()
        mergeable = (lines[:, :-1] == lines[:, 1:]) & (lines[:, :-1] > 0)
        merges += lines[:, :-1][mergeable].sum()
    corner = max(exponents[0, 0], exponents[0, 3], exponents[3, 0], exponents[3, 3])
    return float(merges - disorder + 10 * np.sum(board == 0) + 2 * corner)


def test_board_heuristic_matches_reference() -> None:
    for board in random_boards(2_000, seed=20):
        assert board_heuristic(board) == reference_heuristic(board), board


@pytest.mark.parametrize(("board", "expected_action", "expected_scores"), RECORDED_CHOICES)
def test_choices_match_recorded_values(board: list[list[int]], expected_action: Action, expected_scores: dict[str, float]) -> None:
    array = np.array(board, dtype=np.int32)
    action, scores = heuristic_select(array, get_valid_actions(array))
    assert action == expected_action
    assert scores == expected_scores


def test_only_the_given_actions_are_scored() -> None:
    board = np.array([[64, 0, 2, 0], [128, 0, 0, 0], [32, 0, 2, 0], [4, 4, 8, 0]], dtype=np.int32)
    action, scores = heuristic_select(board, [Action.UP, Action.RIGHT])
    assert action == Action.UP
    assert scores == {"UP": 92.2293, "DOWN": 0.0, "LEFT": 0.0, "RIGHT": 79.1581}
