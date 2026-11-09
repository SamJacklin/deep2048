"""Expectimax search over bitboards, scoring the leaves with the hand-crafted evaluator."""

import numpy as np
from numba import njit

from deep_2048.agents.heuristic.evaluation import evaluate
from deep_2048.game import bitboard as bb
from deep_2048.game.constants import Action

_SPAWN_EXPONENTS = (np.uint64(1), np.uint64(2))
_SPAWN_PROBABILITIES = (1.0 - bb.SPAWN_FOUR_PROBABILITY, bb.SPAWN_FOUR_PROBABILITY)


@njit
def _player_node(board: np.uint64, moves_left: int) -> float:
    """Return the best move's expected value, or the board's score at the horizon or game over."""
    if moves_left == 0:
        return evaluate(board)
    best = -np.inf
    for action in range(bb.NUM_ACTIONS):
        moved, _ = bb.move(board, action)
        if moved != board:
            best = max(best, _chance_node(moved, moves_left - 1))
    return best if best > -np.inf else evaluate(board)


# Inlined so its mutual recursion with _player_node becomes self-recursion, which numba can compile
@njit(inline="always")
def _chance_node(board: np.uint64, moves_left: int) -> float:
    """Return the expected value over every tile that could spawn."""
    n_empty = bb.count_empty(board)
    if n_empty == 0:
        return _player_node(board, moves_left)

    cell_probability = 1.0 / n_empty
    expected = 0.0
    for i in range(bb.NUM_CELLS):
        if bb.get_cell(board, i) != 0:
            continue
        for exponent, spawn_probability in zip(_SPAWN_EXPONENTS, _SPAWN_PROBABILITIES):
            spawned = board | (exponent << np.uint64(4 * i))
            expected += cell_probability * spawn_probability * _player_node(spawned, moves_left)
    return expected


@njit
def _action_values(board: np.uint64, depth: int) -> np.ndarray:
    """Return each action's expected value, or NaN if it doesn't change the board."""
    values = np.full(bb.NUM_ACTIONS, np.nan)
    for action in range(bb.NUM_ACTIONS):
        moved, _ = bb.move(board, action)
        if moved != board:
            values[action] = _chance_node(moved, depth - 1)
    return values


def heuristic_select(board: np.ndarray, valid_actions: list[Action], depth: int = 2) -> tuple[Action, dict[str, float]]:
    """Choose the action with the best expected score after `depth` player moves.

    Returns (action, scores), where scores maps every action name to its expected score rounded
    to 4dp, or 0.0 for invalid actions. Ties go to the earliest action in Action order.
    """
    values = _action_values(bb.from_array(board), depth)
    scores = {action.name: 0.0 for action in Action}
    for action in valid_actions:
        if not np.isnan(values[action]):
            scores[action.name] = float(np.round(values[action], 4))

    chosen = max(sorted(valid_actions), key=lambda action: scores[action.name])
    return chosen, scores
