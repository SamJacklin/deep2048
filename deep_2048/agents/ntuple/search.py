"""Expectimax search over bitboards, with the n-tuple network valuing afterstates at the horizon.

A move is worth its reward plus what follows. Below the horizon that's the expected value over
every tile that could spawn. At the horizon it's the network's value of the afterstate, which
is exactly what TD training taught it to predict. Depth 1 is the greedy agent.
"""

import numpy as np
from numba import njit

from deep_2048.agents.ntuple.network import TupleLayout, evaluate
from deep_2048.game import bitboard as bb

_SPAWN_EXPONENTS = (np.uint64(1), np.uint64(2))
_SPAWN_PROBABILITIES = (1.0 - bb.SPAWN_FOUR_PROBABILITY, bb.SPAWN_FOUR_PROBABILITY)


@njit
def _state_value(weights: np.ndarray, layout: TupleLayout, board: np.uint64, depth: int) -> float:
    """Return the best move's value from a state, searching `depth` moves ahead, or 0 if the game is over."""
    best = -np.inf
    for action in range(bb.NUM_ACTIONS):
        afterstate, reward = bb.move(board, action)
        if afterstate != board:
            best = max(best, reward + _afterstate_value(weights, layout, afterstate, depth - 1))
    return best if best > -np.inf else 0.0


# Inlined so its mutual recursion with _state_value becomes self-recursion, which numba can compile
@njit(inline="always")
def _afterstate_value(weights: np.ndarray, layout: TupleLayout, afterstate: np.uint64, depth: int) -> float:
    """Return the network's value at the horizon, otherwise the expected value over every tile that could spawn."""
    if depth == 0:
        return evaluate(weights, layout, afterstate)

    n_empty = bb.count_empty(afterstate)
    cell_probability = 1.0 / n_empty
    expected = 0.0
    for i in range(bb.NUM_CELLS):
        if bb.get_cell(afterstate, i) != 0:
            continue
        for exponent, spawn_probability in zip(_SPAWN_EXPONENTS, _SPAWN_PROBABILITIES):
            spawned = afterstate | (exponent << np.uint64(4 * i))
            expected += cell_probability * spawn_probability * _state_value(weights, layout, spawned, depth)
    return expected


@njit
def action_values(weights: np.ndarray, layout: TupleLayout, board: np.uint64, depth: int) -> np.ndarray:
    """Return each action's value searching `depth` moves ahead, or NaN if it doesn't change the board."""
    values = np.full(bb.NUM_ACTIONS, np.nan)
    for action in range(bb.NUM_ACTIONS):
        afterstate, reward = bb.move(board, action)
        if afterstate != board:
            values[action] = reward + _afterstate_value(weights, layout, afterstate, depth - 1)
    return values
