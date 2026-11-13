"""Expectimax search over bitboards, with the n-tuple network valuing afterstates at the horizon.

A move is worth its reward plus what follows. Below the horizon that's the expected value over
every tile that could spawn. At the horizon it's the network's value of the afterstate, which
is exactly what TD training taught it to predict. Depth 1 is the greedy agent.
"""

from typing import NamedTuple

import numpy as np
from numba import njit

from deep_2048.agents.ntuple.network import TupleLayout, evaluate
from deep_2048.game import bitboard as bb

_SPAWN_EXPONENTS = (np.uint64(1), np.uint64(2))
_SPAWN_PROBABILITIES = (1.0 - bb.SPAWN_FOUR_PROBABILITY, bb.SPAWN_FOUR_PROBABILITY)
_TABLE_BITS = 16
_HASH_MULTIPLIER = np.uint64(0x9E3779B97F4A7C15)


class Transpositions(NamedTuple):
    """Afterstate values found in the current search, as a direct-mapped hash table that always replaces.

    Each entry is stamped with the search that wrote it, so a new search starts empty without clearing
    the table. With a cutoff, a value reached by a likelier path is reused for less likely ones, which is
    approximate but deterministic, so act() and play() still agree. With `reuse` off, nothing is looked up,
    which gives the exact pruned search to measure that approximation against.
    """

    boards: np.ndarray
    depths: np.ndarray
    values: np.ndarray
    searches: np.ndarray
    current: np.ndarray  # one element: the search being run
    reuse: bool


@njit
def new_transpositions(reuse: bool = True) -> Transpositions:
    size = 1 << _TABLE_BITS
    return Transpositions(np.zeros(size, np.uint64), np.zeros(size, np.int64), np.zeros(size), np.zeros(size, np.int64), np.zeros(1, np.int64), reuse)


@njit
def _state_value(weights: np.ndarray, layout: TupleLayout, board: np.uint64, depth: int, probability: float, cutoff: float, table: Transpositions) -> float:
    """Return the best move's value from a state, searching `depth` moves ahead, or 0 if the game is over."""
    best = -np.inf
    for action in range(bb.NUM_ACTIONS):
        afterstate, reward = bb.move(board, action)
        if afterstate != board:
            best = max(best, reward + _afterstate_value(weights, layout, afterstate, board, depth - 1, probability, cutoff, table))
    return best if best > -np.inf else 0.0


# Inlined so its mutual recursion with _state_value becomes self-recursion, which numba can compile
@njit(inline="always")
def _afterstate_value(weights: np.ndarray, layout: TupleLayout, afterstate: np.uint64, state: np.uint64, depth: int, probability: float, cutoff: float, table: Transpositions) -> float:
    """Return the network's value at the horizon, otherwise the expected value over every tile that could spawn.

    `probability` is the chance of reaching this afterstate from the root, and below `cutoff` it counts as the horizon.
    """
    if depth == 0 or probability < cutoff:
        # At the horizon the network values it in the stage of `state`, the board the move was made from
        return evaluate(weights, layout, afterstate, state)

    slot = (afterstate * _HASH_MULTIPLIER + np.uint64(depth)) >> np.uint64(64 - _TABLE_BITS)
    search = table.current[0]
    if table.reuse and table.searches[slot] == search and table.boards[slot] == afterstate and table.depths[slot] == depth:
        return table.values[slot]

    n_empty = bb.count_empty(afterstate)
    cell_probability = 1.0 / n_empty
    expected = 0.0
    for i in range(bb.NUM_CELLS):
        if bb.get_cell(afterstate, i) != 0:
            continue
        for exponent, spawn_probability in zip(_SPAWN_EXPONENTS, _SPAWN_PROBABILITIES):
            spawned = afterstate | (exponent << np.uint64(4 * i))
            branch = cell_probability * spawn_probability
            expected += branch * _state_value(weights, layout, spawned, depth, probability * branch, cutoff, table)
    table.boards[slot], table.depths[slot], table.values[slot], table.searches[slot] = afterstate, depth, expected, search
    return expected


@njit(nogil=True)
def play_seeded(weights: np.ndarray, layout: TupleLayout, depth: int, cutoff: float, rng: np.random.Generator) -> tuple[int, int]:
    """Play a whole game searching `depth` moves ahead and return (score, max tile).

    Given np.random.default_rng(seed), it's the same game the benchmark runner plays on the Python engine
    with the agent's act(), including ties going to the first action, without that engine's per-move overhead.

    The bitboard caps tiles at 32768, so it can't merge two of them as Game2048 can. A game that comes to hold
    two ends there instead, reported with a max tile of 65536 ("would have made 65536") and its score so far,
    a lower bound. So the >=65536 rate counts the games the cap cut short.
    """
    # The runner resets Game2048 after creating it, so the seed's first board is dealt and thrown away
    bb.spawn_tile_seeded(bb.spawn_tile_seeded(np.uint64(0), rng), rng)
    board = bb.spawn_tile_seeded(bb.spawn_tile_seeded(np.uint64(0), rng), rng)
    score = 0
    table = new_transpositions()
    while True:
        values = _search(weights, layout, board, depth, cutoff, table)
        best = -1
        for action in range(bb.NUM_ACTIONS):
            if not np.isnan(values[action]) and (best < 0 or values[action] > values[best]):
                best = action
        if best < 0:
            return score, 1 << bb.max_exponent(board)
        board, reward = bb.move(board, best)
        score += reward
        board = bb.spawn_tile_seeded(board, rng)
        if has_two_top_tiles(board):
            return score, 1 << (bb.MAX_EXPONENT + 1)


@njit
def has_two_top_tiles(board: np.uint64) -> bool:
    """Whether the board has two tiles at the bitboard's cap (32768), which only Game2048 can merge."""
    if bb.max_exponent(board) < bb.MAX_EXPONENT:
        return False
    count = 0
    for i in range(bb.NUM_CELLS):
        count += bb.get_cell(board, i) == bb.MAX_EXPONENT
    return count >= 2


@njit
def action_values(weights: np.ndarray, layout: TupleLayout, board: np.uint64, depth: int, cutoff: float = 0.0, reuse: bool = True) -> np.ndarray:
    """Return each action's value searching `depth` moves ahead, or NaN if it doesn't change the board.

    Spawn branches less likely than `cutoff` from the root stop at the network's value instead (0 searches everything).
    `reuse` off turns off the transposition table, for measuring what it changes.
    """
    return _search(weights, layout, board, depth, cutoff, new_transpositions(reuse))


@njit
def _search(weights: np.ndarray, layout: TupleLayout, board: np.uint64, depth: int, cutoff: float, table: Transpositions) -> np.ndarray:
    table.current[0] += 1
    values = np.full(bb.NUM_ACTIONS, np.nan)
    for action in range(bb.NUM_ACTIONS):
        afterstate, reward = bb.move(board, action)
        if afterstate != board:
            values[action] = reward + _afterstate_value(weights, layout, afterstate, board, depth - 1, 1.0, cutoff, table)
    return values
