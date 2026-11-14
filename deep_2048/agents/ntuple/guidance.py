"""Search guidance: greedy play that searches deeper only at selected decisions.

A decision is searched when the board's largest tile has reached 2**min_exponent and greedy play's relative margin
between its top two moves (best minus second, as a share of the best's reward + value) lies in [margin_low,
margin_high). Everything else is played greedily, exactly as training plays. So the policy runs from pure greedy play
(an empty band) to depth-3 search on every late decision ([0, inf)), and bands between show which decisions search
helps most.
"""

import numpy as np
from numba import njit

from deep_2048.agents.ntuple.diagnostics import greedy_values, relative_margin
from deep_2048.agents.ntuple.network import TupleLayout
from deep_2048.agents.ntuple.search import _search, has_two_top_tiles, new_transpositions
from deep_2048.game import bitboard as bb


@njit(inline="always")
def _first_best(values: np.ndarray) -> int:
    """The first action with the highest value, as play_seeded picks ties; -1 if none is legal."""
    best = -1
    for action in range(bb.NUM_ACTIONS):
        if not np.isnan(values[action]) and (best < 0 or values[action] > values[best]):
            best = action
    return best


@njit(nogil=True)
def play_selective(weights: np.ndarray, layout: TupleLayout, start: np.uint64, depth: int, min_exponent: int, margin_low: float, margin_high: float,
                   rng: np.random.Generator) -> tuple[int, int, int, int, int, int, int]:
    """Play on from `start`, searching `depth` moves ahead at selected decisions and greedily elsewhere.

    Returns (score gained, max tile, decisions, decisions searched, searched decisions where search chose another move
    than greedy play, chance nodes expanded, afterstates valued at the horizon).
    """
    board = np.uint64(start)
    score = decisions = searched = disagreed = 0
    table = new_transpositions()
    while True:
        values = greedy_values(weights, layout, board)
        greedy = _first_best(values)
        if greedy < 0:
            return score, 1 << bb.max_exponent(board), decisions, searched, disagreed, int(table.current[1]), int(table.current[2])
        choice = greedy
        if bb.max_exponent(board) >= min_exponent:
            margin = relative_margin(values)
            if margin_low <= margin < margin_high:
                choice = _first_best(_search(weights, layout, board, depth, 0.0, table))
                searched += 1
                disagreed += choice != greedy
        decisions += 1
        board, reward = bb.move(board, choice)
        score += reward
        board = bb.spawn_tile_seeded(board, rng)
        if has_two_top_tiles(board):
            return score, 1 << (bb.MAX_EXPONENT + 1), decisions, searched, disagreed, int(table.current[1]), int(table.current[2])
