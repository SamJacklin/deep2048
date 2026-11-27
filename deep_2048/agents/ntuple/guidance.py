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
from deep_2048.agents.ntuple.models import SearchGuide
from deep_2048.agents.ntuple.network import TupleLayout, evaluate, update, weight_indices
from deep_2048.agents.ntuple.search import Transpositions, _search, has_two_top_tiles, new_transpositions
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


@njit
def _gap_sensitivity(layout: TupleLayout, raised: np.uint64, lowered: np.uint64, state: np.uint64) -> float:
    """How much the gap between two afterstates' values moves per unit of a +1 / -1 update to every weight each reads:
    the sum over weights of (reads by `raised` - reads by `lowered`) squared. Weights both read, as neighbouring
    afterstates often do, partly or wholly cancel; symmetries and downgrading come from the network's own reads."""
    up, down = weight_indices(layout, raised, state), weight_indices(layout, lowered, state)
    indices = np.concatenate((up, down))
    signs = np.concatenate((np.ones(up.size), -np.ones(down.size)))
    order = np.argsort(indices)
    total = 0.0
    run = 0.0
    for k in range(order.size):
        run += signs[order[k]]
        if k + 1 == order.size or indices[order[k + 1]] != indices[order[k]]:
            total += run * run
            run = 0.0
    return total


@njit
def guided_move(weights: np.ndarray, layout: TupleLayout, board: np.uint64, guide: SearchGuide, table: Transpositions) -> tuple[np.uint64, int, float]:
    """Choose a training move under the guide: greedy, or searched if the decision is in the guide's band.

    Returns the chosen afterstate, its reward, and reward + value (-inf if no move is legal), as training's greedy
    choice does, so TD learns on the guided trajectory. A ranking update, if any, is applied before that value is read.
    """
    values = greedy_values(weights, layout, board)
    choice = _first_best(values)
    if choice < 0:
        return board, 0, -np.inf
    guide.counts[0] += 1
    if bb.max_exponent(board) >= guide.min_exponent:
        margin = relative_margin(values)
        if guide.margin_low <= margin < guide.margin_high:
            searched = _search(weights, layout, board, guide.depth, 0.0, table)
            teacher = _first_best(searched)
            guide.counts[1] += 1
            if teacher != choice:
                guide.counts[2] += 1
                preference = searched[teacher] - searched[choice]
                if guide.rank_step > 0 and preference >= guide.rank_min_gap * abs(searched[teacher]):
                    teacher_after, _ = bb.move(board, teacher)
                    greedy_after, _ = bb.move(board, choice)
                    sensitivity = _gap_sensitivity(layout, teacher_after, greedy_after, board)
                    if sensitivity > 0:
                        # Raise the teacher's afterstate and lower greedy play's by delta per read: the gap between their
                        # reward + value moves by exactly delta * sensitivity, closing rank_step of it (plus rank_margin)
                        gap = values[choice] - values[teacher] + guide.rank_margin
                        delta = guide.rank_step * gap / sensitivity
                        update(weights, layout, teacher_after, board, delta)
                        update(weights, layout, greedy_after, board, -delta)
                        guide.counts[3] += 1
                choice = teacher
    afterstate, reward = bb.move(board, choice)
    return afterstate, reward, reward + evaluate(weights, layout, afterstate, board)
