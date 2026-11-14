"""Stage 7: the anatomy of the late game. Plays that record what happened, and policies that search for a while.

record_trajectory plays greedily on from a board and keeps every board a decision was made from, so points along a
real game can be replayed under other policies. play_mixed searches `depth` moves ahead for the first `deep_moves`
decisions and plays greedily after (a temporary intervention), counting what the search cost. Both draw spawns from
the rng they're given, so policies started from the same board with the same seed share spawns until their moves differ.
"""

import numpy as np
from numba import njit, types
from numba.typed import List

from deep_2048.agents.ntuple.network import TupleLayout
from deep_2048.agents.ntuple.search import _search, has_two_top_tiles, new_transpositions
from deep_2048.game import bitboard as bb

TOP_EXPONENT = bb.MAX_EXPONENT  # 32768


@njit(inline="always")
def _first_best(values: np.ndarray) -> int:
    best = -1
    for action in range(bb.NUM_ACTIONS):
        if not np.isnan(values[action]) and (best < 0 or values[action] > values[best]):
            best = action
    return best


@njit(nogil=True)
def record_trajectory(weights: np.ndarray, layout: TupleLayout, start: np.uint64, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Play greedily from `start` until the game ends or first holds a 32768, and return (every board a move was made
    from, in order, starting with `start`; the score gained before each; the final max tile; the total score gained)."""
    board = np.uint64(start)
    boards = List.empty_list(types.uint64)
    scores = List.empty_list(types.int64)
    score = 0
    table = new_transpositions(True, 0)
    while bb.max_exponent(board) < TOP_EXPONENT:
        best = _first_best(_search(weights, layout, board, 1, 0.0, table))
        if best < 0:
            break
        boards.append(board)
        scores.append(score)
        board, reward = bb.move(board, best)
        score += reward
        board = bb.spawn_tile_seeded(board, rng)
    return np.asarray(boards), np.asarray(scores), 1 << bb.max_exponent(board), score


@njit(nogil=True)
def play_mixed(weights: np.ndarray, layout: TupleLayout, start: np.uint64, depth: int, deep_moves: int, cutoff: float,
               rng: np.random.Generator) -> tuple[int, int, int, int]:
    """Play on from `start` to the end, searching `depth` moves ahead (with `cutoff`) for the first `deep_moves` decisions
    (all of them if negative) and greedily after. Returns (score gained, max tile, moves, chance nodes expanded)."""
    board = np.uint64(start)
    score = moves = 0
    table = new_transpositions()
    while True:
        searching = depth > 1 and (deep_moves < 0 or moves < deep_moves)
        best = _first_best(_search(weights, layout, board, depth if searching else 1, cutoff if searching else 0.0, table))
        if best < 0:
            return score, 1 << bb.max_exponent(board), moves, int(table.current[1])
        board, reward = bb.move(board, best)
        score += reward
        moves += 1
        board = bb.spawn_tile_seeded(board, rng)
        if has_two_top_tiles(board):
            return score, 1 << (bb.MAX_EXPONENT + 1), moves, int(table.current[1])
