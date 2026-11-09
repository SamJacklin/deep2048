"""Hand-crafted board evaluator for the heuristic agent.

A board's score adds four signals, all measured on tile exponents (log2 values):
  - Monotonicity: minus how far each row and column is from sorted, in its better direction.
  - Merge potential: the exponent of each pair of equal adjacent tiles.
  - Empty cells: 10 per empty cell, since running out of space ends the game.
  - Corner anchor: twice the largest exponent sitting in a corner.

Design notes:
  - A snake-pattern positional bonus was removed because it penalised merging: two 128s
    weighted 15 and 14 scored 203, but merging them into a 256 weighted 15 scored only 120.
  - Empty cells use a linear weight rather than a log scale, so survival weighs about as much
    as board organisation.
  - Merge potential replaced smoothness, which also penalised good gradients like [8, 4, 2, 1].

Every signal but the corner is a sum over rows and columns, so each row's share is looked up in
a table of all 65,536 rows, and columns are read as the rows of the transposed board.
"""

import numpy as np
from numba import njit

from deep_2048.game import bitboard as bb

_EMPTY_CELL_WEIGHT = 10.0
_CORNER_WEIGHT = 2.0
_ROW_MASK = np.uint64(0xFFFF)


def _build_line_scores() -> np.ndarray:
    """Score every possible row for monotonicity and merge potential."""
    rows = np.arange(1 << 16)
    cells = (rows[:, None] >> (4 * np.arange(4))) & 0xF
    steps = np.diff(cells, axis=1)
    disorder = np.minimum(np.maximum(-steps, 0).sum(axis=1), np.maximum(steps, 0).sum(axis=1))
    mergeable = (cells[:, :-1] == cells[:, 1:]) & (cells[:, :-1] > 0)
    merges = np.where(mergeable, cells[:, :-1], 0).sum(axis=1)
    return (merges - disorder).astype(np.float64)


_LINE_SCORES = _build_line_scores()


@njit(inline="always")
def _sum_row_scores(board: np.uint64) -> float:
    total = 0.0
    for r in range(4):
        total += _LINE_SCORES[(board >> np.uint64(16 * r)) & _ROW_MASK]
    return total


@njit(inline="always")
def evaluate(board: np.uint64) -> float:
    """Score a bitboard. Higher is better."""
    corner = max(bb.get_cell(board, 0), bb.get_cell(board, 3), bb.get_cell(board, 12), bb.get_cell(board, 15))
    return _sum_row_scores(board) + _sum_row_scores(bb.transpose(board)) + _EMPTY_CELL_WEIGHT * bb.count_empty(board) + _CORNER_WEIGHT * corner


def board_heuristic(board: np.ndarray) -> float:
    """Score a 4x4 array of tiles. Higher is better."""
    return float(evaluate(bb.from_array(board)))
