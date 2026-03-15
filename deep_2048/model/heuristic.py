"""Board heuristic evaluation and heuristic expectimax action selection.

The heuristic scores a board position using four signals:
  - Monotonicity:     tiles form a gradient in every row and column
  - Merge potential:  reward for adjacent tiles with equal values (available merges)
  - Empty cells:      dominant survival factor — losing space leads to game-over
  - Corner anchor:    reward for the highest tile sitting in any corner

All signals operate in log2 space so tile magnitudes are comparable.

Design notes:
  - The old corner_bonus (snake-pattern dot product) was removed because it
    actively penalised merging: two 128s at weights 15,14 scored 203; merging
    them into one 256 at weight 15 scored only 120 — a drop of 83 per merge.
  - Empty cells now use a linear per-cell weight (10.0) rather than log-scale,
    giving survival roughly equal weight to board organisation.
  - Smoothness was replaced by merge_potential: penalising tile differences
    also penalised good monotone gradients like [8,4,2,1].
"""

import numpy as np

from deep_2048.game_logic.constants import Action
from deep_2048.game_logic.moves import apply_move, get_valid_actions, is_game_over


def _log_board(board: np.ndarray) -> np.ndarray:
    safe = np.maximum(board, 1).astype(float)
    return np.where(board > 0, np.log2(safe), 0.0)


def _monotonicity(lb: np.ndarray) -> float:
    """Penalise rows/columns that are non-monotone in either direction."""
    row_diff = np.diff(lb, axis=1)                          # (4, 3)
    col_diff = np.diff(lb, axis=0)                          # (3, 4)
    row_left  = np.maximum(-row_diff, 0).sum(axis=1)
    row_right = np.maximum( row_diff, 0).sum(axis=1)
    col_up    = np.maximum(-col_diff, 0).sum(axis=0)
    col_down  = np.maximum( col_diff, 0).sum(axis=0)
    return -(np.minimum(row_left, row_right).sum() + np.minimum(col_up, col_down).sum())


def _merge_potential(board: np.ndarray, lb: np.ndarray) -> float:
    """Reward adjacent same-value tiles (available merges), weighted by log2 value."""
    h_match = (board[:, :-1] == board[:, 1:]) & (board[:, :-1] > 0)
    v_match = (board[:-1, :] == board[1:, :]) & (board[:-1, :] > 0)
    return float(lb[:, :-1][h_match].sum() + lb[:-1, :][v_match].sum())


def _corner_max(lb: np.ndarray) -> float:
    """Log2 value of the highest tile sitting in any corner."""
    return float(max(lb[0, 0], lb[0, 3], lb[3, 0], lb[3, 3]))


def board_heuristic(board: np.ndarray) -> float:
    """Combined board quality score. Higher = better position."""
    lb = _log_board(board)
    n_empty = float(np.sum(board == 0))
    return (
        1.0 * _monotonicity(lb)            # gradient structure   (~-20 to 0)
        + 1.0 * _merge_potential(board, lb) # merge readiness       (~0 to 15)
        + 10.0 * n_empty                    # survival factor       (~0 to 140)
        + 2.0 * _corner_max(lb)             # corner anchor         (~0 to 22)
    )


# ── Expectimax tree search ────────────────────────────────────────────────────

def _expectimax(board: np.ndarray, moves_left: int) -> float:
    """Player node: maximise over valid actions."""
    if moves_left == 0 or is_game_over(board):
        return board_heuristic(board)
    valid = get_valid_actions(board)
    if not valid:
        return board_heuristic(board)
    best = float("-inf")
    for action in valid:
        result = apply_move(board, action)
        if result.changed:
            val = _chance_node(result.board, moves_left - 1)
            if val > best:
                best = val
    return best


def _chance_node(board: np.ndarray, moves_left: int) -> float:
    """Chance node: expected value over all possible tile spawns."""
    empty = [(r, c) for r in range(4) for c in range(4) if board[r, c] == 0]
    if not empty:
        return _expectimax(board, moves_left)
    prob_per_cell = 1.0 / len(empty)
    expected = 0.0
    for r, c in empty:
        for tile, tile_prob in ((2, 0.9), (4, 0.1)):
            b = board.copy()
            b[r, c] = tile
            expected += prob_per_cell * tile_prob * _expectimax(b, moves_left)
    return expected


def heuristic_select(
    board: np.ndarray,
    valid_actions: list[Action],
    depth: int = 2,
) -> tuple[Action, dict[str, float]]:
    """Choose an action using heuristic expectimax search.

    Args:
        board:         Raw 4x4 board (tile values, 0 = empty).
        valid_actions: Actions that change the board state.
        depth:         Player-move plies to search (~3000 leaf evals at depth 2,
                       manageable for real-time use).

    Returns:
        (chosen_action, scores) where scores maps every action name to its
        expected heuristic value (0.0 for invalid/unchanged actions).
    """
    valid_set = set(valid_actions)
    scores: dict[str, float] = {}

    for action in Action:
        if action not in valid_set:
            scores[action.name] = 0.0
            continue
        result = apply_move(board, action)
        if not result.changed:
            scores[action.name] = 0.0
            continue
        scores[action.name] = round(_chance_node(result.board, depth - 1), 4)

    chosen = max(valid_set, key=lambda a: scores[a.name])
    return chosen, scores
