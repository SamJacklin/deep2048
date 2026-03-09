import numpy as np

from deep_2048.game_logic.constants import ACTIONS, Action
from deep_2048.game_logic.models import MoveAnalysis, MoveResult


def _collapse_row_left(row: np.ndarray) -> tuple[np.ndarray, int]:
    non_zero_values = [int(value) for value in row if int(value) != 0]

    merged_values: list[int] = []
    reward = 0
    index = 0

    while index < len(non_zero_values):
        current_value = non_zero_values[index]

        if index + 1 < len(non_zero_values) and non_zero_values[index + 1] == current_value:
            merged_value = current_value * 2
            merged_values.append(merged_value)
            reward += merged_value
            index += 2
        else:
            merged_values.append(current_value)
            index += 1

    padded_row = merged_values + [0] * (len(row) - len(merged_values))
    return np.array(padded_row, dtype=np.int32), reward


def _move_left(board: np.ndarray) -> MoveResult:
    new_board = np.zeros_like(board)
    total_reward = 0

    for row_index in range(board.shape[0]):
        collapsed_row, row_reward = _collapse_row_left(board[row_index])
        new_board[row_index] = collapsed_row
        total_reward += row_reward

    changed = not np.array_equal(board, new_board)
    return MoveResult(board=new_board, reward=total_reward, changed=changed)


def _move_right(board: np.ndarray) -> MoveResult:
    reversed_board = np.fliplr(board)
    move_result = _move_left(reversed_board)
    restored_board = np.fliplr(move_result.board)

    changed = not np.array_equal(board, restored_board)
    return MoveResult(board=restored_board, reward=move_result.reward, changed=changed)


def _move_up(board: np.ndarray) -> MoveResult:
    transposed_board = board.T
    move_result = _move_left(transposed_board)
    restored_board = move_result.board.T

    changed = not np.array_equal(board, restored_board)
    return MoveResult(board=restored_board, reward=move_result.reward, changed=changed)


def _move_down(board: np.ndarray) -> MoveResult:
    transposed_board = board.T
    move_result = _move_right(transposed_board)
    restored_board = move_result.board.T

    changed = not np.array_equal(board, restored_board)
    return MoveResult(board=restored_board, reward=move_result.reward, changed=changed)


def apply_move(board: np.ndarray, action: Action) -> MoveResult:
    if action == Action.LEFT:
        return _move_left(board)
    if action == Action.RIGHT:
        return _move_right(board)
    if action == Action.UP:
        return _move_up(board)
    if action == Action.DOWN:
        return _move_down(board)

    raise ValueError(f"Unsupported action: {action}")


def get_valid_actions(board: np.ndarray) -> list[Action]:
    valid_actions: list[Action] = []

    for action in ACTIONS:
        result = apply_move(board, action)
        if result.changed:
            valid_actions.append(action)

    return valid_actions


def is_game_over(board: np.ndarray) -> bool:
    return len(get_valid_actions(board)) == 0


def analyse_actions(board: np.ndarray) -> list[MoveAnalysis]:
    analyses: list[MoveAnalysis] = []

    for action in ACTIONS:
        result = apply_move(board, action)
        analyses.append(
            MoveAnalysis(
                action=action,
                changed=result.changed,
                reward=result.reward,
            )
        )

    return analyses