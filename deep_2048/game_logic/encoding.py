import numpy as np

from deep_2048.game_logic.models import BoardEncoding


def encode_board(board: np.ndarray) -> BoardEncoding:
    encoded_board = np.zeros_like(board, dtype=np.float32)
    non_zero_mask = board > 0
    encoded_board[non_zero_mask] = np.log2(board[non_zero_mask]).astype(np.float32)

    flattened_vector = encoded_board.flatten().astype(np.float32)
    return BoardEncoding(vector=flattened_vector)


def board_to_list(board: np.ndarray) -> list[list[int]]:
    return board.astype(int).tolist()