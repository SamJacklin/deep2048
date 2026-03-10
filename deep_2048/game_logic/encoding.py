import numpy as np

from deep_2048.config import get_settings
from deep_2048.game_logic.models import BoardEncoding

settings = get_settings()


def encode_board(board: np.ndarray) -> BoardEncoding:
    """One-hot encode the board.

    Produces a (NUM_TILE_LEVELS, board_size, board_size) array, then flattens
    it. Each cell is represented as a one-hot vector over tile levels, where
    level 0 = empty and level k = tile value 2^k. This layout is CNN-ready:
    when switching to a convolutional network, reshape the flat vector back to
    (NUM_TILE_LEVELS, board_size, board_size) before passing to the network.
    """
    num_levels = settings.NUM_TILE_LEVELS
    board_size = board.shape[0]

    one_hot = np.zeros((num_levels, board_size, board_size), dtype=np.float32)
    for r in range(board_size):
        for c in range(board_size):
            val = board[r, c]
            level = 0 if val == 0 else min(int(np.log2(val)), num_levels - 1)
            one_hot[level, r, c] = 1.0

    return BoardEncoding(vector=one_hot.flatten())


def board_to_list(board: np.ndarray) -> list[list[int]]:
    return board.astype(int).tolist()
