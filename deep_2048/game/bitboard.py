"""Bitboard version of the game engine, for agents that search or train on many boards.

The board is a uint64 of 16 four-bit cells, with cell (row, col) at bit 4 * (4 * row + col).
Each cell holds its tile's exponent (0 = empty, 1 = 2, 2 = 4, ...). Tiles cap at 32768, so
two 32768s won't merge.

The numba functions here and in the modules that call them aren't cached, because numba only
notices changes to a cached function's own file and would keep running stale callees. Small
hot-path helpers use inline="always": numba otherwise calls them as separate functions, which
roughly doubles the cost of evaluating a board.
"""

import numpy as np
from numba import njit

from deep_2048.game.constants import SPAWN_TILE_WEIGHTS, Action

UP = int(Action.UP)
DOWN = int(Action.DOWN)
LEFT = int(Action.LEFT)
RIGHT = int(Action.RIGHT)
NUM_ACTIONS = 4

NUM_CELLS = 16
NUM_SYMMETRIES = 8
MAX_EXPONENT = 15
SPAWN_FOUR_PROBABILITY = SPAWN_TILE_WEIGHTS[4] / sum(SPAWN_TILE_WEIGHTS.values())

_CELL_BITS = np.uint64(4)
_CELL_MASK = np.uint64(0xF)
_LOW_BIT_OF_EACH_CELL = np.uint64(0x1111111111111111)
_ROW_MASK = np.uint64(0xFFFF)

_THREE_CELLS = np.uint64(12)
_SIX_CELLS = np.uint64(24)
_KEEP_CELLS = np.uint64(0xF0F00F0FF0F00F0F)
_UPPER_CELLS = np.uint64(0x0000F0F00000F0F0)
_LOWER_CELLS = np.uint64(0x0F0F00000F0F0000)
_KEEP_BLOCKS = np.uint64(0xFF00FF0000FF00FF)
_UPPER_BLOCK = np.uint64(0x00000000FF00FF00)
_LOWER_BLOCK = np.uint64(0x00FF00FF00000000)

_ONE_ROW = np.uint64(16)
_THREE_ROWS = np.uint64(48)
_SECOND_ROW = np.uint64(0x00000000FFFF0000)
_THIRD_ROW = np.uint64(0x0000FFFF00000000)
_ONE_CELL = np.uint64(4)
_OUTER_COLUMNS = np.uint64(0x000F000F000F000F)
_INNER_COLUMNS = np.uint64(0x00F000F000F000F0)


def _slide_row_left(cells: list[int]) -> tuple[list[int], int]:
    """Slide a row of exponents left and return (new row, reward)."""
    tiles = [c for c in cells if c != 0]
    slid: list[int] = []
    reward = 0
    i = 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1] and tiles[i] < MAX_EXPONENT:
            slid.append(tiles[i] + 1)
            reward += 1 << (tiles[i] + 1)
            i += 2
        else:
            slid.append(tiles[i])
            i += 1
    return slid + [0] * (4 - len(slid)), reward


def _pack_row(cells: list[int]) -> int:
    return sum(c << (4 * i) for i, c in enumerate(cells))


def _build_row_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precompute the left and right move and reward for every possible row."""
    left = np.zeros(1 << 16, dtype=np.uint64)
    right = np.zeros(1 << 16, dtype=np.uint64)
    left_reward = np.zeros(1 << 16, dtype=np.int64)
    right_reward = np.zeros(1 << 16, dtype=np.int64)

    for row in range(1 << 16):
        cells = [(row >> (4 * i)) & 0xF for i in range(4)]

        slid, reward = _slide_row_left(cells)
        left[row] = _pack_row(slid)
        left_reward[row] = reward

        slid, reward = _slide_row_left(cells[::-1])
        right[row] = _pack_row(slid[::-1])
        right_reward[row] = reward

    return left, left_reward, right, right_reward


_ROW_LEFT, _ROW_LEFT_REWARD, _ROW_RIGHT, _ROW_RIGHT_REWARD = _build_row_tables()


@njit
def transpose(board: np.uint64) -> np.uint64:
    """Swap rows and columns.

    Swaps the off-diagonal cells within each 2x2 block, then swaps the two off-diagonal blocks.
    """
    board = (board & _KEEP_CELLS) | ((board & _UPPER_CELLS) << _THREE_CELLS) | ((board & _LOWER_CELLS) >> _THREE_CELLS)
    return (board & _KEEP_BLOCKS) | ((board & _UPPER_BLOCK) << _SIX_CELLS) | ((board & _LOWER_BLOCK) >> _SIX_CELLS)


@njit
def flip_vertical(board: np.uint64) -> np.uint64:
    """Reverse the order of the rows."""
    return (board >> _THREE_ROWS) | ((board >> _ONE_ROW) & _SECOND_ROW) | ((board << _ONE_ROW) & _THIRD_ROW) | (board << _THREE_ROWS)


@njit
def flip_horizontal(board: np.uint64) -> np.uint64:
    """Reverse the order of the cells in each row."""
    return ((board & _OUTER_COLUMNS) << _THREE_CELLS) | ((board & _INNER_COLUMNS) << _ONE_CELL) | ((board >> _ONE_CELL) & _INNER_COLUMNS) | ((board >> _THREE_CELLS) & _OUTER_COLUMNS)


@njit
def symmetries(board: np.uint64) -> tuple[np.uint64, ...]:
    """Return the board's 4 rotations, then the 4 rotations of its transpose.

    Rotations are anticlockwise, matching np.rot90.
    """
    transposed = transpose(board)
    mirrored = flip_horizontal(board)
    mirrored_transposed = flip_horizontal(transposed)
    return (
        board,
        flip_vertical(transposed),
        flip_vertical(mirrored),
        mirrored_transposed,
        transposed,
        flip_vertical(board),
        flip_vertical(mirrored_transposed),
        mirrored,
    )


@njit
def _move_rows(board: np.uint64, row_table: np.ndarray, reward_table: np.ndarray) -> tuple[np.uint64, int]:
    """Look up each row in a move table and return (new board, total reward)."""
    moved = np.uint64(0)
    reward = 0
    for r in range(4):
        shift = np.uint64(16 * r)
        row = (board >> shift) & _ROW_MASK
        moved |= row_table[row] << shift
        reward += reward_table[row]
    return moved, reward


@njit
def move(board: np.uint64, action: int) -> tuple[np.uint64, int]:
    """Apply a move and return (new board, reward). An illegal move leaves the board unchanged."""
    if action == LEFT:
        return _move_rows(board, _ROW_LEFT, _ROW_LEFT_REWARD)
    if action == RIGHT:
        return _move_rows(board, _ROW_RIGHT, _ROW_RIGHT_REWARD)

    if action == UP:
        moved, reward = _move_rows(transpose(board), _ROW_LEFT, _ROW_LEFT_REWARD)
    else:
        moved, reward = _move_rows(transpose(board), _ROW_RIGHT, _ROW_RIGHT_REWARD)
    return transpose(moved), reward


@njit
def is_game_over(board: np.uint64) -> bool:
    for action in range(NUM_ACTIONS):
        moved, _ = move(board, action)
        if moved != board:
            return False
    return True


@njit
def get_cell(board: np.uint64, index: int) -> int:
    # Cast to int64 because numba turns arithmetic mixing uint64 and int64 into floats
    return int(np.int64((board >> (_CELL_BITS * np.uint64(index))) & _CELL_MASK))


@njit(inline="always")
def _empty_cells(board: np.uint64) -> np.uint64:
    """Return a mask with the lowest bit of every empty cell set."""
    occupied = board | (board >> np.uint64(1))
    occupied |= occupied >> np.uint64(2)
    return ~occupied & _LOW_BIT_OF_EACH_CELL


@njit(inline="always")
def _count_cells(cells: np.uint64) -> int:
    """Count the cells in a mask from _empty_cells."""
    if cells == _LOW_BIT_OF_EACH_CELL:
        return NUM_CELLS
    # Multiplying adds every cell's bit into the top cell, which can hold up to 15
    return int(np.int64((cells * _LOW_BIT_OF_EACH_CELL) >> np.uint64(60)))


@njit
def count_empty(board: np.uint64) -> int:
    return _count_cells(_empty_cells(board))


@njit
def max_exponent(board: np.uint64) -> int:
    highest = 0
    for i in range(NUM_CELLS):
        highest = max(highest, get_cell(board, i))
    return highest


@njit
def seed(value: int) -> None:
    """Seed numba's RNG, which is separate from numpy's."""
    np.random.seed(value)


@njit
def spawn_tile(board: np.uint64) -> np.uint64:
    """Add a 2 or 4 to a random empty cell."""
    empty = _empty_cells(board)
    n_empty = _count_cells(empty)
    if n_empty == 0:
        return board

    target = np.random.randint(n_empty)
    exponent = np.uint64(2) if np.random.random() < SPAWN_FOUR_PROBABILITY else np.uint64(1)
    for _ in range(target):
        empty &= empty - np.uint64(1)
    lowest_empty = empty & (~empty + np.uint64(1))
    return board | (exponent * lowest_empty)


@njit
def initial_board() -> np.uint64:
    return spawn_tile(spawn_tile(np.uint64(0)))


def from_array(board: np.ndarray) -> np.uint64:
    """Pack a 4x4 array of tiles into a bitboard."""
    packed = 0
    for i, tile in enumerate(np.asarray(board).reshape(-1).tolist()):
        if tile == 0:
            continue
        exponent = int(tile).bit_length() - 1
        if tile != 1 << exponent or not 1 <= exponent <= MAX_EXPONENT:
            raise ValueError(f"Can't pack tile {tile}")
        packed |= exponent << (4 * i)
    return np.uint64(packed)


def to_array(board: np.uint64) -> np.ndarray:
    """Unpack a bitboard into a 4x4 array of tiles."""
    exponents = [(int(board) >> (4 * i)) & 0xF for i in range(NUM_CELLS)]
    return np.array([0 if e == 0 else 1 << e for e in exponents], dtype=np.int32).reshape(4, 4)
