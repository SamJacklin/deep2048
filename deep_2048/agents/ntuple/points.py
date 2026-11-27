"""Points: the stages of a game named by its big tiles, for measuring where search would play better than the network.

A point is the board's largest tile and which of the next `detail` tile sizes below it are present, so "16384 with
8192 and 2048" is one point. Within a stage those tiles are the stage's progress written in binary (16384, 8192 and
2048 present means 16384 + 10240 of the 32768), so the points of a stage run from just arrived to nearly done.

The key packs a point into one int: the largest tile's exponent times KEY_STRIDE plus a bit for each smaller size
present, bit 0 for the size just below the largest.
"""

import numpy as np
from numba import njit

from deep_2048.agents.ntuple.network import TupleLayout
from deep_2048.agents.ntuple.search import _search, has_two_top_tiles, new_transpositions
from deep_2048.game import bitboard as bb

KEY_STRIDE = 256
MAX_POINTS = 64  # a game records at most this many points


@njit
def point_key(board: np.uint64, detail: int) -> int:
    """The board's point: its largest tile's exponent * KEY_STRIDE, plus bit k - 1 if it holds 2**(largest - k)."""
    present = 0
    for i in range(bb.NUM_CELLS):
        present |= 1 << bb.get_cell(board, i)
    top = bb.max_exponent(board)
    mask = 0
    for k in range(1, detail + 1):
        if top - k >= 1 and present & (1 << (top - k)):
            mask |= 1 << (k - 1)
    return top * KEY_STRIDE + mask


def point_name(key: int, detail: int) -> str:
    """'16384 + 8192 + 2048' for a key."""
    top, mask = divmod(int(key), KEY_STRIDE)
    tiles = [1 << top] + [1 << (top - k) for k in range(1, detail + 1) if mask & (1 << (k - 1))]
    return " + ".join(str(tile) for tile in tiles)


def stage_progress(key: int, detail: int) -> float:
    """How far through its stage a point is, from 0 (only the largest tile) to just under 1."""
    top, mask = divmod(int(key), KEY_STRIDE)
    return sum(1 / (1 << k) for k in range(1, detail + 1) if mask & (1 << (k - 1)))


@njit
def _best_action(values: np.ndarray) -> int:
    best = -1
    for action in range(bb.NUM_ACTIONS):
        if not np.isnan(values[action]) and (best < 0 or values[action] > values[best]):
            best = action
    return best


@njit(nogil=True)
def point_boards(weights: np.ndarray, layout: TupleLayout, low: int, high: int, detail: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    """Play a seeded greedy game, dealt as play_seeded deals it, and record the first board of each point whose
    largest tile is from 2**low to 2**high.

    Returns (keys, boards, moves, scores so far, points recorded, final score, final max tile). Each board is the one
    the next decision is made from, after the spawn that brought the game to the point.
    """
    keys = np.zeros(MAX_POINTS, dtype=np.int64)
    boards = np.zeros(MAX_POINTS, dtype=np.uint64)
    moves_at = np.zeros(MAX_POINTS, dtype=np.int64)
    scores_at = np.zeros(MAX_POINTS, dtype=np.int64)
    seen = np.zeros((bb.MAX_EXPONENT + 2) * KEY_STRIDE, dtype=np.bool_)
    n = 0
    bb.spawn_tile_seeded(bb.spawn_tile_seeded(np.uint64(0), rng), rng)
    board = bb.spawn_tile_seeded(bb.spawn_tile_seeded(np.uint64(0), rng), rng)
    table = new_transpositions()
    score = moves = 0
    while True:
        top = bb.max_exponent(board)
        if low <= top <= high and n < MAX_POINTS:
            key = point_key(board, detail)
            if not seen[key]:
                seen[key] = True
                keys[n], boards[n], moves_at[n], scores_at[n] = key, board, moves, score
                n += 1
        best = _best_action(_search(weights, layout, board, 1, 0.0, table))
        if best < 0:
            return keys, boards, moves_at, scores_at, n, score, 1 << bb.max_exponent(board)
        board, reward = bb.move(board, best)
        score += reward
        moves += 1
        board = bb.spawn_tile_seeded(board, rng)
        if has_two_top_tiles(board):
            return keys, boards, moves_at, scores_at, n, score, 1 << (bb.MAX_EXPONENT + 1)


@njit(nogil=True)
def play_on(weights: np.ndarray, layout: TupleLayout, start: np.uint64, depth: int, cutoff: float, rng: np.random.Generator) -> tuple[int, int, int]:
    """Play on from `start` to the end, searching `depth` moves ahead, and return (score gained, max tile, moves)."""
    board = np.uint64(start)
    score = moves = 0
    table = new_transpositions()
    while True:
        best = _best_action(_search(weights, layout, board, depth, cutoff, table))
        if best < 0:
            return score, 1 << bb.max_exponent(board), moves
        board, reward = bb.move(board, best)
        score += reward
        moves += 1
        board = bb.spawn_tile_seeded(board, rng)
        if has_two_top_tiles(board):
            return score, 1 << (bb.MAX_EXPONENT + 1), moves


@njit(nogil=True)
def play_to(weights: np.ndarray, layout: TupleLayout, start: np.uint64, depth: int, cutoff: float, rng: np.random.Generator, goal: int) -> tuple[int, int, int, float]:
    """Play on from `start`, searching `depth` moves ahead, until the board holds a 2**goal tile or the game ends.

    Returns (score gained, max tile, moves, the network's value of the board it arrived at): that value is the best
    reward-plus-afterstate value over its moves, the network's own estimate of the score still to come from there,
    and 0 if the game ended first.
    """
    board = np.uint64(start)
    score = moves = 0
    table = new_transpositions()
    while True:
        if bb.max_exponent(board) >= goal:
            values = _search(weights, layout, board, 1, 0.0, table)
            best = _best_action(values)
            return score, 1 << bb.max_exponent(board), moves, 0.0 if best < 0 else float(values[best])
        best = _best_action(_search(weights, layout, board, depth, cutoff, table))
        if best < 0:
            return score, 1 << bb.max_exponent(board), moves, 0.0
        board, reward = bb.move(board, best)
        score += reward
        moves += 1
        board = bb.spawn_tile_seeded(board, rng)
