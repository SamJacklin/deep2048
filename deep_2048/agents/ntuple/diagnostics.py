"""Late-game diagnostics: record positions from play, roll games out from them, and count training's weight updates.

Positions are recorded from whole seeded games played by greedy play or by search, as play_seeded plays them:
- the first board holding an 8192, and the first holding a 16384 (after the spawn, so the next decision is made from it);
- every SAMPLE_EVERY-th board, with how close greedy play's top two moves are.

Rollouts play on from a board or an afterstate with a given depth (1 is greedy play), drawing spawns from the rng they
are given, so rollouts that share a seed share their spawns until their moves differ (common random numbers).
"""

import numpy as np
from numba import njit, types
from numba.typed import List

from deep_2048.agents.ntuple.network import TupleLayout, evaluate, weight_indices
from deep_2048.agents.ntuple.search import _search, has_two_top_tiles, new_transpositions, play_from
from deep_2048.game import bitboard as bb

FIRST_8192, FIRST_16384, SAMPLE = 0, 1, 2
SAMPLE_EVERY = 50
LATE_EXPONENT = 14


@njit
def greedy_values(weights: np.ndarray, layout: TupleLayout, board: np.uint64) -> np.ndarray:
    """Return each move's reward + afterstate value, as greedy play and training judge it, or NaN if it's illegal."""
    values = np.full(bb.NUM_ACTIONS, np.nan)
    for action in range(bb.NUM_ACTIONS):
        afterstate, reward = bb.move(board, action)
        if afterstate != board:
            values[action] = reward + evaluate(weights, layout, afterstate, board)
    return values


@njit
def relative_margin(values: np.ndarray) -> float:
    """Return how far the best value is ahead of the second, as a share of the best (inf with one legal move)."""
    best = second = -np.inf
    for value in values:
        if np.isnan(value):
            continue
        if value > best:
            best, second = value, best
        elif value > second:
            second = value
    return np.inf if second == -np.inf else (best - second) / max(abs(best), 1.0)


@njit(nogil=True)
def record_game(weights: np.ndarray, layout: TupleLayout, depth: int, cutoff: float, rng: np.random.Generator) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Play a seeded game as play_seeded does and return (score, max tile, kinds, moves, scores, boards, margins).

    Each record is a board the game decided from, the moves made and score reached before it, and greedy play's
    relative margin there. Kinds are FIRST_8192, FIRST_16384 or SAMPLE.
    """
    bb.spawn_tile_seeded(bb.spawn_tile_seeded(np.uint64(0), rng), rng)
    board = bb.spawn_tile_seeded(bb.spawn_tile_seeded(np.uint64(0), rng), rng)
    kinds = List.empty_list(types.int64)
    moves = List.empty_list(types.int64)
    scores = List.empty_list(types.int64)
    boards = List.empty_list(types.uint64)
    margins = List.empty_list(types.float64)
    score = 0
    move = 0
    # The next milestone to record: FIRST_8192, then FIRST_16384, then none
    milestone = FIRST_8192
    table = new_transpositions()
    while True:
        top = bb.max_exponent(board)
        recording = []
        while milestone <= FIRST_16384 and top >= 13 + milestone:
            recording.append(milestone)
            milestone += 1
        if move % SAMPLE_EVERY == 0:
            recording.append(SAMPLE)
        if len(recording) > 0:
            margin = relative_margin(greedy_values(weights, layout, board))
            for kind in recording:
                kinds.append(kind)
                moves.append(move)
                scores.append(score)
                boards.append(board)
                margins.append(margin)
        values = _search(weights, layout, board, depth, cutoff, table)
        best = -1
        for action in range(bb.NUM_ACTIONS):
            if not np.isnan(values[action]) and (best < 0 or values[action] > values[best]):
                best = action
        if best < 0:
            break
        board, reward = bb.move(board, best)
        score += reward
        move += 1
        board = bb.spawn_tile_seeded(board, rng)
        if has_two_top_tiles(board):
            break
    final_tile = 1 << (bb.MAX_EXPONENT + 1) if has_two_top_tiles(board) else 1 << bb.max_exponent(board)
    return (
        score,
        final_tile,
        np.asarray(kinds),
        np.asarray(moves),
        np.asarray(scores),
        np.asarray(boards),
        np.asarray(margins),
    )


@njit(nogil=True)
def rollout(weights: np.ndarray, layout: TupleLayout, afterstate: np.uint64, depth: int, cutoff: float, rng: np.random.Generator) -> tuple[int, int]:
    """Spawn a tile into `afterstate` and play on to the end; return (score gained after it, max tile)."""
    return play_from(weights, layout, bb.spawn_tile_seeded(afterstate, rng), depth, cutoff, rng)


@njit(nogil=True)
def count_updates(weights: np.ndarray, layout: TupleLayout, starts: np.ndarray, seed: int, counts: np.ndarray) -> np.ndarray:
    """Play a greedy game from each start (0 for a fresh board), as training plays them, without learning.

    Adds one to `counts` for every weight each chosen afterstate reads, which is the weight updates training makes on
    those games: to row 0 if the board the move is made from has no tile of 2**LATE_EXPONENT (16384) or larger, else
    to row 1. With downgrading, row 0's share of a weight's updates is how much of its training it shares with
    earlier-stage boards. Threads may share `counts`: a lost increment when two collide is rare and only undercounts.
    Returns where each fresh game first made an 8192 (0 if it didn't), for building late starts as training does.
    """
    np.random.seed(seed)
    harvested = np.zeros(starts.size, dtype=np.uint64)
    for g in range(starts.size):
        board = bb.initial_board() if starts[g] == np.uint64(0) else starts[g]
        while True:
            best, best_value = board, -np.inf
            for action in range(bb.NUM_ACTIONS):
                afterstate, reward = bb.move(board, action)
                if afterstate != board:
                    value = reward + evaluate(weights, layout, afterstate, board)
                    if value > best_value:
                        best, best_value = afterstate, value
            if best == board:
                break
            stage = 1 if bb.max_exponent(board) >= LATE_EXPONENT else 0
            for i in weight_indices(layout, best, board):
                counts[stage, i] += 1
            board = bb.spawn_tile(best)
            if starts[g] == np.uint64(0) and harvested[g] == np.uint64(0) and bb.max_exponent(board) >= 13:
                harvested[g] = board
    return harvested


@njit(nogil=True)
def boundary_boards(weights: np.ndarray, layout: TupleLayout, depth: int, cutoff: float, rng: np.random.Generator) -> np.ndarray:
    """Replay a seeded game as record_game does and return every board, before the game's first 16384, from which some
    legal move makes one: the decisions where moves differ in whether their afterstate holds a 16384."""
    bb.spawn_tile_seeded(bb.spawn_tile_seeded(np.uint64(0), rng), rng)
    board = bb.spawn_tile_seeded(bb.spawn_tile_seeded(np.uint64(0), rng), rng)
    found = List.empty_list(types.uint64)
    table = new_transpositions()
    while bb.max_exponent(board) < LATE_EXPONENT:
        for action in range(bb.NUM_ACTIONS):
            afterstate, _ = bb.move(board, action)
            if bb.max_exponent(afterstate) >= LATE_EXPONENT:
                found.append(board)
                break
        values = _search(weights, layout, board, depth, cutoff, table)
        best = -1
        for action in range(bb.NUM_ACTIONS):
            if not np.isnan(values[action]) and (best < 0 or values[action] > values[best]):
                best = action
        if best < 0:
            break
        board, _ = bb.move(board, best)
        board = bb.spawn_tile_seeded(board, rng)
    return np.asarray(found) if len(found) > 0 else np.zeros(0, dtype=np.uint64)
