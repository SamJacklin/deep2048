"""The arrival test: when the last step to a 32768 fails, was the position already poor, or did a good one go wrong?

A game "arrives" when it first holds the whole chain 16384, 8192, 4096 and 2048 with nothing larger. From there it
needs one more 2048's worth of tiles. Three things are measured:
- where the chain's tiles sit at the arrival (placement);
- how often continuations from each arrival position make the 32768, so positions can be told apart by their chances;
- how each continuation goes wrong: the first move that knocks the big tiles out of formation (a break), whether any
  legal move would have kept it (chosen) or none would (forced), and how far the chain had run by then.

Formation: the board's big tiles (2048 and above, at most the four largest) sit in order from a corner along one edge.
That stays true while a cascade merges them, so only moves that displace them count as breaks.

Continuations draw spawns from the rng they are given, so two with the same seed share spawns until their moves differ.
"""

import numpy as np
from numba import njit

from deep_2048.agents.ntuple.network import TupleLayout
from deep_2048.agents.ntuple.search import _search, has_two_top_tiles, new_transpositions, tile_sum
from deep_2048.game import bitboard as bb

CHAIN_TOP = 14
CHAIN_LENGTH = 4
IN_ORDER, CORNER_ONLY, OFF_CORNER = 0, 1, 2
PLACEMENT_NAMES = {IN_ORDER: "in order along one edge", CORNER_ONLY: "largest in a corner, chain out of order", OFF_CORNER: "largest off its corner"}
# The 8 edge lines, each read from a corner outwards
_LINES = np.array([[0, 1, 2, 3], [3, 2, 1, 0], [12, 13, 14, 15], [15, 14, 13, 12], [0, 4, 8, 12], [12, 8, 4, 0], [3, 7, 11, 15], [15, 11, 7, 3]], dtype=np.int64)
_CORNERS = (0, 3, 12, 15)

# The fields trace_from returns, in order
FIELDS = ("converted", "stage_moves", "reach", "break_move", "break_forced", "break_empty", "break_reach", "break_safe_before", "tight_run", "moves_after_break", "start_placement")
CONVERTED, STAGE_MOVES, REACH, BREAK_MOVE, BREAK_FORCED, BREAK_EMPTY, BREAK_REACH, BREAK_SAFE_BEFORE, TIGHT_RUN, MOVES_AFTER_BREAK, START_PLACEMENT = range(len(FIELDS))


@njit
def chain_reach(board: np.uint64) -> int:
    """How many tiles the chain runs to: the largest tile, half of it, a quarter, and so on, until one is missing."""
    present = 0
    for i in range(bb.NUM_CELLS):
        present |= 1 << bb.get_cell(board, i)
    top = bb.max_exponent(board)
    reach = 0
    while top - reach >= 1 and present & (1 << (top - reach)):
        reach += 1
    return reach


@njit
def holds_chain(board: np.uint64, top: int, length: int) -> bool:
    """Whether the board's largest tile is 2**top and it holds every tile of the chain of `length` from it."""
    return bb.max_exponent(board) == top and chain_reach(board) >= length


@njit
def in_formation(board: np.uint64, floor: int, length: int) -> bool:
    """Whether the board's big tiles (exponent `floor` and above, at most the `length` largest) sit in order from a
    corner along one edge. A cascade keeps this true: 16384 8192 4096 4096 and 16384 16384 are both in formation."""
    big = np.zeros(bb.NUM_CELLS, dtype=np.int64)
    n = 0
    for i in range(bb.NUM_CELLS):
        exponent = bb.get_cell(board, i)
        if exponent >= floor:
            big[n] = exponent
            n += 1
    if n == 0:
        return True
    largest = np.sort(big[:n])[::-1][: min(n, length)]
    for line in range(_LINES.shape[0]):
        matches = True
        for j in range(largest.size):
            if bb.get_cell(board, _LINES[line, j]) != largest[j]:
                matches = False
                break
        if matches:
            return True
    return False


@njit
def placement(board: np.uint64, floor: int, length: int) -> int:
    """IN_ORDER if the big tiles are in formation, else CORNER_ONLY if the largest tile is in a corner, else OFF_CORNER."""
    if in_formation(board, floor, length):
        return IN_ORDER
    top = bb.max_exponent(board)
    for corner in _CORNERS:
        if bb.get_cell(board, corner) == top:
            return CORNER_ONLY
    return OFF_CORNER


@njit
def _best_action(values: np.ndarray) -> int:
    """The first action with the highest value, as play_seeded picks it, or -1 if no move is legal."""
    best = -1
    for action in range(bb.NUM_ACTIONS):
        if not np.isnan(values[action]) and (best < 0 or values[action] > values[best]):
            best = action
    return best


@njit(nogil=True)
def arrival_board(weights: np.ndarray, layout: TupleLayout, depth: int, cutoff: float, top: int, length: int, rng: np.random.Generator) -> tuple[np.uint64, int, int]:
    """Play a seeded game as play_seeded does until it first holds the chain of `length` tiles from 2**top.

    Returns (that board, after the spawn, so the next decision is made from it; moves played; score so far), or a board
    of 0 if the game ends, or makes a larger tile, first.
    """
    bb.spawn_tile_seeded(bb.spawn_tile_seeded(np.uint64(0), rng), rng)
    board = bb.spawn_tile_seeded(bb.spawn_tile_seeded(np.uint64(0), rng), rng)
    table = new_transpositions()
    moves = score = 0
    while not holds_chain(board, top, length):
        if bb.max_exponent(board) > top:
            return np.uint64(0), moves, score
        best = _best_action(_search(weights, layout, board, depth, cutoff, table))
        if best < 0:
            return np.uint64(0), moves, score
        board, reward = bb.move(board, best)
        board = bb.spawn_tile_seeded(board, rng)
        moves += 1
        score += reward
    return board, moves, score


@njit(nogil=True)
def trace_from(weights: np.ndarray, layout: TupleLayout, start: np.uint64, depth: int, cutoff: float, top: int, length: int, rng: np.random.Generator) -> tuple[np.ndarray, np.uint64, np.uint64]:
    """Play on from an arrival board until the game makes a 2**(top + 1) tile or ends, and return (fields, the board
    the first breaking move was made from or 0, the last board).

    The fields, named by FIELDS:
    - converted: 1 if the larger tile was made;
    - stage_moves: moves played before that, or before the game ended;
    - reach: the longest the chain ran during those moves;
    - break_move: the move, counted from the start, that first took the big tiles out of formation, or -1 if none did
      (always -1 when the start itself wasn't in formation);
    - break_forced: 1 if no legal move there kept the formation, 0 if one did and play chose otherwise, -1 if no break;
    - break_empty, break_reach: empty cells and chain reach on the board that move was made from;
    - break_safe_before: how many legal moves kept the formation at the decision before the break (-1 if none);
    - tight_run: how many decisions in a row, up to the break, had at most one legal move that kept the formation;
    - moves_after_break: moves played between the break and the end of the stage;
    - start_placement: IN_ORDER, CORNER_ONLY or OFF_CORNER.
    """
    floor = top - length + 1
    board = np.uint64(start)
    fields = np.full(len(FIELDS), -1, dtype=np.int64)
    fields[CONVERTED] = 0
    fields[START_PLACEMENT] = placement(board, floor, length)
    fields[REACH] = chain_reach(board)
    fields[TIGHT_RUN] = 0
    break_board = np.uint64(0)
    tracking = fields[START_PLACEMENT] == IN_ORDER
    tight = 0
    safe_before = -1
    table = new_transpositions()
    moves = 0
    while True:
        best = _best_action(_search(weights, layout, board, depth, cutoff, table))
        if best < 0:
            break
        if tracking:
            safe = 0
            for action in range(bb.NUM_ACTIONS):
                afterstate, _ = bb.move(board, action)
                if afterstate != board and in_formation(afterstate, floor, length):
                    safe += 1
            chosen, _ = bb.move(board, best)
            if not in_formation(chosen, floor, length):
                fields[BREAK_MOVE] = moves
                fields[BREAK_FORCED] = 1 if safe == 0 else 0
                fields[BREAK_EMPTY] = bb.count_empty(board)
                fields[BREAK_REACH] = chain_reach(board)
                fields[BREAK_SAFE_BEFORE] = safe_before
                fields[TIGHT_RUN] = tight
                break_board = board
                tracking = False
            else:
                tight = tight + 1 if safe <= 1 else 0
                safe_before = safe
        board, _ = bb.move(board, best)
        moves += 1
        board = bb.spawn_tile_seeded(board, rng)
        if bb.max_exponent(board) > top:
            fields[CONVERTED] = 1
            break
        fields[REACH] = max(fields[REACH], chain_reach(board))
    fields[STAGE_MOVES] = moves
    if fields[BREAK_MOVE] >= 0:
        fields[MOVES_AFTER_BREAK] = moves - fields[BREAK_MOVE]
    return fields, break_board, board


@njit
def ends_at_once(afterstate: np.uint64) -> float:
    """The chance that the tile spawned into `afterstate` leaves no legal move, so the game ends there."""
    empty = bb.count_empty(afterstate)
    if empty == 0:
        return 0.0
    dead = 0.0
    for cell in range(bb.NUM_CELLS):
        if bb.get_cell(afterstate, cell) != 0:
            continue
        for exponent, chance in ((1, 1.0 - bb.SPAWN_FOUR_PROBABILITY), (2, bb.SPAWN_FOUR_PROBABILITY)):
            board = afterstate | (np.uint64(exponent) << np.uint64(4 * cell))
            if bb.is_game_over(board):
                dead += chance / empty
    return dead


@njit
def safest_keeping_move(board: np.uint64, floor: int, length: int) -> float:
    """Over the legal moves from `board` that keep the formation, the lowest chance that the game ends at once after
    one, or -1 if no legal move keeps it."""
    best = -1.0
    for action in range(bb.NUM_ACTIONS):
        afterstate, _ = bb.move(board, action)
        if afterstate != board and in_formation(afterstate, floor, length):
            chance = ends_at_once(afterstate)
            if best < 0 or chance < best:
                best = chance
    return best


@njit(nogil=True)
def tail_boards(weights: np.ndarray, layout: TupleLayout, start: np.uint64, depth: int, cutoff: float, top: int, offsets: np.ndarray, rng: np.random.Generator) -> tuple[int, int, np.ndarray]:
    """Play on from an arrival board as trace_from does and return (1 if it made the larger tile, moves played, the
    boards decided from `offsets` moves before the last move of the stage, 0 where the stage was shorter than that).

    Given the same rng as trace_from, it is the same game. An offset of 1 is the board the last move was made from.
    """
    size = int(offsets.max()) + 1
    ring = np.zeros(size, dtype=np.uint64)
    board = np.uint64(start)
    table = new_transpositions()
    moves = 0
    converted = 0
    while True:
        best = _best_action(_search(weights, layout, board, depth, cutoff, table))
        if best < 0:
            break
        ring[moves % size] = board
        board, _ = bb.move(board, best)
        moves += 1
        board = bb.spawn_tile_seeded(board, rng)
        if bb.max_exponent(board) > top:
            converted = 1
            break
    found = np.zeros(offsets.size, dtype=np.uint64)
    for k in range(offsets.size):
        if offsets[k] <= moves:
            found[k] = ring[(moves - offsets[k]) % size]
    return converted, moves, found


@njit(nogil=True)
def converts(weights: np.ndarray, layout: TupleLayout, start: np.uint64, depth: int, cutoff: float, top: int, rng: np.random.Generator) -> int:
    """Play on from `start` until the game makes a 2**(top + 1) tile (1) or ends (0)."""
    board = np.uint64(start)
    table = new_transpositions()
    while True:
        best = _best_action(_search(weights, layout, board, depth, cutoff, table))
        if best < 0:
            return 0
        board, _ = bb.move(board, best)
        board = bb.spawn_tile_seeded(board, rng)
        if bb.max_exponent(board) > top:
            return 1


@njit(nogil=True)
def finish_board(weights: np.ndarray, layout: TupleLayout, start: np.uint64, depth: int, cutoff: float, top: int, margin: int, rng: np.random.Generator) -> np.uint64:
    """Play on from an arrival board until its tiles first sum to within `margin` of 2**(top + 1), and return that
    board (after the spawn, so the next decision is made from it), or 0 if the game ends first.

    Given the same rng as trace_from, it is the same game up to there. Spawns add 2 or 4, so the board returned is
    the first one at or past the mark, and the larger tile can't have been made yet while margin is above 0.
    """
    target = (1 << (top + 1)) - margin
    board = np.uint64(start)
    table = new_transpositions()
    while tile_sum(board) < target:
        best = _best_action(_search(weights, layout, board, depth, cutoff, table))
        if best < 0:
            return np.uint64(0)
        board, _ = bb.move(board, best)
        board = bb.spawn_tile_seeded(board, rng)
        if bb.max_exponent(board) > top:
            return np.uint64(0)
    return board


@njit(nogil=True)
def play_with_finish(weights: np.ndarray, layout: TupleLayout, start: np.uint64, depth: int, deep_depth: int, deep_cutoff: float, top: int, margin: int, stop_at_goal: bool,
                     rng: np.random.Generator) -> tuple[int, int, int, int]:
    """Play on from `start` searching `depth` moves ahead, as play_from does, except in the last stretch before a
    2**(top + 1) tile: while the largest tile is 2**top and the tiles sum to within `margin` of 2**(top + 1), each
    move is searched `deep_depth` ahead, pruning spawn branches less likely than `deep_cutoff`.

    Returns (score gained, max tile, moves, moves searched deep). With stop_at_goal the game stops once the larger
    tile is made. With deep_depth == depth and deep_cutoff == 0 it is exactly play_from's game.
    """
    target = (1 << (top + 1)) - margin
    board = np.uint64(start)
    score = 0
    moves = 0
    deep_moves = 0
    table = new_transpositions()
    while True:
        deep = bb.max_exponent(board) == top and tile_sum(board) >= target
        if deep:
            values = _search(weights, layout, board, deep_depth, deep_cutoff, table)
        else:
            values = _search(weights, layout, board, depth, 0.0, table)
        best = _best_action(values)
        if best < 0:
            return score, 1 << bb.max_exponent(board), moves, deep_moves
        board, reward = bb.move(board, best)
        score += reward
        moves += 1
        deep_moves += 1 if deep else 0
        board = bb.spawn_tile_seeded(board, rng)
        if has_two_top_tiles(board):
            return score, 1 << (bb.MAX_EXPONENT + 1), moves, deep_moves
        if stop_at_goal and bb.max_exponent(board) > top:
            return score, 1 << bb.max_exponent(board), moves, deep_moves


@njit(nogil=True)
def play_seeded_with_finish(weights: np.ndarray, layout: TupleLayout, depth: int, deep_depth: int, deep_cutoff: float, top: int, margin: int, rng: np.random.Generator) -> tuple[int, int, int, int]:
    """A whole seeded game, dealt as play_seeded deals it, played by play_with_finish to the end."""
    bb.spawn_tile_seeded(bb.spawn_tile_seeded(np.uint64(0), rng), rng)
    board = bb.spawn_tile_seeded(bb.spawn_tile_seeded(np.uint64(0), rng), rng)
    return play_with_finish(weights, layout, board, depth, deep_depth, deep_cutoff, top, margin, False, rng)
