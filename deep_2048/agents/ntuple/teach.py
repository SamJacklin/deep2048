"""Teaching a network its own search's play across a window of the game, in its own weights.

The stage 1 search-gain map found that greedy play loses about a fifth of its games between 4096 and 16384 and that
searching 3 moves ahead saves most of them, with the gain spread across the whole stretch. This teaches that stretch:

- The window: boards whose largest tile is from 2**low to 2**high (4096 to 8192 for stage 1). An attempt starts from a
  real board as a game enters the window and plays until it leaves it (makes 2**(high + 1)) or the game ends.
- The teacher is a frozen copy of the network before teaching, searching `depth` moves ahead. At each decision every
  legal move's afterstate is moved towards the value the teacher's search gives it (an expectation over spawns, so
  the targets carry no spawn luck), as the finisher does.
- The student is the network itself: its own weights are taught, not a separate patch, so the result is a single
  network that plays the whole game. Weights are shared across stages of the game, so teaching the window also moves
  play elsewhere; whole-game evaluations decide whether to keep the result.
- A share of attempts follow the student's own greedy move rather than the teacher's, so it is also taught on boards
  its own play reaches. Each weight's step is scaled by temporal coherence, from fresh sums.
- With `centred`, each decision's targets are shifted together so that their mean error is zero: the student learns
  only how the teacher ranks the moves, not the teacher's level. Without it, the first run collapsed (greedy play from
  4096 made 16384 0.2% of the time after 1,500 attempts, from 78.7%): the search's values sit at a different level
  from the network's, every visited afterstate was pulled towards that level and every unvisited one wasn't, so play
  drifted to the unvisited ones, and the shared weights carried the damage across the whole game.
"""

import numpy as np
from numba import njit

from deep_2048.agents.ntuple.network import TupleLayout, evaluate, fill_weight_indices
from deep_2048.agents.ntuple.search import _search, has_two_top_tiles, new_transpositions
from deep_2048.game import bitboard as bb


@njit
def in_stretch(board: np.uint64, low: int, high: int) -> bool:
    """Whether the board's largest tile is from 2**low to 2**high."""
    top = bb.max_exponent(board)
    return low <= top <= high


@njit
def _value(weights: np.ndarray, layout: TupleLayout, afterstate: np.uint64, state: np.uint64) -> float:
    return evaluate(weights, layout, afterstate, state)


@njit
def _learn(weights: np.ndarray, errors: np.ndarray, abs_errors: np.ndarray, layout: TupleLayout, afterstate: np.uint64, state: np.uint64, target: float, learning_rate: float,
           indices: np.ndarray) -> None:
    """Move the afterstate's value towards target, each weight's step scaled by its temporal coherence."""
    error = target - _value(weights, layout, afterstate, state)
    for k in range(fill_weight_indices(layout, afterstate, state, indices)):
        i = indices[k]
        coherence = abs(errors[i]) / abs_errors[i] if abs_errors[i] > 0 else 1.0
        weights[i] += learning_rate * coherence * error
        errors[i] += error
        abs_errors[i] += abs(error)


@njit(nogil=True)
def teach_attempts(weights: np.ndarray, errors: np.ndarray, abs_errors: np.ndarray, teacher: np.ndarray, layout: TupleLayout, starts: np.ndarray, attempts: int, depth: int,
                   cutoff: float, learning_rate: float, low: int, high: int, student_share: float, seed: int, centred: bool = True) -> tuple[int, int, int, int, int]:
    """Play `attempts` attempts at the window from boards drawn from `starts`, teaching every decision, and return
    (teacher-led attempts, those that left the window by making 2**(high + 1), student-led attempts, those that did,
    moves played). Releases the GIL, so several threads can teach the same weights at once without locks."""
    np.random.seed(seed)
    table = new_transpositions()
    indices = np.empty(len(layout.offsets) * bb.NUM_SYMMETRIES + 1, dtype=np.int64)
    led = np.zeros(2, dtype=np.int64)
    made = np.zeros(2, dtype=np.int64)
    moves = 0
    for _ in range(attempts):
        student = 1 if np.random.random() < student_share else 0
        led[student] += 1
        board = starts[np.random.randint(starts.size)]
        while in_stretch(board, low, high):
            values = _search(teacher, layout, board, depth, cutoff, table)
            teacher_best = -1
            student_best = -1
            student_value = -np.inf
            for action in range(bb.NUM_ACTIONS):
                if np.isnan(values[action]):
                    continue
                if teacher_best < 0 or values[action] > values[teacher_best]:
                    teacher_best = action
                afterstate, reward = bb.move(board, action)
                own = reward + _value(weights, layout, afterstate, board)
                if own > student_value:
                    student_best, student_value = action, own
            if teacher_best < 0:
                break
            shift = 0.0
            if centred:
                legal = 0
                for action in range(bb.NUM_ACTIONS):
                    if np.isnan(values[action]):
                        continue
                    afterstate, reward = bb.move(board, action)
                    shift += values[action] - reward - _value(weights, layout, afterstate, board)
                    legal += 1
                shift /= legal
            for action in range(bb.NUM_ACTIONS):
                if np.isnan(values[action]):
                    continue
                afterstate, reward = bb.move(board, action)
                _learn(weights, errors, abs_errors, layout, afterstate, board, values[action] - reward - shift, learning_rate, indices)
            board, _ = bb.move(board, student_best if student == 1 else teacher_best)
            board = bb.spawn_tile(board)
            moves += 1
            if has_two_top_tiles(board):
                break
        if bb.max_exponent(board) > high:
            made[student] += 1
    return led[0], made[0], led[1], made[1], moves


@njit
def _rank_update(weights: np.ndarray, layout: TupleLayout, better: np.uint64, worse: np.uint64, state: np.uint64, change: float, indices: np.ndarray,
                 others: np.ndarray) -> None:
    """Raise `better`'s value over `worse`'s by `change`, moving only the weights the two boards don't share.

    Each weight moves by c times (its reads by `better` less its reads by `worse`), so a weight both read equally
    doesn't move at all, and c is set so the gap between the two values grows by exactly `change`.
    """
    n = fill_weight_indices(layout, better, state, indices)
    m = fill_weight_indices(layout, worse, state, others)
    a = np.sort(indices[:n])
    b = np.sort(others[:m])
    unique = np.empty(n + m, dtype=np.int64)
    counts = np.empty(n + m, dtype=np.int64)
    u = 0
    i = j = 0
    while i < n or j < m:
        index = a[i] if j >= m or (i < n and a[i] <= b[j]) else b[j]
        count = 0
        while i < n and a[i] == index:
            count += 1
            i += 1
        while j < m and b[j] == index:
            count -= 1
            j += 1
        if count != 0:
            unique[u], counts[u] = index, count
            u += 1
    total = 0.0
    for k in range(u):
        total += counts[k] * counts[k]
    if total == 0:
        return
    c = change / total
    for k in range(u):
        weights[unique[k]] += c * counts[k]


@njit(nogil=True)
def teach_rank_attempts(weights: np.ndarray, teacher: np.ndarray, layout: TupleLayout, starts: np.ndarray, attempts: int, depth: int, cutoff: float, step: float,
                        margin: float, low: int, high: int, student_share: float, seed: int) -> tuple[int, int, int, int, int, int]:
    """Teach only the move choice: wherever the student's greedy move differs from the teacher's search's, raise the
    teacher's move over the student's by `step` of what it takes to put it `margin` ahead, and nothing else.

    Returns (teacher-led attempts, those that left the window by making 2**(high + 1), student-led attempts, those that
    did, decisions, decisions where the two disagreed). Releases the GIL like teach_attempts.
    """
    np.random.seed(seed)
    table = new_transpositions()
    indices = np.empty(len(layout.offsets) * bb.NUM_SYMMETRIES + 1, dtype=np.int64)
    others = np.empty(len(layout.offsets) * bb.NUM_SYMMETRIES + 1, dtype=np.int64)
    led = np.zeros(2, dtype=np.int64)
    made = np.zeros(2, dtype=np.int64)
    decisions = disagreed = 0
    for _ in range(attempts):
        student = 1 if np.random.random() < student_share else 0
        led[student] += 1
        board = starts[np.random.randint(starts.size)]
        while in_stretch(board, low, high):
            values = _search(teacher, layout, board, depth, cutoff, table)
            teacher_best = -1
            student_best = -1
            student_value = -np.inf
            teacher_own = 0.0
            for action in range(bb.NUM_ACTIONS):
                if np.isnan(values[action]):
                    continue
                if teacher_best < 0 or values[action] > values[teacher_best]:
                    teacher_best = action
                afterstate, reward = bb.move(board, action)
                own = reward + _value(weights, layout, afterstate, board)
                if own > student_value:
                    student_best, student_value = action, own
            if teacher_best < 0:
                break
            decisions += 1
            if student_best != teacher_best:
                disagreed += 1
                better, better_reward = bb.move(board, teacher_best)
                worse, worse_reward = bb.move(board, student_best)
                teacher_own = better_reward + _value(weights, layout, better, board)
                need = student_value - teacher_own + margin
                if need > 0:
                    _rank_update(weights, layout, better, worse, board, step * need, indices, others)
            board, _ = bb.move(board, student_best if student == 1 else teacher_best)
            board = bb.spawn_tile(board)
            if has_two_top_tiles(board):
                break
        if bb.max_exponent(board) > high:
            made[student] += 1
    return led[0], made[0], led[1], made[1], decisions, disagreed
