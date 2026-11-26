"""The finisher: a second set of weights for the last stretch before a tile, taught by the network's own deeper search.

The arrival test found that a game holding 16384, 8192, 4096 and 2048 fails to make the 32768 three times in four,
most often within the last few dozen moves, on a nearly full board, and that from boards within 64 of the 32768's tile
sum the same network makes it 44% of the time playing greedily but 71% of the time searching 8 moves ahead. The
finisher teaches greedy play that foresight, for one window of the game and nothing else:

- The window: boards whose largest tile is 2**top and whose tiles sum to within `margin` of 2**(top + 1). Tiles only
  ever sum to more, so a game enters it once and leaves when the larger tile is made (or the game ends).
- A second stage of weights plays the window, chosen by the board a move is made from, as every second stage here is.
  It starts as a copy of the base, so an untrained finisher plays exactly as the base does. Outside the window the
  base plays, and its weights are never touched.
- The teacher is the frozen base searching `depth` moves ahead (pruning spawn branches less likely than `cutoff`).
  Training starts every attempt from a real board at the window's edge and plays to its end. At each decision every
  legal move's afterstate is moved towards the value the teacher's search gave it. The targets are expectations over
  spawns rather than one sampled spawn, so they carry no spawn luck as far as the search goes.
- The teacher never changes, so the targets don't move as the finisher learns, and they stay on the base's own scale
  of values. (Teaching from a search of the finisher's own weights was tried first on a small network and made play
  worse: positions it had visited were revalued and the rest weren't, so play drifted to the unvisited ones.)
- A share of attempts follow the finisher's own greedy move rather than the teacher's, so it is also taught on the
  boards its own play reaches. Either way the targets are the teacher's.
- Optionally (`centred`) each decision's targets are shifted by one amount so that their mean error is zero: the
  finisher is then taught only how the teacher ranks the moves, and its overall level stays the base's.
- Each weight's step is scaled by temporal coherence, from fresh sums kept only for the second stage.

- Optionally (`area`) the window's stage also reads a work-area table: the half of the board the large tiles have left
  free, as one 8-cell pattern (network.AreaLayout). With `td` the finisher learns by plain TD(0) from its own greedy
  play, with no teacher.

A trained finisher is stored as a patch: the second-stage weights that differ from the base, with the base
checkpoint's path and SHA-256. Loading refuses a base that has changed.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
from numba import njit

from deep_2048.agents.ntuple.network import AREA_WEIGHTS, AreaLayout, NTupleNetwork, TupleLayout, evaluate, fill_indices
from deep_2048.agents.ntuple.search import _search, new_transpositions, tile_sum
from deep_2048.bench import file_sha256
from deep_2048.game import bitboard as bb


# Room for every weight one board reads: far more tuples than any layout here has, in 8 symmetries, plus a work area
MAX_READS = 512


@njit
def in_window(board: np.uint64, top: int, low: int) -> bool:
    """Whether the board's largest tile is 2**top and its tiles sum to at least `low`."""
    return bb.max_exponent(board) == top and tile_sum(board) >= low


@njit
def _value(weights: np.ndarray, layout: TupleLayout, afterstate: np.uint64, state: np.uint64) -> float:
    """The network's value of the afterstate. A call rather than inlined, which numba's inliner mishandles when one
    function values several boards."""
    return evaluate(weights, layout, afterstate, state)


@njit
def _learn(weights: np.ndarray, errors: np.ndarray, abs_errors: np.ndarray, layout: TupleLayout, afterstate: np.uint64, state: np.uint64, target: float, learning_rate: float,
           first: int, indices: np.ndarray, learn_from: int = 0) -> None:
    """Move the afterstate's value towards target, each weight's step scaled by its temporal coherence.

    `state` is in the window, so every weight read is a second-stage one (from index `first` on, apart from a scale
    weight, which is left alone), and its sums are at its index less `first`. Weights below `learn_from` are left
    alone too.
    """
    error = target - _value(weights, layout, afterstate, state)
    for k in range(fill_indices(layout, afterstate, state, indices)):
        i = indices[k] - first
        if i < 0 or i >= errors.size or indices[k] < learn_from:
            continue
        coherence = abs(errors[i]) / abs_errors[i] if abs_errors[i] > 0 else 1.0
        weights[indices[k]] += learning_rate * coherence * error
        errors[i] += error
        abs_errors[i] += abs(error)


@njit
def mean_difference(weights: np.ndarray, layout: TupleLayout, board: np.uint64, values: np.ndarray) -> float:
    """The mean, over the board's legal moves, of the teacher's value of the move less the network's (`values` holds
    the teacher's, NaN for a move that isn't legal)."""
    total = 0.0
    legal = 0
    for action in range(bb.NUM_ACTIONS):
        if np.isnan(values[action]):
            continue
        afterstate, reward = bb.move(board, action)
        total += values[action] - reward - _value(weights, layout, afterstate, board)
        legal += 1
    return total / legal if legal else 0.0


@njit
def _greedy(weights: np.ndarray, layout: TupleLayout, board: np.uint64) -> tuple[int, np.uint64, int]:
    """(the network's greedy move or -1, its afterstate, its reward)."""
    best, best_value, best_after, best_reward = -1, -np.inf, board, 0
    for action in range(bb.NUM_ACTIONS):
        afterstate, reward = bb.move(board, action)
        if afterstate == board:
            continue
        value = reward + _value(weights, layout, afterstate, board)
        if value > best_value:
            best, best_value, best_after, best_reward = action, value, afterstate, reward
    return best, best_after, best_reward


@njit
def _spawn_drawn(board: np.uint64, position: float, four: float) -> np.uint64:
    """Spawn as spawn_tile does, from two uniform draws given rather than taken, so moves can share them."""
    empty = bb._empty_cells(board)
    n_empty = bb._count_cells(empty)
    if n_empty == 0:
        return board
    exponent = np.uint64(1) if four < 1.0 - bb.SPAWN_FOUR_PROBABILITY else np.uint64(2)
    return bb._place_in_empty(board, empty, min(int(position * n_empty), n_empty - 1), exponent)


@njit
def _rollout(weights: np.ndarray, layout: TupleLayout, afterstate: np.uint64, state: np.uint64, draws: np.ndarray) -> float:
    """The rewards of greedy play for len(draws) moves from the afterstate, plus the value after them (0 if the game ends)."""
    total = 0.0
    for t in range(draws.shape[0]):
        board = _spawn_drawn(afterstate, draws[t, 0], draws[t, 1])
        action, after, reward = _greedy(weights, layout, board)
        if action < 0:
            return total
        total += reward
        afterstate, state = after, board
    return total + _value(weights, layout, afterstate, state)


@njit
def rollout_teacher(weights: np.ndarray, layout: TupleLayout, board: np.uint64, length: int, rollouts: int, z: float) -> tuple[np.ndarray, int]:
    """(each move's value: its reward plus the mean of `rollouts` rollouts of `length` greedy moves, NaN if illegal;
    the teacher's move). Every move's rollouts share the same spawn draws. The teacher plays the network's greedy move
    unless another beats it by more than z standard errors of the paired difference: the mean alone picks among
    near-ties at random, since the value after a short rollout barely tells safe positions from doomed ones."""
    greedy_move, _, _ = _greedy(weights, layout, board)
    values = np.full(bb.NUM_ACTIONS, np.nan)
    if greedy_move < 0:
        return values, -1
    draws = np.random.random((rollouts, length, 2))
    totals = np.full((bb.NUM_ACTIONS, rollouts), np.nan)
    for action in range(bb.NUM_ACTIONS):
        afterstate, reward = bb.move(board, action)
        if afterstate == board:
            continue
        for j in range(rollouts):
            totals[action, j] = reward + _rollout(weights, layout, afterstate, board, draws[j])
        values[action] = totals[action].mean()
    best, margin = greedy_move, 0.0
    for action in range(bb.NUM_ACTIONS):
        if action == greedy_move or np.isnan(values[action]):
            continue
        difference = totals[action] - totals[greedy_move]
        mean = difference.mean()
        if mean > margin and mean > z * difference.std() / np.sqrt(rollouts):
            best, margin = action, mean
    return values, best


@njit
def _td_attempt(weights: np.ndarray, errors: np.ndarray, abs_errors: np.ndarray, layout: TupleLayout, board: np.uint64, learning_rate: float, top: int, low: int, first: int,
                learn_from: int, indices: np.ndarray) -> tuple[int, int]:
    """One window attempt by the finisher's own greedy play, learning by TD(0) with no teacher: (made the larger tile,
    moves played). Each afterstate's value moves towards the reward and value of the move played next, 0 if the game
    ends. The move that follows the window is valued by the base, which plays from there."""
    moves = 0
    learning = False
    previous, previous_state = board, board
    while True:
        action, afterstate, reward = _greedy(weights, layout, board)
        if learning:
            target = 0.0 if action < 0 else reward + _value(weights, layout, afterstate, board)
            _learn(weights, errors, abs_errors, layout, previous, previous_state, target, learning_rate, first, indices, learn_from)
        if action < 0 or not in_window(board, top, low):
            return int(bb.max_exponent(board) > top), moves
        previous, previous_state, learning = afterstate, board, True
        board = bb.spawn_tile(afterstate)
        moves += 1


@njit(nogil=True)
def train_attempts(weights: np.ndarray, errors: np.ndarray, abs_errors: np.ndarray, layout: TupleLayout, base_layout: TupleLayout, starts: np.ndarray, attempts: int, depth: int,
                   cutoff: float, learning_rate: float, top: int, low: int, first: int, student_share: float, centred: bool, seed: int,
                   mistakes: bool = False, rollout_length: int = 0, rollouts: int = 32, z: float = 2.0, td: bool = False, learn_from: int = 0) -> tuple[int, int, int, int, int]:
    """Play `attempts` window attempts from boards drawn from `starts`, learning from the teacher's search at every
    decision, and return (teacher-led attempts, those that made the larger tile, student-led attempts, those that
    made it, moves played).

    With `centred`, the targets at each decision are shifted together so that their mean error is zero.
    With `mistakes`, nothing is learned from a decision where the finisher already picks the teacher's move. Where it
    picks another, its value of the teacher's move is raised and of its own pick lowered, half each, until the gap
    between them is the teacher's (so a costly mistake moves the weights most). Value matching changes the weights at
    every decision, and through weights boards share, it undoes choices the base made right; this leaves them alone.
    The teacher is the base, the first `first` weights read through `base_layout`, which nothing here changes. An
    attempt is student-led with chance `student_share`: it then plays the finisher's own greedy move at each decision.
    With `rollout_length` > 0 the teacher is rollout_teacher (the base's own greedy play, not its search), whose move
    is good given that greedy play follows; the search teacher's is good only if its search plays on.
    With `td` there is no teacher: every attempt is the finisher's own greedy play, learning by TD(0) (_td_attempt).
    Weights below `learn_from` never change (so a work-area table can learn alone, over a stage left as the base).
    Releases the GIL, so several threads can train the same weights at once without locks, as training does elsewhere.
    """
    np.random.seed(seed)
    base_weights = weights[:first]
    table = new_transpositions()
    indices = np.empty(MAX_READS, dtype=np.int64)
    led = np.zeros(2, dtype=np.int64)
    made = np.zeros(2, dtype=np.int64)
    moves = 0
    for _ in range(attempts):
        student = 1 if td or np.random.random() < student_share else 0
        led[student] += 1
        board = starts[np.random.randint(starts.size)]
        if td:
            made_it, played = _td_attempt(weights, errors, abs_errors, layout, board, learning_rate, top, low, first, learn_from, indices)
            made[1] += made_it
            moves += played
            continue
        while in_window(board, top, low):
            if rollout_length > 0:
                values, teacher_best = rollout_teacher(base_weights, base_layout, board, rollout_length, rollouts, z)
            else:
                values = _search(base_weights, base_layout, board, depth, cutoff, table)
                teacher_best = -1
            student_best = -1
            student_value = -np.inf
            for action in range(bb.NUM_ACTIONS):
                if np.isnan(values[action]):
                    continue
                if rollout_length == 0 and (teacher_best < 0 or values[action] > values[teacher_best]):
                    teacher_best = action
                afterstate, reward = bb.move(board, action)
                own = reward + _value(weights, layout, afterstate, board)
                if own > student_value:
                    student_best, student_value = action, own
            if teacher_best < 0:
                break
            # Both players have chosen before anything is learned from this decision
            if mistakes:
                if student_best != teacher_best:
                    best_after, best_reward = bb.move(board, teacher_best)
                    own_after, own_reward = bb.move(board, student_best)
                    best_value = _value(weights, layout, best_after, board)
                    own_value = _value(weights, layout, own_after, board)
                    # how far the finisher's gap between the two falls short of the teacher's (positive: it prefers its own)
                    short = (values[teacher_best] - values[student_best]) - (best_reward + best_value - own_reward - own_value)
                    if short > 0:
                        _learn(weights, errors, abs_errors, layout, best_after, board, best_value + short / 2, learning_rate, first, indices, learn_from)
                        _learn(weights, errors, abs_errors, layout, own_after, board, own_value - short / 2, learning_rate, first, indices, learn_from)
                board, _ = bb.move(board, student_best if student == 1 else teacher_best)
                board = bb.spawn_tile(board)
                moves += 1
                continue
            shift = mean_difference(weights, layout, board, values) if centred else 0.0
            for action in range(bb.NUM_ACTIONS):
                if np.isnan(values[action]):
                    continue
                afterstate, reward = bb.move(board, action)
                _learn(weights, errors, abs_errors, layout, afterstate, board, values[action] - reward - shift, learning_rate, first, indices, learn_from)
            board, _ = bb.move(board, student_best if student == 1 else teacher_best)
            board = bb.spawn_tile(board)
            moves += 1
        if bb.max_exponent(board) > top:
            made[student] += 1
    return led[0], made[0], led[1], made[1], moves


@njit(nogil=True)
def fit_to_teacher(weights: np.ndarray, layout: TupleLayout, base_layout: TupleLayout, starts: np.ndarray, depth: int, cutoff: float, top: int, low: int, first: int,
                   seed: int) -> tuple[int, int, float, float, float]:
    """How closely the finisher's one-move choices match the teacher's, along the teacher's own play from each board
    in `starts`: (choices between two or more moves, those where the finisher picks the teacher's move, the mean
    difference between the two's values of a move, the same once each choice's mean difference is taken off, and the
    mean value the teacher puts on its move over the finisher's).

    Nothing is learned. The third number can fall while play stays the same, when the finisher only learns the
    teacher's level; the second and the last are what greedy play depends on.
    """
    np.random.seed(seed)
    base_weights = weights[:first]
    table = new_transpositions()
    differences = np.zeros(bb.NUM_ACTIONS)
    choices = 0
    agreed = 0
    pairs = 0
    error = 0.0
    centred_error = 0.0
    regret = 0.0
    for i in range(starts.size):
        board = starts[i]
        while in_window(board, top, low):
            values = _search(base_weights, base_layout, board, depth, cutoff, table)
            teacher_best = -1
            student_best = -1
            student_value = -np.inf
            legal = 0
            for action in range(bb.NUM_ACTIONS):
                if np.isnan(values[action]):
                    continue
                if teacher_best < 0 or values[action] > values[teacher_best]:
                    teacher_best = action
                afterstate, reward = bb.move(board, action)
                own = reward + _value(weights, layout, afterstate, board)
                differences[legal] = own - values[action]
                legal += 1
                if own > student_value:
                    student_best, student_value = action, own
            if teacher_best < 0:
                break
            if legal > 1:
                mean = differences[:legal].mean()
                for k in range(legal):
                    error += abs(differences[k])
                    centred_error += abs(differences[k] - mean)
                pairs += legal
                choices += 1
                agreed += student_best == teacher_best
                regret += values[teacher_best] - values[student_best]
            board, _ = bb.move(board, teacher_best)
            board = bb.spawn_tile(board)
    return choices, agreed, error / max(pairs, 1), centred_error / max(pairs, 1), regret / max(choices, 1)


class FinisherNetwork:
    """A base network plus a second stage for one window, as weights and a layout every player here can use."""

    def __init__(self, base: NTupleNetwork, base_checkpoint: str | Path, base_sha256: str, top: int, margin: int, progress: dict[str, Any] | None = None,
                 extra: tuple[tuple[int, ...], ...] = (), area: bool = False) -> None:
        if base.stage_tile is not None or base.scale_feature or base.small_tables:
            raise ValueError("The finisher needs a single-stage base without a scale term or small-tile tables")
        if not 0 < margin < 1 << top:
            raise ValueError(f"margin must be between 1 and {(1 << top) - 1}, not {margin}")
        self.base_checkpoint, self.base_sha256 = str(base_checkpoint), base_sha256
        self.top, self.margin, self.low = top, margin, (1 << (top + 1)) - margin
        self.progress = progress or {}
        # `extra` tuples are read after the base's, from zero weights: in the first stage they stay zero, so outside the
        # window play is the base's; in the window's stage they start at zero too, so an untrained finisher plays as the base
        self.extra = tuple(tuple(t) for t in extra)
        self.base_size = base.weights.size
        zeros = np.zeros(sum(16 ** len(t) for t in self.extra), dtype=base.weights.dtype)
        self.first = self.base_size + zeros.size
        layout = TupleLayout.build(tuple(base.tuples) + self.extra, base.downgrade_tile, stage_tile=1 << top, downgrade_repeat=base.downgrade_repeat)
        layout = layout._replace(sharing=layout.sharing._replace(stage_sum=self.low))
        # `area` adds a work-area table after the window's stage, from zero weights, so play still starts as the base's
        self.area = area
        self.weights = np.concatenate([base.weights, zeros, base.weights, zeros, np.zeros(AREA_WEIGHTS if area else 0, dtype=base.weights.dtype)])
        self.layout: TupleLayout | AreaLayout = AreaLayout(layout, 2 * self.first) if area else layout
        self.base_layout = base.layout
        self.tuples, self.downgrade_tile = base.tuples, base.downgrade_tile

    @classmethod
    def from_base(cls, base_checkpoint: str | Path, top: int, margin: int, extra: tuple[tuple[int, ...], ...] = (), area: bool = False) -> "FinisherNetwork":
        base, _ = NTupleNetwork.load(base_checkpoint)
        return cls(base, base_checkpoint, file_sha256(Path(base_checkpoint)), top, margin, extra=extra, area=area)

    def base_unchanged(self, base: NTupleNetwork) -> bool:
        return bool(np.array_equal(self.weights[: self.base_size], base.weights)) and not self.weights[self.base_size : self.first].any()

    def patch(self) -> tuple[np.ndarray, np.ndarray]:
        """The second-stage weights that differ from the base, and the work-area weights that aren't zero: (their
        indices from the stage's start, their values)."""
        stage = self.weights[self.first :]
        changed = np.concatenate([np.flatnonzero(stage[: self.first] != self.weights[: self.first]), self.first + np.flatnonzero(stage[self.first :])])
        return changed.astype(np.uint32), stage[changed]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        indices, values = self.patch()
        about = {"base_checkpoint": self.base_checkpoint, "base_sha256": self.base_sha256, "top": self.top, "margin": self.margin, "progress": self.progress} | ({"extra": [list(t) for t in self.extra]} if self.extra else {}) | ({"area": True} if self.area else {})
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as f:
            np.savez(f, finisher=json.dumps(about), indices=indices, values=values)
        tmp.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> "FinisherNetwork":
        with np.load(path) as data:
            about = json.loads(str(data["finisher"]))
            indices, values = data["indices"], data["values"]
        if file_sha256(Path(about["base_checkpoint"])) != about["base_sha256"]:
            raise ValueError(f"{about['base_checkpoint']} has changed since {path} was made from it")
        base, _ = NTupleNetwork.load(about["base_checkpoint"])
        network = cls(base, about["base_checkpoint"], about["base_sha256"], about["top"], about["margin"], about["progress"], extra=tuple(map(tuple, about.get("extra", []))), area=about.get("area", False))
        network.weights[network.first + indices.astype(np.int64)] = values
        return network


def combine_windows(path: str | Path, second_path: str | Path) -> tuple[np.ndarray, TupleLayout | AreaLayout]:
    """Two finishers on the same base as one player: (weights, layout). The second finisher's window must lie below
    the first's (a smaller largest tile), so the two never overlap; outside both, the base plays. Either may have a
    work-area table: the weights are then the base, the two windows' stages, and the tables after them."""
    first = FinisherNetwork.load(path)
    with np.load(second_path) as data:
        about = json.loads(str(data["finisher"]))
        indices, values = data["indices"], data["values"]
    if about["base_sha256"] != first.base_sha256 or first.extra or about.get("extra"):
        raise ValueError("Both finishers must patch the same base, without extra tuples")
    if about["top"] >= first.top:
        raise ValueError("The second finisher's window must be below the first's")
    indices = indices.astype(np.int64)
    in_stage = indices < first.first  # the rest are the second finisher's work-area weights
    window = first.weights[: first.first].copy()
    window[indices[in_stage]] = values[in_stage]
    low = (1 << (about["top"] + 1)) - about["margin"]
    staged = first.layout.base if first.area else first.layout
    staged = staged._replace(sharing=staged.sharing._replace(window2_exponent=about["top"], window2_sum=low))
    parts = [first.weights[: 2 * first.first], window]
    if not first.area and not about.get("area"):
        return np.concatenate(parts), staged
    area_offset = area2_offset = -1
    if first.area:
        area_offset = 3 * first.first
        parts.append(first.weights[2 * first.first :])
    if about.get("area"):
        area2_offset = 3 * first.first + (AREA_WEIGHTS if first.area else 0)
        table = np.zeros(AREA_WEIGHTS, dtype=first.weights.dtype)
        table[indices[~in_stage] - first.first] = values[~in_stage]
        parts.append(table)
    return np.concatenate(parts), AreaLayout(staged, area_offset, area2_offset)
