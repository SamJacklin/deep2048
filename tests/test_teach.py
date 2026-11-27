import numpy as np

from deep_2048.agents.ntuple import NTupleNetwork
from deep_2048.agents.ntuple.teach import in_stretch, teach_attempts
from deep_2048.game import bitboard as bb

from tests.helpers import SMALL_TUPLES


def board(rows: list[list[int]]) -> np.uint64:
    return np.uint64(bb.from_array(np.array(rows)))


def network(seed: int) -> NTupleNetwork:
    n = NTupleNetwork(tuples=SMALL_TUPLES)
    n.weights[:] = np.random.default_rng(seed).normal(scale=10, size=n.weights.shape).astype(np.float32)
    return n


START = board([[64, 32, 0, 2], [4, 0, 0, 0], [0] * 4, [0] * 4])


def test_the_stretch_is_the_largest_tile_between_low_and_high() -> None:
    assert in_stretch(START, 5, 6) and in_stretch(START, 6, 6)
    assert not in_stretch(START, 7, 9) and not in_stretch(START, 4, 5)


def test_a_teacher_that_agrees_with_the_student_teaches_nothing() -> None:
    student = network(1)
    before = student.weights.copy()
    errors, abs_errors = np.zeros(student.weights.size), np.zeros(student.weights.size)
    led, made, student_led, student_made, moves = teach_attempts(student.weights, errors, abs_errors, before.copy(), student.layout, np.array([START]), 5, 1, 0.0, 0.01, 5, 6,
                                                                 0.5, 3)
    assert led + student_led == 5 and moves > 0
    assert np.allclose(student.weights, before, atol=1e-3)


def test_a_different_teacher_moves_the_student_towards_its_values() -> None:
    student, teacher = network(1), network(2)
    before = student.weights.copy()
    errors, abs_errors = np.zeros(student.weights.size), np.zeros(student.weights.size)
    teach_attempts(student.weights, errors, abs_errors, teacher.weights.copy(), student.layout, np.array([START]), 20, 1, 0.0, 0.01, 5, 6, 0.0, 4)
    touched = student.weights != before
    assert touched.any()
    # with depth 1 the teacher's target for an afterstate is its own value, so touched weights move towards the teacher's
    assert np.abs(student.weights[touched] - teacher.weights[touched]).mean() < np.abs(before[touched] - teacher.weights[touched]).mean()


def test_a_rank_update_moves_only_unshared_weights_and_opens_the_gap_exactly() -> None:
    from deep_2048.agents.ntuple.network import evaluate, weight_indices
    from deep_2048.agents.ntuple.teach import _rank_update

    student = network(5)
    before = student.weights.copy()
    state = START
    better, worse = np.uint64(bb.move(state, 0)[0]), np.uint64(bb.move(state, 2)[0])
    gap = evaluate(student.weights, student.layout, better, state) - evaluate(student.weights, student.layout, worse, state)
    indices = np.empty(student.layout.offsets.size * 8 + 1, dtype=np.int64)
    others = np.empty_like(indices)
    _rank_update(student.weights, student.layout, better, worse, state, 100.0, indices, others)
    after = evaluate(student.weights, student.layout, better, state) - evaluate(student.weights, student.layout, worse, state)
    assert abs((after - gap) - 100.0) < 0.05
    read_by_better = set(weight_indices(student.layout, better, state).tolist())
    read_by_worse = set(weight_indices(student.layout, worse, state).tolist())
    moved = set(np.flatnonzero(student.weights != before).tolist())
    assert moved <= read_by_better ^ read_by_worse | (read_by_better & read_by_worse)  # nothing outside the two boards
    counts = {}
    for i in weight_indices(student.layout, better, state):
        counts[int(i)] = counts.get(int(i), 0) + 1
    for i in weight_indices(student.layout, worse, state):
        counts[int(i)] = counts.get(int(i), 0) - 1
    assert all(student.weights[i] == before[i] for i, c in counts.items() if c == 0)  # read equally: untouched


def test_rank_teaching_makes_the_student_agree_with_the_teacher_more() -> None:
    from deep_2048.agents.ntuple.teach import teach_rank_attempts

    student, teacher = network(1), network(2)
    first = teach_rank_attempts(student.weights, teacher.weights.copy(), student.layout, np.array([START]), 30, 1, 0.0, 0.5, 1.0, 5, 6, 0.0, 9)
    later = teach_rank_attempts(student.weights, teacher.weights.copy(), student.layout, np.array([START]), 30, 1, 0.0, 0.5, 1.0, 5, 6, 0.0, 9)
    assert later[5] / later[4] < first[5] / first[4]
