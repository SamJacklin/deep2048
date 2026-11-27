from pathlib import Path

import numpy as np
import pytest

from deep_2048.agents.ntuple import NTupleNetwork
from deep_2048.agents.ntuple.finisher import FinisherNetwork, combine_windows, fit_to_teacher, in_window, mean_difference, rollout_teacher, train_attempts
from deep_2048.agents.ntuple.models import CheckpointMetadata
from deep_2048.agents.ntuple.network import AREA_WEIGHTS, area_index, evaluate
from deep_2048.agents.ntuple.search import _play_from, _search, new_transpositions, tile_sum
from deep_2048.game import bitboard as bb

from tests.helpers import SMALL_TUPLES

TOP, MARGIN = 10, 64  # the window before a 2048: largest tile 1024, tiles summing to 1984 or more
READS = len(SMALL_TUPLES) * 8


def board(rows: list[list[int]]) -> np.uint64:
    return bb.from_array(np.array(rows))


INSIDE = board([[1024, 512, 256, 128], [64, 32, 16, 8], [4, 2, 0, 0], [0] * 4])  # sums to 2046
BEFORE = board([[1024, 512, 256, 128], [0] * 4, [0] * 4, [0] * 4])  # sums to 1920, short of the window
BEYOND = board([[2048, 4, 2, 0], [0] * 4, [0] * 4, [0] * 4])  # the larger tile is made
EARLIER = board([[512, 512, 512, 256], [128, 64, 16, 8], [0] * 4, [0] * 4])  # sums to 2008 but its largest tile is 512


def saved_base(tmp_path: Path, seed: int = 0, downgrade_tile: int | None = None) -> Path:
    network = NTupleNetwork(tuples=SMALL_TUPLES, downgrade_tile=downgrade_tile)
    network.weights[:] = np.random.default_rng(seed).normal(scale=10, size=network.weights.shape)
    network.weights[0] += 200
    path = tmp_path / "base.npz"
    network.save(path, CheckpointMetadata(games_played=0, moves_played=0, learning_rate=0.0, temporal_coherence=False))
    return path


def test_the_window_is_the_largest_tile_and_the_tile_sum() -> None:
    low = (1 << (TOP + 1)) - MARGIN
    assert in_window(INSIDE, TOP, low)
    assert not in_window(BEFORE, TOP, low) and not in_window(BEYOND, TOP, low) and not in_window(EARLIER, TOP, low)
    assert tile_sum(EARLIER) >= low


@pytest.mark.parametrize("downgrade_tile", [None, 1024])
def test_only_boards_in_the_window_read_the_second_stage(tmp_path: Path, downgrade_tile: int | None) -> None:
    path = saved_base(tmp_path, downgrade_tile=downgrade_tile)
    base, _ = NTupleNetwork.load(path)
    finisher = FinisherNetwork.from_base(path, TOP, MARGIN)
    afterstate = np.uint64(bb.move(INSIDE, bb.RIGHT)[0])
    for state in (INSIDE, BEFORE, BEYOND, EARLIER):
        assert evaluate(finisher.weights, finisher.layout, afterstate, state) == evaluate(base.weights, base.layout, afterstate, state)
    finisher.weights[finisher.first :] += 1000
    assert evaluate(finisher.weights, finisher.layout, afterstate, INSIDE) == pytest.approx(evaluate(base.weights, base.layout, afterstate, INSIDE) + 1000 * READS)
    for state in (BEFORE, BEYOND, EARLIER):
        assert evaluate(finisher.weights, finisher.layout, afterstate, state) == evaluate(base.weights, base.layout, afterstate, state)


def test_an_untrained_finisher_plays_exactly_as_its_base(tmp_path: Path) -> None:
    path = saved_base(tmp_path, seed=1)
    base, _ = NTupleNetwork.load(path)
    finisher = FinisherNetwork.from_base(path, TOP, MARGIN)
    for start in (INSIDE, BEFORE):
        for depth in (1, 2):
            for seed in range(5):
                assert _play_from(finisher.weights, finisher.layout, start, depth, 0.0, np.random.default_rng(seed)) == _play_from(base.weights, base.layout, start, depth, 0.0, np.random.default_rng(seed))


def test_training_changes_only_the_second_stage_and_a_patch_restores_it(tmp_path: Path) -> None:
    path = saved_base(tmp_path, seed=2)
    base, _ = NTupleNetwork.load(path)
    finisher = FinisherNetwork.from_base(path, TOP, MARGIN)
    errors, abs_errors = np.zeros(finisher.first), np.zeros(finisher.first)
    starts = np.array([INSIDE], dtype=np.uint64)
    teacher_led, teacher_made, student_led, student_made, moves = train_attempts(
        finisher.weights, errors, abs_errors, finisher.layout, finisher.base_layout, starts, 20, 2, 0.0, 0.01, TOP, finisher.low, finisher.first, 0.5, False, 3)
    assert teacher_led + student_led == 20 and 0 <= teacher_made <= teacher_led and 0 <= student_made <= student_led and moves >= 20
    assert finisher.base_unchanged(base)
    indices, values = finisher.patch()
    assert 0 < len(indices) < finisher.first and (abs_errors > 0).sum() == len(np.flatnonzero(abs_errors))
    # every weight that moved has sums, and nothing outside the patch differs from the base
    assert set(indices.tolist()) <= set(np.flatnonzero(abs_errors).tolist())

    finisher.progress = {"attempts": 20}
    saved = tmp_path / "finisher.npz"
    finisher.save(saved)
    again = FinisherNetwork.load(saved)
    assert np.array_equal(again.weights, finisher.weights) and again.progress == {"attempts": 20}
    assert (again.top, again.margin, again.low) == (TOP, MARGIN, finisher.low)
    assert saved.stat().st_size < path.stat().st_size  # a patch, not a second network

    # a base that has changed since is refused
    base.weights[0] += 1
    base.save(path, CheckpointMetadata(games_played=0, moves_played=0, learning_rate=0.0, temporal_coherence=False))
    with pytest.raises(ValueError):
        FinisherNetwork.load(saved)


def test_learning_moves_an_afterstate_towards_the_teachers_value(tmp_path: Path) -> None:
    """One attempt from a full board: after training, greedy play's value of each move it learned from is closer to
    what a depth-2 search of the base gave it."""
    from deep_2048.agents.ntuple.search import action_values

    path = saved_base(tmp_path, seed=3)
    finisher = FinisherNetwork.from_base(path, TOP, MARGIN)
    start = board([[1024, 512, 256, 128], [2, 4, 2, 8], [4, 2, 4, 2], [8, 16, 32, 64]])  # full: sums to 2068
    legal = [a for a in range(4) if np.uint64(bb.move(start, a)[0]) != start]
    if not legal:  # nothing to learn from a dead board
        return
    before = action_values(finisher.weights, finisher.layout, start, 1)
    teacher = action_values(finisher.weights, finisher.layout, start, 2)
    errors, abs_errors = np.zeros(finisher.first), np.zeros(finisher.first)
    train_attempts(finisher.weights, errors, abs_errors, finisher.layout, finisher.base_layout, np.array([start], dtype=np.uint64), 1, 2, 0.0, 0.01, TOP, finisher.low, finisher.first, 0.0, False, 0)
    after = action_values(finisher.weights, finisher.layout, start, 1)
    for a in legal:
        if abs(teacher[a] - before[a]) > 1e-6:
            assert abs(teacher[a] - after[a]) < abs(teacher[a] - before[a])


def test_the_teacher_is_the_base_however_much_the_finisher_has_learned(tmp_path: Path) -> None:
    """Teacher-led attempts with the same seed play the same games before and after training: the base decides them."""
    path = saved_base(tmp_path, seed=4)
    starts = np.array([INSIDE], dtype=np.uint64)
    results = []
    for _ in range(2):
        finisher = FinisherNetwork.from_base(path, TOP, MARGIN)
        errors, abs_errors = np.zeros(finisher.first), np.zeros(finisher.first)
        if results:  # the second time, train first with other seeds
            train_attempts(finisher.weights, errors, abs_errors, finisher.layout, finisher.base_layout, starts, 30, 2, 0.0, 0.01, TOP, finisher.low, finisher.first, 0.5, False, 99)
        results.append(train_attempts(finisher.weights, errors, abs_errors, finisher.layout, finisher.base_layout, starts, 15, 2, 0.0, 0.01, TOP, finisher.low, finisher.first, 0.0, False, 7))
    assert results[0] == results[1] and results[0][2] == 0


def test_fit_to_teacher_counts_agreement_and_learns_nothing(tmp_path: Path) -> None:
    """At depth 1 the teacher's values are the base's own, so an untrained finisher agrees at every choice with no
    difference in value. Training towards a depth-2 teacher then leaves the finisher closer to it than it started."""
    path = saved_base(tmp_path, seed=5)
    finisher = FinisherNetwork.from_base(path, TOP, MARGIN)
    starts = np.array([INSIDE] * 4, dtype=np.uint64)
    weights = finisher.weights.copy()
    choices, agreed, error, centred_error, regret = fit_to_teacher(finisher.weights, finisher.layout, finisher.base_layout, starts, 1, 0.0, TOP, finisher.low, finisher.first, 0)
    assert choices > 0 and agreed == choices and error < 1e-6 and centred_error < 1e-6 and regret == 0
    assert np.array_equal(finisher.weights, weights)

    before = fit_to_teacher(finisher.weights, finisher.layout, finisher.base_layout, starts, 2, 0.0, TOP, finisher.low, finisher.first, 0)
    assert before[4] >= 0 and before[3] <= before[2] + 1e-9
    errors, abs_errors = np.zeros(finisher.first), np.zeros(finisher.first)
    train_attempts(finisher.weights, errors, abs_errors, finisher.layout, finisher.base_layout, starts, 200, 2, 0.0, 0.01, TOP, finisher.low, finisher.first, 0.0, False, 1)
    after = fit_to_teacher(finisher.weights, finisher.layout, finisher.base_layout, starts, 2, 0.0, TOP, finisher.low, finisher.first, 0)
    assert after[0] == before[0] and after[2] < before[2]  # the same teacher-led games, valued more as the teacher does


def test_centred_targets_take_off_the_mean_difference(tmp_path: Path) -> None:
    """The shift is the mean of the teacher's value less the network's over the legal moves, and with it taken off a
    forced move has nothing left to teach."""
    from deep_2048.agents.ntuple.search import action_values

    path = saved_base(tmp_path, seed=6)
    finisher = FinisherNetwork.from_base(path, TOP, MARGIN)
    start = INSIDE
    own = action_values(finisher.weights, finisher.layout, start, 1)
    teacher = np.array(action_values(finisher.weights, finisher.layout, start, 2), dtype=np.float64)
    legal = [a for a in range(4) if np.uint64(bb.move(start, a)[0]) != start]
    assert len(legal) > 1
    teacher[[a for a in range(4) if a not in legal]] = np.nan
    assert mean_difference(finisher.weights, finisher.layout, start, teacher) == pytest.approx(sum(teacher[a] - own[a] for a in legal) / len(legal))

    forced = np.full(4, np.nan)
    forced[legal[0]] = teacher[legal[0]]
    assert mean_difference(finisher.weights, finisher.layout, start, forced) == pytest.approx(teacher[legal[0]] - own[legal[0]])

    # Centred and plain play the same teacher-led games and end with different weights
    ends = []
    for centred in (False, True):
        finisher = FinisherNetwork.from_base(path, TOP, MARGIN)
        errors, abs_errors = np.zeros(finisher.first), np.zeros(finisher.first)
        result = train_attempts(finisher.weights, errors, abs_errors, finisher.layout, finisher.base_layout, np.array([INSIDE], dtype=np.uint64), 10, 2, 0.0, 0.01, TOP, finisher.low, finisher.first, 0.0, centred, 4)
        ends.append((result, finisher.weights))
    assert ends[0][0] == ends[1][0] and not np.array_equal(ends[0][1], ends[1][1])


EXTRA = ((0, 1, 2), (5, 6, 9, 10))


def test_extra_tuples_start_at_zero_train_only_in_the_window_and_round_trip(tmp_path: Path) -> None:
    path = saved_base(tmp_path, seed=4)
    base, _ = NTupleNetwork.load(path)
    finisher = FinisherNetwork.from_base(path, TOP, MARGIN, EXTRA)
    # untrained, it plays exactly as the base, in and out of the window
    for start in (INSIDE, BEFORE):
        for seed in range(5):
            assert _play_from(finisher.weights, finisher.layout, start, 2, 0.0, np.random.default_rng(seed)) == _play_from(base.weights, base.layout, start, 2, 0.0, np.random.default_rng(seed))
    errors, abs_errors = np.zeros(finisher.first), np.zeros(finisher.first)
    train_attempts(finisher.weights, errors, abs_errors, finisher.layout, finisher.base_layout, np.array([INSIDE], dtype=np.uint64), 20, 2, 0.0, 0.01, TOP, finisher.low,
                   finisher.first, 0.5, False, 3)
    # the base and the first stage's extra tables are untouched; the window's extra tables learned
    assert finisher.base_unchanged(base)
    assert finisher.weights[finisher.first + finisher.base_size : 2 * finisher.first].any()
    finisher.progress = {"attempts": 20}
    saved = tmp_path / "finisher.npz"
    finisher.save(saved)
    again = FinisherNetwork.load(saved)
    assert again.extra == EXTRA and np.array_equal(again.weights, finisher.weights)


def test_learning_from_mistakes_only_leaves_agreeing_decisions_alone_and_cuts_regret(tmp_path: Path) -> None:
    path = saved_base(tmp_path, seed=5)
    base, _ = NTupleNetwork.load(path)
    starts = np.array([INSIDE] * 4, dtype=np.uint64)
    # At depth 1 the teacher is the base itself, so the finisher never disagrees and nothing is learned
    finisher = FinisherNetwork.from_base(path, TOP, MARGIN)
    errors, abs_errors = np.zeros(finisher.first), np.zeros(finisher.first)
    train_attempts(finisher.weights, errors, abs_errors, finisher.layout, finisher.base_layout, starts, 50, 1, 0.0, 0.01, TOP, finisher.low, finisher.first, 0.5, False, 1, True)
    assert not abs_errors.any() and len(finisher.patch()[0]) == 0
    # Against a depth-2 teacher it learns only where it disagrees, and its regret along the teacher's play falls
    before = fit_to_teacher(finisher.weights, finisher.layout, finisher.base_layout, starts, 2, 0.0, TOP, finisher.low, finisher.first, 0)
    assert before[1] < before[0]  # there are disagreements to learn from
    train_attempts(finisher.weights, errors, abs_errors, finisher.layout, finisher.base_layout, starts, 200, 2, 0.0, 0.01, TOP, finisher.low, finisher.first, 0.5, False, 1, True)
    after = fit_to_teacher(finisher.weights, finisher.layout, finisher.base_layout, starts, 2, 0.0, TOP, finisher.low, finisher.first, 0)
    assert finisher.base_unchanged(base) and abs_errors.any()
    assert after[4] < before[4] and after[1] >= before[1]


def test_rollout_teacher_keeps_the_greedy_move_unless_confident_and_trains_only_on_its_overrides(tmp_path: Path) -> None:
    path = saved_base(tmp_path, seed=7)
    base, _ = NTupleNetwork.load(path)
    starts = np.array([INSIDE] * 4, dtype=np.uint64)
    greedy = int(np.nanargmax(_search(base.weights, base.layout, INSIDE, 1, 0.0, new_transpositions())))
    assert rollout_teacher(base.weights, base.layout, INSIDE, 5, 16, np.inf)[1] == greedy
    # With an override bar nothing can clear, the teacher is the base itself: nothing is learned
    finisher = FinisherNetwork.from_base(path, TOP, MARGIN)
    errors, abs_errors = np.zeros(finisher.first), np.zeros(finisher.first)
    train_attempts(finisher.weights, errors, abs_errors, finisher.layout, finisher.base_layout, starts, 20, 2, 0.0, 0.01, TOP, finisher.low, finisher.first, 0.5, False, 1, True,
                   5, 8, np.inf)
    assert not abs_errors.any()
    # With an ordinary bar it overrides somewhere and learns, leaving the base alone
    train_attempts(finisher.weights, errors, abs_errors, finisher.layout, finisher.base_layout, starts, 50, 2, 0.0, 0.01, TOP, finisher.low, finisher.first, 0.5, False, 1, True,
                   5, 8, 0.0)
    assert abs_errors.any() and finisher.base_unchanged(base)


def test_area_index_reads_the_top_two_rows_unless_one_holds_more_than_128() -> None:
    rows = [[2, 4, 0, 128], [8, 0, 0, 0], [1024, 512, 256, 256], [2048, 4096, 8192, 16384]]
    assert area_index(board(rows)) == 1 | 2 << 3 | 7 << 9 | 3 << 12
    assert area_index(board([[2, 4, 0, 256], *rows[1:]])) == -1
    assert area_index(board(rows[::-1])) == -1


def test_a_work_area_table_adds_to_the_window_only_and_can_learn_alone_by_td(tmp_path: Path) -> None:
    path = saved_base(tmp_path, seed=8)
    base, _ = NTupleNetwork.load(path)
    finisher = FinisherNetwork.from_base(path, TOP, MARGIN, area=True)
    area = 2 * finisher.first
    assert finisher.weights.size == area + AREA_WEIGHTS
    afterstate = np.uint64(bb.move(INSIDE, bb.RIGHT)[0])
    reads = sum(area_index(np.uint64(s)) >= 0 for s in bb.symmetries(afterstate))
    assert reads == 2  # the half without the large tiles, and its mirror image
    for state in (INSIDE, BEFORE, BEYOND):
        assert evaluate(finisher.weights, finisher.layout, afterstate, state) == evaluate(base.weights, base.layout, afterstate, state)
    finisher.weights[area:] += 1000
    assert evaluate(finisher.weights, finisher.layout, afterstate, INSIDE) == pytest.approx(evaluate(base.weights, base.layout, afterstate, INSIDE) + 1000 * reads)
    assert evaluate(finisher.weights, finisher.layout, afterstate, BEFORE) == evaluate(base.weights, base.layout, afterstate, BEFORE)
    finisher.weights[area:] = 0
    # TD with no teacher, the table learning alone: the stage stays the base's copy
    starts = np.array([INSIDE] * 4, dtype=np.uint64)
    errors, abs_errors = np.zeros(finisher.first + AREA_WEIGHTS), np.zeros(finisher.first + AREA_WEIGHTS)
    led = train_attempts(finisher.weights, errors, abs_errors, finisher.layout, finisher.base_layout, starts, 30, 2, 0.0, 0.01, TOP, finisher.low, finisher.first, 0.5, False, 1,
                         False, 0, 32, 2.0, True, area)
    assert led[0] == 0 and led[2] == 30 and led[4] > 0
    assert finisher.weights[area:].any() and not abs_errors[: finisher.first].any()
    assert np.array_equal(finisher.weights[finisher.first : area], finisher.weights[: finisher.first]) and finisher.base_unchanged(base)
    finisher.save(tmp_path / "area.npz")
    loaded = FinisherNetwork.load(tmp_path / "area.npz")
    assert loaded.area and np.array_equal(loaded.weights, finisher.weights)
    # The same without a table: the stage's own weights learn
    plain = FinisherNetwork.from_base(path, TOP, MARGIN)
    errors, abs_errors = np.zeros(plain.first), np.zeros(plain.first)
    train_attempts(plain.weights, errors, abs_errors, plain.layout, plain.base_layout, starts, 30, 2, 0.0, 0.01, TOP, plain.low, plain.first, 0.5, False, 1, False, 0, 32, 2.0, True)
    assert abs_errors.any() and plain.base_unchanged(base)


@pytest.mark.parametrize("areas", [(False, False), (True, True), (True, False), (False, True)])
def test_two_windows_play_each_finisher_in_its_own_window_and_the_base_elsewhere(tmp_path: Path, areas: tuple[bool, bool]) -> None:
    path = saved_base(tmp_path, seed=6)
    base, _ = NTupleNetwork.load(path)
    lower = board([[512, 256, 128, 32], [16, 8, 4, 2], [2, 0, 0, 0], [0] * 4])  # largest 512, sums to 960: the lower window
    trained = []
    for top, start, area in ((TOP, INSIDE, areas[0]), (TOP - 1, lower, areas[1])):
        finisher = FinisherNetwork.from_base(path, top, MARGIN, area=area)
        errors, abs_errors = np.zeros(finisher.weights.size - finisher.first), np.zeros(finisher.weights.size - finisher.first)
        train_attempts(finisher.weights, errors, abs_errors, finisher.layout, finisher.base_layout, np.array([start], dtype=np.uint64), 20, 2, 0.0, 0.01, top, finisher.low,
                       finisher.first, 0.5, False, 3)
        assert finisher.weights[2 * finisher.first :].any() == area  # a table, where there is one, has learned something
        finisher.progress = {"attempts": 20}
        finisher.save(tmp_path / f"finisher-{top}.npz")
        trained.append(finisher)
    weights, layout = combine_windows(tmp_path / f"finisher-{TOP}.npz", tmp_path / f"finisher-{TOP - 1}.npz")
    for state, net in ((INSIDE, trained[0]), (lower, trained[1]), (BEFORE, None)):
        for action in range(4):
            after = np.uint64(bb.move(state, action)[0])
            if after == state:
                continue
            expected = evaluate(base.weights, base.layout, after, state) if net is None else evaluate(net.weights, net.layout, after, state)
            assert evaluate(weights, layout, after, state) == expected
    with pytest.raises(ValueError):
        combine_windows(tmp_path / f"finisher-{TOP - 1}.npz", tmp_path / f"finisher-{TOP}.npz")


def test_a_base_with_small_tile_tables_gives_each_window_its_own_copy(tmp_path: Path) -> None:
    network = NTupleNetwork(tuples=SMALL_TUPLES, downgrade_tile=None, small_tables=True)
    network.weights[:] = np.random.default_rng(7).normal(scale=10, size=network.weights.shape)
    path = tmp_path / "base.npz"
    network.save(path, CheckpointMetadata(games_played=0, moves_played=0, learning_rate=0.0, temporal_coherence=False))
    base, _ = NTupleNetwork.load(path)
    small = base.layout.sharing.small_offset
    lower = board([[512, 256, 128, 32], [16, 8, 4, 2], [2, 0, 0, 0], [0] * 4])  # largest 512, sums to 960: the lower window
    trained = []
    for top, start in ((TOP, INSIDE), (TOP - 1, lower)):
        finisher = FinisherNetwork.from_base(path, top, MARGIN)
        assert finisher.first == base.weights.size and finisher.weights.size == 2 * finisher.first
        for seed in range(3):  # untrained, it plays exactly as the base
            assert _play_from(finisher.weights, finisher.layout, start, 1, 0.0, np.random.default_rng(seed)) == _play_from(base.weights, base.layout, start, 1, 0.0, np.random.default_rng(seed))
        errors, abs_errors = np.zeros(finisher.first), np.zeros(finisher.first)
        train_attempts(finisher.weights, errors, abs_errors, finisher.layout, finisher.base_layout, np.array([start], dtype=np.uint64), 20, 2, 0.0, 0.01, top, finisher.low,
                       finisher.first, 0.5, False, 3)
        assert finisher.base_unchanged(base)
        assert (finisher.weights[finisher.first + small :] != base.weights[small:]).any()  # the window's own small-tile tables learned
        finisher.progress = {"attempts": 20}
        finisher.save(tmp_path / f"finisher-{top}.npz")
        assert np.array_equal(FinisherNetwork.load(tmp_path / f"finisher-{top}.npz").weights, finisher.weights)
        trained.append(finisher)
    weights, layout = combine_windows(tmp_path / f"finisher-{TOP}.npz", tmp_path / f"finisher-{TOP - 1}.npz")
    for state, net in ((INSIDE, trained[0]), (lower, trained[1]), (BEFORE, None)):
        for action in range(4):
            after = np.uint64(bb.move(state, action)[0])
            if after != state:
                expected = evaluate(base.weights, base.layout, after, state) if net is None else evaluate(net.weights, net.layout, after, state)
                assert evaluate(weights, layout, after, state) == expected
    with pytest.raises(ValueError):
        FinisherNetwork.from_base(path, TOP, MARGIN, extra=((0, 1),))
