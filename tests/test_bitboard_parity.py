import numpy as np
import pytest

from deep_2048.game import bitboard as bb
from deep_2048.game.constants import ACTIONS, Action
from deep_2048.game.game import Game2048
from deep_2048.game.moves import apply_move, get_valid_actions

from tests.helpers import random_boards


def assert_moves_match(board: np.ndarray) -> None:
    packed = bb.from_array(board)
    for action in ACTIONS:
        expected = apply_move(board, action)
        after, reward = bb.move(packed, int(action))
        np.testing.assert_array_equal(bb.to_array(np.uint64(after)), expected.board, err_msg=f"{action.name}\n{board}")
        assert reward == expected.reward, f"{action.name}\n{board}"
        assert (after != packed) == expected.changed, f"{action.name}\n{board}"


@pytest.mark.parametrize("seed", range(4))
def test_moves_match_on_random_boards(seed: int) -> None:
    for board in random_boards(5_000, seed):
        assert_moves_match(board)


def test_moves_match_on_every_row() -> None:
    for row in range(1 << 16):
        cells = [(row >> (4 * i)) & 0xF for i in range(4)]
        if max(cells) == 15:
            continue
        board = np.zeros((4, 4), dtype=np.int32)
        board[0] = [0 if c == 0 else 1 << c for c in cells]
        assert_moves_match(board)


def test_moves_match_through_seeded_games() -> None:
    rng = np.random.default_rng(0)
    for game_seed in range(20):
        game = Game2048(seed=game_seed)
        while not game.is_game_over():
            board = game.get_board()
            assert_moves_match(board)

            packed = bb.from_array(board)
            valid = [a for a in ACTIONS if bb.move(packed, int(a))[0] != packed]
            assert valid == get_valid_actions(board)
            assert not bb.is_game_over(packed)

            game.step(valid[int(rng.integers(len(valid)))])
        assert bb.is_game_over(bb.from_array(game.get_board()))


def test_transpose() -> None:
    for board in random_boards(1_000, seed=10):
        transposed = bb.transpose(bb.from_array(board))
        np.testing.assert_array_equal(bb.to_array(np.uint64(transposed)), board.T)


def test_round_trip_and_cell_helpers() -> None:
    for board in random_boards(1_000, seed=11):
        packed = bb.from_array(board)
        np.testing.assert_array_equal(bb.to_array(packed), board)
        assert bb.count_empty(packed) == int(np.sum(board == 0))
        assert 1 << bb.max_exponent(packed) == max(int(board.max()), 1)


def test_from_array_rejects_bad_tiles() -> None:
    for bad in (3, 1, 65536):
        board = np.zeros((4, 4), dtype=np.int32)
        board[0, 0] = bad
        with pytest.raises(ValueError):
            bb.from_array(board)


def test_32768_tiles_do_not_merge() -> None:
    board = np.zeros((4, 4), dtype=np.int32)
    board[0, :2] = 32768
    packed = bb.from_array(board)
    after, reward = bb.move(packed, int(Action.LEFT))
    assert after == packed and reward == 0


def test_spawn_distribution() -> None:
    bb.seed(123)
    n = 100_000
    cells = np.zeros(16, dtype=np.int64)
    fours = 0
    for _ in range(n):
        spawned = bb.to_array(np.uint64(bb.spawn_tile(np.uint64(0)))).reshape(-1)
        (index,) = np.nonzero(spawned)
        cells[index[0]] += 1
        fours += int(spawned[index[0]] == 4)

    assert abs(fours / n - 0.1) < 0.005
    assert np.all(np.abs(cells / n - 1 / 16) < 0.005)


def test_spawn_only_fills_empty_cells() -> None:
    bb.seed(7)
    for board in random_boards(2_000, seed=12):
        packed = bb.from_array(board)
        spawned = bb.to_array(np.uint64(bb.spawn_tile(packed)))
        changed = spawned != board
        if bb.count_empty(packed) == 0:
            assert not changed.any()
        else:
            assert changed.sum() == 1
            assert board[changed][0] == 0 and spawned[changed][0] in (2, 4)


def test_largest_gap() -> None:
    board = np.array([[8192, 2048, 4, 2], [0] * 4, [0] * 4, [0] * 4])
    assert bb.largest_gap(bb.from_array(board)) == 12  # 4096 missing
    assert bb.largest_gap(bb.from_array(np.array([[8, 4, 2, 0], [0] * 4, [0] * 4, [0] * 4]))) == 0


def test_upgrading_above_a_gap_commutes_with_every_move() -> None:
    checked = 0
    for board in random_boards(2000, seed=21):
        packed = bb.from_array(board)
        gap = bb.largest_gap(packed)
        if gap == 0 or bb.max_exponent(packed) >= bb.MAX_EXPONENT - 1:
            assert bb.upgrade(packed) == packed
            continue
        upgraded = np.uint64(bb.upgrade(packed))
        assert bb.shift_above(upgraded, gap + 1, -1) == packed
        for action in range(bb.NUM_ACTIONS):
            moved, _ = bb.move(packed, action)
            moved_upgraded, _ = bb.move(upgraded, action)
            assert moved_upgraded == bb.shift_above(np.uint64(moved), gap, 1)
        checked += 1
    assert checked > 500
