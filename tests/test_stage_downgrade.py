import numpy as np
import pytest

from deep_2048.agents.ntuple import NTupleNetwork
from deep_2048.agents.ntuple.models import CheckpointMetadata
from deep_2048.agents.ntuple.network import TupleLayout, evaluate, weight_indices, with_repeated_halving
from deep_2048.game import bitboard as bb

from tests.helpers import SMALL_TUPLES


def board(rows: list[list[int]]) -> np.uint64:
    return np.uint64(bb.from_array(np.array(rows)))


EARLY = board([[8192, 4096, 2, 0], [2048, 4, 0, 0], [0] * 4, [0] * 4])
LATE = board([[16384, 8192, 2, 0], [4096, 4, 0, 0], [2, 0, 0, 0], [0] * 4])
LATER = board([[32768, 16384, 2, 0], [8192, 4, 0, 0], [0] * 4, [0] * 4])


def test_the_second_stage_downgrades_from_its_own_tile() -> None:
    staged = TupleLayout.build(SMALL_TUPLES, 16384, stage_tile=16384, stage_downgrade_tile=32768)
    plain = TupleLayout.build(SMALL_TUPLES)
    size = staged.sharing.stage_size
    # the first stage reads as before: downgraded from 16384 (an early board has nothing to downgrade)
    assert list(weight_indices(staged, EARLY, EARLY)) == list(weight_indices(plain, EARLY, EARLY))
    # an afterstate that makes 16384 from an early board is still read by the first stage, downgraded from 16384
    assert list(weight_indices(staged, LATE, EARLY)) == list(weight_indices(plain, np.uint64(bb.downgrade(LATE, 14)), EARLY))
    # the second stage reads a 16384 board as it is, and downgrades only from 32768
    assert list(weight_indices(staged, LATE, LATE)) == [size + i for i in weight_indices(plain, LATE, LATE)]
    assert list(weight_indices(staged, LATER, LATER)) == [size + i for i in weight_indices(plain, np.uint64(bb.downgrade(LATER, 15)), LATER)]


def test_no_stage_downgrade_tile_reads_both_stages_alike() -> None:
    staged = TupleLayout.build(SMALL_TUPLES, 16384, stage_tile=16384)
    plain = TupleLayout.build(SMALL_TUPLES)
    size = staged.sharing.stage_size
    assert list(weight_indices(staged, LATE, LATE)) == [size + i for i in weight_indices(plain, np.uint64(bb.downgrade(LATE, 14)), LATE)]


def test_stage_downgrade_tile_zero_turns_it_off_in_the_second_stage() -> None:
    staged = TupleLayout.build(SMALL_TUPLES, 16384, stage_tile=16384, stage_downgrade_tile=0)
    plain = TupleLayout.build(SMALL_TUPLES)
    size = staged.sharing.stage_size
    assert list(weight_indices(staged, LATER, LATER)) == [size + i for i in weight_indices(plain, LATER, LATER)]


def test_values_follow_the_indices_and_the_setting_survives_saving(tmp_path) -> None:
    network = NTupleNetwork(SMALL_TUPLES, downgrade_tile=16384, stage_tile=16384, stage_downgrade_tile=32768)
    network.weights[:] = np.random.default_rng(0).normal(size=network.weights.shape).astype(np.float32)
    for b, s in ((EARLY, EARLY), (LATE, EARLY), (LATE, LATE), (LATER, LATER)):
        expected = float(network.weights[weight_indices(network.layout, b, s)].astype(np.float64).sum())
        assert evaluate(network.weights, network.layout, b, s) == pytest.approx(expected, rel=1e-5)
    from deep_2048.agents.ntuple.models import CheckpointMetadata

    network.save(tmp_path / "n.npz", CheckpointMetadata(games_played=0, moves_played=0, learning_rate=0.1, temporal_coherence=False, optimizer="td"))
    loaded, _ = NTupleNetwork.load(tmp_path / "n.npz")
    assert loaded.stage_downgrade_tile == 32768 and loaded.features == network.features


def test_stage_downgrade_tile_needs_a_second_stage() -> None:
    with pytest.raises(ValueError):
        TupleLayout.build(SMALL_TUPLES, 16384, stage_downgrade_tile=32768)


def test_stage_two_copies_the_first_stage_and_makes_only_untouched_second_stage_weights_optimistic() -> None:
    from deep_2048.agents.ntuple.stages import stage_two

    base = NTupleNetwork(SMALL_TUPLES, downgrade_tile=16384)
    rng = np.random.default_rng(1)
    used = rng.random(base.weights.size) < 0.3
    base.weights[used] = rng.normal(size=used.sum()).astype(np.float32)
    staged, count = stage_two(base, 16384, 32768, 50.0)
    tables = base.weights.size
    assert count == int((~used).sum())
    assert np.array_equal(staged.weights[:tables], base.weights)  # the first stage is untouched
    second = staged.weights[tables:]
    assert np.array_equal(second[used], base.weights[used]) and np.all(second[~used] == 50.0)
    # the first stage plays exactly as before, including the move that makes the stage tile
    for b, s in ((EARLY, EARLY), (LATE, EARLY)):
        assert evaluate(staged.weights, staged.layout, b, s) == evaluate(base.weights, base.layout, b, s)
    plain, none = stage_two(base, 16384, 32768, None)
    assert none == 0 and np.array_equal(plain.weights[tables:], base.weights)


def test_first_boards_are_each_games_first_board_holding_the_tile() -> None:
    from deep_2048.agents.ntuple.stages import first_boards

    network = NTupleNetwork(SMALL_TUPLES)
    network.weights[:] = np.random.default_rng(2).normal(scale=10, size=network.weights.shape).astype(np.float32)
    boards, finals = first_boards(network, 128, 20, 0, 2)
    assert finals.size == 20 and 0 < boards.size <= 20
    assert all(bb.max_exponent(np.uint64(b)) == 7 for b in boards)


def test_repeated_halving_leaves_no_downgrade_tile_and_stops_when_the_gaps_run_out() -> None:
    # 32768, 16384, 8192, 4 and 2: once halved it still holds a 16384, twice halved it doesn't
    assert bb.downgrade(LATER, 14) == board([[16384, 8192, 2, 0], [4096, 4, 0, 0], [0] * 4, [0] * 4])
    assert bb.downgrade(LATER, 14, True) == board([[8192, 4096, 2, 0], [2048, 4, 0, 0], [0] * 4, [0] * 4])
    # a board with one 16384 is halved once either way
    assert bb.downgrade(LATE, 14, True) == bb.downgrade(LATE, 14)
    # no gap left to halve above: the 16384 stays
    full = board([[32768, 8192, 4096, 2048], [1024, 512, 256, 128], [64, 32, 16, 8], [4, 2, 0, 0]])
    assert bb.downgrade(full, 14, True) == board([[16384, 8192, 4096, 2048], [1024, 512, 256, 128], [64, 32, 16, 8], [4, 2, 0, 0]])


def test_a_network_with_repeated_halving_reads_32768_boards_as_the_stage_before_and_round_trips(tmp_path) -> None:  # type: ignore[no-untyped-def]
    plain = TupleLayout.build(SMALL_TUPLES)
    once = NTupleNetwork(SMALL_TUPLES, downgrade_tile=16384)
    repeated = NTupleNetwork(SMALL_TUPLES, downgrade_tile=16384, downgrade_repeat=True)
    assert list(weight_indices(repeated.layout, LATER, LATER)) == list(weight_indices(plain, np.uint64(bb.downgrade(LATER, 14, True)), LATER))
    assert list(weight_indices(repeated.layout, LATER, LATER)) != list(weight_indices(once.layout, LATER, LATER))
    # below a 32768 nothing changes
    assert list(weight_indices(repeated.layout, LATE, LATE)) == list(weight_indices(once.layout, LATE, LATE))
    assert with_repeated_halving(once.layout).sharing == repeated.layout.sharing
    assert "downgrade_repeat" not in once.features
    repeated.save(tmp_path / "n.npz", CheckpointMetadata(games_played=0, moves_played=0, learning_rate=0.1, temporal_coherence=False, optimizer="td"))
    assert NTupleNetwork.load(tmp_path / "n.npz")[0].downgrade_repeat
    with pytest.raises(ValueError, match="needs downgrade_tile"):
        NTupleNetwork(SMALL_TUPLES, downgrade_repeat=True)


def test_with_repeated_halving_a_32768_board_chooses_its_window_halved_once() -> None:
    # A finisher's window: the largest tile exactly 16384 and the tiles summing to at least 32768 - 64
    layout = TupleLayout.build(SMALL_TUPLES, 16384, stage_tile=16384)
    layout = layout._replace(sharing=layout.sharing._replace(stage_sum=32768 - 64))
    repeated = with_repeated_halving(layout)
    size = layout.sharing.stage_size
    row = [8192, 4096, 2048, 1024]
    rest = [[512, 256, 128, 64], [32, 0, 0, 0], [0] * 4]  # with the row and a 16384: 32768 - 32, inside the window
    in_window = board([[16384, 0, 0, 0], row, *rest[:2]])
    after_32768 = board([[32768, 0, 0, 0], row, *rest[:2]])  # halved once (16384 is the gap), it is the board above
    plain = TupleLayout.build(SMALL_TUPLES)
    for lay in (layout, repeated):
        assert weight_indices(lay, in_window, in_window)[0] >= size
    # without it, a board with a 32768 is outside the window; with it, the window opens for the next 16384
    assert weight_indices(layout, after_32768, after_32768)[0] < size
    assert list(weight_indices(repeated, after_32768, after_32768)) == [size + i for i in weight_indices(plain, np.uint64(bb.downgrade(after_32768, 14, True)), after_32768)]
    # short of the window's sum it stays in the first stage
    early = board([[32768, 0, 0, 0], row, [0] * 4, [0] * 4])
    assert weight_indices(repeated, early, early)[0] < size
