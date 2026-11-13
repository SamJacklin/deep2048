import numpy as np
import pytest

from deep_2048.agents.ntuple import NTupleAgent, NTupleNetwork
from deep_2048.agents.ntuple.models import CheckpointMetadata
from deep_2048.agents.ntuple.network import evaluate, weight_indices
from deep_2048.agents.ntuple.residual import CorrectedNetwork, train_residual
from deep_2048.agents.ntuple.search import action_values, play_seeded
from deep_2048.bench import file_sha256
from deep_2048.game import bitboard as bb

from tests.helpers import SMALL_TUPLES, random_boards


def board_of(*rows: list[int]) -> np.uint64:
    return bb.from_array(np.array(rows))


@pytest.fixture
def base_path(tmp_path):
    network = NTupleNetwork(SMALL_TUPLES, downgrade_tile=16384)
    network.weights[:] = np.random.default_rng(0).normal(scale=50, size=network.weights.shape)
    network.weights[0] += 300  # favour empty cells, so games last a while
    path = tmp_path / "base.npz"
    network.save(path, CheckpointMetadata(games_played=1000, moves_played=10**6, learning_rate=0.02, temporal_coherence=True))
    return path


def test_no_correction_plays_exactly_as_the_base(base_path) -> None:
    base, _ = NTupleNetwork.load(base_path)
    corrected = CorrectedNetwork.from_base(base_path, residual=True)
    for board in random_boards(200, seed=1):
        packed = bb.from_array(board)
        assert evaluate(corrected.weights, corrected.layout, packed, packed) == evaluate(base.weights, base.layout, packed, packed)
    for seed in range(3):
        rng = np.random.default_rng
        assert play_seeded(corrected.weights, corrected.layout, 2, 0.0, rng(seed)) == play_seeded(base.weights, base.layout, 2, 0.0, rng(seed))


def test_the_constant_applies_from_the_real_16384_even_where_downgrading_hides_it(base_path) -> None:
    base, _ = NTupleNetwork.load(base_path)
    corrected = CorrectedNetwork.from_base(base_path, constant=-80_000)
    late = board_of([16384, 2048, 2, 0], [0] * 4, [0] * 4, [0] * 4)  # read by the base as an 8192 board
    early = board_of([8192, 2048, 2, 0], [0] * 4, [0] * 4, [0] * 4)
    assert bb.downgrade(late, 14) == early
    for board, shift in ((late, -80_000), (early, 0)):
        assert evaluate(corrected.weights, corrected.layout, board, board) == pytest.approx(evaluate(base.weights, base.layout, board, board) + shift)


def test_the_residual_reads_real_tiles(base_path) -> None:
    corrected = CorrectedNetwork.from_base(base_path, residual=True)
    late = board_of([16384, 2048, 2, 0], [0] * 4, [0] * 4, [0] * 4)
    doubled = board_of([32768, 2048, 2, 0], [0] * 4, [0] * 4, [0] * 4)  # the base reads both the same way
    before = [evaluate(corrected.weights, corrected.layout, b, b) for b in (late, doubled)]
    first = corrected.layout.residual.offsets[0]
    corrected.weights[first + (14 | 11 << 4 | 1 << 8)] = 1000.0  # cells 0, 1, 2 hold 16384, 2048, 2 in tuple (0, 1, 2, 3, 4)
    after = [evaluate(corrected.weights, corrected.layout, b, b) for b in (late, doubled)]
    assert after[0] - before[0] >= 1000.0 and after[1] == before[1]


def test_search_values_moves_with_the_correction(base_path) -> None:
    corrected = CorrectedNetwork.from_base(base_path, constant=-50_000)
    merge = board_of([8192, 8192, 2, 0], [4, 0, 0, 0], [0] * 4, [0] * 4)
    values = action_values(corrected.weights, corrected.layout, merge, 1)
    for action in range(bb.NUM_ACTIONS):
        afterstate, reward = bb.move(merge, action)
        afterstate = np.uint64(afterstate)
        if afterstate != merge:
            gated = bb.max_exponent(afterstate) >= 14
            expected = reward + evaluate(corrected.base.weights, corrected.base.layout, afterstate, merge) - 50_000 * gated
            assert values[action] == pytest.approx(expected)


def test_training_changes_only_the_residual_and_only_past_the_gate(base_path) -> None:
    corrected = CorrectedNetwork.from_base(base_path, constant=-1000, residual=True)
    early = np.array([board_of([2, 4, 0, 0], [0] * 4, [0] * 4, [0] * 4)], dtype=np.uint64)
    train_residual(corrected.weights, corrected.layout, early, 0.01, 0)
    assert not corrected.residual_weights.any()  # a weak network's game never reaches 16384
    late = np.array([board_of([16384, 2, 4, 0], [2, 0, 0, 0], [0] * 4, [0] * 4)] * 3, dtype=np.uint64)
    _, _, moves, errors = train_residual(corrected.weights, corrected.layout, late, 0.01, 0)
    assert moves.sum() > 0 and corrected.residual_weights.any() and np.isfinite(errors).all()
    assert corrected.base_unchanged() and corrected.weights[-1] == -1000


def test_a_corrected_checkpoint_round_trips_and_guards_its_base(base_path, tmp_path) -> None:
    corrected = CorrectedNetwork.from_base(base_path, constant=-80_000, residual=True)
    corrected.residual_weights[:5] = [1, 2, 3, 4, 5]
    path = tmp_path / "corrected.npz"
    corrected.save(path)
    loaded = CorrectedNetwork.load(path)
    assert np.array_equal(loaded.weights, corrected.weights)
    agent = NTupleAgent(path, depth=1)
    assert "late constant -80,000" in agent.label and "residual 4x5" in agent.label
    base, metadata = NTupleNetwork.load(base_path)
    base.weights[0] += 1
    base.save(base_path, metadata)
    with pytest.raises(ValueError, match="has changed"):
        CorrectedNetwork.load(path)


def test_following_the_base_plays_the_base_games_while_the_residual_learns(base_path) -> None:
    corrected = CorrectedNetwork.from_base(base_path, constant=-1000, residual=True)
    late = np.array([board_of([16384, 2, 4, 0], [2, 0, 0, 0], [0] * 4, [0] * 4)] * 2, dtype=np.uint64)
    first = train_residual(corrected.weights, corrected.layout, late, 0.01, 5, True)
    assert corrected.residual_weights.any() and corrected.base_unchanged()
    # The same seed replays the same games however much the residual has learned
    second = train_residual(corrected.weights, corrected.layout, late, 0.01, 5, True)
    assert np.array_equal(first[0], second[0]) and np.array_equal(first[2], second[2])


def exponents_board(rows: list[list[int]]) -> np.uint64:
    board = np.uint64(0)
    for i, exponent in enumerate(e for row in rows for e in row):
        board |= np.uint64(exponent) << np.uint64(4 * i)
    return board


# Holds a 16384, and every legal move leads to game over whatever spawns
ONE_MOVE_LEFT = exponents_board([[11, 0, 5, 10], [3, 4, 13, 3], [13, 12, 3, 8], [5, 7, 14, 12]])


def test_the_last_update_moves_the_combined_value_towards_zero(base_path) -> None:
    corrected = CorrectedNetwork.from_base(base_path, constant=-500, residual=True)
    corrected.residual_weights[:] = np.random.default_rng(3).normal(scale=20, size=corrected.residual_weights.shape)
    layout, weights = corrected.layout, corrected.weights
    moves = [(np.uint64(a), r) for a, r in (bb.move(ONE_MOVE_LEFT, action) for action in range(bb.NUM_ACTIONS)) if a != ONE_MOVE_LEFT]
    chosen = max(moves, key=lambda m: m[1] + evaluate(weights, layout, m[0], ONE_MOVE_LEFT))[0]
    before = evaluate(weights, layout, chosen, ONE_MOVE_LEFT)
    base_before = evaluate(corrected.base.weights, corrected.base.layout, chosen, ONE_MOVE_LEFT)
    counts = np.unique(weight_indices(corrected.layout.residual, chosen, chosen), return_counts=True)[1]
    rate = 1e-4
    _, _, moves_played, _ = train_residual(weights, layout, np.array([ONE_MOVE_LEFT], dtype=np.uint64), rate, 0)
    assert moves_played[0] == 1
    # One update, towards a target of 0, with the error measured on the combined value (base + constant + residual)
    after = evaluate(weights, layout, chosen, ONE_MOVE_LEFT)
    assert after - before == pytest.approx(rate * (0 - before) * np.sum(counts**2), rel=1e-4)
    assert evaluate(corrected.base.weights, corrected.base.layout, chosen, ONE_MOVE_LEFT) == base_before


def test_training_leaves_the_base_checkpoint_and_its_coherence_sums_alone(base_path) -> None:
    before = file_sha256(base_path)
    corrected = CorrectedNetwork.from_base(base_path, constant=-1000, residual=True)
    late = np.array([board_of([16384, 2, 4, 0], [2, 0, 0, 0], [0] * 4, [0] * 4)] * 4, dtype=np.uint64)
    train_residual(corrected.weights, corrected.layout, late, 0.01, 1)
    assert corrected.base_unchanged() and file_sha256(base_path) == before
    assert not hasattr(corrected, "traces") and not hasattr(corrected.base, "traces")


def test_a_reloaded_correction_predicts_identically(base_path, tmp_path) -> None:
    corrected = CorrectedNetwork.from_base(base_path, constant=-30_000, residual=True)
    corrected.residual_weights[:] = np.random.default_rng(4).normal(scale=100, size=corrected.residual_weights.shape)
    corrected.save(tmp_path / "c.npz")
    loaded = CorrectedNetwork.load(tmp_path / "c.npz")
    for board in random_boards(300, seed=8):
        packed = bb.from_array(board)
        for options in ({}, {"scale": 0.25}, {"gate_on_state": True}, {"clamp": True}):
            assert evaluate(loaded.weights, loaded.variant(**options), packed, packed) == evaluate(corrected.weights, corrected.variant(**options), packed, packed)


def test_strength_gate_and_clamp_options(base_path) -> None:
    corrected = CorrectedNetwork.from_base(base_path, constant=-90_000, residual=True)
    base = corrected.base
    late = board_of([16384, 2048, 2, 0], [0] * 4, [0] * 4, [0] * 4)
    plain = evaluate(base.weights, base.layout, late, late)
    assert evaluate(corrected.weights, corrected.variant(scale=0.0), late, late) == plain
    assert evaluate(corrected.weights, corrected.variant(scale=0.5), late, late) == pytest.approx(plain - 45_000)
    assert evaluate(corrected.weights, corrected.variant(clamp=True), late, late) == max(plain - 90_000, 0.0)
    # Merging two 8192s: gated by its afterstate, but not by the board the move is made from
    before_merge = board_of([8192, 8192, 2, 0], [0] * 4, [0] * 4, [0] * 4)
    merged = np.uint64(bb.move(before_merge, 3)[0])  # right
    assert bb.max_exponent(merged) == 14
    assert evaluate(corrected.weights, corrected.layout, merged, before_merge) == pytest.approx(evaluate(base.weights, base.layout, merged, before_merge) - 90_000)
    assert evaluate(corrected.weights, corrected.variant(gate_on_state=True), merged, before_merge) == evaluate(base.weights, base.layout, merged, before_merge)
