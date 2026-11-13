import numpy as np
import pytest

from deep_2048.agents.ntuple import NTupleAgent, NTupleNetwork
from deep_2048.agents.ntuple.models import CheckpointMetadata
from deep_2048.agents.ntuple.network import evaluate
from deep_2048.agents.ntuple.residual import CorrectedNetwork, train_residual
from deep_2048.agents.ntuple.search import action_values, play_seeded
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
