import numpy as np
import pytest

from deep_2048.agents.ntuple import NTupleNetwork
from deep_2048.agents.ntuple.config import NTupleSettings
from deep_2048.agents.ntuple.diagnostics import first_board
from deep_2048.agents.ntuple.models import LateStarts
from deep_2048.agents.ntuple.network import CoherenceTraces
from deep_2048.agents.ntuple.train import _late_starts, play_games
from deep_2048.game import bitboard as bb

from tests.helpers import SMALL_TUPLES


def network() -> NTupleNetwork:
    net = NTupleNetwork(tuples=SMALL_TUPLES)
    net.weights[:] = np.random.default_rng(0).normal(scale=10, size=net.weights.shape)
    return net


def test_first_board_stops_at_the_first_board_holding_the_tile() -> None:
    net = network()
    found, moves, score = first_board(net.weights, net.layout, 1, 7, np.random.default_rng(5))
    assert bb.max_exponent(found) == 7 and moves > 0 and score > 0
    assert first_board(net.weights, net.layout, 1, 7, np.random.default_rng(5)) == (found, moves, score)
    # A tile the weak network never makes: the game ends first
    assert first_board(net.weights, net.layout, 1, 15, np.random.default_rng(5))[0] == 0


def test_a_start_pool_is_fixed_and_replaces_synthetic_starts(tmp_path) -> None:
    path = tmp_path / "pool.npy"
    np.save(path, np.array([0x1234, 0x5678], dtype=np.uint64))
    starts = _late_starts(NTupleSettings(START_POOL=str(path)))
    assert starts.fraction == 1.0 and starts.harvest_exponent == 0 and list(starts.pool) == [0x1234, 0x5678]
    with pytest.raises(ValueError):
        _late_starts(NTupleSettings(START_POOL=str(path), LATE_START_FRACTION=0.2))


def test_training_from_late_starts_leaves_stage_one_untouched() -> None:
    staged = NTupleNetwork(SMALL_TUPLES, downgrade_tile=1024, stage_tile=64)
    staged.weights[:] = np.random.default_rng(1).normal(scale=10, size=staged.weights.shape)
    n = sum(16 ** len(t) for t in SMALL_TUPLES)
    before = staged.weights.copy()
    # Boards that already hold a 64, so every decision is made from the second stage
    pool = np.array([first_board(staged.weights, staged.layout, 1, 6, np.random.default_rng(s))[0] for s in range(20)], dtype=np.uint64)
    assert (pool != 0).all()
    traces = CoherenceTraces.zeros(staged.weights.size)
    bb.seed(0)
    results = play_games(50, staged.weights, traces, staged.layout, 0.02, True, LateStarts(pool, 1.0, 0))
    assert results.synthetic.all() and (results.harvested == 0).all()
    assert np.array_equal(staged.weights[:n], before[:n])
    assert not np.array_equal(staged.weights[n : 2 * n], before[n : 2 * n])
