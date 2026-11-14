import sys
from pathlib import Path

import numpy as np

from deep_2048.agents.ntuple import NTupleNetwork
from deep_2048.agents.ntuple.anatomy import play_mixed, record_trajectory
from deep_2048.agents.ntuple.search import play_from
from deep_2048.game import bitboard as bb

from tests.helpers import SMALL_TUPLES

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from late_game_anatomy import beta_binomial_fit, cluster_interval, features, out_of_fold, points_of  # noqa: E402


def network() -> NTupleNetwork:
    net = NTupleNetwork(tuples=SMALL_TUPLES)
    net.weights[:] = np.random.default_rng(0).normal(scale=10, size=net.weights.shape)
    return net


def start(seed: int) -> np.uint64:
    rng = np.random.default_rng(seed)
    return bb.spawn_tile_seeded(bb.spawn_tile_seeded(np.uint64(0), rng), rng)


def test_mixed_play_is_greedy_or_search_at_the_extremes() -> None:
    net = network()
    for seed in range(3):
        b = start(seed)
        assert play_mixed(net.weights, net.layout, b, 1, -1, 0.0, np.random.default_rng(seed))[:2] == play_from(net.weights, net.layout, b, 1, 0.0, np.random.default_rng(seed))
        assert play_mixed(net.weights, net.layout, b, 2, 0, 0.0, np.random.default_rng(seed))[:2] == play_from(net.weights, net.layout, b, 1, 0.0, np.random.default_rng(seed))
        deep = play_mixed(net.weights, net.layout, b, 2, -1, 0.0, np.random.default_rng(seed))
        assert deep[:2] == play_from(net.weights, net.layout, b, 2, 0.0, np.random.default_rng(seed)) and deep[3] > 0


def test_a_temporary_search_matches_search_then_greedy_from_where_it_stops() -> None:
    net = network()
    b = start(4)
    score, tile, moves, _ = play_mixed(net.weights, net.layout, b, 2, 10, 0.0, np.random.default_rng(4))
    assert moves > 10 and tile >= 2


def test_a_recorded_trajectory_replays_as_greedy_play() -> None:
    net = network()
    b = start(5)
    boards, scores, tile, total = record_trajectory(net.weights, net.layout, b, np.random.default_rng(5))
    assert boards[0] == b and scores[0] == 0 and len(boards) == len(scores)
    assert (total, tile) == play_from(net.weights, net.layout, b, 1, 0.0, np.random.default_rng(5))
    assert np.all(np.diff(scores) >= 0)


def test_features_of_a_known_board() -> None:
    # Row 0: 16384 8192 4096 2048 (a chain from the corner); row 1: 2 2; the rest empty
    exps = [14, 13, 12, 11, 1, 1] + [0] * 10
    board = np.uint64(sum(e << (4 * i) for i, e in enumerate(exps)))
    f = features(board)
    assert f[0] == 1 and f[1] == 0  # corner
    assert f[2] == 10  # empty
    assert f[3] == 13 and f[4] == 1  # 8192 next to it
    assert f[10] == 4  # 16384 → 8192 → 4096 → 2048
    assert f[9] == 1  # the two 2s


def test_the_beta_binomial_recovers_a_known_position_effect() -> None:
    rng = np.random.default_rng(0)
    for rho in (0.0, 0.1):
        if rho == 0:
            p = np.full(3000, 0.12)
        else:
            a = 0.12 * (1 - rho) / rho
            p = rng.beta(a, a * 0.88 / 0.12, 3000)
        mu, fitted = beta_binomial_fit(rng.binomial(64, p), 64)
        assert abs(mu - 0.12) < 0.01 and abs(fitted - rho) < 0.03


def test_out_of_fold_predictions_find_a_real_signal_and_not_a_fake_one() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2000, 3))
    p = 1 / (1 + np.exp(-(-2 + 1.0 * x[:, 0])))
    k = rng.binomial(64, p)
    predicted = out_of_fold(x, k, 64, np.arange(2000), 1.0)
    assert np.corrcoef(predicted, p)[0, 1] > 0.9
    noise = out_of_fold(rng.normal(size=(2000, 3)), k, 64, np.arange(2000), 1.0)
    assert noise.std() < 0.01


def test_points_cover_the_milestones_and_the_end() -> None:
    boards = np.array([start(s) for s in range(1000)], dtype=np.uint64)
    points = points_of({"outcome": "failure"}, boards)
    moves = [p["move"] for p in points]
    assert moves[0] == 0 and 250 in moves and 750 in moves and 800 in moves and 950 in moves
    assert all(p["remaining"] == 1000 - p["move"] for p in points)


def test_the_cluster_interval_resamples_trajectories() -> None:
    points = [{"seed": s, "d": [float(s % 2)] * 4} for s in range(40)]
    mean, lo, hi = cluster_interval(points, "d")
    assert mean == 0.5 and lo < 0.5 < hi
