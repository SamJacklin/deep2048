import numpy as np

from deep_2048.agents.ntuple import NTupleNetwork
from deep_2048.agents.ntuple import points as pt
from deep_2048.agents.ntuple.search import _play_from, play_seeded
from deep_2048.game import bitboard as bb

from tests.helpers import SMALL_TUPLES


def board(rows: list[list[int]]) -> np.uint64:
    return bb.from_array(np.array(rows))


def noisy_network(seed: int) -> NTupleNetwork:
    network = NTupleNetwork(tuples=SMALL_TUPLES)
    network.weights[:] = np.random.default_rng(seed).normal(scale=10, size=network.weights.shape)
    network.weights[0] += 200
    return network


def test_a_point_is_the_largest_tile_and_which_sizes_below_it_are_there() -> None:
    key = pt.point_key(board([[16384, 8192, 0, 2048], [2, 4, 1024, 0], [0] * 4, [0] * 4]), 3)
    assert key == 14 * pt.KEY_STRIDE + 0b101
    assert pt.point_name(key, 3) == "16384 + 8192 + 2048"
    assert pt.stage_progress(key, 3) == 0.5 + 0.125
    # the 1024 only counts with more detail; duplicates and smaller tiles change nothing
    assert pt.point_key(board([[16384, 8192, 8192, 2048], [2, 4, 1024, 0], [0] * 4, [0] * 4]), 4) == 14 * pt.KEY_STRIDE + 0b1101
    assert pt.point_key(board([[8192, 2, 0, 0], [0] * 4, [0] * 4, [0] * 4]), 3) == 13 * pt.KEY_STRIDE
    assert pt.point_name(13 * pt.KEY_STRIDE, 3) == "8192"


def test_point_boards_follow_the_seeded_greedy_game() -> None:
    network = noisy_network(1)
    weights, layout = network.weights, network.layout
    for seed in range(5):
        keys, boards, moves, scores, n, final, top = pt.point_boards(weights, layout, 4, 8, 2, np.random.default_rng(seed))
        assert (final, top) == play_seeded(weights, layout, 1, 0.0, np.random.default_rng(seed))
        assert n == len(set(keys[:n].tolist()))  # each point once
        assert all(4 <= k // pt.KEY_STRIDE <= 8 for k in keys[:n])
        assert all(pt.point_key(np.uint64(b), 2) == k for k, b in zip(keys[:n], boards[:n]))
        assert list(moves[:n]) == sorted(moves[:n])


def test_play_on_is_play_from() -> None:
    network = noisy_network(2)
    start = board([[64, 32, 0, 2], [4, 0, 0, 0], [0] * 4, [0] * 4])
    for depth in (1, 2):
        assert pt.play_on(network.weights, network.layout, start, depth, 0.0, np.random.default_rng(7)) == \
            _play_from(network.weights, network.layout, start, depth, 0.0, np.random.default_rng(7))


def test_play_to_stops_at_the_goal_and_values_the_arrival() -> None:
    network = noisy_network(3)
    start = board([[64, 32, 0, 2], [4, 0, 0, 0], [0] * 4, [0] * 4])
    for seed in range(6):
        gained, top, moves, value = pt.play_to(network.weights, network.layout, start, 1, 0.0, np.random.default_rng(seed), 7)
        full = pt.play_on(network.weights, network.layout, start, 1, 0.0, np.random.default_rng(seed))
        if top >= 128:  # arrived: stopped no later than the whole game, valued by the network
            assert moves <= full[2] and gained <= full[0] and value != 0.0
        else:  # died first: the same game as play_on, nothing to value
            assert (gained, top, moves) == full and value == 0.0
