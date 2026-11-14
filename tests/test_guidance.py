import numpy as np

from deep_2048.agents.ntuple import NTupleNetwork
from deep_2048.agents.ntuple.guidance import play_selective
from deep_2048.agents.ntuple.search import play_from
from deep_2048.game import bitboard as bb

from tests.helpers import SMALL_TUPLES


def network() -> NTupleNetwork:
    net = NTupleNetwork(tuples=SMALL_TUPLES)
    net.weights[:] = np.random.default_rng(0).normal(scale=10, size=net.weights.shape)
    net.weights[0] += 200
    return net


def start(seed: int) -> np.uint64:
    rng = np.random.default_rng(seed)
    return bb.spawn_tile_seeded(bb.spawn_tile_seeded(np.uint64(0), rng), rng)


def test_an_empty_band_is_greedy_play_and_a_full_one_is_search_everywhere() -> None:
    net = network()
    for seed in range(4):
        board = start(seed)
        greedy = play_from(net.weights, net.layout, board, 1, 0.0, np.random.default_rng(seed))
        never = play_selective(net.weights, net.layout, board, 3, 0, 0.0, 0.0, np.random.default_rng(seed))
        assert never[:2] == greedy and never[3] == 0 and never[5] == never[6] == 0
        deep = play_from(net.weights, net.layout, board, 3, 0.0, np.random.default_rng(seed))
        always = play_selective(net.weights, net.layout, board, 3, 0, 0.0, np.inf, np.random.default_rng(seed))
        assert always[:2] == deep and 0 < always[3] <= always[2] and always[5] > 0 and always[6] > 0


def test_nothing_is_searched_below_the_tile() -> None:
    net = network()
    result = play_selective(net.weights, net.layout, start(1), 3, 14, 0.0, np.inf, np.random.default_rng(1))
    assert result[3] == 0  # a weak network never reaches 16384
