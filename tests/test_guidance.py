import numpy as np
import pytest

from deep_2048.agents.ntuple import NTupleNetwork
from deep_2048.agents.ntuple.diagnostics import greedy_values
from deep_2048.agents.ntuple.guidance import guided_move, play_selective
from deep_2048.agents.ntuple.models import SearchGuide
from deep_2048.agents.ntuple.search import action_values, new_transpositions, play_from
from deep_2048.agents.ntuple.train import play_game
from deep_2048.game import bitboard as bb

from tests.helpers import SMALL_TUPLES, random_boards


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


def guide(min_exponent: int = 0, rank_step: float = 0.0, reads: int = len(SMALL_TUPLES) * 8) -> SearchGuide:
    return SearchGuide(2, min_exponent, 0.0, np.inf, rank_step, 1.0, 0.0, reads, np.zeros(6, np.int64))


def test_guided_training_games_search_and_count() -> None:
    net = network()
    bb.seed(3)
    g = guide()
    play_game(net.weights, None, net.layout, 0.0, False, np.uint64(0), 0, g)
    decisions, searched, disagreed, ranked, nodes, leaves = g.counts
    assert decisions > 0 and searched > 0 and disagreed > 0 and ranked == 0 and nodes > 0 and leaves > 0
    late = guide(min_exponent=14)
    bb.seed(3)
    play_game(net.weights, None, net.layout, 0.0, False, np.uint64(0), 0, late)
    assert late.counts[1] == 0  # a weak network never reaches 16384


def test_a_ranking_update_closes_its_share_of_the_gap() -> None:
    net = network()
    for board in (bb.from_array(b) for b in random_boards(500, seed=4)):
        values = greedy_values(net.weights, net.layout, board)
        if np.isnan(values).all():
            continue
        greedy, teacher = int(np.nanargmax(values)), int(np.nanargmax(action_values(net.weights, net.layout, board, 2)))
        if teacher != greedy:
            break
    gap = values[greedy] - values[teacher]
    g = guide(rank_step=0.5)
    afterstate, _, _ = guided_move(net.weights, net.layout, board, g, new_transpositions())
    after = greedy_values(net.weights, net.layout, board)
    assert g.counts[3] == 1 and afterstate == bb.move(board, teacher)[0]
    # Exactly half of the gap plus the margin, whatever weights the two afterstates share
    assert (after[greedy] - after[teacher]) == pytest.approx(gap - 0.5 * (gap + 1.0), rel=1e-4)
