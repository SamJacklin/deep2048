import numpy as np

from deep_2048.agents.ntuple import NTupleNetwork
from deep_2048.agents.ntuple.diagnostics import SAMPLE, SAMPLE_EVERY, count_updates, greedy_values, record_game, relative_margin, rollout
from deep_2048.agents.ntuple.search import play_from, play_seeded
from deep_2048.game import bitboard as bb

from tests.helpers import SMALL_TUPLES


def trained_looking_network(seed: int) -> NTupleNetwork:
    """Small random weights that favour empty cells a little, so games last a few hundred moves."""
    network = NTupleNetwork(tuples=SMALL_TUPLES)
    network.weights[:] = np.random.default_rng(seed).normal(scale=10, size=network.weights.shape)
    network.weights[0] += 200  # the all-empty pattern of the first tuple
    return network


def test_recorded_games_are_the_games_play_seeded_plays() -> None:
    network = trained_looking_network(0)
    for depth in (1, 2):
        for seed in range(3):
            expected = play_seeded(network.weights, network.layout, depth, 0.0, np.random.default_rng(seed))
            score, tile, kinds, moves, scores, boards, margins = record_game(network.weights, network.layout, depth, 0.0, np.random.default_rng(seed))
            assert (score, tile) == expected
            samples = kinds == SAMPLE
            assert (moves[samples] % SAMPLE_EVERY == 0).all() and moves[samples][0] == 0
            assert (np.diff(scores) >= 0).all() and (margins >= 0).all()


def test_play_from_continues_a_game_exactly() -> None:
    network = trained_looking_network(1)
    rng = np.random.default_rng(7)
    bb.spawn_tile_seeded(bb.spawn_tile_seeded(np.uint64(0), rng), rng)
    board = bb.spawn_tile_seeded(bb.spawn_tile_seeded(np.uint64(0), rng), rng)
    assert play_from(network.weights, network.layout, board, 1, 0.0, rng) == play_seeded(network.weights, network.layout, 1, 0.0, np.random.default_rng(7))


def test_rollouts_with_the_same_seed_agree() -> None:
    network = trained_looking_network(2)
    afterstate = bb.from_array(np.array([[2, 4, 0, 0], [0] * 4, [0] * 4, [0] * 4]))
    first = rollout(network.weights, network.layout, afterstate, 1, 0.0, np.random.default_rng(3))
    assert first == rollout(network.weights, network.layout, afterstate, 1, 0.0, np.random.default_rng(3))


def test_margin_is_the_best_values_lead() -> None:
    assert relative_margin(np.array([100.0, np.nan, 90.0, 50.0])) == 0.1
    assert relative_margin(np.array([np.nan, 5.0, np.nan, np.nan])) == np.inf
    network = trained_looking_network(3)
    stuck = bb.from_array(np.array([[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]]))
    assert np.isnan(greedy_values(network.weights, network.layout, stuck)).all()


def test_update_counts_cover_every_read_of_every_move() -> None:
    network = trained_looking_network(4)
    reads_per_board = len(SMALL_TUPLES) * 8
    counts = np.zeros(network.weights.size, dtype=np.int32)
    count_updates(network.weights, network.layout, np.zeros(3, dtype=np.uint64), 11, counts)
    assert counts.sum() > 0 and counts.sum() % reads_per_board == 0
    again = np.zeros_like(counts)
    count_updates(network.weights, network.layout, np.zeros(3, dtype=np.uint64), 11, again)
    assert np.array_equal(counts, again)
