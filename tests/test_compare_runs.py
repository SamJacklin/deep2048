import numpy as np

from scripts.compare_runs import paired_bootstrap, parse_label, statistics


def test_labels_give_games_and_search() -> None:
    assert parse_label("N-tuple TD (2,500,000 games, TC lr 0.02, 8x6 network)") == (2_500_000, "greedy")
    assert parse_label("N-tuple TD (2,000,000 games, TC lr 0.02) + search depth=3") == (2_000_000, "depth 3")
    assert parse_label("N-tuple TD (1,500,000 games) + search depth=6 cutoff=0.1") == (1_500_000, "depth 6, cutoff 0.1")


def test_conversion_counts_only_games_that_reached_16384() -> None:
    average, reached, made, conversion = statistics(np.array([10.0, 20, 30, 40]), np.array([16384, 32768, 8192, 32768]))
    assert (average, reached, made) == (25.0, 0.75, 0.5)
    assert conversion == 2 / 3


def test_a_constant_shift_is_measured_exactly() -> None:
    rng = np.random.default_rng(1)
    scores, tiles = rng.integers(0, 400_000, 500).astype(float), rng.choice([8192, 16384, 32768], 500)
    change, interval = paired_bootstrap((scores, tiles), (scores + 100, tiles))
    assert np.allclose(change, [100, 0, 0, 0]) and np.allclose(interval, [[100, 100], [0, 0], [0, 0], [0, 0]])
