import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from late_game_anatomy import overdispersion, position_effect  # noqa: E402


def test_the_conversion_rate_is_per_continuation() -> None:
    rng = np.random.default_rng(0)
    tiles = np.where(rng.random((200, 16)) < 0.25, 32768, 16384)
    effect = position_effect(tiles, rng.normal(size=tiles.shape), reps=5)
    assert abs(effect["makes 32768"] - (tiles >= 32768).mean()) < 1e-12
    assert 0 <= effect["makes 32768"] <= 1


def test_overdispersion_is_near_one_without_a_position_effect() -> None:
    made = np.random.default_rng(1).binomial(64, 0.11, size=5_000)
    o = overdispersion(made, 64, sims=500)
    assert 0.9 < o["dispersion ratio"] < 1.1
    assert o["simulated p (variance at least the observed)"] > 0.01
    assert sum(o["counts"]) == 5_000 and abs(sum(o["binomial counts"]) - 5_000) < 1e-6


def test_overdispersion_detects_positions_that_differ() -> None:
    rng = np.random.default_rng(2)
    made = rng.binomial(64, rng.choice([0.05, 0.2], size=5_000))
    o = overdispersion(made, 64, sims=500)
    assert o["dispersion ratio"] > 2
    assert o["simulated p (variance at least the observed)"] < 0.01
    assert o["rho by moments"] > 0.01
