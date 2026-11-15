"""Stage 8A's correctness tests: AutoTIDBD with local normalisation (tests B to H of the repair plan).

Test A, that the published global rule still matches its reference, is tests/test_autotidbd.py, unchanged.
"""

import numpy as np
import pytest
from numba import njit

from deep_2048.agents.ntuple import NTupleAgent, autotidbd
from deep_2048.agents.ntuple.autotidbd import BETA, H, XI, Tidbd, sparse_step
from deep_2048.agents.ntuple.models import CheckpointMetadata
from deep_2048.agents.ntuple.network import NTupleNetwork, evaluate
from deep_2048.agents.ntuple.search import action_values
from deep_2048.agents.ntuple.train import play_games, play_games_threaded, train
from deep_2048.game import bitboard as bb

from tests.test_autotidbd import N, NO_FEATURES, Pair, random_features, small_default_network, sparse, tidbd_settings, vector  # noqa: F401
from tests.helpers import SMALL_TUPLES


def normaliser_sum(pair: Pair, x: np.ndarray, x_next: np.ndarray, z: np.ndarray) -> float:
    """Line 11's sum for these features and trace, from the kernel's step sizes as they stand."""
    return float(-(pair.optimiser.alpha() * (pair.gamma * x_next - x)) @ z)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_b_the_local_correction_brings_the_measured_step_to_exactly_one(dtype: type) -> None:
    pair = Pair(alpha=0.3, theta=0.01, dtype=dtype, local=True, w=np.arange(N, dtype=np.float64))
    trace, x, x_next = vector(i0=2, i3=1, i5=1), vector(i0=2, i3=3, i7=1), vector(i0=1, i8=1)
    pair.step(vector(i1=1), trace, reward=1.0)
    pair.step(trace, x, reward=0.5)
    np.testing.assert_array_equal(pair.z, trace)

    pair.step(x, x_next, reward=1.0)
    # The reference's M is the sum before the correction, taken after line 10 as the listing has it
    assert pair.last_m > 1.2
    assert pair.stats[autotidbd.NORMALISATIONS] >= 1
    # The same sum, same features and same trace, with the corrected step sizes: the bound is met, with equality
    assert normaliser_sum(pair, x, x_next, trace) == pytest.approx(1.0, abs=1e-12 if dtype is np.float64 else 1e-5)
    assert pair.optimiser.shared[0] == 0.0


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("gamma", [1.0, 0.9])
def test_c_the_sparse_local_update_matches_the_dense_reference(dtype: type, gamma: float) -> None:
    rng = np.random.default_rng(10)
    pair = Pair(alpha=0.3, theta=0.05, dtype=dtype, gamma=gamma, local=True, w=rng.normal(size=N))
    for _ in range(30):
        pair.start_episode()
        x = random_features(rng)
        for _ in range(rng.integers(1, 30)):
            # Half the time most of the features carry over, so the trace overlaps and normalisation fires often
            x_next = random_features(rng) if rng.random() < 0.5 else np.where(rng.random(N) < 0.8, x, 0) + (rng.random(N) < 0.1)
            pair.step(x, x_next, float(rng.normal()))
            x = x_next if x_next.any() else random_features(rng)
        pair.step(x, NO_FEATURES, 0.0)
    # With twelve weights the offenders are soon cut down, so events are few, but every step was checked against the reference
    assert pair.stats[autotidbd.NORMALISATIONS] >= 8
    # Under the local rule the shared offset never moves, and rates have parted company
    assert pair.optimiser.shared[0] == 0.0
    assert np.ptp(pair.optimiser.rates[:, BETA]) > 0.1


def test_d_only_the_trace_is_normalised() -> None:
    pair = Pair(alpha=0.5, theta=0.0, local=True)
    control = Pair(alpha=0.5, theta=0.0)
    trace, x = vector(i0=2, i5=1), vector(i0=2, i6=1)
    for each in (pair, control):
        each.step(trace, x, reward=1.0)
        each.step(x, NO_FEATURES, reward=1.0)
    # Line 11's sum: weight 0 gives 0.5 * 2 * 2 = 2, and weights 5 (only in the trace) and 6 (only in x) give nothing
    assert pair.last_m == control.last_m == pytest.approx(2.0)
    alpha, published = pair.optimiser.alpha(), control.optimiser.alpha()
    # Published: all twelve step sizes are halved. Local: the trace's two are, including weight 5, whose z isn't 0
    # although it added nothing to the sum, and the other ten are untouched
    np.testing.assert_allclose(published, 0.25)
    np.testing.assert_allclose(alpha[[0, 5]], 0.25)
    np.testing.assert_array_equal(np.delete(pair.optimiser.rates[:, BETA], [0, 5]), np.full(N - 2, np.log(0.5)))
    assert pair.optimiser.shared[0] == 0.0 and control.optimiser.shared[0] == pytest.approx(-np.log(2))
    # Everything else follows the same equations: with the trace's step sizes equal in both, so are w, h and ξ there
    np.testing.assert_allclose(pair.weights[0], control.weights[0])
    np.testing.assert_allclose(pair.optimiser.rates[0, [H, XI]], control.optimiser.rates[0, [H, XI]])
    # ... while weight 6, updated at its untouched rate, has moved twice as far as under the published rule
    assert pair.weights[6] == pytest.approx(2 * control.weights[6])


@pytest.mark.parametrize(
    "x_next",
    [vector(i1=1, i4=2, i7=1), vector(i1=1, i8=1, i7=1), vector(i2=3, i9=1), vector(i4=1, i7=1), NO_FEATURES],
    ids=["identical", "partial overlap", "appear and disappear", "multiplicities change", "terminal"],
)
def test_e_overlapping_and_persistent_features(x_next: np.ndarray) -> None:
    rng = np.random.default_rng(11)
    x = vector(i1=1, i4=2, i7=1)
    pair = Pair(alpha=0.4, local=True, w=rng.normal(size=N))
    pair.step(vector(i1=1, i4=1, i2=1), x, reward=0.5)
    # x persists: it's the trace and the current features at once, as a long-lived anchor's weights are
    pair.step(x, x, reward=0.5)
    before = pair.optimiser.alpha()
    pair.step(x, x_next, reward=1.0)
    if not (x_next - x).any():
        # Unchanged features give line 11 nothing to sum, however large their step sizes
        assert pair.last_m == 1.0
        np.testing.assert_array_equal(pair.optimiser.alpha(), before)
    else:
        assert pair.last_m > 1.0
        outside = pair.z == 0
        untouched = outside & (x_next == 0)
        np.testing.assert_array_equal(pair.optimiser.alpha()[untouched], before[untouched])


def small_network(dtype: type = np.float32, alpha: float = 0.05, local: bool = False) -> tuple[NTupleNetwork, Tidbd]:
    network = NTupleNetwork(tuples=SMALL_TUPLES, downgrade_tile=64, scale_feature=True)
    network.weights = network.weights.astype(dtype)
    return network, Tidbd.create(network.weights.size, alpha, 0.01, 10_000.0, dtype, local)


def metadata(optimizer: str = "autotidbd") -> CheckpointMetadata:
    return CheckpointMetadata(games_played=40, moves_played=1, learning_rate=0.05, optimizer=optimizer)


def test_f_a_fork_starts_from_exactly_the_original(tmp_path, small_default_network) -> None:  # noqa: F811
    # A global run that has normalised, so there's a shared offset to inherit
    settings = tidbd_settings(tmp_path, TIDBD_INITIAL_ALPHA=0.05)
    original = train(total_games=200, snapshot_at=[200], settings=settings).checkpoint_path
    loaded = autotidbd.load(original)
    assert loaded is not None and loaded[0].shared[0] < 0 and not loaded[0].local

    local_settings = tidbd_settings(tmp_path, TIDBD_INITIAL_ALPHA=0.05, TIDBD_NORMALISATION="local", CHECKPOINT_DIR=str(tmp_path / "fork"), OUTPUT_DIR=str(tmp_path / "fork-output"))
    with pytest.raises(ValueError, match="AutoTIDBD settings"):
        train(resume_from=original, total_games=200, settings=local_settings)
    # Forked with no further games: the fork's checkpoint is the original in everything but its rule
    forked = train(resume_from=original, total_games=200, settings=local_settings, fork_normalisation=True).checkpoint_path
    fork = autotidbd.load(forked)
    assert fork is not None and fork[0].local
    np.testing.assert_array_equal(fork[0].rates, loaded[0].rates)
    np.testing.assert_array_equal(fork[0].shared, loaded[0].shared)
    np.testing.assert_array_equal(fork[0].alpha(), loaded[0].alpha())
    np.testing.assert_array_equal(fork[2], loaded[2])

    agents = [NTupleAgent(path, depth=3) for path in (original, forked)]
    np.testing.assert_array_equal(agents[0].network.weights, agents[1].network.weights)
    assert "AutoTIDBD local" in agents[1].label and "local" not in agents[0].label
    rng = np.random.default_rng(12)
    for _ in range(50):
        board = np.uint64(rng.integers(0, 2**40))
        values = [evaluate(agent.network.weights, agent.network.layout, board, board) for agent in agents]
        assert values[0] == values[1]
        for depth in (1, 3):
            decisions = [action_values(agent.network.weights, agent.network.layout, board, depth) for agent in agents]
            np.testing.assert_array_equal(decisions[0], decisions[1])

    # The fork trains on under the local rule: its offset stays where the original left it
    train(resume_from=forked, total_games=300, settings=local_settings)
    trained = autotidbd.load(forked)
    assert trained is not None and trained[0].local and trained[0].shared[0] == loaded[0].shared[0]
    assert trained[2][autotidbd.UPDATES] > loaded[2][autotidbd.UPDATES]
    # ... and never goes back under the global one, nor forks a second time, nor does a fresh run "fork"
    with pytest.raises(ValueError, match="AutoTIDBD settings"):
        train(resume_from=forked, total_games=400, settings=settings)
    with pytest.raises(ValueError, match="a fork goes from global to local"):
        train(resume_from=forked, total_games=400, settings=local_settings, fork_normalisation=True)
    with pytest.raises(ValueError, match="a fork goes from global to local"):
        train(resume_from=original, total_games=400, settings=settings, fork_normalisation=True)
    with pytest.raises(ValueError, match="needs a checkpoint"):
        train(total_games=50, settings=local_settings, fork_normalisation=True)
    # The original is as it was
    np.testing.assert_array_equal(autotidbd.load(original)[0].rates, loaded[0].rates)


def test_g_a_reloaded_local_checkpoint_carries_on_exactly(tmp_path) -> None:
    def run(reload: bool) -> tuple[np.ndarray, Tidbd, np.ndarray]:
        network, optimiser = small_network(local=True)
        bb.seed(5)
        first = play_games(60, network.weights, None, network.layout, 0.0, True, None, None, None, optimiser)
        if reload:
            network.save(tmp_path / "state.npz", metadata("autotidbd-local"), None, autotidbd.checkpoint_arrays(optimiser, 0.05, np.zeros(len(autotidbd.STATS))), compress=True)
            network, _ = NTupleNetwork.load(tmp_path / "state.npz")
            loaded = autotidbd.load(tmp_path / "state.npz")
            assert loaded is not None and loaded[0].local
            optimiser = loaded[0]
        bb.seed(6)
        second = play_games(60, network.weights, None, network.layout, 0.0, True, None, None, None, optimiser)
        return network.weights, optimiser, first.stats + second.stats

    weights, optimiser, stats = run(reload=False)
    reloaded_weights, reloaded, _ = run(reload=True)
    np.testing.assert_array_equal(reloaded_weights, weights)
    np.testing.assert_array_equal(reloaded.rates, optimiser.rates)
    assert reloaded.shared[0] == optimiser.shared[0] == 0.0
    assert stats[autotidbd.NORMALISATIONS] > 0


@njit
def hammer(weights: np.ndarray, optimiser: Tidbd, steps: int, seed: int, stats: np.ndarray) -> float:
    """Millions of sparse updates: a long-lived anchor read 8 times beside features that come and go, with bursts of
    huge errors and terminal steps. Returns the largest of line 11's sums measured after each step's correction."""
    np.random.seed(seed)
    n = weights.size
    size = 8
    indices = np.zeros((3, size), np.int64)
    counts = np.zeros((3, size))
    sizes = np.zeros(3, np.int64)
    work = (np.empty(2 * size, np.int64), np.empty(2 * size), np.empty(2 * size), np.empty(2 * size))
    trace, current, successor = 0, 1, 2
    worst = 0.0
    for step in range(steps):
        terminal = step % 997 == 996
        k = 0
        if not terminal:
            # The anchor (weight 0) persists for long stretches, its count sometimes changing, then leaves for a while
            if (step // 5_000) % 4 != 3:
                indices[successor, 0] = 0
                counts[successor, 0] = 8.0 if step % 50 else 4.0
                k = 1
            last = 0
            for _ in range(np.random.randint(1, size - 1)):
                last += np.random.randint(1, n // size)
                if last >= n:
                    break
                indices[successor, k] = last
                counts[successor, k] = float(np.random.randint(1, 4))
                k += 1
        sizes[successor] = k
        if sizes[current] > 0:
            reward = (1e6 if (step // 1_000) % 7 == 0 else 1.0) * (np.random.random() - 0.5)
            sparse_step(
                weights, optimiser, 1.0, reward,
                (indices[current], counts[current], sizes[current]),
                (indices[successor], counts[successor], sizes[successor]),
                (indices[trace], counts[trace], sizes[trace]),
                work, stats,
            )  # fmt: skip
            # Line 11's sum again from the features themselves, with the step sizes as the step left them
            total = 0.0
            for p in range(sizes[trace]):
                i = indices[trace, p]
                d = 0.0
                for j in range(sizes[successor]):
                    if indices[successor, j] == i:
                        d += counts[successor, j]
                for j in range(sizes[current]):
                    if indices[current, j] == i:
                        d -= counts[current, j]
                total -= np.exp(optimiser.rates[i, BETA] + optimiser.shared[0]) * d * counts[trace, p]
            worst = max(worst, total)
        if terminal:
            sizes[:] = 0
        trace, current, successor = current, successor, trace
    return worst


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_h_stress(dtype: type) -> None:
    weights = np.zeros(64, dtype)
    # The anchor starts with a step size far too high for its eight reads; the rest start where the real run does
    optimiser = Tidbd.create(64, 0.005, 0.05, 10_000.0, dtype, local=True)
    optimiser.rates[0, BETA] = np.log(0.5)
    stats = np.zeros(len(autotidbd.STATS))
    worst = hammer(weights, optimiser, 3_000_000, 1, stats)
    assert stats[autotidbd.UPDATES] > 2_900_000 and stats[autotidbd.NON_FINITE] == 0
    # Each cut sticks, so a handful of events is all it takes: the anchor's first cut alone was by a factor of 15
    assert stats[autotidbd.NORMALISATIONS] >= 5 and stats[autotidbd.MAX_M] > 10
    assert np.isfinite(weights).all() and np.isfinite(optimiser.rates).all()
    assert (optimiser.rates[:, XI] >= 0).all() and optimiser.shared[0] == 0.0
    # No step leaves the measured sum above 1 (but for float32's rounding of β)
    assert worst <= 1.0 + (1e-9 if dtype is np.float64 else 1e-4)
    # The anchor has been cut to where its eight reads can't overshoot, and no rate has run away
    assert np.exp(optimiser.rates[0, BETA]) < 1 / 16
    assert np.exp(optimiser.rates[:, BETA].max()) < 1.0


def test_h_threads_share_the_local_optimiser() -> None:
    network, optimiser = small_network(local=True)
    bb.seed(12)
    results = play_games_threaded(400, network.weights, None, network.layout, 0.0, True, None, 4, None, None, optimiser)
    stats = autotidbd.combine_stats(results.stats)
    assert stats[autotidbd.UPDATES] == results.moves.sum() and stats[autotidbd.NON_FINITE] == 0
    assert stats[autotidbd.NORMALISATIONS] > 0 and optimiser.shared[0] == 0.0
    assert np.isfinite(network.weights).all() and np.isfinite(optimiser.rates).all()


# The proportional rule: the cut is shared by blame rather than equally


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_proportional_brings_the_measured_step_to_exactly_one(dtype: type) -> None:
    pair = Pair(alpha=0.3, theta=0.01, dtype=dtype, proportional=True, w=np.arange(N, dtype=np.float64))
    # Weight 5 comes back with a higher count, so it pulls the sum down: a negative contribution
    trace, x, x_next = vector(i0=2, i3=1, i5=1), vector(i0=2, i3=3, i5=1, i7=1), vector(i0=1, i5=2, i8=1)
    pair.step(vector(i1=1), trace, reward=1.0)
    pair.step(trace, x, reward=0.5)
    pair.step(x, x_next, reward=1.0)
    assert pair.last_m > 1.1 and pair.stats[autotidbd.NORMALISATIONS] >= 1
    assert normaliser_sum(pair, x, x_next, trace) == pytest.approx(1.0, abs=1e-11 if dtype is np.float64 else 1e-5)
    assert pair.optimiser.shared[0] == 0.0


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("gamma", [1.0, 0.9])
def test_proportional_sparse_update_matches_the_dense_reference(dtype: type, gamma: float) -> None:
    rng = np.random.default_rng(13)
    pair = Pair(alpha=0.03, theta=0.02, dtype=dtype, gamma=gamma, proportional=True, w=rng.normal(size=N))
    # As in the real run: one weight with a step size near the bound, which comes and goes among ordinary ones
    pair.beta[0] = pair.optimiser.rates[0, BETA] = np.log(0.9)
    # The kernel finds λ by Newton's method and the reference by bisection: they agree to rounding at each event (the
    # bound test holds one to 1e-11), which later steps can amplify a little
    pair.tolerance = max(pair.tolerance, 1e-8)

    def features() -> np.ndarray:
        x = (rng.random(N) < 0.4) * rng.integers(1, 3, size=N).astype(np.float64)
        x[0] = rng.random() < 0.7
        return x

    for _ in range(40):
        pair.start_episode()
        x = features()
        for _ in range(rng.integers(1, 30)):
            x_next = features()
            if x.any():
                pair.step(x, x_next, float(rng.normal()))
            x = x_next
        if x.any():
            pair.step(x, NO_FEATURES, 0.0)
    # One cut by blame settles the culprit for good, so events are few; every step was checked against the reference
    assert pair.stats[autotidbd.NORMALISATIONS] >= 2 and pair.optimiser.shared[0] == 0.0
    assert np.abs(pair.weights).max() < 100


def test_proportional_cut_falls_on_the_weight_to_blame() -> None:
    # The diagnosed event in miniature: one weight at 0.9 and five neighbours at 0.03 leave together (sum 1.05); weight 7
    # stays (no contribution), and weight 8 is in the trace but not the update
    def run(**rule: bool) -> tuple[Pair, np.ndarray]:
        pair = Pair(alpha=0.03, theta=0.0, **rule)
        pair.beta[0] = pair.optimiser.rates[0, BETA] = np.log(0.9)
        trace = vector(i0=1, i1=1, i2=1, i3=1, i4=1, i5=1, i7=1, i8=1)
        x = vector(i0=1, i1=1, i2=1, i3=1, i4=1, i5=1, i7=1)
        pair.step(trace, x, reward=0.0)
        before = pair.optimiser.alpha()
        pair.step(x, vector(i7=1, i9=1), reward=0.0)
        return pair, pair.optimiser.alpha() / before

    local, local_kept = run(local=True)
    proportional, kept = run(proportional=True)
    assert local.last_m == proportional.last_m == pytest.approx(1.05)
    # Local: the same 4.8% off all eight in the trace, the bystander and the two that added nothing included
    np.testing.assert_allclose(local_kept[[0, 1, 2, 3, 4, 5, 7, 8]], 1 / 1.05)
    # Proportional: the culprit gives up 5.5%, each neighbour 0.2%, and the rest nothing at all
    assert kept[0] == pytest.approx(0.945, abs=0.001)
    np.testing.assert_allclose(kept[1:6], 0.998, atol=0.0005)
    np.testing.assert_array_equal(kept[6:], 1.0)
    # The culprit bears its share: its log cut is 0.9 / 0.03 = 30 times a neighbour's
    assert np.log(kept[0]) / np.log(kept[1]) == pytest.approx(30)


def test_proportional_forks_resumes_and_keeps_to_itself(tmp_path, small_default_network) -> None:  # noqa: F811
    settings = tidbd_settings(tmp_path, TIDBD_INITIAL_ALPHA=0.05)
    original = train(total_games=200, snapshot_at=[200], settings=settings).checkpoint_path
    loaded = autotidbd.load(original)
    ours = tidbd_settings(tmp_path, TIDBD_INITIAL_ALPHA=0.05, TIDBD_NORMALISATION="proportional", CHECKPOINT_DIR=str(tmp_path / "fork"), OUTPUT_DIR=str(tmp_path / "fork-output"))
    forked = train(resume_from=original, total_games=200, settings=ours, fork_normalisation=True).checkpoint_path
    fork = autotidbd.load(forked)
    assert loaded is not None and fork is not None and fork[0].local and fork[0].proportional
    np.testing.assert_array_equal(fork[0].alpha(), loaded[0].alpha())
    assert "AutoTIDBD proportional" in NTupleAgent(forked).label
    train(resume_from=forked, total_games=300, settings=ours)
    assert autotidbd.load(forked)[0].shared[0] == loaded[0].shared[0]
    # It resumes under neither other rule
    for other in ("global", "local"):
        with pytest.raises(ValueError, match="AutoTIDBD settings"):
            train(resume_from=forked, total_games=400, settings=tidbd_settings(tmp_path, TIDBD_INITIAL_ALPHA=0.05, TIDBD_NORMALISATION=other))


def test_proportional_reloaded_checkpoint_carries_on_exactly(tmp_path) -> None:
    def run(reload: bool) -> tuple[np.ndarray, Tidbd, np.ndarray]:
        network, _ = small_network()
        optimiser = Tidbd.create(network.weights.size, 0.05, 0.01, 10_000.0, np.float32, True, True)
        bb.seed(5)
        first = play_games(60, network.weights, None, network.layout, 0.0, True, None, None, None, optimiser)
        if reload:
            network.save(tmp_path / "state.npz", metadata("autotidbd-proportional"), None, autotidbd.checkpoint_arrays(optimiser, 0.05, np.zeros(len(autotidbd.STATS))))
            network, _ = NTupleNetwork.load(tmp_path / "state.npz")
            optimiser = autotidbd.load(tmp_path / "state.npz")[0]
            assert optimiser.proportional
        bb.seed(6)
        second = play_games(60, network.weights, None, network.layout, 0.0, True, None, None, None, optimiser)
        return network.weights, optimiser, first.stats + second.stats

    weights, optimiser, stats = run(reload=False)
    reloaded_weights, reloaded, _ = run(reload=True)
    np.testing.assert_array_equal(reloaded_weights, weights)
    np.testing.assert_array_equal(reloaded.rates, optimiser.rates)
    assert stats[autotidbd.NORMALISATIONS] > 0


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_proportional_stress(dtype: type) -> None:
    weights = np.zeros(64, dtype)
    optimiser = Tidbd.create(64, 0.005, 0.05, 10_000.0, dtype, local=True, proportional=True)
    optimiser.rates[0, BETA] = np.log(0.5)
    stats = np.zeros(len(autotidbd.STATS))
    worst = hammer(weights, optimiser, 3_000_000, 1, stats)
    assert stats[autotidbd.UPDATES] > 2_900_000 and stats[autotidbd.NON_FINITE] == 0
    assert stats[autotidbd.NORMALISATIONS] >= 5 and stats[autotidbd.MAX_M] > 10
    assert np.isfinite(weights).all() and np.isfinite(optimiser.rates).all() and optimiser.shared[0] == 0.0
    assert worst <= 1.0 + (1e-9 if dtype is np.float64 else 1e-4)
    assert np.exp(optimiser.rates[0, BETA]) < 1 / 16 and np.exp(optimiser.rates[:, BETA].max()) < 1.0


# The mean normaliser: ξ as a running mean of the meta-gradient's size, in place of its running maximum


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("rule", [{}, {"local": True}, {"proportional": True}], ids=["global", "local", "proportional"])
def test_mean_normaliser_sparse_update_matches_the_dense_reference(dtype: type, rule: dict) -> None:
    rng = np.random.default_rng(14)
    pair = Pair(alpha=0.03, theta=0.02, dtype=dtype, mean_decay=0.05, w=rng.normal(size=N), **rule)
    pair.tolerance = max(pair.tolerance, 1e-8)
    for _ in range(40):
        pair.start_episode()
        x = random_features(rng) > 0
        for _ in range(rng.integers(1, 30)):
            x_next = (random_features(rng) > 0) if rng.random() < 0.5 else np.where(rng.random(N) < 0.8, x, 0)
            if x.any():
                pair.step(x.astype(np.float64), x_next.astype(np.float64), float(rng.normal()))
            x = x_next
        if x.any():
            pair.step(x.astype(np.float64), NO_FEATURES, 0.0)
    assert np.ptp(pair.optimiser.rates[:, BETA]) > 0.05 and np.abs(pair.weights).max() < 100


def test_mean_normaliser_keeps_meta_steps_near_theta_after_an_outlier() -> None:
    """One huge error early on, then ordinary ones: as published every later meta-step is dwarfed by the outlier, and
    against the running mean they're soon a good part of θ again. Either way no step exceeds θ."""
    def run(mean_decay: float) -> tuple[float, float]:
        rng = np.random.default_rng(15)
        pair = Pair(alpha=0.01, theta=0.01, tau=10_000.0, mean_decay=mean_decay)
        x, other = vector(i0=1), vector(i1=1)
        steps = []
        for step in range(400):
            before = pair.beta[0]
            pair.step(x, other, reward=1e4 if step == 3 else float(rng.normal()))
            pair.step(other, x, reward=float(rng.normal()))
            steps.append(abs(pair.beta[0] - before))
        return float(np.mean(steps[200:])), float(np.max(steps))

    published, published_largest = run(0.0)
    mean, mean_largest = run(0.05)
    assert published < 0.01 * 0.01 and mean > 0.2 * 0.01
    # Each step() pair above makes two meta-steps on weight 0 (once in x, once in x'), each at most θ
    assert published_largest <= 0.02 * 1.0001 and mean_largest <= 0.02 * 1.0001


def test_mean_normaliser_forks_resumes_and_keeps_to_itself(tmp_path, small_default_network) -> None:  # noqa: F811
    settings = tidbd_settings(tmp_path, TIDBD_INITIAL_ALPHA=0.05)
    original = train(total_games=200, snapshot_at=[200], settings=settings).checkpoint_path
    loaded = autotidbd.load(original)
    ours = tidbd_settings(
        tmp_path, TIDBD_INITIAL_ALPHA=0.05, TIDBD_NORMALISATION="proportional", TIDBD_NORMALISER_DECAY=0.01, CHECKPOINT_DIR=str(tmp_path / "fork"), OUTPUT_DIR=str(tmp_path / "fork-output")
    )
    forked = train(resume_from=original, total_games=200, settings=ours, fork_normalisation=True).checkpoint_path
    fork = autotidbd.load(forked)
    assert loaded is not None and fork is not None and fork[0].proportional and fork[0].mean_decay == 0.01
    np.testing.assert_array_equal(fork[0].rates, loaded[0].rates)
    assert "AutoTIDBD proportional mean" in NTupleAgent(forked).label
    train(resume_from=forked, total_games=300, settings=ours)
    assert autotidbd.load(forked)[0].mean_decay == 0.01
    # It doesn't resume with the published normaliser, or another decay
    for decay in (0.0, 0.02):
        with pytest.raises(ValueError, match="AutoTIDBD settings"):
            train(resume_from=forked, total_games=400, settings=ours.model_copy(update={"TIDBD_NORMALISER_DECAY": decay}))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_mean_normaliser_stress(dtype: type) -> None:
    weights = np.zeros(64, dtype)
    optimiser = Tidbd.create(64, 0.005, 0.05, 10_000.0, dtype, local=True, proportional=True, mean_decay=0.01)
    optimiser.rates[0, BETA] = np.log(0.5)
    stats = np.zeros(len(autotidbd.STATS))
    worst = hammer(weights, optimiser, 3_000_000, 1, stats)
    assert stats[autotidbd.UPDATES] > 2_900_000 and stats[autotidbd.NON_FINITE] == 0
    assert np.isfinite(weights).all() and np.isfinite(optimiser.rates).all() and (optimiser.rates[:, XI] >= 0).all()
    assert worst <= 1.0 + (1e-9 if dtype is np.float64 else 1e-4)
