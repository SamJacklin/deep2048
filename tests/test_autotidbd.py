"""Stage 8's correctness tests: the training kernel against Algorithm 1 written out on dense vectors (tests A to H of the plan)."""

import numpy as np
import pytest

from deep_2048.agents.ntuple import NTupleAgent, autotidbd
from deep_2048.agents.ntuple.autotidbd import BETA, H, VISITS, XI, Tidbd, reference_step, sparse_step
from deep_2048.agents.ntuple.config import NTupleSettings
from deep_2048.agents.ntuple.network import NTupleNetwork, evaluate, weight_indices
from deep_2048.agents.ntuple.search import action_values, play_seeded
from deep_2048.agents.ntuple.train import play_games, play_games_threaded, train
from deep_2048.game import bitboard as bb

from tests.helpers import SMALL_TUPLES

N = 12
NO_FEATURES = np.zeros(N)


def sparse(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    indices = np.flatnonzero(x)
    return indices.astype(np.int64), x[indices].astype(np.float64), len(indices)


class Pair:
    """The reference and the kernel side by side, fed the same transitions."""

    def __init__(
        self, alpha: float = 0.05, theta: float = 0.01, tau: float = 100.0, gamma: float = 1.0, dtype: type = np.float64, w: np.ndarray | None = None, local: bool = False
    ) -> None:
        self.gamma, self.theta, self.tau, self.local = gamma, theta, tau, local
        self.w = np.zeros(N) if w is None else w.astype(np.float64)
        self.beta, self.h, self.xi, self.z = np.full(N, np.log(alpha)), np.zeros(N), np.zeros(N), np.zeros(N)
        self.weights = self.w.astype(dtype)
        self.optimiser = Tidbd.create(N, alpha, theta, tau, dtype, local)
        self.work = (np.empty(2 * N, np.int64), np.empty(2 * N), np.empty(2 * N), np.empty(2 * N))
        self.stats = np.zeros(len(autotidbd.STATS))
        self.tolerance = 1e-10 if dtype is np.float64 else 2e-4
        self.last_m = 1.0

    def start_episode(self) -> None:
        self.z[:] = 0

    def step(self, x: np.ndarray, x_next: np.ndarray, reward: float) -> float:
        """Take one step in both, check that they agree on everything, and return δ."""
        trace = sparse(self.z)
        delta, self.last_m = reference_step(self.w, self.beta, self.h, self.xi, self.z, x, x_next, reward, self.gamma, self.theta, self.tau, local=self.local)
        got = sparse_step(self.weights, self.optimiser, self.gamma, reward, sparse(x), sparse(x_next), trace, self.work, self.stats)
        rates, offset = self.optimiser.rates, self.optimiser.shared[0]
        close = dict(rtol=self.tolerance, atol=self.tolerance)
        np.testing.assert_allclose(got, delta, **close)
        np.testing.assert_allclose(self.weights, self.w, **close)
        np.testing.assert_allclose(rates[:, BETA] + offset, self.beta, **close)
        np.testing.assert_allclose(rates[:, H], self.h, **close)
        np.testing.assert_allclose(rates[:, XI], self.xi, **close)
        return delta


def vector(**counts: float) -> np.ndarray:
    """A feature vector from entries like i3=2 (weight 3 read twice)."""
    x = np.zeros(N)
    for name, count in counts.items():
        x[int(name[1:])] = count
    return x


def random_features(rng: np.random.Generator) -> np.ndarray:
    """A few active weights, some read several times."""
    x = np.zeros(N)
    active = rng.choice(N, size=rng.integers(1, 6), replace=False)
    x[active] = rng.choice([1, 1, 1, 2, 3, 8], size=len(active))
    return x


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("gamma", [1.0, 0.9])
def test_a_matches_the_reference_over_consecutive_transitions(dtype: type, gamma: float) -> None:
    rng = np.random.default_rng(0)
    pair = Pair(dtype=dtype, gamma=gamma, w=rng.normal(size=N))
    for _ in range(20):
        pair.start_episode()
        x = random_features(rng)
        for _ in range(rng.integers(1, 30)):
            x_next = random_features(rng)
            pair.step(x, x_next, float(rng.normal()))
            x = x_next
        pair.step(x, NO_FEATURES, 0.0)
    assert pair.stats[autotidbd.UPDATES] > 200
    # The run must have exercised step-size adaptation, both ways, and the normalisation
    log_alpha = pair.optimiser.rates[:, BETA] + pair.optimiser.shared[0]
    assert log_alpha.max() - log_alpha.min() > 0.05
    assert pair.stats[autotidbd.NORMALISATIONS] > 0


def test_b_a_weight_read_several_times_is_one_feature_with_that_value() -> None:
    pair = Pair(w=np.arange(N, dtype=np.float64))
    x = vector(i3=3, i5=1)
    # V = 3 w3 + w5 = 14, and V' = 2 w3 = 6
    assert pair.step(x, vector(i3=2), reward=1.0) == pytest.approx(1 + 6 - 14)
    # Line 16 moves the weight by α δ x, so three reads move it three steps and the value nine
    assert pair.w[3] - 3 == pytest.approx(3 * 0.05 * -7)
    # ... and line 17's memory follows the same step, having been 0
    assert pair.h[3] == pytest.approx(3 * 0.05 * -7)
    pair.step(vector(i3=2), vector(i3=8, i5=2), reward=0.0)
    pair.step(vector(i3=8, i5=2), x, reward=2.0)
    assert pair.optimiser.rates[3, VISITS] == 3


def test_b_features_count_repeated_reads_on_real_boards() -> None:
    network = NTupleNetwork(tuples=SMALL_TUPLES, downgrade_tile=64, scale_feature=True)
    rng = np.random.default_rng(1)
    reads = len(SMALL_TUPLES) * bb.NUM_SYMMETRIES + 1
    indices, counts = np.empty(reads, np.int64), np.empty(reads)
    seen_repeats = False
    # The empty-cornered and symmetric boards are the ones whose symmetries read the same weight
    boards = [np.uint64(0), np.uint64(0x1), np.uint64(0x1001), *(np.uint64(rng.integers(0, 2**62)) for _ in range(200))]
    for board in boards:
        n = autotidbd.features.py_func(network.layout, board, board, indices, counts)
        expected, expected_counts = np.unique(weight_indices(network.layout, board, board), return_counts=True)
        np.testing.assert_array_equal(indices[:n], expected)
        np.testing.assert_array_equal(counts[:n], expected_counts)
        # The feature vector must give back the network's own value
        assert counts[:n].sum() == len(weight_indices(network.layout, board, board))
        seen_repeats |= bool((expected_counts > 1).any())
    assert seen_repeats


@pytest.mark.parametrize(
    "x_next",
    [vector(i1=1, i4=2, i7=1), vector(i2=1, i8=3), vector(i1=1, i8=1, i7=1), vector(i1=2, i4=2, i7=4), NO_FEATURES],
    ids=["identical", "no overlap", "partial overlap", "repeated indices", "terminal"],
)
def test_c_successor_features_enter_the_meta_gradient(x_next: np.ndarray) -> None:
    rng = np.random.default_rng(2)
    x = vector(i1=1, i4=2, i7=1)
    pair = Pair(w=rng.normal(size=N))
    # A first transition into x gives h something to correlate with and leaves a trace
    pair.step(vector(i1=1, i2=1, i4=1), x, reward=0.5)
    h, beta, xi = pair.h.copy(), pair.beta.copy(), pair.xi.copy()
    delta = pair.step(x, x_next, reward=1.0)
    # Line 10, worked by hand: β moves along −δ (γx' − x) h, scaled by ξ
    d = x_next - x
    moved = np.flatnonzero(d * h)
    expected = beta.copy()
    expected[moved] -= pair.theta * delta * d[moved] * h[moved] / pair.xi[moved]
    np.testing.assert_allclose(pair.beta, expected - np.log(pair.last_m), atol=1e-12)
    if not d.any():
        # Identical features: the TD error doesn't depend on these weights, so there's no meta-gradient at all
        np.testing.assert_array_equal(pair.beta, beta)
        np.testing.assert_array_equal(pair.xi, xi)
    # Weights only in the successor learn a step size without being updated themselves
    only_next = (x_next > 0) & (x == 0)
    np.testing.assert_array_equal(pair.h[only_next], h[only_next])


def test_d_an_excessive_effective_step_is_normalised_away() -> None:
    # Rate 0.5 on a weight read twice in the trace and in x: line 11's Σ α (x − γx') z is 0.5 * 2 * 2 = 2
    pair = Pair(alpha=0.5, theta=0.0)
    x = vector(i0=2)
    pair.step(x, x, reward=1.0)
    assert pair.last_m == 1.0
    pair.step(x, NO_FEATURES, reward=1.0)
    assert pair.last_m == pytest.approx(2.0)
    # Every weight's rate is divided by M, visited or not
    np.testing.assert_allclose(pair.optimiser.alpha(), 0.25)
    assert pair.stats[autotidbd.NORMALISATIONS] == 1 and pair.stats[autotidbd.MAX_M] == pytest.approx(2.0)

    # With no trace (a game's first update) nothing is normalised, as the listing has it: the step overshoots and is counted
    fresh = Pair(alpha=0.5, theta=0.0)
    fresh.step(x, NO_FEATURES, reward=1.0)
    assert fresh.last_m == 1.0
    np.testing.assert_allclose(fresh.optimiser.alpha(), 0.5)
    assert fresh.stats[autotidbd.MAX_STEP] == pytest.approx(2.0) and fresh.stats[autotidbd.OVERSHOOTS] == 1


def test_e_zero_meta_rate_leaves_only_the_normalisation() -> None:
    rng = np.random.default_rng(3)
    pair = Pair(alpha=0.2, theta=0.0, w=rng.normal(size=N))
    x = random_features(rng)
    for _ in range(200):
        x_next = random_features(rng)
        pair.step(x, x_next, float(rng.normal()))
        x = x_next
    # No weight's rate moves relative to another's: β is the initial value plus the shared shift
    np.testing.assert_array_equal(pair.optimiser.rates[:, BETA], np.full(N, np.log(0.2)))
    assert pair.optimiser.shared[0] < 0
    assert np.ptp(pair.beta) < 1e-12


def tidbd_setup(dtype: type = np.float32) -> tuple[NTupleNetwork, Tidbd]:
    network = NTupleNetwork(tuples=SMALL_TUPLES, downgrade_tile=64, scale_feature=True)
    network.weights = network.weights.astype(dtype)
    return network, Tidbd.create(network.weights.size, 0.01, 0.01, 10_000.0, dtype)


def test_f_a_reloaded_checkpoint_carries_on_exactly(tmp_path) -> None:
    def run(reload: bool) -> tuple[np.ndarray, Tidbd]:
        network, optimiser = tidbd_setup()
        bb.seed(5)
        play_games(40, network.weights, None, network.layout, 0.0, True, None, None, None, optimiser)
        if reload:
            metadata = autotidbd_metadata()
            network.save(tmp_path / "state.npz", metadata, None, autotidbd.checkpoint_arrays(optimiser, 0.01, np.zeros(len(autotidbd.STATS))), compress=True)
            network, _ = NTupleNetwork.load(tmp_path / "state.npz")
            loaded = autotidbd.load(tmp_path / "state.npz")
            assert loaded is not None
            optimiser = loaded[0]
        bb.seed(6)
        play_games(40, network.weights, None, network.layout, 0.0, True, None, None, None, optimiser)
        return network.weights, optimiser

    weights, optimiser = run(reload=False)
    reloaded_weights, reloaded = run(reload=True)
    # Single-threaded and with everything saved at full precision, the tolerance is zero
    np.testing.assert_array_equal(reloaded_weights, weights)
    np.testing.assert_array_equal(reloaded.rates, optimiser.rates)
    np.testing.assert_array_equal(reloaded.shared, optimiser.shared)
    assert (reloaded.theta, reloaded.tau) == (optimiser.theta, optimiser.tau)
    assert weights.any() and optimiser.rates[:, H].any()


def autotidbd_metadata():  # type: ignore[no-untyped-def]
    from deep_2048.agents.ntuple.models import CheckpointMetadata

    return CheckpointMetadata(games_played=40, moves_played=1, learning_rate=0.01, optimizer="autotidbd")


@pytest.fixture
def small_default_network(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Make the default layout tiny, so train()'s checkpoints (five arrays the size of the weights) are kilobytes, not most of a gigabyte."""
    from deep_2048.agents.ntuple import network

    monkeypatch.setitem(network.NETWORKS, network.DEFAULT_NETWORK, SMALL_TUPLES)


def tidbd_settings(tmp_path, **overrides) -> NTupleSettings:  # type: ignore[no-untyped-def]
    defaults = dict(
        OPTIMIZER="autotidbd", LOG_EVERY=50, CHECKPOINT_EVERY=50, DOWNGRADE_TILE=16384, LATE_START_FRACTION=0.2,
        CHECKPOINT_DIR=str(tmp_path / "checkpoints"), OUTPUT_DIR=str(tmp_path / "output"),
    )  # fmt: skip
    return NTupleSettings(**(defaults | overrides))


def test_f_training_resumes_only_with_its_own_state_and_settings(tmp_path, small_default_network) -> None:
    settings = tidbd_settings(tmp_path)
    first = train(total_games=100, snapshot_at=[100], settings=settings)
    assert autotidbd.load(first.checkpoint_path) is not None
    assert (tmp_path / "output" / "optimizer_stats" / "games_100.json").exists()
    assert len((tmp_path / "output" / "optimizer_history.csv").read_text().splitlines()) == 3

    resumed = train(resume_from=first.checkpoint_path, total_games=150, settings=settings)
    loaded = autotidbd.load(resumed.checkpoint_path)
    assert resumed.games_played == 150 and loaded is not None
    # The counters carry over a resume
    assert loaded[2][autotidbd.UPDATES] > autotidbd.load(tmp_path / "checkpoints" / "games_100.npz")[2][autotidbd.UPDATES]

    with pytest.raises(ValueError, match="AutoTIDBD settings"):
        train(resume_from=first.checkpoint_path, total_games=200, settings=tidbd_settings(tmp_path, TIDBD_META_RATE=0.003))
    plain = train(total_games=50, settings=NTupleSettings(LOG_EVERY=50, DOWNGRADE_TILE=16384, CHECKPOINT_DIR=str(tmp_path / "plain"), OUTPUT_DIR=str(tmp_path / "plain-output")))
    with pytest.raises(ValueError, match="no AutoTIDBD state"):
        train(resume_from=plain.checkpoint_path, total_games=100, settings=tidbd_settings(tmp_path))
    with pytest.raises(ValueError, match="doesn't work with"):
        train(total_games=50, settings=tidbd_settings(tmp_path, LAMBDA=0.5))


def test_g_playing_without_learning_is_untouched(tmp_path) -> None:
    network, optimiser = tidbd_setup()
    bb.seed(7)
    play_games(300, network.weights, None, network.layout, 0.0, True, None, None, None, optimiser)
    weights, rates = network.weights.copy(), optimiser.rates.copy()

    # With learning off, the optimiser's presence changes nothing: the same games, and no state moves
    bb.seed(8)
    with_optimiser = play_games(30, network.weights, None, network.layout, 0.0, False, None, None, None, optimiser)
    bb.seed(8)
    without = play_games(30, network.weights, None, network.layout, 0.0, False)
    np.testing.assert_array_equal(with_optimiser.scores, without.scores)
    np.testing.assert_array_equal(network.weights, weights)
    np.testing.assert_array_equal(optimiser.rates, rates)

    # A saved TIDBD checkpoint evaluates, plays greedily and searches at depth 3 exactly as the same weights saved plainly
    network.save(tmp_path / "tidbd.npz", autotidbd_metadata(), None, autotidbd.checkpoint_arrays(optimiser, 0.01, np.zeros(len(autotidbd.STATS))))
    network.save(tmp_path / "plain.npz", autotidbd_metadata().model_copy(update={"optimizer": None}))
    tidbd_agent, plain_agent = NTupleAgent(tmp_path / "tidbd.npz", depth=3), NTupleAgent(tmp_path / "plain.npz", depth=3)
    assert "AutoTIDBD" in tidbd_agent.label
    rng = np.random.default_rng(9)
    for _ in range(50):
        board = np.uint64(rng.integers(0, 2**40))
        assert evaluate(tidbd_agent.network.weights, tidbd_agent.network.layout, board, board) == evaluate(network.weights, network.layout, board, board)
        for depth in (1, 3):
            np.testing.assert_array_equal(
                action_values(tidbd_agent.network.weights, tidbd_agent.network.layout, board, depth), action_values(plain_agent.network.weights, plain_agent.network.layout, board, depth)
            )
    for depth in (1, 3):
        played = [play_seeded(agent.network.weights, agent.network.layout, depth, 0.0, np.random.default_rng(3)) for agent in (tidbd_agent, plain_agent)]
        assert played[0] == played[1]


def test_g_training_games_follow_the_reference_move_for_move() -> None:
    """Whole training games against the dense reference: the features, their repeats, the successor and its reward,
    the trace across moves and its reset each game, and the terminal update.

    Mirror-image moves tie exactly, and which wins turns on the weights' last bits, so the reference can't choose the
    moves itself. A step-by-step driver here plays the games from its own copy of the kernel's weights, feeding every
    transition to the kernel and to the reference: the trainer must match that copy bit for bit, and the reference
    must match it to rounding.
    """
    network, optimiser = tidbd_setup(np.float64)
    driven, driven_optimiser = tidbd_setup(np.float64)
    layout, n = network.layout, network.weights.size
    w, beta, h, xi, z = np.zeros(n), np.full(n, np.log(0.01)), np.zeros(n), np.zeros(n), np.zeros(n)
    work = (np.empty(2 * n, np.int64), np.empty(2 * n), np.empty(2 * n), np.empty(2 * n))
    stats = np.zeros(len(autotidbd.STATS))

    def dense(afterstate: np.uint64, state: np.uint64) -> np.ndarray:
        return np.bincount(weight_indices(layout, np.uint64(afterstate), np.uint64(state)), minlength=n).astype(np.float64)

    games = 25
    bb.seed(11)
    scores = []
    repeats = 0
    for _ in range(games):
        board, score, previous = np.uint64(bb.initial_board()), 0, None
        z[:] = 0
        while True:
            best = None
            for action in range(bb.NUM_ACTIONS):
                afterstate, reward = bb.move(board, action)
                if afterstate != board:
                    value = reward + evaluate(driven.weights, layout, np.uint64(afterstate), board)
                    if best is None or value > best[0]:
                        best = (value, afterstate, reward)
            if previous is not None:
                x_next, reward = (np.zeros(n), 0.0) if best is None else (dense(best[1], board), float(best[2]))
                repeats += int((previous > 1).any())
                sparse_step(driven.weights, driven_optimiser, 1.0, reward, sparse(previous), sparse(x_next), sparse(z), work, stats)
                reference_step(w, beta, h, xi, z, previous, x_next, reward, 1.0, 0.01, 10_000.0)
            if best is None:
                break
            previous = dense(best[1], board)
            score += best[2]
            board = np.uint64(bb.spawn_tile(np.uint64(best[1])))
        scores.append(score)

    bb.seed(11)
    results = play_games(games, network.weights, None, layout, 0.0, True, None, None, None, optimiser)
    assert repeats > 100
    np.testing.assert_array_equal(results.scores, scores)
    np.testing.assert_array_equal(network.weights, driven.weights)
    np.testing.assert_array_equal(optimiser.rates, driven_optimiser.rates)
    np.testing.assert_array_equal(optimiser.shared, driven_optimiser.shared)
    assert results.stats[autotidbd.UPDATES] == results.moves.sum() == stats[autotidbd.UPDATES]
    close = dict(rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(network.weights, w, **close)
    np.testing.assert_allclose(optimiser.rates[:, BETA] + optimiser.shared[0], beta, **close)
    np.testing.assert_allclose(optimiser.rates[:, H], h, **close)
    np.testing.assert_allclose(optimiser.rates[:, XI], xi, **close)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_h_stress(dtype: type) -> None:
    rng = np.random.default_rng(4)
    pair = Pair(alpha=0.1, theta=0.05, tau=50.0, dtype=dtype, w=rng.normal(size=N))
    # float32 state can't follow the reference through 1e6-sized errors to 1e-4, so there it's only held to staying finite
    check = pair.step if dtype is np.float64 else lambda x, x_next, reward: sparse_step(pair.weights, pair.optimiser, 1.0, reward, sparse(x), sparse(x_next), sparse(x), pair.work, pair.stats)
    x = vector(i0=8, i1=1)
    for step in range(3_000):
        # Alternating signs, bursts of huge and of vanishing errors, and weights 6 to 11 left dormant for long stretches
        scale = (1e6, 1e-30, 1.0)[(step // 100) % 3]
        dormant = (step // 500) % 2 == 0
        x_next = random_features(rng)
        if dormant:
            x_next[6:] = 0
        check(x, x_next, float(scale * (-1) ** step))
        x = x_next if x_next.any() else vector(i0=8, i1=1)
    assert np.isfinite(pair.weights).all() and np.isfinite(pair.optimiser.rates).all() and np.isfinite(pair.optimiser.shared).all()
    assert pair.stats[autotidbd.NON_FINITE] == 0
    assert (pair.optimiser.rates[:, XI] >= 0).all()
    # Every meta-step is at most θ, however large the error, so no rate can have run away
    assert np.abs(pair.optimiser.rates[:, BETA] - np.log(0.1)).max() <= 3_000 * 0.05 * 1.0001


def test_h_a_non_finite_error_is_skipped_and_counted_and_stops_training(tmp_path, small_default_network) -> None:
    pair = Pair(w=np.full(N, np.inf))
    before = pair.optimiser.rates.copy()
    x = vector(i0=1)
    delta = sparse_step(pair.weights, pair.optimiser, 1.0, 1.0, sparse(x), sparse(vector(i1=1)), sparse(NO_FEATURES), pair.work, pair.stats)
    assert np.isnan(delta) and pair.stats[autotidbd.NON_FINITE] == 1 and pair.stats[autotidbd.UPDATES] == 0
    np.testing.assert_array_equal(pair.optimiser.rates, before)

    settings = tidbd_settings(tmp_path)
    first = train(total_games=50, settings=settings)
    saved = first.checkpoint_path.read_bytes()
    with np.load(first.checkpoint_path) as data:
        arrays = dict(data)
    arrays["weights"][:] = np.inf
    np.savez(tmp_path / "broken.npz", **arrays)
    with pytest.raises(FloatingPointError, match="non-finite"):
        train(resume_from=tmp_path / "broken.npz", total_games=100, settings=settings)
    # The failed session wrote no checkpoint over the last good one
    assert first.checkpoint_path.read_bytes() == saved


def test_threads_share_the_optimiser() -> None:
    network, optimiser = tidbd_setup()
    bb.seed(12)
    results = play_games_threaded(200, network.weights, None, network.layout, 0.0, True, None, 4, None, None, optimiser)
    stats = autotidbd.combine_stats(results.stats)
    assert stats[autotidbd.UPDATES] == results.moves.sum() and stats[autotidbd.NON_FINITE] == 0
    assert np.isfinite(network.weights).all() and np.isfinite(optimiser.rates).all()


def test_existing_rules_are_unchanged_by_the_optimizer_setting() -> None:
    assert NTupleSettings().OPTIMIZER is None and not NTupleSettings().TEMPORAL_COHERENCE
    assert NTupleSettings(TEMPORAL_COHERENCE=True).TEMPORAL_COHERENCE
    assert NTupleSettings(OPTIMIZER="tc").TEMPORAL_COHERENCE
    assert not NTupleSettings(OPTIMIZER="autotidbd", TEMPORAL_COHERENCE=True).TEMPORAL_COHERENCE
