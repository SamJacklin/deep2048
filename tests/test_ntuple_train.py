import csv

import numpy as np
import pytest

from deep_2048.agents.ntuple import NTupleAgent
from deep_2048.agents.ntuple.config import NTupleSettings
from deep_2048.agents.ntuple.models import GameResults, LambdaReturn, LateStarts
from deep_2048.agents.ntuple.network import NETWORKS, NTupleNetwork, load_traces
from deep_2048.agents.ntuple.train import lambda_at, lambda_return, learning_rate_at, play_games, play_games_threaded, train
from deep_2048.experiments import RunManifest, run_lock
from deep_2048.game import bitboard as bb

from tests.helpers import SMALL_TUPLES, random_boards


def play(network: NTupleNetwork, n: int, learn: bool, seed: int) -> GameResults:
    bb.seed(seed)
    return play_games(n, network.weights, None, network.layout, 0.01, learn)


def test_no_learning_leaves_weights_alone() -> None:
    network = NTupleNetwork(tuples=SMALL_TUPLES)
    results = play(network, 20, learn=False, seed=0)

    assert not network.weights.any()
    assert (results.scores > 0).all() and (results.moves > 0).all() and (results.max_exponents >= 3).all()


def test_learning_improves_play() -> None:
    network = NTupleNetwork(tuples=SMALL_TUPLES)
    before = play(network, 200, learn=False, seed=0)
    play(network, 2_000, learn=True, seed=1)
    after = play(network, 200, learn=False, seed=0)

    assert after.scores.mean() > 1.5 * before.scores.mean()


def test_train_checkpoints_and_resumes(tmp_path) -> None:
    settings = NTupleSettings(
        LOG_EVERY=100,
        CHECKPOINT_EVERY=100,
        CHECKPOINT_DIR=str(tmp_path / "checkpoints"),
        OUTPUT_DIR=str(tmp_path / "output"),
    )

    first = train(total_games=200, settings=settings)
    _, metadata = NTupleNetwork.load(first.checkpoint_path)
    assert metadata.games_played == 200

    second = train(resume_from=first.checkpoint_path, total_games=300, settings=settings)
    _, resumed = NTupleNetwork.load(second.checkpoint_path)
    assert second.games_played == 300 and resumed.games_played == 300
    assert resumed.moves_played > metadata.moves_played > 0

    with open(tmp_path / "output" / "training_history.csv") as f:
        rows = list(csv.DictReader(f))
    assert [int(row["games"]) for row in rows] == [100, 200, 300]
    assert int(rows[-1]["moves"]) == resumed.moves_played

    manifest = RunManifest.load_or_create(tmp_path / "output")
    first_session, second_session = manifest.sessions
    assert first_session.resumed_from is None and second_session.resumed_from == str(first.checkpoint_path)
    assert first_session.finished_with == second_session.started_with == {"games_played": 200, "moves_played": metadata.moves_played}
    assert second_session.settings["TRAINING_GAMES"] == 300 and second_session.interrupted is False


def test_train_saves_snapshots_and_starts_a_fresh_history(tmp_path) -> None:
    settings = NTupleSettings(
        LOG_EVERY=100,
        CHECKPOINT_DIR=str(tmp_path / "checkpoints"),
        OUTPUT_DIR=str(tmp_path / "output"),
    )
    train(total_games=100, settings=settings)
    train(total_games=250, snapshot_at=[50, 250], settings=settings)

    _, snapshot = NTupleNetwork.load(tmp_path / "checkpoints" / "games_50.npz")
    assert snapshot.games_played == 50
    assert (tmp_path / "checkpoints" / "games_250.npz").exists()

    with open(tmp_path / "output" / "training_history.csv") as f:
        assert [int(row["games"]) for row in csv.DictReader(f)] == [50, 150, 250]


def test_learning_rate_decays_geometrically_over_the_session() -> None:
    settings = NTupleSettings(LEARNING_RATE=0.01, LEARNING_RATE_END=0.0001)
    rates = [learning_rate_at(settings, 100, 300, games) for games in (100, 200, 300)]
    assert np.allclose(rates, [0.01, 0.001, 0.0001])
    assert learning_rate_at(NTupleSettings(LEARNING_RATE=0.01), 100, 300, 200) == 0.01


def test_temporal_coherence_traces_are_checkpointed_and_resumed(tmp_path) -> None:
    settings = NTupleSettings(
        TEMPORAL_COHERENCE=True,
        LOG_EVERY=100,
        CHECKPOINT_DIR=str(tmp_path / "checkpoints"),
        OUTPUT_DIR=str(tmp_path / "output"),
    )
    first = train(total_games=100, settings=settings)
    _, metadata = NTupleNetwork.load(first.checkpoint_path)
    traces = load_traces(first.checkpoint_path)
    assert metadata.temporal_coherence and traces is not None
    first_errors = traces.abs_errors.sum()
    assert first_errors > 0

    train(resume_from=first.checkpoint_path, total_games=200, settings=settings)
    resumed = load_traces(first.checkpoint_path)
    assert resumed is not None and resumed.abs_errors.sum() > first_errors

    # Resetting keeps the weights but starts the sums from zero, so one more game leaves far smaller sums
    reset_settings = settings.model_copy(update={"CHECKPOINT_DIR": str(tmp_path / "reset")})
    reset = train(resume_from=first.checkpoint_path, total_games=201, settings=reset_settings, reset_coherence=True)
    reset_traces = load_traces(reset.checkpoint_path)
    assert reset_traces is not None and 0 < reset_traces.abs_errors.sum() < first_errors / 10


def test_train_builds_the_configured_network(tmp_path) -> None:
    settings = NTupleSettings(NETWORK="4x6", LOG_EVERY=50, CHECKPOINT_DIR=str(tmp_path / "checkpoints"), OUTPUT_DIR=str(tmp_path / "output"))
    summary = train(total_games=50, settings=settings)
    network, _ = NTupleNetwork.load(summary.checkpoint_path)
    assert network.tuples == NETWORKS["4x6"]
    assert NTupleAgent(summary.checkpoint_path).label == "N-tuple TD (50 games, lr 0.0025, 4x6 network)"
    assert RunManifest.load_or_create(tmp_path / "output").sessions[0].settings["NETWORK"] == "4x6"

    with pytest.raises(ValueError, match="holds a 4x6 network"):
        train(resume_from=summary.checkpoint_path, total_games=100, settings=settings.model_copy(update={"NETWORK": "8x6"}))


def test_every_game_can_start_from_a_late_position() -> None:
    network = NTupleNetwork(tuples=SMALL_TUPLES)
    pool = np.array([bb.from_array(board) for board in random_boards(5, seed=30)], dtype=np.uint64)
    bb.seed(0)
    results = play_games(20, network.weights, None, network.layout, 0.01, True, LateStarts(pool, 1.0, 8))
    assert results.synthetic.all() and not results.harvested.any()


def late_start_settings(tmp_path, name: str) -> NTupleSettings:
    return NTupleSettings(
        LATE_START_FRACTION=0.5,
        LATE_START_TILE=256,
        LOG_EVERY=100,
        CHECKPOINT_DIR=str(tmp_path / name / "checkpoints"),
        OUTPUT_DIR=str(tmp_path / name / "output"),
    )


def test_late_starts_are_reported_separately_and_reproducible(tmp_path) -> None:
    for name in ("first", "second"):
        train(total_games=300, settings=late_start_settings(tmp_path, name))

    histories = []
    for name in ("first", "second"):
        with open(tmp_path / name / "output" / "training_history.csv") as f:
            histories.append([{k: v for k, v in row.items() if k != "games_per_second"} for row in csv.DictReader(f)])
    assert histories[0] == histories[1]
    # The first batch has no pool yet, so its games are all normal
    assert [int(row["synthetic_games"]) for row in histories[0]][0] == 0
    assert sum(int(row["synthetic_games"]) for row in histories[0]) > 50


def test_late_start_tile_must_be_a_power_of_two(tmp_path) -> None:
    settings = late_start_settings(tmp_path, "bad").model_copy(update={"LATE_START_TILE": 300})
    with pytest.raises(ValueError, match="power of two"):
        train(total_games=100, settings=settings)


def test_late_start_tile_must_be_doublable(tmp_path) -> None:
    settings = late_start_settings(tmp_path, "too-big").model_copy(update={"LATE_START_TILE": 16384})
    with pytest.raises(ValueError, match="can't be doubled"):
        train(total_games=100, settings=settings)


def test_train_with_scale_sharing(tmp_path) -> None:
    settings = NTupleSettings(DOWNGRADE_TILE=256, SCALE_FEATURE=True, LOG_EVERY=50, CHECKPOINT_DIR=str(tmp_path / "checkpoints"), OUTPUT_DIR=str(tmp_path / "output"))
    summary = train(total_games=50, settings=settings)
    assert NTupleAgent(summary.checkpoint_path).label == "N-tuple TD (50 games, lr 0.0025, downgraded from 256, scale term)"

    with pytest.raises(ValueError, match="'scale_feature': True"):
        train(resume_from=summary.checkpoint_path, total_games=100, settings=settings.model_copy(update={"SCALE_FEATURE": False}))


def test_train_refuses_a_folder_another_process_is_training(tmp_path) -> None:
    settings = NTupleSettings(LOG_EVERY=50, CHECKPOINT_DIR=str(tmp_path / "checkpoints"), OUTPUT_DIR=str(tmp_path / "output"))
    with run_lock(tmp_path / "output"):
        with pytest.raises(RuntimeError, match="already being trained"):
            train(total_games=50, settings=settings)
    assert not (tmp_path / "checkpoints").exists()


def test_scale_term_when_downgraded_needs_both_features(tmp_path) -> None:
    settings = NTupleSettings(SCALE_WHEN_DOWNGRADED=True, SCALE_FEATURE=True, CHECKPOINT_DIR=str(tmp_path / "checkpoints"), OUTPUT_DIR=str(tmp_path / "output"))
    with pytest.raises(ValueError, match="needs SCALE_FEATURE and DOWNGRADE_TILE"):
        train(total_games=10, settings=settings)

    settings = settings.model_copy(update={"DOWNGRADE_TILE": 256, "LOG_EVERY": 50})
    summary = train(total_games=50, settings=settings)
    assert NTupleAgent(summary.checkpoint_path).label == "N-tuple TD (50 games, lr 0.0025, downgraded from 256, scale term when downgraded)"


def test_threaded_learning_improves_play() -> None:
    network = NTupleNetwork(tuples=SMALL_TUPLES)
    before = play(network, 200, learn=False, seed=0)
    bb.seed(1)
    learned = play_games_threaded(2_000, network.weights, None, network.layout, 0.01, True, None, threads=2)
    after = play(network, 200, learn=False, seed=0)

    assert (learned.moves > 0).all() and (learned.scores > 0).all()
    assert after.scores.mean() > 1.5 * before.scores.mean()


def test_train_switches_to_threads_after_the_warm_up(tmp_path) -> None:
    settings = NTupleSettings(THREADS=2, THREADS_AFTER=100, LOG_EVERY=100, CHECKPOINT_DIR=str(tmp_path / "checkpoints"), OUTPUT_DIR=str(tmp_path / "output"))
    summary = train(total_games=300, settings=settings)
    _, metadata = NTupleNetwork.load(summary.checkpoint_path)
    assert metadata.games_played == 300 and metadata.moves_played > 0

    session = RunManifest.load_or_create(tmp_path / "output").sessions[0]
    assert session.settings["THREADS"] == 2 and session.settings["THREADS_AFTER"] == 100
    with open(tmp_path / "output" / "training_history.csv") as f:
        assert [int(row["games"]) for row in csv.DictReader(f)] == [100, 200, 300]


def test_threads_work_with_scale_sharing_and_late_starts(tmp_path) -> None:
    settings = NTupleSettings(
        THREADS=2,
        THREADS_AFTER=50,
        LOG_EVERY=50,
        DOWNGRADE_TILE=256,
        SCALE_FEATURE=True,
        SCALE_WHEN_DOWNGRADED=True,
        LATE_START_FRACTION=0.5,
        LATE_START_TILE=256,
        CHECKPOINT_DIR=str(tmp_path / "checkpoints"),
        OUTPUT_DIR=str(tmp_path / "output"),
    )
    summary = train(total_games=200, settings=settings)
    with open(tmp_path / "output" / "training_history.csv") as f:
        rows = list(csv.DictReader(f))
    assert summary.games_played == 200 and sum(int(row["synthetic_games"]) for row in rows) > 0


def test_thread_count_must_be_available(tmp_path) -> None:
    settings = NTupleSettings(THREADS=10_000, CHECKPOINT_DIR=str(tmp_path / "checkpoints"), OUTPUT_DIR=str(tmp_path / "output"))
    with pytest.raises(ValueError, match="THREADS must be between"):
        train(total_games=10, settings=settings)


def test_lambda_return_blends_n_step_returns() -> None:
    # Ring of 4 with the oldest afterstate at 0: rewards and values for the three steps after it
    rewards = np.array([0.0, 4.0, 8.0, 16.0])
    values = np.array([0.0, 100.0, 200.0, 300.0])
    one, two, three = 4 + 100, 4 + 8 + 200, 4 + 8 + 16 + 300
    assert lambda_return(rewards, values, 0, 3, False, 0.5, 3) == 0.5 * one + 0.25 * two + 0.25 * three

    # The game ended after two steps, so the longer returns are just the rewards collected
    assert lambda_return(rewards, values, 0, 2, True, 0.5, 3) == 0.5 * one + 0.25 * (4 + 8) + 0.25 * (4 + 8)
    assert lambda_return(rewards, values, 0, 0, True, 0.5, 3) == 0.0


def test_lambda_with_horizon_one_is_exactly_td_zero() -> None:
    results = []
    for td_lambda in (None, LambdaReturn(0.7, 1)):
        network = NTupleNetwork(tuples=SMALL_TUPLES)
        bb.seed(3)
        played = play_games(300, network.weights, None, network.layout, 0.01, True, None, td_lambda)
        results.append((played.scores.copy(), network.weights.copy()))
    np.testing.assert_array_equal(results[0][0], results[1][0])
    np.testing.assert_array_equal(results[0][1], results[1][1])


def test_td_lambda_learning_improves_play() -> None:
    network = NTupleNetwork(tuples=SMALL_TUPLES)
    before = play(network, 200, learn=False, seed=0)
    bb.seed(1)
    play_games(2_000, network.weights, None, network.layout, 0.01, True, None, LambdaReturn(0.5, 3))
    after = play(network, 200, learn=False, seed=0)
    assert after.scores.mean() > 1.5 * before.scores.mean()


def test_lambda_moves_linearly_over_the_session() -> None:
    settings = NTupleSettings(LAMBDA=0.8, LAMBDA_END=0.0)
    assert [lambda_at(settings, 100, 300, games) for games in (100, 200, 300)] == pytest.approx([0.8, 0.4, 0.0])
    assert lambda_at(NTupleSettings(LAMBDA=0.5), 100, 300, 200) == 0.5


def test_lambda_settings_are_checked(tmp_path) -> None:
    settings = NTupleSettings(LAMBDA=1.5, CHECKPOINT_DIR=str(tmp_path / "checkpoints"), OUTPUT_DIR=str(tmp_path / "output"))
    with pytest.raises(ValueError, match="must be between 0 and 1"):
        train(total_games=10, settings=settings)


def test_settings_ignore_dqn_env_vars(monkeypatch) -> None:
    monkeypatch.setenv("LEARNING_RATE", "5e-5")
    monkeypatch.setenv("NTUPLE_TRAINING_GAMES", "123")
    settings = NTupleSettings()
    assert settings.LEARNING_RATE == 0.0025
    assert settings.TRAINING_GAMES == 123
