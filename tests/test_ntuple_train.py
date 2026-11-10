import csv

import numpy as np

from deep_2048.agents.ntuple.config import NTupleSettings
from deep_2048.agents.ntuple.models import GameResults
from deep_2048.agents.ntuple.network import NTupleNetwork, load_traces
from deep_2048.agents.ntuple.train import learning_rate_at, play_games, train
from deep_2048.experiments import RunManifest
from deep_2048.game import bitboard as bb

from tests.helpers import SMALL_TUPLES


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


def test_settings_ignore_dqn_env_vars(monkeypatch) -> None:
    monkeypatch.setenv("LEARNING_RATE", "5e-5")
    monkeypatch.setenv("NTUPLE_TRAINING_GAMES", "123")
    settings = NTupleSettings()
    assert settings.LEARNING_RATE == 0.0025
    assert settings.TRAINING_GAMES == 123
