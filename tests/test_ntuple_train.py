import csv

from deep_2048.agents.ntuple import bitboard as bb
from deep_2048.agents.ntuple.config import NTupleSettings
from deep_2048.agents.ntuple.models import GameResults
from deep_2048.agents.ntuple.network import NTupleNetwork
from deep_2048.agents.ntuple.train import play_games, train

from tests.helpers import SMALL_TUPLES


def play(network: NTupleNetwork, n: int, learn: bool, seed: int) -> GameResults:
    bb.seed(seed)
    return play_games(n, network.weights, network.layout, 0.01, learn)


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
    _, metadata = NTupleNetwork.load(second.checkpoint_path)
    assert second.games_played == 300 and metadata.games_played == 300

    with open(tmp_path / "output" / "training_history.csv") as f:
        assert [int(row["games"]) for row in csv.DictReader(f)] == [100, 200, 300]


def test_settings_ignore_dqn_env_vars(monkeypatch) -> None:
    monkeypatch.setenv("LEARNING_RATE", "5e-5")
    monkeypatch.setenv("NTUPLE_TRAINING_GAMES", "123")
    settings = NTupleSettings()
    assert settings.LEARNING_RATE == 0.0025
    assert settings.TRAINING_GAMES == 123
