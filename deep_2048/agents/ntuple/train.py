"""TD(0) afterstate training (Szubert & Jaśkowski, 2014).

The agent always plays the move with the highest reward + V(afterstate), where the afterstate
is the board after the move but before the new tile spawns. The previous afterstate's value is
then moved towards that number, or towards 0 when the game ends. There's no exploration, since
the random tile spawns already give enough variety.
"""

import csv
import time
from pathlib import Path

import numpy as np
from numba import njit

from deep_2048.agents.ntuple.config import NTupleSettings, get_ntuple_settings
from deep_2048.agents.ntuple.models import CheckpointMetadata, GameResults, TrainingHistoryRow, TrainingSummary
from deep_2048.agents.ntuple.network import NTupleNetwork, TupleLayout, evaluate, update
from deep_2048.experiments import Environment, RunManifest, TrainingSession, command_line, now
from deep_2048.game import bitboard as bb

CHECKPOINT_NAME = "latest.npz"
HISTORY_NAME = "training_history.csv"


@njit
def _learn(weights: np.ndarray, layout: TupleLayout, afterstate: np.uint64, target: float, learning_rate: float) -> None:
    """Move the afterstate's value towards target."""
    error = target - evaluate(weights, layout, afterstate)
    update(weights, layout, afterstate, learning_rate * error)


@njit(inline="always")
def _best_move(weights: np.ndarray, layout: TupleLayout, board: np.uint64) -> tuple[np.uint64, int, float]:
    """Return the afterstate, reward and reward + value of the best move, with a value of -inf if none are legal."""
    best_afterstate = board
    best_reward = 0
    best_value = -np.inf
    for action in range(bb.NUM_ACTIONS):
        afterstate, reward = bb.move(board, action)
        if afterstate == board:
            continue
        value = reward + evaluate(weights, layout, afterstate)
        if value > best_value:
            best_afterstate, best_reward, best_value = afterstate, reward, value
    return best_afterstate, best_reward, best_value


@njit
def play_game(weights: np.ndarray, layout: TupleLayout, learning_rate: float, learn: bool) -> tuple[int, int, int]:
    """Play one greedy game, learning from it if `learn`, and return (score, max exponent, moves)."""
    board = bb.initial_board()
    score = 0
    moves = 0
    previous_afterstate = np.uint64(0)

    while True:
        best_afterstate, best_reward, best_value = _best_move(weights, layout, board)
        game_over = best_value == -np.inf
        if learn and moves > 0:
            _learn(weights, layout, previous_afterstate, 0.0 if game_over else best_value, learning_rate)

        if game_over:
            return score, bb.max_exponent(board), moves

        previous_afterstate = best_afterstate
        score += best_reward
        moves += 1
        board = bb.spawn_tile(best_afterstate)


@njit
def play_games(n: int, weights: np.ndarray, layout: TupleLayout, learning_rate: float, learn: bool) -> GameResults:
    results = GameResults(np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64))
    for i in range(n):
        results.scores[i], results.max_exponents[i], results.moves[i] = play_game(weights, layout, learning_rate, learn)
    return results


def train(
    resume_from: str | None = None,
    total_games: int | None = None,
    snapshot_at: list[int] | None = None,
    settings: NTupleSettings | None = None,
) -> TrainingSummary:
    """Train until total_games have been played, including any in the resumed checkpoint.

    The network is also saved as games_<n>.npz at each game count in snapshot_at, and each call is
    recorded as a session in the run's manifest. Ctrl-C saves and stops.
    """
    settings = settings or get_ntuple_settings()
    total_games = settings.TRAINING_GAMES if total_games is None else total_games
    snapshots = sorted(set(settings.SNAPSHOT_AT if snapshot_at is None else snapshot_at))
    checkpoint_dir = Path(settings.CHECKPOINT_DIR)
    checkpoint_path = checkpoint_dir / CHECKPOINT_NAME
    output_dir = Path(settings.OUTPUT_DIR)
    history_path = output_dir / HISTORY_NAME

    network, progress = _load_or_create(resume_from, settings.LEARNING_RATE)
    if resume_from is None:
        history_path.unlink(missing_ok=True)
    manifest = RunManifest(name=output_dir.name) if resume_from is None else RunManifest.load_or_create(output_dir)
    session = TrainingSession(
        command=command_line(),
        settings=settings.model_dump(mode="json") | {"TRAINING_GAMES": total_games, "SNAPSHOT_AT": snapshots},
        resumed_from=None if resume_from is None else str(resume_from),
        environment=Environment.capture(),
        started_at=now(),
        started_with=_counts(progress),
    )
    manifest.sessions.append(session)
    manifest.save(output_dir)

    # Offset by games played so a resumed run doesn't replay the same tile spawns
    bb.seed(settings.SEED + progress.games_played)
    print(f"Training n-tuple network ({network.weights.size:,} weights) to {total_games:,} games, lr {settings.LEARNING_RATE}")
    # Compile before timing starts, otherwise the first batch's games/s includes compilation
    play_games(0, network.weights, network.layout, settings.LEARNING_RATE, True)

    interrupted = False
    try:
        while progress.games_played < total_games:
            next_stop = min([total_games, *(s for s in snapshots if s > progress.games_played)])
            batch = min(settings.LOG_EVERY, next_stop - progress.games_played)
            started = time.monotonic()
            results = play_games(batch, network.weights, network.layout, settings.LEARNING_RATE, True)
            progress.games_played += batch
            progress.moves_played += int(results.moves.sum())

            row = TrainingHistoryRow.from_results(progress.games_played, progress.moves_played, results, time.monotonic() - started)
            print(row, flush=True)
            _append_history(history_path, row)

            if progress.games_played in snapshots:
                _save_checkpoint(network, checkpoint_dir / f"games_{progress.games_played}.npz", progress)
            if progress.games_played % settings.CHECKPOINT_EVERY == 0 and progress.games_played < total_games:
                _save_checkpoint(network, checkpoint_path, progress)
    except KeyboardInterrupt:
        interrupted = True
        print(f"\nStopped at {progress.games_played:,} games")

    _save_checkpoint(network, checkpoint_path, progress)
    session.finished_at, session.finished_with, session.interrupted = now(), _counts(progress), interrupted
    manifest.save(output_dir)
    return TrainingSummary(games_played=progress.games_played, checkpoint_path=checkpoint_path, interrupted=interrupted)


def _load_or_create(resume_from: str | None, learning_rate: float) -> tuple[NTupleNetwork, CheckpointMetadata]:
    """Return the network and how much training it has had, recording the learning rate used from now on."""
    if resume_from is None:
        return NTupleNetwork(), CheckpointMetadata(games_played=0, moves_played=0, learning_rate=learning_rate)

    network, metadata = NTupleNetwork.load(resume_from)
    print(f"Resuming from {resume_from} at {metadata.games_played:,} games")
    return network, metadata.model_copy(update={"learning_rate": learning_rate})


def _counts(progress: CheckpointMetadata) -> dict[str, int]:
    return {"games_played": progress.games_played, "moves_played": progress.moves_played}


def _append_history(path: Path, row: TrainingHistoryRow) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(TrainingHistoryRow.model_fields))
        if is_new:
            writer.writeheader()
        writer.writerow(row.model_dump())


def _save_checkpoint(network: NTupleNetwork, path: Path, metadata: CheckpointMetadata) -> None:
    network.save(path, metadata)
    print(f"  saved {path} ({metadata.games_played:,} games)", flush=True)
