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

from deep_2048.agents.ntuple import bitboard as bb
from deep_2048.agents.ntuple.config import NTupleSettings, get_ntuple_settings
from deep_2048.agents.ntuple.models import CheckpointMetadata, GameResults, TrainingHistoryRow, TrainingSummary
from deep_2048.agents.ntuple.network import NTupleNetwork, TupleLayout, evaluate, update

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


def train(resume_from: str | None = None, total_games: int | None = None, settings: NTupleSettings | None = None) -> TrainingSummary:
    """Train until total_games have been played, including any in the resumed checkpoint. Ctrl-C saves and stops."""
    settings = settings or get_ntuple_settings()
    total_games = settings.TRAINING_GAMES if total_games is None else total_games
    checkpoint_path = Path(settings.CHECKPOINT_DIR) / CHECKPOINT_NAME
    history_path = Path(settings.OUTPUT_DIR) / HISTORY_NAME

    network, games_played = _load_or_create(resume_from)
    # Offset by games played so a resumed run doesn't replay the same tile spawns
    bb.seed(settings.SEED + games_played)
    print(f"Training n-tuple network ({network.weights.size:,} weights) to {total_games:,} games, lr {settings.LEARNING_RATE}")
    # Compile before timing starts, otherwise the first batch's games/s includes compilation
    play_games(0, network.weights, network.layout, settings.LEARNING_RATE, True)

    interrupted = False
    try:
        while games_played < total_games:
            batch = min(settings.LOG_EVERY, total_games - games_played)
            started = time.monotonic()
            results = play_games(batch, network.weights, network.layout, settings.LEARNING_RATE, True)
            games_played += batch

            row = TrainingHistoryRow.from_results(games_played, results, time.monotonic() - started)
            print(row, flush=True)
            _append_history(history_path, row)

            if games_played % settings.CHECKPOINT_EVERY == 0 and games_played < total_games:
                _save_checkpoint(network, checkpoint_path, games_played, settings.LEARNING_RATE)
    except KeyboardInterrupt:
        interrupted = True
        print(f"\nStopped at {games_played:,} games")

    _save_checkpoint(network, checkpoint_path, games_played, settings.LEARNING_RATE)
    return TrainingSummary(games_played=games_played, checkpoint_path=checkpoint_path, interrupted=interrupted)


def _load_or_create(resume_from: str | None) -> tuple[NTupleNetwork, int]:
    if resume_from is None:
        return NTupleNetwork(), 0

    network, metadata = NTupleNetwork.load(resume_from)
    print(f"Resuming from {resume_from} at {metadata.games_played:,} games")
    return network, metadata.games_played


def _append_history(path: Path, row: TrainingHistoryRow) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(TrainingHistoryRow.model_fields))
        if is_new:
            writer.writeheader()
        writer.writerow(row.model_dump())


def _save_checkpoint(network: NTupleNetwork, path: Path, games_played: int, learning_rate: float) -> None:
    network.save(path, CheckpointMetadata(games_played=games_played, learning_rate=learning_rate))
    print(f"  saved {path} ({games_played:,} games)")
