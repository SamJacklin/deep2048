"""TD(0) afterstate training (Szubert & Jaśkowski, 2014).

The agent always plays the move with the highest reward + V(afterstate), where the afterstate
is the board after the move but before the new tile spawns. The previous afterstate's value is
then moved towards that number, or towards 0 when the game ends. There's no exploration, since
the random tile spawns already give enough variety.

The learning rate is constant by default. It can instead decay over the session
(LEARNING_RATE_END), or be scaled per weight by temporal coherence (Beal & Smith, 1999).

A share of games can start from synthetic late-game positions (LATE_START_FRACTION), made by
doubling the tiles above a gap in positions from normal play, which keeps the game's dynamics.
"""

import csv
import time
from pathlib import Path

import numpy as np
from numba import njit

from deep_2048.agents.ntuple.config import NTupleSettings, get_ntuple_settings
from deep_2048.agents.ntuple.models import CheckpointMetadata, GameResults, LateStarts, TrainingHistoryRow, TrainingSummary
from deep_2048.agents.ntuple.network import NETWORKS, CoherenceTraces, NTupleNetwork, TupleLayout, evaluate, load_traces, network_name, update, update_coherent
from deep_2048.experiments import Environment, RunManifest, TrainingSession, command_line, now
from deep_2048.game import bitboard as bb

CHECKPOINT_NAME = "latest.npz"
HISTORY_NAME = "training_history.csv"


@njit
def _learn(weights: np.ndarray, traces: CoherenceTraces | None, layout: TupleLayout, afterstate: np.uint64, target: float, learning_rate: float) -> None:
    """Move the afterstate's value towards target, scaling each weight's step by its coherence if there are traces."""
    error = target - evaluate(weights, layout, afterstate)
    # numba compiles a separate version for traces=None with this branch removed, keeping the constant-rate path as fast as before
    if traces is None:
        update(weights, layout, afterstate, learning_rate * error)
    else:
        update_coherent(weights, traces, layout, afterstate, learning_rate, error)


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
def play_game(
    weights: np.ndarray,
    traces: CoherenceTraces | None,
    layout: TupleLayout,
    learning_rate: float,
    learn: bool,
    start: np.uint64,
    harvest_exponent: int,
) -> tuple[int, int, int, np.uint64]:
    """Play one greedy game, learning from it if `learn`, and return (score, max exponent, moves, harvested).

    The game starts from `start`, or a fresh board if it's 0. If `harvest_exponent` is set, `harvested`
    is the position where the game first made a 2**harvest_exponent tile, otherwise 0.
    """
    board = bb.initial_board() if start == np.uint64(0) else start
    harvested = np.uint64(0)
    harvest_reward = 1 << harvest_exponent if harvest_exponent > 0 else 0
    score = 0
    moves = 0
    previous_afterstate = np.uint64(0)

    while True:
        best_afterstate, best_reward, best_value = _best_move(weights, layout, board)
        game_over = best_value == -np.inf
        if learn and moves > 0:
            _learn(weights, traces, layout, previous_afterstate, 0.0 if game_over else best_value, learning_rate)

        if game_over:
            return score, bb.max_exponent(board), moves, harvested

        previous_afterstate = best_afterstate
        score += best_reward
        moves += 1
        board = bb.spawn_tile(best_afterstate)
        # Only a merge scoring at least the harvest tile can make it, so most moves skip the board scan
        if harvest_reward > 0 and harvested == np.uint64(0) and best_reward >= harvest_reward and bb.max_exponent(board) >= harvest_exponent:
            harvested = board


@njit
def play_games(
    n: int,
    weights: np.ndarray,
    traces: CoherenceTraces | None,
    layout: TupleLayout,
    learning_rate: float,
    learn: bool,
    late_starts: LateStarts | None = None,
) -> GameResults:
    """Play n games, starting a share of them from `late_starts` if given."""
    results = GameResults(np.zeros(n, np.int64), np.zeros(n, np.int64), np.zeros(n, np.int64), np.zeros(n, np.bool_), np.zeros(n, np.uint64))
    for i in range(n):
        start = np.uint64(0)
        harvest_exponent = 0
        if late_starts is not None:
            harvest_exponent = late_starts.harvest_exponent
            if late_starts.pool.size > 0 and np.random.random() < late_starts.fraction:
                start = late_starts.pool[np.random.randint(late_starts.pool.size)]
                # Synthetic games don't donate positions, so starts aren't upgraded again and again
                harvest_exponent = 0
                results.synthetic[i] = True
        game = play_game(weights, traces, layout, learning_rate, learn, start, harvest_exponent)
        results.scores[i], results.max_exponents[i], results.moves[i], results.harvested[i] = game
    return results


@njit
def upgrade_all(boards: np.ndarray) -> np.ndarray:
    """Upgrade each board above its largest gap, keeping only those that changed."""
    upgraded = np.zeros(boards.size, dtype=np.uint64)
    kept = 0
    for board in boards:
        candidate = bb.upgrade(board)
        if candidate != board:
            upgraded[kept] = candidate
            kept += 1
    return upgraded[:kept]


def train(
    resume_from: str | Path | None = None,
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

    network, progress = _load_or_create(resume_from, settings)
    traces = _load_or_create_traces(resume_from, network, settings)
    start_games = progress.games_played
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
    schedule = f" decaying to {settings.LEARNING_RATE_END}" if settings.LEARNING_RATE_END is not None else ""
    coherence = ", temporal coherence" if settings.TEMPORAL_COHERENCE else ""
    print(f"Training {settings.NETWORK} n-tuple network ({network.weights.size:,} weights) to {total_games:,} games, lr {settings.LEARNING_RATE}{schedule}{coherence}")
    late_starts = _late_starts(settings)
    # Compile before timing starts, otherwise the first batch's games/s includes compilation
    play_games(0, network.weights, traces, network.layout, settings.LEARNING_RATE, True, late_starts)

    interrupted = False
    try:
        while progress.games_played < total_games:
            next_stop = min([total_games, *(s for s in snapshots if s > progress.games_played)])
            batch = min(settings.LOG_EVERY, next_stop - progress.games_played)
            progress.learning_rate = learning_rate_at(settings, start_games, total_games, progress.games_played)
            started = time.monotonic()
            results = play_games(batch, network.weights, traces, network.layout, progress.learning_rate, True, late_starts)
            if late_starts is not None:
                late_starts = late_starts._replace(pool=_grow_pool(late_starts.pool, results, settings.LATE_START_POOL))
            progress.games_played += batch
            progress.moves_played += int(results.moves.sum())

            row = TrainingHistoryRow.from_results(progress.games_played, progress.moves_played, results, time.monotonic() - started)
            print(row, flush=True)
            _append_history(history_path, row)

            if progress.games_played in snapshots:
                _save_checkpoint(network, checkpoint_dir / f"games_{progress.games_played}.npz", progress, traces)
            if progress.games_played % settings.CHECKPOINT_EVERY == 0 and progress.games_played < total_games:
                _save_checkpoint(network, checkpoint_path, progress, traces)
    except KeyboardInterrupt:
        interrupted = True
        print(f"\nStopped at {progress.games_played:,} games")

    _save_checkpoint(network, checkpoint_path, progress, traces)
    session.finished_at, session.finished_with, session.interrupted = now(), _counts(progress), interrupted
    manifest.save(output_dir)
    return TrainingSummary(games_played=progress.games_played, checkpoint_path=checkpoint_path, interrupted=interrupted)


def _load_or_create(resume_from: str | Path | None, settings: NTupleSettings) -> tuple[NTupleNetwork, CheckpointMetadata]:
    """Return the network and how much training it has had, recording the learning setup used from now on."""
    learning = {"learning_rate": settings.LEARNING_RATE, "temporal_coherence": settings.TEMPORAL_COHERENCE}
    if settings.DOWNGRADE_TILE is not None and settings.DOWNGRADE_TILE & (settings.DOWNGRADE_TILE - 1):
        raise ValueError(f"DOWNGRADE_TILE must be a power of two, not {settings.DOWNGRADE_TILE}")
    features = {"downgrade_tile": settings.DOWNGRADE_TILE, "scale_feature": settings.SCALE_FEATURE}
    if resume_from is None:
        network = NTupleNetwork(NETWORKS[settings.NETWORK], downgrade_tile=settings.DOWNGRADE_TILE, scale_feature=settings.SCALE_FEATURE)
        return network, CheckpointMetadata(games_played=0, moves_played=0, **learning)

    network, metadata = NTupleNetwork.load(resume_from)
    if network_name(network.tuples) != settings.NETWORK:
        raise ValueError(f"{resume_from} holds a {network_name(network.tuples) or 'custom'} network, not {settings.NETWORK}: set NTUPLE_NETWORK to match")
    if {"downgrade_tile": network.downgrade_tile, "scale_feature": network.scale_feature} != features:
        raise ValueError(f"{resume_from} was trained with downgrade_tile={network.downgrade_tile}, scale_feature={network.scale_feature}: set the settings to match")
    print(f"Resuming from {resume_from} at {metadata.games_played:,} games")
    return network, metadata.model_copy(update=learning)


def _load_or_create_traces(resume_from: str | Path | None, network: NTupleNetwork, settings: NTupleSettings) -> CoherenceTraces | None:
    """Return the coherence traces to train with: None if temporal coherence is off, else resumed or fresh."""
    if not settings.TEMPORAL_COHERENCE:
        return None
    resumed = load_traces(resume_from) if resume_from is not None else None
    return resumed if resumed is not None else CoherenceTraces.zeros(network.weights.size)


def learning_rate_at(settings: NTupleSettings, start_games: int, total_games: int, games_played: int) -> float:
    """Return the rate for the next batch, decaying geometrically over the session if LEARNING_RATE_END is set."""
    if settings.LEARNING_RATE_END is None or total_games <= start_games:
        return settings.LEARNING_RATE
    progress = (games_played - start_games) / (total_games - start_games)
    return settings.LEARNING_RATE * (settings.LEARNING_RATE_END / settings.LEARNING_RATE) ** progress


def _late_starts(settings: NTupleSettings) -> LateStarts | None:
    """Return an empty pool of synthetic starts if they're enabled. The pool isn't checkpointed: a resumed run refills it."""
    if settings.LATE_START_FRACTION <= 0:
        return None
    if settings.LATE_START_TILE & (settings.LATE_START_TILE - 1):
        raise ValueError(f"LATE_START_TILE must be a power of two, not {settings.LATE_START_TILE}")
    return LateStarts(np.zeros(0, dtype=np.uint64), settings.LATE_START_FRACTION, settings.LATE_START_TILE.bit_length() - 1)


def _grow_pool(pool: np.ndarray, results: GameResults, size: int) -> np.ndarray:
    """Add upgraded versions of the batch's donated positions to the pool, keeping the newest `size`."""
    donated = results.harvested[(results.harvested != 0) & ~results.synthetic]
    return np.concatenate([pool, upgrade_all(donated)])[-size:]


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


def _save_checkpoint(network: NTupleNetwork, path: Path, metadata: CheckpointMetadata, traces: CoherenceTraces | None) -> None:
    network.save(path, metadata, traces)
    print(f"  saved {path} ({metadata.games_played:,} games)", flush=True)
