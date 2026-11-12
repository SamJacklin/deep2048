"""TD(0) afterstate training (Szubert & Jaśkowski, 2014).

The agent always plays the move with the highest reward + V(afterstate), where the afterstate
is the board after the move but before the new tile spawns. The previous afterstate's value is
then moved towards that number, or towards 0 when the game ends. There's no exploration, since
the random tile spawns already give enough variety.

The learning rate is constant by default. It can instead decay over the session
(LEARNING_RATE_END), or be scaled per weight by temporal coherence (Beal & Smith, 1999).

A share of games can start from synthetic late-game positions (LATE_START_FRACTION), made by
doubling the tiles above a gap in positions from normal play, which keeps the game's dynamics.

With LAMBDA > 0, learning is delayed TD(λ) as in Jaśkowski (2018): each afterstate is updated
LAMBDA_HORIZON moves after it was played, towards a blend of its 1- to LAMBDA_HORIZON-step returns
built from the rewards and the values computed when those moves were chosen.

With THREADS > 1, a batch's games are shared across threads that update the shared weights without
locks, as in Jaśkowski (2018). Updates are sparse, so threads rarely touch the same weight at once, but runs are
no longer bit-reproducible. The first THREADS_AFTER games stay single-threaded, because a fresh
network's few common positions make collisions frequent.
"""

import csv
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from numba import njit

from deep_2048.agents.ntuple.config import NTupleSettings, get_ntuple_settings
from deep_2048.agents.ntuple.models import CheckpointMetadata, GameResults, LambdaReturn, LateStarts, TrainingHistoryRow, TrainingSummary
from deep_2048.agents.ntuple.network import NETWORKS, CoherenceTraces, NTupleNetwork, TupleLayout, evaluate, load_traces, network_name, update, update_coherent
from deep_2048.experiments import Environment, RunManifest, TrainingSession, command_line, now, run_lock
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


@njit(inline="always")
def lambda_return(rewards: np.ndarray, values: np.ndarray, oldest: int, after: int, ended: bool, lam: float, horizon: int) -> float:
    """Return the truncated λ-return for the afterstate at ring position `oldest`.

    `after` steps follow it in the ring (rewards and afterstate values at those positions). If the
    game has ended, every return longer than those steps is just the rewards collected, since
    nothing more can be scored.
    """
    size = rewards.size
    target = 0.0
    collected = 0.0
    for n in range(1, horizon + 1):
        if n <= after:
            i = (oldest + n) % size
            collected += rewards[i]
            n_step = collected if ended and n == after else collected + values[i]
        else:
            n_step = collected
        weight = (1.0 - lam) * lam ** (n - 1) if n < horizon else lam ** (horizon - 1)
        target += weight * n_step
    return target


@njit
def play_game_lambda(
    weights: np.ndarray,
    traces: CoherenceTraces | None,
    layout: TupleLayout,
    learning_rate: float,
    learn: bool,
    start: np.uint64,
    harvest_exponent: int,
    td_lambda: LambdaReturn,
) -> tuple[int, int, int, np.uint64]:
    """As play_game, but each afterstate learns `horizon` moves later towards its truncated λ-return."""
    horizon = td_lambda.horizon
    size = horizon + 1
    afterstates = np.zeros(size, np.uint64)
    rewards = np.zeros(size)
    values = np.zeros(size)
    oldest = 0
    pending = 0
    board = bb.initial_board() if start == np.uint64(0) else start
    harvested = np.uint64(0)
    harvest_reward = 1 << harvest_exponent if harvest_exponent > 0 else 0
    score = 0
    moves = 0

    while True:
        best_afterstate, best_reward, best_value = _best_move(weights, layout, board)
        if best_value == -np.inf:
            while learn and pending > 0:
                target = lambda_return(rewards, values, oldest, pending - 1, True, td_lambda.lam, horizon)
                _learn(weights, traces, layout, afterstates[oldest], target, learning_rate)
                oldest = (oldest + 1) % size
                pending -= 1
            return score, bb.max_exponent(board), moves, harvested

        newest = (oldest + pending) % size
        afterstates[newest], rewards[newest], values[newest] = best_afterstate, best_reward, best_value - best_reward
        pending += 1
        if pending == size:
            if learn:
                target = lambda_return(rewards, values, oldest, horizon, False, td_lambda.lam, horizon)
                _learn(weights, traces, layout, afterstates[oldest], target, learning_rate)
            oldest = (oldest + 1) % size
            pending -= 1

        score += best_reward
        moves += 1
        board = bb.spawn_tile(best_afterstate)
        if harvest_reward > 0 and harvested == np.uint64(0) and best_reward >= harvest_reward and bb.max_exponent(board) >= harvest_exponent:
            harvested = board


@njit(nogil=True)
def play_games(
    n: int,
    weights: np.ndarray,
    traces: CoherenceTraces | None,
    layout: TupleLayout,
    learning_rate: float,
    learn: bool,
    late_starts: LateStarts | None = None,
    td_lambda: LambdaReturn | None = None,
) -> GameResults:
    """Play n games, starting a share of them from `late_starts` if given, and learning by TD(λ) if `td_lambda` is.

    Releases the GIL, so several threads can run it at once on the same weights.
    """
    scores = np.zeros(n, np.int64)
    max_exponents = np.zeros(n, np.int64)
    moves = np.zeros(n, np.int64)
    synthetic = np.zeros(n, np.bool_)
    harvested = np.zeros(n, np.uint64)
    for i in range(n):
        start = np.uint64(0)
        harvest_exponent = 0
        if late_starts is not None:
            harvest_exponent = late_starts.harvest_exponent
            if late_starts.pool.size > 0 and np.random.random() < late_starts.fraction:
                start = late_starts.pool[np.random.randint(late_starts.pool.size)]
                # Synthetic games don't donate positions, so starts aren't upgraded again and again
                harvest_exponent = 0
                synthetic[i] = True
        if td_lambda is None:
            game = play_game(weights, traces, layout, learning_rate, learn, start, harvest_exponent)
        else:
            game = play_game_lambda(weights, traces, layout, learning_rate, learn, start, harvest_exponent, td_lambda)
        scores[i], max_exponents[i], moves[i], harvested[i] = game
    return GameResults(scores, max_exponents, moves, synthetic, harvested)


def play_games_threaded(
    n: int,
    weights: np.ndarray,
    traces: CoherenceTraces | None,
    layout: TupleLayout,
    learning_rate: float,
    learn: bool,
    late_starts: LateStarts | None,
    threads: int,
    td_lambda: LambdaReturn | None = None,
) -> GameResults:
    """Play n games split across `threads` threads, which update the shared weights without locks."""
    # Game lengths vary a lot, so small shares handed out as threads free up keep every thread busy
    shares = [len(games) for games in np.array_split(np.arange(n), 8 * threads)]
    with ThreadPoolExecutor(max_workers=threads) as pool:
        parts = list(pool.map(lambda share: play_games(share, weights, traces, layout, learning_rate, learn, late_starts, td_lambda), shares))
    return GameResults(*(np.concatenate(field) for field in zip(*parts)))


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
    reset_coherence: bool = False,
) -> TrainingSummary:
    """Train until total_games have been played, including any in the resumed checkpoint.

    reset_coherence starts the temporal coherence sums from zero instead of the checkpoint's, which
    gives every weight back its full rate.

    The network is also saved as games_<n>.npz at each game count in snapshot_at, and each call is
    recorded as a session in the run's manifest. Ctrl-C saves and stops. Refuses to start if another
    process is already training into the same output folder.
    """
    settings = settings or get_ntuple_settings()
    with run_lock(Path(settings.OUTPUT_DIR)):
        return _train(resume_from, total_games, snapshot_at, settings, reset_coherence)


def _train(resume_from: str | Path | None, total_games: int | None, snapshot_at: list[int] | None, settings: NTupleSettings, reset_coherence: bool) -> TrainingSummary:
    total_games = settings.TRAINING_GAMES if total_games is None else total_games
    snapshots = sorted(set(settings.SNAPSHOT_AT if snapshot_at is None else snapshot_at))
    checkpoint_dir = Path(settings.CHECKPOINT_DIR)
    checkpoint_path = checkpoint_dir / CHECKPOINT_NAME
    output_dir = Path(settings.OUTPUT_DIR)
    history_path = output_dir / HISTORY_NAME

    network, progress = _load_or_create(resume_from, settings)
    traces = _load_or_create_traces(None if reset_coherence else resume_from, network, settings)
    if reset_coherence and traces is not None:
        print("Temporal coherence sums reset: every weight starts back at the full rate")
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
    if settings.LAMBDA > 0:
        schedule_end = f" to {settings.LAMBDA_END}" if settings.LAMBDA_END is not None else ""
        coherence += f", TD(λ={settings.LAMBDA}{schedule_end}) over {settings.LAMBDA_HORIZON} moves"
    threads = f", {settings.THREADS} threads after {settings.THREADS_AFTER:,} games" if settings.THREADS > 1 else ""
    print(f"Training {settings.NETWORK} n-tuple network ({network.weights.size:,} weights) to {total_games:,} games, lr {settings.LEARNING_RATE}{schedule}{coherence}{threads}")
    late_starts = _late_starts(settings)
    td_lambda = _td_lambda(settings)
    if not 1 <= settings.THREADS <= (os.cpu_count() or 1):
        raise ValueError(f"THREADS must be between 1 and {os.cpu_count()}, not {settings.THREADS}")
    # Compile before timing starts, otherwise the first batch's games/s includes compilation
    play_games(0, network.weights, traces, network.layout, settings.LEARNING_RATE, True, late_starts, td_lambda)

    interrupted = False
    try:
        while progress.games_played < total_games:
            stops = [total_games, *(s for s in snapshots if s > progress.games_played)]
            if settings.THREADS > 1 and settings.THREADS_AFTER > progress.games_played:
                stops.append(settings.THREADS_AFTER)
            batch = min(settings.LOG_EVERY, min(stops) - progress.games_played)
            progress.learning_rate = learning_rate_at(settings, start_games, total_games, progress.games_played)
            if td_lambda is not None:
                td_lambda = td_lambda._replace(lam=lambda_at(settings, start_games, total_games, progress.games_played))
            threaded = settings.THREADS > 1 and progress.games_played >= settings.THREADS_AFTER
            started = time.monotonic()
            if threaded:
                results = play_games_threaded(batch, network.weights, traces, network.layout, progress.learning_rate, True, late_starts, settings.THREADS, td_lambda)
            else:
                results = play_games(batch, network.weights, traces, network.layout, progress.learning_rate, True, late_starts, td_lambda)
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
    if settings.SCALE_WHEN_DOWNGRADED and not (settings.SCALE_FEATURE and settings.DOWNGRADE_TILE):
        raise ValueError("SCALE_WHEN_DOWNGRADED needs SCALE_FEATURE and DOWNGRADE_TILE")
    features = {"downgrade_tile": settings.DOWNGRADE_TILE, "scale_feature": settings.SCALE_FEATURE, "scale_when_downgraded": settings.SCALE_WHEN_DOWNGRADED}
    if resume_from is None:
        network = NTupleNetwork(
            NETWORKS[settings.NETWORK],
            downgrade_tile=settings.DOWNGRADE_TILE,
            scale_feature=settings.SCALE_FEATURE,
            scale_when_downgraded=settings.SCALE_WHEN_DOWNGRADED,
        )
        return network, CheckpointMetadata(games_played=0, moves_played=0, **learning)

    network, metadata = NTupleNetwork.load(resume_from)
    if network_name(network.tuples) != settings.NETWORK:
        raise ValueError(f"{resume_from} holds a {network_name(network.tuples) or 'custom'} network, not {settings.NETWORK}: set NTUPLE_NETWORK to match")
    if network.features != features:
        raise ValueError(f"{resume_from} was trained with {network.features}: set the settings to match")
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


def _td_lambda(settings: NTupleSettings) -> LambdaReturn | None:
    """Return the TD(λ) settings, or None for plain TD(0)."""
    lambdas = [settings.LAMBDA] if settings.LAMBDA_END is None else [settings.LAMBDA, settings.LAMBDA_END]
    if not all(0 <= lam <= 1 for lam in lambdas) or settings.LAMBDA_HORIZON < 1:
        raise ValueError(f"LAMBDA and LAMBDA_END must be between 0 and 1 and LAMBDA_HORIZON at least 1, not {lambdas} and {settings.LAMBDA_HORIZON}")
    return LambdaReturn(settings.LAMBDA, settings.LAMBDA_HORIZON) if settings.LAMBDA > 0 else None


def lambda_at(settings: NTupleSettings, start_games: int, total_games: int, games_played: int) -> float:
    """Return λ for the next batch, moving linearly from LAMBDA to LAMBDA_END over the session if it's set."""
    if settings.LAMBDA_END is None or total_games <= start_games:
        return settings.LAMBDA
    progress = (games_played - start_games) / (total_games - start_games)
    return settings.LAMBDA + (settings.LAMBDA_END - settings.LAMBDA) * progress


def _late_starts(settings: NTupleSettings) -> LateStarts | None:
    """Return an empty pool of synthetic starts if they're enabled. The pool isn't checkpointed: a resumed run refills it."""
    if settings.LATE_START_FRACTION <= 0:
        return None
    if settings.LATE_START_TILE & (settings.LATE_START_TILE - 1):
        raise ValueError(f"LATE_START_TILE must be a power of two, not {settings.LATE_START_TILE}")
    if settings.LATE_START_TILE.bit_length() - 1 >= bb.MAX_EXPONENT - 1:
        # upgrade() leaves such boards alone, since doubled they could need two 32768s merged, so no starts would ever be made
        raise ValueError(f"LATE_START_TILE must be below {1 << (bb.MAX_EXPONENT - 1)}, not {settings.LATE_START_TILE}: its positions can't be doubled")
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
