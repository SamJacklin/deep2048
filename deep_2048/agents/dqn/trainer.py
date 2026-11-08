import csv
import math
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from deep_2048.agents.dqn.agent import DQNAgent, get_device
from deep_2048.agents.dqn.config import get_dqn_settings
from deep_2048.agents.dqn.models import (
    CheckpointMetadata,
    CheckpointResult,
    EpisodeSummary,
    EvaluationResult,
    TrainingHistoryPoint,
    TrainingSummary,
    Transition,
)
from deep_2048.agents.dqn.replay_buffer import NStepBuffer, PrioritizedReplayBuffer
from deep_2048.game import Game2048

settings = get_dqn_settings()


def train(resume_from: str | None = None) -> TrainingSummary:
    _set_global_seeds()
    _ensure_output_directories()

    device = get_device()
    agent = DQNAgent(device=device)
    replay_buffer = PrioritizedReplayBuffer(
        capacity=settings.REPLAY_CAPACITY,
        device=device,
    )

    game = Game2048()

    training_step = 0
    episode_index = 0
    epsilon = settings.EPSILON_START

    if resume_from:
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        training_state = checkpoint.get("training_state", {})
        training_step = training_state.get("training_step", 0)
        episode_index = training_state.get("episode_index", 0)
        epsilon = _calculate_epsilon(
            current_step=training_step,
            epsilon_start=settings.EPSILON_START,
            epsilon_end=settings.EPSILON_END,
            epsilon_decay=settings.EPSILON_DECAY,
        )
        agent.load(resume_from)
        print(f"Resumed from {resume_from} at step={training_step} episode={episode_index} epsilon={epsilon:.4f}")

    _print_training_banner(agent=agent, device=device, start_step=training_step)

    history: list[TrainingHistoryPoint] = []
    checkpoint_paths: list[str] = []

    latest_eval_result: EvaluationResult | None = None
    latest_loss: float | None = None
    latest_lr: float | None = None
    last_eval_step: int = (training_step // settings.EVAL_EVERY) * settings.EVAL_EVERY
    best_eval_score: float = -1.0

    train_start_time = time.monotonic()
    score_window: deque[int] = deque(maxlen=50)

    while training_step < settings.TRAINING_STEPS:
        episode_index += 1
        episode_summary, training_step, epsilon, latest_loss, latest_lr = _run_training_episode(
            game=game,
            agent=agent,
            replay_buffer=replay_buffer,
            training_step=training_step,
            episode_index=episode_index,
            epsilon=epsilon,
        )

        score_window.append(episode_summary.score)

        eval_average_score: float | None = None
        eval_average_max_tile: float | None = None

        next_eval_step = (training_step // settings.EVAL_EVERY) * settings.EVAL_EVERY
        if next_eval_step > last_eval_step:
            last_eval_step = next_eval_step
            latest_eval_result = evaluate_agent(
                agent=agent,
                episodes=settings.EVAL_EPISODES,
            )
            eval_average_score = latest_eval_result.average_score
            eval_average_max_tile = latest_eval_result.average_max_tile

            checkpoint_result = _save_checkpoint(
                agent=agent,
                training_step=training_step,
                episode_index=episode_index,
                epsilon=epsilon,
                average_score=latest_eval_result.average_score,
                average_max_tile=latest_eval_result.average_max_tile,
            )
            checkpoint_paths.append(str(checkpoint_result.path))

            if latest_eval_result.average_score > best_eval_score:
                best_eval_score = latest_eval_result.average_score
                _save_checkpoint(
                    agent=agent,
                    training_step=training_step,
                    episode_index=episode_index,
                    epsilon=epsilon,
                    average_score=latest_eval_result.average_score,
                    average_max_tile=latest_eval_result.average_max_tile,
                    checkpoint_name="best.pt",
                )
                print(f"[best]  step={training_step} new best avg_score={latest_eval_result.average_score:.2f}")

            dist = latest_eval_result.tile_distribution
            dist_str = "  ".join(f"≥{t}: {dist.get(t, 0.0)*100:.0f}%" for t in (256, 512, 1024, 2048))
            print(
                f"[eval]  step={training_step}"
                f"  avg_score={latest_eval_result.average_score:.1f}"
                f"  avg_tile={latest_eval_result.average_max_tile:.0f}"
                f"  |  {dist_str}"
            )

        history_point = TrainingHistoryPoint(
            training_step=training_step,
            episode_index=episode_index,
            epsilon=epsilon,
            episode_score=episode_summary.score,
            episode_max_tile=episode_summary.max_tile,
            episode_reward=episode_summary.total_reward,
            episode_steps=episode_summary.steps,
            eval_average_score=eval_average_score,
            eval_average_max_tile=eval_average_max_tile,
            loss=latest_loss,
            lr=latest_lr,
        )
        history.append(history_point)

        elapsed = time.monotonic() - train_start_time
        steps_per_sec = training_step / elapsed if elapsed > 0 else 0.0
        remaining = settings.TRAINING_STEPS - training_step
        eta_str = _format_duration(int(remaining / steps_per_sec)) if steps_per_sec > 0 else "?"
        progress_pct = 100.0 * training_step / settings.TRAINING_STEPS
        rolling_avg = sum(score_window) / len(score_window)

        loss_str = f"  loss={latest_loss:.4f}" if latest_loss is not None else ""
        lr_str = f"  lr={latest_lr:.2e}" if latest_lr is not None else ""
        print(
            f"[train] ep={episode_index}  step={training_step} ({progress_pct:.1f}%)"
            f"  score={episode_summary.score}  tile={episode_summary.max_tile}"
            f"  avg50={rolling_avg:.0f}  ε={epsilon:.3f}"
            f"{loss_str}{lr_str}"
            f"  {steps_per_sec:.0f} steps/s  eta={eta_str}"
        )

    final_checkpoint = _save_checkpoint(
        agent=agent,
        training_step=training_step,
        episode_index=episode_index,
        epsilon=epsilon,
        average_score=latest_eval_result.average_score if latest_eval_result else None,
        average_max_tile=latest_eval_result.average_max_tile if latest_eval_result else None,
        checkpoint_name="final.pt",
    )
    checkpoint_paths.append(str(final_checkpoint.path))

    _write_training_history_csv(history=history)

    return TrainingSummary(
        total_training_steps=training_step,
        total_episodes=episode_index,
        final_epsilon=epsilon,
        latest_eval_average_score=(latest_eval_result.average_score if latest_eval_result else None),
        latest_eval_average_max_tile=(latest_eval_result.average_max_tile if latest_eval_result else None),
        checkpoint_paths=checkpoint_paths,
    )


def evaluate_agent(
    agent: DQNAgent,
    episodes: int,
) -> EvaluationResult:
    agent.set_eval_mode()

    evaluation_game = Game2048(seed=settings.SEED + 1)

    scores: list[int] = []
    max_tiles: list[int] = []
    rewards: list[float] = []

    for _ in range(episodes):
        state = evaluation_game.reset()
        done = False
        total_reward = 0.0

        while not done:
            valid_actions = evaluation_game.get_valid_actions()
            selection_result = agent.select_action(
                state=state,
                valid_actions=valid_actions,
                epsilon=0.0,
            )
            step_result = evaluation_game.step(selection_result.action)
            state = step_result.state
            total_reward += step_result.reward
            done = step_result.done

        scores.append(evaluation_game.score)
        max_tiles.append(evaluation_game.max_tile)
        rewards.append(total_reward)

    agent.set_train_mode()

    _TILE_THRESHOLDS = (256, 512, 1024, 2048)
    n = len(max_tiles)
    tile_distribution = {
        t: sum(1 for tile in max_tiles if tile >= t) / n
        for t in _TILE_THRESHOLDS
    } if n > 0 else {}

    return EvaluationResult(
        average_score=float(np.mean(scores)) if scores else 0.0,
        average_max_tile=float(np.mean(max_tiles)) if max_tiles else 0.0,
        average_reward=float(np.mean(rewards)) if rewards else 0.0,
        episodes=episodes,
        tile_distribution=tile_distribution,
    )


def _run_training_episode(
    game: Game2048,
    agent: DQNAgent,
    replay_buffer: PrioritizedReplayBuffer,
    training_step: int,
    episode_index: int,
    epsilon: float,
) -> tuple[EpisodeSummary, int, float, float | None, float | None]:
    state = game.reset()
    done = False

    total_reward = 0.0
    episode_steps = 0
    latest_loss: float | None = None
    latest_lr: float | None = None
    episode_max_tile = game.max_tile

    n_step_buffer = NStepBuffer(n_steps=settings.N_STEPS, gamma=settings.GAMMA)

    while not done and training_step < settings.TRAINING_STEPS:
        valid_actions = game.get_valid_actions()

        selection_result = agent.select_action(
            state=state,
            valid_actions=valid_actions,
            epsilon=epsilon,
        )

        step_result = game.step(selection_result.action)

        shaped_reward = math.log1p(step_result.reward) if step_result.reward > 0 else 0.0

        new_max_tile = step_result.info.max_tile
        if new_max_tile > episode_max_tile:
            shaped_reward += math.log1p(new_max_tile) * settings.MILESTONE_BONUS_SCALE
            episode_max_tile = new_max_tile

        raw_transition = Transition(
            state=state.copy(),
            action=selection_result.action,
            reward=shaped_reward,
            next_state=step_result.state.copy(),
            done=step_result.done,
        )
        for n_step_transition in n_step_buffer.push(raw_transition):
            replay_buffer.push(n_step_transition)

        state = step_result.state
        total_reward += step_result.reward
        done = step_result.done
        episode_steps += 1
        training_step += 1

        epsilon = _calculate_epsilon(
            current_step=training_step,
            epsilon_start=settings.EPSILON_START,
            epsilon_end=settings.EPSILON_END,
            epsilon_decay=settings.EPSILON_DECAY,
        )

        if len(replay_buffer) >= settings.MIN_REPLAY_SIZE:
            beta = min(
                settings.PER_BETA_END,
                settings.PER_BETA_START + (settings.PER_BETA_END - settings.PER_BETA_START) * training_step / settings.TRAINING_STEPS,
            )
            batch = replay_buffer.sample(settings.BATCH_SIZE, beta=beta)
            optimisation_result = agent.optimise(batch)
            replay_buffer.update_priorities(batch.indices, optimisation_result.td_errors)
            latest_loss = optimisation_result.loss
            latest_lr = optimisation_result.lr
            if training_step % settings.TARGET_UPDATE_FREQUENCY == 0:
                agent.update_target_network()

    for n_step_transition in n_step_buffer.flush():
        replay_buffer.push(n_step_transition)

    episode_summary = EpisodeSummary(
        episode_index=episode_index,
        score=game.score,
        max_tile=game.max_tile,
        total_reward=total_reward,
        steps=episode_steps,
        epsilon=epsilon,
    )

    return episode_summary, training_step, epsilon, latest_loss, latest_lr


def _calculate_epsilon(
    current_step: int,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay: int,
) -> float:
    if epsilon_decay <= 0:
        return epsilon_end

    progress = min(current_step / epsilon_decay, 1.0)
    return epsilon_start + progress * (epsilon_end - epsilon_start)


def _save_checkpoint(
    agent: DQNAgent,
    training_step: int,
    episode_index: int,
    epsilon: float,
    average_score: float | None,
    average_max_tile: float | None,
    checkpoint_name: str | None = None,
) -> CheckpointResult:
    checkpoint_dir = Path(settings.CHECKPOINT_DIR)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    path = checkpoint_dir / (checkpoint_name if checkpoint_name else f"step_{training_step}.pt")

    agent.save(path)

    # Patch training state into the saved file so resume works from any checkpoint.
    saved = torch.load(path, map_location="cpu", weights_only=False)
    saved["training_state"] = {"training_step": training_step, "episode_index": episode_index}
    torch.save(saved, path)

    metadata = CheckpointMetadata(
        training_step=training_step,
        episode_index=episode_index,
        epsilon=epsilon,
        average_score=average_score,
        average_max_tile=average_max_tile,
    )

    metadata_path = path.with_suffix(".json")
    metadata_path.write_text(metadata.model_dump_json(indent=2))

    return CheckpointResult(path=path, metadata=metadata)


def _write_training_history_csv(
    history: list[TrainingHistoryPoint],
) -> None:
    output_dir = Path(settings.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "training_history.csv"

    fieldnames = [
        "training_step",
        "episode_index",
        "epsilon",
        "episode_score",
        "episode_max_tile",
        "episode_reward",
        "episode_steps",
        "eval_average_score",
        "eval_average_max_tile",
        "loss",
        "lr",
    ]

    # Append if the file already exists (resume), otherwise write fresh with header.
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for point in history:
            writer.writerow(point.model_dump())


def _ensure_output_directories() -> None:
    Path(settings.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def _set_global_seeds() -> None:
    random.seed(settings.SEED)
    np.random.seed(settings.SEED)
    torch.manual_seed(settings.SEED)


def _format_duration(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    h = seconds // 3600
    m = (seconds % 3600) // 60
    return f"{h}h{m:02d}m"


def _print_training_banner(agent: DQNAgent, device: torch.device, start_step: int) -> None:
    param_count = sum(p.numel() for p in agent.online_network.parameters())

    def _fmt(n: int) -> str:
        return f"{n:,}"

    decay_pct = 100.0 * settings.EPSILON_DECAY / settings.TRAINING_STEPS
    remaining = settings.TRAINING_STEPS - start_step

    print("=" * 44)
    print("  Deep 2048 — DQN Training")
    print("=" * 44)
    print(f"  Device:          {device}")
    print(f"  Steps:           {_fmt(remaining)} (total {_fmt(settings.TRAINING_STEPS)})")
    print(f"  Epsilon decay:   {_fmt(settings.EPSILON_DECAY)} ({decay_pct:.0f}% of training)")
    print(f"  Batch size:      {settings.BATCH_SIZE}")
    print(f"  Replay buffer:   {_fmt(settings.REPLAY_CAPACITY)}")
    print(f"  N-step:          {settings.N_STEPS}")
    print(f"  Target update:   every {_fmt(settings.TARGET_UPDATE_FREQUENCY)} steps (τ={settings.TAU})")
    print(f"  LR eta_min:      {settings.LR_ETA_MIN}")
    print(f"  Hidden dim:      {settings.HIDDEN_DIM}")
    print(f"  Network params:  {param_count:,}")
    print("=" * 44)
