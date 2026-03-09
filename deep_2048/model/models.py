from pathlib import Path

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field

from deep_2048.game_logic.constants import Action


class Transition(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    state: np.ndarray
    action: Action
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBatch(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor


class ActionSelectionResult(BaseModel):
    action: Action
    epsilon: float
    is_random: bool
    q_values: list[float] = Field(default_factory=list)
    valid_actions: list[Action] = Field(default_factory=list)


class OptimisationResult(BaseModel):
    loss: float
    mean_q_value: float
    mean_target_q_value: float


class EvaluationResult(BaseModel):
    average_score: float
    average_max_tile: float
    average_reward: float
    episodes: int


class EpisodeSummary(BaseModel):
    episode_index: int
    score: int
    max_tile: int
    total_reward: float
    steps: int
    epsilon: float


class CheckpointMetadata(BaseModel):
    training_step: int
    episode_index: int
    epsilon: float
    average_score: float | None = None
    average_max_tile: float | None = None


class CheckpointResult(BaseModel):
    path: Path
    metadata: CheckpointMetadata


class TrainingHistoryPoint(BaseModel):
    training_step: int
    episode_index: int
    epsilon: float
    episode_score: int
    episode_max_tile: int
    episode_reward: float
    episode_steps: int
    eval_average_score: float | None = None
    eval_average_max_tile: float | None = None
    loss: float | None = None


class TrainingSummary(BaseModel):
    total_training_steps: int
    total_episodes: int
    final_epsilon: float
    latest_eval_average_score: float | None = None
    latest_eval_average_max_tile: float | None = None
    checkpoint_paths: list[str] = Field(default_factory=list)
