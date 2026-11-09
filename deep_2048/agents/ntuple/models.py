from pathlib import Path
from typing import NamedTuple

import numpy as np
from pydantic import BaseModel


class CheckpointMetadata(BaseModel):
    games_played: int
    moves_played: int
    learning_rate: float


class GameResults(NamedTuple):
    """Per-game results from a batch of training games. A NamedTuple so numba can build it."""

    scores: np.ndarray
    max_exponents: np.ndarray
    moves: np.ndarray


class TrainingHistoryRow(BaseModel):
    games: int
    moves: int
    avg_score: float
    max_score: int
    avg_moves: float
    rate_2048: float
    rate_4096: float
    rate_8192: float
    rate_16384: float
    games_per_second: float

    @classmethod
    def from_results(cls, games_played: int, moves_played: int, results: GameResults, seconds: float) -> "TrainingHistoryRow":
        tiles = np.left_shift(1, results.max_exponents)
        return cls(
            games=games_played,
            moves=moves_played,
            avg_score=float(results.scores.mean()),
            max_score=int(results.scores.max()),
            avg_moves=float(results.moves.mean()),
            rate_2048=100 * float(np.mean(tiles >= 2048)),
            rate_4096=100 * float(np.mean(tiles >= 4096)),
            rate_8192=100 * float(np.mean(tiles >= 8192)),
            rate_16384=100 * float(np.mean(tiles >= 16384)),
            games_per_second=len(results.scores) / max(seconds, 1e-9),
        )

    def __str__(self) -> str:
        return (
            f"  games {self.games:>9,}  moves {self.moves / 1e6:>8.1f}M  avg score {self.avg_score:>8.0f}  max {self.max_score:>7}  "
            f"≥2048 {self.rate_2048:5.1f}%  ≥4096 {self.rate_4096:5.1f}%  ≥8192 {self.rate_8192:5.1f}%  "
            f"| {self.games_per_second:6.1f} games/s"
        )


class TrainingSummary(BaseModel):
    games_played: int
    checkpoint_path: Path
    interrupted: bool
