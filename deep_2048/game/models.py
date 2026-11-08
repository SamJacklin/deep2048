from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from deep_2048.game.constants import Action


class BoardEncoding(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    vector: np.ndarray

    @property
    def as_list(self) -> list[float]:
        return self.vector.tolist()


class MoveResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    board: np.ndarray
    reward: int
    changed: bool


class GameInfo(BaseModel):
    score: int
    max_tile: int
    board_changed: bool
    action: str
    valid_actions: list[Action] = Field(default_factory=list)


class StepResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    state: np.ndarray
    reward: float
    done: bool
    info: GameInfo


class GameSnapshot(BaseModel):
    board: list[list[int]]
    score: int
    done: bool
    valid_actions: list[Action]


class SpawnResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    board: np.ndarray
    spawned_value: int | None = None
    spawned_position: tuple[int, int] | None = None


class RenderedBoard(BaseModel):
    lines: list[str]

    def __str__(self) -> str:
        return "\n".join(self.lines)

    def print(self) -> None:
        print(str(self))


class MoveAnalysis(BaseModel):
    action: Action
    changed: bool
    reward: int


class BoardStats(BaseModel):
    empty_tiles: int
    max_tile: int
    has_empty_tiles: bool


class GameState(BaseModel):
    board: list[list[int]]
    score: int = 0
    done: bool = False
    move_count: int = 0

    @classmethod
    def from_numpy(
        cls,
        board: np.ndarray,
        score: int = 0,
        done: bool = False,
        move_count: int = 0,
    ) -> GameState:
        return cls(
            board=board.astype(int).tolist(),
            score=score,
            done=done,
            move_count=move_count,
        )

    def to_numpy(self) -> np.ndarray:
        return np.array(self.board, dtype=np.int32)

    def to_snapshot(self, valid_actions: list[Action]) -> GameSnapshot:
        return GameSnapshot(
            board=self.board,
            score=self.score,
            done=self.done,
            valid_actions=valid_actions,
        )
