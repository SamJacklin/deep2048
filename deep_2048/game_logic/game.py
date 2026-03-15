import numpy as np

from deep_2048.config import get_settings
from deep_2048.game_logic.board import (
    copy_board,
    get_max_tile,
    initialise_board,
    render_board,
    spawn_random_tile,
)
from deep_2048.game_logic.constants import SPAWN_TILE_WEIGHTS, Action
from deep_2048.game_logic.encoding import encode_board
from deep_2048.game_logic.models import GameInfo, GameSnapshot, GameState, StepResult
from deep_2048.game_logic.moves import apply_move, get_valid_actions, is_game_over

settings = get_settings()


class Game2048:
    def __init__(self, seed: int | None = None) -> None:
        self.board_size: int = settings.BOARD_SIZE
        self.rng: np.random.Generator = np.random.default_rng(seed if seed is not None else settings.SEED)

        self.board: np.ndarray = np.zeros((self.board_size, self.board_size), dtype=np.int32)
        self.score: int = 0
        self.done: bool = False
        self.move_count: int = 0

        self.reset()

    def reset(self) -> np.ndarray:
        self.board = initialise_board(
            board_size=self.board_size,
            rng=self.rng,
            tile_weights=SPAWN_TILE_WEIGHTS,
        )
        self.score = 0
        self.done = False
        self.move_count = 0
        return self.get_state()

    def step(self, action: Action) -> StepResult:
        if self.done:
            return StepResult(
                state=self.get_state(),
                reward=0.0,
                done=True,
                info=self._build_info(action=action, board_changed=False),
            )

        move_result = apply_move(self.board, action)

        if not move_result.changed:
            self.done = is_game_over(self.board)
            return StepResult(
                state=self.get_state(),
                reward=0.0,
                done=self.done,
                info=self._build_info(action=action, board_changed=False),
            )

        self.board = move_result.board
        self.score += move_result.reward
        self.move_count += 1

        spawn_result = spawn_random_tile(
            board=self.board,
            rng=self.rng,
            tile_weights=SPAWN_TILE_WEIGHTS,
        )
        self.board = spawn_result.board
        self.done = is_game_over(self.board)

        return StepResult(
            state=self.get_state(),
            reward=float(move_result.reward),
            done=self.done,
            info=self._build_info(action=action, board_changed=True),
        )

    def get_state(self) -> np.ndarray:
        return encode_board(self.board).vector

    def get_board(self) -> np.ndarray:
        return copy_board(self.board)

    def get_valid_actions(self) -> list[Action]:
        return get_valid_actions(self.board)

    def is_game_over(self) -> bool:
        return self.done

    def render(self) -> None:
        rendered = render_board(self.board, score=self.score)
        rendered.print()

    def get_snapshot(self) -> GameSnapshot:
        return GameState.from_numpy(
            board=self.board,
            score=self.score,
            done=self.done,
            move_count=self.move_count,
        ).to_snapshot(valid_actions=self.get_valid_actions())

    @property
    def max_tile(self) -> int:
        return get_max_tile(self.board)

    def _build_info(self, action: Action, board_changed: bool) -> GameInfo:
        return GameInfo(
            score=self.score,
            max_tile=self.max_tile,
            board_changed=board_changed,
            action=action.name,
            valid_actions=self.get_valid_actions(),
        )
