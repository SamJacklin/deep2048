"""Play seeded games with any Agent and collect results.

Game i is played with seed `seed_offset + i`, so every agent sees exactly the same
sequence of tile spawns for the same moves and results are directly comparable.
Games are independent, so they can be spread across worker processes.
"""

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from deep_2048.agents.base import Agent
from deep_2048.game.game import Game2048


@dataclass
class EvalResult:
    label: str
    n_games: int
    scores: list[int]
    max_tiles: list[int]

    @property
    def avg_score(self) -> float:
        return float(np.mean(self.scores))

    @property
    def avg_max_tile(self) -> float:
        return float(np.mean(self.max_tiles))

    def tile_pct(self, tile: int) -> float:
        return 100.0 * sum(t >= tile for t in self.max_tiles) / self.n_games


class GameOutcome(NamedTuple):
    score: int
    max_tile: int


def play_game(agent: Agent, seed: int) -> GameOutcome:
    """Play one game to the end with `agent`."""
    game = Game2048(seed=seed)
    game.reset()
    while not game.is_game_over():
        decision = agent.act(game.get_board(), game.get_valid_actions())
        game.step(decision.action)
    return GameOutcome(score=int(game.score), max_tile=int(game.max_tile))


def run_games(agent: Agent, n: int, seed_offset: int = 0, workers: int = 1) -> EvalResult:
    """Play n games with `agent`, printing a live progress line.

    With more than one worker, each worker process gets its own pickled copy of the agent.
    """
    seeds = range(seed_offset, seed_offset + n)
    outcomes: dict[int, GameOutcome] = {}
    progress = _Progress(n)

    if workers == 1:
        for seed in seeds:
            outcomes[seed] = play_game(agent, seed)
            progress.update(outcomes[seed])
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(agent,)) as pool:
            futures = {pool.submit(_play_in_worker, seed): seed for seed in seeds}
            for future in as_completed(futures):
                outcomes[futures[future]] = future.result()
                progress.update(outcomes[futures[future]])

    progress.finish()
    return EvalResult(
        label=agent.label,
        n_games=n,
        scores=[outcomes[seed].score for seed in seeds],
        max_tiles=[outcomes[seed].max_tile for seed in seeds],
    )


_worker_agent: Agent | None = None


def _init_worker(agent: Agent) -> None:
    global _worker_agent
    _worker_agent = agent


def _play_in_worker(seed: int) -> GameOutcome:
    assert _worker_agent is not None
    return play_game(_worker_agent, seed)


class _Progress:
    """Live one-line summary of the games finished so far."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.scores: list[int] = []
        self.best_tile = 0
        self.start = time.monotonic()

    def update(self, outcome: GameOutcome) -> None:
        self.scores.append(outcome.score)
        self.best_tile = max(self.best_tile, outcome.max_tile)
        done = len(self.scores)
        elapsed = time.monotonic() - self.start
        eta = elapsed / done * (self.n - done)
        print(
            f"  {done:>{len(str(self.n))}}/{self.n}  "
            f"score {outcome.score:>6}  tile {outcome.max_tile:>4}  "
            f"| running avg {np.mean(self.scores):>7.0f}  best tile {self.best_tile:>4}  "
            f"| {elapsed / done:>5.1f}s/game  ETA {eta:>5.0f}s",
            end="\r",
            flush=True,
        )

    def finish(self) -> None:
        print(f"  {self.n}/{self.n} done in {time.monotonic() - self.start:.1f}s{' ' * 40}")
