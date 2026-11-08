"""Play seeded games with any Agent and collect results.

Game i is played with seed `seed_offset + i`, so every agent sees exactly the same
sequence of tile spawns for the same moves — results are directly comparable.
"""

import time
from dataclasses import dataclass

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


def run_games(agent: Agent, n: int, seed_offset: int = 0) -> EvalResult:
    """Play n games with `agent`, printing a live progress line."""
    scores, max_tiles = [], []
    t_start = time.monotonic()

    for i in range(n):
        t_game = time.monotonic()
        game = Game2048(seed=i + seed_offset)
        game.reset()
        while not game.is_game_over():
            decision = agent.act(game.get_board(), game.get_valid_actions())
            game.step(decision.action)
        scores.append(game.score)
        max_tiles.append(game.max_tile)

        elapsed = time.monotonic() - t_start
        game_time = time.monotonic() - t_game
        done = i + 1
        avg_s = sum(scores) / done
        best = max(max_tiles)
        eta = (elapsed / done) * (n - done)
        print(
            f"  {done:>{len(str(n))}}/{n}  "
            f"score {game.score:>6}  tile {game.max_tile:>4}  "
            f"| running avg {avg_s:>7.0f}  best tile {best:>4}  "
            f"| {game_time:>5.1f}s/game  ETA {eta:>5.0f}s",
            end="\r",
            flush=True,
        )

    elapsed = time.monotonic() - t_start
    print(f"  {n}/{n} done in {elapsed:.1f}s{' ' * 40}")
    return EvalResult(label=agent.label, n_games=n, scores=scores, max_tiles=max_tiles)
