"""Evaluate an agent over N games and print a summary table.

Usage examples:
    uv run python scripts/evaluate.py                          # DQN best, 100 games
    uv run python scripts/evaluate.py --mode drl_search -n 50
    uv run python scripts/evaluate.py --mode heuristic -n 50
    uv run python scripts/evaluate.py --mode all -n 100        # DQN vs heuristic side-by-side
    uv run python scripts/evaluate.py checkpoints/step_350068.pt --mode drl -n 200
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from deep_2048.game_logic.game import Game2048
from deep_2048.model.agent import DQNAgent, get_device
from deep_2048.model.heuristic import heuristic_select
from deep_2048.model.search import dqn_search_select


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


def _run_games(move_fn, n: int, seed_offset: int = 0) -> tuple[list[int], list[int]]:
    """Run n games, returning (scores, max_tiles)."""
    scores, max_tiles = [], []
    t_start = time.monotonic()

    for i in range(n):
        t_game = time.monotonic()
        game = Game2048(seed=i + seed_offset)
        game.reset()
        while not game.is_game_over():
            valid_actions = game.get_valid_actions()
            action = move_fn(game, valid_actions)
            game.step(action)
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
    return scores, max_tiles


def evaluate_mode(
    mode: str,
    depth: int,
    checkpoint: str,
    n: int,
    seed_offset: int = 0,
) -> EvalResult:
    agent: DQNAgent | None = None
    if mode in ("drl", "drl_search"):
        device = get_device()
        agent = DQNAgent(device=device)
        agent.load(Path(checkpoint))
        agent.set_eval_mode()

    if mode == "drl":
        def move_fn(game: Game2048, valid_actions):
            state = game.get_state()
            return agent.select_action(state=state, valid_actions=valid_actions, epsilon=0.0).action
        label = f"DQN ({Path(checkpoint).stem})"

    elif mode == "drl_search":
        _depth = depth
        def move_fn(game: Game2048, valid_actions):
            action, _ = dqn_search_select(agent, game.board, valid_actions, depth=_depth)
            return action
        label = f"DQN+Search depth={depth} ({Path(checkpoint).stem})"

    else:  # heuristic
        _depth = depth
        def move_fn(game: Game2048, valid_actions):
            action, _ = heuristic_select(game.board, valid_actions, depth=_depth)
            return action
        label = f"Heuristic depth={depth}"

    scores, max_tiles = _run_games(move_fn, n=n, seed_offset=seed_offset)
    return EvalResult(label=label, n_games=n, scores=scores, max_tiles=max_tiles)


def print_results(results: list[EvalResult]) -> None:
    col = 42
    tiles = [256, 512, 1024, 2048]

    header = f"{'Agent':<{col}}  {'Avg score':>10}  {'Avg tile':>9}" + "".join(
        f"  {f'≥{t}':>7}" for t in tiles
    )
    sep = "─" * len(header)

    print(f"\n{sep}")
    print(header)
    print(sep)
    for r in results:
        tile_cols = "".join(f"  {r.tile_pct(t):>6.1f}%" for t in tiles)
        print(
            f"{r.label:<{col}}  {r.avg_score:>10.0f}  {r.avg_max_tile:>9.0f}{tile_cols}"
            f"  ({r.n_games} games)"
        )
    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a 2048 agent over multiple games")
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default="checkpoints/best.pt",
        help="Path to .pt checkpoint (default: checkpoints/best.pt)",
    )
    parser.add_argument(
        "--mode",
        choices=["drl", "drl_search", "heuristic", "all"],
        default="drl",
        help="Agent mode. 'all' runs DQN and heuristic side-by-side (default: drl)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Search depth — default 2 for drl_search and heuristic",
    )
    parser.add_argument(
        "-n",
        "--num-games",
        type=int,
        default=100,
        help="Number of games to evaluate (default: 100)",
    )
    parser.add_argument(
        "--json",
        metavar="PATH",
        default=None,
        help="Save results as JSON to this path (e.g. frontend/public/benchmark.json)",
    )
    args = parser.parse_args()

    if args.mode == "all":
        modes = [
            ("drl", 1),
            ("heuristic", 2 if args.depth is None else args.depth),
        ]
        results = []
        for mode, depth in modes:
            print(f"\nEvaluating {mode} (depth={depth if mode != 'drl' else 'n/a'})…")
            results.append(evaluate_mode(mode, depth, args.checkpoint, args.num_games))
        print_results(results)
    else:
        depth = args.depth if args.depth is not None else 2
        print(f"\nEvaluating {args.mode}…")
        result = evaluate_mode(args.mode, depth, args.checkpoint, args.num_games)
        print_results([result])
        results = [result]

    if args.json:
        tiles = [256, 512, 1024, 2048]
        payload = [
            {
                "label": r.label,
                "n_games": r.n_games,
                "avg_score": round(r.avg_score),
                "avg_max_tile": round(r.avg_max_tile),
                "tile_pcts": {str(t): round(r.tile_pct(t), 1) for t in tiles},
            }
            for r in results
        ]
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved benchmark JSON → {out}")


if __name__ == "__main__":
    main()
