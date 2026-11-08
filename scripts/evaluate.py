"""Evaluate one or more agents over N seeded games and print a summary table.

Usage examples:
    uv run python scripts/evaluate.py                              # DQN best, 100 games
    uv run python scripts/evaluate.py --agent dqn_search -n 50
    uv run python scripts/evaluate.py --agent heuristic -n 50
    uv run python scripts/evaluate.py --agent ntuple -n 100         # checkpoints/ntuple/latest.npz
    uv run python scripts/evaluate.py --agent all -n 100           # DQN vs heuristic side-by-side
    uv run python scripts/evaluate.py checkpoints/dqn/step_350068.pt --agent dqn -n 200
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from deep_2048.agents import AGENT_NAMES, DEFAULT_DQN_CHECKPOINT, DEFAULT_NTUPLE_CHECKPOINT, load_agent
from deep_2048.bench import EvalResult, run_games

# Agents run by `--agent all`, in table order.
ALL_AGENTS = ("dqn", "heuristic")


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
        default=None,
        help=f"Checkpoint for learned agents (default: {DEFAULT_DQN_CHECKPOINT} for DQN agents, {DEFAULT_NTUPLE_CHECKPOINT} for ntuple)",
    )
    parser.add_argument(
        "--agent",
        choices=[*AGENT_NAMES, "all"],
        default="dqn",
        help=f"Agent to evaluate. 'all' runs {' and '.join(ALL_AGENTS)} side-by-side (default: dqn)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Search depth for search-based agents (default: 2)",
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

    names = ALL_AGENTS if args.agent == "all" else (args.agent,)
    results: list[EvalResult] = []
    for name in names:
        agent = load_agent(name, checkpoint=args.checkpoint, depth=args.depth)
        print(f"\nEvaluating {agent.label}…")
        results.append(run_games(agent, n=args.num_games))
    print_results(results)

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
