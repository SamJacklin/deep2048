"""Evaluate one or more agents over N seeded games and print a summary table.

Usage examples:
    uv run python scripts/evaluate.py                              # DQN best, 1000 games
    uv run python scripts/evaluate.py --agent dqn_search -n 50
    uv run python scripts/evaluate.py --agent heuristic -n 50
    uv run python scripts/evaluate.py --agent ntuple               # checkpoints/ntuple/best.npz
    uv run python scripts/evaluate.py --agent ntuple_search --depth 3
    uv run python scripts/evaluate.py --agent ntuple_search --depth 5 --cutoff 0.1   # prune unlikely spawns
    uv run python scripts/evaluate.py --agent all                  # every benchmark agent side-by-side
    uv run python scripts/evaluate.py checkpoints/dqn/step_350068.pt --agent dqn -n 200
    uv run python scripts/evaluate.py --agent all --rerun           # ignore cached results
    uv run python scripts/evaluate.py --agent dqn --workers 1       # play games one at a time
    uv run python scripts/evaluate.py checkpoints/ntuple/games_1000.npz --agent ntuple --experiment experiments/ntuple-td-1m

Results are cached in output/benchmark_cache/, keyed by agent label, checkpoint hash and game count.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from deep_2048.agents import AGENT_NAMES, DEFAULT_DQN_CHECKPOINT, DEFAULT_NTUPLE_CHECKPOINT, load_agent
from deep_2048.bench import EvalResult, ResultCache, RunKey, run_games
from deep_2048.experiments import Environment, EvaluationRecord, now, record_evaluation

# Agents run by `--agent all`, in table order.
# (agent, search depth) pairs run by `--agent all`, in table order. None uses the agent's default.
ALL_AGENTS: tuple[tuple[str, int | None], ...] = (("dqn", None), ("heuristic", 2), ("ntuple", None), ("ntuple_search", 3))
TILES = (256, 512, 1024, 2048, 4096, 8192, 16384)


def print_results(results: list[EvalResult]) -> None:
    col = max(42, *(len(r.label) for r in results))

    header = f"{'Agent':<{col}}  {'Avg score':>10}  {'Avg tile':>9}" + "".join(
        f"  {f'≥{t}':>7}" for t in TILES
    )
    sep = "─" * len(header)

    print(f"\n{sep}")
    print(header)
    print(sep)
    for r in results:
        tile_cols = "".join(f"  {r.tile_pct(t):>6.1f}%" for t in TILES)
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
        help=f"Agent to evaluate. 'all' runs {', '.join(name for name, _ in ALL_AGENTS)} side-by-side (default: dqn)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Search depth for search-based agents (default: 2)",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=0.0,
        help="For ntuple_search, don't expand spawn branches less likely than this from the root (default: 0, exact)",
    )
    parser.add_argument(
        "-n",
        "--num-games",
        type=int,
        default=1000,
        help="Number of games to evaluate (default: 1000)",
    )
    parser.add_argument(
        "--json",
        metavar="PATH",
        default=None,
        help="Save results as JSON to this path (e.g. frontend/public/benchmark.json)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Processes to play games in parallel (default: one per CPU core)",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Replay games even if cached results exist (use after changing an agent's code)",
    )
    parser.add_argument(
        "--experiment",
        metavar="DIR",
        type=Path,
        default=None,
        help="Also record per-game results in DIR/evaluations.json (e.g. experiments/ntuple-td-1m)",
    )
    args = parser.parse_args()

    specs = ALL_AGENTS if args.agent == "all" else ((args.agent, args.depth),)
    cache = ResultCache()
    results: list[EvalResult] = []
    for name, depth in specs:
        agent = load_agent(name, checkpoint=args.checkpoint, depth=depth, cutoff=args.cutoff)
        key = RunKey.for_agent(agent, n_games=args.num_games)
        result = None if args.rerun else cache.get(key)
        from_cache = result is not None
        if result is None:
            print(f"\nEvaluating {agent.label}…")
            result = run_games(agent, n=args.num_games, workers=args.workers)
            cache.put(key, result)
        else:
            print(f"\nUsing cached results for {agent.label}")
        results.append(result)

        if args.experiment:
            checkpoint = str(agent.checkpoint) if agent.checkpoint else None
            record = EvaluationRecord(key=key, checkpoint=checkpoint, environment=Environment.capture(), recorded_at=now(), from_cache=from_cache, result=result)
            record_evaluation(args.experiment, record)
    print_results(results)

    if args.json:
        payload = [
            {
                "label": r.label,
                "n_games": r.n_games,
                "avg_score": round(r.avg_score),
                "avg_max_tile": round(r.avg_max_tile),
                "tile_pcts": {str(t): round(r.tile_pct(t), 1) for t in TILES},
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
