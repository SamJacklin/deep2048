"""Stage 8's evaluations and tables: AutoStep TIDBD against the temporal coherence champion.

    uv run python scripts/stage8.py evaluate CHECKPOINT --depth 3 --split dev    # 1,000 seeded games, saved per game
    uv run python scripts/stage8.py bench                                        # the thread benchmark's table
    uv run python scripts/stage8.py screen                                       # the screen's table (also screening.csv)
    uv run python scripts/stage8.py curve ARM                                    # an arm's dev evaluations against the champion

Seeds, fixed before any training: the screen's rounds use the archive's usual seeds 0-999 (greedy only); development
uses 800,000 onwards (1,000 games, the next 1,000 if a decision is close); the final test uses 900,000 onwards (2,000
games, the next 2,000 if it's close), and isn't looked at until one checkpoint has been chosen on the development seeds.
Results are stored per game in experiments/stage8-tidbd/evaluations/, named by network, depth and seeds, so comparisons
are paired game for game. An evaluation already stored isn't played again.
"""

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from deep_2048.agents.ntuple.network import NTupleNetwork
from deep_2048.agents.ntuple.search import play_seeded_counted
from deep_2048.bench import file_sha256

OUTPUT = Path("experiments/stage8-tidbd")
EVALUATIONS = OUTPUT / "evaluations"
CHECKPOINTS = Path("checkpoints/ntuple/stage8-tidbd")
INCUMBENT = Path("checkpoints/ntuple/decision-8x6-tc-reset/games_2000000.npz")
# The champion's own history: the long run to 1.5M games, then the reset arm to 2M
TC_SNAPSHOTS = {
    **{games: Path(f"checkpoints/ntuple/long-8x6/games_{games}.npz") for games in (500_000, 1_500_000)},
    **{games: Path(f"checkpoints/ntuple/decision-8x6-tc-reset/games_{games}.npz") for games in (1_750_000, 2_000_000)},
}
SPLITS = {"screen": 0, "dev": 800_000, "dev2": 801_000, "test": 900_000, "test2": 902_000}
SPLIT_GAMES = {"screen": 1_000, "dev": 1_000, "dev2": 1_000, "test": 2_000, "test2": 2_000}
ARMS = {"A": (0.00125, 0.003), "B": (0.00125, 0.01), "C": (0.004, 0.003), "D": (0.004, 0.01), "E": (0.01, 0.003), "F": (0.01, 0.01)}
TAU = 10_000


def evaluation_path(name: str, depth: int, split: str) -> Path:
    return EVALUATIONS / f"{name}-depth{depth}-{split}.npz"


def evaluate(checkpoint: Path, name: str, depth: int, split: str) -> dict[str, np.ndarray]:
    """Play the split's seeded games with the checkpoint at this depth, or load them if they've been played."""
    path = evaluation_path(name, depth, split)
    if path.exists():
        return dict(np.load(path))
    network, metadata = NTupleNetwork.load(checkpoint)
    seeds = np.arange(SPLITS[split], SPLITS[split] + SPLIT_GAMES[split])
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
        games = list(pool.map(lambda seed: play_seeded_counted(network.weights, network.layout, depth, 0.0, np.random.default_rng(int(seed))), seeds))
    scores, max_tiles, moves = (np.array(column) for column in zip(*games))
    result = {
        "seeds": seeds, "scores": scores, "max_tiles": max_tiles, "moves": moves, "seconds": np.array(time.monotonic() - started),
        "checkpoint": np.array(str(checkpoint)), "sha256": np.array(file_sha256(checkpoint)),
        "training_games": np.array(metadata.games_played), "training_moves": np.array(metadata.moves_played),
    }  # fmt: skip
    EVALUATIONS.mkdir(parents=True, exist_ok=True)
    np.savez(path, **result)
    return result


def interval(values: np.ndarray) -> float:
    """The half-width of a 95% confidence interval for the mean."""
    return float(1.96 * values.std(ddof=1) / np.sqrt(len(values)))


def summary(result: dict[str, np.ndarray]) -> dict[str, float]:
    scores, tiles = result["scores"], result["max_tiles"]
    reached_16384 = tiles >= 16384
    return {
        "mean": float(scores.mean()), "ci": interval(scores), "median": float(np.median(scores)),
        "r8192": float(np.mean(tiles >= 8192)), "r16384": float(reached_16384.mean()), "r32768": float(np.mean(tiles >= 32768)),
        "conversion": float(np.mean(tiles[reached_16384] >= 32768)) if reached_16384.any() else float("nan"),
        "moves": float(result["moves"].mean()),
    }  # fmt: skip


def describe(label: str, result: dict[str, np.ndarray], control: dict[str, np.ndarray] | None = None) -> str:
    """One table row: the values, then the paired difference from the control on the same seeds."""
    s = summary(result)
    row = (
        f"| {label} | {s['mean']:,.0f} ± {s['ci']:,.0f} | {s['median']:,.0f} | {s['r8192']:.1%} | {s['r16384']:.1%} | {s['r32768']:.1%} "
        f"| {s['conversion']:.1%} | {s['moves']:,.0f} |"
    )
    if control is not None:
        assert (result["seeds"] == control["seeds"]).all()
        difference = result["scores"] - control["scores"]
        conversion = (result["max_tiles"] >= 32768).astype(float) - (control["max_tiles"] >= 32768)
        row += f" {difference.mean():+,.0f} ± {interval(difference):,.0f} | {conversion.mean():+.1%} ± {interval(conversion):.1%} |"
    return row


HEADER = "| Network | Mean score ± 95% | Median | ≥8192 | ≥16384 | ≥32768 | 16384→32768 | Moves |"
RULE = "|---|---|---|---|---|---|---|---|"


def history(directory: Path, name: str) -> list[dict[str, str]]:
    path = directory / name
    return list(csv.DictReader(path.open())) if path.exists() else []


def bench_table() -> None:
    """Throughput of each benchmark run, from the batches it logged."""
    print("| Run | Threads | Games | Moves | Wall s | Moves/s | Games/s | CPU s | Peak memory GB | Avg training score | Max effective step | Non-finite |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for directory in sorted((OUTPUT / "bench").glob("*/")):
        manifest = json.loads((directory / "manifest.json").read_text())
        session = manifest["sessions"][-1]
        rows = history(directory, "training_history.csv")
        first_games = session["started_with"]["games_played"]
        rows = [r for r in rows if int(r["games"]) > first_games]
        previous = [first_games, *(int(r["games"]) for r in rows)]
        seconds = sum((int(r["games"]) - before) / float(r["games_per_second"]) for r, before in zip(rows, previous))
        moves = session["finished_with"]["moves_played"] - session["started_with"]["moves_played"]
        games = session["finished_with"]["games_played"] - first_games
        score = np.mean([float(r["avg_score"]) for r in rows])
        optimiser = [r for r in history(directory, "optimizer_history.csv") if int(r["games"]) > first_games]
        step = f"{max(float(r['max_effective_step']) for r in optimiser):.3f}" if optimiser else "n/a"
        non_finite = f"{sum(float(r['non_finite']) for r in optimiser):.0f}" if optimiser else "n/a"
        threads = session["settings"]["THREADS"]
        print(
            f"| {directory.name} | {threads} | {games:,} | {moves / 1e6:.1f}M | {seconds:.0f} | {moves / seconds:,.0f} | {games / seconds:.1f} "
            f"| {session['cpu_seconds']:.0f} | {session['peak_memory_gb']:.2f} | {score:,.0f} | {step} | {non_finite} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("checkpoint", type=Path)
    evaluation.add_argument("--name", required=True, help="What to call the network in the stored results, such as arm-D-100000 or tc-2000000")
    evaluation.add_argument("--depth", type=int, default=1)
    evaluation.add_argument("--split", choices=list(SPLITS), default="dev")
    commands.add_parser("bench")
    args = parser.parse_args()

    if args.command == "evaluate":
        result = evaluate(args.checkpoint, args.name, args.depth, args.split)
        print(HEADER + "\n" + RULE + "\n" + describe(f"{args.name}, depth {args.depth}, {args.split} ({float(result['seconds']):.0f} s)", result))
    elif args.command == "bench":
        bench_table()


if __name__ == "__main__":
    main()
