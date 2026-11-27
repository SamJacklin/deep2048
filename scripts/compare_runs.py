"""Compare runs' evaluations game for game against a reference, with paired bootstrap 95% intervals.

Usage:
    uv run python scripts/compare_runs.py decision2-8x6-continue decision11-8x6-rate-0.005
    uv run python scripts/compare_runs.py decision12-8x6-two-stage-reset decision12b-8x6-two-stage-continued --reference-games 2500000

Each evaluation of a run is paired with the reference's evaluation with the same search, at the same game
count, or at --reference-games if given. Evaluations play game i with seed i, so the pairs share their tile
spawns until the agents' moves differ. Reports the change in average score, in the rates of reaching 16384 and
32768, and in the conditional rate of making a 32768 in games that reached 16384, each with an interval from
resampling games (pairs kept together).
"""

import argparse
import json
from pathlib import Path

import numpy as np

RESAMPLES = 10_000


def parse_label(label: str) -> tuple[int, str]:
    """Return an evaluation's training games and search ("greedy" or "depth N") from its label."""
    games = int(label[label.index("(") + 1 : label.index(" games")].replace(",", ""))
    search = f"depth {label.split('depth=')[1].split()[0]}" if "depth=" in label else "greedy"
    if "cutoff=" in label:
        search += f", cutoff {label.split('cutoff=')[1].split()[0]}"
    return games, search


def load(run: str) -> dict[tuple[int, str], dict]:
    records = json.loads((Path("experiments") / run / "evaluations.json").read_text())
    return {parse_label(r["key"]["label"]): r for r in records}


def statistics(scores: np.ndarray, tiles: np.ndarray) -> np.ndarray:
    """Average score, ≥16384 rate, ≥32768 rate and the 16384→32768 conversion rate, along the last axis's games."""
    reached = tiles >= 16384
    made = tiles >= 32768
    conversion = made.sum(-1) / np.maximum(reached.sum(-1), 1)
    return np.stack([scores.mean(-1), reached.mean(-1), made.mean(-1), conversion], -1)


def paired_bootstrap(reference: tuple[np.ndarray, np.ndarray], other: tuple[np.ndarray, np.ndarray], seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return the statistics' differences (other minus reference) and their 95% intervals, shape (4,) and (4, 2)."""
    n = len(reference[0])
    picks = np.random.default_rng(seed).integers(0, n, size=(RESAMPLES, n))
    differences = statistics(other[0][picks], other[1][picks]) - statistics(reference[0][picks], reference[1][picks])
    observed = statistics(*other) - statistics(*reference)
    return observed, np.percentile(differences, [2.5, 97.5], axis=0).T


def arrays(record: dict) -> tuple[np.ndarray, np.ndarray]:
    return np.array(record["result"]["scores"], dtype=float), np.array(record["result"]["max_tiles"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare runs' evaluations with a reference's, game for game")
    parser.add_argument("reference", help="Run folder under experiments/ to compare against")
    parser.add_argument("runs", nargs="+", help="Run folders to compare")
    parser.add_argument("--reference-games", type=int, help="Pair every evaluation with the reference's at this game count")
    args = parser.parse_args()

    reference = load(args.reference)
    print("| Run | Games | Search | Average | Δ average | Δ ≥16384 | Δ ≥32768 | 16384→32768 | Δ 16384→32768 |")
    print("|---|---:|---|---:|---:|---:|---:|---:|---:|")
    for run in [args.reference, *args.runs]:
        for (games, search), record in sorted(load(run).items()):
            scores, tiles = arrays(record)
            conversion = statistics(scores, tiles)[3]
            pair = reference.get((args.reference_games or games, search))
            if run == args.reference or pair is None:
                print(f"| {run} | {games:,} | {search} | {scores.mean():,.0f} | — | — | — | {conversion:.1%} | — |")
                continue
            key, pair_key = record["key"], pair["key"]
            if (key["n_games"], key["seed_offset"]) != (pair_key["n_games"], pair_key["seed_offset"]):
                raise SystemExit(f"{run} at {games:,} ({search}) didn't play the same seeded games as {args.reference}")
            change, interval = paired_bootstrap(arrays(pair), (scores, tiles))
            score = f"{change[0]:+,.0f} [{interval[0, 0]:+,.0f}, {interval[0, 1]:+,.0f}]"
            rates = [f"{change[i]:+.1%} [{interval[i, 0]:+.1%}, {interval[i, 1]:+.1%}]" for i in (1, 2, 3)]
            print(f"| {run} | {games:,} | {search} | {scores.mean():,.0f} | {score} | {rates[0]} | {rates[1]} | {conversion:.1%} | {rates[2]} |")


if __name__ == "__main__":
    main()
