"""Print each run's evaluations as a Markdown table: games, search depth, average ± 95% interval, and tile rates.

Usage:
    uv run python scripts/summarise_runs.py ryzen-recipe ryzen-optimism
"""

import json
import sys
from pathlib import Path

import numpy as np


def main(runs: list[str]) -> None:
    for run in runs:
        records = json.loads((Path("experiments") / run / "evaluations.json").read_text())
        print(f"\n### {run}\n")
        print("| Games | Search | Average ± 95% | Median | ≥16384 | ≥32768 |")
        print("|---:|---|---:|---:|---:|---:|")
        rows = []
        for record in records:
            label = record["key"]["label"]
            games = int(label[label.index("(") + 1 : label.index(" games")].replace(",", ""))
            search = "depth 3" if "depth=3" in label else "greedy"
            scores, tiles = np.array(record["result"]["scores"]), np.array(record["result"]["max_tiles"])
            interval = 1.96 * scores.std(ddof=1) / np.sqrt(len(scores))
            average = f"{scores.mean():,.0f} ± {interval:,.0f}"
            rates = f"{(tiles >= 16384).mean():.1%} | {(tiles >= 32768).mean():.1%}"
            rows.append((games, search, f"| {games:,} | {search} | {average} | {np.median(scores):,.0f} | {rates} |"))
        for _, _, row in sorted(rows):
            print(row)


if __name__ == "__main__":
    main(sys.argv[1:])
