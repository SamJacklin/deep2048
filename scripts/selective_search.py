"""Stage 5, before any training: how much of depth-3 search's late-game gain does searching only some decisions recover?

Usage:
    uv run python scripts/selective_search.py run checkpoints/ntuple/decision-8x6-tc-reset/games_2000000.npz
    uv run python scripts/selective_search.py report

Every policy plays greedily except at late decisions (largest tile 16384 or more) whose greedy relative margin between
the top two moves lies in its band, where it searches exactly to depth 3. Bands are percentiles of that margin over
the dev split's late positions: the closest 10%, 25% and 50% of decisions, each quarter-band on its own, and every
late decision. All play on from the dev split's first-16384 positions, 8 continuations each, with the same spawns
for every policy (common random numbers), so differences are paired by position and continuation.
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

from deep_2048.agents.ntuple import NTupleNetwork
from deep_2048.agents.ntuple.guidance import play_selective

CORPUS = Path("experiments/late-game-corpus/positions.csv")
SAMPLES = Path("output/late-game-corpus/samples.csv")
OUTPUT = Path("experiments/search-teacher/selective")
CONTINUATIONS = 8
POOL = ThreadPoolExecutor(max_workers=os.cpu_count())


def read_rows(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def seed_of(row: dict) -> int:
    return (1 if row["policy"] == "greedy" else 2) * 10**10 + int(row["seed"]) * 10**4 + int(row["move"]) % 10**4


def margin_percentiles() -> dict[str, float]:
    margins = [float(r["greedy_margin"]) for r in read_rows(SAMPLES) if r["split"] == "dev" and int(r["max_tile"]) >= 16384]
    finite = np.array([m for m in margins if np.isfinite(m)])
    return {f"p{p}": float(np.percentile(finite, p)) for p in (10, 25, 50, 75)} | {"positions": len(finite)}


def policies(cuts: dict[str, float]) -> dict[str, tuple[float, float]]:
    p10, p25, p50, p75 = (cuts[k] for k in ("p10", "p25", "p50", "p75"))
    return {
        "greedy": (0.0, 0.0),
        "closest 10%": (0.0, p10),
        "closest 25%": (0.0, p25),
        "closest 50%": (0.0, p50),
        "band 10-25%": (p10, p25),
        "band 25-50%": (p25, p50),
        "band 50-75%": (p50, p75),
        "band 75-100%": (p75, np.inf),
        "every late decision": (0.0, np.inf),
    }


def run(base_path: str) -> None:
    network, _ = NTupleNetwork.load(base_path)
    rows = [r for r in read_rows(CORPUS) if r["kind"] == "first-16384" and r["split"] == "dev"]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cuts = margin_percentiles()
    (OUTPUT / "margins.json").write_text(json.dumps(cuts, indent=2) + "\n")
    jobs = [(p, j) for p in range(len(rows)) for j in range(CONTINUATIONS)]
    for name, (low, high) in policies(cuts).items():
        path = OUTPUT / f"{name}.npz"
        if path.exists():
            continue
        start, cpu = time.perf_counter(), time.process_time()
        games = list(POOL.map(lambda job: play_selective(network.weights, network.layout, np.uint64(int(rows[job[0]]["board"], 16)), 3, 14, low, high,
                                                         np.random.default_rng([seed_of(rows[job[0]]), job[1]])), jobs))
        fields = np.array(games, dtype=np.int64).reshape(len(rows), CONTINUATIONS, 7)
        np.savez(path, scores=fields[..., 0], tiles=fields[..., 1], decisions=fields[..., 2], searched=fields[..., 3], disagreed=fields[..., 4],
                 nodes=fields[..., 5], leaves=fields[..., 6], wall_seconds=time.perf_counter() - start, cpu_seconds=time.process_time() - cpu)
        print(f"Played {name}", flush=True)


def interval(values: np.ndarray, rate: bool = False) -> str:
    mean, half = values.mean(), 1.96 * values.std(ddof=1) / np.sqrt(len(values))
    return f"{mean:+.1%} ± {half:.1%}" if rate else f"{mean:+,.0f} ± {half:,.0f}"


def report() -> None:
    cuts = json.loads((OUTPUT / "margins.json").read_text())
    print(f"Margin cut points over {cuts['positions']:,} dev late positions: " + ", ".join(f"{k} {cuts[k]:.2e}" for k in ("p10", "p25", "p50", "p75")))
    results = {name: np.load(OUTPUT / f"{name}.npz") for name in policies(cuts) if (OUTPUT / f"{name}.npz").exists()}
    greedy, full = results.get("greedy"), results.get("every late decision")
    if greedy is None:
        return
    g_made, g_added = (greedy["tiles"] >= 32768).mean(axis=1), greedy["scores"].mean(axis=1)
    gap_made = gap_added = None
    if full is not None:
        gap_made = (full["tiles"] >= 32768).mean() - (greedy["tiles"] >= 32768).mean()
        gap_added = full["scores"].mean() - greedy["scores"].mean()
    print(f"\n### Selective depth-3 search from {g_made.size} dev first-16384 positions, {greedy['scores'].shape[1]} continuations each (paired with greedy, ± 95%)\n")
    print("| Policy | Decisions searched | Search disagrees with greedy | Makes 32768 | vs greedy | Recovered | Score added | vs greedy | Recovered | "
          "Score gained per 1,000 searches | Chance nodes per searched decision | CPU s per continuation |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, d in results.items():
        made, added = (d["tiles"] >= 32768).mean(axis=1), d["scores"].mean(axis=1)
        searched, decisions = d["searched"].sum(), d["decisions"].sum()
        disagree = d["disagreed"].sum() / searched if searched else 0.0
        recovered_made = f"{(made.mean() - g_made.mean()) / gap_made:.0%}" if gap_made else ""
        recovered_added = f"{(added.mean() - g_added.mean()) / gap_added:.0%}" if gap_added else ""
        per_search = f"{(d['scores'].sum() - greedy['scores'].sum()) / searched * 1000:,.0f}" if searched else ""
        nodes = f"{d['nodes'].sum() / searched:,.0f}" if searched else ""
        cpu = float(d["cpu_seconds"]) / d["scores"].size
        print(f"| {name} | {searched / decisions:.1%} | {disagree:.1%} | {made.mean():.1%} | {interval(made - g_made, True)} | {recovered_made} | "
              f"{added.mean():,.0f} | {interval(added - g_added)} | {recovered_added} | {per_search} | {nodes} | {cpu:.2f} |")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 5: selective depth-3 search, before training")
    parser.add_argument("step", choices=("run", "report"))
    parser.add_argument("base", nargs="?")
    args = parser.parse_args()
    if args.step == "run":
        run(args.base)
    report()


if __name__ == "__main__":
    main()
