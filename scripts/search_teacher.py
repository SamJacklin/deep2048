"""Stage 5's training arms: does search during training move search's decisions into the network?

Usage:
    BASE=checkpoints/ntuple/decision-8x6-tc-reset/games_2000000.npz
    uv run python scripts/search_teacher.py choose "$BASE"         # pick the search band and ranking threshold from dev
    uv run python scripts/search_teacher.py train "$BASE" A        # likewise B, C1, C2
    uv run python scripts/search_teacher.py evaluate "$BASE" A     # likewise baseline, B, C1, C2
    uv run python scripts/search_teacher.py report

Every arm continues the 2M network for GAMES games with the long run's recipe and its temporal coherence sums reset:
- A: greedy TD(0), as trained so far;
- B: TD(0) on games that search the chosen band of late decisions to depth 3 and play the searched move;
- C1, C2: B plus ranking updates where search disagrees with greedy play by at least the dev median preference,
  closing 10% (C1) or 25% (C2) of the gap between the two moves' reward + value each time.
The band is the narrowest of the closest 10%, 25% or 50% of late decisions by greedy margin that recovered at least half
of depth 3's conversion gain in scripts/selective_search.py, or the closest 50% if none did.

Each arm is evaluated on dev data only: 32 greedy and 8 depth-3 continuations from the dev split's first-16384
positions (the same spawns for every arm), and 1,000 greedy and 1,000 depth-3 full games on seeds 10,000-10,999.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from deep_2048.agents.ntuple import NTupleNetwork
from deep_2048.agents.ntuple.diagnostics import greedy_values, relative_margin
from deep_2048.agents.ntuple.guidance import _first_best
from deep_2048.agents.ntuple.search import action_values, play_from, play_seeded
from deep_2048.experiments import RunManifest

CORPUS = Path("experiments/late-game-corpus/positions.csv")
SAMPLES = Path("output/late-game-corpus/samples.csv")
SELECTIVE = Path("experiments/search-teacher/selective")
OUTPUT = Path("experiments/search-teacher")
GAMES = 50_000
FULL_GAME_SEEDS = range(10_000, 11_000)
RECIPE = {"NTUPLE_NETWORK": "8x6", "NTUPLE_LEARNING_RATE": "0.02", "NTUPLE_TEMPORAL_COHERENCE": "true", "NTUPLE_DOWNGRADE_TILE": "16384",
          "NTUPLE_LATE_START_FRACTION": "0.2", "NTUPLE_LATE_START_TILE": "8192", "NTUPLE_THREADS": "8"}
RANK_STEPS = {"C1": 0.1, "C2": 0.25}
POOL = ThreadPoolExecutor(max_workers=os.cpu_count())


def read_rows(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def seed_of(row: dict) -> int:
    return (1 if row["policy"] == "greedy" else 2) * 10**10 + int(row["seed"]) * 10**4 + int(row["move"]) % 10**4


def run_name(arm: str) -> str:
    return f"search-teacher-{arm}"


# ---------------------------------------------------------------- choosing the band and threshold


def choose(base_path: str) -> None:
    cuts = json.loads((SELECTIVE / "margins.json").read_text())
    greedy, full = (np.load(SELECTIVE / f"{name}.npz") for name in ("greedy", "every late decision"))
    gap = (full["tiles"] >= 32768).mean() - (greedy["tiles"] >= 32768).mean()
    recovered = {}
    for percent in (10, 25, 50):
        d = np.load(SELECTIVE / f"closest {percent}%.npz")
        recovered[percent] = float(((d["tiles"] >= 32768).mean() - (greedy["tiles"] >= 32768).mean()) / gap)
    chosen = next((p for p in (10, 25, 50) if recovered[p] >= 0.5), 50)
    high = cuts[f"p{chosen}"]

    # The ranking threshold: the median of search's preference over greedy play's move, as a share of its value,
    # where they disagree on dev late positions in the band
    network, _ = NTupleNetwork.load(base_path)
    rows = [r for r in read_rows(SAMPLES) if r["split"] == "dev" and int(r["max_tile"]) >= 16384 and float(r["greedy_margin"]) < high]
    rows = [rows[i] for i in np.random.default_rng(0).choice(len(rows), min(2000, len(rows)), replace=False)]

    def preference(row: dict) -> float:
        board = np.uint64(int(row["board"], 16))
        values = greedy_values(network.weights, network.layout, board)
        searched = action_values(network.weights, network.layout, board, 3)
        greedy_move, teacher = _first_best(values), _first_best(searched)
        return np.nan if teacher == greedy_move else (searched[teacher] - searched[greedy_move]) / abs(searched[teacher])

    preferences = np.array(list(POOL.map(preference, rows)))
    disagreements = preferences[~np.isnan(preferences)]
    choice = {"recovered by band": recovered, "closest percent": chosen, "margin high": high, "dev positions": len(rows),
              "disagreements": int(disagreements.size), "rank min gap": float(np.median(disagreements)),
              "preference percentiles": {f"p{p}": float(np.percentile(disagreements, p)) for p in (10, 25, 50, 75, 90)}}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "choice.json").write_text(json.dumps(choice, indent=2) + "\n")
    print(json.dumps(choice, indent=2))


# ---------------------------------------------------------------- training


def train(base_path: str, arm: str) -> None:
    if (Path("checkpoints/ntuple") / run_name(arm) / f"games_{2_000_000 + GAMES}.npz").exists():
        print(f"{arm} already trained")
        return
    choice = json.loads((OUTPUT / "choice.json").read_text())
    env = os.environ | RECIPE
    if arm != "A":
        env |= {"NTUPLE_GUIDE_DEPTH": "3", "NTUPLE_GUIDE_TILE": "16384", "NTUPLE_GUIDE_MARGIN_LOW": "0", "NTUPLE_GUIDE_MARGIN_HIGH": str(choice["margin high"])}
    if arm in RANK_STEPS:
        env |= {"NTUPLE_RANK_STEP": str(RANK_STEPS[arm]), "NTUPLE_RANK_MARGIN": "1", "NTUPLE_RANK_MIN_GAP": str(choice["rank min gap"])}
    end = 2_000_000 + GAMES
    command = [sys.executable, "-u", "scripts/train_ntuple.py", "--run", run_name(arm), "--resume", base_path, "--reset-coherence", "--games", str(end), "--snapshot-at", str(end)]
    with open(OUTPUT / f"train-{arm}.log", "w") as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


# ---------------------------------------------------------------- evaluation


def checkpoint_of(base_path: str, arm: str) -> str:
    return base_path if arm == "baseline" else f"checkpoints/ntuple/{run_name(arm)}/games_{2_000_000 + GAMES}.npz"


def evaluate(base_path: str, arm: str) -> None:
    path = OUTPUT / "evaluations" / f"{arm}.npz"
    if path.exists():
        return
    network, _ = NTupleNetwork.load(checkpoint_of(base_path, arm))
    w, layout = network.weights, network.layout
    rows = [r for r in read_rows(CORPUS) if r["kind"] == "first-16384" and r["split"] == "dev"]
    out = {}
    for name, depth, n in (("greedy", 1, 32), ("depth3", 3, 8)):
        jobs = [(p, j) for p in range(len(rows)) for j in range(n)]
        games = list(POOL.map(lambda job: play_from(w, layout, np.uint64(int(rows[job[0]]["board"], 16)), depth, 0.0, np.random.default_rng([seed_of(rows[job[0]]), job[1]])), jobs))
        out[f"{name}_scores"] = np.array([s for s, _ in games]).reshape(len(rows), n)
        out[f"{name}_tiles"] = np.array([t for _, t in games]).reshape(len(rows), n)
        full = list(POOL.map(lambda s: play_seeded(w, layout, depth, 0.0, np.random.default_rng(s)), FULL_GAME_SEEDS))
        out[f"full_{name}_scores"], out[f"full_{name}_tiles"] = np.array([s for s, _ in full]), np.array([t for _, t in full])
    # How often its greedy move differs from the base's on dev late positions
    base, _ = NTupleNetwork.load(base_path)
    late = [np.uint64(int(r["board"], 16)) for r in read_rows(SAMPLES) if r["split"] == "dev" and int(r["max_tile"]) >= 16384 and np.isfinite(float(r["greedy_margin"]))]
    late = [late[i] for i in np.random.default_rng(0).choice(len(late), min(3000, len(late)), replace=False)]
    out["late_changed"] = np.array([_first_best(greedy_values(w, layout, b)) != _first_best(greedy_values(base.weights, base.layout, b)) for b in late])
    out["late_margin"] = np.array([relative_margin(greedy_values(w, layout, b)) for b in late])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)


# ---------------------------------------------------------------- report


def interval(values: np.ndarray, rate: bool = False) -> str:
    mean, half = values.mean(), 1.96 * values.std(ddof=1) / np.sqrt(len(values))
    return f"{mean:+.1%} ± {half:.1%}" if rate else f"{mean:+,.0f} ± {half:,.0f}"


def training_cost(arm: str) -> dict:
    if arm == "baseline":
        return {}
    directory = OUTPUT.parent / run_name(arm)
    history = read_rows(directory / "training_history.csv")
    manifest = RunManifest.load_or_create(directory)
    session = manifest.sessions[-1]
    cost = {"moves": int(history[-1]["moves"]) - int(session.started_with["moves_played"]), "cpu hours": (session.cpu_seconds or 0) / 3600,
            "wall hours": ((session.finished_at - session.started_at).total_seconds() / 3600) if session.finished_at else 0.0}
    guide_path = directory / "guide_history.csv"
    if guide_path.exists():
        totals = np.array([[int(v) for v in list(r.values())[1:]] for r in read_rows(guide_path)]).sum(axis=0)
        cost |= dict(zip(("decisions", "searched", "disagreed", "ranked", "chance nodes", "leaf values"), (int(v) for v in totals)))
    return cost


def report() -> None:
    arms = {a: np.load(OUTPUT / "evaluations" / f"{a}.npz") for a in ("baseline", "A", "B", "C1", "C2") if (OUTPUT / "evaluations" / f"{a}.npz").exists()}
    if "baseline" not in arms:
        return
    choice = json.loads((OUTPUT / "choice.json").read_text())
    print(f"Search band: the closest {choice['closest percent']}% of late decisions (greedy relative margin < {choice['margin high']:.3g}); "
          f"recovered by band: {choice['recovered by band']}; ranking threshold {choice['rank min gap']:.3g} ({choice['disagreements']} dev disagreements)\n")
    b = arms["baseline"]
    print("| Arm | Greedy conversion | Δ | Greedy score added | Δ | Depth-3 conversion | Δ | Depth-3 score added | Δ | Full greedy | Δ | Full depth 3 | Δ | "
          "32768, full depth 3 | Late moves changed |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for arm, d in arms.items():
        cells = []
        for name in ("greedy", "depth3"):
            made, base_made = (d[f"{name}_tiles"] >= 32768).mean(axis=1), (b[f"{name}_tiles"] >= 32768).mean(axis=1)
            added, base_added = d[f"{name}_scores"].mean(axis=1), b[f"{name}_scores"].mean(axis=1)
            cells += [f"{made.mean():.1%}", "—" if arm == "baseline" else interval(made - base_made, True), f"{added.mean():,.0f}", "—" if arm == "baseline" else interval(added - base_added)]
        for name in ("greedy", "depth3"):
            full = d[f"full_{name}_scores"].astype(float)
            cells += [f"{full.mean():,.0f}", "—" if arm == "baseline" else interval(full - b[f"full_{name}_scores"])]
        cells += [f"{(d['full_depth3_tiles'] >= 32768).mean():.1%}", f"{d['late_changed'].mean():.1%}"]
        print(f"| {arm} | " + " | ".join(cells) + " |")
    print("\n| Arm | Training moves | Decisions searched | Search disagreed | Ranking updates | Chance nodes | CPU hours | Wall hours |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for arm in arms:
        c = training_cost(arm)
        if c:
            searches = f"{c.get('searched', 0):,} | {c.get('disagreed', 0):,} | {c.get('ranked', 0):,} | {c.get('chance nodes', 0):,}"
            print(f"| {arm} | {c['moves']:,} | {searches} | {c['cpu hours']:.2f} | {c['wall hours']:.2f} |")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 5's training arms")
    parser.add_argument("step", choices=("choose", "train", "evaluate", "report"))
    parser.add_argument("base", nargs="?")
    parser.add_argument("arm", nargs="?")
    args = parser.parse_args()
    if args.step == "choose":
        choose(args.base)
    elif args.step == "train":
        train(args.base, args.arm)
    elif args.step == "evaluate":
        evaluate(args.base, args.arm)
    report()


if __name__ == "__main__":
    main()
