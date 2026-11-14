"""Stage 6: a dedicated late-game network. Stage 1 is the frozen 2M network, and a second stage, trained only from
genuine first-16384 positions, plays every board holding a 16384 or more.

Usage:
    uv run python scripts/late_network.py pools     # genuine first-16384 positions from seeded games
    uv run python scripts/late_network.py build     # the three two-stage start checkpoints
    uv run python scripts/late_network.py check     # equivalence and frozen-stage checks, before any training
    uv run python scripts/late_network.py run       # all of the above, then every arm in segments, evaluating at each stop
    uv run python scripts/late_network.py report

Every arm's stage 1 is round 1's TC-reset network at 2M games. Its stage 2 starts as a copy of that network, of the
long run's 500k network, or as zeros, with fresh temporal coherence sums and a continuation count of 0, and trains at
each base rate in RATES. Training games each start from a position drawn from the training pool and run to the end.
Since every board in them holds a 16384, only stage 2 is ever read or updated, so stage 1 stays frozen.

Arms train in segments to each stop in STOPS, resuming from their latest checkpoint, and are evaluated at each stop,
so only one checkpoint per arm is ever on disk. Evaluations play on from the dev pools (greedy continuations at every
stop, depth 3 at DEPTH3_AT) and whole games on seeds 10,000-10,999, with the same spawns for every arm, so all of
them pair with the control: the 2M network alone, which the 2M-copy start plays identically to (checked).
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from add_stage import add_stage  # noqa: E402

from deep_2048.agents.ntuple import NTupleNetwork  # noqa: E402
from deep_2048.agents.ntuple.diagnostics import boundary_boards, first_board, greedy_values  # noqa: E402
from deep_2048.agents.ntuple.guidance import play_selective  # noqa: E402
from deep_2048.agents.ntuple.models import CheckpointMetadata  # noqa: E402
from deep_2048.agents.ntuple.search import action_values, play_from, play_seeded  # noqa: E402

BASE = "checkpoints/ntuple/decision-8x6-tc-reset/games_2000000.npz"
SOURCES = {"2m": BASE, "500k": "checkpoints/ntuple/long-8x6/games_500000.npz", "scratch": None}
NETWORKS = {"2m": BASE, "1.5m": "checkpoints/ntuple/long-8x6/games_1500000.npz"}
RATES = {"2m": (0.02, 0.005, 0.00125), "500k": (0.02, 0.005, 0.00125), "scratch": (0.02, 0.005)}
STOPS = tuple(range(100_000, 1_000_001, 100_000))
DEPTH3_AT = (500_000, 1_000_000)
# (pool, policy, network that played, first seed, positions): seed ranges disjoint from each other and every earlier one
POOLS = (
    ("train", "greedy", "2m", 20_000_000, 100_000),
    ("dev", "greedy", "2m", 30_000_000, 5_000),
    ("test", "greedy", "2m", 40_000_000, 5_000),
    ("dev", "depth3", "2m", 31_000_000, 2_000),
    ("test", "depth3", "2m", 41_000_000, 2_000),
    ("dev", "greedy", "1.5m", 32_000_000, 2_000),
    ("test", "greedy", "1.5m", 42_000_000, 2_000),
)
FULL_GAME_SEEDS = range(10_000, 11_000)
# Continuations per position at every stop, by dev pool; depth-3 continuations use the first 1,000 greedy-entry positions
CONTINUATIONS = {("greedy", "2m"): 2, ("depth3", "2m"): 2, ("greedy", "1.5m"): 2}
DEPTH3_POSITIONS = 1_000

OUTPUT = Path("experiments/late-network")
POSITIONS = OUTPUT / "positions.csv"
TRAIN_POOL = OUTPUT / "train-pool.npy"
EVALUATIONS = OUTPUT / "evaluations"
STARTS = Path("checkpoints/ntuple/late-network-start")
MIN_FREE_GB = 8
POOL = ThreadPoolExecutor(max_workers=os.cpu_count())


def run_name(start: str, rate: float) -> str:
    return f"late-network-{start}-{rate:g}"


def tables(network: NTupleNetwork) -> int:
    return sum(16 ** len(t) for t in network.tuples)


def read_positions(pool: str, policy: str | None = None, network: str | None = None) -> list[dict]:
    with open(POSITIONS) as f:
        return [r for r in csv.DictReader(f) if r["pool"] == pool and (policy is None or r["policy"] == policy) and (network is None or r["network"] == network)]


def board(row: dict) -> np.uint64:
    return np.uint64(int(row["board"], 16))


# ---------------------------------------------------------------- positions


def pools() -> None:
    if POSITIONS.exists():
        print(f"{POSITIONS} exists")
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    networks = {name: NTupleNetwork.load(path)[0] for name, path in NETWORKS.items()}
    rows, seen = [], set()
    for pool, policy, name, offset, count in POOLS:
        net, depth = networks[name], 1 if policy == "greedy" else 3
        found, games, duplicates, started = [], 0, 0, time.perf_counter()
        while len(found) < count:
            batch = 2_000 if depth == 1 else 200
            seeds = range(offset + games, offset + games + batch)
            results = POOL.map(lambda s: (s, first_board(net.weights, net.layout, depth, 14, np.random.default_rng(s))), seeds)
            for seed, (b, moves, score) in results:
                if b == 0 or len(found) == count:
                    continue
                if int(b) in seen:
                    duplicates += 1
                    continue
                seen.add(int(b))
                found.append((pool, policy, name, seed, moves, score, f"{int(b):016x}"))
            games += batch
        rows += found
        print(f"{pool} {policy} {name}: {count:,} distinct positions from seeds {offset:,}-{found[-1][3]:,} "
              f"({games:,} games played, {duplicates} duplicates dropped) in {time.perf_counter() - started:.0f} s", flush=True)
    with open(POSITIONS, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("pool", "policy", "network", "seed", "moves", "score", "board"))
        writer.writerows(rows)
    np.save(TRAIN_POOL, np.array([int(r[6], 16) for r in rows if r[0] == "train"], dtype=np.uint64))


# ---------------------------------------------------------------- start checkpoints


def build() -> None:
    base, _ = NTupleNetwork.load(BASE)
    n = tables(base)
    for start, source in SOURCES.items():
        path = STARTS / f"{start}.npz"
        if path.exists():
            continue
        network = add_stage(base, 16384)
        if source is None:
            network.weights[n : 2 * n] = 0
        elif source != BASE:
            donor, _ = NTupleNetwork.load(source)
            if donor.features != base.features or donor.tuples != base.tuples:
                raise ValueError(f"{source} doesn't read boards as {BASE} does")
            network.weights[n : 2 * n] = donor.weights[:n]
        # No coherence sums, so training starts them fresh; the count starts at 0 so it counts continuations
        network.save(path, CheckpointMetadata(games_played=0, moves_played=0, learning_rate=0.0, temporal_coherence=True))
        print(f"Saved {path}: stage 1 from {BASE}, stage 2 from {source or 'zeros'}")


# ---------------------------------------------------------------- checks


def same(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.array_equal(a, b, equal_nan=True))


def check() -> None:
    """Before any training: the 2M-copy start plays exactly as the 2M network, and every start's stage 1 is the 2M's."""
    path = OUTPUT / "checks.json"
    if path.exists():
        print(f"{path} exists")
        return
    base, _ = NTupleNetwork.load(BASE)
    copy, _ = NTupleNetwork.load(STARTS / "2m.npz")
    n = tables(base)
    results = {}
    late = [board(r) for r in read_positions("dev")][:2_000]
    rng = np.random.default_rng(0)
    early = [np.uint64(b) for b in (int(r["board"], 16) for r in csv.DictReader(open("output/late-game-corpus/samples.csv")) if r["split"] == "dev")]
    early = [early[i] for i in rng.choice(len(early), 2_000, replace=False)]
    boundary = np.concatenate([boundary_boards(base.weights, base.layout, 1, 0.0, np.random.default_rng(s)) for s in range(30_000_000, 30_000_040)])
    results["boards"] = {"late": len(late), "sampled": len(early), "boundary": int(boundary.size)}
    for name, boards in (("late", late), ("sampled", early), ("boundary", list(boundary))):
        results[f"greedy values, {name}"] = all(same(greedy_values(base.weights, base.layout, b), greedy_values(copy.weights, copy.layout, b)) for b in boards)
        picks = boards[:300]
        results[f"depth-3 values, {name}"] = all(same(action_values(base.weights, base.layout, b, 3), action_values(copy.weights, copy.layout, b, 3)) for b in picks)
    play = lambda net, depth, seeds: list(POOL.map(lambda s: play_seeded(net.weights, net.layout, depth, 0.0, np.random.default_rng(s)), seeds))  # noqa: E731
    results["greedy games (1,000)"] = play(base, 1, FULL_GAME_SEEDS) == play(copy, 1, FULL_GAME_SEEDS)
    results["depth-3 games (50)"] = play(base, 3, FULL_GAME_SEEDS[:50]) == play(copy, 3, FULL_GAME_SEEDS[:50])
    for start, source in SOURCES.items():
        net, _ = NTupleNetwork.load(STARTS / f"{start}.npz")
        results[f"{start}: stage 1 is the 2M network"] = same(net.weights[:n], base.weights[:n])
        expected = np.zeros(n, np.float32) if source is None else NTupleNetwork.load(source)[0].weights[:n]
        results[f"{start}: stage 2 is its source"] = same(net.weights[n : 2 * n], expected)
    path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    if not all(v for k, v in results.items() if k != "boards"):
        raise SystemExit("A check failed: fix the two-stage network before training")


# ---------------------------------------------------------------- training


def recipe(rate: float) -> dict[str, str]:
    return {"NTUPLE_NETWORK": "8x6", "NTUPLE_LEARNING_RATE": str(rate), "NTUPLE_TEMPORAL_COHERENCE": "true", "NTUPLE_DOWNGRADE_TILE": "16384",
            "NTUPLE_STAGE_TILE": "16384", "NTUPLE_START_POOL": str(TRAIN_POOL), "NTUPLE_LATE_START_FRACTION": "0", "NTUPLE_THREADS": "8",
            "NTUPLE_THREADS_AFTER": "0", "NTUPLE_CHECKPOINT_EVERY": "1000000000", "NTUPLE_LOG_EVERY": "1000"}


def latest(start: str, rate: float) -> Path:
    return Path("checkpoints/ntuple") / run_name(start, rate) / "latest.npz"


def games_in(path: Path) -> int:
    with np.load(path) as data:
        return CheckpointMetadata.model_validate_json(str(data["metadata"])).games_played


def train_to(start: str, rate: float, stop: int) -> None:
    resume = latest(start, rate) if latest(start, rate).exists() else STARTS / f"{start}.npz"
    if games_in(resume) >= stop:
        return
    free = shutil.disk_usage(".").free / 1e9
    if free < MIN_FREE_GB:
        raise SystemExit(f"Only {free:.1f} GB free: stopping before training {run_name(start, rate)}")
    command = [sys.executable, "-u", "scripts/train_ntuple.py", "--run", run_name(start, rate), "--resume", str(resume), "--games", str(stop)]
    with open(OUTPUT / f"train-{run_name(start, rate)}.log", "a") as log:
        subprocess.run(command, env=os.environ | recipe(rate), stdout=log, stderr=subprocess.STDOUT, check=True)


def strip_traces(start: str, rate: float) -> None:
    """Keep only the weights of an arm that won't be extended (a third of the size)."""
    path = latest(start, rate)
    network, metadata = NTupleNetwork.load(path)
    network.save(path, metadata)


# ---------------------------------------------------------------- evaluation


def seed_of(row: dict) -> int:
    return int(row["seed"])


def continuations(network: NTupleNetwork, rows: list[dict], n: int, depth: int) -> dict[str, np.ndarray]:
    """Play on from each position n times with the same spawns for every network: (score added, max tile, moves)."""
    jobs = [(r, j) for r in rows for j in range(n)]
    if depth == 1:
        # An empty search band is exactly greedy play, and also counts the moves
        games = list(POOL.map(lambda job: play_selective(network.weights, network.layout, board(job[0]), 1, 99, 0.0, 0.0, np.random.default_rng([seed_of(job[0]), job[1]]))[:3], jobs))
    else:
        games = list(POOL.map(lambda job: (*play_from(network.weights, network.layout, board(job[0]), depth, 0.0, np.random.default_rng([seed_of(job[0]), job[1]])), 0), jobs))
    fields = np.array(games, dtype=np.int64).reshape(len(rows), n, 3)
    return {"scores": fields[..., 0], "tiles": fields[..., 1], "moves": fields[..., 2]}


def evaluate(name: str, path: str | Path, depth3: bool) -> None:
    """Evaluate one network into evaluations/<name>.npz, adding depth 3 if asked and not done yet."""
    out_path = EVALUATIONS / f"{name}.npz"
    done = dict(np.load(out_path)) if out_path.exists() else {}
    if done and (not depth3 or "depth3_full_scores" in done):
        return
    network, metadata = NTupleNetwork.load(path)
    out = done or {"games": np.array(metadata.games_played), "moves_played": np.array(metadata.moves_played)}
    if not done:
        for (policy, played_by), n in CONTINUATIONS.items():
            for k, v in continuations(network, read_positions("dev", policy, played_by), n, 1).items():
                out[f"greedy_{policy}_{played_by}_{k}"] = v
        full = list(POOL.map(lambda s: play_seeded(network.weights, network.layout, 1, 0.0, np.random.default_rng(s)), FULL_GAME_SEEDS))
        out["greedy_full_scores"], out["greedy_full_tiles"] = np.array(full).T
        if network.stage_tile is not None:
            base, _ = NTupleNetwork.load(BASE)
            out["stage1_unchanged"] = np.array(same(network.weights[: tables(base)], base.weights[: tables(base)]))
    if depth3:
        for k, v in continuations(network, read_positions("dev", "greedy", "2m")[:DEPTH3_POSITIONS], 1, 3).items():
            out[f"depth3_greedy_2m_{k}"] = v
        full = list(POOL.map(lambda s: play_seeded(network.weights, network.layout, 3, 0.0, np.random.default_rng(s)), FULL_GAME_SEEDS))
        out["depth3_full_scores"], out["depth3_full_tiles"] = np.array(full).T
    EVALUATIONS.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out)


# ---------------------------------------------------------------- report


def interval(values: np.ndarray, rate: bool = False) -> str:
    mean, half = values.mean(), 1.96 * values.std(ddof=1) / np.sqrt(len(values))
    return f"{mean:+.1%} ± {half:.1%}" if rate else f"{mean:+,.0f} ± {half:,.0f}"


def per_position(d: dict, key: str) -> tuple[np.ndarray, np.ndarray]:
    """Each position's mean score added and 32768 rate."""
    return d[f"{key}_scores"].mean(axis=1), (d[f"{key}_tiles"] >= 32768).mean(axis=1)


def leader(start: str) -> float | None:
    """The start's best rate at 1M by dev greedy score added, or a lower rate within its 95% interval (short-run caution)."""
    done = {rate: dict(np.load(EVALUATIONS / f"{run_name(start, rate)}-{STOPS[-1]}.npz")) for rate in RATES[start] if (EVALUATIONS / f"{run_name(start, rate)}-{STOPS[-1]}.npz").exists()}
    if len(done) < len(RATES[start]):
        return None
    added = {rate: per_position(d, "greedy_greedy_2m")[0] for rate, d in done.items()}
    best = max(added, key=lambda r: added[r].mean())
    for rate in sorted(added):
        diff = added[rate] - added[best]
        if rate < best and diff.mean() + 1.96 * diff.std(ddof=1) / np.sqrt(len(diff)) >= 0:
            return rate
    return best


def report() -> None:
    control_path = EVALUATIONS / "control.npz"
    if not control_path.exists():
        return
    c = dict(np.load(control_path))
    c_added, c_made = per_position(c, "greedy_greedy_2m")
    print(f"Control (the 2M network alone): dev greedy continuations add {c_added.mean():,.0f} and make 32768 in {c_made.mean():.1%}; "
          f"full greedy games {c['greedy_full_scores'].mean():,.0f}" + (f", depth 3 {c['depth3_full_scores'].mean():,.0f}" if "depth3_full_scores" in c else ""))
    print("\n### Paired with the control (± 95%)\n")
    print("| Arm | Continuations | Moves per continuation | Score added | Makes 32768 | Depth-3 entries: added | 1.5M entries: added | Full greedy | ≥32768 | "
          "Depth-3 added | Depth-3 full | ≥32768 at depth 3 | Stage 1 unchanged |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for path in sorted(EVALUATIONS.glob("*.npz"), key=lambda p: (p.stem.rsplit("-", 1)[0], int(p.stem.rsplit("-", 1)[1]) if p.stem[-1].isdigit() else -1)):
        if path.stem == "control":
            continue
        d = dict(np.load(path))
        added, made = per_position(d, "greedy_greedy_2m")
        moves = d["greedy_greedy_2m_moves"].mean()
        cells = [f"{int(d['games']):,}", f"{moves:,.0f}", interval(added - c_added), interval(made - c_made, True)]
        for key in ("greedy_depth3_2m", "greedy_greedy_1.5m"):
            cells.append(interval(per_position(d, key)[0] - per_position(c, key)[0]))
        cells += [interval(d["greedy_full_scores"] - c["greedy_full_scores"]), f"{(d['greedy_full_tiles'] >= 32768).mean():.1%}"]
        if "depth3_full_scores" in d and "depth3_full_scores" in c:
            cells += [interval(d["depth3_greedy_2m_scores"][:, 0] - c["depth3_greedy_2m_scores"][:, 0]), interval(d["depth3_full_scores"] - c["depth3_full_scores"]),
                      f"{(d['depth3_full_tiles'] >= 32768).mean():.1%}"]
        else:
            cells += ["", "", ""]
        cells.append({True: "yes", False: "**NO**"}.get(bool(d["stage1_unchanged"]), "") if "stage1_unchanged" in d else "")
        print(f"| {path.stem} | " + " | ".join(cells) + " |")
    leaders = {start: leader(start) for start in RATES}
    print("\nLeading rate by start (dev greedy score added at 1M, preferring a lower rate within the interval): "
          + ", ".join(f"{s} {'—' if r is None else f'{r:g}'}" for s, r in leaders.items()))


# ---------------------------------------------------------------- the whole chain


def run() -> None:
    pools()
    build()
    check()
    evaluate("control", BASE, depth3=True)
    for start in ("500k", "scratch"):
        evaluate(f"{start}-start-0", STARTS / f"{start}.npz", depth3=False)
    for start, rates in RATES.items():
        for rate in rates:
            for stop in STOPS:
                train_to(start, rate, stop)
                evaluate(f"{run_name(start, rate)}-{stop}", latest(start, rate), depth3=stop in DEPTH3_AT)
                print(f"{time.strftime('%H:%M')} {run_name(start, rate)} evaluated at {stop:,}", flush=True)
        best = leader(start)
        for rate in rates:
            if rate != best:
                strip_traces(start, rate)
        print(f"{start}: leading rate {best:g}; the others keep weights only", flush=True)
    report()


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 6: a dedicated late-game network")
    parser.add_argument("step", choices=("pools", "build", "check", "run", "report"))
    args = parser.parse_args()
    {"pools": pools, "build": build, "check": check, "run": run, "report": report}[args.step]()


if __name__ == "__main__":
    main()
