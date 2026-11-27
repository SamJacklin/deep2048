"""A deep finish: does searching deeper only in the last stretch before a 32768 pay off at play time?

Usage:
    CHECKPOINT=checkpoints/ntuple/decision-8x6-tc-reset/games_2000000.npz
    uv run python scripts/deep_finish.py arrival "$CHECKPOINT"    # from the arrival test's positions (about 15 minutes)
    uv run python scripts/deep_finish.py games "$CHECKPOINT"      # whole games on seeds 10,000-10,999 (minutes; --depth3 adds about 20)
    uv run python scripts/deep_finish.py report

No training is involved. The arrival test's finish table found that, from boards within 64 of a 32768's tile sum,
pruned depth-8 search makes the 32768 far more often than greedy play or depth 3 (71% against 44% and 52%). Here the
ordinary player (greedy play or depth 3) plays everywhere except that last stretch: while the largest tile is 16384
and the tiles sum to within a margin of 32768, each move is searched to depth 8, pruning spawn branches less likely
than 0.02. Nearly full boards have few spawns, so that costs milliseconds a move, for a few dozen moves a game.

arrival: every policy plays on from the arrival test's well-placed positions (experiments/arrival-test/positions.csv)
until the game makes a 32768 or ends, with the same spawns, so each pairs with its ordinary player.

games: whole seeded games, on the seeds stage 6's control used (10,000-10,999), so the ordinary players' results
should reproduce experiments/late-network/evaluations/control.npz, which the report checks if that file exists.
"""

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from deep_2048.agents.ntuple import NTupleNetwork  # noqa: E402
from deep_2048.agents.ntuple import arrival as ar  # noqa: E402

OUTPUT = Path("experiments/deep-finish")
POSITIONS = Path("experiments/arrival-test/positions.csv")
CONTROL = Path("experiments/late-network/evaluations/control.npz")
POOL = ThreadPoolExecutor(max_workers=os.cpu_count())
TOP = ar.CHAIN_TOP
DEEP_DEPTH, DEEP_CUTOFF = 8, 0.02
FULL_GAME_SEEDS = range(10_000, 11_000)
# (name, the ordinary player's depth, the last stretch's margin or 0 for none, the name of the policy it pairs with)
ARRIVAL_POLICIES = (
    ("greedy", 1, 0, None),
    ("greedy, deep in the last 64", 1, 64, "greedy"),
    ("greedy, deep in the last 256", 1, 256, "greedy"),
    ("depth 3", 3, 0, None),
    ("depth 3, deep in the last 64", 3, 64, "depth 3"),
)
GAME_POLICIES = (("greedy", 1, 0, None), ("greedy, deep in the last 64", 1, 64, "greedy"), ("greedy, deep in the last 256", 1, 256, "greedy"))
GAME_POLICIES_DEPTH3 = (("depth 3", 3, 0, None), ("depth 3, deep in the last 64", 3, 64, "depth 3"))


def deep_settings(depth: int, margin: int) -> tuple[int, float]:
    """With no last stretch, the 'deep' search is the ordinary one, which makes the game exactly play_from's."""
    return (DEEP_DEPTH, DEEP_CUTOFF) if margin > 0 else (depth, 0.0)


def save(path: Path, done: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **done)


def arrival(network: NTupleNetwork, positions: int, depth3_positions: int, continuations: int) -> None:
    if not POSITIONS.exists():
        raise SystemExit(f"{POSITIONS} is missing: run scripts/arrival_test.py pools first")
    path = OUTPUT / "arrival.npz"
    done = dict(np.load(path)) if path.exists() else {}
    with open(POSITIONS) as f:
        rows = [r for r in csv.DictReader(f) if r["policy"] == "greedy" and int(r["placement"]) == ar.IN_ORDER][:positions]
    weights, layout = network.weights, network.layout
    for name, depth, margin, _ in ARRIVAL_POLICIES:
        if f"tiles {name}" in done:
            continue
        chosen = rows if depth == 1 else rows[:depth3_positions]
        deep_depth, deep_cutoff = deep_settings(depth, margin)
        jobs = [(r, j) for r in chosen for j in range(continuations)]
        started = time.perf_counter()
        results = list(POOL.map(lambda job: ar.play_with_finish(weights, layout, np.uint64(int(job[0]["board"], 16)), depth, deep_depth, deep_cutoff, TOP, margin, True,
                                                                np.random.default_rng([int(job[0]["seed"]), job[1]])), jobs))
        fields = np.array(results, dtype=np.int64).reshape(len(chosen), continuations, 4)
        done[f"tiles {name}"], done[f"moves {name}"], done[f"deep {name}"] = fields[..., 1], fields[..., 2], fields[..., 3]
        done[f"seconds {name}"] = np.array(time.perf_counter() - started)
        save(path, done)  # after each policy, so an interrupted run resumes at the next one
        print(f"arrival: {name}: makes {1 << (TOP + 1)} in {(fields[..., 1] > 1 << TOP).mean():.1%} of {len(jobs):,} continuations, in {done[f'seconds {name}']:.0f} s", flush=True)


def games(network: NTupleNetwork, depth3: bool) -> None:
    path = OUTPUT / "games.npz"
    done = dict(np.load(path)) if path.exists() else {}
    weights, layout = network.weights, network.layout
    for name, depth, margin, _ in GAME_POLICIES + (GAME_POLICIES_DEPTH3 if depth3 else ()):
        if f"scores {name}" in done:
            continue
        deep_depth, deep_cutoff = deep_settings(depth, margin)
        started = time.perf_counter()
        results = list(POOL.map(lambda s: ar.play_seeded_with_finish(weights, layout, depth, deep_depth, deep_cutoff, TOP, margin, np.random.default_rng(s)), FULL_GAME_SEEDS))
        fields = np.array(results, dtype=np.int64)
        done[f"scores {name}"], done[f"tiles {name}"], done[f"moves {name}"], done[f"deep {name}"] = fields.T
        done[f"seconds {name}"] = np.array(time.perf_counter() - started)
        save(path, done)
        print(f"games: {name}: {fields[:, 0].mean():,.0f} average, {1 << (TOP + 1)} in {(fields[:, 1] > 1 << TOP).mean():.1%}, in {done[f'seconds {name}']:.0f} s", flush=True)


def interval(values: np.ndarray, rate: bool = False) -> str:
    mean, half = values.mean(), 1.96 * values.std(ddof=1) / np.sqrt(len(values))
    return f"{mean:+.1%} ± {half:.1%}" if rate else f"{mean:+,.0f} ± {half:,.0f}"


def report() -> None:
    goal = 1 << (TOP + 1)
    lines = ["# Deep finish\n", f"The last stretch is searched to depth {DEEP_DEPTH}, pruning spawn branches less likely than {DEEP_CUTOFF}. Intervals are 95%.\n"]
    path = OUTPUT / "arrival.npz"
    if path.exists():
        d = np.load(path)
        lines += ["## From well-placed arrival positions\n", f"| Policy | Positions | Makes {goal} | vs its ordinary player (paired) | Moves searched deep, per continuation | Seconds |", "|---|---:|---:|---:|---:|---:|"]
        for name, _, _, pair in ARRIVAL_POLICIES:
            if f"tiles {name}" not in d:
                continue
            made = (d[f"tiles {name}"] > 1 << TOP).mean(axis=1)
            versus = "—"
            if pair is not None and f"tiles {pair}" in d:
                versus = interval(made - (d[f"tiles {pair}"] > 1 << TOP).mean(axis=1)[: len(made)], rate=True)
            deep = d[f"deep {name}"].mean() if pair is not None else 0
            lines.append(f"| {name} | {len(made):,} | {made.mean():.1%} | {versus} | {deep:.0f} | {float(d[f'seconds {name}']):.0f} |")
        lines.append("")
    path = OUTPUT / "games.npz"
    if path.exists():
        d = np.load(path)
        lines += [f"## Whole games (seeds {FULL_GAME_SEEDS[0]:,}-{FULL_GAME_SEEDS[-1]:,})\n",
                  f"| Policy | Average score | vs its ordinary player (paired) | ≥{goal // 2} | ≥{goal} | vs its ordinary player | Moves searched deep, per game | Seconds |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for name, _, _, pair in GAME_POLICIES + GAME_POLICIES_DEPTH3:
            if f"scores {name}" not in d:
                continue
            scores, tiles = d[f"scores {name}"], d[f"tiles {name}"]
            versus = rate_versus = "—"
            if pair is not None and f"scores {pair}" in d:
                versus = interval(scores - d[f"scores {pair}"])
                rate_versus = interval((tiles >= goal).astype(float) - (d[f"tiles {pair}"] >= goal), rate=True)
            deep = d[f"deep {name}"].mean() if pair is not None else 0
            lines.append(f"| {name} | {scores.mean():,.0f} | {versus} | {(tiles >= goal // 2).mean():.1%} | {(tiles >= goal).mean():.1%} | {rate_versus} | {deep:.1f} | {float(d[f'seconds {name}']):.0f} |")
        if CONTROL.exists():
            c = np.load(CONTROL)
            checks = [f"{name}: {'identical' if np.array_equal(d[f'scores {name}'], c[key]) else '**DIFFERENT**'}"
                      for name, key in (("greedy", "greedy_full_scores"), ("depth 3", "depth3_full_scores")) if f"scores {name}" in d and key in c]
            if checks:
                lines.append(f"\nThe ordinary players' games against stage 6's stored control ({CONTROL}): " + "; ".join(checks) + ".")
        lines.append("")
    text = "\n".join(lines)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "report.md").write_text(text + "\n")
    print(text)


def main() -> None:
    global OUTPUT, POSITIONS, TOP
    parser = argparse.ArgumentParser(description="Deeper search only in the last stretch before a 32768")
    parser.add_argument("command", choices=("arrival", "games", "report"))
    parser.add_argument("checkpoint", type=Path, nargs="?")
    parser.add_argument("--positions", type=int, default=1_000, help="arrival: well-placed positions for the greedy policies (default: 1,000)")
    parser.add_argument("--depth3-positions", type=int, default=300, help="arrival: positions for the depth-3 policies (default: 300)")
    parser.add_argument("--continuations", type=int, default=4, help="arrival: continuations per position (default: 4)")
    parser.add_argument("--depth3", action="store_true", help="games: also play depth 3 with and without the deep finish (about 20 minutes more)")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--positions-file", type=Path, default=POSITIONS, help="arrival: the arrival test's positions.csv")
    parser.add_argument("--top", type=int, default=TOP, help="exponent of the tile before the goal (default: 14, so the goal is a 32768)")
    args = parser.parse_args()
    OUTPUT, POSITIONS, TOP = args.output, args.positions_file, args.top
    if args.command != "report":
        if args.checkpoint is None:
            parser.error("arrival and games need a checkpoint")
        network, _ = NTupleNetwork.load(args.checkpoint)
        if args.command == "arrival":
            arrival(network, args.positions, args.depth3_positions, args.continuations)
        else:
            games(network, args.depth3)
    report()


if __name__ == "__main__":
    main()
