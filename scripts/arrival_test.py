"""The arrival test: when the last step to a 32768 fails, was the position already poor, or did a good one go wrong?

Usage:
    CHECKPOINT=checkpoints/ntuple/decision-8x6-tc-reset/games_2000000.npz
    uv run python scripts/arrival_test.py pools "$CHECKPOINT"    # arrival positions from seeded games (minutes)
    uv run python scripts/arrival_test.py trace "$CHECKPOINT"    # traced continuations from each (minutes)
    uv run python scripts/arrival_test.py fate "$CHECKPOINT"     # how late each game's outcome was settled (a minute or two)
    uv run python scripts/arrival_test.py finish "$CHECKPOINT"   # can deeper search finish the last stretch? (see its timings)
    uv run python scripts/arrival_test.py report                 # the tables, also written to report.md
    uv run python scripts/arrival_test.py run "$CHECKPOINT"      # pools, trace and report

A game arrives when it first holds 16384, 8192, 4096 and 2048 at once, with nothing larger. No training is involved.

pools: plays seeded games by greedy play and by depth-3 search until each has its share of arrival positions, one per
game, and records each board with where its chain sits. Seeds start at 60,000,000 (greedy) and 61,000,000 (depth 3),
clear of the evaluation seeds, the corpus, stage 6's pools and stage 8's splits.

trace: plays on from every position, by greedy play (--greedy continuations each) and, from the first --depth3-positions
of each pool, by depth-3 search (--depth3 each), until the game makes a 32768 or ends. Each continuation records
whether it converted, how far the chain ran, and the first move that broke the big tiles' formation: whether any legal
move would have kept it, and the board it was made from. Continuation j of a position uses the rng seeded with
[position seed, j] under either policy, so the two share spawns until their moves differ.

fate: replays greedy continuations from well-placed positions (the same games trace played), takes the boards they
decided from 1 to 200 moves before the stage ended, and plays on from each many times with fresh spawns. The share
that make the 32768 from a board is that board's chance, so the table shows how early a failed game was already lost.

finish: replays greedy continuations from well-placed positions (again the games trace played) to the first board whose
tiles sum to within --finish-margin of 32768, then plays on from each of those boards --finish-continuations times with
each policy in FINISH_POLICIES. Continuation k of a board uses the same spawns under every policy until their moves
differ, so each policy pairs with greedy play. It shows whether looking further ahead finishes the last stretch.

report: conversion by placement at arrival; whether positions differ in their chances (positions are ranked on half
their continuations and measured on the other half, so the spread isn't inflated by luck); how far failed
continuations got; and how they failed.
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
from deep_2048.bench import file_sha256  # noqa: E402
from deep_2048.game import bitboard as bb  # noqa: E402

POLICIES = {"greedy": (1, 60_000_000), "depth3": (3, 61_000_000)}
OUTPUT = Path("experiments/arrival-test")
POOL = ThreadPoolExecutor(max_workers=os.cpu_count())
COLUMNS = ("policy", "seed", "moves", "score", "board", "placement", "empty")


def positions_path(output: Path) -> Path:
    return output / "positions.csv"


def read_positions(output: Path) -> list[dict]:
    with open(positions_path(output)) as f:
        return list(csv.DictReader(f))


def board_of(row: dict) -> np.uint64:
    return np.uint64(int(row["board"], 16))


def show(board: np.uint64) -> str:
    cells = [bb.get_cell(np.uint64(board), i) for i in range(16)]
    return " / ".join(" ".join(f"{(1 << c) if c else 0:>5}" for c in cells[4 * r : 4 * r + 4]) for r in range(4))


# ---------------------------------------------------------------- pools


def pools(network: NTupleNetwork, checkpoint: Path, output: Path, counts: dict[str, int], top: int, length: int) -> None:
    if positions_path(output).exists():
        print(f"{positions_path(output)} exists")
        return
    output.mkdir(parents=True, exist_ok=True)
    weights, layout, floor = network.weights, network.layout, top - length + 1
    rows = []
    for policy, (depth, offset) in POLICIES.items():
        found, games, started = [], 0, time.perf_counter()
        while len(found) < counts[policy]:
            batch = 2_000 if depth == 1 else 200
            seeds = range(offset + games, offset + games + batch)
            results = POOL.map(lambda s: (s, ar.arrival_board(weights, layout, depth, 0.0, top, length, np.random.default_rng(s))), seeds)
            for seed, (board, moves, score) in results:
                if board == 0 or len(found) == counts[policy]:
                    continue
                found.append((policy, seed, moves, score, f"{int(board):016x}", int(ar.placement(np.uint64(board), floor, length)), int(bb.count_empty(np.uint64(board)))))
            games += batch
        rows += found
        print(f"{policy}: {len(found):,} arrival positions from {games:,} games (seeds from {offset:,}) in {time.perf_counter() - started:.0f} s", flush=True)
    with open(positions_path(output), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        writer.writerows(rows)
    (output / "source.txt").write_text(f"{checkpoint}\nsha256 {file_sha256(checkpoint)}\nchain top 2**{top}, length {length}\n")


# ---------------------------------------------------------------- trace


def trace(network: NTupleNetwork, output: Path, greedy: int, depth3: int, depth3_positions: int, top: int, length: int) -> None:
    rows = read_positions(output)
    weights, layout = network.weights, network.layout
    for name, depth, continuations in (("greedy", 1, greedy), ("depth3", 3, depth3)):
        path = output / f"traces-{name}.npz"
        if path.exists() or continuations == 0:
            print(f"{path} {'exists' if path.exists() else 'skipped'}")
            continue
        chosen = rows if depth == 1 else [r for policy in POLICIES for r in [x for x in rows if x["policy"] == policy][:depth3_positions]]
        jobs = [(p, j) for p in range(len(chosen)) for j in range(continuations)]
        started = time.perf_counter()

        def run(job: tuple[int, int]) -> tuple[np.ndarray, np.uint64, np.uint64]:
            p, j = job
            return ar.trace_from(weights, layout, board_of(chosen[p]), depth, 0.0, top, length, np.random.default_rng([int(chosen[p]["seed"]), j]))

        results = list(POOL.map(run, jobs))
        shape = (len(chosen), continuations)
        np.savez(
            path,
            fields=np.array([f for f, _, _ in results]).reshape(*shape, len(ar.FIELDS)),
            break_boards=np.array([b for _, b, _ in results], dtype=np.uint64).reshape(shape),
            last_boards=np.array([b for _, _, b in results], dtype=np.uint64).reshape(shape),
            entry=np.array([r["policy"] for r in chosen]),
            seed=np.array([int(r["seed"]) for r in chosen]),
            board=np.array([int(r["board"], 16) for r in chosen], dtype=np.uint64),
            placement=np.array([int(r["placement"]) for r in chosen]),
            empty=np.array([int(r["empty"]) for r in chosen]),
        )
        print(f"{name}: {len(jobs):,} continuations from {len(chosen):,} positions in {time.perf_counter() - started:.0f} s", flush=True)


# ---------------------------------------------------------------- fate

FATE_OFFSETS = (1, 5, 10, 25, 50, 100, 200)


def fate(network: NTupleNetwork, output: Path, positions: int, continuations: int, rollouts: int, top: int) -> None:
    path = output / "fate.npz"
    if path.exists():
        print(f"{path} exists")
        return
    chosen = [r for r in read_positions(output) if r["policy"] == "greedy" and int(r["placement"]) == ar.IN_ORDER][:positions]
    weights, layout = network.weights, network.layout
    offsets = np.array(FATE_OFFSETS, dtype=np.int64)
    started = time.perf_counter()
    games = [(p, j) for p in range(len(chosen)) for j in range(continuations)]
    tails = list(POOL.map(lambda g: ar.tail_boards(weights, layout, board_of(chosen[g[0]]), 1, 0.0, top, offsets, np.random.default_rng([int(chosen[g[0]]["seed"]), g[1]])), games))
    boards = np.array([b for _, _, b in tails], dtype=np.uint64)
    jobs = [(g, k) for g in range(len(games)) for k in range(len(offsets)) if boards[g, k] != 0]

    def chance(job: tuple[int, int]) -> float:
        g, k = job
        p, j = games[g]
        seed = int(chosen[p]["seed"])
        return float(np.mean([ar.converts(weights, layout, boards[g, k], 1, 0.0, top, np.random.default_rng([seed, j, int(offsets[k]), r])) for r in range(rollouts)]))

    chances = np.full(boards.shape, np.nan)
    for (g, k), value in zip(jobs, POOL.map(chance, jobs)):
        chances[g, k] = value
    np.savez(path, converted=np.array([c for c, _, _ in tails]), moves=np.array([m for _, m, _ in tails]), boards=boards, chances=chances, offsets=offsets,
             rollouts=np.array(rollouts))
    print(f"fate: {len(jobs):,} boards from {len(games):,} games, {rollouts} rollouts each, in {time.perf_counter() - started:.0f} s", flush=True)


def report_fate(output: Path, goal: int) -> list[str]:
    path = output / "fate.npz"
    if not path.exists():
        return []
    d = np.load(path)
    converted, chances = d["converted"] == 1, d["chances"]
    lines = [f"## How late was the outcome settled? ({len(converted):,} greedy games from well-placed positions, {int(d['rollouts'])} fresh continuations per board)\n",
             f"The chance of making the {goal} from the board a game decided from, some moves before its stage ended. {converted.mean():.1%} of these games made it.\n",
             f"| Moves before the end | Games that failed | Games that made the {goal} |\n|---:|---:|---:|"]
    for k, offset in enumerate(d["offsets"]):
        cells = []
        for chosen in (~converted, converted):
            values = chances[chosen, k]
            values = values[~np.isnan(values)]
            cells.append(f"{values.mean():.1%} ({len(values):,})" if len(values) else "n/a")
        lines.append(f"| {offset} | {cells[0]} | {cells[1]} |")
    lines.append("\nBoard counts are in brackets: stages shorter than the offset have no board there.\n")
    return lines


# ---------------------------------------------------------------- finish

# (name, depth, cutoff): exact search to the depth, or pruned where a spawn branch is less likely than the cutoff
# Exact depth 5 costs over 100 ms a move once a cascade opens the board up, so the deeper policies are pruned: they
# still search to their full depth along likely spawns, which on a nearly full board is almost every line.
FINISH_POLICIES = (("greedy", 1, 0.0), ("depth 3", 3, 0.0), ("depth 5, cutoff 0.02", 5, 0.02), ("depth 8, cutoff 0.02", 8, 0.02))


def finish(network: NTupleNetwork, output: Path, limit: int, continuations: int, margin: int, top: int) -> None:
    path = output / "finish.npz"
    weights, layout = network.weights, network.layout
    done = dict(np.load(path)) if path.exists() else {}
    if "boards" not in done:
        chosen = [r for r in read_positions(output) if r["policy"] == "greedy" and int(r["placement"]) == ar.IN_ORDER]
        started = time.perf_counter()
        games = [(r, j) for j in range(4) for r in chosen]
        found = list(POOL.map(lambda g: ar.finish_board(weights, layout, board_of(g[0]), 1, 0.0, top, margin, np.random.default_rng([int(g[0]["seed"]), g[1]])), games[: 4 * limit]))
        reached = [(int(b), int(g[0]["seed"]), g[1]) for b, g in zip(found, games) if b != 0]
        done = {"boards": np.array([b for b, _, _ in reached][:limit], dtype=np.uint64), "seeds": np.array([s for _, s, _ in reached][:limit]),
                "game": np.array([j for _, _, j in reached][:limit]), "reached": np.array(len(reached) / len(found)), "margin": np.array(margin)}
        np.savez(path, **done)
        print(f"finish: {len(done['boards']):,} boards within {margin} of the sum ({done['reached']:.1%} of {len(found):,} continuations got that far) in {time.perf_counter() - started:.0f} s", flush=True)
    boards, seeds, game = done["boards"], done["seeds"], done["game"]
    jobs = [(i, k) for i in range(len(boards)) for k in range(continuations)]
    for name, depth, cutoff in FINISH_POLICIES:
        if f"converted {name}" in done:
            continue
        started = time.perf_counter()
        results = list(POOL.map(lambda job: ar.converts(weights, layout, boards[job[0]], depth, cutoff, top, np.random.default_rng([int(seeds[job[0]]), int(game[job[0]]), 7, job[1]])), jobs))
        done[f"converted {name}"] = np.array(results).reshape(len(boards), continuations)
        done[f"seconds {name}"] = np.array(time.perf_counter() - started)
        np.savez(path, **done)  # saved after each policy, so an interrupted run resumes at the next one
        print(f"finish: {name}: {done[f'converted {name}'].mean():.1%} in {done[f'seconds {name}']:.0f} s", flush=True)


def report_finish(output: Path, goal: int) -> list[str]:
    path = output / "finish.npz"
    if not path.exists():
        return []
    d = np.load(path)
    names = [name for name, _, _ in FINISH_POLICIES if f"converted {name}" in d]
    if not names:
        return []
    base = d[f"converted {names[0]}"]
    lines = [f"## Can looking further ahead finish the last stretch? ({len(d['boards']):,} boards, {base.shape[1]} continuations each)\n",
             f"Boards are the first whose tiles sum to within {int(d['margin'])} of {goal}, from greedy continuations of well-placed positions; {float(d['reached']):.1%} of those "
             "continuations got that far. Every policy plays on from the same boards with the same spawns.\n",
             f"| Played on by | Makes {goal} | vs greedy play (paired) | Seconds for all continuations |\n|---|---:|---:|---:|"]
    for name in names:
        converted = d[f"converted {name}"]
        diff = (converted - base).mean(axis=1)
        versus = "—" if name == names[0] else f"{diff.mean():+.1%} ± {1.96 * diff.std(ddof=1) / np.sqrt(len(diff)):.1%}"
        lines.append(f"| {name} | {rate(converted)} | {versus} | {float(d[f'seconds {name}']):.0f} |")
    lines.append("")
    return lines


# ---------------------------------------------------------------- report


def rate(converted: np.ndarray) -> str:
    """A conversion rate with a 95% interval from the spread between positions (rows), which continuations share."""
    per_position = converted.mean(axis=1)
    half = 1.96 * per_position.std(ddof=1) / np.sqrt(len(per_position)) if len(per_position) > 1 else float("nan")
    return f"{per_position.mean():.1%} ± {half:.1%}"


def share(mask: np.ndarray, of: np.ndarray) -> str:
    return f"{mask[of].mean():.0%}" if of.any() else "n/a"


def report(output: Path) -> None:
    source = (output / "source.txt").read_text().splitlines()
    top, length = int(source[2].split("**")[1].split(",")[0]), int(source[2].rsplit(" ", 1)[1])
    goal = 1 << (top + 1)
    lines = [f"# Arrival test\n\nSource: {source[0]}. Arrival: the first board holding the {length} tiles from {1 << top:,} down, with nothing larger. "
             f"Intervals are 95%, from the spread between positions.\n"]
    for name in ("greedy", "depth3"):
        path = output / f"traces-{name}.npz"
        if not path.exists():
            continue
        data = np.load(path)
        fields, entry, place = data["fields"], data["entry"], data["placement"]
        converted = fields[:, :, ar.CONVERTED]
        continuations = converted.shape[1]
        lines.append(f"## Continued by {'greedy play' if name == 'greedy' else 'depth-3 search'} ({continuations} continuations per position)\n")

        lines.append(f"**Placement at arrival.**\n\n| Arrived from | Placement | Positions | Share | Makes {goal} |\n|---|---|---:|---:|---:|")
        for policy in POLICIES:
            of_policy = entry == policy
            if not of_policy.any():
                continue
            lines.append(f"| {policy} games | all | {of_policy.sum():,} | 100% | {rate(converted[of_policy])} |")
            for code, label in ar.PLACEMENT_NAMES.items():
                chosen = of_policy & (place == code)
                if chosen.sum() > 1:
                    lines.append(f"| {policy} games | {label} | {chosen.sum():,} | {chosen.sum() / of_policy.sum():.0%} | {rate(converted[chosen])} |")

        good = place == ar.IN_ORDER
        if good.sum() >= 10 and continuations >= 4:
            lines.append("\n**Do well-placed positions differ in their chances?** Positions in order along one edge, ranked by conversion on "
                         "their even-numbered continuations and measured on the odd-numbered ones.\n\n| Fifth of positions, by the ranking half | Ranking half | Measured half |\n|---|---:|---:|")
            ranking, measured = converted[good][:, 0::2].mean(axis=1), converted[good][:, 1::2].mean(axis=1)
            order = np.argsort(ranking, kind="stable")
            for k, part in enumerate(np.array_split(order, 5)):
                lines.append(f"| {('lowest', 'second', 'middle', 'fourth', 'highest')[k]} | {ranking[part].mean():.1%} | {measured[part].mean():.1%} |")
            per_position = converted[good].mean(axis=1)
            p = per_position.mean()
            luck = p * (1 - p) / continuations
            between = max(per_position.var(ddof=1) - luck, 0.0)
            lines.append(f"\nSpread of positions' true chances, after removing the luck of {continuations} continuations: standard deviation {np.sqrt(between):.1%} around {p:.1%}.")
            lines.append(f"\n| Empty cells at arrival | Positions | Makes {goal} |\n|---:|---:|---:|")
            for empty in sorted(set(data["empty"][good])):
                chosen = good & (data["empty"] == empty)
                if chosen.sum() > 1:
                    lines.append(f"| {empty} | {chosen.sum():,} | {rate(converted[chosen])} |")

        f = fields[good].reshape(-1, len(ar.FIELDS))
        if len(f):
            failed = f[:, ar.CONVERTED] == 0
            broke = f[:, ar.BREAK_MOVE] >= 0
            forced = f[:, ar.BREAK_FORCED] == 1
            lines.append(f"\n**How far continuations from well-placed positions got** ({len(f):,} continuations).\n\n| Chain ran to at least | Share |\n|---|---:|")
            for reach in range(length + 1, length + 5):
                tile = 1 << (top - reach + 1)
                lines.append(f"| {tile} | {((f[:, ar.REACH] >= reach) | (f[:, ar.CONVERTED] == 1)).mean():.1%} |")
            lines.append(f"| a {goal} | {(~failed).mean():.1%} |")

            lines.append(f"\n**How they failed** ({failed.sum():,} failed continuations from well-placed positions).\n\n| | Share of failures |\n|---|---:|")
            lines.append(f"| The formation broke before the end | {share(broke, failed)} |")
            lines.append(f"| ...by a forced move (no legal move kept it) | {share(broke & forced, failed)} |")
            lines.append(f"| ...by a chosen move (another legal move kept it) | {share(broke & ~forced, failed)} |")
            lines.append(f"| Ended with the formation never broken | {share(~broke, failed)} |")
            lines.append(f"\nOf continuations whose formation broke, {share(f[:, ar.CONVERTED] == 1, broke)} still made the {goal}; of those where it never broke, {share(f[:, ar.CONVERTED] == 1, ~broke)} did. "
                         "So a break doesn't by itself decide the outcome: games that convert break formation too, on the way into their last cascade.")
            chose = np.flatnonzero(broke & ~forced & failed)
            if len(chose):
                sample = chose[np.random.default_rng(0).choice(len(chose), min(len(chose), 5_000), replace=False)]
                all_boards = data["break_boards"][good].reshape(-1)
                floor = top - length + 1
                at_once = np.array([ar.safest_keeping_move(np.uint64(all_boards[i]), floor, length) for i in sample])
                lines.append(f"\n**Was a chosen break a mistake?** Not necessarily. In {np.mean(at_once > 0.5):.0%} of chosen breaks in failures, even the safest move that kept the formation "
                             f"would more likely than not have ended the game on the spot (no legal move after the spawn); in {np.mean(at_once == 0):.0%} it couldn't have.")
            b = f[broke & failed]
            if len(b):
                lines.append(f"\nAt the break, in failures (medians): {int(np.median(b[:, ar.BREAK_EMPTY]))} empty cells on the board the move was made from; "
                             f"{int(np.median(b[:, ar.MOVES_AFTER_BREAK]))} moves from the break to the end; {int(np.median(b[:, ar.BREAK_MOVE]))} moves from arrival to the break; "
                             f"{int(np.median(b[:, ar.TIGHT_RUN]))} decisions in a row before it with at most one safe move.")
                lines.append("\n| Chain ran to, at the break | Share of breaks in failures | Forced |\n|---|---:|---:|")
                for reach in range(length, length + 4):
                    at = b[:, ar.BREAK_REACH] == reach
                    if at.any():
                        lines.append(f"| {1 << (top - reach + 1)} | {at.mean():.0%} | {(b[at, ar.BREAK_FORCED] == 1).mean():.0%} |")
                at = b[:, ar.BREAK_REACH] >= length + 4
                if at.any():
                    lines.append(f"| {1 << (top - length - 3)} or beyond | {at.mean():.0%} | {(b[at, ar.BREAK_FORCED] == 1).mean():.0%} |")
                boards = data["break_boards"][good].reshape(-1)[broke & failed]
                lines.append("\nFive boards a breaking move was made from (first continuations of the first positions):\n")
                lines += [f"- `{show(board)}`" for board in boards[:5]]
        lines.append("")
    lines += report_fate(output, goal)
    lines += report_finish(output, goal)
    text = "\n".join(lines)
    (output / "report.md").write_text(text + "\n")
    print(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Where the last step to a 32768 goes wrong")
    parser.add_argument("command", choices=("pools", "trace", "fate", "finish", "report", "run"))
    parser.add_argument("checkpoint", type=Path, nargs="?")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--greedy-positions", type=int, default=2_000, help="arrival positions from greedy games (default: 2,000)")
    parser.add_argument("--depth3-entry-positions", type=int, default=500, help="arrival positions from depth-3 games (default: 500)")
    parser.add_argument("--greedy", type=int, default=32, help="greedy continuations per position (default: 32)")
    parser.add_argument("--depth3", type=int, default=8, help="depth-3 continuations per position (default: 8; 0 skips them)")
    parser.add_argument("--depth3-positions", type=int, default=500, help="positions per pool that get depth-3 continuations (default: 500)")
    parser.add_argument("--fate-positions", type=int, default=500, help="fate: well-placed greedy-entry positions replayed (default: 500)")
    parser.add_argument("--fate-continuations", type=int, default=4, help="fate: games replayed per position (default: 4)")
    parser.add_argument("--fate-rollouts", type=int, default=32, help="fate: fresh continuations per board (default: 32)")
    parser.add_argument("--finish-boards", type=int, default=1_000, help="finish: boards near the end to play on from (default: 1,000)")
    parser.add_argument("--finish-continuations", type=int, default=8, help="finish: continuations per board and policy (default: 8)")
    parser.add_argument("--finish-margin", type=int, default=64, help="finish: how close to the full sum a board must be (default: 64)")
    parser.add_argument("--top", type=int, default=ar.CHAIN_TOP, help="exponent of the chain's largest tile (default: 14, a 16384)")
    parser.add_argument("--length", type=int, default=ar.CHAIN_LENGTH, help="tiles in the chain (default: 4)")
    args = parser.parse_args()
    if args.command != "report":
        if args.checkpoint is None:
            parser.error("pools, trace, fate, finish and run need a checkpoint")
        network, _ = NTupleNetwork.load(args.checkpoint)
        if args.command in ("pools", "run"):
            pools(network, args.checkpoint, args.output, {"greedy": args.greedy_positions, "depth3": args.depth3_entry_positions}, args.top, args.length)
        if args.command in ("trace", "run"):
            trace(network, args.output, args.greedy, args.depth3, args.depth3_positions, args.top, args.length)
        if args.command == "fate":
            fate(network, args.output, args.fate_positions, args.fate_continuations, args.fate_rollouts, args.top)
        if args.command == "finish":
            finish(network, args.output, args.finish_boards, args.finish_continuations, args.finish_margin, args.top)
    if args.command in ("report", "run", "fate", "finish"):
        report(args.output)


if __name__ == "__main__":
    main()
