"""The search-gain map: at which points of a game would searching ahead play better than the network does greedily?

Usage:
    CHECKPOINT=checkpoints/ntuple/decision-8x6-tc-reset/games_2000000.npz
    uv run python scripts/search_gain.py run "$CHECKPOINT"       # pools, play and report (resumable)
    uv run python scripts/search_gain.py pools "$CHECKPOINT"     # positions at each point, from greedy games (a minute)
    uv run python scripts/search_gain.py play "$CHECKPOINT"      # continuations from them (the long part; resumable)
    uv run python scripts/search_gain.py report                  # the tables, also written to report.md

A point is a game's largest tile with the tile sizes just below it that it holds (deep_2048/agents/ntuple/points.py):
"16384 + 8192 + 2048" is a game 10240 of the way through its 16384 stage. No training is involved: the network is the
one given, frozen.

pools: plays --games seeded greedy games (seeds from 70,000,000, clear of the evaluation seeds and every earlier
pool) and records the first board of each point whose largest tile is from 2**--low to 2**--high (default the 8192
and 16384 stages). Each point keeps the boards of its first --positions games to reach it. It also records which
points every game reached, so the report can say how many games each point matters to.

play: from every kept board, --greedy continuations by greedy play and --search continuations searching --depth ahead
(pruning spawn branches less likely than --cutoff), each to the end of the game. Continuation j of a board uses the rng
seeded with [game seed, point key, j] under both policies, so greedy and search continuation 0 share spawns until their
moves differ and the gap between them is paired. Positions are played in turns across the points, so a run stopped
early still covers every point, and results are saved every --save-every continuations: rerunning resumes.

report: for each point, how many games reach it; from its boards, the final score, the share making the next largest
tile and the share making 32768, by greedy play and by search; and search's gain, paired, with a 95% interval.
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from deep_2048.agents.ntuple import NTupleNetwork  # noqa: E402
from deep_2048.agents.ntuple import points as pt  # noqa: E402
from deep_2048.bench import file_sha256  # noqa: E402

OUTPUT = Path("experiments/search-gain")
SEED_OFFSET = 70_000_000
GOAL = 32768


def interval(values: np.ndarray) -> float:
    return float(1.96 * values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else float("nan")


# ---------------------------------------------------------------- pools


def pools(network: NTupleNetwork, checkpoint: Path, output: Path, games: int, positions: int, low: int, high: int, detail: int, threads: int) -> None:
    path = output / "positions.npz"
    if path.exists():
        print(f"{path} exists: delete it to draw new positions")
        return
    output.mkdir(parents=True, exist_ok=True)
    weights, layout = network.weights, network.layout
    started = time.perf_counter()
    seeds = np.arange(SEED_OFFSET, SEED_OFFSET + games)
    with ThreadPoolExecutor(max_workers=threads) as pool:
        results = list(pool.map(lambda s: pt.point_boards(weights, layout, low, high, detail, np.random.default_rng(int(s))), seeds))
    finals = np.array([r[5] for r in results])
    tops = np.array([r[6] for r in results])
    print(f"{games:,} greedy games in {time.perf_counter() - started:.0f} s: mean score {finals.mean():,.0f}, 32768 in {(tops >= GOAL).mean():.1%}", flush=True)
    rows = {"key": [], "board": [], "moves": [], "score": [], "seed": []}
    reached: dict[int, int] = {}
    kept: dict[int, int] = {}
    for seed, (keys, boards, moves, scores, n, _, _) in zip(seeds, results):
        for k in range(n):
            key = int(keys[k])
            reached[key] = reached.get(key, 0) + 1
            if kept.get(key, 0) < positions:
                kept[key] = kept.get(key, 0) + 1
                for name, value in (("key", key), ("board", int(boards[k])), ("moves", int(moves[k])), ("score", int(scores[k])), ("seed", int(seed))):
                    rows[name].append(value)
    order = sorted(reached)
    np.savez(path, key=np.array(rows["key"], dtype=np.int64), board=np.array(rows["board"], dtype=np.uint64), moves=np.array(rows["moves"], dtype=np.int64),
             score=np.array(rows["score"], dtype=np.int64), seed=np.array(rows["seed"], dtype=np.int64), point=np.array(order, dtype=np.int64),
             reached=np.array([reached[k] for k in order], dtype=np.int64), games=games, final_score=finals, final_top=tops, detail=detail)
    (output / "source.txt").write_text(f"{checkpoint}\nsha256 {file_sha256(checkpoint)}\nlow 2**{low}, high 2**{high}, detail {detail}\n")
    print(f"{len(rows['key']):,} positions at {len(order)} points, written to {path}", flush=True)


# ---------------------------------------------------------------- play


def turn_order(keys: np.ndarray) -> np.ndarray:
    """Position indices taking turns across the points: every point's first, then every point's second, and so on."""
    rank = np.zeros(keys.size, dtype=np.int64)
    counts: dict[int, int] = {}
    for i, key in enumerate(keys):
        rank[i] = counts.get(int(key), 0)
        counts[int(key)] = rank[i] + 1
    return np.lexsort((keys, rank))


def play_name(depth: int, cutoff: float, goal: int | None) -> str:
    return f"play-d{depth}-c{cutoff:g}" + (f"-g{goal}" if goal else "") + ".npz"


def play(network: NTupleNetwork, output: Path, greedy: int, search: int, depth: int, cutoff: float, threads: int, save_every: int, goal: int | None = None) -> None:
    positions = np.load(output / "positions.npz")
    keys, boards, seeds = positions["key"], positions["board"], positions["seed"]
    path = output / play_name(depth, cutoff, goal)
    n = keys.size
    if path.exists():
        saved = dict(np.load(path))
        if saved["greedy_score"].shape != (n, greedy) or saved["search_score"].shape != (n, search):
            raise SystemExit(f"{path} was made with other continuation counts: rerun with those or delete it")
        results = saved
        results.setdefault("greedy_value", np.zeros((n, greedy)))
        results.setdefault("search_value", np.zeros((n, search)))
    else:
        results = {"greedy_score": np.zeros((n, greedy), dtype=np.int64), "greedy_top": np.zeros((n, greedy), dtype=np.int64),
                   "search_score": np.zeros((n, search), dtype=np.int64), "search_top": np.zeros((n, search), dtype=np.int64),
                   "search_moves": np.zeros((n, search), dtype=np.int64),
                   "greedy_value": np.zeros((n, greedy)), "search_value": np.zeros((n, search))}
    weights, layout = network.weights, network.layout
    order = turn_order(keys)
    tasks = [(i, "greedy", j) for j in range(greedy) for i in order if results["greedy_top"][i, j] == 0]
    tasks += [(i, "search", j) for j in range(search) for i in order if results["search_top"][i, j] == 0]
    if not tasks:
        print(f"{path} is complete")
        return

    goal_exponent = goal.bit_length() - 1 if goal else 0

    def one(task: tuple[int, str, int]) -> tuple[int, str, int, int, int, int, float]:
        i, policy, j = task
        rng = np.random.default_rng([int(seeds[i]), int(keys[i]), j])
        d, c = (1, 0.0) if policy == "greedy" else (depth, cutoff)
        if goal:
            gained, top, moves, value = pt.play_to(weights, layout, np.uint64(boards[i]), d, c, rng, goal_exponent)
        else:
            (gained, top, moves), value = pt.play_on(weights, layout, np.uint64(boards[i]), d, c, rng), 0.0
        return i, policy, j, gained, top, moves, value

    with ThreadPoolExecutor(max_workers=threads) as pool:
        for policy in ("greedy", "search"):
            todo = [task for task in tasks if task[1] == policy]
            if not todo:
                continue
            what = ("greedy" if policy == "greedy" else f"searching {depth} ahead (cutoff {cutoff:g})") + (f", each to {goal} or the end" if goal else ", each to the end")
            print(f"{len(todo):,} continuations {what} on {threads} threads", flush=True)
            started = time.perf_counter()
            for chunk_start in range(0, len(todo), save_every):
                for i, kind, j, gained, top, moves, value in pool.map(one, todo[chunk_start : chunk_start + save_every]):
                    results[f"{kind}_score"][i, j] = gained
                    results[f"{kind}_top"][i, j] = top
                    results[f"{kind}_value"][i, j] = value
                    if kind == "search":
                        results["search_moves"][i, j] = moves
                tmp = path.with_suffix(".tmp.npz")
                np.savez(tmp, **results)
                os.replace(tmp, path)
                finished = min(chunk_start + save_every, len(todo))
                elapsed = time.perf_counter() - started
                left = elapsed / finished * (len(todo) - finished)
                print(f"  {finished:,} of {len(todo):,} in {elapsed / 60:.1f} min, about {left / 60:.0f} min left", flush=True)
    print(f"Written to {path}")


# ---------------------------------------------------------------- report


def report_goal(output: Path, depth: int, cutoff: float, goal: int) -> None:
    """The map aimed at a tile: from each point, how often the game reaches `goal`, in what shape, by greedy and by search."""
    positions = np.load(output / "positions.npz")
    results = np.load(output / play_name(depth, cutoff, goal))
    detail, games = int(positions["detail"]), int(positions["games"])
    keys, start_scores = positions["key"], positions["score"]
    reached = dict(zip(positions["point"].tolist(), positions["reached"].tolist()))
    g_done, s_done = results["greedy_top"] > 0, results["search_top"] > 0
    finals = positions["final_top"]
    lines = [f"# Search-gain map aimed at {goal}: greedy play against depth-{depth} search (cutoff {cutoff:g})", "",
             f"Positions from {games:,} seeded greedy games of `{(output / 'source.txt').read_text().splitlines()[0]}`: mean score "
             f"{positions['final_score'].mean():,.0f}, {goal} in {(finals >= goal).mean():.1%} of them. From each point's boards (the first time a game "
             f"reached it) play goes on until the board holds {goal} or the game ends. Arrival value is the network's own estimate of the score still "
             f"to come from the board it reached {goal} on: the shape it arrived in. Expected final score is the score at arrival plus that value, or "
             f"the final score if the game ended first. Gains are search minus greedy on continuation 0, which share spawns until their moves differ, "
             f"with 95% intervals.", ""]
    header = (f"| Point | Through stage | Games reaching | Boards | {goal}, greedy | {goal}, search | {goal} gain (points) | Arrival value, greedy "
              "| Arrival value, search | Expected final, greedy | Expected final, search | Expected final gain |")
    rule = "|" + "|".join(["---"] + ["---:"] * 11) + "|"
    ranking = []
    for top in sorted({int(k) // pt.KEY_STRIDE for k in reached}):
        lines += [f"## The {1 << top} stage", "", header, rule]
        for key in sorted((k for k in reached if k // pt.KEY_STRIDE == top), key=lambda k: pt.stage_progress(k, detail)):
            rows = np.flatnonzero(keys == key)
            both = rows[g_done[rows, 0] & s_done[rows, 0]]
            name, share = pt.point_name(key, detail), reached[key] / games
            if both.size < 2:
                lines.append(f"| {name} | {pt.stage_progress(key, detail):.0%} | {share:.1%} | {rows.size} | " + " | ".join(["–"] * 8) + " |")
                continue
            g_rows = rows[g_done[rows].all(axis=1)]
            s_rows = rows[s_done[rows].all(axis=1)]
            g_arrive = results["greedy_top"][g_rows] >= goal
            s_arrive = results["search_top"][s_rows] >= goal
            g_final = start_scores[g_rows, None] + results["greedy_score"][g_rows] + results["greedy_value"][g_rows]
            s_final = start_scores[s_rows, None] + results["search_score"][s_rows] + results["search_value"][s_rows]
            arrive_gap = 100.0 * ((results["search_top"][both, 0] >= goal).astype(float) - (results["greedy_top"][both, 0] >= goal))
            final_gap = (results["search_score"][both, 0] + results["search_value"][both, 0]) - (results["greedy_score"][both, 0] + results["greedy_value"][both, 0])
            g_value = results["greedy_value"][g_rows][g_arrive]
            s_value = results["search_value"][s_rows][s_arrive]
            lines.append(f"| {name} | {pt.stage_progress(key, detail):.0%} | {share:.1%} | {both.size} | {g_arrive.mean():.1%} | {s_arrive.mean():.1%} | "
                         f"{arrive_gap.mean():+.1f} ± {interval(arrive_gap):.1f} | {g_value.mean() if g_value.size else float('nan'):,.0f} | "
                         f"{s_value.mean() if s_value.size else float('nan'):,.0f} | {g_final.mean():,.0f} | {s_final.mean():,.0f} | "
                         f"{final_gap.mean():+,.0f} ± {interval(final_gap):,.0f} |")
            ranking.append((share * arrive_gap.mean(), share * final_gap.mean(), name, share, arrive_gap.mean(), final_gap.mean()))
        lines.append("")
    lines += ["## Where teaching could pay most", "",
              f"Search's gain at a point times the share of games that reach it. Points along one game overlap, so these don't add up; they rank the points.", "",
              f"| Point | Games reaching | {goal} gain | Per game: {goal} points | Expected final gain | Per game: expected final |", "|---|---:|---:|---:|---:|---:|"]
    for per_game_goal, per_game_final, name, share, goal_gain, final_gain in sorted(ranking, key=lambda r: r[1], reverse=True)[:12]:
        lines.append(f"| {name} | {share:.1%} | {goal_gain:+.1f} | {per_game_goal:+.2f} | {final_gain:+,.0f} | {per_game_final:+,.0f} |")
    text = "\n".join(lines) + "\n"
    (output / f"report-g{goal}.md").write_text(text)
    print(text)


def report(output: Path, depth: int, cutoff: float) -> None:
    positions = np.load(output / "positions.npz")
    play_path = output / f"play-d{depth}-c{cutoff:g}.npz"
    results = np.load(play_path)
    detail, games = int(positions["detail"]), int(positions["games"])
    keys, start_scores = positions["key"], positions["score"]
    reached = dict(zip(positions["point"].tolist(), positions["reached"].tolist()))
    g_done = results["greedy_top"] > 0
    s_done = results["search_top"] > 0
    lines = [f"# Search-gain map: greedy play against depth-{depth} search (cutoff {cutoff:g})", "",
             f"Positions from {games:,} seeded greedy games of `{(output / 'source.txt').read_text().splitlines()[0]}`: mean score "
             f"{positions['final_score'].mean():,.0f}, 32768 in {(positions['final_top'] >= GOAL).mean():.1%}. Each point's boards are the first time "
             f"a game reached it. Final scores include the score at the point. Gains are search minus greedy on continuation 0, which share spawns "
             f"until their moves differ, with 95% intervals.", ""]
    header = ("| Point | Through stage | Games reaching | Boards | Greedy score | Search score | Score gain | Next tile, greedy | Next tile, search "
              "| 32768, greedy | 32768, search | 32768 gain (points) |")
    rule = "|" + "|".join(["---"] + ["---:"] * 11) + "|"
    ranking = []
    for top in sorted({int(k) // pt.KEY_STRIDE for k in reached}):
        lines += [f"## The {1 << top} stage", "", header, rule]
        stage = sorted((k for k in reached if k // pt.KEY_STRIDE == top), key=lambda k: pt.stage_progress(k, detail))
        for key in stage:
            rows = np.flatnonzero(keys == key)
            both = rows[g_done[rows, 0] & s_done[rows, 0]]
            name, share = pt.point_name(key, detail), reached[key] / games
            if both.size < 2:
                lines.append(f"| {name} | {pt.stage_progress(key, detail):.0%} | {share:.1%} | {rows.size} | " + " | ".join(["–"] * 8) + " |")
                continue
            g_rows = rows[g_done[rows].all(axis=1)] if g_done.shape[1] else rows
            g_final = (start_scores[g_rows, None] + results["greedy_score"][g_rows]).ravel()
            s_rows = rows[s_done[rows].all(axis=1)]
            s_final = (start_scores[s_rows, None] + results["search_score"][s_rows]).ravel()
            g_next = (results["greedy_top"][g_rows] > 1 << top).ravel()
            s_next = (results["search_top"][s_rows] > 1 << top).ravel()
            g_goal = (results["greedy_top"][g_rows] >= GOAL).ravel()
            s_goal = (results["search_top"][s_rows] >= GOAL).ravel()
            score_gap = (results["search_score"][both, 0] - results["greedy_score"][both, 0]).astype(float)
            goal_gap = 100.0 * ((results["search_top"][both, 0] >= GOAL).astype(float) - (results["greedy_top"][both, 0] >= GOAL))
            lines.append(f"| {name} | {pt.stage_progress(key, detail):.0%} | {share:.1%} | {both.size} | {g_final.mean():,.0f} | {s_final.mean():,.0f} | "
                         f"{score_gap.mean():+,.0f} ± {interval(score_gap):,.0f} | {g_next.mean():.1%} | {s_next.mean():.1%} | {g_goal.mean():.1%} | {s_goal.mean():.1%} | "
                         f"{goal_gap.mean():+.1f} ± {interval(goal_gap):.1f} |")
            ranking.append((share * goal_gap.mean(), share * score_gap.mean(), name, share, goal_gap.mean(), score_gap.mean()))
        lines.append("")
    lines += ["## Where teaching could pay most", "",
              "Search's gain at a point times the share of games that reach it: what the average game would gain if the network learned to play "
              "like search from that point on. Points along one game overlap, so these don't add up; they rank the points.", "",
              "| Point | Games reaching | 32768 gain | Per game: 32768 points | Score gain | Per game: score |", "|---|---:|---:|---:|---:|---:|"]
    for per_game_goal, per_game_score, name, share, goal_gain, score_gain in sorted(ranking, reverse=True)[:12]:
        lines.append(f"| {name} | {share:.1%} | {goal_gain:+.1f} | {per_game_goal:+.2f} | {score_gain:+,.0f} | {per_game_score:+,.0f} |")
    text = "\n".join(lines) + "\n"
    (output / "report.md").write_text(text)
    print(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Where search would play better than the network")
    parser.add_argument("command", choices=("pools", "play", "report", "run"))
    parser.add_argument("checkpoint", type=Path, nargs="?")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--games", type=int, default=2_000, help="pools: seeded greedy games to draw positions from (default 2,000)")
    parser.add_argument("--positions", type=int, default=200, help="pools: boards kept at each point (default 200)")
    parser.add_argument("--low", type=int, default=13, help="pools: the smallest largest tile's exponent (default 13, the 8192 stage)")
    parser.add_argument("--high", type=int, default=14, help="pools: the largest largest tile's exponent (default 14, the 16384 stage)")
    parser.add_argument("--detail", type=int, default=3, help="pools: tile sizes below the largest that name a point (default 3: eighths of a stage)")
    parser.add_argument("--greedy", type=int, default=4, help="play: greedy continuations per board (default 4)")
    parser.add_argument("--search", type=int, default=1, help="play: search continuations per board (default 1)")
    parser.add_argument("--depth", type=int, default=3, help="play: search depth (default 3)")
    parser.add_argument("--cutoff", type=float, default=0.0, help="play: prune spawn branches less likely than this (default 0, none)")
    parser.add_argument("--threads", type=int, default=os.cpu_count(), help="threads (default: every core)")
    parser.add_argument("--save-every", type=int, default=200, help="play: continuations between saves (default 200)")
    parser.add_argument("--goal", type=int, help="play and report: stop each continuation once the board holds this tile, and report how often and in what shape it got there")
    args = parser.parse_args()
    if args.command != "report":
        if args.checkpoint is None:
            parser.error("pools, play and run need a checkpoint")
        network, _ = NTupleNetwork.load(args.checkpoint)
        if args.command in ("pools", "run"):
            pools(network, args.checkpoint, args.output, args.games, args.positions, args.low, args.high, args.detail, args.threads)
        if args.command in ("play", "run"):
            play(network, args.output, args.greedy, args.search, args.depth, args.cutoff, args.threads, args.save_every, args.goal)
    if args.command in ("report", "run"):
        if args.goal:
            report_goal(args.output, args.depth, args.cutoff, args.goal)
        else:
            report(args.output, args.depth, args.cutoff)


if __name__ == "__main__":
    main()
