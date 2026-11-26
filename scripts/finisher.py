"""The finisher: teach greedy play the last stretch before a 32768 from the network's own deeper search.

Usage:
    uv run python scripts/finisher.py pools     # boards at the window's edge, for training and for measuring (minutes)
    uv run python scripts/finisher.py check     # an untrained finisher plays exactly as the base
    uv run python scripts/finisher.py train     # train in segments, measuring after each
    uv run python scripts/finisher.py report
    uv run python scripts/finisher.py evaluate  # the trained finisher against the base: whole games and the test set

See deep_2048/agents/ntuple/finisher.py for the method. The base is round 1's TC-reset network at 2M games, frozen.
The window is the last --margin of tile sum before a 32768 (default 64), while the largest tile is 16384. Deep search
in the last 256 made the 32768 no more often than in the last 64 (scripts/deep_finish.py), so 64 is where the teacher
has something to teach.

pools: training boards come from fresh greedy games of the base (seeds from 70,000,000): each game that arrives
(first holds 16384, 8192, 4096 and 2048) is played on by greedy play several times, and each continuation's first
board inside the window is kept. Measuring boards are made the same way from the arrival test's greedy-entry positions
(experiments/arrival-test/positions.csv, seeds from 60,000,000), which training never sees: positions with an even
seed are the dev set, odd the test set, so no game's boards are in both.

train: every attempt starts from a training board and plays to the end of the window, learning at each decision from
the teacher: the frozen base searching --depth moves ahead, pruning spawn branches less likely than --cutoff. A share
of attempts (--student-share) follow the finisher's own greedy move, the rest the teacher's. With --centred it is
taught only how the teacher ranks the moves at each choice. After every --every attempts the finisher is saved as a
patch on the base and measured on the dev set, paired with the base on the same boards and spawns:
- how often its one-move choice is the teacher's, along the teacher's play, and what the teacher thinks the
  difference costs (the first sign of whether greedy play is learning the search's choices);
- from the window's edge: greedy play with the finisher, against greedy play and the teacher's search with the base;
- the same at depth 3;
- from arrival: greedy play with the finisher from well-placed arrival positions, against the base's.
A run can be stopped and resumed: the patch and its temporal coherence sums are saved after each segment.
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

from deep_2048.agents.ntuple import NTupleNetwork  # noqa: E402
from deep_2048.agents.ntuple import arrival as ar  # noqa: E402
from deep_2048.agents.ntuple.diagnostics import greedy_values  # noqa: E402
from deep_2048.agents.ntuple.finisher import FinisherNetwork, fit_to_teacher, train_attempts  # noqa: E402
from deep_2048.agents.ntuple.network import REDUNDANT_TUPLES, SHAPE_TUPLES  # noqa: E402
from deep_2048.agents.ntuple.search import _play_from, play_seeded  # noqa: E402

BASE = Path("checkpoints/ntuple/decision-8x6-tc-reset/games_2000000.npz")
ARRIVAL_POSITIONS = Path("experiments/arrival-test/positions.csv")
OUTPUT = Path("experiments/finisher")
CHECKPOINTS = Path("checkpoints/ntuple/finisher")
POOL = ThreadPoolExecutor(max_workers=os.cpu_count())
TOP = ar.CHAIN_TOP
LENGTH = ar.CHAIN_LENGTH  # pools: a game arrives when it holds this many tiles from 2**TOP down (--length)
TRAIN_SEEDS = 70_000_000
TRAIN_BOARDS = 50_000
TRAIN_CONTINUATIONS = 8
MEASURE_CONTINUATIONS = 2
EVAL_CONTINUATIONS = 4
ARRIVAL_POSITIONS_MEASURED = 500
FULL_GAME_SEEDS = range(10_000, 11_000)
FIT_BOARDS = 400
# Tuples the window's stage reads besides the base's own, from zero weights (--extra)
EXTRA = {
    "none": (),
    "redundant": REDUNDANT_TUPLES,  # Jaśkowski's small tuples: straight 4s, 2x2 squares, straight 3s
    "shapes": SHAPE_TUPLES,  # four 6-cell shapes the 8x6 set doesn't cover
}
HISTORY = ("attempts", "moves", "seconds", "teacher_made", "student_made", "patch_weights", "agrees", "regret", "value_error", "ranking_error", "dev_greedy", "dev_greedy_vs_base", "dev_depth3", "dev_depth3_vs_base", "arrival_greedy", "arrival_greedy_vs_base")


def pool_dir(margin: int) -> Path:
    return OUTPUT / f"pool-{margin}"


def read_arrivals() -> list[dict]:
    if not ARRIVAL_POSITIONS.exists():
        raise SystemExit(f"{ARRIVAL_POSITIONS} is missing: run scripts/arrival_test.py pools first")
    with open(ARRIVAL_POSITIONS) as f:
        return [r for r in csv.DictReader(f) if r["policy"] == "greedy"]


# ---------------------------------------------------------------- pools


def pools(margin: int, boards_wanted: int, natural: bool) -> None:
    """Collect the training boards (and, the first time, the measuring boards).

    The first training pool (train.npy) kept each board once. Games funnel into a limited set of boards at the
    window's edge, the clean ones many times over, so keeping each once left the pool mostly one-off, harder boards:
    its teacher made the 32768 in 42% against 71% from the measuring boards. `natural` keeps every board as often as
    games reach it (train-natural.npy).
    """
    out = pool_dir(margin)
    target = out / ("train-natural.npy" if natural else "train.npy")
    if target.exists() and (out / "measure.npz").exists():
        print(f"{target} exists")
        return
    out.mkdir(parents=True, exist_ok=True)
    base, _ = NTupleNetwork.load(BASE)
    weights, layout, length = base.weights, base.layout, LENGTH

    def window_boards(starts: list[tuple[int, int]], continuations: int) -> list[tuple[int, int, int]]:
        """Each arrival board's greedy continuations to the window: (board, seed, continuation) for those that got there."""
        jobs = [(board, seed, j) for board, seed in starts for j in range(continuations)]
        found = POOL.map(lambda job: ar.finish_board(weights, layout, np.uint64(job[0]), 1, 0.0, TOP, margin, np.random.default_rng([job[1], job[2]])), jobs)
        return [(int(b), seed, j) for b, (_, seed, j) in zip(found, jobs) if b != 0]

    started, games, train, seen = time.perf_counter(), 0, [], set()
    while not target.exists() and len(train) < boards_wanted:
        seeds = range(TRAIN_SEEDS + games, TRAIN_SEEDS + games + 2_000)
        arrivals = POOL.map(lambda s: (s, ar.arrival_board(weights, layout, 1, 0.0, TOP, length, np.random.default_rng(s))[0]), seeds)
        starts = [(int(b), s) for s, b in arrivals if b != 0]
        for b, _, _ in window_boards(starts, TRAIN_CONTINUATIONS):
            if natural or b not in seen:
                seen.add(b)
                train.append(b)
        games += 2_000
        print(f"pools: {len(train):,} training boards ({len(seen):,} distinct) from {games:,} games", flush=True)
    if not target.exists():
        np.save(target, np.array(train[:boards_wanted], dtype=np.uint64))
    if (out / "measure.npz").exists():
        print(f"pools: {target} written in {time.perf_counter() - started:.0f} s; the measuring boards are unchanged")
        return

    rows = read_arrivals()
    measured = window_boards([(int(r["board"], 16), int(r["seed"])) for r in rows], MEASURE_CONTINUATIONS)
    np.savez(out / "measure.npz", boards=np.array([b for b, _, _ in measured], dtype=np.uint64), seeds=np.array([s for _, s, _ in measured]),
             continuation=np.array([j for _, _, j in measured]))
    dev = sum(1 for _, s, _ in measured if s % 2 == 0)
    print(f"pools: {boards_wanted:,} training boards and {len(measured):,} measuring boards ({dev:,} dev, {len(measured) - dev:,} test) within {margin} of the sum, in {time.perf_counter() - started:.0f} s")


def load_pools(margin: int, split: str, natural: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """(training boards, the split's measuring boards, their seeds, the split's well-placed arrival positions)."""
    out = pool_dir(margin)
    train_file = out / ("train-natural.npy" if natural else "train.npy")
    if not natural and not train_file.exists():  # callers wanting only the measuring boards take whichever pool was made
        train_file = out / "train-natural.npy"
    if not train_file.exists() or not (out / "measure.npz").exists():
        raise SystemExit(f"{train_file} is missing: run scripts/finisher.py pools{' --natural' if natural else ''} first")
    m = np.load(out / "measure.npz")
    chosen = (m["seeds"] % 2 == 0) if split == "dev" else (m["seeds"] % 2 == 1)
    seeds = m["seeds"][chosen] * 10 + m["continuation"][chosen]  # one spawn stream per board
    arrivals = [r for r in read_arrivals() if int(r["placement"]) == ar.IN_ORDER and (int(r["seed"]) % 2 == 0) == (split == "dev")][:ARRIVAL_POSITIONS_MEASURED]
    return np.load(train_file), m["boards"][chosen], seeds, arrivals


# ---------------------------------------------------------------- measuring


def from_window(weights: np.ndarray, layout, boards: np.ndarray, seeds: np.ndarray, depth: int, cutoff: float) -> np.ndarray:
    """Whether each continuation from each window board makes the 32768: every caller shares spawns, board by board."""
    jobs = [(i, k) for i in range(len(boards)) for k in range(EVAL_CONTINUATIONS)]
    made = POOL.map(lambda job: ar.converts(weights, layout, boards[job[0]], depth, cutoff, TOP, np.random.default_rng([int(seeds[job[0]]), 11, job[1]])), jobs)
    return np.array(list(made)).reshape(len(boards), EVAL_CONTINUATIONS)


def from_arrival(weights: np.ndarray, layout, arrivals: list[dict]) -> np.ndarray:
    """Whether each greedy continuation from each arrival position makes the 32768."""
    jobs = [(r, j) for r in arrivals for j in range(EVAL_CONTINUATIONS)]
    made = POOL.map(lambda job: ar.converts(weights, layout, np.uint64(int(job[0]["board"], 16)), 1, 0.0, TOP, np.random.default_rng([int(job[0]["seed"]), job[1]])), jobs)
    return np.array(list(made)).reshape(len(arrivals), EVAL_CONTINUATIONS)


def baseline(margin: int, split: str, depth: int, cutoff: float) -> dict[str, np.ndarray]:
    """The base alone on the split: greedy play, depth 3 and the teacher's search from the window, and greedy play from arrival."""
    path = pool_dir(margin) / f"baseline-{split}.npz"
    done = dict(np.load(path)) if path.exists() else {}
    teacher = f"teacher depth {depth} cutoff {cutoff:g}"
    wanted = {"greedy": (1, 0.0), "depth3": (3, 0.0), teacher: (depth, cutoff)}
    if all(k in done for k in wanted) and "arrival" in done:
        return done | {"teacher": done[teacher]}
    base, _ = NTupleNetwork.load(BASE)
    _, boards, seeds, arrivals = load_pools(margin, split)
    for key, (d, c) in wanted.items():
        if key not in done:
            started = time.perf_counter()
            done[key] = from_window(base.weights, base.layout, boards, seeds, d, c)
            print(f"baseline ({split}): {key} from the window makes the 32768 in {done[key].mean():.1%}, in {time.perf_counter() - started:.0f} s", flush=True)
            np.savez(path, **done)
    if "arrival" not in done:
        done["arrival"] = from_arrival(base.weights, base.layout, arrivals)
        print(f"baseline ({split}): greedy play from arrival makes the 32768 in {done['arrival'].mean():.1%}", flush=True)
        np.savez(path, **done)
    return done | {"teacher": done[teacher]}


def fit(finisher: FinisherNetwork, boards: np.ndarray, depth: int, cutoff: float) -> dict[str, float]:
    """How closely the finisher's one-move choices match the teacher's along the teacher's play from the first
    FIT_BOARDS measuring boards (the same boards and spawns every time)."""
    parts = list(POOL.map(lambda k: fit_to_teacher(finisher.weights, finisher.layout, finisher.base_layout, boards[:FIT_BOARDS][k::8], depth, cutoff, TOP, finisher.low, finisher.first, 500 + k), range(8)))
    choices = max(sum(p[0] for p in parts), 1)
    return {"agrees": round(sum(p[1] for p in parts) / choices, 4), "regret": round(sum(p[4] * p[0] for p in parts) / choices, 1),
            "value_error": round(sum(p[2] * p[0] for p in parts) / choices, 1), "ranking_error": round(sum(p[3] * p[0] for p in parts) / choices, 1)}


def versus(ours: np.ndarray, theirs: np.ndarray) -> tuple[float, float]:
    """The paired difference in conversion, position by position: (mean, 95% half-width)."""
    diff = ours.mean(axis=1) - theirs.mean(axis=1)
    return float(diff.mean()), float(1.96 * diff.std(ddof=1) / np.sqrt(len(diff)))


# ---------------------------------------------------------------- check


def check(margin: int, extra: str = "none", area: bool = False) -> None:
    base, _ = NTupleNetwork.load(BASE)
    finisher = FinisherNetwork.from_base(BASE, TOP, margin, EXTRA[extra], area)
    train, boards, seeds, _ = load_pools(margin, "dev")
    sample = [np.uint64(b) for b in train[:: max(1, len(train) // 2_000)]]
    same_values = all(np.array_equal(greedy_values(base.weights, base.layout, b), greedy_values(finisher.weights, finisher.layout, b), equal_nan=True) for b in sample)

    def play(net, depth: int, n: int) -> list:
        return list(POOL.map(lambda i: _play_from(net.weights, net.layout, boards[i], depth, 0.0, np.random.default_rng(int(seeds[i]))), range(n)))

    results = {
        "greedy values on training boards": same_values,
        "greedy continuations from the window (200)": play(base, 1, min(200, len(boards))) == play(finisher, 1, min(200, len(boards))),
        "depth-3 continuations from the window (20)": play(base, 3, min(20, len(boards))) == play(finisher, 3, min(20, len(boards))),
        "whole greedy games (100)": list(POOL.map(lambda s: play_seeded(base.weights, base.layout, 1, 0.0, np.random.default_rng(s)), FULL_GAME_SEEDS[:100]))
        == list(POOL.map(lambda s: play_seeded(finisher.weights, finisher.layout, 1, 0.0, np.random.default_rng(s)), FULL_GAME_SEEDS[:100])),
        "first stage is the base": finisher.base_unchanged(base),
    }
    (pool_dir(margin) / "checks.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    if not all(results.values()):
        raise SystemExit("A check failed: an untrained finisher must play exactly as the base")


# ---------------------------------------------------------------- train


def sums_path(name: str) -> Path:
    return CHECKPOINTS / f"{name}.sums.npz"


def train(name: str, margin: int, rate: float, depth: int, cutoff: float, attempts: int, every: int, threads: int, student_share: float, centred: bool, natural: bool,
          extra: str = "none", mistakes: bool = False, rollout: tuple[int, int, float] = (0, 32, 2.0), area: bool = False, td: bool = False) -> None:
    if depth < 2:
        raise SystemExit("The teacher must search at least 2 moves ahead: at depth 1 its values are the network's own")
    out = OUTPUT / name
    out.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINTS / f"{name}.npz"
    settings = {"margin": margin, "rate": rate, "depth": depth, "cutoff": cutoff, "threads": threads, "student_share": student_share, "centred": centred} | ({"natural": True} if natural else {}) | ({"extra": extra} if extra != "none" else {}) | ({"mistakes": True} if mistakes else {}) | ({"rollout": list(rollout)} if rollout[0] > 0 else {}) | ({"area": True} if area else {}) | ({"td": True} if td else {})
    starts, boards, seeds, arrivals = load_pools(margin, "dev", natural)
    base_results = baseline(margin, "dev", depth, cutoff)
    if path.exists():
        finisher = FinisherNetwork.load(path)
        if finisher.progress.get("settings") != settings:
            raise SystemExit(f"{path} was trained with {finisher.progress.get('settings')}, not {settings}: use another --name")
        errors, abs_errors = np.zeros(finisher.weights.size - finisher.first), np.zeros(finisher.weights.size - finisher.first)
        with np.load(sums_path(name)) as sums:
            errors[sums["indices"]], abs_errors[sums["indices"]] = sums["errors"], sums["abs_errors"]
        print(f"Resuming {name} at {finisher.progress['attempts']:,} attempts")
    else:
        finisher = FinisherNetwork.from_base(BASE, TOP, margin, EXTRA[extra], area)
        finisher.progress = {"attempts": 0, "moves": 0, "settings": settings}
        errors, abs_errors = np.zeros(finisher.weights.size - finisher.first), np.zeros(finisher.weights.size - finisher.first)
        (out / "settings.json").write_text(json.dumps(settings | {"base": str(BASE), "base_sha256": finisher.base_sha256, "top": TOP}, indent=2) + "\n")
    base, _ = NTupleNetwork.load(BASE)
    weights, layout = finisher.weights, finisher.layout
    # By TD, a work-area table learns alone: the stage under it stays the base's copy, so nothing the base knows is unsettled
    learn_from = 2 * finisher.first if td and area else 0
    # Compile before timing starts
    train_attempts(weights, errors, abs_errors, layout, finisher.base_layout, starts, 0, depth, cutoff, rate, TOP, finisher.low, finisher.first, student_share, centred, 0, mistakes, *rollout, td, learn_from)
    if finisher.progress["attempts"] == 0:
        start = fit(finisher, boards, depth, cutoff)
        (out / "fit-untrained.json").write_text(json.dumps(start) + "\n")
        print(f"Before training, along the teacher's play from {min(FIT_BOARDS, len(boards))} dev boards: the base's one-move choice is the teacher's {start['agrees']:.1%} of the time; "
              f"the teacher values its own move {start['regret']:,.0f} above the base's on average", flush=True)

    while finisher.progress["attempts"] < attempts:
        segment = min(every, attempts - finisher.progress["attempts"])
        shares = [len(part) for part in np.array_split(np.arange(segment), threads)]
        seed = 1_000 * (finisher.progress["attempts"] // every + 1)
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=threads) as workers:
            parts = list(workers.map(lambda k: train_attempts(weights, errors, abs_errors, layout, finisher.base_layout, starts, shares[k], depth, cutoff, rate, TOP, finisher.low,
                                                              finisher.first, student_share, centred, seed + k, mistakes, *rollout, td, learn_from), range(threads)))
        seconds = time.perf_counter() - started
        teacher_led, teacher_made, student_led, student_made, moves = (sum(p[i] for p in parts) for i in range(5))
        teacher_rate, student_rate = teacher_made / max(teacher_led, 1), student_made / max(student_led, 1)
        finisher.progress["attempts"] += segment
        finisher.progress["moves"] += moves
        if not finisher.base_unchanged(base):
            raise SystemExit("Training changed the base's weights: stopping")
        finisher.save(path)
        touched = np.flatnonzero(abs_errors)
        np.savez(sums_path(name), indices=touched.astype(np.uint32), errors=errors[touched], abs_errors=abs_errors[touched])

        greedy = from_window(weights, layout, boards, seeds, 1, 0.0)
        depth3 = from_window(weights, layout, boards, seeds, 3, 0.0)
        arrival = from_arrival(weights, layout, arrivals)
        done = finisher.progress["attempts"]
        np.savez(out / f"eval-{done}.npz", greedy=greedy, depth3=depth3, arrival=arrival)
        fitted = fit(finisher, boards, depth, cutoff)
        row = {"attempts": done, "moves": finisher.progress["moves"], "seconds": round(seconds), "teacher_made": round(teacher_rate, 4), "student_made": round(student_rate, 4), "patch_weights": len(finisher.patch()[0]), **fitted,
               "dev_greedy": round(float(greedy.mean()), 4), "dev_greedy_vs_base": round(versus(greedy, base_results["greedy"])[0], 4),
               "dev_depth3": round(float(depth3.mean()), 4), "dev_depth3_vs_base": round(versus(depth3, base_results["depth3"])[0], 4),
               "arrival_greedy": round(float(arrival.mean()), 4), "arrival_greedy_vs_base": round(versus(arrival, base_results["arrival"])[0], 4)}
        new = not (out / "history.csv").exists()
        with open(out / "history.csv", "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=HISTORY)
            if new:
                writer.writeheader()
            writer.writerow(row)
        print(f"{time.strftime('%H:%M')} {name} at {done:,} attempts ({seconds:.0f} s, {moves / max(seconds, 1e-9) / threads:,.0f} moves/s a thread): in training the teacher made {teacher_rate:.1%} and the finisher {student_rate:.1%}; picks the teacher's move {fitted['agrees']:.1%}, regret {fitted['regret']:,.0f}; "
              f"dev from the window: greedy {greedy.mean():.1%} ({row['dev_greedy_vs_base']:+.1%} vs base), depth 3 {depth3.mean():.1%} ({row['dev_depth3_vs_base']:+.1%}); "
              f"from arrival: greedy {arrival.mean():.1%} ({row['arrival_greedy_vs_base']:+.1%})", flush=True)
    report(name, margin, depth, cutoff)


# ---------------------------------------------------------------- evaluate


def paired(ours: np.ndarray, theirs: np.ndarray) -> tuple[float, float]:
    """The paired difference game by game: (mean, 95% half-width)."""
    diff = np.asarray(ours, dtype=np.float64) - np.asarray(theirs, dtype=np.float64)
    return float(diff.mean()), float(1.96 * diff.std(ddof=1) / np.sqrt(len(diff)))


def evaluate(name: str, margin: int, depth: int, cutoff: float, depth3: bool) -> None:
    """The trained finisher against the base: whole games on the usual evaluation seeds, and the held-out test boards."""
    out = OUTPUT / name
    finisher = FinisherNetwork.load(CHECKPOINTS / f"{name}.npz")
    base, _ = NTupleNetwork.load(BASE)
    if not finisher.base_unchanged(base):
        raise SystemExit("The finisher's first stage is not the base")
    path = out / "evaluation.npz"
    done = dict(np.load(path)) if path.exists() and int(np.load(path)["attempts"]) == finisher.progress["attempts"] else {"attempts": np.array(finisher.progress["attempts"])}

    def games(key: str, net, search: int) -> np.ndarray:
        if key not in done:
            started = time.perf_counter()
            done[key] = np.array(list(POOL.map(lambda s: play_seeded(net.weights, net.layout, search, 0.0, np.random.default_rng(s)), FULL_GAME_SEEDS)))
            np.savez(path, **done)
            print(f"{key}: {len(FULL_GAME_SEEDS):,} whole games, mean score {done[key][:, 0].mean():,.0f}, in {time.perf_counter() - started:.0f} s", flush=True)
        return done[key]

    lines = [f"# Finisher {name}: evaluation at {finisher.progress['attempts']:,} attempts\n",
             f"Whole games on seeds {FULL_GAME_SEEDS[0]:,} to {FULL_GAME_SEEDS[-1]:,}, the base and the finisher on the same seeds. Differences are paired game by game, ± 95%.\n",
             "| Play | Network | Mean score | vs base | ≥16384 | ≥32768 | vs base |", "|---|---|---:|---:|---:|---:|---:|"]
    for label, search in (("Greedy", 1),) + ((("Depth 3", 3),) if depth3 else ()):
        b, f = games(f"base depth {search}", base, search), games(f"finisher depth {search}", finisher, search)
        lines.append(f"| {label} | Base | {b[:, 0].mean():,.0f} | | {(b[:, 1] >= 16384).mean():.1%} | {(b[:, 1] >= 32768).mean():.1%} | |")
        score, big = paired(f[:, 0], b[:, 0]), paired(f[:, 1] >= 32768, b[:, 1] >= 32768)
        lines.append(f"| {label} | Finisher | {f[:, 0].mean():,.0f} | {score[0]:+,.0f} ± {score[1]:,.0f} | {(f[:, 1] >= 16384).mean():.1%} | {(f[:, 1] >= 32768).mean():.1%} | {pm(big)} |")

    _, boards, seeds, arrivals = load_pools(margin, "test")
    b = baseline(margin, "test", depth, cutoff)
    if "test greedy" not in done:
        done["test greedy"] = from_window(finisher.weights, finisher.layout, boards, seeds, 1, 0.0)
        done["test arrival"] = from_arrival(finisher.weights, finisher.layout, arrivals)
        np.savez(path, **done)
    lines += ["", f"The held-out test set, which no setting was chosen on: {len(boards):,} window boards and {len(arrivals):,} well-placed arrival positions, {EVAL_CONTINUATIONS} continuations each.\n",
              "| From | Base greedy | Finisher greedy | Difference | Base's deep search |", "|---|---:|---:|---:|---:|",
              f"| The window's edge | {b['greedy'].mean():.1%} | {done['test greedy'].mean():.1%} | {pm(versus(done['test greedy'], b['greedy']))} | {b['teacher'].mean():.1%} |",
              f"| Arrival | {b['arrival'].mean():.1%} | {done['test arrival'].mean():.1%} | {pm(versus(done['test arrival'], b['arrival']))} | |"]
    text = "\n".join(lines)
    (out / "evaluation.md").write_text(text + "\n")
    print(text)


# ---------------------------------------------------------------- report


def pm(pair: tuple[float, float]) -> str:
    return f"{pair[0]:+.1%} ± {pair[1]:.1%}"


def report(name: str, margin: int, depth: int, cutoff: float) -> None:
    out = OUTPUT / name
    evaluations = sorted(out.glob("eval-*.npz"), key=lambda p: int(p.stem.split("-")[1]))
    if not evaluations:
        print(f"No evaluations in {out}")
        return
    b = baseline(margin, "dev", depth, cutoff)
    history = {int(r["attempts"]): r for r in csv.DictReader(open(out / "history.csv"))}
    lines = [f"# Finisher {name}\n", f"Window: the last {margin} of tile sum before a 32768. Teacher: depth {depth}, cutoff {cutoff:g}. Dev set: {len(b['greedy']):,} window boards and "
             f"{len(b['arrival']):,} well-placed arrival positions, {EVAL_CONTINUATIONS} continuations each. Differences are paired with the base, ± 95%.\n",
             "The base alone, from the window: greedy play makes the 32768 in " + f"{b['greedy'].mean():.1%}, depth 3 in {b['depth3'].mean():.1%}, the teacher's search in {b['teacher'].mean():.1%}. "
             f"From arrival, greedy play makes it in {b['arrival'].mean():.1%}.\n",
             "| Attempts | Window, greedy | vs base greedy | Share of the teacher's gain | Window, depth 3 | vs base depth 3 | Arrival, greedy | vs base | Picks the teacher's move | Regret | Value error | Ranking error | Patch weights | In training: teacher / finisher |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    if (out / "fit-untrained.json").exists():
        u = json.loads((out / "fit-untrained.json").read_text())
        lines.append(f"| 0 | {b['greedy'].mean():.1%} | | | {b['depth3'].mean():.1%} | | {b['arrival'].mean():.1%} | | {u['agrees']:.1%} | {u['regret']:,.0f} | {u['value_error']:,.0f} | {u['ranking_error']:,.0f} | 0 | |")
    gap = b["teacher"].mean() - b["greedy"].mean()
    for path in evaluations:
        attempts = int(path.stem.split("-")[1])
        d = np.load(path)
        h = history.get(attempts, {})
        gain = versus(d["greedy"], b["greedy"])
        lines.append(f"| {attempts:,} | {d['greedy'].mean():.1%} | {pm(gain)} | {gain[0] / gap:.0%} | {d['depth3'].mean():.1%} | {pm(versus(d['depth3'], b['depth3']))} | "
                     f"{d['arrival'].mean():.1%} | {pm(versus(d['arrival'], b['arrival']))} | {float(h.get('agrees', 0)):.1%} | {float(h.get('regret', 0)):,.0f} | {float(h.get('value_error', 0)):,.0f} | {float(h.get('ranking_error', 0)):,.0f} | {int(h.get('patch_weights', 0)):,} | {float(h.get('teacher_made', 0)):.1%} / {float(h.get('student_made', 0)):.1%} |")
    lines += ["", "Picks the teacher's move: how often the finisher's one-move choice is the teacher's, along the teacher's play from dev boards. Regret: how much more the teacher values its own move than the "
              "finisher's, on average. Value error: the mean difference between the two's values of a move. Ranking error: the same once each choice's mean difference is taken off."]
    text = "\n".join(lines)
    (out / "report.md").write_text(text + "\n")
    print(text)


def main() -> None:
    global BASE, ARRIVAL_POSITIONS, OUTPUT, CHECKPOINTS, TOP, LENGTH
    parser = argparse.ArgumentParser(description="The finisher: the last stretch before a 32768, taught by deeper search")
    parser.add_argument("command", choices=("pools", "check", "train", "report", "evaluate"))
    parser.add_argument("--name", default=None, help="the run's name (default: from its settings)")
    parser.add_argument("--margin", type=int, default=64, help="the window: this much tile sum before a 32768 (default: 64)")
    parser.add_argument("--rate", type=float, default=0.01, help="base learning rate per weight (default: 0.01)")
    parser.add_argument("--depth", type=int, default=8, help="the teacher's search depth (default: 8)")
    parser.add_argument("--cutoff", type=float, default=0.02, help="the teacher prunes spawn branches less likely than this (default: 0.02)")
    parser.add_argument("--attempts", type=int, default=200_000, help="window attempts to train to (default: 200,000)")
    parser.add_argument("--every", type=int, default=20_000, help="attempts between saves and measurements (default: 20,000)")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--student-share", type=float, default=0.5, help="share of attempts that follow the finisher's own greedy move (default: 0.5)")
    parser.add_argument("--centred", action="store_true", help="teach only how the teacher ranks the moves at each choice, not its level of value")
    parser.add_argument("--mistakes", action="store_true", help="train: learn only where the finisher's move isn't the teacher's, closing the gap between the two moves")
    parser.add_argument("--rollout", type=int, default=0, help="train: the teacher is the base's own greedy rollouts of this many moves, not its search (default: 0, search)")
    parser.add_argument("--rollouts", type=int, default=32, help="train: rollouts per move for --rollout (default: 32)")
    parser.add_argument("--z", type=float, default=2.0, help="train: --rollout overrides the base's greedy move only by more than this many standard errors (default: 2)")
    parser.add_argument("--natural", action="store_true", help="pools, train: training boards as often as games reach them, not each once")
    parser.add_argument("--depth3", action="store_true", help="evaluate: also play the whole games searching 3 moves ahead (much slower)")
    parser.add_argument("--train-boards", type=int, default=TRAIN_BOARDS, help="pools: training boards to collect (default: 50,000)")
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--arrival-positions", type=Path, default=ARRIVAL_POSITIONS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--checkpoints", type=Path, default=CHECKPOINTS)
    parser.add_argument("--extra", choices=tuple(EXTRA), default="none", help="check, train: tuples the window reads besides the base's, from zero (default: none)")
    parser.add_argument("--area", action="store_true", help="check, train: the window also reads a work-area table, the half of the board without the large tiles as one pattern")
    parser.add_argument("--td", action="store_true", help="train: no teacher; the finisher learns by TD(0) from its own greedy play (with --area, only the table learns)")
    parser.add_argument("--length", type=int, default=LENGTH, help="pools: training games arrive at this many tiles from 2**top down (default: 4; 1 is the first 2**top)")
    parser.add_argument("--top", type=int, default=TOP, help="exponent of the window's largest tile (default: 14, a 16384)")
    args = parser.parse_args()
    BASE, ARRIVAL_POSITIONS, OUTPUT, CHECKPOINTS, TOP = args.base, args.arrival_positions, args.output, args.checkpoints, args.top
    LENGTH = args.length
    name = args.name or f"m{args.margin}-d{args.depth}-r{args.rate:g}" + ("-centred" if args.centred else "") + ("-natural" if args.natural else "") + ("" if args.extra == "none" else f"-{args.extra}") + ("-mistakes" if args.mistakes else "") + (f"-rollout{args.rollout}" if args.rollout else "") + ("-area" if args.area else "") + ("-td" if args.td else "")
    if args.command == "pools":
        pools(args.margin, args.train_boards, args.natural)
    elif args.command == "check":
        check(args.margin, args.extra, args.area)
    elif args.command == "train":
        train(name, args.margin, args.rate, args.depth, args.cutoff, args.attempts, args.every, args.threads, args.student_share, args.centred, args.natural, args.extra, args.mistakes, (args.rollout, args.rollouts, args.z), args.area, args.td)
    elif args.command == "evaluate":
        evaluate(name, args.margin, args.depth, args.cutoff, args.depth3)
    else:
        report(name, args.margin, args.depth, args.cutoff)


if __name__ == "__main__":
    main()
