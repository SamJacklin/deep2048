"""Teach stage 1 its own search's play across the 4096 to 8192 stretch (deep_2048/agents/ntuple/teach.py), in its own weights.

Two ways to teach (--mode):
- rank (the default): only the move choice. Where the network's greedy move differs from the search's, the search's
  move is raised over the network's by --step of the gap (plus --margin), moving only the weights the two afterstates
  don't share.
- value: every legal move's value pulled towards the search's. This collapsed play at every rate tried (a weight
  shared by a deadly move and its safe neighbour is pulled both ways for ever; see scripts/teach_diagnose.py).

Usage:
    caffeinate -i uv run python scripts/teach_window.py                     # defaults: the best network, depth 3, 30,000 attempts
    caffeinate -i uv run python scripts/teach_window.py --attempts 60000    # rerun to carry on: it resumes from latest.npz

Pools (built once, in experiments/teach-stage1/): the first board holding 4096 from the network's own seeded greedy
games, 20,000 games from seed 100,000,000 to teach from and 1,000 from seed 110,000,000 to measure on.

Every --every attempts it measures: from each measuring board, the taught network plays greedily (spawns seeded by
the board's index) until 16384 or the game ends, and the share that makes 16384 is compared with the untaught
network's on the same boards and spawns (paired, with a 95% interval). Lines go to history.csv; the network is saved
as latest.npz and, when it measures best so far, best.npz, both whole networks evaluate.py can play. The temporal
coherence sums are saved alongside (sums.npz) so a rerun carries on as if it had never stopped.

Then evaluate whole games on the usual seeds, against the untaught network's 320.0k / 77.4% 16384 / 8.5% 32768:
    uv run python scripts/evaluate.py experiments/teach-stage1/<name>/best.npz --agent ntuple -n 1000
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
from deep_2048.agents.ntuple import points as pt  # noqa: E402
from deep_2048.agents.ntuple.stages import first_boards  # noqa: E402
from deep_2048.agents.ntuple.teach import teach_attempts, teach_rank_attempts  # noqa: E402
from deep_2048.bench import file_sha256  # noqa: E402

BEST = Path("checkpoints/ntuple/decision-8x6-tc-reset/games_2000000.npz")
OUTPUT = Path("experiments/teach-stage1")
TRAIN_SEED, MEASURE_SEED = 100_000_000, 110_000_000
FIELDS = ["attempts", "minutes", "teacher_led", "teacher_made", "student_led", "student_made", "moves", "disagreed", "measured", "made", "untaught", "gain", "gain_ci"]


def pools(network: NTupleNetwork, low: int, games: int, measure: int, threads: int, checkpoint: Path) -> tuple[np.ndarray, np.ndarray]:
    path = OUTPUT / f"pool-{1 << low}.npz"
    if path.exists():
        data = np.load(path)
        return data["train"], data["measure"]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    train, _ = first_boards(network, 1 << low, games, TRAIN_SEED, threads)
    measured, _ = first_boards(network, 1 << low, measure, MEASURE_SEED, threads)
    np.savez(path, train=train, measure=measured, source=str(checkpoint), sha256=file_sha256(checkpoint))
    print(f"Pools: {train.size:,} boards to teach from and {measured.size:,} to measure on, first holding {1 << low} ({time.perf_counter() - started:.0f} s)", flush=True)
    return train, measured


def reach(weights: np.ndarray, network: NTupleNetwork, boards: np.ndarray, goal: int, threads: int) -> np.ndarray:
    """1 for each measuring board from which greedy play makes 2**goal, its spawns seeded by the board's index."""
    def one(i: int) -> int:
        _, top, _, _ = pt.play_to(weights, network.layout, np.uint64(boards[i]), 1, 0.0, np.random.default_rng([MEASURE_SEED, i]), goal)
        return int(top >= 1 << goal)

    with ThreadPoolExecutor(max_workers=threads) as pool:
        return np.array(list(pool.map(one, range(boards.size))))


def main() -> None:
    parser = argparse.ArgumentParser(description="Teach a network its own search's play across a window of the game")
    parser.add_argument("--checkpoint", type=Path, default=BEST)
    parser.add_argument("--name", help="the run's folder under experiments/teach-stage1 (default: from the settings)")
    parser.add_argument("--low", type=int, default=12, help="the window's smallest largest tile's exponent (default 12, 4096)")
    parser.add_argument("--high", type=int, default=13, help="the window's largest largest tile's exponent (default 13, 8192)")
    parser.add_argument("--depth", type=int, default=3, help="the teacher's search depth (default 3)")
    parser.add_argument("--cutoff", type=float, default=0.0, help="the teacher prunes spawn branches less likely than this (default 0)")
    parser.add_argument("--mode", choices=("rank", "value"), default="rank")
    parser.add_argument("--step", type=float, default=0.5, help="rank: the share of the gap closed at each disagreement (default 0.5)")
    parser.add_argument("--margin", type=float, default=1.0, help="rank: how far ahead the search's move is aimed for, in score (default 1)")
    parser.add_argument("--rate", type=float, default=0.001, help="base learning rate a weight, scaled by temporal coherence (default 0.001)")
    parser.add_argument("--uncentred", action="store_true", help="teach the teacher's level as well as its ranking of the moves (the first run's setting, which collapsed)")
    parser.add_argument("--attempts", type=int, default=30_000, help="attempts in total (default 30,000); rerun with more to carry on")
    parser.add_argument("--every", type=int, default=500, help="attempts between measurements and saves (default 500)")
    parser.add_argument("--student-share", type=float, default=0.5, help="share of attempts that follow the student's own move (default 0.5)")
    parser.add_argument("--pool-games", type=int, default=20_000)
    parser.add_argument("--measure", type=int, default=1_000, help="games whose first 4096 boards are measured on (default 1,000)")
    parser.add_argument("--threads", type=int, default=os.cpu_count())
    args = parser.parse_args()

    if args.mode == "rank":
        name = args.name or f"w{1 << args.low}-{1 << args.high}-d{args.depth}-rank{args.step:g}"
    else:
        name = args.name or f"w{1 << args.low}-{1 << args.high}-d{args.depth}-r{args.rate:g}" + ("-uncentred" if args.uncentred else "-centred")
    folder = OUTPUT / name
    folder.mkdir(parents=True, exist_ok=True)
    base, metadata = NTupleNetwork.load(args.checkpoint)
    teacher = base.weights.copy()  # frozen: the targets never move
    train, measured = pools(base, args.low, args.pool_games, args.measure, args.threads, args.checkpoint)
    goal = args.high + 1
    untaught = reach(teacher, base, measured, goal, args.threads)
    print(f"Untaught greedy play makes {1 << goal} from {untaught.mean():.1%} of the {measured.size:,} measuring boards", flush=True)

    weights = base.weights
    errors, abs_errors = np.zeros(weights.size), np.zeros(weights.size)
    history_path = folder / "history.csv"
    done, best_made, minutes = 0, -1.0, 0.0
    if (folder / "latest.npz").exists():
        weights[:] = NTupleNetwork.load(folder / "latest.npz")[0].weights
        sums = np.load(folder / "sums.npz")
        errors[:], abs_errors[:] = sums["errors"], sums["abs_errors"]
        rows = list(csv.DictReader(history_path.open()))
        done, minutes = int(rows[-1]["attempts"]), float(rows[-1]["minutes"])
        best_made = max(float(row["made"]) for row in rows)
        print(f"Resuming {folder} at {done:,} attempts", flush=True)
    else:
        with history_path.open("w", newline="") as f:
            csv.DictWriter(f, FIELDS).writeheader()
        (folder / "source.txt").write_text(f"{args.checkpoint}\nsha256 {file_sha256(args.checkpoint)}\n{vars(args)}\n")

    how = f"move choice, step {args.step:g}, margin {args.margin:g}" if args.mode == "rank" else f"values, rate {args.rate:g}, {'uncentred' if args.uncentred else 'centred'} targets"
    print(f"Teaching {name}: window {1 << args.low}-{1 << args.high}, teacher depth {args.depth}, {how}, to {args.attempts:,} attempts on {args.threads} threads", flush=True)
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        while done < args.attempts:
            chunk = min(args.every, args.attempts - done)
            shares = [chunk // args.threads + (k < chunk % args.threads) for k in range(args.threads)]
            started = time.perf_counter()
            if args.mode == "rank":
                results = list(pool.map(lambda k: teach_rank_attempts(weights, teacher, base.layout, train, shares[k], args.depth, args.cutoff, args.step, args.margin,
                                                                      args.low, args.high, args.student_share, TRAIN_SEED + done + k * 1_000_003), range(args.threads)))
            else:
                results = [(*r, 0) for r in pool.map(lambda k: teach_attempts(weights, errors, abs_errors, teacher, base.layout, train, shares[k], args.depth, args.cutoff,
                                                                              args.rate, args.low, args.high, args.student_share, TRAIN_SEED + done + k * 1_000_003,
                                                                              not args.uncentred), range(args.threads))]
            done += chunk
            minutes += (time.perf_counter() - started) / 60
            totals = np.array(results).sum(axis=0)
            made = reach(weights, base, measured, goal, args.threads)
            difference = made - untaught
            gain_ci = 1.96 * difference.std(ddof=1) / np.sqrt(difference.size)
            row = {"attempts": done, "minutes": round(minutes, 1), "teacher_led": totals[0], "teacher_made": totals[1], "student_led": totals[2], "student_made": totals[3],
                   "moves": totals[4], "disagreed": round(float(totals[5] / max(totals[4], 1)), 4), "measured": measured.size, "made": round(float(made.mean()), 4), "untaught": round(float(untaught.mean()), 4),
                   "gain": round(float(difference.mean()), 4), "gain_ci": round(float(gain_ci), 4)}
            with history_path.open("a", newline="") as f:
                csv.DictWriter(f, FIELDS).writerow(row)
            base.save(folder / "latest.npz", metadata)
            np.savez(folder / "sums.npz", errors=errors.astype(np.float32), abs_errors=abs_errors.astype(np.float32))
            better = made.mean() > best_made
            if better:
                best_made = float(made.mean())
                base.save(folder / "best.npz", metadata)
            led = lambda n, m: f"{m / n:.0%}" if n else "–"  # noqa: E731
            disagreement = f", disagreed on {totals[5] / max(totals[4], 1):.0%} of moves" if args.mode == "rank" else ""
            print(f"  {done:,} attempts, {minutes:.0f} min: made {1 << goal} teacher-led {led(totals[0], totals[1])}, student-led {led(totals[2], totals[3])}{disagreement} | "
                  f"greedy from the measuring boards {made.mean():.1%} (untaught {untaught.mean():.1%}, {difference.mean():+.1%} ± {gain_ci:.1%})" + ("  best so far" if better else ""), flush=True)
    print(f"\nDone. Evaluate whole games with:\n  uv run python scripts/evaluate.py {folder / 'best.npz'} --agent ntuple -n 1000")


if __name__ == "__main__":
    main()
