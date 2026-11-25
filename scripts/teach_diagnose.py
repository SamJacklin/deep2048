"""Diagnose the stage 1 teacher's collapse: is it a bug or what teaching does to shared weights?

    caffeinate -i uv run python scripts/teach_diagnose.py      # about 10-20 minutes; stop the teaching run first

Uses the teaching pools in experiments/teach-stage1/pool-4096.npz and the best network. Prints, then writes to
experiments/teach-stage1/diagnose.txt:

A. The size of what is taught, before anything is learned: along the network's own greedy play through the window from
   200 measuring boards, the teacher's depth-3 value of each legal move against the network's own, raw and centred,
   and how often the two pick the same move. Anything not finite here is a bug in the targets.
B. Controls that must not collapse, each from a fresh copy of the network, single-threaded:
   - rate 0: nothing may change;
   - a depth-1 teacher (the network's own values): every error starts at 0, so play should stay where it is.
   If either falls, the teaching code is broken.
C. The real setting, single-threaded, measured after 1, 5, 20, 50, 100 and 200 attempts: the share of measuring
   boards greedy play takes to 16384, weights that aren't finite, the largest weight and the largest change, and how
   many weights moved. This shows when the fall starts and whether weights blow up (a bug) or drift (the method).
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from numba import njit

sys.path.insert(0, str(Path(__file__).parent.parent))

from deep_2048.agents.ntuple import NTupleNetwork  # noqa: E402
from deep_2048.agents.ntuple import points as pt  # noqa: E402
from deep_2048.agents.ntuple.network import TupleLayout, evaluate  # noqa: E402
from deep_2048.agents.ntuple.search import _search, new_transpositions  # noqa: E402
from deep_2048.agents.ntuple.teach import in_stretch, teach_attempts, teach_rank_attempts  # noqa: E402
from deep_2048.game import bitboard as bb  # noqa: E402

BEST = Path("checkpoints/ntuple/decision-8x6-tc-reset/games_2000000.npz")
POOL = Path("experiments/teach-stage1/pool-4096.npz")
OUT = Path("experiments/teach-stage1/diagnose.txt")  # written next to the pools
lines: list[str] = []


def say(text: str = "") -> None:
    print(text, flush=True)
    lines.append(text)


@njit
def target_gaps(weights: np.ndarray, layout: TupleLayout, starts: np.ndarray, depth: int, cap: int, seed: int, low: int, high: int) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    """Along greedy play from each start through the window: (raw gaps, centred gaps, decisions, agreements, non-finite targets)."""
    np.random.seed(seed)
    table = new_transpositions()
    raw = np.zeros(cap * 4)
    centred = np.zeros(cap * 4)
    n = decisions = agreed = bad = 0
    for s in range(starts.size):
        board = starts[s]
        while in_stretch(board, low, high) and decisions < cap:
            values = _search(weights, layout, board, depth, 0.0, table)
            own = np.full(4, np.nan)
            gaps = np.full(4, np.nan)
            teacher_best = student_best = -1
            for a in range(4):
                if np.isnan(values[a]):
                    continue
                if not np.isfinite(values[a]):
                    bad += 1
                after, reward = bb.move(board, a)
                own[a] = reward + evaluate(weights, layout, after, board)
                gaps[a] = values[a] - own[a]
                if teacher_best < 0 or values[a] > values[teacher_best]:
                    teacher_best = a
                if student_best < 0 or own[a] > own[student_best]:
                    student_best = a
            if student_best < 0:
                break
            mean = 0.0
            legal = 0
            for a in range(4):
                if not np.isnan(gaps[a]):
                    mean += gaps[a]
                    legal += 1
            mean /= legal
            for a in range(4):
                if not np.isnan(gaps[a]):
                    raw[n] = gaps[a]
                    centred[n] = gaps[a] - mean
                    n += 1
            decisions += 1
            agreed += teacher_best == student_best
            board, _ = bb.move(board, student_best)
            board = bb.spawn_tile(board)
    return raw[:n], centred[:n], decisions, agreed, bad


def reach(weights: np.ndarray, layout: TupleLayout, boards: np.ndarray, threads: int) -> float:
    def one(i: int) -> int:
        _, top, _, _ = pt.play_to(weights, layout, np.uint64(boards[i]), 1, 0.0, np.random.default_rng([110_000_000, i]), WINDOW[1] + 1)
        return int(top >= 1 << (WINDOW[1] + 1))

    with ThreadPoolExecutor(max_workers=threads) as pool:
        return float(np.mean(list(pool.map(one, range(boards.size)))))


def describe(values: np.ndarray) -> str:
    q = np.percentile(np.abs(values), [50, 90, 99])
    return f"mean {values.mean():+,.0f}, mean size {np.abs(values).mean():,.0f}, sizes p50/p90/p99 {q[0]:,.0f}/{q[1]:,.0f}/{q[2]:,.0f}, largest {np.abs(values).max():,.0f}"


def run(label: str, base: NTupleNetwork, train: np.ndarray, measure: np.ndarray, depth: int, rate: float, centred: bool, checkpoints: list[int], threads: int,
        rank: bool = False) -> None:
    how = f"move choice, step {rate:g}" if rank else f"values, rate {rate:g}, {'centred' if centred else 'uncentred'}"
    say(f"\n{label}: teacher depth {depth}, {how}, one thread")
    say(f"{'attempts':>9} {'reach goal':>12} {'not finite':>11} {'largest weight':>15} {'largest change':>15} {'weights moved':>14} {'minutes':>8}")
    original = base.weights.copy()
    teacher = original.copy()
    weights = original.copy()
    errors, abs_errors = np.zeros(weights.size), np.zeros(weights.size)
    done, started = 0, time.perf_counter()
    for target in checkpoints:
        if target > done:
            if rank:
                result = teach_rank_attempts(weights, teacher, base.layout, train, target - done, depth, 0.0, rate, 1.0, WINDOW[0], WINDOW[1], 0.5, 7 + done)
                say(f"{'':>9} (disagreed on {result[5] / max(result[4], 1):.0%} of {result[4]:,} moves since the last line)")
            else:
                teach_attempts(weights, errors, abs_errors, teacher, base.layout, train, target - done, depth, 0.0, rate, WINDOW[0], WINDOW[1], 0.5, 7 + done, centred)
            done = target
        finite = np.isfinite(weights)
        change = np.abs(weights[finite].astype(np.float64) - original[finite])
        say(f"{done:>9,} {reach(weights, base.layout, measure, threads):>12.1%} {int((~finite).sum()):>11,} {float(np.abs(weights[finite]).max()):>15,.0f} "
            f"{float(change.max()):>15,.0f} {int((change > 0).sum()):>14,} {(time.perf_counter() - started) / 60:>8.1f}")


WINDOW = [12, 13]


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose the stage 1 teacher")
    parser.add_argument("--checkpoint", type=Path, default=BEST)
    parser.add_argument("--pool", type=Path, default=POOL)
    parser.add_argument("--low", type=int, default=12)
    parser.add_argument("--high", type=int, default=13)
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 8)
    parser.add_argument("--sweep", action="store_true", help="skip A and B; run C at each --rates, centred and not, to 50 attempts")
    parser.add_argument("--rates", type=float, nargs="+", default=[1e-5, 1e-4])
    parser.add_argument("--rank", action="store_true", help="run C for the move-choice teaching at each --steps, to 100 attempts")
    parser.add_argument("--steps", type=float, nargs="+", default=[0.1, 0.5])
    args = parser.parse_args()
    WINDOW[:] = [args.low, args.high]
    threads = args.threads
    base, _ = NTupleNetwork.load(args.checkpoint)
    pools = np.load(args.pool)
    train, measure = pools["train"], pools["measure"]
    say(f"Network {args.checkpoint}: {base.weights.size:,} weights, {int((~np.isfinite(base.weights)).sum())} not finite, largest {float(np.abs(base.weights).max()):,.0f}")
    say(f"Untaught: greedy play takes {reach(base.weights, base.layout, measure, threads):.1%} of the {measure.size} measuring boards to {1 << (args.high + 1)}")

    if args.rank:
        global OUT
        OUT = OUT.with_name("diagnose-rank.txt")
        for step in args.steps:
            run(f"Move choice: step {step:g}", base, train, measure, 3, step, False, [1, 5, 20, 50, 100], threads, rank=True)
        OUT.write_text("\n".join(lines) + "\n")
        say(f"\nWritten to {OUT}")
        return
    if args.sweep:
        OUT = OUT.with_name("diagnose-sweep.txt")
        for rate in args.rates:
            for centred in (False, True):
                run(f"Sweep: rate {rate:g}", base, train, measure, 3, rate, centred, [1, 5, 20, 50], threads)
        OUT.write_text("\n".join(lines) + "\n")
        say(f"\nWritten to {OUT}")
        return

    say("\nA. What the depth-3 teacher would teach, along the network's own play from 200 measuring boards (nothing learned)")
    raw, centred, decisions, agreed, bad = target_gaps(base.weights, base.layout, measure[:200], 3, 60_000, 1, args.low, args.high)
    say(f"  {decisions:,} decisions, {raw.size:,} moves valued; teacher and network pick the same move {agreed / decisions:.1%} of the time; targets not finite: {bad}")
    say(f"  teacher value minus the network's, raw:     {describe(raw)}")
    say(f"  the same, centred on each decision's mean:  {describe(centred)}")

    say("\nB. Controls (must not collapse)")
    run("B1 rate 0", base, train, measure, 3, 0.0, True, [20], threads)
    run("B2 depth-1 teacher (the network's own values)", base, train, measure, 1, 0.001, True, [20, 100], threads)

    say("\nC. The real setting")
    run("C1", base, train, measure, 3, 0.001, True, [1, 5, 20, 50, 100, 200], threads)
    OUT.write_text("\n".join(lines) + "\n")
    say(f"\nWritten to {OUT}")


if __name__ == "__main__":
    main()
