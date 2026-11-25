"""Stage 6 follow-up: the scratch late network again, carried on to 3M continuations without a reset.

Usage:
    uv run python scripts/late_network_continue.py run       # about 3.5 hours on 8 threads; needs about 8 GB of free disk
    uv run python scripts/late_network_continue.py report

Stage 6's scratch arm at 0.02 was still improving at 2M continuations (-23k score added at 1M, -10k at 1.5M, -4k at
2M) when its coherence sums were reset, the continuation without a reset was never run, and its depth-3 evaluations
at 2M and 3M were cancelled. Its checkpoint is no longer on disk, so this trains the same arm again from zeros, under
its own name, straight through to 3M continuations with no reset. The first 2M are therefore a second run of the
original arm (training on 8 threads isn't bit-reproducible), which also shows how repeatable its climb was.

Everything else is stage 6's, imported from late_network.py: the frozen 2M network as stage 1, the 100,000-position
training pool, the recipe, and the evaluations, paired with stage 6's stored control on the same positions and spawns.
Greedy dev continuations and full greedy games every 100k continuations; depth 3 at 1M, 2M and 3M.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import late_network as ln  # noqa: E402
from add_stage import add_stage  # noqa: E402

from deep_2048.agents.ntuple import NTupleNetwork  # noqa: E402
from deep_2048.agents.ntuple.models import CheckpointMetadata  # noqa: E402

NAME = "late-network-scratch-0.02-b"
RATE = 0.02
TARGET = 3_000_000
STEP = 100_000
DEPTH3_AT = (1_000_000, 2_000_000, 3_000_000)
START = ln.STARTS / "scratch.npz"
CHECKPOINT = Path("checkpoints/ntuple") / NAME / "latest.npz"
REQUIRED = (Path(ln.BASE), ln.POSITIONS, ln.TRAIN_POOL, ln.PREFIXES, ln.EVALUATIONS / "control.npz")


def build_start() -> None:
    """The scratch start, as late_network.build makes it: stage 1 the 2M network, stage 2 zeros, no coherence sums."""
    if START.exists():
        return
    base, _ = NTupleNetwork.load(ln.BASE)
    n = ln.tables(base)
    network = add_stage(base, 16384)
    network.weights[n : 2 * n] = 0
    network.save(START, CheckpointMetadata(games_played=0, moves_played=0, learning_rate=0.0, temporal_coherence=True))
    print(f"Saved {START}: stage 1 from {ln.BASE}, stage 2 zeros", flush=True)


def train_to(stop: int) -> None:
    resume = CHECKPOINT if CHECKPOINT.exists() else START
    if ln.games_in(resume) >= stop:
        return
    free = shutil.disk_usage(".").free / 1e9
    if free < ln.MIN_FREE_GB:
        raise SystemExit(f"Only {free:.1f} GB free: stopping before training {NAME} to {stop:,}")
    command = [sys.executable, "-u", "scripts/train_ntuple.py", "--run", NAME, "--resume", str(resume), "--games", str(stop)]
    # The optimiser is named outright so a shell left set up for stage 8 can't switch this run to AutoTIDBD
    with open(ln.OUTPUT / f"train-{NAME}.log", "a") as log:
        subprocess.run(command, env=os.environ | ln.recipe(RATE) | {"NTUPLE_OPTIMIZER": "tc"}, stdout=log, stderr=subprocess.STDOUT, check=True)


def run() -> None:
    missing = [str(p) for p in REQUIRED if not p.exists()]
    if missing:
        raise SystemExit("Stage 6's files are needed and missing: " + ", ".join(missing))
    build_start()
    for stop in range(STEP, TARGET + 1, STEP):
        train_to(stop)
        ln.evaluate(f"{NAME}-{stop}", CHECKPOINT, depth3=stop in DEPTH3_AT)
        print(f"{time.strftime('%H:%M')} {NAME} evaluated at {stop:,}", flush=True)
    report()


def report() -> None:
    """This arm against the control, and against stage 6's first run of it where that was evaluated."""
    c = dict(np.load(ln.EVALUATIONS / "control.npz"))
    c_added, c_made = ln.per_position(c, "greedy_greedy_2m")
    print(f"Control (the 2M network alone): dev greedy continuations add {c_added.mean():,.0f} and make 32768 in {c_made.mean():.1%}; "
          f"full greedy games {c['greedy_full_scores'].mean():,.0f}, depth 3 {c['depth3_full_scores'].mean():,.0f}\n")
    print("| Continuations | Score added | Makes 32768 | Full greedy | Depth-3 added | Depth-3 full | ≥32768 at depth 3 | First run: score added |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for stop in range(STEP, TARGET + 1, STEP):
        path = ln.EVALUATIONS / f"{NAME}-{stop}.npz"
        if not path.exists():
            continue
        d = dict(np.load(path))
        added, made = ln.per_position(d, "greedy_greedy_2m")
        cells = [f"{stop:,}", ln.interval(added - c_added), ln.interval(made - c_made, True), ln.interval(d["greedy_full_scores"] - c["greedy_full_scores"])]
        if "depth3_full_scores" in d:
            cells += [ln.interval(d["depth3_greedy_2m_scores"][:, 0] - c["depth3_greedy_2m_scores"][:, 0]), ln.interval(d["depth3_full_scores"] - c["depth3_full_scores"]),
                      f"{(d['depth3_full_tiles'] >= 32768).mean():.1%} (control {(c['depth3_full_tiles'] >= 32768).mean():.1%})"]
        else:
            cells += ["", "", ""]
        first = ln.EVALUATIONS / f"{ln.run_name('scratch', RATE)}-{stop}.npz"
        cells.append(ln.interval(ln.per_position(dict(np.load(first)), "greedy_greedy_2m")[0] - c_added) if first.exists() else "")
        print("| " + " | ".join(cells) + " |")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 6 follow-up: the scratch late network to 3M continuations without a reset")
    parser.add_argument("step", choices=("run", "report"))
    args = parser.parse_args()
    {"run": run, "report": report}[args.step]()


if __name__ == "__main__":
    main()
