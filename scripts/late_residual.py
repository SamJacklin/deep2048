"""Stage 3: correct the frozen network's late-game values with a constant or a learned residual, and test what it buys.

Usage:
    BASE=checkpoints/ntuple/decision-8x6-tc-reset/games_2000000.npz
    uv run python scripts/late_residual.py fit-constant "$BASE"                 # the constant arm, from the train split
    uv run python scripts/late_residual.py train "$BASE" --games 20000          # the residual arm, from the train split
    uv run python scripts/late_residual.py evaluate "$BASE" late-constant late-residual   # each arm on the test split
    uv run python scripts/late_residual.py report

The base network is never changed. Arms are compared on the corpus's held-out test split (scripts/late_game_corpus.py),
which neither fitting nor training sees:
- conversion from its first-16384 positions: 32 greedy and 8 depth-3 continuations each, with the same spawns for
  every arm (common random numbers), so arms are compared position by position;
- the network's estimate of the score to come there, against what its own greedy continuations score;
- how often each arm's greedy choice differs from the base's, on late positions (largest tile 16384 or more) and on
  boundary positions, where a move makes the game's first 16384. There some moves' afterstates are past the gate and
  some aren't, so even a constant changes which move looks best.
Full games at exact depth 3 come from scripts/evaluate.py, since corrected checkpoints load as agents.
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
from deep_2048.agents.ntuple.diagnostics import greedy_values, rollout
from deep_2048.agents.ntuple.network import evaluate
from deep_2048.agents.ntuple.residual import RESIDUAL_TUPLES, CorrectedNetwork, is_corrected_checkpoint, train_residual
from deep_2048.agents.ntuple.search import play_from
from deep_2048.game import bitboard as bb

CORPUS = Path("experiments/late-game-corpus/positions.csv")
SAMPLES = Path("output/late-game-corpus/samples.csv")
OUTPUT = Path("experiments/late-residual")
CHECKPOINTS = Path("checkpoints/ntuple/late-residual")
CONTINUATIONS = {"greedy": (1, 32), "depth3": (3, 8)}  # name -> (search depth, continuations per position)
POOL = ThreadPoolExecutor(max_workers=os.cpu_count())


def read_rows(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def board_of(row: dict) -> np.uint64:
    return np.uint64(int(row["board"], 16))


def seed_of(row: dict) -> int:
    return (1 if row["policy"] == "greedy" else 2) * 10**10 + int(row["seed"]) * 10**4 + int(row["move"]) % 10**4


def load(path: str | Path) -> CorrectedNetwork | NTupleNetwork:
    return CorrectedNetwork.load(path) if is_corrected_checkpoint(path) else NTupleNetwork.load(path)[0]


def arm_path(arm: str, base: str) -> Path:
    return Path(base) if arm == "baseline" else CHECKPOINTS / arm / "latest.npz"


# ---------------------------------------------------------------- the constant


def fit_constant(base_path: str, positions: int, rollouts: int) -> None:
    """The mean gap between greedy play's returns and the base's values, over gated afterstates from the train split."""
    base, _ = NTupleNetwork.load(base_path)
    rows = [r for r in read_rows(SAMPLES) if r["split"] == "train" and int(r["max_tile"]) >= 16384 and np.isfinite(float(r["greedy_margin"]))]
    rows = [rows[i] for i in np.random.default_rng(0).choice(len(rows), min(positions, len(rows)), replace=False)]
    afterstates, predicted = [], []
    for row in rows:
        board = board_of(row)
        values = greedy_values(base.weights, base.layout, board)
        afterstate, _ = bb.move(board, int(np.nanargmax(values)))
        afterstates.append(np.uint64(afterstate))
        predicted.append(evaluate(base.weights, base.layout, np.uint64(afterstate), board))
    jobs = [(k, j) for k in range(len(rows)) for j in range(rollouts)]
    returns = np.array(list(POOL.map(lambda job: rollout(base.weights, base.layout, afterstates[job[0]], 1, 0.0, np.random.default_rng([seed_of(rows[job[0]]), job[1]]))[0], jobs)))
    gaps = returns.reshape(len(rows), rollouts).mean(axis=1) - np.array(predicted)
    constant = float(gaps.mean())
    interval = 1.96 * gaps.std(ddof=1) / np.sqrt(len(gaps))
    corrected = CorrectedNetwork.from_base(base_path, constant=round(constant), notes=f"fitted on {len(rows)} train-split late afterstates, {rollouts} greedy rollouts each")
    corrected.save(CHECKPOINTS / "late-constant" / "latest.npz")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fit = {"constant": constant, "interval": interval, "positions": len(rows), "rollouts": rollouts,
           "mean_predicted": float(np.mean(predicted)), "mean_return": float(returns.mean())}
    (OUTPUT / "constant_fit.json").write_text(json.dumps(fit, indent=2) + "\n")
    print(f"Constant {constant:+,.0f} ± {interval:,.0f}: greedy returns average {returns.mean():,.0f} where the base predicts {np.mean(predicted):,.0f}")


# ---------------------------------------------------------------- the residual


def train(base_path: str, games: int, board_step: float, batch: int, name: str, resume: bool) -> None:
    """Train the residual on greedy games from the train split's first-16384 positions, a batch at a time."""
    checkpoint = CHECKPOINTS / name / "latest.npz"
    corrected = CorrectedNetwork.load(checkpoint) if resume and checkpoint.exists() else CorrectedNetwork.from_base(base_path, residual=True)
    rate = board_step / (len(RESIDUAL_TUPLES) * bb.NUM_SYMMETRIES)
    starts = np.array([board_of(r) for r in read_rows(CORPUS) if r["kind"] == "first-16384" and r["split"] == "train"], dtype=np.uint64)
    history_path = OUTPUT / name / "training_history.csv"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    if not resume or not history_path.exists():
        history_path.write_text("games,moves,score_added,rate_32768,mean_abs_td_error,games_per_second\n")
    rng = np.random.default_rng(corrected.correction.residual_games)
    threads = os.cpu_count() or 1
    print(f"Training {name} from {corrected.correction.residual_games:,} to {games:,} games from {starts.size} first-16384 positions, rate {rate:.5f} per weight ({board_step} per board)")
    while corrected.correction.residual_games < games:
        n = min(batch, games - corrected.correction.residual_games)
        chosen = rng.choice(starts, n)
        start = time.perf_counter()
        seeds = rng.integers(0, 2**31, threads)
        parts = list(POOL.map(lambda k: train_residual(corrected.weights, corrected.layout, np.array_split(chosen, threads)[k], rate, int(seeds[k])), range(threads)))
        scores, tops, moves, errors = (np.concatenate(field) for field in zip(*parts))
        seconds = time.perf_counter() - start
        corrected.correction.residual_games += n
        corrected.correction.residual_moves += int(moves.sum())
        corrected.correction.residual_rate = rate
        corrected.save(checkpoint)
        with open(history_path, "a") as f:
            f.write(f"{corrected.correction.residual_games},{corrected.correction.residual_moves},{scores.mean():.1f},{np.mean(tops >= 15):.4f},{errors.mean():.1f},{n / seconds:.1f}\n")
        print(f"{corrected.correction.residual_games:>8,} games: score added {scores.mean():>9,.0f}, 32768 in {np.mean(tops >= 15):6.1%}, "
              f"mean |TD error| {errors.mean():>8,.0f}, {n / seconds:,.0f} games/s", flush=True)
    assert corrected.base_unchanged()


# ---------------------------------------------------------------- evaluation


def evaluate_arm(base_path: str, arm: str) -> None:
    network = load(arm_path(arm, base_path))
    weights, layout = network.weights, network.layout
    base = network.base if isinstance(network, CorrectedNetwork) else network
    rows = [r for r in read_rows(CORPUS) if r["kind"] == "first-16384" and r["split"] == "test"]
    results = {}
    for name, (depth, n) in CONTINUATIONS.items():
        jobs = [(p, j) for p in range(len(rows)) for j in range(n)]
        outcome = list(POOL.map(lambda job: play_from(weights, layout, board_of(rows[job[0]]), depth, 0.0, np.random.default_rng([seed_of(rows[job[0]]), job[1]])), jobs))
        results[f"{name}_scores"] = np.array([s for s, _ in outcome]).reshape(len(rows), n)
        results[f"{name}_tiles"] = np.array([t for _, t in outcome]).reshape(len(rows), n)
    results["predicted"] = np.array([np.nanmax(greedy_values(weights, layout, board_of(r))) for r in rows])

    # Where the arm's greedy choice differs from the base's
    samples = [r for r in read_rows(SAMPLES) if r["split"] == "test"]
    late = [board_of(r) for r in samples if int(r["max_tile"]) >= 16384]
    boundary = [board_of(r) for r in samples if int(r["max_tile"]) == 8192 and any(
        bb.max_exponent(np.uint64(bb.move(board_of(r), a)[0])) >= 14 for a in range(bb.NUM_ACTIONS))]
    for label, boards in (("late", late), ("boundary", boundary)):
        mine = np.array([np.nanargmax(greedy_values(weights, layout, b)) for b in boards])
        theirs = np.array([np.nanargmax(greedy_values(base.weights, base.layout, b)) for b in boards])
        results[f"{label}_changed"] = mine != theirs
        if label == "boundary":
            results["boundary_merges"] = np.array([bb.max_exponent(np.uint64(bb.move(b, int(a))[0])) >= 14 for b, a in zip(boards, mine)])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUTPUT / f"{arm}.npz", **results)


def paired(values: np.ndarray, reference: np.ndarray) -> str:
    differences = values - reference
    return f"{differences.mean():+,.0f} ± {1.96 * differences.std(ddof=1) / np.sqrt(len(differences)):,.0f}"


def paired_rate(values: np.ndarray, reference: np.ndarray) -> str:
    differences = values - reference
    return f"{differences.mean():+.1%} ± {1.96 * differences.std(ddof=1) / np.sqrt(len(differences)):.1%}"


def report() -> None:
    arms = {path.stem: np.load(path) for path in sorted(OUTPUT.glob("*.npz"))}
    if "baseline" not in arms:
        raise SystemExit("evaluate the baseline arm first")
    reference = arms.pop("baseline")
    arms = {"baseline": reference, **arms}
    positions = len(reference["predicted"])
    print(f"### Conversion from the test split's {positions} first-16384 positions (differences paired by position, ± 95%)\n")
    print("| Arm | Continued by | Makes 32768 | vs baseline | Score added | vs baseline |")
    print("|---|---|---:|---:|---:|---:|")
    for arm, d in arms.items():
        for name in CONTINUATIONS:
            made = (d[f"{name}_tiles"] >= 32768).mean(axis=1)
            added = d[f"{name}_scores"].mean(axis=1)
            base_made = (reference[f"{name}_tiles"] >= 32768).mean(axis=1)
            base_added = reference[f"{name}_scores"].mean(axis=1)
            versus = ("—", "—") if arm == "baseline" else (paired_rate(made, base_made), paired(added, base_added))
            print(f"| {arm} | {name}, {d[f'{name}_scores'].shape[1]} each | {made.mean():.1%} | {versus[0]} | {added.mean():,.0f} | {versus[1]} |")
    print("\n### Estimates and choices\n")
    print("| Arm | Estimate of the score to come at a first 16384 | Its own greedy play's | Estimate ÷ actual | "
          "Late positions with a different greedy move from the base | Boundary positions with a different move | Boundary positions where it makes the 16384 |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for arm, d in arms.items():
        actual = d["greedy_scores"].mean()
        print(f"| {arm} | {d['predicted'].mean():,.0f} | {actual:,.0f} | {d['predicted'].mean() / actual:.2f} | "
              f"{d['late_changed'].mean():.1%} of {d['late_changed'].size:,} | {d['boundary_changed'].mean():.1%} of {d['boundary_changed'].size:,} | {d['boundary_merges'].mean():.1%} |")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3: a late-game correction for the frozen network")
    parser.add_argument("step", choices=("fit-constant", "train", "evaluate", "report"))
    parser.add_argument("base", nargs="?", help="The frozen base checkpoint")
    parser.add_argument("arms", nargs="*", help="evaluate: arms to evaluate (baseline, late-constant, or a residual run's name)")
    parser.add_argument("--positions", type=int, default=1000, help="fit-constant: train-split late positions (default: 1,000)")
    parser.add_argument("--rollouts", type=int, default=8, help="fit-constant: greedy rollouts per position (default: 8)")
    parser.add_argument("--games", type=int, default=20000, help="train: total late games (default: 20,000)")
    parser.add_argument("--board-step", type=float, default=0.1, help="train: the residual's step per board, shared over its 32 reads (default: 0.1)")
    parser.add_argument("--batch", type=int, default=1000, help="train: games per batch (default: 1,000)")
    parser.add_argument("--name", default="late-residual", help="train: the run's name (default: late-residual)")
    parser.add_argument("--resume", action="store_true", help="train: carry on from the run's latest checkpoint")
    args = parser.parse_args()
    if args.step == "fit-constant":
        fit_constant(args.base, args.positions, args.rollouts)
    elif args.step == "train":
        train(args.base, args.games, args.board_step, args.batch, args.name, args.resume)
    elif args.step == "evaluate":
        for arm in args.arms:
            evaluate_arm(args.base, arm)
            print(f"Evaluated {arm}")
        report()
    else:
        report()


if __name__ == "__main__":
    main()
