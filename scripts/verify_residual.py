"""Verify stage 3: the corrected network's integration, and whether any correction improves play rather than values.

Usage:
    BASE=checkpoints/ntuple/decision-8x6-tc-reset/games_2000000.npz
    uv run python scripts/verify_residual.py equivalence "$BASE"   # 1: a zero correction plays exactly as the base
    uv run python scripts/verify_residual.py sweep "$BASE"         # 3, 4, 5: gate, clamping and strength variants
    uv run python scripts/verify_residual.py calibration "$BASE"   # 6: every variant against the base's own returns
    uv run python scripts/verify_residual.py rankings "$BASE"      # 7: what corrections do to stage 1's ranked moves
    uv run python scripts/verify_residual.py report

Item 2 (training correctness) is tests/test_residual.py. Everything here uses the corpus's dev split, apart from item 7,
which reuses stage 1's action-ranking positions and rollouts. The test split stays untouched for final confirmation.
Full games use seeds 10,000 to 10,999, clear of the 0-999 evaluation seeds.
"""

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from deep_2048.agents.ntuple import NTupleNetwork
from deep_2048.agents.ntuple.diagnostics import boundary_boards, greedy_values, rollout
from deep_2048.agents.ntuple.network import evaluate
from deep_2048.agents.ntuple.residual import CorrectedNetwork
from deep_2048.agents.ntuple.search import action_values, play_from, play_seeded
from deep_2048.game import bitboard as bb
from scripts.late_game_tests import ranking_errors

CORPUS = Path("experiments/late-game-corpus/positions.csv")
SAMPLES = Path("output/late-game-corpus/samples.csv")
CHECKPOINTS = Path("checkpoints/ntuple/late-residual")
OUTPUT = Path("experiments/late-residual/verification")
CACHE = Path("output/late-residual")
FULL_GAME_SEEDS = range(10_000, 11_000)
POOL = ThreadPoolExecutor(max_workers=os.cpu_count())

# name -> (correction checkpoint or None for the base, strength, gate on the pre-move board, clamp at 0)
VARIANTS: dict[str, tuple[str | None, float, bool, bool]] = {
    "baseline": (None, 1.0, False, False),
    "constant": ("late-constant", 1.0, False, False),
    "constant, pre-move gate": ("late-constant", 1.0, True, False),
    "constant, clamped": ("late-constant", 1.0, False, True),
    **{f"residual 0.01, strength {s:g}": ("residual-own-0.01", s, False, False) for s in (0.05, 0.1, 0.25, 0.5, 1.0)},
    "residual 0.01, pre-move gate": ("residual-own-0.01", 1.0, True, False),
    "residual 0.01, clamped": ("residual-own-0.01", 1.0, False, True),
    **{f"residual 0.1, strength {s:g}": ("late-residual", s, False, False) for s in (0.05, 0.1, 0.25)},
}


def read_rows(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def board_of(row: dict) -> np.uint64:
    return np.uint64(int(row["board"], 16))


def seed_of(row: dict) -> int:
    return (1 if row["policy"] == "greedy" else 2) * 10**10 + int(row["seed"]) * 10**4 + int(row["move"]) % 10**4


def legal_afterstates(board: np.uint64) -> list[tuple[int, np.uint64, int]]:
    moves = []
    for action in range(bb.NUM_ACTIONS):
        afterstate, reward = bb.move(board, action)
        if afterstate != board:
            moves.append((action, np.uint64(afterstate), reward))
    return moves


class Models:
    """The base and each correction checkpoint, loaded once; variants share their weights."""

    def __init__(self, base_path: str) -> None:
        self.base, _ = NTupleNetwork.load(base_path)
        self.corrections: dict[str, CorrectedNetwork] = {}

    def get(self, variant: str):  # type: ignore[no-untyped-def]
        checkpoint, scale, pre_move, clamp = VARIANTS[variant]
        if checkpoint is None:
            return self.base.weights, self.base.layout
        if checkpoint not in self.corrections:
            self.corrections[checkpoint] = CorrectedNetwork.load(CHECKPOINTS / checkpoint / "latest.npz")
        corrected = self.corrections[checkpoint]
        return corrected.weights, corrected.variant(scale=scale, gate_on_state=pre_move, clamp=clamp)


def dev_late_boards(limit: int, seed: int = 0) -> list[np.uint64]:
    rows = [r for r in read_rows(SAMPLES) if r["split"] == "dev" and int(r["max_tile"]) >= 16384 and not bb.is_game_over(board_of(r))]
    picks = np.random.default_rng(seed).choice(len(rows), min(limit, len(rows)), replace=False)
    return [board_of(rows[i]) for i in picks]


def dev_boundary_boards(base: NTupleNetwork) -> np.ndarray:
    path = CACHE / "boundary-dev.npy"
    if path.exists():
        return np.load(path)
    games = sorted({(r["policy"], int(r["seed"])) for r in read_rows(CORPUS) if r["split"] == "dev"})
    depths = {"greedy": 1, "depth3": 3}
    found = list(POOL.map(lambda g: boundary_boards(base.weights, base.layout, depths[g[0]], 0.0, np.random.default_rng(g[1])), games))
    boards = np.concatenate(found).astype(np.uint64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, boards)
    return boards


# ---------------------------------------------------------------- 1: equivalence


def equivalence(base_path: str) -> None:
    base, _ = NTupleNetwork.load(base_path)
    zero = CorrectedNetwork.from_base(base_path, residual=True)
    trained = CorrectedNetwork.load(CHECKPOINTS / "residual-own-0.01" / "latest.npz")
    candidates = {"zero correction": (zero.weights, zero.layout), "trained residual at strength 0": (trained.weights, trained.variant(scale=0.0))}
    rows = [r for r in read_rows(SAMPLES) if r["split"] == "dev" and not bb.is_game_over(board_of(r))]
    boards = [board_of(rows[i]) for i in np.random.default_rng(0).choice(len(rows), 2000, replace=False)]
    late = [b for b in boards if bb.max_exponent(b) >= 14][:300]
    results = {}
    for name, (weights, layout) in candidates.items():
        checks = {
            "greedy values, 2,000 dev boards": all(np.array_equal(greedy_values(weights, layout, b), greedy_values(base.weights, base.layout, b), equal_nan=True) for b in boards),
            "depth-3 values, 300 late dev boards": all(
                np.array_equal(action_values(weights, layout, b, 3), action_values(base.weights, base.layout, b, 3), equal_nan=True) for b in late
            ),
        }
        def both(seed: int, depth: int) -> bool:
            rng = np.random.default_rng
            return play_seeded(weights, layout, depth, 0.0, rng(seed)) == play_seeded(base.weights, base.layout, depth, 0.0, rng(seed))

        checks["1,000 greedy games, identical scores and tiles"] = all(POOL.map(lambda s: both(s, 1), FULL_GAME_SEEDS))
        checks["50 depth-3 games, identical scores and tiles"] = all(POOL.map(lambda s: both(s, 3), FULL_GAME_SEEDS[:50]))
        results[name] = checks
        for check, passed in checks.items():
            print(f"{name}: {check}: {'PASS' if passed else 'FAIL'}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "equivalence.json").write_text(json.dumps(results, indent=2) + "\n")


# ---------------------------------------------------------------- 3, 4, 5: the sweep


def sweep(base_path: str, variants: list[str], greedy_runs: int, depth3_runs: int) -> None:
    models = Models(base_path)
    rows = [r for r in read_rows(CORPUS) if r["kind"] == "first-16384" and r["split"] == "dev"]
    late = dev_late_boards(2000)
    boundary = list(dev_boundary_boards(models.base))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for variant in variants:
        path = OUTPUT / f"sweep-{variant}.npz"
        if path.exists():
            continue
        weights, layout = models.get(variant)
        out: dict[str, np.ndarray] = {}
        for name, depth, n in (("greedy", 1, greedy_runs), ("depth3", 3, depth3_runs)):
            jobs = [(p, j) for p in range(len(rows)) for j in range(n)]
            games = list(POOL.map(lambda job: play_from(weights, layout, board_of(rows[job[0]]), depth, 0.0, np.random.default_rng([seed_of(rows[job[0]]), job[1]])), jobs))
            out[f"{name}_scores"] = np.array([s for s, _ in games]).reshape(len(rows), n)
            out[f"{name}_tiles"] = np.array([t for _, t in games]).reshape(len(rows), n)
        full = list(POOL.map(lambda s: play_seeded(weights, layout, 1, 0.0, np.random.default_rng(s)), FULL_GAME_SEEDS))
        out["full_scores"], out["full_tiles"] = np.array([s for s, _ in full]), np.array([t for _, t in full])
        # 4: values below 0 on late boards, by how many moves are legal
        negative, legal_counts = [], []
        for board in late:
            moves = legal_afterstates(board)
            negative.append(np.mean([evaluate(weights, layout, a, board) < 0 for _, a, _ in moves]))
            legal_counts.append(len(moves))
        out["late_negative"], out["late_legal"] = np.array(negative), np.array(legal_counts)
        # 3: at the boundary, whether greedy play makes the 16384
        out["boundary_merges"] = np.array([bb.max_exponent(np.uint64(bb.move(b, int(np.nanargmax(greedy_values(weights, layout, b))))[0])) >= 14 for b in boundary])
        np.savez(path, **out)
        print(f"Swept {variant}", flush=True)


# ---------------------------------------------------------------- 6: calibration against one fixed policy


def calibration(base_path: str, positions: int, rollouts: int) -> None:
    models = Models(base_path)
    base = models.base
    late = dev_late_boards(positions, seed=1)
    afterstates = []
    for board in late:
        _, afterstate, _ = max(legal_afterstates(board), key=lambda m: m[2] + evaluate(base.weights, base.layout, m[1], board))
        afterstates.append(afterstate)
    jobs = [(k, j) for k in range(len(late)) for j in range(rollouts)]
    returns = np.array(list(POOL.map(lambda job: rollout(base.weights, base.layout, afterstates[job[0]], 1, 0.0, np.random.default_rng([7, job[0], job[1]]))[0], jobs)))
    actual = returns.reshape(len(late), rollouts).mean(axis=1)
    results = {"positions": len(late), "rollouts": rollouts, "mean greedy return": float(actual.mean()), "variants": {}}
    for variant in VARIANTS:
        weights, layout = models.get(variant)
        predicted = np.array([evaluate(weights, layout, a, b) for a, b in zip(afterstates, late)])
        error = predicted - actual
        results["variants"][variant] = {"mean predicted": float(predicted.mean()), "bias": float(error.mean()), "bias interval": float(1.96 * error.std(ddof=1) / np.sqrt(len(error))),
                                        "mean absolute error": float(np.abs(error).mean()), "rms error": float(np.sqrt(np.mean(error**2)))}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "calibration.json").write_text(json.dumps(results, indent=2) + "\n")


# ---------------------------------------------------------------- 7: rankings, on stage 1's positions and rollouts


def rankings(base_path: str) -> None:
    models = Models(base_path)
    d = np.load("experiments/late-game-diagnostics/actions.npz")
    boards = [np.uint64(b) for b in d["board"]]
    rewards, base_predicted = d["rewards"], d["predicted"]
    totals = rewards[:, :, None, None] + d["returns"]
    n_greedy = int(np.sum(~np.isnan(d["returns"][0, int(np.nanargmax(rewards[0])), 0])))
    greedy_totals = totals[:, :, 0, :n_greedy]  # [position, move, rollout] under the base's greedy play
    base_choice = np.nanargmax(rewards + base_predicted, axis=1)
    base_sorted = np.sort(np.where(np.isnan(base_predicted), -np.inf, rewards + base_predicted), axis=1)
    margin = base_sorted[:, -1] - base_sorted[:, -2]
    split_of = {(r["policy"], int(r["board"], 16)): r["split"] for r in read_rows(SAMPLES)}
    held_out = np.array([split_of.get((p, int(b))) != "train" for p, b in zip(d["policy"], d["board"])])
    results = {"positions": len(boards), "held-out positions": int(held_out.sum()), "greedy rollouts": n_greedy,
               "median base margin": float(np.median(margin[np.isfinite(margin)])), "variants": {}}
    for variant in VARIANTS:
        weights, layout = models.get(variant)
        values = np.full_like(base_predicted, np.nan)
        for p, board in enumerate(boards):
            for action, afterstate, _ in legal_afterstates(board):
                values[p, action] = evaluate(weights, layout, afterstate, board)
        choice = np.nanargmax(rewards + values, axis=1)
        changed = choice != base_choice
        rows = np.arange(len(boards))
        difference = greedy_totals[rows, choice] - greedy_totals[rows, base_choice]  # [position, rollout]
        mean = difference.mean(axis=1)
        se = difference.std(axis=1, ddof=1) / np.sqrt(n_greedy)
        correction = values - base_predicted
        second = np.argsort(np.where(np.isnan(base_predicted), -np.inf, rewards + base_predicted), axis=1)[:, -2]
        swing = np.abs(correction[rows, base_choice] - correction[rows, second])
        _, clear, loss = ranking_errors(greedy_totals, choice)
        stats = {}
        for label, mask in (("all", np.ones(len(boards), bool)), ("held-out", held_out)):
            c = changed & mask
            stats[label] = {
                "changed": float(c.sum() / mask.sum()),
                "changes better by > 2 SE": int(np.sum(c & (mean > 2 * se))), "changes worse by > 2 SE": int(np.sum(c & (mean < -2 * se))), "changes": int(c.sum()),
                "score change from changes, per position": float(mean[mask].mean()), "interval": float(1.96 * mean[mask].std(ddof=1) / np.sqrt(mask.sum())),
                "clearly worse than the best": float(clear[mask].mean()), "cross-validated loss": float(loss[mask].mean()),
                "median correction swing between the base's top two": float(np.nanmedian(swing[mask])),
                "swing larger than the base's margin": float(np.nanmean((swing > margin)[mask])),
            }
        results["variants"][variant] = stats
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "rankings.json").write_text(json.dumps(results, indent=2) + "\n")


# ---------------------------------------------------------------- report


def interval(values: np.ndarray) -> str:
    return f"{values.mean():+,.0f} ± {1.96 * values.std(ddof=1) / np.sqrt(len(values)):,.0f}"


def rate_interval(values: np.ndarray) -> str:
    return f"{values.mean():+.1%} ± {1.96 * values.std(ddof=1) / np.sqrt(len(values)):.1%}"


def report() -> None:
    path = OUTPUT / "equivalence.json"
    if path.exists():
        print("### 1. Equivalence\n")
        for name, checks in json.loads(path.read_text()).items():
            for check, passed in checks.items():
                print(f"- {name}: {check}: {'pass' if passed else 'FAIL'}")
    swept = {v: np.load(OUTPUT / f"sweep-{v}.npz") for v in VARIANTS if (OUTPUT / f"sweep-{v}.npz").exists()}
    if "baseline" in swept:
        b = swept["baseline"]
        print("\n### 3-5. Variants on the dev split (differences paired with the baseline, ± 95%)\n")
        print("| Variant | Greedy conversion | Δ | Greedy score added | Δ | Depth-3 conversion | Δ | Depth-3 score added | Δ | Full greedy games | Δ | 32768 in full games | "
              "Greedy continuations ending under 20k | Late afterstates valued below 0 | Boundary: makes the 16384 |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for v, d in swept.items():
            cells = []
            for name in ("greedy", "depth3"):
                made, base_made = (d[f"{name}_tiles"] >= 32768).mean(axis=1), (b[f"{name}_tiles"] >= 32768).mean(axis=1)
                added, base_added = d[f"{name}_scores"].mean(axis=1), b[f"{name}_scores"].mean(axis=1)
                cells += [f"{made.mean():.1%}", rate_interval(made - base_made) if v != "baseline" else "—", f"{added.mean():,.0f}", interval(added - base_added) if v != "baseline" else "—"]
            full = d["full_scores"].astype(float)
            cells += [f"{full.mean():,.0f}", interval(full - b["full_scores"]) if v != "baseline" else "—", f"{(d['full_tiles'] >= 32768).mean():.1%}",
                      f"{(d['greedy_scores'] < 20_000).mean():.1%}", f"{d['late_negative'].mean():.1%}", f"{d['boundary_merges'].mean():.0%}"]
            print(f"| {v} | " + " | ".join(cells) + " |")
        d = swept[next(iter(swept))]
        print(f"\n({d['greedy_scores'].shape[0]} dev first-16384 positions, {d['greedy_scores'].shape[1]} greedy and {d['depth3_scores'].shape[1]} depth-3 "
              f"continuations each; {d['full_scores'].size:,} full games; {d['late_legal'].size:,} late boards; {d['boundary_merges'].size} boundary boards)")
        print("\nLate afterstates valued below 0, by legal moves:\n")
        print("| Variant | 1 legal move | 2 | 3 | 4 |")
        print("|---|---:|---:|---:|---:|")
        for v, d in swept.items():
            print(f"| {v} | " + " | ".join(f"{d['late_negative'][d['late_legal'] == k].mean():.1%}" if (d["late_legal"] == k).any() else "" for k in (1, 2, 3, 4)) + " |")
    path = OUTPUT / "calibration.json"
    if path.exists():
        c = json.loads(path.read_text())
        print(f"\n### 6. Calibration against the base's greedy play: {c['positions']} dev late afterstates the base chose, {c['rollouts']} greedy rollouts each "
              f"(mean return {c['mean greedy return']:,.0f})\n")
        print("| Variant | Mean predicted | Bias | Mean absolute error | RMS error |")
        print("|---|---:|---:|---:|---:|")
        for v, s in c["variants"].items():
            print(f"| {v} | {s['mean predicted']:,.0f} | {s['bias']:+,.0f} ± {s['bias interval']:,.0f} | {s['mean absolute error']:,.0f} | {s['rms error']:,.0f} |")
    path = OUTPUT / "rankings.json"
    if path.exists():
        r = json.loads(path.read_text())
        print(f"\n### 7. Rankings on stage 1's {r['positions']} late positions ({r['held-out positions']} from dev or test games), judged by {r['greedy rollouts']} "
              f"greedy rollouts of every move; the base's median margin between its top two moves is {r['median base margin']:,.0f}\n")
        print("| Variant | Moves changed | Changes better by > 2 SE | Worse by > 2 SE | Score change from changes, per position | Clearly worse than the best | "
              "Cross-validated loss | Median correction swing between the base's top two | Swing larger than the base's margin |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for v, s in r["variants"].items():
            a = s["all"]
            swing, larger = a["median correction swing between the base's top two"], a["swing larger than the base's margin"]
            print(f"| {v} | {a['changed']:.0%} ({a['changes']}) | {a['changes better by > 2 SE']} | {a['changes worse by > 2 SE']} | "
                  f"{a['score change from changes, per position']:+,.0f} ± {a['interval']:,.0f} | {a['clearly worse than the best']:.0%} | "
                  f"{a['cross-validated loss']:+,.0f} | {swing:,.0f} | {larger:.0%} |")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify stage 3's corrected network")
    parser.add_argument("step", choices=("equivalence", "sweep", "calibration", "rankings", "report"))
    parser.add_argument("base", nargs="?")
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS), help="sweep: variants to run (default: all)")
    parser.add_argument("--greedy-runs", type=int, default=32, help="sweep: greedy continuations per position (default: 32)")
    parser.add_argument("--depth3-runs", type=int, default=2, help="sweep: depth-3 continuations per position (default: 2)")
    args = parser.parse_args()
    if args.step == "equivalence":
        equivalence(args.base)
    elif args.step == "sweep":
        sweep(args.base, args.variants, args.greedy_runs, args.depth3_runs)
    elif args.step == "calibration":
        calibration(args.base, 1000, 8)
    elif args.step == "rankings":
        rankings(args.base)
    report()


if __name__ == "__main__":
    main()
