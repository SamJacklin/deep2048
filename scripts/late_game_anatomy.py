"""Stage 7: why does the 2M network convert so few first 16384s into a 32768? A diagnostic: no weights change.

Usage (scripts/run_late_game_anatomy.sh runs them all in order, resuming):
    uv run python scripts/late_game_anatomy.py manifest
    uv run python scripts/late_game_anatomy.py p1-greedy      # Part 1: 64 greedy continuations from each dev entry position
    uv run python scripts/late_game_anatomy.py p1-depth3      # 8 depth-3 continuations from a stratified 1,000 of them
    uv run python scripts/late_game_anatomy.py p1-report
    uv run python scripts/late_game_anatomy.py p2-trajectories --phase pilot   # Part 2: greedy games from entry positions
    uv run python scripts/late_game_anatomy.py p2-run --phase pilot            # rescue policies from points along them
    uv run python scripts/late_game_anatomy.py p2-report --phase pilot
    uv run python scripts/late_game_anatomy.py p2-plan        # the expansion, chosen by a fixed rule from the pilot
    uv run python scripts/late_game_anatomy.py p2-trajectories --phase expansion
    uv run python scripts/late_game_anatomy.py p2-run --phase expansion
    uv run python scripts/late_game_anatomy.py p2-report --phase final

Positions are stage 6's dev pool: 5,000 genuine first-16384 boards from distinct greedy games of the 2M network (seeds
30,000,000 on). Its test pool is never read. Seeds are kept apart: games made the positions; continuation j of Part 1
uses rng([game seed, j]) (so j = 0, 1 repeat stage 6's control evaluation exactly); Part 2's trajectories use
rng([game seed, 9000]); and its continuations rng([game seed, move, j, 77]), the same for every policy, so policies are
paired. Big arrays go to output/late-game-anatomy (outside git); compact reports to experiments/late-game-anatomy.
SMOKE=1 shrinks everything for a quick end-to-end check.
"""

import argparse
import csv
import json
import os
import platform
import resource
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from deep_2048.agents.ntuple import NTupleNetwork  # noqa: E402
from deep_2048.agents.ntuple.anatomy import play_mixed, record_trajectory  # noqa: E402
from deep_2048.bench.cache import file_sha256  # noqa: E402
from deep_2048.game import bitboard as bb  # noqa: E402

SMOKE = os.environ.get("SMOKE") == "1"
BASE = Path("checkpoints/ntuple/decision-8x6-tc-reset/games_2000000.npz")
POSITIONS = Path("experiments/late-network/positions.csv")
STAGE6_CONTROL = Path("experiments/late-network/evaluations/control.npz")
OUTPUT = Path("output/late-game-anatomy-smoke" if SMOKE else "output/late-game-anatomy")
REPORTS = Path("experiments/late-game-anatomy") / ("smoke" if SMOKE else "")

P1_POSITIONS = 60 if SMOKE else 5_000
P1_GREEDY = 4 if SMOKE else 64
P1_DEPTH3_POSITIONS = 12 if SMOKE else 1_000
P1_DEPTH3 = 2 if SMOKE else 8
CHUNK = 20 if SMOKE else 250

# Part 2. Policies: (search depth, decisions searched (-1: all), probability cutoff)
POLICIES = {"A": (1, 0, 0.0), "B": (3, -1, 0.0), "C": (3, 200, 0.0), "D": (5, 200, 0.01)}
POLICY_NAMES = {"A": "greedy", "B": "depth 3 from here", "C": "depth 3 for 200 moves, then greedy", "D": "pruned depth 5 (cutoff 0.01) for 200 moves, then greedy"}
PILOT_PER_OUTCOME = 2 if SMOKE else 20
EXPANSION_PER_OUTCOME = 2 if SMOKE else 20
PILOT_CONTINUATIONS = 2 if SMOKE else 8
EXPANDED_CONTINUATIONS = 4 if SMOKE else 32
EVERY = 250
BEFORE_END = (200, 50)
BUDGET_HOURS = float(os.environ.get("BUDGET_HOURS", "10"))
TIME_BINS = ((0, 0, "at the first 16384"), (1, 500, "1–500 moves after"), (501, 1500, "501–1,500"), (1501, 3000, "1,501–3,000"), (3001, 10**9, "over 3,000"))
REMAINING_BINS = ((2001, 10**9, "over 2,000 moves left"), (501, 2000, "501–2,000 left"), (201, 500, "201–500 left"), (0, 200, "200 or fewer left"))
EMPTY_BINS = ((0, 1, "0–1 empty"), (2, 3, "2–3 empty"), (4, 6, "4–6 empty"), (7, 16, "7+ empty"))
POOL = ThreadPoolExecutor(max_workers=os.cpu_count())


def load_base() -> NTupleNetwork:
    return NTupleNetwork.load(BASE)[0]


def dev_rows() -> list[dict]:
    with open(POSITIONS) as f:
        rows = [r for r in csv.DictReader(f) if r["pool"] == "dev" and r["policy"] == "greedy" and r["network"] == "2m"]
    return rows[:P1_POSITIONS]


def board_of(row: dict) -> np.uint64:
    return np.uint64(int(row["board"], 16))


def log_cost(step: str, started: float, cpu: float, **counts: int) -> None:
    """Append a step's compute to the cost log: wall and CPU time, peak memory and its counts."""
    REPORTS.mkdir(parents=True, exist_ok=True)
    row = {"step": step, "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "wall_s": round(time.perf_counter() - started, 1),
           "cpu_s": round(time.process_time() - cpu, 1), "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 if sys.platform == "linux" else 1024) / 1024, 1)} | counts
    with open(REPORTS / "costs.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


# ---------------------------------------------------------------- features


def grid(board: np.uint64) -> np.ndarray:
    return np.array([(int(board) >> (4 * i)) & 15 for i in range(16)]).reshape(4, 4)


def chain_length(g: np.ndarray, r: int, c: int) -> int:
    """The longest path from (r, c) through neighbours whose exponents fall by exactly one each step."""
    best = 1
    for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < 4 and 0 <= cc < 4 and g[rr, cc] > 0 and g[rr, cc] == g[r, c] - 1:
            best = max(best, 1 + chain_length(g, rr, cc))
    return best


FEATURES = ("top tile in a corner", "top tile on an edge", "empty cells", "2nd largest (exponent)", "2nd largest: distance from top",
            "3rd largest (exponent)", "3rd largest: distance from top", "monotonicity", "legal moves", "adjacent equal pairs",
            "chain from top tile", "tiles of 256+ besides the top", "other tiles' sum / 1,000")


def features(board: np.uint64) -> np.ndarray:
    """A small interpretable description of a board, in the order of FEATURES."""
    g = grid(board)
    flat = g.ravel()
    top = int(np.argmax(flat))
    r, c = divmod(top, 4)
    corner = r in (0, 3) and c in (0, 3)
    edge = (r in (0, 3) or c in (0, 3)) and not corner
    order = np.argsort(-flat, kind="stable")
    second, third = order[1], order[2]
    distance = lambda i: abs(i // 4 - r) + abs(i % 4 - c)  # noqa: E731
    lines = [g[i, :] for i in range(4)] + [g[:, i] for i in range(4)]
    mono = np.mean([max(np.sum(line[:-1] >= line[1:]), np.sum(line[:-1] <= line[1:])) / 3 for line in lines])
    legal = sum(bb.move(board, a)[0] != board for a in range(bb.NUM_ACTIONS))
    pairs = sum(int(g[i, j] > 0 and g[i, j] == g[i, j + 1]) for i in range(4) for j in range(3)) + sum(int(g[i, j] > 0 and g[i, j] == g[i + 1, j]) for i in range(3) for j in range(4))
    others = np.delete(flat, top)
    return np.array([corner, edge, np.sum(flat == 0), flat[second], distance(second), flat[third], distance(third), mono, legal, pairs,
                     chain_length(g, r, c), np.sum(others >= 8), np.sum(np.where(others > 0, 2.0 ** others, 0)) / 1000], dtype=float)


# ---------------------------------------------------------------- Part 1


def continuations(network: NTupleNetwork, rows: list[dict], n: int, depth: int) -> np.ndarray:
    """(score added, max tile, moves) for continuations j < n from each row, j drawn from rng([game seed, j])."""
    jobs = [(r, j) for r in rows for j in range(n)]
    out = POOL.map(lambda job: play_mixed(network.weights, network.layout, board_of(job[0]), depth, -1, 0.0, np.random.default_rng([int(job[0]["seed"]), job[1]]))[:3], jobs)
    return np.array(list(out), dtype=np.int64).reshape(len(rows), n, 3)


def run_chunks(name: str, rows: list[dict], n: int, depth: int) -> None:
    network = load_base()
    folder = OUTPUT / name
    folder.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(rows), CHUNK):
        path = folder / f"chunk_{start:05d}.npz"
        if path.exists():
            continue
        began, cpu = time.perf_counter(), time.process_time()
        part = rows[start : start + CHUNK]
        played = continuations(network, part, n, depth)
        np.savez(path, seeds=np.array([int(r["seed"]) for r in part]), scores=played[..., 0], tiles=played[..., 1], moves=played[..., 2])
        log_cost(f"{name} positions {start:,}-{start + len(part) - 1:,}", began, cpu, continuations=played.shape[0] * n, moves=int(played[..., 2].sum()))


def load_chunks(name: str) -> dict[str, np.ndarray]:
    parts = [dict(np.load(p)) for p in sorted((OUTPUT / name).glob("chunk_*.npz"))]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]} if parts else {}


def depth3_subset() -> list[dict]:
    """1,000 positions stratified by where the 16384 sits and by empty cells, chosen from the boards alone (before any outcome)."""
    path = REPORTS / "depth3-subset.json"
    rows = dev_rows()
    if path.exists():
        chosen = set(json.loads(path.read_text()))
        return [r for r in rows if int(r["seed"]) in chosen]
    f = np.array([features(board_of(r)) for r in rows])
    place = np.where(f[:, 0] == 1, 0, np.where(f[:, 1] == 1, 1, 2))
    empties = np.digitize(f[:, 2], np.quantile(f[:, 2], [1 / 3, 2 / 3]))
    strata = place * 3 + empties
    rng = np.random.default_rng(7007)
    picked: list[int] = []
    for s in np.unique(strata):
        members = np.flatnonzero(strata == s)
        take = min(len(members), max(min(30, len(members)), round(P1_DEPTH3_POSITIONS * len(members) / len(rows))))
        picked += list(rng.choice(members, take, replace=False))
    picked = sorted(rng.choice(picked, P1_DEPTH3_POSITIONS, replace=False)) if len(picked) > P1_DEPTH3_POSITIONS else sorted(picked)
    REPORTS.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([int(rows[i]["seed"]) for i in picked]) + "\n")
    return [rows[i] for i in picked]


def p1_greedy() -> None:
    run_chunks("p1-greedy", dev_rows(), P1_GREEDY, 1)
    check_against_stage6()


def p1_depth3() -> None:
    run_chunks("p1-depth3", depth3_subset(), P1_DEPTH3, 3)


def check_against_stage6() -> None:
    """Continuations 0 and 1 replay stage 6's control evaluation exactly: the same network, positions and spawns."""
    if not STAGE6_CONTROL.exists():
        return
    control, mine = dict(np.load(STAGE6_CONTROL)), load_chunks("p1-greedy")
    n = len(mine["scores"])
    same = bool(np.array_equal(control["greedy_greedy_2m_scores"][:n, :2], mine["scores"][:, :2]) and np.array_equal(control["greedy_greedy_2m_tiles"][:n, :2], mine["tiles"][:, :2]))
    (REPORTS / "stage6-replay-check.json").write_text(json.dumps({"positions": n, "continuations 0 and 1 identical to stage 6's control": same}, indent=2) + "\n")
    if not same:
        raise SystemExit("Part 1's first two continuations differ from stage 6's control evaluation: the setup isn't the same")


# ---------------------------------------------------------------- statistics


def beta_binomial_fit(k: np.ndarray, n: int) -> tuple[float, float]:
    """Maximum-likelihood mean and intraclass correlation (rho) of a beta-binomial for successes k out of n per position,
    over a fine grid. rho is the share of the variance in single outcomes that's between positions."""
    hist = np.bincount(k, minlength=n + 1).astype(float)
    mus = np.linspace(0.005, 0.995, 199)
    rhos = np.concatenate([[1e-6], np.geomspace(1e-4, 0.9, 200)])
    m, r = np.meshgrid(mus, rhos, indexing="ij")
    a, b = (m * (1 - r) / r).ravel(), ((1 - m) * (1 - r) / r).ravel()
    t = np.arange(n)
    # S_x(j) = sum_{i<j} log(x + i), for j = 0..n
    s = lambda x: np.concatenate([np.zeros((x.size, 1)), np.cumsum(np.log(x[:, None] + t[None, :]), axis=1)], axis=1)  # noqa: E731
    ks = np.arange(n + 1)
    ll = (s(a)[:, ks] + s(b)[:, n - ks]) @ hist - hist.sum() * s(a + b)[:, n]
    best = int(np.argmax(ll))
    return float(m.ravel()[best]), float(r.ravel()[best])


def position_effect(tiles: np.ndarray, scores: np.ndarray, reps: int = 200) -> dict:
    """How much the starting position matters: beta-binomial rho for making 32768, and the between-position share of the
    variance in score added (one-way random effects), each with a bootstrap 95% interval over positions."""
    made = (tiles >= 32768).sum(axis=1)
    n = tiles.shape[1]

    def score_share(s: np.ndarray) -> float:
        within = s.var(axis=1, ddof=1).mean()
        between = max(s.mean(axis=1).var(ddof=1) - within / n, 0.0)
        return between / (between + within)

    mu, rho = beta_binomial_fit(made, n)
    rng = np.random.default_rng(0)
    boot = [(beta_binomial_fit(made[i], n)[1], score_share(scores[i])) for i in (rng.integers(0, len(made), len(made)) for _ in range(reps))]
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    return {"positions": len(made), "continuations each": n, "makes 32768": float(made.sum() / made.size), "beta-binomial mean": mu,
            "rho (32768)": rho, "rho 95%": [float(lo[0]), float(hi[0])], "SD of true conversion across positions": float(np.sqrt(mu * (1 - mu) * rho)),
            "mean score added": float(scores.mean()), "between-position share of score variance": score_share(scores), "share 95%": [float(lo[1]), float(hi[1])]}


def ridge_logistic(x: np.ndarray, k: np.ndarray, n: int, lam: float) -> np.ndarray:
    """L2-penalised logistic regression on binomial counts (k of n), by Newton's method; the intercept isn't penalised."""
    xb = np.column_stack([np.ones(len(x)), x])
    w = np.zeros(xb.shape[1])
    penalty = lam * np.eye(xb.shape[1])
    penalty[0, 0] = 0
    for _ in range(50):
        p = 1 / (1 + np.exp(-xb @ w))
        gradient = xb.T @ (k - n * p) - penalty @ w
        hessian = xb.T @ (xb * (n * p * (1 - p))[:, None]) + penalty
        step = np.linalg.solve(hessian, gradient)
        w += step
        if np.abs(step).max() < 1e-8:
            break
    return w


def folds_of(seeds: np.ndarray, k: int = 5) -> np.ndarray:
    """Fold of each position by its source game, so no game is split across folds."""
    games = np.unique(seeds)
    order = np.random.default_rng(11).permutation(len(games))
    fold_of_game = dict(zip(games[order], np.arange(len(games)) % k))
    return np.array([fold_of_game[s] for s in seeds])


def out_of_fold(x: np.ndarray, k: np.ndarray, n: int, seeds: np.ndarray, lam: float) -> np.ndarray:
    folds, predicted = folds_of(seeds), np.zeros(len(k))
    for f in np.unique(folds):
        train, test = folds != f, folds == f
        mean, sd = x[train].mean(axis=0), x[train].std(axis=0) + 1e-9
        w = ridge_logistic((x[train] - mean) / sd, k[train], n, lam)
        predicted[test] = 1 / (1 + np.exp(-(np.column_stack([np.ones(test.sum()), (x[test] - mean) / sd]) @ w)))
    return predicted


def scores_of(p: np.ndarray, k: np.ndarray, n: int) -> tuple[float, float]:
    """Brier score and log loss per continuation."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(np.sum(k * (1 - p) ** 2 + (n - k) * p**2) / (len(k) * n)), float(-np.sum(k * np.log(p) + (n - k) * np.log(1 - p)) / (len(k) * n))


def interval(values: np.ndarray, reps: int = 2000, rate: bool = False) -> str:
    rng = np.random.default_rng(1)
    boot = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(reps)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return f"{values.mean():+.1%} [{lo:+.1%}, {hi:+.1%}]" if rate else f"{values.mean():+,.0f} [{lo:+,.0f}, {hi:+,.0f}]"


def p1_report() -> None:
    greedy = load_chunks("p1-greedy")
    rows = {int(r["seed"]): r for r in dev_rows()}
    x = np.array([features(board_of(rows[s])) for s in greedy["seeds"]])
    n = greedy["tiles"].shape[1]
    made = (greedy["tiles"] >= 32768).sum(axis=1)
    lines = ["# Stage 7, Part 1: does the entry position matter?\n", f"Generated {time.strftime('%Y-%m-%d %H:%M')} from `p1-greedy`" + (" and `p1-depth3`" if (OUTPUT / "p1-depth3").exists() else "") + ".\n"]

    lines.append("## 6.1 Position-dependent conversion\n")
    effects = {"greedy": position_effect(greedy["tiles"], greedy["scores"])}
    depth3 = load_chunks("p1-depth3")
    if depth3:
        effects["depth 3"] = position_effect(depth3["tiles"], depth3["scores"])
    lines.append("| Policy | Positions × continuations | Makes 32768 | rho (32768) [95%] | SD of true conversion | Mean score added | Between-position share of score variance [95%] |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, e in effects.items():
        lines.append(f"| {name} | {e['positions']:,} × {e['continuations each']} | {e['makes 32768']:.1%} | {e['rho (32768)']:.3f} [{e['rho 95%'][0]:.3f}, {e['rho 95%'][1]:.3f}] | "
                     f"{e['SD of true conversion across positions']:.1%} | {e['mean score added']:,.0f} | {e['between-position share of score variance']:.1%} "
                     f"[{e['share 95%'][0]:.1%}, {e['share 95%'][1]:.1%}] |")
    if depth3:
        greedy_rate = dict(zip(greedy["seeds"].tolist(), (greedy["tiles"] >= 32768).mean(axis=1)))
        d = (depth3["tiles"] >= 32768).mean(axis=1)
        g = np.array([greedy_rate[s] for s in depth3["seeds"].tolist()])
        lines.append(f"\nOn the depth-3 subset's {len(d):,} positions, greedy makes 32768 in {g.mean():.1%} and depth 3 in {d.mean():.1%}; "
                     f"the per-position rates correlate at r = {np.corrcoef(g, d)[0, 1]:.2f} (both noisy, so this understates the true correlation).")

    lines.append("\n## 6.2 Predictive board features\n")
    lines.append("Out-of-fold predictions from L2-penalised logistic regression on all features, 5 folds grouped by source game, against a constant predictor fitted on each training fold. Scores are per continuation; lower is better.\n")
    folds = folds_of(greedy["seeds"])
    constant = np.zeros(len(made))
    for f in np.unique(folds):
        constant[folds == f] = made[folds != f].sum() / ((folds != f).sum() * n)
    base_brier, base_ll = scores_of(constant, made, n)
    lines.append("| Model | Penalty | Brier | Log loss | Brier skill vs constant | SD of predictions |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    lines.append(f"| Constant | — | {base_brier:.5f} | {base_ll:.5f} | 0 | 0 |")
    predictions = {}
    for lam in (0.1, 1.0, 10.0, 100.0):
        p = out_of_fold(x, made, n, greedy["seeds"], lam)
        predictions[lam] = p
        brier, ll = scores_of(p, made, n)
        lines.append(f"| All features | {lam:g} | {brier:.5f} | {ll:.5f} | {1 - brier / base_brier:+.2%} | {p.std():.2%} |")
    p = predictions[1.0]
    true_sd = effects["greedy"]["SD of true conversion across positions"]
    lines.append(f"\nAt penalty 1, the predictions' spread is {p.std():.2%} against an estimated {true_sd:.2%} spread in true conversion: "
                 f"the features capture roughly {min(p.var() / max(true_sd**2, 1e-12), 9.99):.0%} of the between-position variance.\n")
    lines.append("Calibration (penalty 1), by tenths of predicted conversion:\n")
    lines.append("| Tenth | Predicted | Observed |")
    lines.append("|---:|---:|---:|")
    tenth = np.minimum((np.argsort(np.argsort(p)) * 10) // len(p), 9)
    for t in range(10):
        lines.append(f"| {t + 1} | {p[tenth == t].mean():.1%} | {made[tenth == t].sum() / ((tenth == t).sum() * n):.1%} |")

    lines.append("\nEach feature on its own: conversion by the feature's level, over all continuations.\n")
    lines.append("| Feature | Level | Positions | Makes 32768 | Mean score added |")
    lines.append("|---|---|---:|---:|---:|")
    place = np.where(x[:, 0] == 1, "corner", np.where(x[:, 1] == 1, "edge", "interior"))
    for level in ("corner", "edge", "interior"):
        m = place == level
        if m.any():
            lines.append(f"| Top tile | {level} | {m.sum():,} | {made[m].sum() / (m.sum() * n):.1%} | {greedy['scores'][m].mean():,.0f} |")
    for i, name in enumerate(FEATURES[2:], start=2):
        cuts = np.unique(np.quantile(x[:, i], [1 / 3, 2 / 3]))
        level = np.digitize(x[:, i], cuts)
        for v in np.unique(level):
            m = level == v
            lo, hi = x[m, i].min(), x[m, i].max()
            lines.append(f"| {name} | {lo:g}–{hi:g} | {m.sum():,} | {made[m].sum() / (m.sum() * n):.1%} | {greedy['scores'][m].mean():,.0f} |")

    lines.append("\n## 6.3 Predicted best against predicted worst tenth (out of fold, penalty 1)\n")
    best, worst = tenth == 9, tenth == 0
    rates, added = made / n, greedy["scores"].mean(axis=1)
    rng = np.random.default_rng(2)
    diff = [rates[best][rng.integers(0, best.sum(), best.sum())].mean() - rates[worst][rng.integers(0, worst.sum(), worst.sum())].mean() for _ in range(2000)]
    lo, hi = np.percentile(diff, [2.5, 97.5])
    lines.append("| Group | Positions | Makes 32768 | Mean score added |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| Predicted best 10% | {best.sum():,} | {rates[best].mean():.1%} | {added[best].mean():,.0f} |")
    lines.append(f"| Predicted worst 10% | {worst.sum():,} | {rates[worst].mean():.1%} | {added[worst].mean():,.0f} |")
    lines.append(f"\nDifference in conversion: {rates[best].mean() - rates[worst].mean():+.1%} [{lo:+.1%}, {hi:+.1%}].\n")
    lines.append("| Feature | Best 10% | Worst 10% | All |")
    lines.append("|---|---:|---:|---:|")
    for i, name in enumerate(FEATURES):
        lines.append(f"| {name} | {x[best, i].mean():.2f} | {x[worst, i].mean():.2f} | {x[:, i].mean():.2f} |")
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "part1.md").write_text("\n".join(lines) + "\n")
    (REPORTS / "part1.json").write_text(json.dumps(effects, indent=2) + "\n")
    print("\n".join(lines))


# ---------------------------------------------------------------- Part 2


def trajectories_index(phase: str) -> Path:
    return REPORTS / f"p2-trajectories-{phase}.json"


def p2_trajectories(phase: str) -> None:
    """Greedy games from entry positions, in seed order, until there are enough that made a 32768 and enough that didn't.
    The expansion skips every game the pilot looked at, kept or not."""
    path = trajectories_index(phase)
    if path.exists():
        return
    per_outcome = PILOT_PER_OUTCOME if phase == "pilot" else EXPANSION_PER_OUTCOME
    skip = set()
    if phase == "expansion":
        skip = set(json.loads(trajectories_index("pilot").read_text())["examined"])
    network = load_base()
    folder = OUTPUT / "p2-trajectories"
    folder.mkdir(parents=True, exist_ok=True)
    kept, examined = {"success": [], "failure": []}, []
    began, cpu = time.perf_counter(), time.process_time()
    rows = [r for r in dev_rows() if int(r["seed"]) not in skip]
    for start in range(0, len(rows), 64):
        batch = rows[start : start + 64]
        games = list(POOL.map(lambda r: record_trajectory(network.weights, network.layout, board_of(r), np.random.default_rng([int(r["seed"]), 9000])), batch))
        for r, (boards, scores, tile, total) in zip(batch, games):
            examined.append(int(r["seed"]))
            outcome = "success" if tile >= 32768 else "failure"
            if len(kept[outcome]) < per_outcome:
                np.savez(folder / f"{r['seed']}.npz", boards=boards, scores=scores)
                kept[outcome].append({"seed": int(r["seed"]), "outcome": outcome, "moves": len(boards), "score": int(total), "tile": int(tile)})
        if all(len(v) >= per_outcome for v in kept.values()):
            break
    path.write_text(json.dumps({"kept": kept["success"] + kept["failure"], "examined": examined, "examined count": len(examined)}, indent=2) + "\n")
    log_cost(f"p2-trajectories {phase}", began, cpu, games=len(examined), kept=len(kept["success"]) + len(kept["failure"]))


def points_of(trajectory: dict, boards: np.ndarray) -> list[dict]:
    """The decisions replayed: the first 16384, every EVERY moves, the first board with at most one empty cell, and for
    failures 200 and 50 moves before the end. Each records moves since the 16384 and (looking back) moves remaining."""
    length = len(boards)
    marks: dict[int, str] = {m: "every 250" for m in range(0, length, EVERY)}
    marks[0] = "first 16384"
    crowded = next((m for m in range(1, length) if bb.count_empty(boards[m]) <= 1), None)
    if crowded is not None:
        marks.setdefault(crowded, "first with ≤1 empty")
    if trajectory["outcome"] == "failure":
        for before in BEFORE_END:
            if length - before > 0:
                marks.setdefault(length - before, f"{before} before the end")
    return [{"move": m, "kind": k, "remaining": length - m, "empty": int(bb.count_empty(boards[m])), "legal": int(sum(bb.move(boards[m], a)[0] != boards[m] for a in range(4)))}
            for m, k in sorted(marks.items())]


def time_bin(move: int) -> int:
    return next(i for i, (lo, hi, _) in enumerate(TIME_BINS) if lo <= move <= hi)


def p2_run(phase: str) -> None:
    """Play each policy from each point of each trajectory. The pilot plays PILOT_CONTINUATIONS per point; the expansion
    follows p2-plan: more continuations at the pilot's informative points, and the new trajectories' informative points."""
    network = load_base()
    folder = OUTPUT / "p2-continuations"
    folder.mkdir(parents=True, exist_ok=True)
    plan = json.loads((REPORTS / "p2-plan.json").read_text()) if phase == "expansion" else None
    trajectories = json.loads(trajectories_index("pilot").read_text())["kept"]
    if phase == "expansion":
        trajectories += json.loads(trajectories_index("expansion").read_text())["kept"]
    for t in trajectories:
        boards = np.load(OUTPUT / "p2-trajectories" / f"{t['seed']}.npz")["boards"]
        points = points_of(t, boards)
        for policy, (depth, deep, cutoff) in POLICIES.items():
            if phase == "pilot":
                target, chosen = PILOT_CONTINUATIONS, points
            else:
                if policy not in plan["policies"]:
                    continue
                groups = {tuple(g) for g in plan["groups"]}
                chosen = [p for p in points if (t["outcome"], time_bin(p["move"])) in groups]
                target = plan["continuations"]
            path = folder / f"{t['seed']}-{policy}.npz"
            done = dict(np.load(path)) if path.exists() else {"moves_at": np.zeros(0, np.int64)}
            have = {(int(m), int(j)) for m, j in zip(done.get("moves_at", []), done.get("j", []))}
            jobs = [(p["move"], j) for p in chosen for j in range(target) if (p["move"], j) not in have]
            if not jobs:
                continue
            began, cpu = time.perf_counter(), time.process_time()
            played = np.array(list(POOL.map(lambda job: play_mixed(network.weights, network.layout, np.uint64(boards[job[0]]), depth, deep, cutoff,
                                                                    np.random.default_rng([t["seed"], job[0], job[1], 77])), jobs)), dtype=np.int64)
            new = {"moves_at": np.array([m for m, _ in jobs]), "j": np.array([j for _, j in jobs]), "scores": played[:, 0], "tiles": played[:, 1],
                   "moves": played[:, 2], "nodes": played[:, 3], "cpu_s": np.full(len(jobs), (time.process_time() - cpu) / len(jobs))}
            merged = {k: np.concatenate([done[k], new[k]]) if k in done else new[k] for k in new}
            np.savez(path, **merged)
            log_cost(f"p2-run {phase} {t['seed']} {policy}", began, cpu, continuations=len(jobs), moves=int(played[:, 2].sum()), nodes=int(played[:, 3].sum()))


def p2_table() -> list[dict]:
    """One row per (trajectory, point, continuation j) where every policy that played that point has played j."""
    rows = []
    for index in ("pilot", "expansion"):
        if not trajectories_index(index).exists():
            continue
        for t in json.loads(trajectories_index(index).read_text())["kept"]:
            boards = np.load(OUTPUT / "p2-trajectories" / f"{t['seed']}.npz")["boards"]
            played = {p: dict(np.load(OUTPUT / "p2-continuations" / f"{t['seed']}-{p}.npz")) for p in POLICIES if (OUTPUT / "p2-continuations" / f"{t['seed']}-{p}.npz").exists()}
            if "A" not in played:
                continue
            for point in points_of(t, boards):
                for policy, d in played.items():
                    at = d["moves_at"] == point["move"]
                    for j, score, tile, cpu, nodes in zip(d["j"][at], d["scores"][at], d["tiles"][at], d["cpu_s"][at], d["nodes"][at]):
                        rows.append(point | {"seed": t["seed"], "outcome": t["outcome"], "policy": policy, "j": int(j), "score": int(score), "made": int(tile >= 32768),
                                             "cpu_s": float(cpu), "nodes": int(nodes)})
    return rows


def uplifts(rows: list[dict]) -> dict:
    """Per policy and point, the mean paired difference from greedy over the continuations both played."""
    a = {(r["seed"], r["move"], r["j"]): r for r in rows if r["policy"] == "A"}
    points: dict = {}
    for r in rows:
        if r["policy"] == "A" or (r["seed"], r["move"], r["j"]) not in a:
            continue
        base = a[(r["seed"], r["move"], r["j"])]
        key = (r["policy"], r["seed"], r["move"])
        p = points.setdefault(key, {"seed": r["seed"], "move": r["move"], "outcome": r["outcome"], "remaining": r["remaining"], "empty": r["empty"],
                                    "kind": r["kind"], "d_score": [], "d_made": [], "cpu": [], "nodes": [], "base_made": []})
        p["d_score"].append(r["score"] - base["score"])
        p["d_made"].append(r["made"] - base["made"])
        p["base_made"].append(base["made"])
        p["cpu"].append(r["cpu_s"])
        p["nodes"].append(r["nodes"])
    return points


def cluster_interval(points: list[dict], field: str, reps: int = 2000) -> tuple[float, float, float]:
    """Mean of the points' mean paired differences, with a 95% bootstrap interval resampling whole trajectories."""
    by_seed: dict = {}
    for p in points:
        by_seed.setdefault(p["seed"], []).append(np.mean(p[field]))
    seeds = list(by_seed)
    mean = float(np.mean([v for s in seeds for v in by_seed[s]]))
    if len(seeds) < 2:
        return mean, float("nan"), float("nan")
    rng = np.random.default_rng(3)
    boot = []
    for _ in range(reps):
        values = [v for s in rng.choice(seeds, len(seeds)) for v in by_seed[s]]
        boot.append(np.mean(values))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return mean, float(lo), float(hi)


def group_rows(points: dict, policy: str, outcome: str | None, selector, bins) -> list[tuple[str, list[dict]]]:
    chosen = [p for (pol, _, _), p in points.items() if pol == policy and (outcome is None or p["outcome"] == outcome)]
    return [(label, [p for p in chosen if lo <= selector(p) <= hi]) for lo, hi, label in bins]


def p2_report(phase: str) -> None:
    rows = p2_table()
    points = uplifts(rows)
    natural = json.loads((REPORTS / "part1.json").read_text())["greedy"]["makes 32768"] if (REPORTS / "part1.json").exists() else None
    lines = [f"# Stage 7, Part 2: when can failing games be rescued? ({phase})\n", f"Generated {time.strftime('%Y-%m-%d %H:%M')}.\n"]
    lines.append("Policies: " + "; ".join(f"**{k}** {v}" for k, v in POLICY_NAMES.items()) + ".\n")
    lines.append("Each cell is the mean paired difference from greedy (A) over points, then continuations, with a 95% interval from resampling whole trajectories. "
                 "Moves remaining are retrospective (the agent can't know them).\n")
    trajectories = {r["seed"]: r["outcome"] for r in rows}
    lines.append(f"Trajectories: {sum(v == 'success' for v in trajectories.values())} that made 32768 (sampled up to it) and {sum(v == 'failure' for v in trajectories.values())} that didn't. "
                 f"Points: {len({(r['seed'], r['move']) for r in rows}):,}. Continuations: {len(rows):,}.\n")
    for policy in [p for p in POLICIES if p != "A"]:
        lines.append(f"\n## {policy}: {POLICY_NAMES[policy]}\n")
        for title, outcome, selector, bins in (
            ("By moves since the first 16384, failed trajectories", "failure", lambda p: p["move"], TIME_BINS),
            ("By moves since the first 16384, successful trajectories", "success", lambda p: p["move"], TIME_BINS),
            ("By moves remaining, failed trajectories", "failure", lambda p: p["remaining"], REMAINING_BINS),
            ("By empty cells, failed trajectories", "failure", lambda p: p["empty"], EMPTY_BINS),
        ):
            lines.append(f"\n**{title}**\n")
            lines.append("| Group | Points | Trajectories | Greedy makes 32768 | Δ makes 32768 [95%] | Δ score added [95%] | CPU s per continuation | Chance nodes per continuation |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
            for label, members in group_rows(points, policy, outcome, selector, bins):
                if not members:
                    continue
                dm, dml, dmh = cluster_interval(members, "d_made")
                ds, dsl, dsh = cluster_interval(members, "d_score")
                lines.append(f"| {label} | {len(members)} | {len({m['seed'] for m in members})} | {np.mean([np.mean(m['base_made']) for m in members]):.1%} | "
                             f"{dm:+.1%} [{dml:+.1%}, {dmh:+.1%}] | {ds:+,.0f} [{dsl:+,.0f}, {dsh:+,.0f}] | {np.mean([np.mean(m['cpu']) for m in members]):.2f} | "
                             f"{np.mean([np.mean(m['nodes']) for m in members]):,.0f} |")
        if natural is not None:
            first = [p for (pol, _, _), p in points.items() if pol == policy and p["move"] == 0]
            if first:
                s = [p for p in first if p["outcome"] == "success"]
                f = [p for p in first if p["outcome"] == "failure"]
                if s and f:
                    mix = natural * np.mean([np.mean(p["d_made"]) for p in s]) + (1 - natural) * np.mean([np.mean(p["d_made"]) for p in f])
                    lines.append(f"\nAt the first 16384, weighting successes and failures by greedy's natural {natural:.1%} conversion, {policy} changes conversion by {mix:+.1%}.")
    if "D" in {k[0] for k in points}:
        lines.append("\n## D against C: deeper search over the same 200 moves\n")
        lines.append("| Group (failed trajectories) | Points | Δ makes 32768, D − C [95%] | Δ score, D − C [95%] |")
        lines.append("|---|---:|---:|---:|")
        for label, members in group_rows(points, "D", "failure", lambda p: p["move"], TIME_BINS):
            pairs = [m | {"dd_made": np.array(m["d_made"][: len(c["d_made"])]) - np.array(c["d_made"][: len(m["d_made"])]),
                          "dd_score": np.array(m["d_score"][: len(c["d_score"])]) - np.array(c["d_score"][: len(m["d_score"])])}
                     for m in members if (c := points.get(("C", m["seed"], m["move"])))]
            if pairs:
                dm, dml, dmh = cluster_interval(pairs, "dd_made")
                ds, dsl, dsh = cluster_interval(pairs, "dd_score")
                lines.append(f"| {label} | {len(pairs)} | {dm:+.1%} [{dml:+.1%}, {dmh:+.1%}] | {ds:+,.0f} [{dsl:+,.0f}, {dsh:+,.0f}] |")
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"part2-{phase}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def p2_plan() -> None:
    """The expansion, by a fixed rule from the pilot: a (source outcome, time since 16384) group is informative if B's or
    C's effect could matter (upper bound of +5,000 score or +2 points of conversion) but is still uncertain (score
    interval wider than ±2,000). Informative groups get EXPANDED_CONTINUATIONS for A, B and C, at the pilot's points and
    the new trajectories' points. D joins only where its edge over C could matter (upper bound of +5,000)."""
    path = REPORTS / "p2-plan.json"
    if path.exists():
        print(path.read_text())
        return
    points = uplifts(p2_table())
    groups, reasons = [], []
    for outcome in ("failure", "success"):
        for b, (_, _, label) in enumerate(TIME_BINS):
            for policy in ("B", "C"):
                members = [p for (pol, _, _), p in points.items() if pol == policy and p["outcome"] == outcome and time_bin(p["move"]) == b]
                if len({m["seed"] for m in members}) < 2:
                    continue
                ds, dsl, dsh = cluster_interval(members, "d_score")
                dm, dml, dmh = cluster_interval(members, "d_made")
                if (dsh >= 5_000 or dmh >= 0.02) and (dsh - dsl) / 2 >= 2_000:
                    groups.append([outcome, b])
                    reasons.append(f"{outcome}, {label}: {policy} Δ score {ds:+,.0f} [{dsl:+,.0f}, {dsh:+,.0f}], Δ conversion {dm:+.1%} [{dml:+.1%}, {dmh:+.1%}]")
                    break
    policies = ["A", "B", "C"]
    for g in groups:
        members = [m for (pol, _, _), m in points.items() if pol == "D" and m["outcome"] == g[0] and time_bin(m["move"]) == g[1]]
        pairs = [m | {"dd": np.array(m["d_score"][: len(c["d_score"])]) - np.array(c["d_score"][: len(m["d_score"])])} for m in members if (c := points.get(("C", m["seed"], m["move"])))]
        if len({p["seed"] for p in pairs}) >= 2 and cluster_interval(pairs, "dd")[2] >= 5_000:
            policies.append("D")
            reasons.append(f"D joins: its edge over C could reach +5,000 in {g}")
            break
    # Fit the budget, from the pilot's measured cost per continuation by policy
    cost = {p: np.mean([np.mean(m["cpu"]) for (pol, _, _), m in points.items() if pol == p] or [0.0]) for p in policies}
    pilot_points = {g: sum(1 for (pol, _, _), m in points.items() if pol == "B" and (m["outcome"], time_bin(m["move"])) == g) for g in map(tuple, groups)}
    continuations = EXPANDED_CONTINUATIONS
    threads = os.cpu_count() or 1

    def hours(c: int) -> float:
        # pilot points topped up to c, and the new trajectories' points (about as many again) played to c
        return sum(n * ((c - PILOT_CONTINUATIONS) + c) for n in pilot_points.values()) * sum(cost[p] for p in policies) / threads / 3600

    while continuations > PILOT_CONTINUATIONS * 2 and hours(continuations) > BUDGET_HOURS:
        continuations //= 2
    plan = {"groups": groups, "policies": policies, "continuations": continuations, "estimated hours": round(hours(continuations), 2),
            "budget hours": BUDGET_HOURS, "reasons": reasons or ["No group met the rule: no expansion."]}
    path.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(plan, indent=2))


# ---------------------------------------------------------------- run record


def manifest() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    record = {"checkpoint": str(BASE), "checkpoint sha256": file_sha256(BASE), "commit": commit, "machine": platform.platform(), "processor": platform.processor(),
              "cpus": os.cpu_count(), "python": platform.python_version(), "smoke": SMOKE, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    path = REPORTS / "manifest.json"
    history = json.loads(path.read_text()) if path.exists() else []
    path.write_text(json.dumps(history + [record], indent=2) + "\n")
    print(json.dumps(record, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 7: the anatomy of the late game")
    parser.add_argument("step", choices=("manifest", "p1-greedy", "p1-depth3", "p1-report", "p2-trajectories", "p2-run", "p2-report", "p2-plan"))
    parser.add_argument("--phase", choices=("pilot", "expansion", "final"), default="pilot")
    args = parser.parse_args()
    steps = {"manifest": manifest, "p1-greedy": p1_greedy, "p1-depth3": p1_depth3, "p1-report": p1_report, "p2-plan": p2_plan,
             "p2-trajectories": lambda: p2_trajectories(args.phase), "p2-run": lambda: p2_run(args.phase), "p2-report": lambda: p2_report(args.phase)}
    steps[args.step]()


if __name__ == "__main__":
    main()
