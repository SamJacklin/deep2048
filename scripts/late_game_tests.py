"""Stage 1's late-game tests on the corpus from scripts/late_game_corpus.py. No training.

Usage:
    CHECKPOINT=checkpoints/ntuple/decision-8x6-tc-reset/games_2000000.npz
    uv run python scripts/late_game_tests.py coverage "$CHECKPOINT"      # a few minutes
    uv run python scripts/late_game_tests.py conversion "$CHECKPOINT"    # about an hour
    uv run python scripts/late_game_tests.py actions "$CHECKPOINT"       # about half an hour
    uv run python scripts/late_game_tests.py report "$CHECKPOINT"        # the tables, from the saved results

coverage: how often training updates the weights that late-game positions read. Plays greedy games as training does
    (a fifth of them from doubled first-8192 positions) without learning, counting every weight update, then reports
    for groups of corpus positions what share of the weights their legal moves read are updated rarely or never.
conversion: from each first-16384 position, 8 continuations by greedy play and 8 by depth-3 search, sharing their
    spawns (common random numbers): how often each makes a 32768 and what it adds to the score, and how the network's
    own estimate of the score to come compares with greedy play's, the policy it was trained on.
actions: for late-game positions (largest tile 16384 or more), half chosen at random and half where greedy play's top
    two moves are closest, every legal move followed by greedy play (64 rollouts) and by depth-3 search (8 rollouts),
    sharing spawns across moves. How often greedy play and depth-3 search choose a clearly worse move than the best,
    the score they lose by it, and how well the network's values of the afterstates match greedy play's returns.

Results go to experiments/late-game-diagnostics/ (the update counts, too large to commit, to output/).
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
from deep_2048.agents.ntuple.diagnostics import count_updates, greedy_values, rollout
from deep_2048.agents.ntuple.network import evaluate, load_traces, weight_indices
from deep_2048.agents.ntuple.search import action_values, play_from
from deep_2048.agents.ntuple.train import upgrade_all
from deep_2048.game import bitboard as bb

CORPUS = Path("experiments/late-game-corpus")
SAMPLES = Path("output/late-game-corpus/samples.csv")
OUTPUT = Path("experiments/late-game-diagnostics")
COUNTS = Path("output/late-game-diagnostics/counts.npz")
POLICY_SEEDS = {"greedy": 1, "depth3": 2}
# Continuation policies: name -> search depth
CONTINUATIONS = {"greedy": 1, "depth3": 3}
VISIT_BINS = (0, 1, 10, 100)  # weight updates per 1,000 training games
POOL = ThreadPoolExecutor(max_workers=os.cpu_count())


def read_rows(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def board_of(row: dict) -> np.uint64:
    return np.uint64(int(row["board"], 16))


def position_id(row: dict) -> int:
    """A stable id for a corpus position, for seeding its rollouts."""
    return POLICY_SEEDS[row["policy"]] * 10**10 + int(row["seed"]) * 10**4 + int(row["move"]) % 10**4


def legal_moves(board: np.uint64) -> list[tuple[int, np.uint64, int]]:
    moves = []
    for action in range(bb.NUM_ACTIONS):
        afterstate, reward = bb.move(board, action)
        if afterstate != board:
            moves.append((action, np.uint64(afterstate), reward))
    return moves


# ---------------------------------------------------------------- coverage


def count_training_updates(network: NTupleNetwork, games: int) -> tuple[np.ndarray, int]:
    """Count each weight's updates over `games` fresh greedy games plus a quarter as many late starts, as training plays."""
    counts = np.zeros(network.weights.size, dtype=np.int32)
    chunks = np.array_split(np.zeros(games, dtype=np.uint64), os.cpu_count() or 1)
    harvested = list(POOL.map(lambda k: count_updates(network.weights, network.layout, chunks[k], 100 + k, counts), range(len(chunks))))
    pool = upgrade_all(np.concatenate(harvested)[np.concatenate(harvested) != 0])
    late = np.random.default_rng(0).choice(pool, games // 4)
    late_chunks = np.array_split(late, os.cpu_count() or 1)
    list(POOL.map(lambda k: count_updates(network.weights, network.layout, late_chunks[k], 200 + k, counts), range(len(late_chunks))))
    return counts, games + games // 4


def coverage_of(boards: list[np.uint64], network: NTupleNetwork, per_thousand: np.ndarray, never: np.ndarray, coherence: np.ndarray) -> dict:
    """Shares of the reads made by the boards' legal moves, by how often training updates the weight read."""
    reads = [weight_indices(network.layout, afterstate, board) for board in boards for _, afterstate, _ in legal_moves(board)]
    indices = np.concatenate(reads)
    visits = per_thousand[indices]
    shares = {"none seen": float(np.mean(visits == 0)), "<1": float(np.mean((visits > 0) & (visits < 1)))}
    for low, high in zip(VISIT_BINS[1:], VISIT_BINS[2:]):
        shares[f"{low}-{high}"] = float(np.mean((visits >= low) & (visits < high)))
    shares[f"≥{VISIT_BINS[-1]}"] = float(np.mean(visits >= VISIT_BINS[-1]))
    return {"positions": len(boards), "reads": int(indices.size), **shares,
            "never updated in training": float(np.mean(never[indices])), "median coherence": float(np.median(coherence[indices]))}


def coverage(network: NTupleNetwork, checkpoint: Path, games: int) -> None:
    COUNTS.parent.mkdir(parents=True, exist_ok=True)
    if COUNTS.exists():
        saved = np.load(COUNTS)
        counts, total = saved["counts"], int(saved["games"])
    else:
        counts, total = count_training_updates(network, games)
        np.savez(COUNTS, counts=counts, games=total)
    per_thousand = counts * (1000 / total)
    traces = load_traces(checkpoint)
    never = traces.abs_errors == 0
    coherence = np.where(traces.abs_errors > 0, np.abs(traces.errors) / np.where(traces.abs_errors > 0, traces.abs_errors, 1), 1.0)

    rng = np.random.default_rng(0)
    groups: dict[str, list[np.uint64]] = {}
    positions = read_rows(CORPUS / "positions.csv")
    for policy in POLICY_SEEDS:
        for kind in ("first-8192", "first-16384"):
            groups[f"{kind}, {policy} games"] = [board_of(r) for r in positions if r["policy"] == policy and r["kind"] == kind]
    firsts = np.array([board_of(r) for r in positions if r["policy"] == "greedy" and r["kind"] == "first-8192"], dtype=np.uint64)
    groups["late starts: doubled first-8192, greedy games"] = list(upgrade_all(firsts))
    samples = read_rows(SAMPLES)
    for policy in POLICY_SEEDS:
        for tile in (1024, 2048, 4096, 8192, 16384, 32768):
            chosen = [board_of(r) for r in samples if r["policy"] == policy and int(r["max_tile"]) == tile]
            if chosen:
                picks = rng.choice(len(chosen), min(1000, len(chosen)), replace=False)
                groups[f"largest tile {tile}, {policy} games"] = [chosen[i] for i in picks]
    results = {"counting games": total, "groups": {name: coverage_of(boards, network, per_thousand, never, coherence) for name, boards in groups.items()}}
    (OUTPUT / "coverage.json").write_text(json.dumps(results, indent=2) + "\n")


# ---------------------------------------------------------------- conversion


def conversion(network: NTupleNetwork, continuations: int) -> None:
    rows = [r for r in read_rows(CORPUS / "positions.csv") if r["kind"] == "first-16384"]
    jobs = [(p, c, j) for p in range(len(rows)) for c in CONTINUATIONS for j in range(continuations)]

    def run(job: tuple[int, str, int]) -> tuple[int, int]:
        p, c, j = job
        return play_from(network.weights, network.layout, board_of(rows[p]), CONTINUATIONS[c], 0.0, np.random.default_rng([position_id(rows[p]), j]))

    results = list(POOL.map(run, jobs))
    shape = (len(rows), len(CONTINUATIONS), continuations)
    scores = np.array([s for s, _ in results]).reshape(shape)
    tiles = np.array([t for _, t in results]).reshape(shape)
    predicted = np.array([np.nanmax(greedy_values(network.weights, network.layout, board_of(r))) for r in rows])
    np.savez(OUTPUT / "conversion.npz", policy=np.array([r["policy"] for r in rows]), split=np.array([r["split"] for r in rows]),
             score_before=np.array([int(r["score"]) for r in rows]), predicted=predicted, scores=scores, tiles=tiles,
             continuations=np.array(list(CONTINUATIONS)))


# ---------------------------------------------------------------- actions


def choose_action_positions(per_policy: int, seed: int) -> list[dict]:
    """Late-game samples per source policy: half at random, half where greedy play's top two moves are closest."""
    rng = np.random.default_rng(seed)
    late = [r for r in read_rows(SAMPLES) if int(r["max_tile"]) >= 16384 and np.isfinite(float(r["greedy_margin"]))]
    chosen = []
    for policy in POLICY_SEEDS:
        candidates = [r for r in late if r["policy"] == policy]
        order = sorted(range(len(candidates)), key=lambda i: float(candidates[i]["greedy_margin"]))
        closest = [candidates[i] for i in order[: per_policy // 2]]
        rest = [candidates[i] for i in order[per_policy // 2 :]]
        random = [rest[i] for i in rng.choice(len(rest), per_policy - len(closest), replace=False)]
        chosen += [r | {"group": "close call"} for r in closest] + [r | {"group": "random"} for r in random]
    return chosen


def actions(network: NTupleNetwork, per_policy: int, rollouts: dict[str, int]) -> None:
    rows = choose_action_positions(per_policy, seed=1)
    moves = [legal_moves(board_of(r)) for r in rows]
    jobs = [(p, m, c, j) for p in range(len(rows)) for m in range(len(moves[p])) for c in CONTINUATIONS for j in range(rollouts[c])]

    def run(job: tuple[int, int, str, int]) -> tuple[int, int]:
        p, m, c, j = job
        return rollout(network.weights, network.layout, moves[p][m][1], CONTINUATIONS[c], 0.0, np.random.default_rng([position_id(rows[p]), j]))

    results = dict(zip(jobs, POOL.map(run, jobs)))
    longest = max(rollouts.values())
    # Missing moves and rollouts are NaN: [position, move slot (action), continuation, rollout]
    returns = np.full((len(rows), bb.NUM_ACTIONS, len(CONTINUATIONS), longest), np.nan)
    tiles = np.zeros_like(returns)
    rewards = np.full((len(rows), bb.NUM_ACTIONS), np.nan)
    predicted = np.full((len(rows), bb.NUM_ACTIONS), np.nan)
    depth3 = np.zeros(len(rows), dtype=np.int64)
    for p, row in enumerate(rows):
        board = board_of(row)
        for m, (action, afterstate, reward) in enumerate(moves[p]):
            rewards[p, action] = reward
            predicted[p, action] = evaluate(network.weights, network.layout, afterstate, board)
            for c, name in enumerate(CONTINUATIONS):
                for j in range(rollouts[name]):
                    returns[p, action, c, j], tiles[p, action, c, j] = results[(p, m, name, j)]
        depth3[p] = int(np.nanargmax(action_values(network.weights, network.layout, board, 3)))
    np.savez(OUTPUT / "actions.npz", policy=np.array([r["policy"] for r in rows]), group=np.array([r["group"] for r in rows]),
             board=np.array([board_of(r) for r in rows], dtype=np.uint64), max_tile=np.array([int(r["max_tile"]) for r in rows]),
             empty=np.array([int(r["empty"]) for r in rows]), rewards=rewards, predicted=predicted, returns=returns, tiles=tiles,
             depth3_choice=depth3, continuations=np.array(list(CONTINUATIONS)))


# ---------------------------------------------------------------- report


def mean_interval(values: np.ndarray) -> str:
    return f"{values.mean():,.0f} ± {1.96 * values.std(ddof=1) / np.sqrt(len(values)):,.0f}"


def report_coverage() -> None:
    path = OUTPUT / "coverage.json"
    if not path.exists():
        return
    results = json.loads(path.read_text())
    print(f"\n### Coverage: weight updates per 1,000 training games, from {results['counting games']:,} counting games\n")
    groups = results["groups"]
    columns = [k for k in next(iter(groups.values())) if k not in ("positions", "reads")]
    print("| Positions | Boards | " + " | ".join(columns) + " |")
    print("|---|---:|" + "---:|" * len(columns))
    for name, g in groups.items():
        print(f"| {name} | {g['positions']:,} | " + " | ".join(f"{g[c]:.1%}" if c != "median coherence" else f"{g[c]:.4f}" for c in columns) + " |")


def report_conversion() -> None:
    path = OUTPUT / "conversion.npz"
    if not path.exists():
        return
    d = np.load(path)
    print("\n### Conversion from first-16384 positions\n")
    print("| Positions from | Continued by | Positions | Makes 32768 | Score added | Network's estimate of the score to come | Estimate / greedy play's |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for policy in POLICY_SEEDS:
        chosen = d["policy"] == policy
        greedy_added = d["scores"][chosen, 0].mean(axis=1)
        for c, name in enumerate(d["continuations"]):
            added = d["scores"][chosen, c].mean(axis=1)
            made = (d["tiles"][chosen, c] >= 32768).mean()
            estimate = mean_interval(d["predicted"][chosen]) if c == 0 else ""
            ratio = f"{d['predicted'][chosen].mean() / greedy_added.mean():.2f}" if c == 0 else ""
            print(f"| {policy} games | {name} | {chosen.sum()} | {made:.1%} | {mean_interval(added)} | {estimate} | {ratio} |")


def report_actions() -> None:
    path = OUTPUT / "actions.npz"
    if not path.exists():
        return
    d = np.load(path)
    returns, rewards, predicted = d["returns"], d["rewards"], d["predicted"]
    totals = rewards[:, :, None, None] + returns  # each move's score to come, per continuation and rollout
    greedy_choice = np.nanargmax(rewards + predicted, axis=1)
    deciders = {"greedy play": greedy_choice, "depth-3 search": d["depth3_choice"]}

    print("\n### Action ranking: each legal move followed by each continuation, rollouts sharing spawns\n")
    print("| Positions from | Group | Decider | Judged by | Positions | Not the best on average | Clearly worse (> 2 SE) | Score lost: cross-validated mean |")
    print("|---|---|---|---|---:|---:|---:|---:|")
    for policy in POLICY_SEEDS:
        for group in ("random", "close call"):
            chosen = (d["policy"] == policy) & (d["group"] == group)
            for decider, choice in deciders.items():
                for c, name in enumerate(d["continuations"]):
                    rows = np.flatnonzero(chosen)
                    n = int(np.sum(~np.isnan(totals[rows[0], np.nanargmax(rewards[rows[0]]), c])))
                    not_best = clear = 0
                    losses = []
                    for p in rows:
                        values = totals[p, :, c, :n]
                        legal = ~np.isnan(values[:, 0])
                        mine = values[choice[p]]
                        means = np.where(legal, np.nanmean(values, axis=1), -np.inf)
                        not_best += int(np.argmax(means) != choice[p])
                        differences = values[legal] - mine
                        se = differences.std(axis=1, ddof=1) / np.sqrt(n)
                        clear += int(np.any(differences.mean(axis=1) > 2 * np.where(se > 0, se, np.inf)))
                        # Pick the best on half the rollouts and measure what it gains on the other half, both ways
                        halves = (np.arange(n) % 2 == 0, np.arange(n) % 2 == 1)
                        loss = 0.0
                        for pick, judge in (halves, halves[::-1]):
                            best = int(np.argmax(np.where(legal, np.nanmean(values[:, pick], axis=1), -np.inf)))
                            loss += np.mean(values[best, judge] - mine[judge]) / 2
                        losses.append(loss)
                    k = len(rows)
                    print(f"| {policy} games | {group} | {decider} | {name}, {n} rollouts | {k} | {not_best / k:.0%} | {clear / k:.0%} | {mean_interval(np.array(losses))} |")

    print("\n### Calibration: the network's afterstate values against greedy play's returns from them\n")
    print("| Largest tile | Empty cells | Afterstates | Mean predicted | Mean greedy return | Bias | Mean absolute error | Network ranks the best move first |")
    print("|---:|---|---:|---:|---:|---:|---:|---:|")
    realised = np.nanmean(returns[:, :, 0, :], axis=2)
    for tile in (16384, 32768):
        for label, low, high in (("0-3", 0, 3), ("4-7", 4, 7), ("8+", 8, 16)):
            chosen = (d["max_tile"] == tile) & (d["empty"] >= low) & (d["empty"] <= high)
            if not chosen.any():
                continue
            pred, real = predicted[chosen], realised[chosen]
            legal = ~np.isnan(pred)
            agree = np.mean(np.nanargmax(np.where(np.isnan(pred), -np.inf, pred + rewards[chosen]), axis=1)
                            == np.nanargmax(np.where(np.isnan(real), -np.inf, real + rewards[chosen]), axis=1))
            print(f"| {tile:,} | {label} | {legal.sum():,} | {pred[legal].mean():,.0f} | {real[legal].mean():,.0f} | "
                  f"{(pred[legal] - real[legal]).mean():+,.0f} | {np.abs(pred[legal] - real[legal]).mean():,.0f} | {agree:.0%} |")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1's late-game tests")
    parser.add_argument("test", choices=("coverage", "conversion", "actions", "report"))
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--counting-games", type=int, default=2000, help="coverage: fresh greedy games to count updates over (default: 2,000)")
    parser.add_argument("--continuations", type=int, default=8, help="conversion: continuations per position and policy (default: 8)")
    parser.add_argument("--positions", type=int, default=120, help="actions: positions per source policy (default: 120)")
    parser.add_argument("--greedy-rollouts", type=int, default=64, help="actions: greedy rollouts per move (default: 64)")
    parser.add_argument("--depth3-rollouts", type=int, default=8, help="actions: depth-3 rollouts per move (default: 8)")
    args = parser.parse_args()

    network, _ = NTupleNetwork.load(args.checkpoint)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.test == "coverage":
        coverage(network, args.checkpoint, args.counting_games)
    elif args.test == "conversion":
        conversion(network, args.continuations)
    elif args.test == "actions":
        actions(network, args.positions, {"greedy": args.greedy_rollouts, "depth3": args.depth3_rollouts})
    report_coverage()
    report_conversion()
    report_actions()


if __name__ == "__main__":
    main()
