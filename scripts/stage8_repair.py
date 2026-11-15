"""Stage 8A: why AutoTIDBD's step sizes collapsed, and whether normalising locally repairs it.

    uv run python scripts/stage8_repair.py diagnose CHECKPOINT --games 40000    # replay training from a checkpoint, attributing every normalisation
    uv run python scripts/stage8_repair.py compare                               # experiment A1's tables: control against repair

`diagnose` carries on training in memory from the checkpoint under the published rule (nothing is saved over it), on
the usual 8 threads and then, from the same checkpoint, on 1, to see whether threads change how often normalisation
fires. Each update where line 11's M exceeds 1 is recorded with the weights that made up its sum.
"""

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from numba import njit

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from deep_2048.agents.ntuple import autotidbd
from deep_2048.agents.ntuple.autotidbd import BETA, NORMALISATIONS, VISITS, Tidbd
from deep_2048.agents.ntuple.network import NTupleNetwork, TupleLayout
from deep_2048.agents.ntuple.train import _best_move, upgrade_all
from deep_2048.game import bitboard as bb

OUTPUT = Path("experiments/stage8-tidbd-repair")
TOP = 6  # contributions kept per event, largest first
LATE_START_FRACTION, LATE_START_EXPONENT, LATE_START_POOL = 0.2, 13, 10_000


@njit(nogil=True)
def traced_games(n: int, weights: np.ndarray, optimiser: Tidbd, layout: TupleLayout, pool: np.ndarray, max_events: int):  # type: ignore[no-untyped-def]
    """train.play_games with AutoTIDBD, recording each normalisation: the same game loop as train.play_game_tidbd, with
    line 11's sum rebuilt from the features after any update that normalised.

    Returns (harvested boards, moves played, events recorded, per-event numbers, boards and top contributions).
    Per event: M from the rebuilt sum, the shared shift before and after, the sum of the positive and of the negative
    contributions, how many weights contributed, and the move number and whether the game was a synthetic start.
    """
    reads = len(layout.offsets) * bb.NUM_SYMMETRIES + 1
    indices = np.empty((3, reads), np.int64)
    counts = np.empty((3, reads))
    sizes = np.zeros(3, np.int64)
    work = (np.empty(2 * reads, np.int64), np.empty(2 * reads), np.empty(2 * reads), np.empty(2 * reads))
    stats = np.zeros(len(autotidbd.STATS))
    harvested = np.zeros(n, np.uint64)
    numbers = np.zeros((max_events, 8))
    boards = np.zeros((max_events, 4), np.uint64)
    top_indices = np.full((max_events, TOP), -1, np.int64)
    top_values = np.zeros((max_events, TOP, 5))  # contribution, x, x', z, step size
    contributions = np.empty(reads)
    recorded = 0
    moves_played = 0
    for game in range(n):
        start = np.uint64(0)
        synthetic = pool.size > 0 and np.random.random() < LATE_START_FRACTION
        if synthetic:
            start = pool[np.random.randint(pool.size)]
        board = bb.initial_board() if start == np.uint64(0) else start
        moves = 0
        sizes[:] = 0
        trace, current, successor = 0, 1, 2
        previous_afterstate = previous_state = np.uint64(0)
        while True:
            best_afterstate, best_reward, best_value = _best_move(weights, layout, board)
            game_over = best_value == -np.inf
            sizes[successor] = 0
            if not game_over:
                sizes[successor] = autotidbd.features(layout, best_afterstate, board, indices[successor], counts[successor])
            if moves > 0:
                before = optimiser.shared[0]
                events_before = stats[NORMALISATIONS]
                autotidbd.sparse_step(
                    weights, optimiser, 1.0, float(best_reward),
                    (indices[current], counts[current], sizes[current]),
                    (indices[successor], counts[successor], sizes[successor]),
                    (indices[trace], counts[trace], sizes[trace]),
                    work, stats,
                )  # fmt: skip
                if stats[NORMALISATIONS] > events_before and recorded < max_events:
                    after = optimiser.shared[0]
                    positive = negative = 0.0
                    contributors = 0
                    for p in range(sizes[trace]):
                        i = indices[trace, p]
                        x = x_next = 0.0
                        for j in range(sizes[current]):
                            if indices[current, j] == i:
                                x = counts[current, j]
                        for j in range(sizes[successor]):
                            if indices[successor, j] == i:
                                x_next = counts[successor, j]
                        # β is as line 10 left it, and the shift as it was when line 11 summed
                        alpha = np.exp(optimiser.rates[i, BETA] + before)
                        c = -alpha * (x_next - x) * counts[trace, p]
                        contributions[p] = c
                        if c > 0:
                            positive += c
                        else:
                            negative += c
                        contributors += c != 0
                    order = np.argsort(-contributions[: sizes[trace]])
                    for rank in range(min(TOP, sizes[trace])):
                        p = order[rank]
                        i = indices[trace, p]
                        x = x_next = 0.0
                        for j in range(sizes[current]):
                            if indices[current, j] == i:
                                x = counts[current, j]
                        for j in range(sizes[successor]):
                            if indices[successor, j] == i:
                                x_next = counts[successor, j]
                        top_indices[recorded, rank] = i
                        top_values[recorded, rank, 0] = contributions[p]
                        top_values[recorded, rank, 1] = x
                        top_values[recorded, rank, 2] = x_next
                        top_values[recorded, rank, 3] = counts[trace, p]
                        top_values[recorded, rank, 4] = np.exp(optimiser.rates[i, BETA] + before)
                    numbers[recorded, 0] = positive + negative
                    numbers[recorded, 1] = before
                    numbers[recorded, 2] = after
                    numbers[recorded, 3] = positive
                    numbers[recorded, 4] = negative
                    numbers[recorded, 5] = contributors
                    numbers[recorded, 6] = moves
                    numbers[recorded, 7] = synthetic
                    boards[recorded, 0] = previous_afterstate
                    boards[recorded, 1] = previous_state
                    boards[recorded, 2] = best_afterstate
                    boards[recorded, 3] = board
                    recorded += 1
            if game_over:
                break
            trace, current, successor = current, successor, trace
            previous_afterstate, previous_state = best_afterstate, board
            moves += 1
            board = bb.spawn_tile(best_afterstate)
            if not synthetic and harvested[game] == np.uint64(0) and best_reward >= (1 << LATE_START_EXPONENT) and bb.max_exponent(board) >= LATE_START_EXPONENT:
                harvested[game] = board
        moves_played += moves
    return harvested, moves_played, int(stats[NORMALISATIONS]), numbers[:recorded], boards[:recorded], top_indices[:recorded], top_values[:recorded]


def replay(checkpoint: Path, games: int, threads: int, seed: int) -> dict[str, np.ndarray]:
    """Train on from the checkpoint in memory for `games` games on `threads` threads, and return every normalisation."""
    network, _ = NTupleNetwork.load(checkpoint)
    loaded = autotidbd.load(checkpoint)
    assert loaded is not None and not loaded[0].local, "the diagnosis replays the published global rule"
    optimiser = loaded[0]
    shift_start = float(optimiser.shared[0])
    pool = np.zeros(0, np.uint64)
    parts: list[tuple] = []  # type: ignore[type-arg]
    batch_events, moves, started, played = [], 0, time.monotonic(), 0
    bb.seed(seed)
    with ThreadPoolExecutor(max_workers=threads) as executor:
        while played < games:
            batch = min(1_000, games - played)
            shares = [len(share) for share in np.array_split(np.arange(batch), 8 * threads if threads > 1 else 1)]
            results = list(executor.map(lambda share: traced_games(share, network.weights, optimiser, network.layout, pool, 500), shares))
            donated = np.concatenate([r[0] for r in results])
            pool = np.concatenate([pool, upgrade_all(donated[donated != 0])])[-LATE_START_POOL:]
            moves += sum(r[1] for r in results)
            batch_events.append(sum(r[2] for r in results))
            parts += [(np.full(len(r[3]), played), *r[3:]) for r in results if len(r[3])]
            played += batch
            if played % 5_000 == 0:
                print(f"  {threads} threads: {played:,} games, {sum(batch_events):,} normalisations, shift {optimiser.shared[0]:.4f}", flush=True)
    columns = [np.concatenate(column) for column in zip(*parts)] if parts else [np.zeros(0)] * 5
    top_weights = np.unique(columns[3][columns[3] >= 0]) if parts else np.zeros(0, np.int64)
    return {
        "batch": columns[0], "numbers": columns[1], "boards": columns[2], "top_indices": columns[3], "top_values": columns[4],
        "batch_events": np.array(batch_events), "moves": np.array(moves), "seconds": np.array(time.monotonic() - started),
        "shift_start": np.array(shift_start), "shift_end": np.array(float(optimiser.shared[0])), "threads": np.array(threads),
        "top_weights": top_weights, "top_weight_visits": optimiser.rates[top_weights, VISITS] if parts else np.zeros(0),
    }  # fmt: skip


def decode(network: NTupleNetwork, index: int) -> str:
    """A weight as the tiles its tuple reads (after downgrading), in the tuple's own cell order."""
    offsets = network.layout.offsets
    if index >= offsets[-1] + 16 ** len(network.tuples[-1]):
        return "scale term"
    t = int(np.searchsorted(offsets, index, side="right") - 1)
    local = int(index - offsets[t])
    tiles = [(local >> (4 * k)) & 15 for k in range(len(network.tuples[t]))]
    return f"tuple {t} cells {list(network.tuples[t])}: " + " ".join("·" if e == 0 else str(1 << e) for e in tiles)


def board_text(board: int) -> str:
    cells = [(int(board) >> (4 * i)) & 15 for i in range(16)]
    return " / ".join(" ".join("·" if e == 0 else str(1 << e) for e in cells[r : r + 4]) for r in range(0, 16, 4))


def diagnose(checkpoint: Path, games: int, single_games: int) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    network, metadata = NTupleNetwork.load(checkpoint)
    threaded = replay(checkpoint, games, 8, 42 + metadata.games_played)
    single = replay(checkpoint, single_games, 1, 42 + metadata.games_played)
    np.savez(OUTPUT / "normalisation-events.npz", **{f"threads8_{k}": v for k, v in threaded.items()}, **{f"threads1_{k}": v for k, v in single.items()})

    lines = [f"# Normalisation events replayed from {checkpoint} ({metadata.games_played:,} games)", ""]
    lines += ["| Replay | Games | Moves | Normalisations | per 10k games | per 1M moves | Shift at start | Shift at end | Σ log M of recorded events |", "|---|---|---|---|---|---|---|---|---|"]
    for name, run, n in (("8 threads", threaded, games), ("1 thread", single, single_games)):
        events = int(run["batch_events"].sum())
        m = run["numbers"][:, 0] if len(run["numbers"]) else np.zeros(0)
        lines.append(
            f"| {name} | {n:,} | {int(run['moves']) / 1e6:.0f}M | {events:,} | {1e4 * events / n:.1f} | {1e6 * events / int(run['moves']):.2f} "
            f"| {float(run['shift_start']):.4f} | {float(run['shift_end']):.4f} | {np.log(m[m > 1]).sum():.4f} of {float(run['shift_start']) - float(run['shift_end']):.4f} |"
        )
    numbers, top_indices, top_values = threaded["numbers"], threaded["top_indices"], threaded["top_values"]
    if len(numbers) == 0:
        lines.append("\nNo normalisation events in the replay.")
        (OUTPUT / "diagnosis.md").write_text("\n".join(lines) + "\n")
        print("\n".join(lines))
        return
    m = numbers[:, 0]
    per_10k = threaded["batch_events"].reshape(-1, 10).sum(axis=1) if len(threaded["batch_events"]) % 10 == 0 else threaded["batch_events"]
    lines += ["", f"Events per 10k games through the 8-thread replay: {', '.join(str(int(v)) for v in per_10k)}", ""]
    lines += [f"M over the {len(m):,} recorded events (8 threads): " + ", ".join(f"p{q} {np.percentile(m, q):.4f}" for q in (10, 50, 90, 99)) + f", max {m.max():.4f}; mean log M {np.log(m[m > 1]).mean():.4f}", ""]  # noqa: E501
    lines += [
        f"Events from games started at a synthetic late position: {numbers[:, 7].mean():.1%} (20% of games are). Median move number within the game: {np.median(numbers[:, 6]):,.0f}.",
        f"Weights with a non-zero contribution per event: median {np.median(numbers[:, 5]):.0f}, largest {numbers[:, 5].max():.0f}. "
        f"The largest single contribution is {np.median(top_values[:, 0, 0] / numbers[:, 3]):.0%} of the positive part of the sum at the median event, "
        f"and exceeds 1 by itself in {np.mean(top_values[:, 0, 0] > 1):.1%} of events.",
        "",
    ]

    # Each event's log M is shared among its positive contributors in proportion to their contributions
    blame: dict[int, float] = defaultdict(float)
    appearances: dict[int, int] = defaultdict(int)
    largest: dict[int, int] = defaultdict(int)
    details: dict[int, list[np.ndarray]] = defaultdict(list)
    for event in range(len(m)):
        if m[event] <= 1:
            continue
        for rank in range(TOP):
            index, values = int(top_indices[event, rank]), top_values[event, rank]
            if index < 0 or values[0] <= 0:
                continue
            blame[index] += np.log(m[event]) * values[0] / numbers[event, 3]
            appearances[index] += 1
            largest[index] += rank == 0
            details[index].append(values)
    total = np.log(m[m > 1]).sum()
    ranked = sorted(blame, key=lambda i: -blame[i])
    visits = dict(zip(threaded["top_weights"].tolist(), threaded["top_weight_visits"].tolist()))
    lines += [f"{len(ranked):,} distinct weights appear among the top {TOP} positive contributors. Their share of the summed log M ({total:.4f}), each event's split among its contributors by size:", ""]  # noqa: E501
    lines += ["| Weight | Reads | Share of Σ log M | Cumulative | Events | Largest in | Median step size | Median contribution | x → x' (median) | z (median) | Read more than once | Updates |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]  # noqa: E501
    cumulative = 0.0
    for index in ranked[:15]:
        values = np.array(details[index])
        cumulative += blame[index]
        repeated = np.mean((values[:, 1] > 1) | (values[:, 2] > 1) | (values[:, 3] > 1))
        lines.append(
            f"| {index} | {decode(network, index)} | {blame[index] / total:.1%} | {cumulative / total:.1%} | {appearances[index]:,} | {largest[index]:,} | {np.median(values[:, 4]):.3g} "
            f"| {np.median(values[:, 0]):.3g} | {np.median(values[:, 1]):.0f} → {np.median(values[:, 2]):.0f} | {np.median(values[:, 3]):.0f} | {repeated:.0%} | {visits.get(index, 0):,.0f} |"
        )
    shares = np.cumsum([blame[i] for i in ranked]) / total
    lines += ["", "Weights needed to account for a share of the summed log M: " + ", ".join(f"{q:.0%}: {int(np.searchsorted(shares, q) + 1)}" for q in (0.5, 0.8, 0.9, 0.99)), ""]

    lines += ["## Sample events", ""]
    for event in np.linspace(0, len(m) - 1, 6).astype(int):
        boards = threaded["boards"][event]
        lines += [
            f"- Batch from game {int(threaded['batch'][event]):,}, move {int(numbers[event, 6]):,}: M {m[event]:.4f}, shift {numbers[event, 1]:.4f} → {numbers[event, 2]:.4f}, "
            f"positive {numbers[event, 3]:.4f}, negative {numbers[event, 4]:.4f}",
            f"  - afterstate updated: {board_text(boards[0])}",
            f"  - successor afterstate: {board_text(boards[2])}",
        ]
        for rank in range(3):
            index, values = int(top_indices[event, rank]), top_values[event, rank]
            if index >= 0 and values[0] != 0:
                lines.append(f"  - {values[0]:+.4f} from {decode(network, index)} (step size {values[4]:.3g}, x {values[1]:.0f} → x' {values[2]:.0f}, z {values[3]:.0f})")
    (OUTPUT / "diagnosis.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def _rows(directory: Path, name: str) -> list[dict[str, str]]:
    path = directory / name
    return list(csv.DictReader(path.open())) if path.exists() else []


def compare() -> None:
    """Experiment A1 (and A2's scratch run, once it exists): each arm at each stop, on the development seeds."""
    from stage8 import EVALUATIONS, HEADER, RULE, describe, stored

    champion = stored("tc-2000000", 1, "dev")
    arms = {"start": "Shared start", "control": "Control (global)", "local": "Repair (local)", "proportional": "Repair (proportional)", "scratch": "Local from scratch"}
    found: dict[int, dict[str, dict[str, np.ndarray]]] = defaultdict(dict)
    for path in EVALUATIONS.glob("repair-*-depth1-dev.npz"):
        _, arm, games = path.name.split("-")[:3]
        found[int(games)][arm] = dict(np.load(path))
    for depth in (1, 3):
        champion = stored("tc-2000000", depth, "dev")
        print(f"\n{'Greedy' if depth == 1 else 'Exact depth 3'}, dev seeds 800,000 onwards, 1,000 games. Differences are paired on the same seeds.\n")
        print("| Arm | Games " + HEADER[len("| Network ") :] + " vs champion | 32768 vs champion | vs control | 32768 vs control |")
        print("|---|---" + RULE + "---|---|---|---|")
        if champion is not None:
            print("| TC champion | 2,000,000 " + describe("", champion)[2:])
        for games in sorted(found):
            control = stored(f"repair-control-{games}", depth, "dev")
            for arm, label in arms.items():
                result = stored(f"repair-{arm}-{games}", depth, "dev")
                if result is None:
                    continue
                row = f"| {label} | {games:,} " + describe("", result, champion)[2:]
                if arm in ("local", "proportional") and control is not None:
                    difference = result["scores"] - control["scores"]
                    conversion = (result["max_tiles"] >= 32768).astype(float) - (control["max_tiles"] >= 32768)
                    sd = lambda v: 1.96 * v.std(ddof=1) / np.sqrt(len(v))  # noqa: E731
                    row += f" {difference.mean():+,.0f} ± {sd(difference):,.0f} | {conversion.mean():+.1%} ± {sd(conversion):.1%} |"
                print(row)

    print("\nTraining and optimiser state at each stop. Training averages and speeds cover the games since the previous stop; step sizes are over weights updated at least once.\n")
    print("| Arm | Games | Moves | Training avg score | α p10 | α median | α p90 | α max | below 1% of initial | Shared shift | Normalisations since last stop | Σ log M since | Largest M so far | Overshoots since | Moves/s |")  # noqa: E501
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for arm in ("control", "local", "proportional", "scratch"):
        directory = OUTPUT / arm
        training, optimiser = _rows(directory, "training_history.csv"), _rows(directory, "optimizer_history.csv")
        previous = None
        for path in sorted((directory / "optimizer_stats").glob("games_*.json"), key=lambda p: int(p.stem.split("_")[1])):
            report = json.loads(path.read_text())
            games = report["games"]
            since = [r for r in optimiser if (previous or 0) < int(r["games"]) <= games]
            scores = [float(r["avg_score"]) for r in training if (previous or 0) < int(r["games"]) <= games]
            visited = report["alpha_visited"]
            total = lambda key: sum(float(r.get(key) or 0) for r in since)  # noqa: E731
            print(
                f"| {arms[arm]} | {games:,} | {report['moves'] / 1e9:.2f}B | {np.mean(scores):,.0f} | {visited['p10']:.2g} | {visited['median']:.2g} | {visited['p90']:.2g} | {visited['max']:.2g} "
                f"| {report['visited_very_small']:.2%} | {report['log_offset']:.3f} | {total('normalisations'):,.0f} | {total('sum_log_m'):.3f} | {report['totals']['max_m']:.3f} "
                f"| {total('overshoots'):,.0f} | {np.mean([float(r['moves_per_second']) for r in since]):,.0f} |"
            )
            previous = games


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    diagnosis = commands.add_parser("diagnose")
    diagnosis.add_argument("checkpoint", type=Path)
    diagnosis.add_argument("--games", type=int, default=40_000)
    diagnosis.add_argument("--single-games", type=int, default=4_000)
    commands.add_parser("compare")
    args = parser.parse_args()
    if args.command == "diagnose":
        diagnose(args.checkpoint, args.games, args.single_games)
    elif args.command == "compare":
        compare()


if __name__ == "__main__":
    main()
