"""Why greedy games fall short of 16384: what separates the games that convert an 8192 into a 16384 from those that don't.

Usage:
    uv run python scripts/failure_anatomy.py checkpoints/ntuple/methods-8x6-best-long-tc0.005/games_10000000.npz
    uv run python scripts/failure_anatomy.py <checkpoint> --games 5000 --threads 10 --out experiments/failure-anatomy/tc0.005-10m

Plays seeded greedy games (seeds from 70,000,000, not the evaluation seeds) with the network, each until it makes 16384
or ends. For every game it keeps the board at its first 8192, the board where it ended, the boards 25, 100 and 300
moves before that, and the first time after its first 8192 that no largest tile sat in a corner: whether that move was
forced (no legal move kept one there) or chosen (another move would have).

The report (printed and written to report.md beside the raw records, records.npz) has:
1. Outcomes: how many games reach 8192, convert it, or end in the 8192 stage.
2. On the way to 16384: each board feature for the games that convert and those that don't, at three points of the
   8192 stage, and the conversion rate by feature band late in it, so the features that predict failure stand out.
3. How the 8192-stage failures end: the corner lost (forced or chosen) or kept, with the board's features where it ended.
4. Example failures, each with the board 100 moves before the end.

Board features (all symmetries considered, the best one counts):
- corner: a largest tile sits in a corner;
- chain: the length of the snake from that corner (along an edge, back along the next row, and so on) over which tiles
  never increase, the ordered run that merges back into the corner;
- empty: empty cells;
- big: tiles of 256 or more; stranded: those of them outside the chain;
- second: the second-largest tile;
- progress: the board's tile sum as a percentage of 16384, how far it is towards the next tile.
They're compared at five points of the 8192 stage: the first 8192 (just merged, so the rest of the board is small),
then the first 8192 that also holds a 4096, then a 4096 and a 2048, and so on down to a 512.
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
from deep_2048.agents.ntuple.network import TupleLayout, evaluate  # noqa: E402
from deep_2048.game import bitboard as bb  # noqa: E402

SEED_OFFSET = 70_000_000
CORNERS = (0, 3, 12, 15)
LOOKBACK = (25, 100, 300)  # boards kept from before a game's end
RING = 512


@njit(inline="always")
def _max_in_corner(board: np.uint64) -> bool:
    top = bb.max_exponent(board)
    return bb.get_cell(board, 0) == top or bb.get_cell(board, 3) == top or bb.get_cell(board, 12) == top or bb.get_cell(board, 15) == top


@njit(inline="always")
def _holds(board: np.uint64, exponent: int) -> bool:
    for i in range(16):
        if bb.get_cell(board, i) == exponent:
            return True
    return False


@njit(nogil=True)
def play_recorded(weights: np.ndarray, layout: TupleLayout, seed: int, goal: int, mark: int) -> tuple:
    """One greedy game until a 2**goal tile or its end.

    Returns (max exponent, moves, score, first 2**mark board, its move, final board, boards LOOKBACK moves before the
    end, move of the first corner loss after the first 2**mark (-1: none), whether it was forced, and the first boards
    with a 2**mark and every tile from 2**(mark-1) down to 2**(mark-1-k), k = 0..3 (0: never))."""
    np.random.seed(seed)
    board = bb.initial_board()
    ring = np.zeros(RING, dtype=np.uint64)
    moves = 0
    score = 0
    first_mark = np.uint64(0)
    first_mark_move = -1
    loss_move = -1
    loss_forced = False
    chain_points = np.zeros(4, dtype=np.uint64)  # the first board with a 2**mark and every tile from 2**(mark-1) down to 2**(mark-1-k)
    while True:
        ring[moves % RING] = board
        if bb.max_exponent(board) >= goal:
            break
        if first_mark_move < 0 and bb.max_exponent(board) >= mark:
            first_mark, first_mark_move = board, moves
        if first_mark_move >= 0 and bb.max_exponent(board) == mark:
            for k in range(4):
                if not _holds(board, mark - 1 - k):
                    break
                if chain_points[k] == 0:
                    chain_points[k] = board
        best_after = board
        best_reward = 0
        best_value = -np.inf
        keeps_corner = False
        for action in range(4):
            after, reward = bb.move(board, action)
            if after == board:
                continue
            value = reward + evaluate(weights, layout, after, board)
            if _max_in_corner(after):
                keeps_corner = True
            if value > best_value:
                best_after, best_reward, best_value = after, reward, value
        if best_value == -np.inf:
            break
        if first_mark_move >= 0 and loss_move < 0 and _max_in_corner(board) and not _max_in_corner(best_after):
            loss_move, loss_forced = moves, not keeps_corner
        board = bb.spawn_tile(best_after)
        score += best_reward
        moves += 1
    before = np.zeros(len(LOOKBACK), dtype=np.uint64)
    for k in range(len(LOOKBACK)):
        back = LOOKBACK[k]
        if back < RING and moves - back >= 0:
            before[k] = ring[(moves - back) % RING]
    return bb.max_exponent(board), moves, score, first_mark, first_mark_move, board, before, loss_move, loss_forced, chain_points


def exponents(board: int) -> np.ndarray:
    return np.array([[bb.get_cell(np.uint64(board), 4 * r + c) for c in range(4)] for r in range(4)])


def orientations(grid: np.ndarray) -> list[np.ndarray]:
    """The board's 8 symmetries, each with its corner of interest at the top left."""
    out = []
    for g in (grid, grid.T):
        for k in range(4):
            out.append(np.rot90(g, k))
    return out


SNAKE = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (1, 2), (1, 1), (1, 0), (2, 0), (2, 1), (2, 2), (2, 3), (3, 3), (3, 2), (3, 1), (3, 0)]


def features(board: int) -> dict[str, int]:
    grid = exponents(board)
    top = grid.max()
    chain_values: list[int] = []
    for g in orientations(grid):
        if g[0, 0] != top:
            continue
        values, previous = [], 99
        for r, c in SNAKE:
            if g[r, c] == 0 or g[r, c] > previous:
                break
            previous = g[r, c]
            values.append(int(g[r, c]))
        if len(values) > len(chain_values):
            chain_values = values
    big = int((grid >= 8).sum())
    flat = np.sort(grid.ravel())[::-1]
    return {
        "corner": int(any(grid[r, c] == top for r, c in ((0, 0), (0, 3), (3, 0), (3, 3)))),
        "chain": len(chain_values),
        "empty": int((grid == 0).sum()),
        "big": big,
        "stranded": big - sum(1 for v in chain_values if v >= 8),
        "second": int(1 << int(flat[1])) if flat[1] else 0,
        "progress": round(100 * sum((1 << int(v)) for v in grid.ravel() if v) / 16384),
    }


def show(board: int) -> str:
    grid = exponents(board)
    return "\n".join(" ".join(f"{(1 << int(v)) if v else 0:>6}" for v in row) for row in grid)


def band_table(rows: list[dict], key: str, bands: list[tuple[str, callable]]) -> list[str]:
    lines = [f"| {key} | games reaching 8192 | convert to 16384 |", "|---|---:|---:|"]
    for label, test in bands:
        chosen = [r for r in rows if test(r[key])]
        if chosen:
            rate = 100 * np.mean([r["converted"] for r in chosen])
            lines.append(f"| {label} | {len(chosen):,} | {rate:.1f}% |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--games", type=int, default=5000)
    parser.add_argument("--threads", type=int, default=os.cpu_count())
    parser.add_argument("--out", type=Path, help="where to write report.md and records.npz (default experiments/failure-anatomy/<checkpoint's run>)")
    args = parser.parse_args()
    out = args.out or Path("experiments/failure-anatomy") / f"{args.checkpoint.parent.name}-{args.checkpoint.stem}"
    out.mkdir(parents=True, exist_ok=True)

    network, metadata = NTupleNetwork.load(args.checkpoint)
    weights, layout = network.weights, network.layout
    goal, mark = 14, 13
    print(f"{args.checkpoint} ({metadata.games_played:,} games): {args.games:,} greedy games on {args.threads} threads...", flush=True)
    play_recorded(weights, layout, 1, 6, 5)  # compile
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        results = list(pool.map(lambda s: play_recorded(weights, layout, SEED_OFFSET + s, goal, mark), range(args.games)))
    print(f"  played in {time.time() - started:.0f}s", flush=True)

    tops = np.array([r[0] for r in results])
    np.savez(out / "records.npz", max_exponent=tops, moves=np.array([r[1] for r in results]), score=np.array([r[2] for r in results]),
             first_8192=np.array([r[3] for r in results], dtype=np.uint64), first_8192_move=np.array([r[4] for r in results]),
             final=np.array([r[5] for r in results], dtype=np.uint64), before_end=np.array([r[6] for r in results], dtype=np.uint64),
             corner_loss_move=np.array([r[7] for r in results]), corner_loss_forced=np.array([r[8] for r in results]),
             first_8192_with_chain=np.array([r[9] for r in results], dtype=np.uint64))

    reached = [i for i, r in enumerate(results) if r[4] >= 0]
    rows = []
    for i in reached:
        r = results[i]
        row = {"game": i, "converted": r[0] >= goal, "moves_after": r[1] - r[4], "loss": r[7], "forced": r[8]} | features(r[3])
        rows.append(row)
    failures = [r for r in rows if not r["converted"]]
    n = args.games
    lines = [f"# Failure anatomy: {args.checkpoint}", "", f"{metadata.games_played:,} games of training; {n:,} greedy games from seed {SEED_OFFSET:,}.", ""]
    lines += ["## 1. Outcomes", "", "| | Games | Share |", "|---|---:|---:|",
              f"| Ended before 8192 | {n - len(reached):,} | {100 * (n - len(reached)) / n:.1f}% |",
              f"| Reached 8192 | {len(reached):,} | {100 * len(reached) / n:.1f}% |",
              f"| Converted to 16384 | {len(reached) - len(failures):,} | {100 * (len(reached) - len(failures)) / n:.1f}% "
              f"({100 * (1 - len(failures) / max(len(reached), 1)):.1f}% of those reaching 8192) |",
              f"| Ended in the 8192 stage | {len(failures):,} | {100 * len(failures) / n:.1f}% |", ""]

    keys = ["corner", "chain", "empty", "big", "stranded"]
    points = (("first 8192", lambda r: r[3]), ("+4096", lambda r: r[9][0]), ("+4096+2048", lambda r: r[9][1]), ("+4096+2048+1024", lambda r: r[9][2]),
              ("+4096+2048+1024+512", lambda r: r[9][3]))
    lines += ["## 2. On the way to 16384", "",
              "Board features at five points of the 8192 stage (the first board holding the 8192 and the chain named), means for the games that go on "
              "to convert and those that don't (converts / fails). Games that never reach a point are left out of its column.", "",
              "| Feature | " + " | ".join(label for label, _ in points) + " |",
              "|---|" + "---:|" * len(points)]
    by_point = {}
    for label, pick in points:
        by_point[label] = [{"converted": r["converted"]} | features(pick(results[r["game"]])) for r in rows if pick(results[r["game"]])]
    for key in keys + ["progress"]:
        cells = []
        for label, _ in points:
            group = by_point[label]
            good = [g[key] for g in group if g["converted"]]
            bad = [g[key] for g in group if not g["converted"]]
            cells.append(f"{np.mean(good) if good else float('nan'):.2f} / {np.mean(bad) if bad else float('nan'):.2f}")
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    reach = [(label, len(by_point[label]), 100 * np.mean([g["converted"] for g in by_point[label]]) if by_point[label] else float("nan")) for label, _ in points]
    lines += ["", "| Point | Games reaching it | Of those, convert |", "|---|---:|---:|"] + [f"| {label} | {count:,} | {rate:.1f}% |" for label, count, rate in reach] + [""]
    half = by_point["+4096+2048+1024"]
    lines += ["Conversion by feature band, at the first 8192 with a 4096, a 2048 and a 1024:", ""]
    lines += band_table(half, "corner", [("largest tile in a corner", lambda v: v == 1), ("not in a corner", lambda v: v == 0)]) + [""]
    lines += band_table(half, "chain", [("1-2", lambda v: v <= 2), ("3-4", lambda v: 3 <= v <= 4), ("5-6", lambda v: 5 <= v <= 6), ("7-8", lambda v: 7 <= v <= 8), ("9+", lambda v: v >= 9)]) + [""]
    lines += band_table(half, "empty", [("0-2", lambda v: v <= 2), ("3-4", lambda v: 3 <= v <= 4), ("5-6", lambda v: 5 <= v <= 6), ("7+", lambda v: v >= 7)]) + [""]
    lines += band_table(half, "stranded", [("0", lambda v: v == 0), ("1", lambda v: v == 1), ("2", lambda v: v == 2), ("3+", lambda v: v >= 3)]) + [""]

    lines += ["## 3. How the 8192-stage failures end", ""]
    lost_forced = [r for r in failures if r["loss"] >= 0 and r["forced"]]
    lost_chosen = [r for r in failures if r["loss"] >= 0 and not r["forced"]]
    kept = [r for r in failures if r["loss"] < 0]
    winners = [r for r in rows if r["converted"]]
    lines += ["Whether the largest tile left its corner at any point after the first 8192, and if so whether the move was forced:", "",
              "| | Failures | Share of failures | For comparison: converting games |", "|---|---:|---:|---:|",
              f"| Corner lost, forced | {len(lost_forced):,} | {100 * len(lost_forced) / max(len(failures), 1):.1f}% | {100 * np.mean([r['loss'] >= 0 and r['forced'] for r in winners]):.1f}% |",
              f"| Corner lost, chosen | {len(lost_chosen):,} | {100 * len(lost_chosen) / max(len(failures), 1):.1f}% | {100 * np.mean([r['loss'] >= 0 and not r['forced'] for r in winners]):.1f}% |",
              f"| Corner kept | {len(kept):,} | {100 * len(kept) / max(len(failures), 1):.1f}% | {100 * np.mean([r['loss'] < 0 for r in winners]):.1f}% |", ""]
    if failures:
        lines += ["Where the failures ended (means):", "", "| | Games | Moves after the first 8192 | Corner | Chain | Empty | Big | Stranded | Progress % |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for label, group in (("Corner lost, forced", lost_forced), ("Corner lost, chosen", lost_chosen), ("Corner kept", kept)):
            if group:
                ends = [features(results[r["game"]][5]) for r in group]
                lines.append(f"| {label} | {len(group):,} | {np.mean([r['moves_after'] for r in group]):.0f} | "
                             + " | ".join(f"{np.mean([e[k] for e in ends]):.2f}" for k in keys) + f" | {np.mean([e['progress'] for e in ends]):.0f} |")
        lines.append("")
        moves_to_loss = [r["loss"] - results[r["game"]][4] for r in lost_forced + lost_chosen]
        if moves_to_loss:
            lines += [f"Corner lost a median {np.median(moves_to_loss):.0f} moves after the first 8192, and the game ended a median "
                      f"{np.median([results[r['game']][1] - r['loss'] for r in lost_forced + lost_chosen]):.0f} moves after that.", ""]

    lines += ["## 4. Example failures", "", "Each: the board 100 moves before the end, then the end.", ""]
    for label, group in (("Corner lost, forced", lost_forced), ("Corner lost, chosen", lost_chosen), ("Corner kept", kept)):
        for r in group[:3]:
            result = results[r["game"]]
            earlier = show(int(result[6][1])) if result[6][1] else "(no board)"
            lines += [f"**{label}**, game {r['game']} ({result[1] - result[4]} moves after its first 8192):", "", "```", earlier, "", show(int(result[5])), "```", ""]

    report = "\n".join(lines)
    (out / "report.md").write_text(report + "\n")
    print(report)
    print(f"\nWritten to {out / 'report.md'} and {out / 'records.npz'}")


if __name__ == "__main__":
    main()
