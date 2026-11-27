"""E3a-0: how often would a small-tile strip with exactly one tile above 128 be readable, and does it tell moves apart?

Plays seeded greedy games of a checkpoint and, at every decision, looks at each legal afterstate's 16 strip reads
(8 symmetries x top two rows / middle two rows). A read has 0, 1, 2 or 3+ cells above 128: 0 is what the small-tile
tables read today, 1 is what a single-overflow table would add. Run from the repository root:
    uv run python experiments/overflow-count-small-7m/count.py CHECKPOINT [games] [threads]
"""
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numba import njit

sys.path.insert(0, ".")
from deep_2048.agents.ntuple import NTupleNetwork
from deep_2048.agents.ntuple.arrival import _best_action, chain_reach
from deep_2048.agents.ntuple.search import _search, new_transpositions
from deep_2048.game import bitboard as bb

STAGES, REACHES = 4, 9  # largest tile <=4096, 8192, 16384, 32768; chain reach 1..8
# per (stage, reach): decisions, decisions with 2+ moves, of those: a candidate has a one-over read, candidates differ
# in their one-over reads, candidates differ in their exact reads
D_ALL, D_MULTI, D_ANY1, D_DIFF1, D_DIFF0 = range(5)
# per (stage, reach, chosen/rejected): afterstates, with an exact read, with a one-over read, one-over but no exact
A_ALL, A_EXACT, A_ONE, A_ONLY_ONE = range(4)
MIX = np.uint64(0x9E3779B97F4A7C15)


@njit
def strip_reads(afterstate, reads, xs):
    """Count the afterstate's reads by cells above 128 into reads[strip, 0..3] and the lone big tile's size into xs.
    Returns (exact reads, one-over reads, order-free hash of the exact reads, of the one-over reads)."""
    exact = one = 0
    h0 = h1 = np.uint64(0)
    for sym in bb.symmetries(afterstate):
        for strip in range(2):
            half = (sym >> np.uint64(16 * strip)) & np.uint64(0xFFFFFFFF)
            big, where, size = 0, 0, 0
            for c in range(8):
                e = np.int64((half >> np.uint64(4 * c)) & np.uint64(0xF))
                if e >= 8:
                    big += 1
                    where, size = c, e
            reads[strip, min(big, 3)] += 1
            if big == 0:
                exact += 1
                h0 += (half + np.uint64(strip << 32) + np.uint64(1)) * MIX
            elif big == 1:
                one += 1
                xs[min(size, 11) - 8] += 1
                key = (half & ~(np.uint64(0xF) << np.uint64(4 * where))) + (np.uint64(where + 8 * strip + 1) << np.uint64(32))
                h1 += key * MIX
    return exact, one, h0, h1


@njit(nogil=True)
def count_game(weights, layout, rng):
    decisions = np.zeros((STAGES, REACHES, 5), np.int64)
    after = np.zeros((STAGES, REACHES, 2, 4), np.int64)
    reads = np.zeros((STAGES, REACHES, 2, 2, 4), np.int64)  # stage, reach, chosen/rejected, strip, cells above 128
    xs = np.zeros((STAGES, 4), np.int64)  # the lone big tile: 256, 512, 1024, 2048+
    board = bb.spawn_tile_seeded(bb.spawn_tile_seeded(np.uint64(0), rng), rng)
    table = new_transpositions()
    while True:
        values = _search(weights, layout, board, 1, 0.0, table)
        best = _best_action(values)
        if best < 0:
            return decisions, after, reads, xs
        s = min(max(bb.max_exponent(board) - 12, 0), 3)
        r = min(chain_reach(board), 8)
        legal = any1 = 0
        first0 = first1 = np.uint64(0)
        diff0 = diff1 = False
        for action in range(bb.NUM_ACTIONS):
            if np.isnan(values[action]):
                continue
            afterstate, _ = bb.move(board, action)
            which = 0 if action == best else 1
            exact, one, h0, h1 = strip_reads(afterstate, reads[s, r, which], xs[s])
            after[s, r, which, A_ALL] += 1
            after[s, r, which, A_EXACT] += exact > 0
            after[s, r, which, A_ONE] += one > 0
            after[s, r, which, A_ONLY_ONE] += one > 0 and exact == 0
            any1 += one > 0
            if legal == 0:
                first0, first1 = h0, h1
            else:
                diff0 |= h0 != first0
                diff1 |= h1 != first1
            legal += 1
        decisions[s, r, D_ALL] += 1
        if legal > 1:
            decisions[s, r, D_MULTI] += 1
            decisions[s, r, D_ANY1] += any1 > 0
            decisions[s, r, D_DIFF1] += diff1
            decisions[s, r, D_DIFF0] += diff0
        board, _ = bb.move(board, best)
        board = bb.spawn_tile_seeded(board, rng)


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "—"


def main():
    path, games, threads = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1000, int(sys.argv[3]) if len(sys.argv) > 3 else 4
    net = NTupleNetwork.load(path)[0]
    w, lay = net.weights, net.layout
    seeds = range(62_000_000, 62_000_000 + games)
    with ThreadPoolExecutor(threads) as pool:
        parts = list(pool.map(lambda s: count_game(w, lay, np.random.default_rng(s)), seeds))
    dec, aft, rd, xs = (sum(p[i] for p in parts) for i in range(4))
    np.savez(__file__.replace(".py", ".npz"), decisions=dec, after=aft, reads=rd, xs=xs)
    names = ("≤4096", "8192", "16384", "32768")
    print(f"{games:,} greedy games of {path}, seeds from 62,000,000; {dec[..., D_ALL].sum():,} decisions\n")

    def rows(label, d, a, r):
        both = a.sum(0)
        return (f"| {label} | {d[D_ALL]:,} | {pct(both[A_EXACT], both[A_ALL])} | {pct(both[A_ONE], both[A_ALL])} | {pct(both[A_ONLY_ONE], both[A_ALL])} "
                f"| {r[..., 0].sum() / both[A_ALL]:.2f} | {r[..., 1].sum() / both[A_ALL]:.2f} | {pct(d[D_ANY1], d[D_MULTI])} | {pct(d[D_DIFF1], d[D_MULTI])} | {pct(d[D_DIFF0], d[D_MULTI])} |")

    head = ("| {} | Decisions | Afterstates with an exact read | with a one-over read | one-over but no exact | Exact reads per afterstate "
            "| One-over reads per afterstate | Decisions where a move has a one-over read | where moves differ in one-over reads | where moves differ in exact reads |\n" + "|---" * 10 + "|")
    print(head.format("Largest tile"))
    for s in range(STAGES):
        print(rows(names[s], dec[s].sum(0), aft[s].sum(0), rd[s].sum(0)))
    print(rows("All", dec.sum((0, 1)), aft.sum((0, 1)), rd.sum((0, 1))))
    for s in (1, 2, 3):
        print(f"\nLargest tile {names[s]}, by how far the chain runs below it\n")
        print(head.format("Chain runs down to"))
        for r in range(1, REACHES):
            if dec[s, r, D_ALL]:
                print(rows(f"{2 ** (13 + s - r) if s < 3 else 2 ** (16 - r):,}{'+' if r == 8 else ''}", dec[s, r], aft[s, r], rd[s, r]))
    print("\n| Largest tile | Move | Strip | 0 above 128 | 1 | 2 | 3+ |\n|---|---|---|---|---|---|---|")
    for s in range(STAGES):
        for which, wn in enumerate(("chosen", "rejected")):
            for strip, sn in enumerate(("top two rows", "middle two rows")):
                x = rd[s, :, which, strip].sum(0)
                print(f"| {names[s]} | {wn} | {sn} | " + " | ".join(pct(v, x.sum()) for v in x) + " |")
    print("\n| Largest tile | Lone tile is 256 | 512 | 1024 | 2048+ |\n|---|---|---|---|---|")
    for s in range(STAGES):
        print(f"| {names[s]} | " + " | ".join(pct(v, xs[s].sum()) for v in xs[s]) + " |")


if __name__ == "__main__":
    main()
