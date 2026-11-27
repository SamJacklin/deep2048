"""Rollout teacher check: from the 16384 pool's dev window boards (last 1,024 before 16384, 6.4M network), play on to
16384 choosing each move by Q(a) = r_a + mean over K rollouts of [L greedy moves' rewards + V(last afterstate)].
Rollouts follow the base's own greedy policy, share spawn draws across the four moves (common random numbers), and
use their own random stream, so the game's spawns are the same as in baseline-dev.npz (greedy / depth3 / teacher).
L = 0 is greedy play exactly.

usage: rollout.py BOARDS CONTINUATIONS L:K:Z10 ...   (L -1 = to the end, scored by conversion; Z10 = z x 10)   (BOARDS first dev boards, continuations 0..C-1)"""
import os, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from numba import njit
sys.path.insert(0, ".")
from deep_2048.agents.ntuple import NTupleNetwork
from deep_2048.agents.ntuple.network import evaluate
from deep_2048.game import bitboard as bb

TOP = 13
POOL_DIR = "experiments/finisher-6.4m-16384/pool-1024"
BASE = "checkpoints/ntuple/methods-8x6-second-reset/stall_6400000.npz"


@njit(nogil=True)
def greedy(w, lay, board):
    """(best action or -1, its afterstate, its reward)."""
    best, bv, ba, br = -1, -np.inf, board, 0
    for a in range(4):
        after, r = bb.move(board, a)
        if after == board:
            continue
        v = r + evaluate(w, lay, after, board)
        if v > bv:
            best, bv, ba, br = a, v, after, r
    return best, ba, br


@njit(nogil=True)
def spawn_u(board, u1, u2):
    empty = bb._empty_cells(board)
    n = bb._count_cells(empty)
    if n == 0:
        return board
    target = min(int(u1 * n), n - 1)
    e = np.uint64(1) if u2 < 0.9 else np.uint64(2)
    return bb._place_in_empty(board, empty, target, e)


@njit(nogil=True)
def rollout(w, lay, after, state, u, length, top):
    """length >= 0: rewards of that many greedy moves plus the value after them. length < 0: 1 if greedy play makes
    2**(top + 1) before the game ends, else 0 (spawns beyond the drawn ones come from numba's own generator)."""
    if length < 0:
        t = 0
        while True:
            board = spawn_u(after, u[t, 0], u[t, 1]) if t < u.shape[0] else bb.spawn_tile(after)
            if bb.max_exponent(board) > top:
                return 1.0
            a, after, _ = greedy(w, lay, board)
            if a < 0:
                return 0.0
            t += 1
    total = 0.0
    for t in range(length):
        board = spawn_u(after, u[t, 0], u[t, 1])
        a, nxt, r = greedy(w, lay, board)
        if a < 0:
            return total
        total += r
        after, state = nxt, board
    return total + evaluate(w, lay, after, state)


@njit(nogil=True)
def play(w, lay, start, length, k, z, top, rng, rrng):
    """(made the larger tile, decisions, decisions where the rollout move differed from greedy)."""
    board = np.uint64(start)
    decisions = differs = 0
    while True:
        g, _, _ = greedy(w, lay, board)
        if g < 0:
            return 0, decisions, differs
        u = rrng.random((k, length if length > 0 else 600, 2))
        vals = np.full((4, k), np.nan)
        for a in range(4):
            after, r = bb.move(board, a)
            if after == board:
                continue
            for j in range(k):
                vals[a, j] = (r if length >= 0 else 0.0) + rollout(w, lay, after, board, u[j], length, top)
        # override greedy only where a move beats it by more than z standard errors of the paired difference
        best, margin = g, 0.0
        for a in range(4):
            if a == g or np.isnan(vals[a, 0]):
                continue
            d = vals[a] - vals[g]
            m, sd = d.mean(), d.std()
            if m > margin and m > z * sd / np.sqrt(k):
                best, margin = a, m
        decisions += 1
        differs += best != g
        board, _ = bb.move(board, best)
        board = bb.spawn_tile_seeded(board, rng)
        if bb.max_exponent(board) > top:
            return 1, decisions, differs


if __name__ == "__main__":
    n_boards, n_cont, configs = int(sys.argv[1]), int(sys.argv[2]), [tuple(map(int, c.split(":"))) for c in sys.argv[3:]]
    net, _ = NTupleNetwork.load(BASE)
    w, lay = net.weights, net.layout
    m = np.load(f"{POOL_DIR}/measure.npz")
    dev = m["seeds"] % 2 == 0
    boards, seeds = m["boards"][dev][:n_boards], (m["seeds"][dev] * 10 + m["continuation"][dev])[:n_boards]
    base = np.load(f"{POOL_DIR}/baseline-dev.npz")
    ref = {k: base[k][:n_boards, :n_cont].astype(float).ravel() for k in ("greedy", "depth3", "teacher depth 8 cutoff 0.02")}
    pool = ThreadPoolExecutor(os.cpu_count())
    jobs = [(i, c) for i in range(len(boards)) for c in range(n_cont)]
    se = lambda v: v.std(ddof=1) / len(v) ** .5
    print(f"{len(boards)} dev boards x {n_cont} continuations. Stored: " + ", ".join(f"{k} {v.mean():.1%}" for k, v in ref.items()), flush=True)
    print("| Teacher | Makes 16384 | vs greedy | vs depth 3 | Moves differing from greedy | Time |\n|---|---|---|---|---|---|", flush=True)
    for length, k, z10 in configs:
        z = z10 / 10
        t = time.perf_counter()
        out = np.array(list(pool.map(lambda j: play(w, lay, boards[j[0]], length, k, z, TOP, np.random.default_rng([int(seeds[j[0]]), 11, j[1]]),
                                                     np.random.default_rng([int(seeds[j[0]]), 12, j[1]])), jobs)), float)
        np.save(f"{os.path.dirname(__file__)}/games/rollout-L{length}-K{k}-z{z:g}-{n_boards}x{n_cont}.npy", out)
        made = out[:, 0]
        dg, d3 = made - ref["greedy"], made - ref["depth3"]
        print(f"| L {length if length >= 0 else 'to the end'}, K {k}, z {z:g} | {made.mean():.1%} | {100*dg.mean():+.1f} ± {100*se(dg):.1f} | {100*d3.mean():+.1f} ± {100*se(d3):.1f} | "
              f"{out[:,2].sum()/out[:,1].sum():.1%} of {out[:,1].mean():.0f} | {time.perf_counter()-t:.0f} s |", flush=True)
