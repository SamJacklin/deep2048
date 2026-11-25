"""Greedy play with the 32768 finisher, plus search only in two short stretches: the last 1,024 of tile sum before
16384 (largest tile 8192) and the last 64 before 32768 (largest tile 16384). 10,000 games on seeds 0-4999 and
20000-24999, paired with plain greedy play by the same player.

usage: selective.py PATCH  [name:depth16:cutoff16:depth32:cutoff32 ...]   (depth 1 = greedy in that stretch)"""
import os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from numba import njit
sys.path.insert(0, ".")
from deep_2048.agents.ntuple.finisher import FinisherNetwork
from deep_2048.agents.ntuple import arrival as ar
from deep_2048.agents.ntuple.search import _search, new_transpositions, tile_sum
from deep_2048.game import bitboard as bb


@njit(nogil=True)
def game(weights, layout, d16, c16, d32, c32, m16, m32, rng):
    bb.spawn_tile_seeded(bb.spawn_tile_seeded(np.uint64(0), rng), rng)
    board = bb.spawn_tile_seeded(bb.spawn_tile_seeded(np.uint64(0), rng), rng)
    table = new_transpositions()
    score = 0
    deep = 0
    while True:
        top = bb.max_exponent(board)
        depth, cutoff = 1, 0.0
        if top == 13 and tile_sum(board) >= 16384 - m16:
            depth, cutoff = d16, c16
        elif top == 14 and tile_sum(board) >= 32768 - m32:
            depth, cutoff = d32, c32
        if depth > 1:
            deep += 1
        a = ar._best_action(_search(weights, layout, board, depth, cutoff, table))
        if a < 0:
            return score, top, deep
        board, reward = bb.move(board, a)
        score += reward
        board = bb.spawn_tile_seeded(board, rng)


f = FinisherNetwork.load(sys.argv[1])
pool = ThreadPoolExecutor(os.cpu_count())
seeds = [*range(5000), *range(20_000, 25_000)] if not os.environ.get("SEED_START") else list(range(int(os.environ["SEED_START"]), int(os.environ["SEED_START"]) + 10_000))
se = lambda v: v.std(ddof=1) / len(v) ** .5
configs = [("greedy", 1, 0.0, 1, 0.0, 1024, 64)] + [(c.split(":")[0], int(c.split(":")[1]), float(c.split(":")[2]), int(c.split(":")[3]), float(c.split(":")[4]), int(c.split(":")[5]) if c.count(":") > 4 else 1024, int(c.split(":")[6]) if c.count(":") > 5 else 64) for c in sys.argv[2:]]
base = None
print("| Player | Avg score | ≥16384 | ≥32768 | Searched moves a game | Δ score vs greedy (± 1 se) | Δ ≥16384 | Δ ≥32768 |\n|---|---|---|---|---|---|---|---|", flush=True)
for name, d16, c16, d32, c32, m16, m32 in configs:
    t = time.perf_counter()
    g = np.array(list(pool.map(lambda s: game(f.weights, f.layout, d16, c16, d32, c32, m16, m32, np.random.default_rng(s)), seeds)), float)
    np.save(Path(__file__).parent / "games" / f"selective-{name}{os.environ.get('SEED_START', '')}.npy", g)
    base = g if base is None else base
    d = g[:, 0] - base[:, 0]
    r = [(g[:, 1] >= k).astype(float) - (base[:, 1] >= k) for k in (14, 15)]
    print(f"| {name} | {g[:,0].mean():,.0f} | {(g[:,1]>=14).mean():.1%} | {(g[:,1]>=15).mean():.1%} | {g[:,2].mean():.0f} | {d.mean():+,.0f} ± {se(d):,.0f} | "
          f"{100*r[0].mean():+.1f} ± {100*se(r[0]):.1f} | {100*r[1].mean():+.1f} ± {100*se(r[1]):.1f} |  ({time.perf_counter() - t:.0f} s)", flush=True)
