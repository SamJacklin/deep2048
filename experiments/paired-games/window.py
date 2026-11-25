"""Finishers head to head from the 32768 window's edge: every measuring board (dev and test), CONT continuations each,
the same spawns for every player. usage: window.py CONT name=patch ..."""
import os, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
sys.path.insert(0, ".")
from deep_2048.agents.ntuple import NTupleNetwork
from deep_2048.agents.ntuple import arrival as ar
from deep_2048.agents.ntuple.finisher import FinisherNetwork

BASE = "checkpoints/ntuple/methods-8x6-second-reset/stall_6400000.npz"
m = np.load("experiments/finisher-6.4m/pool-64/measure.npz")
boards, seeds = m["boards"], m["seeds"] * 10 + m["continuation"]
cont = int(sys.argv[1])
pool = ThreadPoolExecutor(os.cpu_count())
jobs = [(i, c) for i in range(len(boards)) for c in range(cont)]
players = {"base": NTupleNetwork.load(BASE)[0]} | {a.split("=")[0]: FinisherNetwork.load(a.split("=")[1]) for a in sys.argv[2:]}
made = {}
for name, net in players.items():
    w, lay = net.weights, net.layout
    made[name] = np.array(list(pool.map(lambda j: ar.converts(w, lay, boards[j[0]], 1, 0.0, 14, np.random.default_rng([int(seeds[j[0]]), 21, j[1]])), jobs)), float).reshape(len(boards), cont).mean(axis=1)
se = lambda v: v.std(ddof=1) / len(v) ** .5
names = list(made)
print(f"{len(boards):,} boards x {cont} continuations\n| Player | Makes 32768 | Δ vs base (± 1 se) | Δ vs {names[1]} |\n|---|---|---|---|")
for n in names:
    d, e = made[n] - made["base"], made[n] - made[names[1]]
    print(f"| {n} | {made[n].mean():.1%} | {100*d.mean():+.1f} ± {100*se(d):.1f} | {100*e.mean():+.1f} ± {100*se(e):.1f} |")
