"""Players over seeds 0-4999 and 20000-24999 (10,000 games), paired game by game. Arguments: name=path ... then
optional pairs a-b to compare (default: each against the first). A player's games are cached in games/ beside this file.
A name ending in -repeat plays with repeated halving (network.with_repeated_halving). Run from the repository root."""
import os, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from deep_2048.agents.ntuple import NTupleNetwork
from deep_2048.agents.ntuple.finisher import FinisherNetwork, combine_windows
from deep_2048.agents.ntuple.network import with_repeated_halving
from deep_2048.agents.ntuple.search import play_seeded

CACHE = Path(__file__).parent / "games"
seeds = [*range(5000), *range(20_000, 25_000)]
SUFFIX = ""
if os.environ.get("SEEDS"):  # SEEDS=first:end plays those seeds instead, cached apart
    seeds, SUFFIX = list(range(*map(int, os.environ["SEEDS"].split(":")))), "-" + os.environ["SEEDS"].replace(":", "-")
pool = ThreadPoolExecutor(os.cpu_count())
players = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
pairs = [tuple(a.split(":")) for a in sys.argv[1:] if ":" in a and "=" not in a] or [(n, next(iter(players))) for n in list(players)[1:]]
g = {}
for name, path in players.items():
    cached = CACHE / f"{name}{SUFFIX}.npy"
    if cached.exists():
        g[name] = np.load(cached)
        continue
    if "+" in path:  # two finishers: the higher window first
        w, lay = combine_windows(*path.split("+"))
    else:
        with np.load(path) as d:
            is_patch = "finisher" in d.files
        net = FinisherNetwork.load(path) if is_patch else NTupleNetwork.load(path)[0]
        w, lay = net.weights, net.layout
        del net
    if name.endswith("-repeat"):
        lay = with_repeated_halving(lay)
    g[name] = np.array(list(pool.map(lambda s: play_seeded(w, lay, 1, 0.0, np.random.default_rng(s)), seeds)), float)
    np.save(cached, g[name])
    del w
se = lambda v: v.std(ddof=1) / len(v) ** .5
print(f"| Player ({len(seeds):,} games) | Avg score | ≥8192 | ≥16384 | ≥32768 | 8192→16384 | 16384→32768 |\n|---|---|---|---|---|---|---|")
for n, x in g.items():
    t = x[:, 1]
    print(f"| {n} | {x[:,0].mean():,.0f} | {(t>=8192).mean():.1%} | {(t>=16384).mean():.1%} | {(t>=32768).mean():.1%} | {(t[t>=8192]>=16384).mean():.1%} | {(t[t>=16384]>=32768).mean():.1%} |")
print("\n| Comparison (± 1 se) | Δ score | Δ ≥16384 | Δ ≥32768 |\n|---|---|---|---|")
for a, b in pairs:
    d = g[a][:, 0] - g[b][:, 0]
    cells = [f"{100*((g[a][:,1]>=k).astype(float)-(g[b][:,1]>=k)).mean():+.1f} ± {100*se((g[a][:,1]>=k).astype(float)-(g[b][:,1]>=k)):.1f}" for k in (16384, 32768)]
    print(f"| {a} − {b} | {d.mean():+,.0f} ± {se(d):,.0f} | " + " | ".join(cells) + " |")
