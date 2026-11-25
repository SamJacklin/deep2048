"""Weight averages of one lineage's snapshots, greedy over compare.py's 10,000 seeds, paired with the cached 6.4M games.
A variant is name=games:weight,games:weight,... (weights are normalised); name ending in -nz averages each weight over
the snapshots where it is nonzero instead of over all of them."""
import os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from deep_2048.agents.ntuple import NTupleNetwork
from deep_2048.agents.ntuple.search import play_seeded

C = "checkpoints/ntuple"
PATHS = {"2.4": f"{C}/methods-8x6-scratch-stall/stall_2400000.npz", "3.6": f"{C}/methods-8x6-scratch-stall/stall_3600000.npz",
         "4.6": f"{C}/methods-8x6-scratch-stall/stall_4600000.npz", "5.6": f"{C}/methods-8x6-scratch-stall/stall_5600000.npz",
         "5.8": f"{C}/methods-8x6-second-reset/stall_5800000.npz", "6.4": f"{C}/methods-8x6-second-reset/stall_6400000.npz"}
CACHE = Path(__file__).parent / "games"
seeds = [*range(5000), *range(20_000, 25_000)]
pool = ThreadPoolExecutor(os.cpu_count())
variants = {a.split("=")[0]: [(p.split(":")[0], float(p.split(":")[1])) for p in a.split("=")[1].split(",")] for a in sys.argv[1:]}
nets, layout = {}, None
for k in {k for parts in variants.values() for k, _ in parts}:
    net = NTupleNetwork.load(PATHS[k])[0]
    nets[k], layout = net.weights, net.layout
base = np.load(CACHE / "6.4M.npy")
se = lambda v: v.std(ddof=1) / len(v) ** .5
print("| Average (10,000 games) | Avg score | ≥16384 | ≥32768 | Δ score vs 6.4M (± 1 se) | Δ ≥16384 | Δ ≥32768 |\n|---|---|---|---|---|---|---|", flush=True)
print(f"| 6.4M alone | {base[:,0].mean():,.0f} | {(base[:,1]>=16384).mean():.1%} | {(base[:,1]>=32768).mean():.1%} | — | — | — |", flush=True)
for name, parts in variants.items():
    t = time.perf_counter()
    cached = CACHE / f"avg-{name}.npy"
    if cached.exists():
        g = np.load(cached)
    else:
        w = np.zeros_like(nets[parts[0][0]])
        total = np.zeros_like(w) if name.endswith("-nz") else sum(s for _, s in parts)
        for k, s in parts:
            w += s * nets[k]
            if name.endswith("-nz"):
                total += s * (nets[k] != 0)
        w = np.where(total > 0, w / np.maximum(total, 1e-12), 0).astype(w.dtype) if name.endswith("-nz") else w / total
        g = np.array(list(pool.map(lambda s: play_seeded(w, layout, 1, 0.0, np.random.default_rng(s)), seeds)), float)
        np.save(cached, g)
        del w
    d = g[:, 0] - base[:, 0]
    r = [(g[:, 1] >= k).astype(float) - (base[:, 1] >= k) for k in (16384, 32768)]
    print(f"| {name} | {g[:,0].mean():,.0f} | {(g[:,1]>=16384).mean():.1%} | {(g[:,1]>=32768).mean():.1%} | {d.mean():+,.0f} ± {se(d):,.0f} | "
          f"{100*r[0].mean():+.1f} ± {100*se(r[0]):.1f} | {100*r[1].mean():+.1f} ± {100*se(r[1]):.1f} |  ({time.perf_counter()-t:.0f} s)", flush=True)
