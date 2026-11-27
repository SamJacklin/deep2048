"""Confirm the sweep's only positive variant (the 0.1 residual at strength 0.05) on the test split, paired with the baseline."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from late_residual import CORPUS, POOL, board_of, read_rows, seed_of  # same seeds as the baseline's test evaluation

from deep_2048.agents.ntuple.residual import CorrectedNetwork
from deep_2048.agents.ntuple.search import play_from

corrected = CorrectedNetwork.load("checkpoints/ntuple/late-residual/late-residual/latest.npz")
weights, layout = corrected.weights, corrected.variant(scale=0.05)
rows = [r for r in read_rows(CORPUS) if r["kind"] == "first-16384" and r["split"] == "test"]
base = np.load("experiments/late-residual/baseline.npz")
out = {}
for name, depth, n in (("greedy", 1, 32), ("depth3", 3, 8)):
    jobs = [(p, j) for p in range(len(rows)) for j in range(n)]
    games = list(POOL.map(lambda job: play_from(weights, layout, board_of(rows[job[0]]), depth, 0.0, np.random.default_rng([seed_of(rows[job[0]]), job[1]])), jobs))
    scores = np.array([s for s, _ in games]).reshape(len(rows), n)
    tiles = np.array([t for _, t in games]).reshape(len(rows), n)
    out[f"{name}_scores"], out[f"{name}_tiles"] = scores, tiles
    made, base_made = (tiles >= 32768).mean(1), (base[f"{name}_tiles"] >= 32768).mean(1)
    added, base_added = scores.mean(1), base[f"{name}_scores"].mean(1)
    dm, da = made - base_made, added - base_added
    print(f"{name}: 32768 {made.mean():.1%} vs {base_made.mean():.1%} ({dm.mean():+.1%} ± {1.96 * dm.std(ddof=1) / np.sqrt(len(dm)):.1%}); "
          f"score added {added.mean():,.0f} vs {base_added.mean():,.0f} ({da.mean():+,.0f} ± {1.96 * da.std(ddof=1) / np.sqrt(len(da)):,.0f})", flush=True)
np.savez("experiments/late-residual/verification/test-confirmation-residual-0.1-strength-0.05.npz", **out)
