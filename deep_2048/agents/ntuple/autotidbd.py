"""AutoStep TIDBD(0): every weight learns its own step size by meta-descent on the TD error.

The algorithm is Algorithm 1 of Günther et al. (2020), "Examining the Use of Temporal-Difference Incremental
Delta-Bar-Delta for Real-World Predictive Knowledge Architectures" (https://pmc.ncbi.nlm.nih.gov/articles/PMC7805647/),
with λ = 0. For a transition from features x to x' with reward r, and d = γx' − x:

     4      δ ← r + γ w·x' − w·x
     6–9    ξ_i ← max(|δ d_i h_i|,  ξ_i − (1/τ) α_i d_i z_i (|δ x_i h_i| − ξ_i))
    10      β_i ← β_i − θ δ d_i h_i / ξ_i
    11      M ← max(−Σ_i e^β_i d_i z_i, 1)
    13–14   β_i ← β_i − log M,  α_i ← e^β_i          (every weight)
    15      z_i ← γλ z_i + x_i                        (= x_i, since λ = 0)
    16      w_i ← w_i + α_i δ z_i
    17      h_i ← h_i max(1 + α_i z_i d_i, 0) + α_i δ z_i

Lines 6 to 11 read the trace z before line 15 replaces it, so with λ = 0 they see the previous transition's features
(none at a game's first update), as published.

`reference_step` is that listing on dense float64 vectors: slow, and the one to trust. `sparse_step` is what training
runs. It only visits the weights in x or x', which are the only ones lines 6 to 10, 16 and 17 can change, and it keeps
line 13's shift of every β as one shared offset: weight i's β is rates[i, BETA] + shared[0].

A feature's value is how many times a board reads that weight (symmetric boards read some weights several times), so
features arrive as sorted unique weight indices with their counts.

With `local` set (stage 8A's experimental repair, not the published algorithm), line 13 changes and nothing else:
log M is subtracted only from the β of the weights whose z isn't 0 where line 11 reads it, the previous afterstate's
features, in place of every weight's. Dividing those step sizes by M divides line 11's sum by M, so the bound it
measures is met exactly as before, while weights that took no part keep their step sizes. The shared offset then never
moves again: it stays as a checkpoint trained under the published rule left it.

With `proportional` as well (also experimental), the cut is shared by blame instead of equally. Weight i's part of
line 11's sum is c_i = −α_i d_i z_i. Each weight with c_i > 0 has λ c_i / Σ_{c>0} c subtracted from its β, with the one
λ that brings the sum to exactly 1, found by Newton's method. So a weight that makes up 90% of the positive part takes
90% of λ, the weights that merely sit beside it take almost none, and weights that add nothing or pull the sum down are
left alone. The local rule's equal cut lets the one weight whose step size keeps climbing drag its neighbours' down.

With `mean_decay` above 0 (experimental again), lines 6 to 10 change: ξ_i becomes a running mean of |δ d_i h_i|,
moving that fraction of the way to each new value, and line 10 divides by the larger of ξ_i and the current value, so
a meta-step is still at most θ. As published ξ_i is in effect the largest value the weight has ever seen, since its
decay needs the weight in two afterstates running and is scaled by α/τ. TD errors here are heavy-tailed (the end of a
game is an error the size of the whole value), so the typical meta-step measured at 850k games was 0.3% of θ and step
sizes barely moved in 1.2M games. Against a running mean the typical meta-step is a good part of θ.

Beyond the listing:
- Line 10 is skipped where ξ_i is 0. Its numerator is then 0 as well, so the listing asks for 0/0.
- An update whose TD error isn't finite is skipped and counted, and training stops on it.
- State is stored as float32 (the arithmetic is float64), and a weight's update count stops rising at 2**24.
- With several threads, updates aren't locked. Two threads on the same weight can lose one of the updates, and the
  shared offset can lose a normalisation that lands during another.
"""

import json
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from numba import njit

from deep_2048.agents.ntuple.network import TupleLayout, fill_weight_indices

# Columns of Tidbd.rates
BETA, H, XI, VISITS = 0, 1, 2, 3

# A batch's counters, in the order sparse_step keeps them. The effective step is Σ α_i x_i (x_i − γx'_i): the share of
# its own TD error an update removes, where above 1 overshoots. Normalisations count the updates where line 11's M > 1.
STATS = ("updates", "max_effective_step", "sum_effective_step", "overshoots", "normalisations", "max_m", "max_abs_delta", "non_finite", "sum_log_m")
UPDATES, MAX_STEP, SUM_STEP, OVERSHOOTS, NORMALISATIONS, MAX_M, MAX_ABS_DELTA, NON_FINITE, SUM_LOG_M = range(len(STATS))
# Which counters are maxima rather than sums, for combining batches
STAT_IS_MAX = np.array([name.startswith("max_") for name in STATS])

# For the diagnostics: a weight is frequently visited from this many updates, and its rate very small below this share of the initial rate
FREQUENT_VISITS = 100
VERY_SMALL = 0.01
LOG10_BINS = np.arange(-10.0, 1.01, 0.25)


class Tidbd(NamedTuple):
    """AutoStep TIDBD's state for every weight, and its two parameters. A NamedTuple so numba can use it.

    Row i of `rates` holds weight i's β before the shared offset, h, ξ and how many updates it has had, side by side
    so an update reads them from one cache line.
    """

    rates: np.ndarray
    shared: np.ndarray
    theta: float
    tau: float
    # Normalise only the weights in the trace (stage 8A) rather than every weight (as published)
    local: bool = False
    # With local: share the cut among the weights that made up the sum, by how much of it each made up
    proportional: bool = False
    # Above 0: ξ is a running mean of the meta-gradient's size, moving this fraction towards each new value, not its running maximum
    mean_decay: float = 0.0

    @classmethod
    def create(cls, n_weights: int, initial_alpha: float, theta: float, tau: float, dtype: type = np.float32, local: bool = False, proportional: bool = False, mean_decay: float = 0.0) -> "Tidbd":
        if proportional and not local:
            raise ValueError("proportional normalisation is a kind of local normalisation")
        if initial_alpha <= 0 or theta < 0 or tau <= 0:
            raise ValueError(f"AutoTIDBD needs a positive initial rate and τ and a non-negative θ, not {initial_alpha}, {tau} and {theta}")
        rates = np.zeros((n_weights, 4), dtype=dtype)
        rates[:, BETA] = np.log(initial_alpha)
        if not 0 <= mean_decay <= 1:
            raise ValueError(f"the normaliser's mean decay is a fraction, not {mean_decay}")
        return cls(rates, np.zeros(1), theta, tau, local, proportional, mean_decay)

    def alpha(self, indices: np.ndarray | slice = slice(None)) -> np.ndarray:
        """The step sizes of these weights (default: all of them)."""
        return np.exp(self.rates[indices, BETA].astype(np.float64) + self.shared[0])


def reference_step(
    w: np.ndarray, beta: np.ndarray, h: np.ndarray, xi: np.ndarray, z: np.ndarray, x: np.ndarray, x_next: np.ndarray,
    reward: float, gamma: float, theta: float, tau: float, lam: float = 0.0, local: bool = False, proportional: bool = False, mean_decay: float = 0.0,
) -> tuple[float, float]:
    """One step of Algorithm 1 on dense float64 vectors, changed in place, returning (δ, M). A terminal step has x_next = 0.

    With `local`, line 13 only lowers the β of weights whose z isn't 0 (the module's note on stage 8A's repair), and
    with `proportional` it lowers each positive contributor's by its share of the sum, here solved by bisection.
    With `mean_decay`, ξ is a running mean and line 10 divides by the larger of it and the current value."""
    delta = reward + gamma * (w @ x_next) - w @ x
    d = gamma * x_next - x
    alpha = np.exp(beta)
    if mean_decay > 0:
        now = np.abs(delta * d * h)
        divisor = np.maximum(now, xi)
        known = divisor != 0
        beta[known] -= theta * delta * d[known] * h[known] / divisor[known]
        # As the listing only touches ξ where the weight is in x or x' and has some h to speak of
        moved = (d != 0) & ((h != 0) | (z != 0))
        xi[moved] += mean_decay * (now[moved] - xi[moved])
    else:
        xi[:] = np.maximum(np.abs(delta * d * h), xi - alpha * d * z * (np.abs(delta * x * h) - xi) / tau)
        known = xi != 0
        beta[known] -= theta * delta * d[known] * h[known] / xi[known]
    m = max(float(-(np.exp(beta) * d) @ z), 1.0)
    if proportional and m > 1:
        c = -np.exp(beta) * d * z
        positive = c > 0
        share = c[positive] / c[positive].sum()
        target = 1.0 - c[~positive].sum()
        low, high = 0.0, 1.0
        while (c[positive] * np.exp(-high * share)).sum() > target:
            high *= 2
        for _ in range(200):
            middle = (low + high) / 2
            low, high = (middle, high) if (c[positive] * np.exp(-middle * share)).sum() > target else (low, middle)
        beta[positive] -= high * share
    elif local:
        beta[z != 0] -= np.log(m)
    else:
        beta -= np.log(m)
    alpha = np.exp(beta)
    z[:] = gamma * lam * z + x
    w += alpha * delta * z
    h[:] = h * np.maximum(1 + alpha * z * d, 0) + alpha * delta * z
    return float(delta), m


@njit(inline="always")
def features(layout: TupleLayout, board: np.uint64, state: np.uint64, indices: np.ndarray, counts: np.ndarray) -> int:
    """Write the afterstate's features as sorted unique weight indices and how often each is read, and return how many."""
    n = fill_weight_indices(layout, board, state, indices)
    # Tuples' tables follow one another, so the reads are already in order except within a tuple's symmetries
    for a in range(1, n):
        value = indices[a]
        b = a - 1
        while b >= 0 and indices[b] > value:
            indices[b + 1] = indices[b]
            b -= 1
        indices[b + 1] = value
    unique = 0
    for a in range(n):
        if unique > 0 and indices[unique - 1] == indices[a]:
            counts[unique - 1] += 1.0
        else:
            indices[unique] = indices[a]
            counts[unique] = 1.0
            unique += 1
    return unique


@njit
def sparse_step(
    weights: np.ndarray,
    optimiser: Tidbd,
    gamma: float,
    reward: float,
    current: tuple[np.ndarray, np.ndarray, int],
    successor: tuple[np.ndarray, np.ndarray, int],
    trace: tuple[np.ndarray, np.ndarray, int],
    work: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    stats: np.ndarray,
) -> float:
    """One step of Algorithm 1 with λ = 0 and return δ: reference_step, visiting only the weights in x or x'.

    `current`, `successor` and `trace` are x, x' and the z left by the previous step (the features of the afterstate
    before x, or none), each as (sorted unique indices, counts, how many). A terminal step has no successor features.
    `work` is room for the indices, x, d and z of every weight in x or x'.
    """
    rates, shared = optimiser.rates, optimiser.shared
    current_indices, current_counts, n_current = current
    successor_indices, successor_counts, n_successor = successor
    trace_indices, trace_counts, n_trace = trace
    indices, xs, ds, zs = work

    # Merge x and x' into one list of weights, looking each up in z, and take δ from the weights as they stand
    delta = reward
    n = a = b = p = 0
    while a < n_current or b < n_successor:
        x = x_next = 0.0
        if b >= n_successor or (a < n_current and current_indices[a] <= successor_indices[b]):
            i = current_indices[a]
            x = current_counts[a]
        else:
            i = successor_indices[b]
        if b < n_successor and successor_indices[b] == i:
            x_next = successor_counts[b]
            b += 1
        if x > 0.0:
            a += 1
        while p < n_trace and trace_indices[p] < i:
            p += 1
        d = gamma * x_next - x
        delta += d * weights[i]
        indices[n], xs[n], ds[n] = i, x, d
        zs[n] = trace_counts[p] if p < n_trace and trace_indices[p] == i else 0.0
        n += 1

    if not np.isfinite(delta):
        stats[NON_FINITE] += 1
        return delta
    stats[UPDATES] += 1
    stats[MAX_ABS_DELTA] = max(stats[MAX_ABS_DELTA], abs(delta))

    # Lines 5 to 11. Where d is 0, or h and z both are, they leave ξ and β as they were
    offset = shared[0]
    normaliser = 0.0
    for k in range(n):
        d, z = ds[k], zs[k]
        i = indices[k]
        h = rates[i, H]
        if d == 0.0 or (h == 0.0 and z == 0.0):
            continue
        beta = rates[i, BETA]
        xi = rates[i, XI]
        if optimiser.mean_decay > 0.0:
            now = abs(delta * d * h)
            divisor = max(now, xi)
            if divisor > 0.0:
                beta -= optimiser.theta * delta * d * h / divisor
                rates[i, BETA] = beta
            rates[i, XI] = xi + optimiser.mean_decay * (now - xi)
        else:
            decayed = xi
            if z != 0.0:
                decayed = xi - np.exp(beta + offset) * d * z * (abs(delta * xs[k] * h) - xi) / optimiser.tau
            xi = max(abs(delta * d * h), decayed)
            if xi > 0.0:
                beta -= optimiser.theta * delta * d * h / xi
                rates[i, BETA] = beta
            rates[i, XI] = xi
        if z != 0.0:
            normaliser -= np.exp(beta + offset) * d * z

    # Lines 12 to 14: dividing every step size by M is one subtraction from the shared offset
    # (or, with the local rule, one from each weight in the trace, each of which is listed once whatever its count)
    if normaliser > 1.0:
        log_m = np.log(normaliser)
        if optimiser.proportional:
            # c_k = −α d z for each weight in the sum; the positive ones give up λ × their share, where λ solves
            # Σ_{c>0} c exp(−λ share) = 1 − Σ_{c≤0} c. The left side falls and is convex in λ, so Newton's steps from 0
            # only ever approach the root from below.
            positive = negative = 0.0
            for k in range(n):
                if zs[k] != 0.0 and ds[k] != 0.0:
                    c = -np.exp(rates[indices[k], BETA] + offset) * ds[k] * zs[k]
                    if c > 0.0:
                        positive += c
                    else:
                        negative += c
            target = 1.0 - negative
            lam = 0.0
            for _ in range(50):
                value = slope = 0.0
                for k in range(n):
                    if zs[k] != 0.0 and ds[k] != 0.0:
                        c = -np.exp(rates[indices[k], BETA] + offset) * ds[k] * zs[k]
                        if c > 0.0:
                            term = c * np.exp(-lam * c / positive)
                            value += term
                            slope += term * c / positive
                step = (value - target) / slope
                lam += step
                if step <= 1e-13 * (1.0 + lam):
                    break
            for k in range(n):
                if zs[k] != 0.0 and ds[k] != 0.0:
                    c = -np.exp(rates[indices[k], BETA] + offset) * ds[k] * zs[k]
                    if c > 0.0:
                        rates[indices[k], BETA] -= lam * c / positive
        elif optimiser.local:
            for p in range(n_trace):
                rates[trace_indices[p], BETA] -= log_m
        else:
            shared[0] -= log_m
            offset = shared[0]
        stats[NORMALISATIONS] += 1
        stats[MAX_M] = max(stats[MAX_M], normaliser)
        stats[SUM_LOG_M] += log_m

    # Lines 15 to 17: z is now x, so only the weights in x move
    effective_step = 0.0
    for k in range(n):
        x = xs[k]
        if x == 0.0:
            continue
        i = indices[k]
        alpha = np.exp(rates[i, BETA] + offset)
        step = alpha * delta * x
        weights[i] += step
        rates[i, H] = rates[i, H] * max(1.0 + alpha * x * ds[k], 0.0) + step
        rates[i, VISITS] += 1
        effective_step -= alpha * x * ds[k]
    stats[MAX_STEP] = max(stats[MAX_STEP], effective_step)
    stats[SUM_STEP] += effective_step
    stats[OVERSHOOTS] += effective_step > 1.0
    return delta


def combine_stats(rows: np.ndarray) -> np.ndarray:
    """Combine counters from several batches or threads into one row: maxima stay maxima and the rest add up."""
    rows = rows.reshape(-1, len(STATS))
    return np.where(STAT_IS_MAX, rows.max(axis=0, initial=0.0), rows.sum(axis=0))


def normalisation(optimiser: Tidbd) -> str:
    """The name of the optimiser's normalisation rule, as settings and checkpoints give it."""
    return "proportional" if optimiser.proportional else "local" if optimiser.local else "global"


def checkpoint_arrays(optimiser: Tidbd, initial_alpha: float, totals: np.ndarray) -> dict[str, Any]:
    """The arrays a checkpoint needs to carry on training exactly: the optimiser's whole state, its settings and the run's counters so far."""
    settings = {"initial_alpha": initial_alpha, "theta": optimiser.theta, "tau": optimiser.tau, "normalisation": normalisation(optimiser), "normaliser_decay": optimiser.mean_decay}
    return {"tidbd_rates": optimiser.rates, "tidbd_shared": optimiser.shared, "tidbd_settings": json.dumps(settings), "tidbd_totals": totals}


def load(path: str | Path) -> tuple[Tidbd, float, np.ndarray] | None:
    """Load a checkpoint's (optimiser, initial rate, counters so far), or None if it holds weights only."""
    with np.load(path) as data:
        if "tidbd_rates" not in data:
            return None
        settings = json.loads(str(data["tidbd_settings"]))
        # Checkpoints from before the local rule existed are the published, global one, and have one counter fewer
        rule = settings.get("normalisation", "global")
        optimiser = Tidbd(data["tidbd_rates"], data["tidbd_shared"], settings["theta"], settings["tau"], rule != "global", rule == "proportional", settings.get("normaliser_decay", 0.0))
        totals = np.zeros(len(STATS))
        totals[: len(data["tidbd_totals"])] = data["tidbd_totals"]
        return optimiser, settings["initial_alpha"], totals


def _percentiles(log_alpha: np.ndarray) -> dict[str, float]:
    """Percentiles of the step sizes, taken on their logs since exp keeps their order."""
    if log_alpha.size == 0:
        return {}
    points = np.exp(np.concatenate([np.percentile(log_alpha, [10, 50, 90]), [log_alpha.min(), log_alpha.max()]]).astype(np.float64))
    return dict(zip(("p10", "median", "p90", "min", "max"), points.tolist()))


def rate_diagnostics(optimiser: Tidbd, initial_alpha: float, sample_every: int = 1) -> dict[str, Any]:
    """Describe the learned step sizes, over every weight or, with `sample_every` > 1, every that-many-th weight.

    Visited weights are those with at least one update. A rate has risen or fallen if it differs from the initial rate
    at all, which line 13's shared shift makes true of every weight once any update has been normalised.
    """
    rates = optimiser.rates[::sample_every]
    offset = float(optimiser.shared[0])
    visits = rates[:, VISITS]
    visited = visits > 0
    log_alpha = rates[:, BETA] + np.float32(offset)
    start = np.log(initial_alpha)
    seen = log_alpha[visited]
    frequent = log_alpha[visits >= FREQUENT_VISITS]
    result: dict[str, Any] = {
        "sampled": sample_every > 1,
        "sample_every": sample_every,
        "weights": int(rates.shape[0]),
        "visited": int(visited.sum()),
        "frequent": int(frequent.size),
        "initial_alpha": initial_alpha,
        "log_offset": offset,
        "alpha_all": _percentiles(log_alpha),
        "alpha_visited": _percentiles(seen),
        "alpha_frequent": _percentiles(frequent),
    }
    for name, values in (("visited", seen), ("frequent", frequent)):
        if values.size:
            # float32 β can't hold log(initial rate) exactly, so "unchanged" allows for its rounding
            result[f"{name}_increased"] = float(np.mean(values > start + 1e-6))
            result[f"{name}_decreased"] = float(np.mean(values < start - 1e-6))
            result[f"{name}_very_small"] = float(np.mean(values < start + np.log(VERY_SMALL)))
    # Most weights are rarely updated, so the overall median says little about the weights doing most of the work
    result["alpha_by_visits"] = {}
    for low in (1, 100, 1_000, 10_000, 100_000, 1_000_000):
        band = log_alpha[(visits >= low) & (visits < (100 if low == 1 else 10 * low if low < 1_000_000 else np.inf))]
        if band.size:
            result["alpha_by_visits"][str(low)] = {"weights": int(band.size)} | _percentiles(band)
    result["log10_alpha_bins"] = LOG10_BINS.tolist()
    result["log10_alpha_visited_histogram"] = np.histogram(seen / np.float32(np.log(10)), bins=LOG10_BINS)[0].tolist()
    result["non_finite_state"] = int((~np.isfinite(rates)).sum())
    return result
