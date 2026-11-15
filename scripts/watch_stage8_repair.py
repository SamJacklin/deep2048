"""Live side-by-side chart of stage 8A's fork: the control, local and proportional arms, and proportional with θ 0.1, from their shared 850k start.

    uv run python scripts/watch_stage8_repair.py            # a window, redrawn every 10 seconds
    uv run python scripts/watch_stage8_repair.py --once     # save experiments/stage8-tidbd-repair/fork-progress.png and exit

Training scores are rolling averages over the last 20 batches (20,000 games), ordinary games only.
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT = Path("experiments/stage8-tidbd-repair")
ARMS = {"control": ("Control (global)", "firebrick"), "local": ("Local", "steelblue"), "proportional": ("Proportional", "seagreen"), "theta0.1": ("Proportional, θ 0.1", "darkorange")}
PANELS = (
    ("training_history.csv", "avg_score", "Training score (rolling 20k games)", True, False),
    ("training_history.csv", "rate_16384", "% of training games reaching 16384 (rolling)", True, False),
    ("training_history.csv", "rate_32768", "% of training games reaching 32768 (rolling)", True, False),
    ("optimizer_history.csv", "sampled_alpha_median", "Median step size over all updated weights (mostly rarely used ones)", False, True),
)


def column(path: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        return np.zeros(0), np.zeros(0)
    rows = [row for row in csv.DictReader(path.open()) if row.get(name)]
    return np.array([int(row["games"]) for row in rows]), np.array([float(row[name]) for row in rows])


def rolling(values: np.ndarray, window: int = 20) -> np.ndarray:
    sums = np.cumsum(np.insert(values, 0, 0.0))
    counts = np.minimum(np.arange(1, len(values) + 1), window)
    return (sums[1:] - sums[np.maximum(np.arange(1, len(values) + 1) - window, 0)]) / counts


def draw(fig: plt.Figure) -> None:
    fig.clear()
    for ax, (file, name, title, smooth, log) in zip(fig.subplots(2, 2).flat, PANELS):
        for arm, (label, colour) in ARMS.items():
            games, values = column(OUTPUT / arm / file, name)
            if len(games):
                values = rolling(values) if smooth else values
                ax.plot(games / 1e3, values, color=colour, linewidth=1.5, label=f"{label}: {values[-1]:,.4g} at {games[-1] / 1e3:,.0f}k")
        ax.set_title(title)
        ax.set_xlabel("Games played (thousands)")
        if log:
            ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle("Stage 8A fork from 850k games", fontweight="bold")
    fig.tight_layout()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=float, default=10.0)
    args = parser.parse_args()
    fig = plt.figure(figsize=(13, 8))
    if args.once:
        draw(fig)
        fig.savefig(OUTPUT / "fork-progress.png", dpi=110)
        print(f"Saved {OUTPUT / 'fork-progress.png'}")
        return
    while plt.fignum_exists(fig.number):
        draw(fig)
        plt.pause(args.interval)


if __name__ == "__main__":
    main()
