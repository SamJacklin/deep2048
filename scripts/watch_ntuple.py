"""Chart n-tuple training progress, redrawing as new rows land in the history CSV.

Usage examples:
    uv run python scripts/watch_ntuple.py            # live window, refreshes every 5s
    uv run python scripts/watch_ntuple.py --once     # save the chart and exit
    uv run python scripts/watch_ntuple.py --run lr-decay
    uv run python scripts/watch_ntuple.py --run lr-a lr-b lr-c  # overlay several runs

A single run's chart is also saved as training_progress.png next to its history, and an
overlay is saved to the --image path.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from deep_2048.agents.ntuple.config import get_ntuple_settings
from deep_2048.agents.ntuple.train import HISTORY_NAME
from deep_2048.experiments import EXPERIMENTS_DIR

BENCHMARK_PATH = Path("frontend/public/benchmark.json")
DQN_CHECKPOINT_INFO = Path("checkpoints/dqn/best.json")
REFERENCE_AGENTS = {"DQN (best)": "firebrick", "Heuristic depth=2": "seagreen"}
TILE_COLOURS = {2048: "goldenrod", 4096: "darkorange", 8192: "orangered", 16384: "purple"}
RUN_COLOURS = ("steelblue", "darkorange", "seagreen", "firebrick", "purple", "saddlebrown")


def load_reference_scores() -> dict[str, float]:
    """Return the benchmark average score of each reference agent, if the benchmark exists."""
    if not BENCHMARK_PATH.exists():
        return {}
    rows = json.loads(BENCHMARK_PATH.read_text())
    return {row["label"]: row["avg_score"] for row in rows if row["label"] in REFERENCE_AGENTS}


def load_dqn_training_moves() -> int | None:
    if not DQN_CHECKPOINT_INFO.exists():
        return None
    return json.loads(DQN_CHECKPOINT_INFO.read_text())["training_step"]


def draw(fig: plt.Figure, history: np.ndarray, references: dict[str, float], dqn_moves: int | None) -> None:
    fig.clear()
    (score_ax, tiles_ax), (experience_ax, speed_ax) = fig.subplots(2, 2)
    latest = history[-1]
    fig.suptitle(
        f"N-tuple training: {latest['games']:,.0f} games, {latest['moves'] / 1e6:,.0f}M moves, avg score {latest['avg_score']:,.0f} over the last batch",
        fontweight="bold",
    )

    for ax, x, label in ((score_ax, history["games"], "Games played"), (experience_ax, history["moves"], "Moves played (experience)")):
        ax.plot(x, history["avg_score"], color="steelblue", linewidth=1.5)
        for name, score in references.items():
            ax.axhline(score, color=REFERENCE_AGENTS[name], linestyle="--", linewidth=1, label=f"{name} benchmark ({score:,.0f})")
        ax.set_xscale("log")
        ax.set_xlabel(label)
        ax.set_ylabel("Average score")
        ax.grid(True, which="both", alpha=0.3)

    score_ax.set_title("Score by games played")
    score_ax.legend(fontsize=8, loc="upper left")
    experience_ax.set_title("Score by experience")
    if dqn_moves:
        experience_ax.axvline(dqn_moves, color=REFERENCE_AGENTS["DQN (best)"], linestyle=":", linewidth=1.5, label="DQN's training budget")
        experience_ax.legend(fontsize=8, loc="upper left")

    for tile, colour in TILE_COLOURS.items():
        tiles_ax.plot(history["games"], history[f"rate_{tile}"], color=colour, linewidth=1.5, label=f"≥{tile}")
    tiles_ax.set_title("Games reaching each tile")
    tiles_ax.set_xscale("log")
    tiles_ax.set_xlabel("Games played")
    tiles_ax.set_ylabel("% of games")
    tiles_ax.set_ylim(0, 100)
    tiles_ax.legend(fontsize=8, loc="upper left")
    tiles_ax.grid(True, which="both", alpha=0.3)

    speed_ax.plot(history["games"], history["avg_moves"] * history["games_per_second"] / 1e6, color="slategray", linewidth=1.5)
    speed_ax.set_title("Training speed")
    speed_ax.set_xscale("log")
    speed_ax.set_xlabel("Games played")
    speed_ax.set_ylabel("Million moves / second")
    speed_ax.set_ylim(bottom=0)
    speed_ax.grid(True, which="both", alpha=0.3)
    moves_ax = speed_ax.twinx()
    moves_ax.plot(history["games"], history["avg_moves"], color="tan", linewidth=1, alpha=0.8)
    moves_ax.set_ylabel("Moves per game", color="tan")

    fig.tight_layout()


def draw_comparison(fig: plt.Figure, histories: dict[str, np.ndarray]) -> None:
    """Overlay several runs, one colour each."""
    fig.clear()
    (score_ax, rate_8192_ax), (rate_4096_ax, speed_ax) = fig.subplots(2, 2)
    fig.suptitle("N-tuple training runs: " + ", ".join(histories), fontweight="bold")

    panels = (
        (score_ax, "avg_score", "Average score", None),
        (rate_8192_ax, "rate_8192", "% of games reaching 8192", (0, 100)),
        (rate_4096_ax, "rate_4096", "% of games reaching 4096", (0, 100)),
        (speed_ax, "games_per_second", "Games / second", None),
    )
    for ax, column, title, limits in panels:
        for (name, history), colour in zip(histories.items(), RUN_COLOURS):
            ax.plot(history["games"], history[column], color=colour, linewidth=1, alpha=0.9, label=f"{name} ({history[column][-1]:,.0f})")
        ax.set_title(title)
        ax.set_xlabel("Games played")
        if limits:
            ax.set_ylim(*limits)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()


def load_history(path: Path) -> np.ndarray | None:
    """Return the training history, or None if it's empty or the trainer is mid-write."""
    if not path.exists():
        return None
    try:
        history = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
    except ValueError:
        return None
    return history if history.size else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Chart n-tuple training progress")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--run", metavar="NAME", nargs="+", help="Chart runs recorded in experiments/NAME, overlaid if there are several")
    source.add_argument("--csv", type=Path, default=Path(get_ntuple_settings().OUTPUT_DIR) / HISTORY_NAME)
    parser.add_argument("--image", type=Path, default=Path("output/ntuple/comparison.png"), help="Where to save an overlay of several runs")
    parser.add_argument("--interval", type=float, default=5.0, help="Seconds between refreshes (default: 5)")
    parser.add_argument("--once", action="store_true", help="Save the chart once and exit")
    args = parser.parse_args()

    csv_paths = {name: EXPERIMENTS_DIR / name / HISTORY_NAME for name in args.run} if args.run else {args.csv.parent.name: args.csv}
    image_path = next(iter(csv_paths.values())).parent / "training_progress.png" if len(csv_paths) == 1 else args.image
    references = load_reference_scores()
    dqn_moves = load_dqn_training_moves()
    fig = plt.figure(figsize=(14, 8))
    last_modified: dict[str, float] = {}

    while True:
        modified = {name: path.stat().st_mtime for name, path in csv_paths.items() if path.exists()}
        if modified and modified != last_modified:
            # The trainer may be mid-write, in which case the next refresh tries again
            histories = {name: load_history(path) for name, path in csv_paths.items()}
            loaded = {name: history for name, history in histories.items() if history is not None}
            if loaded and len(loaded) == len(modified):
                if len(csv_paths) == 1:
                    draw(fig, next(iter(loaded.values())), references, dqn_moves)
                else:
                    draw_comparison(fig, loaded)
                image_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(image_path, dpi=110)
                last_modified = modified
        if args.once:
            print(f"Saved {image_path}")
            return
        if not plt.fignum_exists(fig.number):
            return
        plt.pause(args.interval)


if __name__ == "__main__":
    main()
