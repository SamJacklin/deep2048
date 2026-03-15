"""Show training results from output/training_history.csv."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def load_csv(path: str) -> dict[str, list]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    def parse(val: str) -> float | None:
        return float(val) if val != "" else None

    columns: dict[str, list] = {key: [] for key in rows[0]}
    for row in rows:
        for key in columns:
            columns[key].append(parse(row[key]))
    return columns


def main(csv_path: str = "output/training_history.csv", window: int = 50) -> None:
    data = load_csv(csv_path)

    steps = np.array(data["training_step"], dtype=float)
    scores = np.array(data["episode_score"], dtype=float)

    # Eval rows only
    eval_mask = [v is not None for v in data["eval_average_score"]]
    eval_steps = steps[eval_mask]
    eval_scores = np.array([v for v in data["eval_average_score"] if v is not None])
    eval_tiles = np.array([v for v in data["eval_average_max_tile"] if v is not None])

    # Loss rows only
    loss_mask = [v is not None for v in data["loss"]]
    loss_steps = steps[loss_mask]
    losses = np.array([v for v in data["loss"] if v is not None])

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Training Results", fontsize=14, fontweight="bold")

    # --- Eval average score ---
    ax = axes[0, 0]
    ax.plot(eval_steps, eval_scores, marker="o", linewidth=2, color="steelblue")
    ax.set_title("Eval Average Score")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Score")
    ax.grid(True, alpha=0.3)

    # --- Eval average max tile ---
    ax = axes[0, 1]
    ax.plot(eval_steps, eval_tiles, marker="o", linewidth=2, color="darkorange")
    for tile in [64, 128, 256, 512, 1024]:
        if eval_tiles.max() >= tile * 0.5:
            ax.axhline(tile, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.text(eval_steps[0], tile + 2, str(tile), fontsize=7, color="gray")
    ax.set_title("Eval Average Max Tile")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Tile value")
    ax.grid(True, alpha=0.3)

    # --- Episode score (rolling mean) ---
    ax = axes[1, 0]
    ax.plot(steps, scores, color="lightsteelblue", linewidth=0.5, alpha=0.4, label="raw")
    if len(scores) >= window:
        smoothed = rolling_mean(scores, window)
        ax.plot(steps[window - 1 :], smoothed, color="steelblue", linewidth=1.5, label=f"rolling mean ({window})")
    ax.set_title("Episode Score")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Score")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Loss ---
    ax = axes[1, 1]
    ax.plot(loss_steps, losses, color="firebrick", linewidth=0.7, alpha=0.4)
    if len(losses) >= window:
        smoothed_loss = rolling_mean(losses, window)
        ax.plot(loss_steps[window - 1 :], smoothed_loss, color="darkred", linewidth=1.5)
    ax.set_title("Training Loss")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Huber loss")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    out_path = Path("output") / "training_results.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")

    # Print summary table
    print("\n--- Eval progression ---")
    print(f"{'Step':>10}  {'Avg score':>12}  {'Avg max tile':>14}")
    print("-" * 42)
    for step, score, tile in zip(eval_steps, eval_scores, eval_tiles):
        print(f"{int(step):>10}  {score:>12.1f}  {tile:>14.1f}")

    best_idx = int(np.argmax(eval_scores))
    print(f"\nBest checkpoint: step {int(eval_steps[best_idx])}  "
          f"score={eval_scores[best_idx]:.1f}  avg_max_tile={eval_tiles[best_idx]:.1f}")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="output/training_history.csv")
    parser.add_argument("--window", type=int, default=50, help="Rolling mean window for raw episode score and loss")
    args = parser.parse_args()
    main(csv_path=args.csv, window=args.window)
