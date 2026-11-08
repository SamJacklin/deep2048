"""Train the n-tuple TD agent.

Usage examples:
    uv run python scripts/train_ntuple.py                     # 100k games
    uv run python scripts/train_ntuple.py --games 20000
    uv run python scripts/train_ntuple.py --resume checkpoints/ntuple/latest.npz --games 200000
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from deep_2048.agents.ntuple.train import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the 2048 n-tuple TD agent")
    parser.add_argument(
        "--resume",
        metavar="CHECKPOINT",
        default=None,
        help="Checkpoint to resume from (e.g. checkpoints/ntuple/latest.npz)",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=None,
        help="Total games to train to, including games already in a resumed checkpoint (default: 100,000)",
    )
    args = parser.parse_args()

    summary = train(resume_from=args.resume, total_games=args.games)

    print("\nTraining stopped" if summary.interrupted else "\nTraining complete")
    print(summary.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
