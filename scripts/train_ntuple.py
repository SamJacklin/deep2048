"""Train the n-tuple TD agent.

Usage examples:
    uv run python scripts/train_ntuple.py                     # 100k games
    uv run python scripts/train_ntuple.py --games 20000
    uv run python scripts/train_ntuple.py --resume checkpoints/ntuple/latest.npz --games 200000
    uv run python scripts/train_ntuple.py --games 1000000 --snapshot-at 1000 10000 100000
    uv run python scripts/train_ntuple.py --run lr-decay --resume checkpoints/ntuple/games_1000000.npz

Without --run, outputs go to output/ntuple and checkpoints to checkpoints/ntuple. With it, the
history and manifest go to experiments/<run>/, to be committed, and checkpoints to checkpoints/ntuple/<run>/.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from deep_2048.agents.ntuple.config import get_ntuple_settings
from deep_2048.agents.ntuple.train import train
from deep_2048.experiments import EXPERIMENTS_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the 2048 n-tuple TD agent")
    parser.add_argument(
        "--run",
        metavar="NAME",
        default=None,
        help="Record this run in experiments/NAME, keeping checkpoints in checkpoints/ntuple/NAME",
    )
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
    parser.add_argument(
        "--snapshot-at",
        type=int,
        nargs="+",
        default=None,
        metavar="GAMES",
        help="Also save the network as games_<n>.npz at each of these game counts",
    )
    coherence = parser.add_mutually_exclusive_group()
    coherence.add_argument(
        "--reset-coherence",
        dest="coherence_scale",
        action="store_const",
        const=0.0,
        help="Start the temporal coherence sums from zero instead of the resumed checkpoint's, restoring every weight's full rate",
    )
    coherence.add_argument(
        "--coherence-scale",
        type=float,
        metavar="FACTOR",
        help="Multiply the resumed temporal coherence sums by FACTOR: rates stay as they were but adapt to new errors sooner",
    )
    args = parser.parse_args()

    settings = get_ntuple_settings()
    if args.run:
        run_dirs = {"OUTPUT_DIR": str(EXPERIMENTS_DIR / args.run), "CHECKPOINT_DIR": str(Path(settings.CHECKPOINT_DIR) / args.run)}
        settings = settings.model_copy(update=run_dirs)

    summary = train(resume_from=args.resume, total_games=args.games, snapshot_at=args.snapshot_at, settings=settings, coherence_scale=args.coherence_scale)

    print("\nTraining stopped" if summary.interrupted else "\nTraining complete")
    print(summary.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
