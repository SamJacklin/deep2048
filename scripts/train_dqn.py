import argparse

from deep_2048.agents.dqn.trainer import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the 2048 DQN agent")
    parser.add_argument(
        "--resume",
        metavar="CHECKPOINT",
        default=None,
        help="Path to a checkpoint to resume training from (e.g. checkpoints/dqn/final.pt)",
    )
    args = parser.parse_args()

    summary = train(resume_from=args.resume)

    print("\nTraining complete")
    print(summary.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
