from deep_2048.model.trainer import train


def main() -> None:
    summary = train()

    print("\nTraining complete")
    print(summary.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
