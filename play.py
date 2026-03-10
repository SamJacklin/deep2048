import argparse
import time

import torch

from deep_2048.game_logic.game import Game2048
from deep_2048.model.agent import DQNAgent


def play(checkpoint_path: str, render: bool, delay: float) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(device=device)
    agent.load(checkpoint_path)
    agent.set_eval_mode()

    game = Game2048()
    state = game.reset()

    move = 0
    while not game.is_game_over():
        if render:
            print(f"\nMove {move}")
            game.render()
            if delay > 0:
                time.sleep(delay)

        valid_actions = game.get_valid_actions()
        selection = agent.select_action(state=state, valid_actions=valid_actions, epsilon=0.0)
        step_result = game.step(selection.action)
        state = step_result.state
        move += 1

    print(f"\n--- Game over ---")
    game.render()
    print(f"Score:    {game.score}")
    print(f"Max tile: {game.max_tile}")
    print(f"Moves:    {move}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a trained 2048 agent")
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default="checkpoints/final.pt",
        help="Path to checkpoint file (default: checkpoints/final.pt)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the board after each move",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Seconds to pause between moves when rendering (default: 0.1)",
    )
    args = parser.parse_args()

    play(checkpoint_path=args.checkpoint, render=args.render, delay=args.delay)


if __name__ == "__main__":
    main()
