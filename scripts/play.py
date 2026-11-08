"""Play a single game with a trained agent and render it to the terminal.

Usage examples:
    uv run python scripts/play.py                           # DQN best checkpoint
    uv run python scripts/play.py --agent dqn_search        # DQN + depth-1 expectimax
    uv run python scripts/play.py --agent heuristic         # pure heuristic expectimax
    uv run python scripts/play.py --agent heuristic --depth 3
    uv run python scripts/play.py checkpoints/dqn/step_350068.pt --render --delay 0.2
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from deep_2048.agents import AGENT_NAMES, DEFAULT_DQN_CHECKPOINT, load_agent
from deep_2048.game.game import Game2048


def play(name: str, depth: int, checkpoint: str, render: bool, delay: float) -> None:
    agent = load_agent(name, checkpoint=checkpoint, depth=depth)

    game = Game2048()
    game.reset()
    move = 0

    while not game.is_game_over():
        if render:
            print(f"\nMove {move}")
            game.render()
            if delay > 0:
                time.sleep(delay)

        decision = agent.act(game.get_board(), game.get_valid_actions())
        game.step(decision.action)
        move += 1

    print("\n--- Game over ---")
    game.render()
    print(f"Score:    {game.score}")
    print(f"Max tile: {game.max_tile}")
    print(f"Moves:    {move}")
    print(f"Agent:    {agent.label}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Play a single 2048 game with a trained agent")
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default=DEFAULT_DQN_CHECKPOINT,
        help=f"Path to .pt checkpoint for learned agents (default: {DEFAULT_DQN_CHECKPOINT})",
    )
    parser.add_argument(
        "--agent",
        choices=AGENT_NAMES,
        default="dqn",
        help="Agent to play (default: dqn)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Search depth — default 1 for dqn_search, 2 for heuristic",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Print the board after each move",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Seconds to wait between moves when rendering (default: 0.1)",
    )
    args = parser.parse_args()

    if args.depth is None:
        args.depth = 1 if args.agent == "dqn_search" else 2

    play(
        name=args.agent,
        depth=args.depth,
        checkpoint=args.checkpoint,
        render=args.render,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
