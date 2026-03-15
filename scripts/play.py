"""Play a single game with a trained agent and render it to the terminal.

Usage examples:
    uv run python scripts/play.py                           # DQN best checkpoint
    uv run python scripts/play.py --mode drl_search         # DQN + depth-1 expectimax
    uv run python scripts/play.py --mode heuristic          # pure heuristic expectimax
    uv run python scripts/play.py --mode heuristic --depth 3
    uv run python scripts/play.py checkpoints/step_350068.pt --render --delay 0.2
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from deep_2048.game_logic.game import Game2048
from deep_2048.model.agent import DQNAgent, get_device
from deep_2048.model.heuristic import heuristic_select
from deep_2048.model.search import dqn_search_select


def play(mode: str, depth: int, checkpoint: str, render: bool, delay: float) -> None:
    agent: DQNAgent | None = None
    if mode in ("drl", "drl_search"):
        device = get_device()
        agent = DQNAgent(device=device)
        agent.load(Path(checkpoint))
        agent.set_eval_mode()

    game = Game2048()
    game.reset()
    move = 0

    while not game.is_game_over():
        if render:
            print(f"\nMove {move}")
            game.render()
            if delay > 0:
                time.sleep(delay)

        valid_actions = game.get_valid_actions()

        if mode == "drl":
            state = game.get_state()
            action = agent.select_action(state=state, valid_actions=valid_actions, epsilon=0.0).action
        elif mode == "drl_search":
            action, _ = dqn_search_select(agent, game.board, valid_actions, depth=depth)
        else:  # heuristic
            action, _ = heuristic_select(game.board, valid_actions, depth=depth)

        game.step(action)
        move += 1

    print("\n--- Game over ---")
    game.render()
    print(f"Score:    {game.score}")
    print(f"Max tile: {game.max_tile}")
    print(f"Moves:    {move}")
    print(f"Mode:     {mode}" + (f" (depth {depth})" if mode != "drl" else ""))


def main() -> None:
    parser = argparse.ArgumentParser(description="Play a single 2048 game with a trained agent")
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default="checkpoints/best.pt",
        help="Path to .pt checkpoint (default: checkpoints/best.pt)",
    )
    parser.add_argument(
        "--mode",
        choices=["drl", "drl_search", "heuristic"],
        default="drl",
        help="Agent mode (default: drl)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Search depth — default 1 for drl_search, 2 for heuristic",
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
        args.depth = 1 if args.mode == "drl_search" else 2

    play(
        mode=args.mode,
        depth=args.depth,
        checkpoint=args.checkpoint,
        render=args.render,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
