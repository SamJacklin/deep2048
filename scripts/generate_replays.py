"""Generate game replay JSON files for the frontend visualiser.

Runs one greedy game (epsilon=0) per checkpoint and records every frame
including board state, chosen action, Q-values, and score.

Usage:
    uv run python scripts/generate_replays.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path so deep_2048 is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from deep_2048.game_logic.constants import Action
from deep_2048.game_logic.encoding import encode_board
from deep_2048.model.agent import DQNAgent

CHECKPOINTS = [
    {
        "name": "step_25143",
        "label": "Early Training",
        "description": "25K steps — agent is mostly still exploring",
        "pt": "checkpoints/step_25143.pt",
        "json": "checkpoints/step_25143.json",
    },
    {
        "name": "step_75041",
        "label": "Mid Training",
        "description": "75K steps — patterns starting to emerge",
        "pt": "checkpoints/step_75041.pt",
        "json": "checkpoints/step_75041.json",
    },
    {
        "name": "step_125198",
        "label": "Late Training",
        "description": "125K steps — solid strategy visible",
        "pt": "checkpoints/step_125198.pt",
        "json": "checkpoints/step_125198.json",
    },
    {
        "name": "best",
        "label": "Best Agent",
        "description": "150K steps — highest average score achieved",
        "pt": "checkpoints/best.pt",
        "json": "checkpoints/best.json",
    },
]

OUTPUT_DIR = Path("frontend/public/replays")
MAX_FRAMES = 600  # cap replay length for file size


def get_q_values(agent: DQNAgent, board: np.ndarray) -> dict[str, float]:
    encoded = encode_board(board).vector
    state_tensor = torch.tensor(encoded, dtype=torch.float32, device=agent.device).unsqueeze(0)
    with torch.no_grad():
        q = agent.online_network(state_tensor).squeeze(0).cpu().tolist()
    actions = [a.name for a in Action]
    return {a: round(q[i], 4) for i, a in enumerate(actions)}


def run_game(agent: DQNAgent) -> list[dict]:
    from deep_2048.game_logic.board import initialise_board, spawn_random_tile
    from deep_2048.game_logic.constants import SPAWN_TILE_WEIGHTS
    from deep_2048.game_logic.moves import apply_move, get_valid_actions, is_game_over

    rng = np.random.default_rng(0)
    board_size = 4

    board = initialise_board(board_size=board_size, rng=rng, tile_weights=SPAWN_TILE_WEIGHTS)
    score = 0
    frames = []
    move_number = 0

    while not is_game_over(board) and len(frames) < MAX_FRAMES:
        valid_actions = get_valid_actions(board)
        if not valid_actions:
            break

        q_vals = get_q_values(agent, board)

        # Greedy: mask invalid actions, pick best
        masked = {a: (q_vals[a] if a in [v.name for v in valid_actions] else float("-inf"))
                  for a in q_vals}
        chosen = max(masked, key=lambda a: masked[a])
        action = Action[chosen]

        frames.append({
            "board": board.tolist(),
            "score": score,
            "action": chosen,
            "q_values": q_vals,
            "valid_actions": [v.name for v in valid_actions],
            "reward": 0,  # filled in after move
            "move_number": move_number,
        })

        move_result = apply_move(board, action)
        if not move_result.changed:
            frames[-1]["reward"] = 0
            # still progresses to avoid infinite loop — shouldn't normally happen
            # as we only pick valid actions
            break

        board = move_result.board
        reward = move_result.reward
        score += reward
        frames[-1]["reward"] = reward
        move_number += 1

        spawn_result = spawn_random_tile(board=board, rng=rng, tile_weights=SPAWN_TILE_WEIGHTS)
        board = spawn_result.board

    # Append a final frame (game over state, no action)
    if frames:
        frames.append({
            "board": board.tolist(),
            "score": score,
            "action": None,
            "q_values": get_q_values(agent, board),
            "valid_actions": [],
            "reward": 0,
            "move_number": move_number,
        })

    return frames, score, int(board.max())


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).parent.parent

    for cp in CHECKPOINTS:
        pt_path = root / cp["pt"]
        meta_path = root / cp["json"]

        if not pt_path.exists():
            print(f"  Skipping {cp['name']} — checkpoint not found at {pt_path}")
            continue

        print(f"Loading {cp['name']} …")
        agent = DQNAgent()
        agent.load(pt_path)
        agent.set_eval_mode()

        with open(meta_path) as f:
            meta = json.load(f)

        frames, final_score, max_tile = run_game(agent)
        print(f"  → {len(frames)} frames, score={final_score}, max_tile={max_tile}")

        replay = {
            "checkpoint_name": cp["name"],
            "checkpoint_label": cp["label"],
            "checkpoint_description": cp["description"],
            "training_step": meta["training_step"],
            "epsilon": meta["epsilon"],
            "average_score": meta["average_score"],
            "average_max_tile": meta["average_max_tile"],
            "final_score": final_score,
            "max_tile_achieved": max_tile,
            "total_moves": len(frames) - 1,
            "frames": frames,
        }

        out_path = OUTPUT_DIR / f"{cp['name']}.json"
        with open(out_path, "w") as f:
            json.dump(replay, f, separators=(",", ":"))

        size_kb = out_path.stat().st_size / 1024
        print(f"  → Saved to {out_path} ({size_kb:.1f} KB)")

    print("\nDone.")


if __name__ == "__main__":
    main()
