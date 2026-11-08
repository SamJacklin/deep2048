"""Generate game replay JSON files for the frontend visualiser.

Produces one replay per agent:
  - Four DQN training checkpoints (progression)
  - Pure heuristic depth-2 expectimax (no neural network)

Usage:
    uv run python scripts/generate_replays.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from deep_2048.game.board import initialise_board, spawn_random_tile
from deep_2048.game.constants import Action, SPAWN_TILE_WEIGHTS
from deep_2048.game.encoding import encode_board
from deep_2048.game.moves import apply_move, get_valid_actions, is_game_over
from deep_2048.agents.dqn.agent import DQNAgent
from deep_2048.agents.heuristic.expectimax import heuristic_select

OUTPUT_DIR = Path("frontend/public/replays")
MAX_FRAMES = 1400
GAME_SEED = 1


# ── Checkpoint / agent definitions ───────────────────────────────────────────

DQN_CHECKPOINTS = [
    {
        "name": "step_100083",
        "label": "Early Training",
        "description": "100K steps — basic merges learned, heavy exploration",
        "agent_type": "drl",
        "search_depth": None,
        "score_label": "Q-Values",
        "pt": "checkpoints/dqn/step_100083.pt",
        "json": "checkpoints/dqn/step_100083.json",
    },
    {
        "name": "step_350068",
        "label": "Mid Training",
        "description": "350K steps — 256 tile emerging consistently",
        "agent_type": "drl",
        "search_depth": None,
        "score_label": "Q-Values",
        "pt": "checkpoints/dqn/step_350068.pt",
        "json": "checkpoints/dqn/step_350068.json",
    },
    {
        "name": "step_625066",
        "label": "Late Training",
        "description": "625K steps — 512 reached regularly, strategy consolidating",
        "agent_type": "drl",
        "search_depth": None,
        "score_label": "Q-Values",
        "pt": "checkpoints/dqn/step_625066.pt",
        "json": "checkpoints/dqn/step_625066.json",
    },
    {
        "name": "best",
        "label": "Best DQN",
        "description": "925K steps — peak performance: ≥512 in 61%, ≥1024 in 9%",
        "agent_type": "drl",
        "search_depth": None,
        "score_label": "Q-Values",
        "pt": "checkpoints/dqn/best.pt",
        "json": "checkpoints/dqn/best.json",
    },
]

COMPARISON_AGENTS = [
    {
        "name": "heuristic",
        "label": "Heuristic Search",
        "description": "Depth-2 expectimax with hand-crafted board evaluator — no learning",
        "agent_type": "heuristic",
        "search_depth": 2,
        "score_label": "Heuristic Scores",
        "pt": None,
        "json": None,
        "seed": 1,
    },
]


# ── Action-selection functions (replay-specific) ──────────────────────────────

def _get_q_values(agent: DQNAgent, board: np.ndarray) -> dict[str, float]:
    encoded = encode_board(board).vector
    state_tensor = torch.tensor(encoded, dtype=torch.float32, device=agent.device).unsqueeze(0)
    with torch.no_grad():
        q = agent.online_network(state_tensor).squeeze(0).cpu().tolist()
    return {a.name: round(q[i], 4) for i, a in enumerate(Action)}


def _dqn_move(
    agent: DQNAgent, board: np.ndarray, valid_actions: list[Action]
) -> tuple[Action, dict[str, float]]:
    """Greedy DQN: pick the action with the highest Q-value."""
    q_vals = _get_q_values(agent, board)
    chosen = max(valid_actions, key=lambda a: q_vals[a.name])
    return chosen, q_vals


# ── Generic game runner ───────────────────────────────────────────────────────

def run_game(move_fn, seed: int = GAME_SEED) -> tuple[list[dict], int, int]:
    """Play a full game with move_fn(board, valid_actions) -> (Action, scores_dict).

    Returns (frames, final_score, max_tile).
    """
    rng = np.random.default_rng(seed)
    board = initialise_board(board_size=4, rng=rng, tile_weights=SPAWN_TILE_WEIGHTS)
    score = 0
    frames: list[dict] = []
    move_number = 0

    while not is_game_over(board) and len(frames) < MAX_FRAMES:
        valid_actions = get_valid_actions(board)
        if not valid_actions:
            break

        chosen, scores = move_fn(board, valid_actions)

        frames.append({
            "board": board.tolist(),
            "score": score,
            "action": chosen.name,
            "q_values": scores,
            "valid_actions": [v.name for v in valid_actions],
            "reward": 0,
            "move_number": move_number,
        })

        move_result = apply_move(board, chosen)
        if not move_result.changed:
            break

        board = move_result.board
        score += move_result.reward
        frames[-1]["reward"] = move_result.reward
        move_number += 1

        spawn_result = spawn_random_tile(board=board, rng=rng, tile_weights=SPAWN_TILE_WEIGHTS)
        board = spawn_result.board

    # Terminal frame — game-over state, no action.
    if frames:
        frames.append({
            "board": board.tolist(),
            "score": score,
            "action": None,
            "q_values": {a.name: 0.0 for a in Action},
            "valid_actions": [],
            "reward": 0,
            "move_number": move_number,
        })

    return frames, score, int(board.max())


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).parent.parent

    for cp in DQN_CHECKPOINTS + COMPARISON_AGENTS:
        print(f"\n{'─' * 50}")
        print(f"  {cp['label']}  ({cp['agent_type']})")
        print(f"{'─' * 50}")

        agent: DQNAgent | None = None
        if cp["pt"]:
            pt_path = root / cp["pt"]
            if not pt_path.exists():
                print(f"  Skipping — checkpoint not found: {pt_path}")
                continue
            agent = DQNAgent()
            agent.load(pt_path)
            agent.set_eval_mode()
            print(f"  Loaded: {cp['pt']}")

        meta: dict = {"training_step": 0, "epsilon": 0.0, "average_score": 0.0, "average_max_tile": 0.0}
        if cp["json"]:
            meta_path = root / cp["json"]
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)

        agent_type = cp["agent_type"]
        if agent_type == "drl":
            _agent = agent
            def move_fn(board, valid, a=_agent):
                return _dqn_move(a, board, valid)
        else:  # heuristic
            _depth = cp["search_depth"]
            def move_fn(board, valid, d=_depth):
                return heuristic_select(board, valid, d)

        game_seed = cp.get("seed", GAME_SEED)
        print("  Running game…")
        frames, final_score, max_tile = run_game(move_fn, seed=game_seed)
        print(f"  → {len(frames)} frames  score={final_score}  max_tile={max_tile}")

        replay = {
            "checkpoint_name": cp["name"],
            "checkpoint_label": cp["label"],
            "checkpoint_description": cp["description"],
            "agent_type": cp["agent_type"],
            "search_depth": cp["search_depth"],
            "score_label": cp["score_label"],
            "training_step": meta.get("training_step", 0),
            "epsilon": meta.get("epsilon", 0.0),
            "average_score": meta.get("average_score", 0.0),
            "average_max_tile": meta.get("average_max_tile", 0.0),
            "final_score": final_score,
            "max_tile_achieved": max_tile,
            "total_moves": len(frames) - 1,
            "frames": frames,
        }

        out_path = OUTPUT_DIR / f"{cp['name']}.json"
        with open(out_path, "w") as f:
            json.dump(replay, f, separators=(",", ":"))

        size_kb = out_path.stat().st_size / 1024
        print(f"  → Saved: {out_path} ({size_kb:.1f} KB)")

    print(f"\n{'─' * 50}")
    print("  Done.")
    print(f"{'─' * 50}\n")


if __name__ == "__main__":
    main()
