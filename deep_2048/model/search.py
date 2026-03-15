"""DQN + expectimax search action selection.

Wraps the trained DQN in a depth-2 expectimax search.

Why depth 1 fails:
  The DQN is trained with n-step returns where every training transition's
  next_state is already a post-spawn board, so Q(s,a) implicitly contains the
  tile-spawn expectation one step ahead. Depth-1 search just replicates that
  computation noisily without adding information.

  Depth 2 looks two full player-moves ahead, genuinely extending the DQN's
  planning horizon and giving the search real signal.

Batching strategy:
  For each root action we collect every possible leaf board across the full
  depth-2 tree (action → spawn → action → spawn → leaf) in a single flat list,
  run one forward pass, then reconstruct the max/expectation bottom-up.
  The network is called exactly once per root action.

Leaf evaluation:
  max_a Q(s,a) over *valid* actions only. Invalid actions are masked to -inf
  before taking the max so they cannot inflate the leaf estimate.
"""

import numpy as np
import torch

from deep_2048.game_logic.constants import Action
from deep_2048.game_logic.encoding import encode_board
from deep_2048.game_logic.moves import apply_move, get_valid_actions
from deep_2048.model.agent import DQNAgent

_SPAWN_PAIRS = ((2, 0.9), (4, 0.1))


def _spawn_outcomes(board: np.ndarray) -> tuple[list[np.ndarray], list[float]]:
    """Return (boards, probabilities) for all possible tile spawns."""
    empty = [(r, c) for r in range(4) for c in range(4) if board[r, c] == 0]
    if not empty:
        return [board], [1.0]
    p = 1.0 / len(empty)
    boards, weights = [], []
    for r, c in empty:
        for tile, tile_p in _SPAWN_PAIRS:
            b = board.copy()
            b[r, c] = tile
            boards.append(b)
            weights.append(p * tile_p)
    return boards, weights


def _batch_leaf_values(agent: DQNAgent, boards: list[np.ndarray]) -> np.ndarray:
    """Evaluate boards in one forward pass; mask invalid actions before max."""
    encoded = np.stack([encode_board(b).vector for b in boards])
    t = torch.tensor(encoded, dtype=torch.float32, device=agent.device)
    with torch.no_grad():
        q = agent.online_network(t).cpu().numpy()   # (N, 4)

    for i, b in enumerate(boards):
        valid = get_valid_actions(b)
        if valid:
            for a in Action:
                if a not in valid:
                    q[i, int(a)] = -np.inf

    return q.max(axis=1)   # (N,)


def dqn_search_select(
    agent: DQNAgent,
    board: np.ndarray,
    valid_actions: list[Action],
    depth: int = 2,
) -> tuple[Action, dict[str, float]]:
    """Choose an action using depth-2 expectimax with the DQN as leaf evaluator.

    Tree structure:
        root action → spawn layer 1 → depth-1 action → spawn layer 2 → evaluate

    Args:
        agent:         Trained DQNAgent (must be in eval mode).
        board:         Raw 4x4 board (tile values, 0 = empty).
        valid_actions: Actions that change the board state.
        depth:         Search depth (default 2).

    Returns:
        (chosen_action, scores) where scores maps every action name to its
        expected leaf value (0.0 for invalid/unchanged root actions).
    """
    valid_set = set(valid_actions)
    scores: dict[str, float] = {}

    for root_action in Action:
        if root_action not in valid_set:
            scores[root_action.name] = 0.0
            continue

        result0 = apply_move(board, root_action)
        if not result0.changed:
            scores[root_action.name] = 0.0
            continue

        spawn1_boards, spawn1_weights = _spawn_outcomes(result0.board)

        # ── Collect every leaf board across the whole tree for this root action ──
        # Structure per spawn-1 outcome: list of (a1, start, end, spawn2_weights)
        # where [start:end] indexes into all_leaves.
        all_leaves: list[np.ndarray] = []
        # Each entry: (spawn1_weight, [(a1_tag, slice_start, slice_end, sp2_weights)])
        tree: list[tuple[float, list[tuple]]] = []

        for sp1_board, w1 in zip(spawn1_boards, spawn1_weights):
            d1_actions = get_valid_actions(sp1_board)

            if not d1_actions or depth == 1:
                # Leaf is the spawn-1 board itself.
                idx = len(all_leaves)
                all_leaves.append(sp1_board)
                tree.append((w1, [(None, idx, idx + 1, [1.0])]))
                continue

            action_entries = []
            for a1 in d1_actions:
                res1 = apply_move(sp1_board, a1)
                if not res1.changed:
                    continue
                sp2_boards, sp2_weights = _spawn_outcomes(res1.board)
                start = len(all_leaves)
                all_leaves.extend(sp2_boards)
                action_entries.append((a1, start, len(all_leaves), sp2_weights))

            if not action_entries:
                idx = len(all_leaves)
                all_leaves.append(sp1_board)
                tree.append((w1, [(None, idx, idx + 1, [1.0])]))
            else:
                tree.append((w1, action_entries))

        if not all_leaves:
            scores[root_action.name] = 0.0
            continue

        leaf_vals = _batch_leaf_values(agent, all_leaves)

        # ── Reconstruct: expectation over spawn1, max over a1, expectation over spawn2 ──
        score = 0.0
        for w1, action_entries in tree:
            best = float("-inf")
            for _, start, end, sp2_weights in action_entries:
                v = float(np.dot(sp2_weights, leaf_vals[start:end]))
                if v > best:
                    best = v
            score += w1 * (best if best > float("-inf") else 0.0)

        scores[root_action.name] = round(score, 4)

    chosen = max(valid_set, key=lambda a: scores[a.name])
    return chosen, scores
