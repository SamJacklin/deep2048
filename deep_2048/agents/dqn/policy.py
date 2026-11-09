"""Play-time wrappers that expose a trained DQN through the common Agent interface.

`DQNAgent` (agent.py) is the learner: networks, optimiser, replay updates. These classes
load a checkpoint into one and use it purely for choosing moves.
"""

from pathlib import Path

import numpy as np
import torch

from deep_2048.agents.base import Agent, Decision
from deep_2048.agents.dqn.agent import DQNAgent, get_device
from deep_2048.agents.dqn.search import dqn_search_select
from deep_2048.game.constants import Action
from deep_2048.game.encoding import encode_board


def load_dqn(checkpoint: str | Path, device: torch.device | None = None) -> DQNAgent:
    learner = DQNAgent(device=device or get_device())
    learner.load(Path(checkpoint))
    learner.set_eval_mode()
    return learner


class DQNPolicy(Agent):
    """Greedy DQN: pick the valid action with the highest Q-value."""

    def __init__(self, checkpoint: str | Path, device: torch.device | None = None) -> None:
        self.learner = load_dqn(checkpoint, device)
        self.checkpoint = Path(checkpoint)
        self.label = f"DQN ({self.checkpoint.stem})"

    def act(self, board: np.ndarray, valid_actions: list[Action]) -> Decision:
        result = self.learner.select_action(state=encode_board(board).vector, valid_actions=valid_actions, epsilon=0.0)
        scores = {action.name: q for action, q in zip(Action, result.q_values)}
        return Decision(action=result.action, scores=scores)


class DQNSearchPolicy(Agent):
    """Expectimax search with the DQN's max-Q as the leaf evaluator."""

    def __init__(self, checkpoint: str | Path, depth: int = 2, device: torch.device | None = None) -> None:
        self.learner = load_dqn(checkpoint, device)
        self.checkpoint = Path(checkpoint)
        self.depth = depth
        self.label = f"DQN+Search depth={depth} ({self.checkpoint.stem})"

    def act(self, board: np.ndarray, valid_actions: list[Action]) -> Decision:
        action, scores = dqn_search_select(self.learner, board, valid_actions, depth=self.depth)
        return Decision(action=action, scores=scores)
