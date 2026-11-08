import numpy as np

from deep_2048.agents.base import Agent, Decision
from deep_2048.agents.heuristic.expectimax import heuristic_select
from deep_2048.game.constants import Action


class HeuristicAgent(Agent):
    """Depth-N expectimax over a hand-crafted board evaluator. No learning."""

    def __init__(self, depth: int = 2) -> None:
        self.depth = depth
        self.label = f"Heuristic depth={depth}"

    def act(self, board: np.ndarray, valid_actions: list[Action]) -> Decision:
        action, scores = heuristic_select(board, valid_actions, depth=self.depth)
        return Decision(action=action, scores=scores)
