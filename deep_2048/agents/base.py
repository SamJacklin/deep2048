"""The interface every 2048 agent implements.

An agent is anything that looks at a board and picks a move: a hand-written search,
a trained network, or (later) a decision model or LLM behind an API. Keeping them all
behind one method means the evaluation harness never needs to know which kind it is.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from deep_2048.game.constants import Action


@dataclass(frozen=True)
class Decision:
    """A chosen move plus the per-action values the agent used to pick it.

    `scores` is keyed by action name (e.g. Q-values for the DQN, expected heuristic
    value for expectimax). Agents that don't produce per-action values leave it empty.
    """

    action: Action
    scores: dict[str, float] = field(default_factory=dict)


class Agent(ABC):
    label: str  # human-readable name used in results tables and benchmark.json

    @abstractmethod
    def act(self, board: np.ndarray, valid_actions: list[Action]) -> Decision:
        """Choose a move for `board` (raw 4x4 tile values, 0 = empty) from `valid_actions`."""
