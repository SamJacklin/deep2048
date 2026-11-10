from pathlib import Path

import numpy as np

from deep_2048.agents.base import Agent, Decision
from deep_2048.agents.ntuple.network import NTupleNetwork
from deep_2048.agents.ntuple.search import action_values
from deep_2048.game import bitboard as bb
from deep_2048.game.constants import Action


class NTupleAgent(Agent):
    """Plays the move with the best reward + value, searching `depth` moves ahead (1 is greedy).

    Invalid moves are left out of `scores` since they have no afterstate.
    """

    def __init__(self, checkpoint: str | Path, depth: int = 1) -> None:
        self.network, metadata = NTupleNetwork.load(checkpoint)
        self.checkpoint = Path(checkpoint)
        self.depth = depth
        rule = "TC " if metadata.temporal_coherence else ""
        search = f" + search depth={depth}" if depth > 1 else ""
        self.label = f"N-tuple TD ({metadata.games_played:,} games, {rule}lr {metadata.learning_rate:.2g}){search}"

    def act(self, board: np.ndarray, valid_actions: list[Action]) -> Decision:
        values = action_values(self.network.weights, self.network.layout, bb.from_array(board), self.depth)
        scores = {action.name: float(values[action]) for action in valid_actions}
        best = max(valid_actions, key=lambda action: scores[action.name])
        return Decision(action=best, scores=scores)
