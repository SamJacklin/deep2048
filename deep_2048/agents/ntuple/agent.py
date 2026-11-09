from pathlib import Path

import numpy as np

from deep_2048.agents.base import Agent, Decision
from deep_2048.agents.ntuple.network import NTupleNetwork
from deep_2048.game import bitboard as bb
from deep_2048.game.constants import Action


class NTupleAgent(Agent):
    """Greedy agent that plays the move with the highest reward + V(afterstate).

    Invalid moves are left out of `scores` since they have no afterstate.
    """

    def __init__(self, checkpoint: str | Path) -> None:
        self.network, metadata = NTupleNetwork.load(checkpoint)
        self.checkpoint = Path(checkpoint)
        self.label = f"N-tuple TD ({metadata.games_played:,} games, lr {metadata.learning_rate:g})"

    def act(self, board: np.ndarray, valid_actions: list[Action]) -> Decision:
        packed = bb.from_array(board)
        scores: dict[str, float] = {}
        for action in valid_actions:
            afterstate, reward = bb.move(packed, int(action))
            scores[action.name] = reward + self.network.value(np.uint64(afterstate))

        best = max(valid_actions, key=lambda action: scores[action.name])
        return Decision(action=best, scores=scores)
