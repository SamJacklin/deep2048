"""Agent registry: build any agent by name.

Imports are deferred so that, for example, running the heuristic doesn't import torch.
"""

from deep_2048.agents.base import Agent, Decision

DEFAULT_DQN_CHECKPOINT = "checkpoints/dqn/best.pt"
DEFAULT_NTUPLE_CHECKPOINT = "checkpoints/ntuple/best.npz"

AGENT_NAMES: tuple[str, ...] = ("dqn", "dqn_search", "heuristic", "ntuple", "ntuple_search")


def load_agent(name: str, *, checkpoint: str | None = None, depth: int | None = None, cutoff: float = 0.0) -> Agent:
    """Build an agent by registry name.

    Args:
        name:       One of AGENT_NAMES.
        checkpoint: Checkpoint for learned agents (default: that agent's DEFAULT_*_CHECKPOINT).
        depth:      Search depth for search-based agents (default: 2).
        cutoff:     For ntuple_search, the probability below which spawn branches aren't expanded (default: 0, exact).
    """
    if name == "heuristic":
        from deep_2048.agents.heuristic import HeuristicAgent

        return HeuristicAgent(depth=depth or 2)

    if name == "dqn":
        from deep_2048.agents.dqn.policy import DQNPolicy

        return DQNPolicy(checkpoint or DEFAULT_DQN_CHECKPOINT)

    if name == "dqn_search":
        from deep_2048.agents.dqn.policy import DQNSearchPolicy

        return DQNSearchPolicy(checkpoint or DEFAULT_DQN_CHECKPOINT, depth=depth or 2)

    if name == "ntuple":
        from deep_2048.agents.ntuple import NTupleAgent

        return NTupleAgent(checkpoint or DEFAULT_NTUPLE_CHECKPOINT)

    if name == "ntuple_search":
        from deep_2048.agents.ntuple import NTupleAgent

        return NTupleAgent(checkpoint or DEFAULT_NTUPLE_CHECKPOINT, depth=depth or 2, cutoff=cutoff)

    raise ValueError(f"Unknown agent {name!r}. Choose from: {', '.join(AGENT_NAMES)}")


__all__ = ["Agent", "Decision", "AGENT_NAMES", "DEFAULT_DQN_CHECKPOINT", "DEFAULT_NTUPLE_CHECKPOINT", "load_agent"]
