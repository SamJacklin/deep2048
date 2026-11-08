"""Agent registry: build any agent by name.

Imports are deferred so that, for example, running the heuristic doesn't import torch.
"""

from deep_2048.agents.base import Agent, Decision

DEFAULT_DQN_CHECKPOINT = "checkpoints/dqn/best.pt"

AGENT_NAMES: tuple[str, ...] = ("dqn", "dqn_search", "heuristic")


def load_agent(name: str, *, checkpoint: str | None = None, depth: int | None = None) -> Agent:
    """Build an agent by registry name.

    Args:
        name:       One of AGENT_NAMES.
        checkpoint: Model checkpoint for learned agents (default: DEFAULT_DQN_CHECKPOINT).
        depth:      Search depth for search-based agents (default: 2).
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

    raise ValueError(f"Unknown agent {name!r}. Choose from: {', '.join(AGENT_NAMES)}")


__all__ = ["Agent", "Decision", "AGENT_NAMES", "DEFAULT_DQN_CHECKPOINT", "load_agent"]
