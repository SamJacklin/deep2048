import random
from collections import deque

import numpy as np
import torch

from deep_2048.model.models import ReplayBatch, Transition


class ReplayBuffer:
    def __init__(self, capacity: int, device: torch.device) -> None:
        self.capacity = capacity
        self.device = device
        self._buffer: deque[Transition] = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self._buffer.append(transition)

    def sample(self, batch_size: int) -> ReplayBatch:
        transitions = random.sample(self._buffer, batch_size)

        states = np.stack([transition.state for transition in transitions]).astype(np.float32)
        actions = np.array([int(transition.action) for transition in transitions], dtype=np.int64)
        rewards = np.array([transition.reward for transition in transitions], dtype=np.float32)
        next_states = np.stack([transition.next_state for transition in transitions]).astype(np.float32)
        dones = np.array([transition.done for transition in transitions], dtype=np.float32)

        return ReplayBatch(
            states=torch.tensor(states, dtype=torch.float32, device=self.device),
            actions=torch.tensor(actions, dtype=torch.long, device=self.device),
            rewards=torch.tensor(rewards, dtype=torch.float32, device=self.device),
            next_states=torch.tensor(next_states, dtype=torch.float32, device=self.device),
            dones=torch.tensor(dones, dtype=torch.float32, device=self.device),
        )

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def is_ready(self) -> bool:
        return len(self._buffer) > 0


class NStepBuffer:
    """Converts 1-step transitions into n-step transitions.

    For each transition at time t, accumulates the discounted sum of rewards
    over the next n steps and uses s_{t+n} as the bootstrap state. When the
    episode ends before n steps are available, the partial return is used and
    done=True is set (zeroing out the bootstrap term in the agent).
    """

    def __init__(self, n_steps: int, gamma: float) -> None:
        self.n_steps = n_steps
        self.gamma = gamma
        self._pending: deque[Transition] = deque()

    def push(self, transition: Transition) -> list[Transition]:
        """Add a raw transition; returns any n-step transitions that are ready."""
        self._pending.append(transition)

        ready: list[Transition] = []
        while len(self._pending) >= self.n_steps:
            ready.append(self._make_n_step())
        return ready

    def flush(self) -> list[Transition]:
        """Drain remaining transitions at episode end."""
        ready: list[Transition] = []
        while self._pending:
            ready.append(self._make_n_step())
        return ready

    def _make_n_step(self) -> Transition:
        pending = list(self._pending)
        n = min(self.n_steps, len(pending))

        # Truncate at first terminal within the window
        for i in range(n - 1):
            if pending[i].done:
                n = i + 1
                break

        accumulated_reward = sum(self.gamma**i * pending[i].reward for i in range(n))

        self._pending.popleft()

        return Transition(
            state=pending[0].state,
            action=pending[0].action,
            reward=accumulated_reward,
            next_state=pending[n - 1].next_state,
            done=pending[n - 1].done,
        )
