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
