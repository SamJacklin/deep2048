import random
from collections import deque

import numpy as np
import torch

from deep_2048.agents.dqn.config import get_dqn_settings
from deep_2048.agents.dqn.models import ReplayBatch, Transition
from deep_2048.game.constants import Action

settings = get_dqn_settings()

# ---------------------------------------------------------------------------
# Board symmetry augmentation
# ---------------------------------------------------------------------------
# 2048 has 8-fold symmetry: 4 rotations × 2 flips. Each entry is
# (rot_k, flip_horizontal, action_map) where action_map[original] = augmented.
#
# Action enum: UP=0, DOWN=1, LEFT=2, RIGHT=3
#
# For a 90° CCW rotation of the board:
#   UP→LEFT, DOWN→RIGHT, LEFT→DOWN, RIGHT→UP
# Combined transforms are applied as: flip first, then rotate.
_SYMMETRIES: list[tuple[int, bool, tuple[int, int, int, int]]] = [
    (0, False, (0, 1, 2, 3)),  # identity
    (1, False, (2, 3, 1, 0)),  # 90° CCW:         UP→LEFT,  DOWN→RIGHT, LEFT→DOWN,  RIGHT→UP
    (2, False, (1, 0, 3, 2)),  # 180°:             UP→DOWN,  DOWN→UP,   LEFT→RIGHT, RIGHT→LEFT
    (3, False, (3, 2, 0, 1)),  # 270° CCW:         UP→RIGHT, DOWN→LEFT, LEFT→UP,    RIGHT→DOWN
    (0, True,  (0, 1, 3, 2)),  # H-flip:           LEFT↔RIGHT
    (1, True,  (2, 3, 0, 1)),  # H-flip + 90° CCW
    (2, True,  (1, 0, 2, 3)),  # H-flip + 180° (= V-flip): UP↔DOWN
    (3, True,  (3, 2, 1, 0)),  # H-flip + 270° CCW
]


def _apply_random_symmetry(transition: Transition) -> Transition:
    k, flip, action_map = random.choice(_SYMMETRIES)
    if k == 0 and not flip:
        return transition

    board_size = settings.BOARD_SIZE
    num_levels = settings.NUM_TILE_LEVELS

    def transform(state: np.ndarray) -> np.ndarray:
        x = state.reshape(num_levels, board_size, board_size)
        if flip:
            x = np.flip(x, axis=2)
        if k > 0:
            x = np.rot90(x, k=k, axes=(1, 2))
        return np.ascontiguousarray(x).flatten()

    return Transition(
        state=transform(transition.state),
        action=Action(action_map[int(transition.action)]),
        reward=transition.reward,
        next_state=transform(transition.next_state),
        done=transition.done,
    )


# ---------------------------------------------------------------------------
# Sum tree for O(log n) priority sampling
# ---------------------------------------------------------------------------

class SumTree:
    """Binary sum tree with 1-indexed internal storage.

    Leaves are at indices ``capacity`` to ``2*capacity - 1``.
    The root (index 1) holds the total priority sum.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        # Index 0 unused; leaves at [capacity, 2*capacity).
        self.tree = np.zeros(2 * capacity, dtype=np.float64)
        self.data: list = [None] * capacity
        self.write = 0
        self.size = 0

    def _propagate(self, leaf_idx: int, change: float) -> None:
        idx = (leaf_idx + self.capacity) >> 1  # parent of leaf
        while idx >= 1:
            self.tree[idx] += change
            idx >>= 1

    def add(self, priority: float, data: object) -> None:
        old = self.tree[self.capacity + self.write]
        self.tree[self.capacity + self.write] = priority
        self._propagate(self.write, priority - old)
        self.data[self.write] = data
        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, leaf_idx: int, priority: float) -> None:
        old = self.tree[self.capacity + leaf_idx]
        self.tree[self.capacity + leaf_idx] = priority
        self._propagate(leaf_idx, priority - old)

    @property
    def total(self) -> float:
        return float(self.tree[1])

    def get(self, s: float) -> tuple[int, float, object]:
        """Return (leaf_idx, priority, data) for the leaf reached by value s."""
        idx = 1
        while idx < self.capacity:
            left = 2 * idx
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = left + 1
        leaf_idx = idx - self.capacity
        return leaf_idx, float(self.tree[idx]), self.data[leaf_idx]


# ---------------------------------------------------------------------------
# Prioritized replay buffer
# ---------------------------------------------------------------------------

class PrioritizedReplayBuffer:
    """Replay buffer with proportional prioritization (PER) and symmetry augmentation.

    Transitions are sampled proportional to ``priority^alpha``. Importance-
    sampling weights ``(N * P(i))^(-beta)`` are returned with each batch to
    correct the bias introduced by non-uniform sampling. Beta is annealed from
    ``PER_BETA_START`` toward ``PER_BETA_END`` over the course of training.
    """

    def __init__(self, capacity: int, device: torch.device) -> None:
        self.capacity = capacity
        self.device = device
        self._tree = SumTree(capacity)
        self._max_priority = 1.0

    def push(self, transition: Transition) -> None:
        # New transitions get max priority to guarantee they are sampled at least once.
        priority = self._max_priority ** settings.PER_ALPHA
        self._tree.add(priority, transition)

    def sample(self, batch_size: int, beta: float) -> ReplayBatch:
        indices: list[int] = []
        priorities: list[float] = []
        transitions: list[Transition] = []

        total = self._tree.total
        segment = total / batch_size

        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            # Clamp to avoid floating-point overshoot at the upper bound.
            s = min(s, total - 1e-8)
            leaf_idx, priority, data = self._tree.get(s)
            indices.append(leaf_idx)
            priorities.append(priority)
            transitions.append(data)  # type: ignore[arg-type]

        # Importance-sampling weights, normalised so the max weight is 1.
        probs = np.array(priorities, dtype=np.float32) / total
        weights = (self._tree.size * probs) ** (-beta)
        weights /= weights.max()

        # Apply a random board symmetry to each transition in the batch.
        transitions = [_apply_random_symmetry(t) for t in transitions]

        states = np.stack([t.state for t in transitions]).astype(np.float32)
        actions = np.array([int(t.action) for t in transitions], dtype=np.int64)
        rewards = np.array([t.reward for t in transitions], dtype=np.float32)
        next_states = np.stack([t.next_state for t in transitions]).astype(np.float32)
        dones = np.array([t.done for t in transitions], dtype=np.float32)

        return ReplayBatch(
            states=torch.tensor(states, dtype=torch.float32, device=self.device),
            actions=torch.tensor(actions, dtype=torch.long, device=self.device),
            rewards=torch.tensor(rewards, dtype=torch.float32, device=self.device),
            next_states=torch.tensor(next_states, dtype=torch.float32, device=self.device),
            dones=torch.tensor(dones, dtype=torch.float32, device=self.device),
            weights=torch.tensor(weights, dtype=torch.float32, device=self.device),
            indices=indices,
        )

    def update_priorities(self, indices: list[int], td_errors: np.ndarray) -> None:
        for leaf_idx, td_error in zip(indices, td_errors):
            priority = (abs(float(td_error)) + settings.PER_PRIORITY_EPSILON) ** settings.PER_ALPHA
            self._tree.update(leaf_idx, priority)
            self._max_priority = max(self._max_priority, priority)

    def __len__(self) -> int:
        return self._tree.size


# ---------------------------------------------------------------------------
# N-step return buffer (unchanged)
# ---------------------------------------------------------------------------

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
