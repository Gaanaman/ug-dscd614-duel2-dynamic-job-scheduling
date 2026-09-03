"""Replay buffers: uniform and prioritised.

Owner: Daniel

Prioritised experience replay (Schaul et al., 2016, arXiv:1511.05952) samples a
transition in proportion to its last temporal-difference error, so the updates
that still surprise the network are revisited more often. It appears in the
recipe of the closest published precedent to this project -- Han and Yang (2020)
use a dueling double DQN with prioritised replay -- and in Liu et al. (2025).
See docs/report/literature_review.md.

It is a change to *which transitions are sampled*, not to the learning rule, so
the algorithm remains Dueling DQN as the brief requires.

Two corrections are needed for correctness and both are implemented:

  * Sampling non-uniformly biases the expected update. Importance-sampling
    weights w_i = (N * P(i))^-beta, normalised by their maximum, correct it.
    beta is annealed to 1 over training so the correction is exact by the end,
    which is the schedule Schaul et al. recommend.
  * A new transition has no TD error yet. It enters at the current maximum
    priority so it is sampled at least once before being ranked.

The SumTree gives O(log N) sampling and update against O(N) for a naive
probability vector, which matters at a capacity of 200,000.
"""

from __future__ import annotations

import numpy as np
import torch


class SumTree:
    """Fixed-capacity binary tree whose internal nodes hold subtree sums."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)

    def total(self) -> float:
        return float(self.tree[0])

    def update(self, data_idx: int, priority: float) -> None:
        i = data_idx + self.capacity - 1
        delta = priority - self.tree[i]
        self.tree[i] = priority
        while i > 0:
            i = (i - 1) // 2
            self.tree[i] += delta

    def find(self, value: float) -> int:
        """Index of the leaf whose cumulative range contains ``value``."""
        i = 0
        while i < self.capacity - 1:
            left = 2 * i + 1
            if value <= self.tree[left]:
                i = left
            else:
                value -= self.tree[left]
                i = left + 1
        return i - (self.capacity - 1)

    def max_leaf(self) -> float:
        return float(self.tree[self.capacity - 1:].max())


class PrioritisedReplayBuffer:
    """Proportional prioritised replay with importance-sampling correction."""

    def __init__(self, capacity: int, obs_dim: int, n_actions: int,
                 alpha: float = 0.6, beta_start: float = 0.4, eps: float = 1e-3):
        self.capacity = capacity
        self.alpha, self.beta_start, self.eps = alpha, beta_start, eps
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.mask = np.zeros((capacity, n_actions), dtype=bool)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_mask = np.zeros((capacity, n_actions), dtype=bool)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)
        self.discount = np.ones(capacity, dtype=np.float32)
        self.tree = SumTree(capacity)
        self.pos, self.full = 0, False
        self._last_idx: np.ndarray | None = None

    def __len__(self) -> int:
        return self.capacity if self.full else self.pos

    def add(self, obs, mask, action, reward, next_obs, next_mask, terminated, discount=None):
        i = self.pos
        self.obs[i] = obs
        self.mask[i] = mask
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.next_mask[i] = next_mask
        self.terminated[i] = float(terminated)
        self.discount[i] = 1.0 if discount is None else float(discount)
        # enter at maximum priority so every transition is seen at least once
        p_max = self.tree.max_leaf() if len(self) else 1.0
        self.tree.update(i, max(p_max, self.eps) ** 1.0)
        self.pos = (self.pos + 1) % self.capacity
        self.full = self.full or self.pos == 0

    def sample(self, batch_size: int, device, beta: float | None = None):
        n = len(self)
        total = self.tree.total()
        segment = total / batch_size
        idx = np.empty(batch_size, dtype=np.int64)
        for k in range(batch_size):
            v = np.random.uniform(segment * k, segment * (k + 1))
            j = self.tree.find(v)
            idx[k] = min(j, n - 1)
        self._last_idx = idx

        priorities = np.array([self.tree.tree[i + self.capacity - 1] for i in idx])
        probs = np.maximum(priorities, 1e-12) / max(total, 1e-12)
        b = self.beta_start if beta is None else beta
        weights = (n * probs) ** (-b)
        weights /= max(weights.max(), 1e-12)

        t = lambda x, dt=torch.float32: torch.as_tensor(x[idx], dtype=dt, device=device)
        return (t(self.obs), t(self.mask, torch.bool), t(self.actions, torch.int64),
                t(self.rewards), t(self.next_obs), t(self.next_mask, torch.bool),
                t(self.terminated), t(self.discount),
                torch.as_tensor(weights, dtype=torch.float32, device=device))

    def update_priorities(self, td_errors: np.ndarray) -> None:
        """Re-rank the transitions returned by the most recent ``sample``."""
        if self._last_idx is None:
            return
        p = (np.abs(td_errors) + self.eps) ** self.alpha
        for i, pi in zip(self._last_idx, p):
            self.tree.update(int(i), float(pi))
