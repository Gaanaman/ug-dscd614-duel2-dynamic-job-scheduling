"""Dueling Q-network.

Owner: Daniel

    Q(s, a) = V(s) + ( A(s, a) - mean_{a' in valid} A(s, a') )

The mean is taken over *valid* actions only. Averaging over all K*M+1 entries
lets arbitrary values from masked, unreachable actions leak into V(s). This is
asserted in tests/test_mask.py.

Why dueling suits scheduling, for the Background section: the value of a
scheduling state is dominated by system congestion -- how much work is backed up
against how much capacity is free -- while the advantage of one assignment over
another is often small and sometimes exactly zero. Vanilla DQN re-learns that
shared state value once per action across all 51 outputs. The decomposition
learns congestion once and lets the advantage stream model only the differences.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DuelingQNetwork(nn.Module):
    """Shared trunk, separate value and advantage heads."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.value_head = nn.Linear(hidden, 1)
        self.advantage_head = nn.Linear(hidden, n_actions)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Q-values for a batch of observations.

        Args:
            obs:  (B, obs_dim)
            mask: (B, n_actions) boolean, True where the action is valid

        Returns:
            (B, n_actions) with invalid entries set to -inf.
        """
        raise NotImplementedError(
            "TODO: trunk -> V and A; subtract the mean of A over valid actions "
            "only; combine; then set invalid entries to -inf"
        )
