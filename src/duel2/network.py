"""Dueling Q-network with masked advantage aggregation.

Owner: Daniel

    Q(s, a) = V(s) + ( A(s, a) - mean_{a' in valid(s)} A(s, a') )

Two details that are easy to get wrong and are both asserted in tests/test_mask.py:

1. The mean is taken over *valid* actions only. Averaging over all K*M+1 entries
   lets arbitrary values from masked, unreachable actions leak into V(s).
2. Invalid entries are set to -inf on the way out, so every argmax and max
   downstream ignores them without the caller having to remember. This is the
   output-layer masking described by Wang et al. (2016) for the dueling
   architecture and applied to scheduling with masking by Han and Yang (2020).

Why dueling suits this problem, for the Background section: the value of a
scheduling state is dominated by congestion -- how much work is backed up
against how much capacity is free -- while the advantage of one assignment over
another is often small and sometimes exactly zero, because two idle machines and
two similar jobs make several actions equivalent. Vanilla DQN re-learns that
shared state value once per action across all 51 outputs. The decomposition
learns congestion once and lets the advantage stream model only the differences.
"""

from __future__ import annotations

import torch
import torch.nn as nn

NEG_INF = -1e9   # finite stand-in for -inf: keeps gradients and losses well defined


class DuelingQNetwork(nn.Module):
    """Shared trunk, separate value and advantage heads."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: tuple[int, ...] = (256, 256)):
        super().__init__()
        layers, last = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        self.trunk = nn.Sequential(*layers)
        self.value_head = nn.Linear(last, 1)
        self.advantage_head = nn.Linear(last, n_actions)
        self.n_actions = n_actions

    def forward(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Q-values for a batch.

        Args:
            obs:  (B, obs_dim) float
            mask: (B, n_actions) bool, True where the action is valid

        Returns:
            (B, n_actions) with invalid entries at NEG_INF.
        """
        h = self.trunk(obs)
        value = self.value_head(h)                       # (B, 1)
        advantage = self.advantage_head(h)               # (B, A)

        mask_f = mask.to(advantage.dtype)
        n_valid = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean_valid_adv = (advantage * mask_f).sum(dim=1, keepdim=True) / n_valid

        q = value + (advantage - mean_valid_adv)
        return q.masked_fill(~mask, NEG_INF)

    def state_value(self, obs: torch.Tensor) -> torch.Tensor:
        """V(s) alone. Used for the value/advantage figure in the report."""
        return self.value_head(self.trunk(obs))
