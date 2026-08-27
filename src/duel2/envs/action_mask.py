"""Action validity masking.

Owner: Kyeremeh Faithful (22424515)

The mask must be applied in three places. All three are required:

1. exploration      -- epsilon-greedy samples uniformly over valid actions only
2. greedy selection -- argmax over Q with invalid entries set to -inf
3. bootstrap target -- max_a' Q_target(s', a') restricted to the *next* state's
                       mask

Point 3 is the one that gets missed. Without it the target backs up the value of
an action the agent can never take, and training silently converges to a policy
shaped by unreachable actions. The next-state mask therefore has to be stored in
the replay buffer alongside the next observation.

The dueling aggregation has the same trap: the mean subtracted from the advantage
stream must be taken over valid actions only, or arbitrary values from masked
entries leak into V(s).
"""

from __future__ import annotations

import numpy as np


def build_mask(pending_slots, machines, n_machines: int, queue_window: int) -> np.ndarray:
    """Return a boolean mask of shape (queue_window * n_machines + 1,).

    ``mask[k * n_machines + m]`` is True iff slot k is occupied and machine m is
    idle. The final entry (the no-op) is always True.
    """
    raise NotImplementedError("TODO")


def decode_action(action: int, n_machines: int) -> tuple[int, int] | None:
    """Map a flat action index to ``(slot, machine)``, or None for the no-op."""
    raise NotImplementedError("TODO")


def apply_mask(q_values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Set invalid entries to -inf so argmax and max ignore them."""
    raise NotImplementedError("TODO")
