"""Action validity masking.

Owner: Faithful

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


def build_mask(
    n_visible_jobs: int,
    idle_machines: list[bool],
    queue_window: int,
    future_event_exists: bool,
) -> np.ndarray:
    """Boolean mask of shape ``(queue_window * n_machines + 1,)``.

    ``mask[k * n_machines + m]`` is True iff slot k holds a job and machine m is
    idle.

    The final entry is the no-op. It is valid only when a future event exists --
    a running job that will complete, or a job that has not yet arrived. Without
    that condition an agent could no-op forever in a state where nothing else
    can happen, and simulated time would never advance.
    """
    n_machines = len(idle_machines)
    mask = np.zeros(queue_window * n_machines + 1, dtype=bool)

    occupied = np.zeros(queue_window, dtype=bool)
    occupied[:n_visible_jobs] = True
    mask[: queue_window * n_machines] = np.outer(
        occupied, np.asarray(idle_machines, dtype=bool)
    ).ravel()
    mask[-1] = future_event_exists

    return mask


def decode_action(action: int, n_machines: int, queue_window: int) -> tuple[int, int] | None:
    """Map a flat action index to ``(slot, machine)``, or None for the no-op."""
    if action == queue_window * n_machines:
        return None
    if not 0 <= action < queue_window * n_machines:
        raise ValueError(f"action {action} out of range for {queue_window}x{n_machines}")
    return divmod(action, n_machines)


def apply_mask(q_values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Set invalid entries to -inf so argmax and max ignore them."""
    out = np.array(q_values, dtype=np.float64, copy=True)
    out[~mask] = -np.inf
    return out
