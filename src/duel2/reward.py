"""Reward function.

Written jointly -- this module is the interface between the environment and the
metrics, and both owners have to agree on it.

    r_i = -(alpha * dt * |Q_i| + beta * dt * |I_i|) / Z
          + gamma_c * sum_{j in F_i} w_j / Z
          - delta * sum_{j in F_i} w_j * max(0, C_j - d_j) / Z

    Q_i  pending jobs at epoch i
    I_i  idle machines at epoch i
    F_i  jobs completing during [t_i, t_{i+1})
    dt   t_{i+1} - t_i
    Z    N * mean(p), so returns are O(1) across instance sizes

The first term is worth understanding rather than treating as shaping: summed
over an episode, ``sum_i dt_i * |Q_i|`` telescopes to the total waiting time
accumulated across all jobs. It is the true objective decomposed over decision
epochs, which is why the agent gets a dense signal that stays consistent with
the metric it is scored on.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RewardWeights:
    alpha: float = 1.0      # queue waiting cost
    beta: float = 0.3       # machine idleness
    gamma_c: float = 1.0    # weighted completion
    delta: float = 2.0      # weighted tardiness


def compute_reward(delta_t, pending, idle_machines, completed, weights, normaliser):
    """Reward for the interval between two decision epochs.

    Return the total and the four component terms separately -- logging the
    components is what makes the reward-design discussion in the report
    evidence-based rather than speculative, and it is how you diagnose an agent
    that optimises the reward while losing on the metric.
    """
    raise NotImplementedError("TODO")
