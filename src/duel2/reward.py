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
    Z    n_jobs * mean processing time, so returns are O(1) across instance sizes

The first term is worth understanding rather than treating as shaping: summed
over an episode, ``sum_i dt_i * |Q_i|`` telescopes to the total waiting time
accumulated across all jobs. It is the true objective decomposed over decision
epochs, which is why the agent gets a dense signal that stays consistent with the
metric it is scored on. tests/test_reward.py asserts that identity holds.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RewardWeights:
    alpha: float = 1.0      # queue waiting cost
    beta: float = 0.3       # machine idleness
    gamma_c: float = 1.0    # weighted completion
    delta: float = 2.0      # weighted tardiness


@dataclass
class RewardTerms:
    """The four components, logged separately.

    Logging the components is what makes the reward-design discussion in the
    report evidence-based, and it is how you diagnose an agent that maximises
    reward while losing on the metric.
    """

    waiting: float = 0.0
    idle: float = 0.0
    completion: float = 0.0
    tardiness: float = 0.0

    def total(self, w: RewardWeights, normaliser: float) -> float:
        return (
            -w.alpha * self.waiting
            - w.beta * self.idle
            + w.gamma_c * self.completion
            - w.delta * self.tardiness
        ) / normaliser

    def __iadd__(self, other: "RewardTerms") -> "RewardTerms":
        self.waiting += other.waiting
        self.idle += other.idle
        self.completion += other.completion
        self.tardiness += other.tardiness
        return self


def interval_terms(delta_t: float, n_pending: int, n_idle: int) -> RewardTerms:
    """Waiting and idleness accrued over a sub-interval of constant occupancy."""
    return RewardTerms(waiting=delta_t * n_pending, idle=delta_t * n_idle)


def completion_terms(weight: float, completion_time: float, deadline: float) -> RewardTerms:
    """Completion credit and tardiness penalty for one finished job."""
    return RewardTerms(
        completion=weight,
        tardiness=weight * max(0.0, completion_time - deadline),
    )
