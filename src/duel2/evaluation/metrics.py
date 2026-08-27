"""Episode metrics.

Owner: Caleb Abakah Mensah (22424188)

Computed from a completed episode's job records. The agent and every baseline
use this module -- there is exactly one implementation of each metric in the
repository.

    makespan            max_j C_j - min_j a_j                     lower better
    avg_waiting_time    mean_j (start_j - a_j)                    lower better
    machine_utilisation busy machine-time / (M * makespan)        higher better
    missed_deadlines    |{ j : C_j > d_j }| / N                   lower better
    weighted_tardiness  sum_j w_j * max(0, C_j - d_j)             lower better
    cumulative_reward   episode return                            higher better

Verify these against a hand-computed 3-job, 2-machine instance in
tests/test_metrics.py. A metric bug invalidates every number in the report, and
it is the cheapest possible thing to get wrong.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EpisodeMetrics:
    makespan: float
    avg_waiting_time: float
    machine_utilisation: float
    missed_deadlines: float
    weighted_tardiness: float
    cumulative_reward: float


def compute_metrics(job_records, machine_records, episode_return: float) -> EpisodeMetrics:
    raise NotImplementedError("TODO")
