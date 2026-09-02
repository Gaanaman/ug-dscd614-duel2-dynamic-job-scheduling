"""Episode metrics.

Owner: Caleb

Computed from a completed episode. The agent and every baseline use this module
-- there is exactly one implementation of each metric in the repository.

    makespan            max_j C_j - min_j a_j                     lower better
    avg_waiting_time    mean_j (start_j - a_j)                    lower better
    machine_utilisation busy machine-time / (M * makespan)        higher better
    missed_deadlines    |{ j : C_j > d_j }| / N                   lower better
    weighted_tardiness  sum_j w_j * max(0, C_j - d_j)             lower better
    cumulative_reward   episode return                            higher better

tests/test_metrics.py verifies these against a schedule small enough to work out
on paper. A metric bug invalidates every number in the report and is the cheapest
possible thing to get wrong.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class EpisodeMetrics:
    makespan: float
    avg_waiting_time: float
    machine_utilisation: float
    missed_deadlines: float
    weighted_tardiness: float
    cumulative_reward: float
    jobs_completed: int

    def as_dict(self) -> dict:
        return asdict(self)


def compute_metrics(completed, machine_busy_time, episode_return: float) -> EpisodeMetrics:
    """Metrics for one episode.

    Args:
        completed: CompletedJob records, one per finished job.
        machine_busy_time: per-machine accumulated processing time.
        episode_return: sum of rewards over the episode.
    """
    if not completed:
        return EpisodeMetrics(0.0, 0.0, 0.0, 0.0, 0.0, episode_return, 0)

    first_arrival = min(c.job.arrival for c in completed)
    last_finish = max(c.finish for c in completed)
    makespan = last_finish - first_arrival

    n_machines = len(machine_busy_time)
    utilisation = (
        float(sum(machine_busy_time)) / (n_machines * makespan) if makespan > 0 else 0.0
    )

    return EpisodeMetrics(
        makespan=makespan,
        avg_waiting_time=sum(c.start - c.job.arrival for c in completed) / len(completed),
        machine_utilisation=utilisation,
        missed_deadlines=sum(1 for c in completed if c.finish > c.job.deadline) / len(completed),
        weighted_tardiness=sum(
            c.job.weight * max(0.0, c.finish - c.job.deadline) for c in completed
        ),
        cumulative_reward=episode_return,
        jobs_completed=len(completed),
    )
