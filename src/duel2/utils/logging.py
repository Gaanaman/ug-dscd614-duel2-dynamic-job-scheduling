"""Structured run logging.

Owner: Daniel K. Adotey (22424924)

Everything plotted in the report must come out of these files, so log more than
feels necessary now -- re-running training on 3 September to recover a column is
not a plan.

Training (logs/train/seed_{s}/progress.csv), one row per episode:
    global_step, episode, episode_return, episode_length, loss, epsilon,
    mean_q, r_waiting, r_idle, r_completion, r_tardiness

The four reward components are what let the Discussion section argue about
reward design from evidence, and what diagnoses an agent that maximises reward
while losing on the metric.

Evaluation (logs/eval/{policy}_seed{s}.jsonl), one object per episode:
    instance_seed, makespan, avg_waiting_time, machine_utilisation,
    missed_deadlines, weighted_tardiness, cumulative_reward
"""

from __future__ import annotations


class RunLogger:
    def __init__(self, path, fields):
        raise NotImplementedError("TODO")

    def log(self, **kwargs) -> None:
        raise NotImplementedError("TODO")
