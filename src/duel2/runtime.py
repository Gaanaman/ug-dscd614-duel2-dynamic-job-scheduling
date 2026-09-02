"""Seeding, run logging and plotting.

Owners: Daniel (seeding, logging), Caleb (plotting)

SEEDING
Three independent streams per run, so agent stochasticity and instance
difficulty never become confounded:

    seed          -> network init, exploration
    seed + 1000   -> training job instances  (must stay below 9000)
    9000 + i      -> evaluation instances    (fixed across seeds AND policies)

The third stream is what makes the agent-vs-baseline comparison paired.

LOGGING
Everything plotted in the report comes out of these files, so log more than
feels necessary now. Re-running training on 3 September to recover a column is
not a plan.

    logs/train/seed_{s}/progress.csv   one row per episode
        global_step, episode, episode_return, episode_length, loss, epsilon,
        mean_q, r_waiting, r_idle, r_completion, r_tardiness

    logs/eval/{policy}_seed{s}.jsonl   one object per episode
        instance_seed, makespan, avg_waiting_time, machine_utilisation,
        missed_deadlines, weighted_tardiness, cumulative_reward

The four reward components turn the Discussion section into an evidence-based
argument, and they are how you diagnose an agent that maximises reward while
losing on the metric.

PLOTTING
Shared style, so every figure in the report looks like it came from one project.
Training curves show the mean across seeds with a shaded spread -- never a
single seed, and never seeds overplotted without an aggregate.
"""

from __future__ import annotations

# --------------------------------------------------------------------- seeding


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch. Record the value in the report."""
    raise NotImplementedError("TODO")


def training_instance_seed(seed: int, episode: int) -> int:
    raise NotImplementedError("TODO: derive from seed + 1000, assert < 9000")


# --------------------------------------------------------------------- logging


class RunLogger:
    def __init__(self, path, fields):
        raise NotImplementedError("TODO")

    def log(self, **kwargs) -> None:
        raise NotImplementedError("TODO")


# -------------------------------------------------------------------- plotting


def training_curve(per_seed_frames, ax=None):
    """Mean episode return vs. environment steps, shaded by spread across seeds."""
    raise NotImplementedError("TODO")


def baseline_bars(aggregated, metric: str, ax=None):
    """Agent vs. baselines on one metric, with error bars from seed spread."""
    raise NotImplementedError("TODO")


def rollout_gantt(rollout_records, ax=None):
    """Machine lanes over simulated time for a single episode.

    This is the rollout figure the demonstration requires for an environment
    with no graphical renderer.
    """
    raise NotImplementedError("TODO")
