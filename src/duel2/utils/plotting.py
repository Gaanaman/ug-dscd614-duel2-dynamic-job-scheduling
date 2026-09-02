"""Plot helpers.

Owner: Caleb

Shared style so every figure in the report looks like it came from one project.
Training curves show the mean across seeds with a shaded spread -- never a
single seed, and never seeds overplotted without an aggregate.
"""

from __future__ import annotations


def training_curve(per_seed_frames, ax=None):
    """Mean episode return vs. environment steps, shaded by spread across seeds."""
    raise NotImplementedError("TODO")


def baseline_bars(aggregated, metric: str, ax=None):
    """Agent vs. baselines on one metric, with error bars from seed spread."""
    raise NotImplementedError("TODO")


def rollout_gantt(rollout_records, ax=None):
    """Machine lanes over simulated time for a single episode.

    This is the figure that shows a rollout of the trained agent, which the
    demonstration requires for an environment with no graphical renderer.
    """
    raise NotImplementedError("TODO")
