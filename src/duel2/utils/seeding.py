"""Seeding.

Owner: Daniel K. Adotey (22424924)

Three independent streams per run, so that agent stochasticity and instance
difficulty never become confounded:

    seed          -> network init, exploration
    seed + 1000   -> training job instances  (must stay below 9000)
    9000 + i      -> evaluation instances    (fixed across seeds AND policies)

The third stream is what makes the agent-vs-baseline comparison paired.
"""

from __future__ import annotations


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch. Record the value in the report."""
    raise NotImplementedError("TODO")


def training_instance_seed(seed: int, episode: int) -> int:
    raise NotImplementedError("TODO: derive from seed + 1000, assert < 9000")
