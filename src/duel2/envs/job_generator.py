"""Synthetic job instance generation.

Owner: Kyeremeh Faithful (22424515)

Instances are generated from a dedicated RNG stream so that the *instances* an
agent sees are independent of the *agent's* stochasticity. See
docs/experimental_protocol.md: training draws from ``seed + 1000``, evaluation
uses the fixed held-out range 9000-9029, and the harness asserts the two never
overlap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EVAL_SEED_START = 9000
EVAL_SEED_END = 9030          # exclusive
TRAIN_SEED_CEILING = 9000     # training instance seeds must stay below this


@dataclass(frozen=True)
class Job:
    job_id: int
    arrival: float
    processing_time: float
    weight: float
    deadline: float


def generate_instance(instance_seed: int, cfg) -> list[Job]:
    """Generate one episode's job stream.

    Arrivals are a Poisson process with rate ``cfg.arrival_rate``. Deadlines are
    set from a tightness factor over processing time so that a non-trivial
    fraction of jobs is at risk under a naive policy — if every deadline is
    comfortably met, the tardiness term in the reward carries no signal and the
    baselines are already near-optimal.
    """
    raise NotImplementedError("TODO")


def assert_held_out(instance_seed: int) -> None:
    """Fail loudly if a training instance seed collides with the eval range."""
    if EVAL_SEED_START <= instance_seed < EVAL_SEED_END:
        raise ValueError(
            f"instance seed {instance_seed} is in the held-out evaluation range "
            f"[{EVAL_SEED_START}, {EVAL_SEED_END}) and must not be used for training"
        )
