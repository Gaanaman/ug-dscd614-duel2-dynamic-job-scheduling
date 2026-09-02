"""Synthetic job instance generation.

Owner: Faithful

Instances come from a dedicated RNG stream so the *instances* an agent sees are
independent of the *agent's* stochasticity. See docs/experimental_protocol.md:
training draws from ``seed + 1000``, evaluation uses the fixed held-out range
9000-9029, and ``assert_held_out`` makes an accidental overlap fail loudly
rather than silently inflating the result.
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

    Arrivals are a Poisson process with rate ``cfg.arrival_rate``, shifted so the
    first job arrives at t=0. Deadlines are a tightness factor over processing
    time, which keeps a job's deadline proportional to its own size instead of
    penalising long jobs by construction.
    """
    rng = np.random.default_rng(instance_seed)
    n = cfg.n_jobs

    gaps = rng.exponential(1.0 / cfg.arrival_rate, size=n)
    arrivals = np.cumsum(gaps)
    arrivals -= arrivals[0]

    processing = rng.integers(
        cfg.processing_time_min, cfg.processing_time_max + 1, size=n
    ).astype(float)
    weights = rng.choice(cfg.weight_values, size=n, p=cfg.weight_probs)
    tightness = rng.uniform(
        cfg.deadline_tightness_min, cfg.deadline_tightness_max, size=n
    )
    deadlines = arrivals + processing * tightness

    return [
        Job(i, float(a), float(p), float(w), float(d))
        for i, (a, p, w, d) in enumerate(zip(arrivals, processing, weights, deadlines))
    ]


def assert_held_out(instance_seed: int) -> None:
    """Fail loudly if a training instance seed collides with the eval range."""
    if EVAL_SEED_START <= instance_seed < EVAL_SEED_END:
        raise ValueError(
            f"instance seed {instance_seed} is in the held-out evaluation range "
            f"[{EVAL_SEED_START}, {EVAL_SEED_END}) and must not be used for training"
        )
