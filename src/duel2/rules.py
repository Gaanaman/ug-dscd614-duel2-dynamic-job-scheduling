"""Dispatching rules as an action space.

Owner: Daniel

Motivation from the literature. Recent reviews of DRL for dynamic job-shop
scheduling report that using dispatching rules as the action space is
consistently better than selecting an eligible operation directly: the action
carries domain structure, the choice stays consistent over a decision, and the
learned policy inherits a guaranteed floor -- an agent that always picks one
rule reproduces that rule exactly. See docs/report/references.md.

Our first formulation used the direct (job slot, machine) action space, which is
the "eligible operation" design. It reached third of five on every metric and
lost to Shortest-Job-First. The diagnosis in the report was that a flat
concatenated observation makes cross-slot comparison unnatural for an MLP. A
rule action space removes that requirement entirely: the comparison is performed
by the rule, and the network only has to learn WHEN each rule is the right one.

Each rule scores the visible jobs and the highest score wins. The machine is
always the fastest idle machine, identically for every rule, so the choice
isolates job selection.
"""

from __future__ import annotations

import math

import numpy as np


def _spt(job, now, cfg):      return -job.processing_time
def _lpt(job, now, cfg):      return job.processing_time
def _edd(job, now, cfg):      return -job.deadline
def _fcfs(job, now, cfg):     return -job.arrival
def _wspt(job, now, cfg):     return job.weight / max(job.processing_time, 1e-9)
def _min_slack(job, now, cfg):
    return -(job.deadline - now - job.processing_time)
def _critical_ratio(job, now, cfg):
    return -(job.deadline - now) / max(job.processing_time, 1e-9)


def _atc(job, now, cfg):
    """Apparent Tardiness Cost: weighted shortest processing time, discounted by slack.

    The standard composite rule for weighted tardiness. k = 2.0 is the usual
    look-ahead; p_bar is the mean processing time.
    """
    slack = max(0.0, job.deadline - now - job.processing_time)
    p_bar = cfg.mean_processing_time
    return (job.weight / max(job.processing_time, 1e-9)) * math.exp(-slack / (2.0 * p_bar))


RULES = [
    ("SPT", _spt, "shortest processing time"),
    ("LPT", _lpt, "longest processing time"),
    ("EDD", _edd, "earliest due date"),
    ("FCFS", _fcfs, "earliest arrival"),
    ("WSPT", _wspt, "weighted shortest processing time"),
    ("MS", _min_slack, "minimum slack"),
    ("CR", _critical_ratio, "critical ratio"),
    ("ATC", _atc, "apparent tardiness cost"),
]
N_RULES = len(RULES)
RULE_NAMES = [r[0] for r in RULES]


def apply_rule(rule_index: int, visible_jobs, idle_machines, machine_speeds, now, cfg):
    """Return ``(slot, machine)`` chosen by rule ``rule_index``.

    ``visible_jobs`` is the K-slot window in its sorted order, so the returned
    slot index maps directly onto the direct action encoding.
    """
    if not visible_jobs or not any(idle_machines):
        return None
    _, score, _ = RULES[rule_index]
    slot = max(range(len(visible_jobs)), key=lambda s: score(visible_jobs[s], now, cfg))
    idle = [m for m, free in enumerate(idle_machines) if free]
    machine = max(idle, key=lambda m: machine_speeds[m])
    return slot, machine


def rule_mask(visible_jobs, idle_machines) -> np.ndarray:
    """Every rule is applicable whenever a dispatch is possible.

    A decision epoch guarantees both an idle machine and a pending job, so this
    is all-True in practice. It is kept because the training loop, the replay
    buffer and the dueling aggregation all take a mask, and because it keeps the
    two action modes interchangeable.
    """
    ok = bool(visible_jobs) and any(idle_machines)
    return np.full(N_RULES, ok, dtype=bool)
