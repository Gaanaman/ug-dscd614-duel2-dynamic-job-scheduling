"""Metrics, verified against a schedule small enough to work out on paper.

Owner: Caleb

A metric bug invalidates every number in the report and is the cheapest possible
thing to get wrong.
"""

import numpy as np
import pytest

from duel2.env import CompletedJob
from duel2.jobs import Job
from duel2.metrics import compute_metrics


def hand_schedule():
    """
    machines 2, jobs 3
      j0: arrival 0, p 4, w 1, deadline 10
      j1: arrival 0, p 2, w 1, deadline 10
      j2: arrival 1, p 3, w 2, deadline  4   <- finishes at 5, so 1 late

    schedule: m0 <- j0 [0, 4);  m1 <- j1 [0, 2);  m1 <- j2 [2, 5)

      makespan          5 - 0 = 5
      waiting times     j0 0, j1 0, j2 2 - 1 = 1   -> mean 1/3
      busy machine-time 4 + 2 + 3 = 9 over 2 * 5 = 10 -> utilisation 0.9
      missed deadlines  1 of 3
      weighted tardiness 2 * (5 - 4) = 2
    """
    j0 = Job(0, 0.0, 4.0, 1.0, 10.0)
    j1 = Job(1, 0.0, 2.0, 1.0, 10.0)
    j2 = Job(2, 1.0, 3.0, 2.0, 4.0)
    completed = [
        CompletedJob(j0, 0, 0.0, 4.0),
        CompletedJob(j1, 1, 0.0, 2.0),
        CompletedJob(j2, 1, 2.0, 5.0),
    ]
    return completed, np.array([4.0, 5.0])


def test_metrics_on_hand_computed_instance():
    completed, busy = hand_schedule()
    m = compute_metrics(completed, busy, episode_return=-1.25)

    assert m.makespan == pytest.approx(5.0)
    assert m.avg_waiting_time == pytest.approx(1 / 3)
    assert m.machine_utilisation == pytest.approx(0.9)
    assert m.missed_deadlines == pytest.approx(1 / 3)
    assert m.weighted_tardiness == pytest.approx(2.0)
    assert m.cumulative_reward == pytest.approx(-1.25)
    assert m.jobs_completed == 3


def test_utilisation_stays_in_the_unit_interval():
    completed, busy = hand_schedule()
    assert 0.0 <= compute_metrics(completed, busy, 0.0).machine_utilisation <= 1.0


def test_empty_episode_does_not_divide_by_zero():
    m = compute_metrics([], np.zeros(2), episode_return=0.0)
    assert m.jobs_completed == 0
    assert m.makespan == 0.0
