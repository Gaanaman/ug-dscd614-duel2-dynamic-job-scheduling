"""Metrics.

Owner: Caleb Abakah Mensah (22424188)

A metric bug invalidates every number in the report. Verify against a schedule
small enough to work out on paper: 3 jobs, 2 machines, known arrival and
processing times, makespan and utilisation computed by hand in the docstring.
"""

import pytest


@pytest.mark.xfail(reason="not implemented yet", strict=False)
def test_metrics_on_hand_computed_instance():
    """
    machines 2, jobs 3
      j0: arrival 0, p 4
      j1: arrival 0, p 2
      j2: arrival 1, p 3
    schedule: m0 <- j0 [0,4); m1 <- j1 [0,2); m1 <- j2 [2,5)
      makespan          5 - 0 = 5
      waiting times     j0 0, j1 0, j2 1  -> mean 1/3
      busy machine-time 4 + 2 + 3 = 9 over 2 * 5 = 10 -> utilisation 0.9
    """
    raise AssertionError("assert compute_metrics reproduces the numbers above")


@pytest.mark.xfail(reason="not implemented yet", strict=False)
def test_utilisation_bounded_in_unit_interval():
    raise AssertionError("write me")


@pytest.mark.xfail(reason="not implemented yet", strict=False)
def test_aggregate_reports_mean_and_spread():
    """Never a single number, and never a selected seed."""
    raise AssertionError("write me")
