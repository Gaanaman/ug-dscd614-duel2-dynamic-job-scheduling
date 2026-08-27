"""Reward function.

Owner: joint (rewards/reward_fn.py is the environment/metrics interface).
"""

import pytest


@pytest.mark.xfail(reason="not implemented yet", strict=False)
def test_components_match_hand_computation():
    raise AssertionError("one interval, 2 pending jobs, 1 idle machine, 1 completion")


@pytest.mark.xfail(reason="not implemented yet", strict=False)
def test_waiting_term_telescopes_to_total_waiting_time():
    """The property claimed in docs/mdp_spec.md section 5 and in the report.

    Summed over a full episode, sum_i dt_i * |Q_i| must equal the total waiting
    time reported by metrics.compute_metrics. If this test does not pass, the
    claim in the report is false and must be removed.
    """
    raise AssertionError("write me")


@pytest.mark.xfail(reason="not implemented yet", strict=False)
def test_no_tardiness_penalty_when_all_deadlines_met():
    raise AssertionError("write me")
