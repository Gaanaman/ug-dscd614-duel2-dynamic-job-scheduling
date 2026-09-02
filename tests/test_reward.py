"""Reward function.

Owner: joint (reward.py is the environment/metrics interface).
"""

import pytest

from duel2.baselines import FCFS, SJF
from duel2.env import DynamicJobShopEnv
from duel2.metrics import compute_metrics
from duel2.reward import RewardTerms, RewardWeights, completion_terms, interval_terms


def test_interval_terms_are_time_weighted_counts():
    t = interval_terms(delta_t=2.5, n_pending=4, n_idle=1)
    assert t.waiting == pytest.approx(10.0)
    assert t.idle == pytest.approx(2.5)


def test_no_tardiness_penalty_when_the_deadline_is_met():
    t = completion_terms(weight=3.0, completion_time=4.0, deadline=9.0)
    assert t.completion == pytest.approx(3.0)
    assert t.tardiness == pytest.approx(0.0)


def test_tardiness_is_weighted_by_priority_and_lateness():
    t = completion_terms(weight=3.0, completion_time=11.0, deadline=9.0)
    assert t.tardiness == pytest.approx(6.0)


def test_waiting_term_telescopes_to_total_waiting_time():
    """The identity claimed in docs/mdp_spec.md section 5 and in the report.

    Summed over an episode, sum_i dt_i * |Q_i| is the area under the queue-length
    curve, which equals the total time all jobs spend waiting. If this fails, the
    claim that the first reward term is the objective rather than a shaping
    heuristic is false and must come out of the report.
    """
    env = DynamicJobShopEnv()
    obs, info = env.reset(options={"instance_seed": 9001})
    policy = SJF()
    policy.reset()

    accumulated = RewardTerms()
    while True:
        obs, _, terminated, truncated, info = env.step(
            policy.act(obs, info["action_mask"], info)
        )
        accumulated += info["reward_terms"]
        if terminated or truncated:
            break

    total_waiting = sum(c.start - c.job.arrival for c in env.completed)
    assert accumulated.waiting == pytest.approx(total_waiting, rel=1e-6)


def test_total_applies_the_weights_and_normaliser():
    terms = RewardTerms(waiting=10.0, idle=4.0, completion=3.0, tardiness=1.0)
    w = RewardWeights(alpha=1.0, beta=0.5, gamma_c=2.0, delta=3.0)
    # -(1.0*10 + 0.5*4) + 2.0*3 - 3.0*1 = -12 + 6 - 3 = -9, over Z = 2
    assert terms.total(w, normaliser=2.0) == pytest.approx(-4.5)
