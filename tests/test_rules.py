"""Dispatching-rule action space.

Owner: Daniel

The equivalence tests are what license comparing the two action modes: a fixed
rule in the rules environment must produce exactly the schedule the equivalent
baseline produces in the direct environment. If they diverge, the comparison in
the report is not like-for-like.
"""

import statistics as st
from dataclasses import replace

import pytest

from duel2.baselines import FCFS, SJF, FixedRule
from duel2.env import DynamicJobShopEnv, EnvConfig
from duel2.harness import run_policy
from duel2.rules import N_RULES, RULE_NAMES, apply_rule


def direct_env():
    return DynamicJobShopEnv(EnvConfig(allow_noop=False))


def rules_env():
    return DynamicJobShopEnv(replace(EnvConfig(allow_noop=False), action_mode="rules"))


def test_action_space_size_matches_the_rule_count():
    assert rules_env().action_space.n == N_RULES


def test_every_rule_is_always_applicable_at_a_decision_epoch():
    env = rules_env()
    obs, info = env.reset(options={"instance_seed": 9000})
    for _ in range(60):
        assert info["action_mask"].all(), "a rule was unavailable at a decision epoch"
        obs, _, terminated, truncated, info = env.step(0)
        if terminated or truncated:
            break


@pytest.mark.parametrize("rule,baseline", [("SPT", SJF()), ("FCFS", FCFS())])
def test_fixed_rule_reproduces_its_baseline_exactly(rule, baseline):
    idx = RULE_NAMES.index(rule)
    a, _ = run_policy(baseline, direct_env(), n_episodes=10)
    b, _ = run_policy(FixedRule(idx), rules_env(), n_episodes=10)
    for field in ("avg_waiting_time", "weighted_tardiness", "makespan", "missed_deadlines"):
        assert st.mean(r[field] for r in a) == pytest.approx(
            st.mean(r[field] for r in b), abs=1e-9
        ), f"{rule} diverges from {baseline.name} on {field}"


def test_spt_picks_the_shortest_visible_job():
    env = rules_env()
    env.reset(options={"instance_seed": 9003})
    visible = env.visible_jobs()
    idle = [r is None for r in env._running]
    slot, _ = apply_rule(RULE_NAMES.index("SPT"), visible, idle,
                         env.cfg.machine_speeds, env.now, env.cfg)
    assert visible[slot].processing_time == min(j.processing_time for j in visible)


def test_edd_picks_the_earliest_deadline():
    env = rules_env()
    env.reset(options={"instance_seed": 9004})
    visible = env.visible_jobs()
    idle = [r is None for r in env._running]
    slot, _ = apply_rule(RULE_NAMES.index("EDD"), visible, idle,
                         env.cfg.machine_speeds, env.now, env.cfg)
    assert visible[slot].deadline == min(j.deadline for j in visible)


def test_rules_always_choose_the_fastest_idle_machine():
    """Every rule shares one machine tie-break, so the choice isolates job selection."""
    env = rules_env()
    env.reset(options={"instance_seed": 9005})
    visible = env.visible_jobs()
    idle = [r is None for r in env._running]
    speeds = env.cfg.machine_speeds
    fastest = max((m for m in range(len(idle)) if idle[m]), key=lambda m: speeds[m])
    for i in range(N_RULES):
        _, machine = apply_rule(i, visible, idle, speeds, env.now, env.cfg)
        assert machine == fastest


def test_n_step_returns_use_the_matching_discount():
    """gamma**k must match the k rewards actually accumulated.

    A partial window flushed at an episode boundary carries fewer than n rewards,
    so a fixed gamma**n would over-discount the bootstrap and bias every target
    near the end of an episode.
    """
    from dataclasses import replace as dc_replace

    import numpy as np

    from duel2.agent import AgentConfig, MaskedDuelingDQN

    for n in (1, 3, 5):
        agent = MaskedDuelingDQN(
            rules_env(),
            dc_replace(AgentConfig(), n_step=n, total_timesteps=2000, learning_starts=10_000),
            seed=0,
        )
        agent.train(2000)
        used = {round(float(d), 6) for d in agent.buffer.discount[: len(agent.buffer)]}
        allowed = {round(0.99 ** k, 6) for k in range(1, n + 1)}
        assert used <= allowed, f"n={n} produced discounts outside gamma^1..gamma^{n}: {used - allowed}"
        assert round(0.99 ** n, 6) in used or n == 1


def test_uniform_replay_weights_reduce_to_the_plain_huber_loss():
    """With no importance weights the loss must equal the unweighted mean.

    Guards the branch: if the weighted path silently applied to uniform replay,
    every headline result would shift without any config change.
    """
    import torch
    from dataclasses import replace as dc_replace

    from duel2.agent import AgentConfig, MaskedDuelingDQN

    agent = MaskedDuelingDQN(rules_env(), AgentConfig(), seed=0)
    n_actions = int(rules_env().action_space.n)
    obs = torch.randn(6, 69)
    m = torch.ones(6, n_actions, dtype=torch.bool)
    batch8 = (obs, m, torch.zeros(6, dtype=torch.int64), torch.zeros(6),
              obs, m, torch.zeros(6), torch.full((6,), 0.99))
    batch9 = batch8 + (torch.ones(6),)
    a, _ = agent.compute_loss(batch8)
    b, _ = agent.compute_loss(batch9)
    assert torch.isclose(a, b), "unit importance weights changed the loss"


def test_prioritised_replay_trains_and_reranks():
    from dataclasses import replace as dc_replace

    from duel2.agent import AgentConfig, MaskedDuelingDQN
    from duel2.replay import PrioritisedReplayBuffer

    cfg = dc_replace(AgentConfig(), prioritised_replay=True, total_timesteps=4000,
                     learning_starts=500, buffer_size=4096)
    agent = MaskedDuelingDQN(rules_env(), cfg, seed=0)
    assert isinstance(agent.buffer, PrioritisedReplayBuffer)
    before = agent.buffer.tree.total()
    agent.train(4000)
    assert agent.buffer.tree.total() != before, "priorities never updated"
    assert len(agent.buffer) > 0


def test_required_baselines_run_in_a_direct_environment_whatever_the_config():
    """FCFS, SJF and Round Robin must always be evaluable.

    They emit (slot, machine) action indices, so they are only meaningful in a
    direct-assignment environment. When the headline configuration switched to
    the rule action space, deriving the baseline environment from the default
    config turned it into a Discrete(8) space and every required baseline
    crashed with an index error -- which would have broken run_all.sh, the entry
    point the rubric requires to reproduce the headline result.
    """
    from duel2.baselines import FCFS, RoundRobin, SJF
    from duel2.env import EnvConfig
    from duel2.harness import run_policy

    # whatever the default config says, the baselines must still run
    default = EnvConfig.from_yaml("configs/env_default.yaml")
    env = DynamicJobShopEnv(dc_replace_mode(default, "direct"))
    assert env.action_space.n == default.queue_window * default.n_machines + 1
    for pol in (FCFS(), SJF(), RoundRobin(default.n_machines)):
        rows, _ = run_policy(pol, env, n_episodes=2)
        assert len(rows) == 2
        assert rows[0]["jobs_completed"] == default.n_jobs


def dc_replace_mode(cfg, mode):
    from dataclasses import replace as r
    return r(cfg, action_mode=mode)
