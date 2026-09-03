"""Gymnasium API conformance and environment invariants.

Owner: Faithful

The instructions require the Gymnasium API and forbid the deprecated OpenAI Gym
package. check_env catches the whole class of five-tuple, dtype and bounds
mistakes that otherwise surface as an inexplicable training failure on day nine.
"""

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from duel2.baselines import SJF
from duel2.env import DynamicJobShopEnv


def rollout(env, policy, instance_seed):
    obs, info = env.reset(options={"instance_seed": instance_seed})
    policy.reset()
    total = 0.0
    while True:
        obs, reward, terminated, truncated, info = env.step(
            policy.act(obs, info["action_mask"], info)
        )
        total += reward
        if terminated or truncated:
            return total, terminated, truncated, info


def test_passes_gymnasium_checker():
    # check_env samples uniformly from the full action space and cannot respect
    # a mask, so the conformance run uses the permissive mode. Training and
    # evaluation always run strict.
    check_env(DynamicJobShopEnv(strict_actions=False), skip_render_check=True)


def test_spaces_match_the_specification():
    env = DynamicJobShopEnv()
    K, M = env.cfg.queue_window, env.cfg.n_machines
    assert env.observation_space.shape == (5 * K + 3 * M + 4,)
    assert env.action_space.n == K * M + 1


def test_observations_stay_inside_the_declared_box():
    env = DynamicJobShopEnv()
    obs, info = env.reset(options={"instance_seed": 4242})
    policy = SJF()
    policy.reset()
    while True:
        assert env.observation_space.contains(obs), "observation outside Box(-1, 1)"
        obs, _, terminated, truncated, info = env.step(
            policy.act(obs, info["action_mask"], info)
        )
        if terminated or truncated:
            break


def test_same_instance_seed_gives_an_identical_episode():
    a = DynamicJobShopEnv().reset(options={"instance_seed": 4242})[0]
    b = DynamicJobShopEnv().reset(options={"instance_seed": 4242})[0]
    assert np.array_equal(a, b)


def test_different_instance_seeds_give_different_episodes():
    a = DynamicJobShopEnv().reset(options={"instance_seed": 1})[0]
    b = DynamicJobShopEnv().reset(options={"instance_seed": 2})[0]
    assert not np.array_equal(a, b)


def test_episode_terminates_with_every_job_completed():
    env = DynamicJobShopEnv()
    _, terminated, truncated, _ = rollout(env, SJF(), 9000)
    assert terminated and not truncated
    assert len(env.completed) == env.cfg.n_jobs


def test_simulated_time_never_moves_backwards():
    env = DynamicJobShopEnv()
    obs, info = env.reset(options={"instance_seed": 7})
    policy = SJF()
    policy.reset()
    last = info["now"]
    while True:
        obs, _, terminated, truncated, info = env.step(
            policy.act(obs, info["action_mask"], info)
        )
        assert info["now"] >= last - 1e-9
        last = info["now"]
        if terminated or truncated:
            break


def test_strict_mode_rejects_a_masked_action():
    env = DynamicJobShopEnv(strict_actions=True)
    _, info = env.reset(options={"instance_seed": 3})
    invalid = int(np.flatnonzero(~info["action_mask"])[0])
    with pytest.raises(ValueError, match="masked out"):
        env.step(invalid)


def test_training_instances_never_enter_the_held_out_range():
    """The bug this catches would silently invalidate every number in the report."""
    from duel2.jobs import EVAL_SEED_END, EVAL_SEED_START
    from duel2.runtime import training_instance_seed

    for seed in range(6):
        for episode in range(0, 20000, 7):
            s = training_instance_seed(seed, episode)
            assert not (EVAL_SEED_START <= s < EVAL_SEED_END), (seed, episode, s)
            assert s >= 0


def test_each_training_seed_gets_its_own_instances():
    """Seeds must not train on each other's instances, or the runs are not independent."""
    from duel2.runtime import training_instance_seed

    bands = [{training_instance_seed(s, e) for e in range(3000)} for s in range(3)]
    assert bands[0].isdisjoint(bands[1])
    assert bands[1].isdisjoint(bands[2])
    assert bands[0].isdisjoint(bands[2])
