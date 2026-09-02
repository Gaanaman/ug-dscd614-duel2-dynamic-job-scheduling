"""Gymnasium API conformance.

Owner: Faithful

The instructions require the Gymnasium API and forbid the deprecated OpenAI Gym
package. check_env catches the whole class of five-tuple, dtype and bounds
mistakes that otherwise show up as an inexplicable training failure on day nine.
"""

import pytest

pytest.importorskip("gymnasium")


@pytest.mark.xfail(reason="environment not implemented yet", strict=False)
def test_passes_gymnasium_checker():
    from gymnasium.utils.env_checker import check_env

    from duel2.envs.scheduling_env import DynamicJobShopEnv

    check_env(DynamicJobShopEnv(), skip_render_check=True)


@pytest.mark.xfail(reason="environment not implemented yet", strict=False)
def test_observation_dimension_matches_spec():
    from duel2.envs.scheduling_env import DynamicJobShopEnv

    env = DynamicJobShopEnv()
    K, M = env.cfg.queue_window, env.cfg.n_machines
    assert env.observation_space.shape == (5 * K + 3 * M + 4,)
    assert env.action_space.n == K * M + 1


@pytest.mark.xfail(reason="environment not implemented yet", strict=False)
def test_same_instance_seed_gives_identical_episode():
    """Reproducibility: identical instance seed, identical job stream."""
    from duel2.envs.scheduling_env import DynamicJobShopEnv

    a = DynamicJobShopEnv().reset(options={"instance_seed": 4242})[0]
    b = DynamicJobShopEnv().reset(options={"instance_seed": 4242})[0]
    assert (a == b).all()


@pytest.mark.xfail(reason="environment not implemented yet", strict=False)
def test_termination_and_truncation_are_distinct():
    """terminated on all jobs complete; truncated on the epoch limit. Never both
    conflated -- the bootstrap target treats them differently."""
    raise AssertionError("write this against a tiny instance")
