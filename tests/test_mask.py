"""Action mask correctness.

Owner: Faithful

The highest-value tests in the repository. A wrong mask does not crash; it
silently trains a policy shaped by actions that cannot be taken, and the failure
is invisible until the results make no sense.
"""

import numpy as np
import pytest

from duel2.action_mask import apply_mask, build_mask, decode_action
from duel2.env import DynamicJobShopEnv


def test_mask_is_the_outer_product_of_occupied_slots_and_idle_machines():
    # 3 slots, 2 machines; slots 0 and 1 occupied; machine 0 idle, machine 1 busy
    mask = build_mask(
        n_visible_jobs=2, idle_machines=[True, False], queue_window=3,
        future_event_exists=True,
    )
    assert mask.tolist() == [
        True, False,    # slot 0 -> m0 valid, m1 busy
        True, False,    # slot 1 -> m0 valid, m1 busy
        False, False,   # slot 2 empty
        True,           # no-op
    ]


def test_empty_slot_invalidates_the_whole_row():
    mask = build_mask(1, [True, True, True], queue_window=2, future_event_exists=True)
    assert mask[:3].all()            # slot 0 occupied, all machines idle
    assert not mask[3:6].any()       # slot 1 empty


def test_busy_machine_invalidates_the_whole_column():
    mask = build_mask(3, [True, False], queue_window=3, future_event_exists=True)
    assert mask[1::2][:3].tolist() == [False, False, False]


def test_noop_is_invalid_when_no_future_event_exists():
    """Otherwise the agent could no-op forever and the clock would never move."""
    assert build_mask(1, [True], 1, future_event_exists=True)[-1]
    assert not build_mask(1, [True], 1, future_event_exists=False)[-1]


def test_decode_action_round_trips():
    for slot in range(4):
        for machine in range(3):
            assert decode_action(slot * 3 + machine, 3, 4) == (slot, machine)
    assert decode_action(12, 3, 4) is None          # the no-op
    with pytest.raises(ValueError):
        decode_action(13, 3, 4)


def test_apply_mask_sends_invalid_entries_to_negative_infinity():
    q = np.array([1.0, 5.0, 3.0])
    masked = apply_mask(q, np.array([True, False, True]))
    assert masked[1] == -np.inf
    assert int(np.argmax(masked)) == 2      # 5.0 was the max but is masked out


def test_environment_never_offers_an_empty_action_set():
    env = DynamicJobShopEnv()
    obs, info = env.reset(options={"instance_seed": 11})
    while True:
        assert info["action_mask"].any(), "no valid action available"
        action = int(np.flatnonzero(info["action_mask"])[0])
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break


def test_mask_agrees_with_the_environment_state():
    env = DynamicJobShopEnv()
    obs, info = env.reset(options={"instance_seed": 12})
    for _ in range(40):
        mask = info["action_mask"]
        n_visible, idle = len(info["visible"]), info["idle_machines"]
        M = len(idle)
        for slot in range(env.cfg.queue_window):
            for m in range(M):
                expected = slot < n_visible and idle[m]
                assert bool(mask[slot * M + m]) == expected
        obs, _, terminated, truncated, info = env.step(int(np.flatnonzero(mask)[0]))
        if terminated or truncated:
            break


@pytest.mark.xfail(reason="agent not implemented yet", strict=False)
def test_bootstrap_target_ignores_invalid_next_actions():
    """The one that gets missed.

    Give the target network a large Q-value on an action that is invalid in s'.
    The computed target must not contain it.
    """
    raise AssertionError("write this alongside agent.py")


@pytest.mark.xfail(reason="network not implemented yet", strict=False)
def test_dueling_mean_is_over_valid_actions_only():
    """Changing the advantage of a masked action must not change any Q-value.

    If it does, the mean subtracted in the aggregation is being taken over all
    actions and masked entries are leaking into V(s).
    """
    raise AssertionError("write this alongside network.py")
