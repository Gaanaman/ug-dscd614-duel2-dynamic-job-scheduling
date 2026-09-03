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


def test_bootstrap_target_ignores_invalid_next_actions():
    """The one that gets missed.

    Give the network a huge Q-value on one action, then mask that action out.
    The maximum the bootstrap target would take must not contain it. Without the
    mask in the target, training backs up the value of a move the agent can
    never make, nothing crashes, and the results quietly stop meaning anything.
    """
    import torch
    from duel2.network import DuelingQNetwork

    net = DuelingQNetwork(69, 51)
    with torch.no_grad():
        net.advantage_head.bias.zero_()
        net.advantage_head.bias[7] = 1000.0

    obs = torch.zeros(1, 69)
    allowed = torch.ones(1, 51, dtype=torch.bool)
    assert int(net(obs, allowed).argmax()) == 7            # sanity: it is the max when legal

    blocked = allowed.clone()
    blocked[0, 7] = False
    q = net(obs, blocked)
    assert int(q.argmax()) != 7
    assert float(q.max()) < 100.0, "the masked action leaked into the maximum"


def test_masked_target_changes_the_loss():
    """End-to-end: the same batch scores differently once the mask is applied."""
    import torch
    from duel2.agent import AgentConfig, MaskedDuelingDQN
    from duel2.env import DynamicJobShopEnv

    agent = MaskedDuelingDQN(DynamicJobShopEnv(), AgentConfig(), seed=0)
    with torch.no_grad():
        agent.target.advantage_head.bias.zero_()
        agent.target.advantage_head.bias[7] = 500.0

    obs = torch.zeros(4, 69)
    actions = torch.zeros(4, dtype=torch.int64)
    rewards = torch.zeros(4)
    terminated = torch.zeros(4)
    allowed = torch.ones(4, 51, dtype=torch.bool)
    blocked = allowed.clone(); blocked[:, 7] = False

    cur = torch.ones(4, 51, dtype=torch.bool)
    loss_all = agent.compute_loss((obs, cur, actions, rewards, obs, allowed, terminated))
    loss_masked = agent.compute_loss((obs, cur, actions, rewards, obs, blocked, terminated))
    assert not torch.isclose(loss_all, loss_masked), "the next-state mask had no effect"


def test_dueling_mean_is_over_valid_actions_only():
    """Changing the advantage of a masked action must not move any Q-value.

    If it does, the mean subtracted in the aggregation is being taken over all
    51 actions, and arbitrary values from unreachable moves are leaking into
    V(s).
    """
    import torch
    from duel2.network import DuelingQNetwork

    net = DuelingQNetwork(69, 51)
    obs = torch.randn(1, 69)
    mask = torch.ones(1, 51, dtype=torch.bool)
    mask[0, 7] = False

    before = net(obs, mask).clone()
    with torch.no_grad():
        net.advantage_head.bias[7] += 500.0
    after = net(obs, mask)

    valid = mask[0]
    assert torch.allclose(before[0][valid], after[0][valid], atol=1e-5), \
        "a masked action's advantage leaked into the valid Q-values"


def test_training_uses_the_current_state_mask_for_predicted_q():
    """Q(s,a) must be computed with the same mask select_action uses.

    The dueling mean is taken over valid actions, so supplying an all-ones mask
    during the update optimises a different function from the one the agent acts
    with. Nothing errors; the agent simply gets worse.
    """
    import torch
    from duel2.agent import AgentConfig, MaskedDuelingDQN
    from duel2.env import DynamicJobShopEnv

    agent = MaskedDuelingDQN(DynamicJobShopEnv(), AgentConfig(), seed=0)
    obs = torch.randn(4, 69)
    actions = torch.zeros(4, dtype=torch.int64)
    rewards, terminated = torch.zeros(4), torch.zeros(4)
    next_mask = torch.ones(4, 51, dtype=torch.bool)

    allowed = torch.ones(4, 51, dtype=torch.bool)
    narrow = allowed.clone(); narrow[:, 5:40] = False

    a = agent.compute_loss((obs, allowed, actions, rewards, obs, next_mask, terminated))
    b = agent.compute_loss((obs, narrow, actions, rewards, obs, next_mask, terminated))
    assert not torch.isclose(a, b), "the current-state mask had no effect on the loss"
