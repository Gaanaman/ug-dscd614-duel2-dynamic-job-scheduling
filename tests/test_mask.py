"""Action mask correctness.

Owner: Faithful

These are the highest-value tests in the repository. A wrong mask does not
crash; it silently trains a policy shaped by actions that cannot be taken, and
the failure is invisible until the results make no sense.
"""

import pytest


@pytest.mark.xfail(reason="not implemented yet", strict=False)
def test_mask_true_only_for_occupied_slot_and_idle_machine():
    raise AssertionError("hand-construct a 3-slot 2-machine state and assert every entry")


@pytest.mark.xfail(reason="not implemented yet", strict=False)
def test_noop_always_valid():
    """Otherwise a state with no idle machine has an empty action set."""
    raise AssertionError("write me")


@pytest.mark.xfail(reason="not implemented yet", strict=False)
def test_env_never_accepts_an_invalid_action():
    raise AssertionError("step with a masked action; expect a raise, not silent success")


@pytest.mark.xfail(reason="not implemented yet", strict=False)
def test_bootstrap_target_ignores_invalid_next_actions():
    """The one that gets missed.

    Give the target network a large Q-value on an action that is invalid in s'.
    The computed target must not contain it.
    """
    raise AssertionError("write me")


@pytest.mark.xfail(reason="not implemented yet", strict=False)
def test_dueling_mean_is_over_valid_actions_only():
    """Changing the advantage of a masked action must not change any Q-value.

    If it does, the mean subtracted in the aggregation is being taken over all
    actions and masked entries are leaking into V(s).
    """
    raise AssertionError("write me")
