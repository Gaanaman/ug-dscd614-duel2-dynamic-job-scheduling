"""Masked DQN training loop.

Owner: Daniel

Attribution: if this is adapted from a reference implementation (CleanRL's
dqn.py is the recommended base -- single file, transparent, easy to modify), add
the attribution comment here AND a row in docs/attribution.md. Rule 8: an
unattributed adaptation is plagiarism regardless of how much was modified.

Modifications over a standard DQN that must be described in the report:
  1. dueling value/advantage heads (agents/dueling_network.py)
  2. action mask applied to the behaviour policy
  3. action mask applied to the bootstrap target -- requires storing the
     next-state mask in the replay buffer
  4. optional double-Q target; state clearly whether it is on, since dueling and
     double are independent choices and conflating them confuses the reader
"""

from __future__ import annotations


class MaskedDuelingDQN:
    """Dueling DQN with action masking over a Discrete action space."""

    def __init__(self, env, config, seed: int, logger):
        raise NotImplementedError("TODO")

    def select_action(self, obs, mask, epsilon: float) -> int:
        """Epsilon-greedy over valid actions only.

        With probability epsilon, sample uniformly from ``mask.nonzero()`` --
        not from the full action space, which would spend most exploration on
        actions the environment rejects.
        """
        raise NotImplementedError("TODO")

    def compute_target(self, batch):
        """Bootstrap target with the next-state mask applied.

            y = r + gamma * (1 - terminated) * max_{a' valid in s'} Q_target(s', a')

        Note ``terminated``, not ``terminated or truncated``. A truncated episode
        is cut off by the harness, not by the task, so its final value is
        bootstrapped rather than zeroed.
        """
        raise NotImplementedError("TODO")

    def train(self, total_timesteps: int):
        """Run training, logging to logs/train/seed_{seed}/progress.csv.

        Log at minimum: environment step, episode return, episode length, loss,
        epsilon, mean Q, and the four reward components. The reward components
        are what turn the Discussion section into an evidence-based argument.
        """
        raise NotImplementedError("TODO")
