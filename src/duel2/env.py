"""Gymnasium environment for dynamic job scheduling on parallel machines.

Owner: Faithful

Formulation reference: docs/mdp_spec.md. Keep the two in sync — the report is
marked against the specification, and a divergence between spec and code is the
kind of thing a viva finds immediately.

Design summary
--------------
M parallel machines, N jobs arriving as a Poisson process. Decision epochs occur
only when at least one machine is idle and at least one job is pending; between
epochs the simulator jumps to the next event. Observations are fixed-dimension
(5K + 3M + 4) via a K-slot queue window sorted by earliest deadline. Actions are
Discrete(K*M + 1) with a validity mask emitted in ``info``.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class EnvConfig:
    """Environment parameters. Mirrors configs/env_default.yaml."""

    n_machines: int = 5
    n_jobs: int = 50
    queue_window: int = 10          # K
    arrival_rate: float = 0.0       # lambda; TODO set from load analysis
    horizon: float = 0.0            # H; TODO time normalisation constant
    max_epochs_factor: int = 4      # T_max = factor * n_jobs
    # TODO: processing-time distribution, deadline tightness, machine speeds


class DynamicJobShopEnv(gym.Env):
    """Assign queued jobs to idle machines to minimise waiting and tardiness."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, config: EnvConfig | None = None, render_mode: str | None = None):
        self.cfg = config or EnvConfig()
        self.render_mode = render_mode

        K, M = self.cfg.queue_window, self.cfg.n_machines
        self.obs_dim = 5 * K + 3 * M + 4
        self.n_actions = K * M + 1          # last index is the no-op

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.n_actions)

    # ------------------------------------------------------------------ API

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Generate a fresh job instance and advance to the first decision epoch.

        The instance seed is separate from the agent seed — see
        docs/experimental_protocol.md. ``options["instance_seed"]`` overrides it,
        which is how the evaluation harness pins the held-out instances.
        """
        super().reset(seed=seed)
        raise NotImplementedError("TODO: generate instance, init machines, advance to first epoch")

    def step(self, action: int):
        """Apply one dispatch decision and advance to the next decision epoch.

        Returns ``(obs, reward, terminated, truncated, info)`` where
        ``info["action_mask"]`` is the mask for the *returned* observation. The
        training loop stores that mask in the replay buffer; the bootstrap target
        needs it.
        """
        raise NotImplementedError(
            "TODO: decode action -> (slot, machine); commit assignment; advance "
            "simulator to next event; accumulate reward over the interval"
        )

    def action_masks(self) -> np.ndarray:
        """Boolean mask over the action space for the current state.

        ``mask[k * M + m]`` is True iff slot k holds a job and machine m is idle.
        The no-op is always valid, otherwise a state with no idle machine would
        have an empty action set.
        """
        raise NotImplementedError("TODO: delegate to action_mask.build_mask")

    def render(self):
        """ANSI text render: one line per machine showing its current job and
        remaining time, plus the head of the pending queue.

        Worth implementing — the demonstration must show a rollout, and a text
        trace of the agent's decisions satisfies that requirement for an
        environment with no graphical rendering.
        """
        raise NotImplementedError

    # ------------------------------------------------------------- internals

    def _advance_to_next_epoch(self) -> float:
        """Advance simulated time to the next decision epoch. Returns delta_t."""
        raise NotImplementedError

    def _is_terminal(self) -> bool:
        """True when all N jobs have completed. Distinct from truncation."""
        raise NotImplementedError
