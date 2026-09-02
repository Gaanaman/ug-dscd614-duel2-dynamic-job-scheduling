"""Gymnasium environment for dynamic job scheduling on parallel machines.

Owner: Faithful

Formulation reference: docs/mdp_spec.md. Keep the two in sync -- the report is
marked against the specification, and a divergence between spec and code is the
kind of thing a viva finds immediately.

M parallel machines with heterogeneous speeds, N jobs arriving as a Poisson
process. A decision epoch occurs only when at least one machine is idle and at
least one job is pending; between epochs the simulator jumps to the next event,
so episode length is proportional to the number of jobs rather than to the
length of the simulated clock. Observations are fixed-dimension (5K + 3M + 4)
via a K-slot queue window sorted by earliest deadline. Actions are
Discrete(K*M + 1) with a validity mask returned in ``info["action_mask"]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces

from .action_mask import build_mask, decode_action
from .jobs import Job, generate_instance
from .observation import build_observation, observation_dim
from .reward import RewardTerms, RewardWeights, completion_terms, interval_terms

_EPS = 1e-9


@dataclass
class EnvConfig:
    """Environment parameters. Mirrors configs/env_default.yaml."""

    n_machines: int = 5
    machine_speeds: tuple[float, ...] = (1.0, 1.0, 1.25, 0.8, 0.8)
    n_jobs: int = 50
    queue_window: int = 10
    arrival_rate: float = 1.0
    processing_time_min: int = 2
    processing_time_max: int = 10
    weight_values: tuple[float, ...] = (1.0, 2.0, 5.0)
    weight_probs: tuple[float, ...] = (0.6, 0.3, 0.1)
    deadline_tightness_min: float = 1.3
    deadline_tightness_max: float = 2.5
    horizon: float = 120.0
    max_epochs_factor: int = 4

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EnvConfig":
        data = yaml.safe_load(Path(path).read_text())
        for key in ("machine_speeds", "weight_values", "weight_probs"):
            if key in data and data[key] is not None:
                data[key] = tuple(data[key])
        return cls(**data)

    def __post_init__(self):
        if len(self.machine_speeds) != self.n_machines:
            raise ValueError(
                f"machine_speeds has {len(self.machine_speeds)} entries but "
                f"n_machines is {self.n_machines}"
            )

    @property
    def p_max(self) -> float:
        return float(self.processing_time_max)

    @property
    def w_max(self) -> float:
        return float(max(self.weight_values))

    @property
    def s_max(self) -> float:
        return float(max(self.machine_speeds))

    @property
    def mean_processing_time(self) -> float:
        return (self.processing_time_min + self.processing_time_max) / 2.0

    @property
    def capacity(self) -> float:
        """Jobs per unit time the machine bank can clear on average."""
        return sum(self.machine_speeds) / self.mean_processing_time

    @property
    def reward_normaliser(self) -> float:
        """Z in the reward equation. Keeps episode returns O(1) across sizes."""
        return self.n_jobs * self.mean_processing_time

    @property
    def max_epochs(self) -> int:
        return self.max_epochs_factor * self.n_jobs


@dataclass
class CompletedJob:
    """One finished job, as the metrics module needs it."""

    job: Job
    machine: int
    start: float
    finish: float


class DynamicJobShopEnv(gym.Env):
    """Assign queued jobs to idle machines to minimise waiting and tardiness."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        config: EnvConfig | None = None,
        weights: RewardWeights | None = None,
        render_mode: str | None = None,
        strict_actions: bool = True,
    ):
        self.cfg = config or EnvConfig()
        self.weights = weights or RewardWeights()
        self.render_mode = render_mode

        # strict_actions=True raises on a masked action, so a mask bug in the
        # agent surfaces immediately instead of training a policy shaped by
        # moves the environment silently rewrote. Gymnasium's check_env samples
        # uniformly from the full action space and cannot respect a mask, so
        # tests/test_env_api.py runs the conformance check with strict_actions
        # off, where an invalid action falls back to the no-op and is counted in
        # info["invalid_actions"]. Training and evaluation always run strict.
        self.strict_actions = strict_actions
        self.invalid_actions = 0

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(observation_dim(self.cfg),), dtype=np.float32
        )
        self.action_space = spaces.Discrete(
            self.cfg.queue_window * self.cfg.n_machines + 1
        )

        self.now: float = 0.0
        self.pending: list[Job] = []
        self.completed: list[CompletedJob] = []

    # ------------------------------------------------------------------ API

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Generate a fresh job instance and advance to the first decision epoch.

        ``options["instance_seed"]`` pins the job stream independently of the
        agent seed. That separation is what lets the evaluation harness run every
        policy on the same held-out instances -- see
        docs/experimental_protocol.md.
        """
        super().reset(seed=seed)

        options = options or {}
        instance_seed = options.get("instance_seed")
        if instance_seed is None:
            instance_seed = int(self.np_random.integers(0, 2**31 - 1))

        self._instance = generate_instance(instance_seed, self.cfg)
        self._next_arrival = 0
        self.now = 0.0
        self.pending = []
        self.completed = []
        self._epoch = 0
        self.invalid_actions = 0
        self._running: list[CompletedJob | None] = [None] * self.cfg.n_machines
        self.machine_free_at = np.zeros(self.cfg.n_machines)
        self.machine_busy_time = np.zeros(self.cfg.n_machines)

        self._process_events_now()
        self._advance_to_decision_epoch()

        return build_observation(self), self._info(instance_seed)

    def step(self, action: int):
        """Apply one dispatch decision and advance to the next decision epoch."""
        action = int(action)
        mask = self.action_masks()
        if not mask[action]:
            if self.strict_actions:
                raise ValueError(
                    f"action {action} is masked out in this state; the policy "
                    "must select from info['action_mask']"
                )
            self.invalid_actions += 1
            action = int(np.flatnonzero(mask)[-1])   # no-op if valid, else first valid

        terms = RewardTerms()
        decoded = decode_action(action, self.cfg.n_machines, self.cfg.queue_window)

        if decoded is None:
            terms += self._advance_one_event()
        else:
            slot, machine = decoded
            self._dispatch(self.visible_jobs()[slot], machine)

        terms += self._advance_to_decision_epoch()
        self._epoch += 1

        reward = terms.total(self.weights, self.cfg.reward_normaliser)
        terminated = len(self.completed) == self.cfg.n_jobs
        truncated = (not terminated) and (
            self._epoch >= self.cfg.max_epochs or self.now > self.cfg.horizon
        )

        info = self._info()
        info["reward_terms"] = terms
        return build_observation(self), reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Boolean mask over the action space for the current state."""
        return build_mask(
            n_visible_jobs=len(self.visible_jobs()),
            idle_machines=[r is None for r in self._running],
            queue_window=self.cfg.queue_window,
            future_event_exists=self._next_event_time() is not None,
        )

    def visible_jobs(self) -> list[Job]:
        """The K most urgent pending jobs, by (deadline, processing time)."""
        ordered = sorted(self.pending, key=lambda j: (j.deadline, j.processing_time))
        return ordered[: self.cfg.queue_window]

    def render(self):
        """One line per machine, plus the head of the pending queue.

        The demonstration must show a rollout of the trained agent. This
        environment has no graphical renderer, so a text trace of its decisions
        is what satisfies that requirement.
        """
        lines = [f"t={self.now:7.2f}  epoch={self._epoch:4d}  "
                 f"pending={len(self.pending):3d}  done={len(self.completed):3d}"]
        for m, run in enumerate(self._running):
            if run is None:
                lines.append(f"  m{m} (x{self.cfg.machine_speeds[m]:.2f})  idle")
            else:
                lines.append(
                    f"  m{m} (x{self.cfg.machine_speeds[m]:.2f})  job {run.job.job_id:3d}"
                    f"  finishes {run.finish:7.2f}  deadline {run.job.deadline:7.2f}"
                )
        for job in self.visible_jobs()[:5]:
            lines.append(
                f"  queued job {job.job_id:3d}  p={job.processing_time:5.2f}"
                f"  w={job.weight:3.1f}  slack={job.deadline - self.now:7.2f}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------- internals

    def _dispatch(self, job: Job, machine: int) -> None:
        duration = job.processing_time / self.cfg.machine_speeds[machine]
        finish = self.now + duration
        self._running[machine] = CompletedJob(job, machine, self.now, finish)
        self.machine_free_at[machine] = finish
        self.machine_busy_time[machine] += duration
        self.pending.remove(job)

    def _next_event_time(self) -> float | None:
        """Earliest future completion or arrival, or None if neither exists."""
        candidates = [r.finish for r in self._running if r is not None]
        if self._next_arrival < self.cfg.n_jobs:
            candidates.append(self._instance[self._next_arrival].arrival)
        future = [c for c in candidates if c > self.now + _EPS]
        return min(future) if future else None

    def _process_events_now(self) -> RewardTerms:
        """Complete jobs and release arrivals that are due at the current time."""
        terms = RewardTerms()
        for m, run in enumerate(self._running):
            if run is not None and run.finish <= self.now + _EPS:
                self.completed.append(run)
                self._running[m] = None
                terms += completion_terms(run.job.weight, run.finish, run.job.deadline)
        while (
            self._next_arrival < self.cfg.n_jobs
            and self._instance[self._next_arrival].arrival <= self.now + _EPS
        ):
            self.pending.append(self._instance[self._next_arrival])
            self._next_arrival += 1
        return terms

    def _advance_one_event(self) -> RewardTerms:
        """Move the clock to the next event. Used by the no-op action."""
        t_next = self._next_event_time()
        if t_next is None:
            return RewardTerms()
        terms = interval_terms(t_next - self.now, len(self.pending), self._n_idle())
        self.now = t_next
        terms += self._process_events_now()
        return terms

    def _advance_to_decision_epoch(self) -> RewardTerms:
        """Run the simulator until the agent has a decision to make."""
        terms = RewardTerms()
        while not self._is_decision_epoch():
            t_next = self._next_event_time()
            if t_next is None:
                break
            terms += interval_terms(t_next - self.now, len(self.pending), self._n_idle())
            self.now = t_next
            terms += self._process_events_now()
        return terms

    def _is_decision_epoch(self) -> bool:
        return self._n_idle() > 0 and len(self.pending) > 0

    def _n_idle(self) -> int:
        return sum(1 for r in self._running if r is None)

    def _info(self, instance_seed: int | None = None) -> dict:
        info = {
            "action_mask": self.action_masks(),
            "now": self.now,
            "pending": list(self.pending),
            "visible": self.visible_jobs(),
            "idle_machines": [r is None for r in self._running],
            "machine_speeds": self.cfg.machine_speeds,
            "invalid_actions": self.invalid_actions,
        }
        if instance_seed is not None:
            info["instance_seed"] = instance_seed
        return info
