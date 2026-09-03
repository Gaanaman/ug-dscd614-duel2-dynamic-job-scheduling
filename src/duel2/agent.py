"""Masked dueling DQN: replay buffer, training loop, greedy policy wrapper.

Owner: Daniel

Attribution
-----------
The training loop follows the standard DQN recipe of Mnih et al. (2015) --
replay buffer, target network, epsilon-greedy -- and the dueling decomposition
of Wang et al. (2016), structured after the single-file CleanRL reference
implementation (Huang et al., 2022, MIT licence). It is written here rather than
imported so that action masking can be threaded through every place it is
needed. See docs/attribution.md.

Modifications over a standard DQN, all of which the report must describe:

  1. Dueling value/advantage heads with the mean taken over valid actions only
     (network.py).
  2. The mask applied to epsilon-greedy exploration, so random actions are drawn
     from the legal set rather than the full 51.
  3. The mask applied to the bootstrap target, which requires storing the
     next-state mask in the replay buffer. Omitting this is the silent failure
     described in docs/mdp_spec.md section 4.
  4. Optional double-Q targets (van Hasselt et al., 2016), off by default. The
     brief binds the algorithm to Dueling DQN, so the headline run is dueling
     alone; the flag exists so the ablation in the Discussion is a config change
     rather than a code change.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .action_mask import apply_mask
from .env import DynamicJobShopEnv
from .metrics import compute_metrics
from .network import NEG_INF, DuelingQNetwork
from .replay import PrioritisedReplayBuffer
from .runtime import training_instance_seed


@dataclass
class AgentConfig:
    hidden: tuple[int, ...] = (256, 256)
    learning_rate: float = 1e-4
    batch_size: int = 128
    buffer_size: int = 200_000
    learning_starts: int = 5_000
    train_frequency: int = 4
    target_update_interval: int = 1_000
    gamma: float = 0.99
    grad_clip: float = 10.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_fraction: float = 0.3
    total_timesteps: int = 300_000
    double_q: bool = False
    prioritised_replay: bool = False
    """Sample transitions in proportion to their last TD error.

    Schaul et al. (2016). Present in the recipe of the closest published
    precedent (Han and Yang, 2020, dueling double DQN with prioritised replay)
    and in Liu et al. (2025). Changes which transitions are sampled, not the
    learning rule, so the algorithm remains Dueling DQN.
    """
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    n_step: int = 1
    """Length of the multi-step return used in the bootstrap target.

    n = 1 is ordinary DQN. Larger n propagates a delayed consequence back to the
    action that caused it in one update instead of n, which matters here: a
    dispatch returns reward 0 at the instant it is committed and its cost only
    appears when the queue drains, tens of decisions later. Declared in
    docs/hyperparameters.md as a deviation.
    """


class ReplayBuffer:
    """Fixed-size circular buffer.

    Stores *two* masks, and both are load-bearing:

      ``next_mask``  the bootstrap target must restrict its maximum to actions
                     legal in the next state.
      ``mask``       the dueling aggregation subtracts the mean advantage over
                     valid actions, so Q(s,a) depends on the current mask too.
                     Training with an all-ones mask here optimises a different
                     function from the one select_action evaluates, and the
                     agent gets steadily worse while nothing errors.

    That second one is why this class exists instead of a library buffer.
    """

    def __init__(self, capacity: int, obs_dim: int, n_actions: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.mask = np.zeros((capacity, n_actions), dtype=bool)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_mask = np.zeros((capacity, n_actions), dtype=bool)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)
        self.discount = np.ones(capacity, dtype=np.float32)
        self.pos = 0
        self.full = False

    def add(self, obs, mask, action, reward, next_obs, next_mask, terminated, discount=None):
        i = self.pos
        self.obs[i] = obs
        self.mask[i] = mask
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.next_mask[i] = next_mask
        # terminated only -- a truncated episode is cut off by the harness, not
        # by the task, so its final value is bootstrapped rather than zeroed.
        self.terminated[i] = float(terminated)
        self.discount[i] = 1.0 if discount is None else float(discount)
        self.pos = (self.pos + 1) % self.capacity
        self.full = self.full or self.pos == 0

    def __len__(self) -> int:
        return self.capacity if self.full else self.pos

    def sample(self, batch_size: int, device):
        idx = np.random.randint(0, len(self), size=batch_size)
        t = lambda x, dt=torch.float32: torch.as_tensor(x[idx], dtype=dt, device=device)
        return (t(self.obs), t(self.mask, torch.bool), t(self.actions, torch.int64),
                t(self.rewards), t(self.next_obs), t(self.next_mask, torch.bool),
                t(self.terminated), t(self.discount))


class MaskedDuelingDQN:
    """Dueling DQN with action masking over a Discrete action space."""

    def __init__(self, env: DynamicJobShopEnv, cfg: AgentConfig, seed: int, logger=None,
                 device: str = "cpu"):
        self.env, self.cfg, self.seed, self.logger = env, cfg, seed, logger
        self.device = torch.device(device)

        obs_dim = env.observation_space.shape[0]
        n_actions = int(env.action_space.n)
        self.q = DuelingQNetwork(obs_dim, n_actions, cfg.hidden).to(self.device)
        self.target = DuelingQNetwork(obs_dim, n_actions, cfg.hidden).to(self.device)
        self.target.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=cfg.learning_rate)
        self.loss_fn = nn.SmoothL1Loss(reduction="none")
        self.buffer = (
            PrioritisedReplayBuffer(cfg.buffer_size, obs_dim, n_actions,
                                    alpha=cfg.per_alpha, beta_start=cfg.per_beta_start)
            if cfg.prioritised_replay
            else ReplayBuffer(cfg.buffer_size, obs_dim, n_actions)
        )
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------ policy

    def epsilon(self, step: int) -> float:
        span = max(1.0, self.cfg.epsilon_decay_fraction * self.cfg.total_timesteps)
        frac = min(1.0, step / span)
        return self.cfg.epsilon_start + frac * (self.cfg.epsilon_end - self.cfg.epsilon_start)

    @torch.no_grad()
    def q_values(self, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        m = torch.as_tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        return self.q(o, m).squeeze(0).cpu().numpy()

    def select_action(self, obs: np.ndarray, mask: np.ndarray, epsilon: float) -> int:
        """Epsilon-greedy over valid actions only.

        The random branch samples from ``mask.nonzero()``, not from the full
        action space. Sampling from all 51 would spend most of the exploration
        budget on actions the environment rejects outright.
        """
        if self.rng.random() < epsilon:
            return int(self.rng.choice(np.flatnonzero(mask)))
        return int(np.argmax(apply_mask(self.q_values(obs, mask), mask)))

    # ------------------------------------------------------------ learning

    def compute_loss(self, batch):
        """Returns (scalar loss, per-sample TD errors).

        The loss is per-sample so importance-sampling weights can correct the
        bias that prioritised sampling introduces. With uniform replay every
        weight is 1 and this reduces exactly to the mean Huber loss.
        """
        weights = None
        if len(batch) == 9:
            obs, mask, actions, rewards, next_obs, next_mask, terminated, discount, weights = batch
        else:
            obs, mask, actions, rewards, next_obs, next_mask, terminated, discount = batch

        with torch.no_grad():
            if self.cfg.double_q:
                # online net picks the action, target net scores it
                greedy = self.q(next_obs, next_mask).argmax(dim=1, keepdim=True)
                next_q = self.target(next_obs, next_mask).gather(1, greedy).squeeze(1)
            else:
                next_q = self.target(next_obs, next_mask).max(dim=1).values
            # a state with no legal next action contributes no bootstrap value
            next_q = torch.where(next_mask.any(dim=1), next_q, torch.zeros_like(next_q))
            next_q = next_q.clamp(min=NEG_INF / 2)
            # discount is gamma**k for the k actually accumulated, so a partial
            # n-step return flushed at an episode boundary stays correct.
            target = rewards + discount * (1.0 - terminated) * next_q

        # the CURRENT mask, not ones: the dueling mean is taken over valid
        # actions, so Q(s,a) is only the quantity select_action uses when the
        # same mask is supplied here.
        predicted = self.q(obs, mask).gather(1, actions.unsqueeze(1)).squeeze(1)
        elementwise = self.loss_fn(predicted, target)
        if weights is not None:
            elementwise = elementwise * weights
        td = (target - predicted).detach().cpu().numpy()
        return elementwise.mean(), td

    def train(self, total_timesteps: int | None = None) -> dict:
        """Run training. Returns a summary dict; per-episode rows go to the logger."""
        total = total_timesteps or self.cfg.total_timesteps
        env = self.env
        started = time.time()

        episode, ep_return, ep_len, losses, qs = 0, 0.0, 0, [], []
        nstep: deque = deque(maxlen=self.cfg.n_step)

        def flush(final: bool):
            """Emit n-step transitions from the pending window."""
            while nstep:
                o0, m0, a0 = nstep[0][0], nstep[0][1], nstep[0][2]
                ret, disc = 0.0, 1.0
                term_k, no, nm = 0.0, nstep[-1][4], nstep[-1][5]
                for (_, _, _, r_k, o_k, m_k, t_k) in nstep:
                    ret += disc * r_k
                    disc *= self.cfg.gamma
                    no, nm, term_k = o_k, m_k, t_k
                    if t_k:
                        break
                self.buffer.add(o0, m0, a0, ret, no, nm, term_k, discount=disc)
                if not final and len(nstep) == self.cfg.n_step:
                    nstep.popleft(); return
                nstep.popleft()
        obs, info = env.reset(
            seed=self.seed,
            options={"instance_seed": training_instance_seed(self.seed, 0)},
        )

        for step in range(1, total + 1):
            eps = self.epsilon(step)
            mask = info["action_mask"]
            action = self.select_action(obs, mask, eps)
            next_obs, reward, terminated, truncated, next_info = env.step(action)

            nstep.append((obs, mask, action, reward, next_obs,
                          next_info["action_mask"], float(terminated)))
            if len(nstep) == self.cfg.n_step:
                flush(final=False)
            obs, info = next_obs, next_info
            ep_return += reward
            ep_len += 1

            if terminated or truncated:
                flush(final=True)
                m = compute_metrics(env.completed, env.machine_busy_time, ep_return)
                if self.logger:
                    self.logger.log(
                        global_step=step, episode=episode, episode_return=round(ep_return, 5),
                        episode_length=ep_len, epsilon=round(eps, 4),
                        loss=round(float(np.mean(losses)), 6) if losses else "",
                        mean_q=round(float(np.mean(qs)), 4) if qs else "",
                        makespan=round(m.makespan, 3),
                        avg_waiting_time=round(m.avg_waiting_time, 4),
                        missed_deadlines=round(m.missed_deadlines, 4),
                        weighted_tardiness=round(m.weighted_tardiness, 2),
                    )
                episode += 1
                ep_return, ep_len, losses, qs = 0.0, 0, [], []
                obs, info = env.reset(options={
                    "instance_seed": training_instance_seed(self.seed, episode)})

            if step > self.cfg.learning_starts and step % self.cfg.train_frequency == 0:
                if self.cfg.prioritised_replay:
                    # beta annealed to 1 so the correction is exact by the end,
                    # the schedule Schaul et al. (2016) recommend
                    frac = step / max(1, total)
                    beta = self.cfg.per_beta_start + frac * (1.0 - self.cfg.per_beta_start)
                    batch = self.buffer.sample(self.cfg.batch_size, self.device, beta=beta)
                else:
                    batch = self.buffer.sample(self.cfg.batch_size, self.device)
                loss, td = self.compute_loss(batch)
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.grad_clip)
                self.opt.step()
                if self.cfg.prioritised_replay:
                    self.buffer.update_priorities(td)
                losses.append(loss.item())

            if step % self.cfg.target_update_interval == 0:
                self.target.load_state_dict(self.q.state_dict())

        return {"episodes": episode, "steps": total, "seconds": round(time.time() - started, 1)}

    # ------------------------------------------------------------ persistence

    def save(self, path) -> None:
        torch.save({"state_dict": self.q.state_dict(), "cfg": self.cfg.__dict__,
                    "seed": self.seed}, path)


class GreedyAgentPolicy:
    """The trained agent, wearing the same Policy interface as the baselines.

    This is what makes the comparison legitimate: the agent is handed to
    ``harness.run_policy`` exactly as FCFS, SJF and Round Robin are, so it runs
    on the same instances through the same metric code. Exploration is off --
    action selection is a deterministic masked argmax.
    """

    def __init__(self, network: DuelingQNetwork, name: str = "DuelingDQN", device="cpu"):
        self.net, self.name, self.device = network, name, torch.device(device)
        self.net.eval()

    @classmethod
    def load(cls, path, obs_dim: int, n_actions: int, name: str = "DuelingDQN", device="cpu"):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        hidden = tuple(ckpt["cfg"].get("hidden", (256, 256)))
        net = DuelingQNetwork(obs_dim, n_actions, hidden).to(device)
        net.load_state_dict(ckpt["state_dict"])
        return cls(net, name=name, device=device)

    def reset(self) -> None:
        pass

    @torch.no_grad()
    def act(self, obs: np.ndarray, mask: np.ndarray, info: dict) -> int:
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        m = torch.as_tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        return int(self.net(o, m).squeeze(0).argmax().item())
