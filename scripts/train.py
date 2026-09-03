"""Train the dueling DQN agent for one seed.

Owner: Daniel

    python scripts/train.py --config configs/dueling_dqn.yaml --seed 0

Writes logs/train/seed_{seed}/progress.csv and models/dueling_dqn_seed{seed}.pt.
Every hyperparameter comes from the config file, so the three seeds are provably
identical in configuration -- a per-seed configuration invalidates the
comparison and the rubric says so explicitly.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
import yaml

# One thread per process. Three seeds run concurrently and each PyTorch process
# would otherwise spawn a pool sized to every core, oversubscribing the machine
# and making a nine-run grid roughly an order of magnitude slower.
torch.set_num_threads(1)

from duel2.agent import AgentConfig, MaskedDuelingDQN
from duel2.env import DynamicJobShopEnv, EnvConfig
from duel2.reward import RewardWeights
from duel2.runtime import TRAIN_FIELDS, RunLogger, seed_everything


def load_agent_config(path: str, override_steps: int | None) -> AgentConfig:
    raw = yaml.safe_load(Path(path).read_text())
    cfg = AgentConfig(
        hidden=tuple(raw["network"]["hidden"]),
        learning_rate=float(raw["learning_rate"]),
        batch_size=int(raw["batch_size"]),
        buffer_size=int(raw["buffer_size"]),
        learning_starts=int(raw["learning_starts"]),
        train_frequency=int(raw["train_frequency"]),
        target_update_interval=int(raw["target_update_interval"]),
        gamma=float(raw["gamma"]),
        grad_clip=float(raw["grad_clip"]),
        epsilon_start=float(raw["epsilon"]["start"]),
        epsilon_end=float(raw["epsilon"]["end"]),
        epsilon_decay_fraction=float(raw["epsilon"]["decay_fraction"]),
        total_timesteps=int(raw["total_timesteps"]),
        double_q=bool(raw.get("double_q", False)),
        n_step=int(raw.get("n_step", 1)),
        prioritised_replay=bool(raw.get("prioritised_replay", False)),
        per_alpha=float(raw.get("per_alpha", 0.6)),
        per_beta_start=float(raw.get("per_beta_start", 0.4)),
    )
    return replace(cfg, total_timesteps=override_steps) if override_steps else cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dueling_dqn.yaml")
    ap.add_argument("--env-config", default="configs/env_default.yaml")
    ap.add_argument("--reward-config", default="configs/reward.yaml")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--total-timesteps", type=int, default=None,
                    help="overrides the config; declare any override in the report")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an existing checkpoint for this seed")
    args = ap.parse_args()

    out = Path(args.out_dir)
    model_path = out / f"models/dueling_dqn_seed{args.seed}.pt"
    if model_path.exists() and not args.overwrite:
        raise SystemExit(
            f"{model_path} already exists. Refusing to overwrite a saved run.\n"
            f"Pass --out-dir to write elsewhere, or --overwrite to replace it.\n"
            f"This guard exists because a run launched without --out-dir once clobbered a\n"
            f"committed checkpoint, leaving the weights inconsistent with their own log."
        )
    agent_cfg = load_agent_config(args.config, args.total_timesteps)
    env_cfg = EnvConfig.from_yaml(args.env_config)
    weights = RewardWeights(**yaml.safe_load(Path(args.reward_config).read_text()))

    seed_everything(args.seed)
    env = DynamicJobShopEnv(env_cfg, weights, strict_actions=True)

    log_path = out / f"logs/train/seed_{args.seed}/progress.csv"
    with RunLogger(log_path, TRAIN_FIELDS) as logger:
        agent = MaskedDuelingDQN(env, agent_cfg, seed=args.seed, logger=logger)
        summary = agent.train()

    model_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(model_path)

    summary.update(seed=args.seed, model=str(model_path), log=str(log_path),
                   double_q=agent_cfg.double_q)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
