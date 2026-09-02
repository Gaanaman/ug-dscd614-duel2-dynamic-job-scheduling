"""Train the dueling DQN agent for one seed.

Owner: Daniel

    python scripts/train.py --config configs/dueling_dqn.yaml \
        --env-config configs/env_default.yaml --seed 0

Writes logs/train/seed_{seed}/progress.csv and models/dueling_dqn_seed{seed}.pt.
Hyperparameters come entirely from the config file so that the three seeds are
provably identical in configuration.
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--env-config", default="configs/env_default.yaml")
    p.add_argument("--reward-config", default="configs/reward.yaml")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--total-timesteps", type=int, default=None,
                   help="overrides the config; declare any override in the report")
    return p.parse_args()


def main() -> None:
    raise NotImplementedError("TODO")


if __name__ == "__main__":
    main()
