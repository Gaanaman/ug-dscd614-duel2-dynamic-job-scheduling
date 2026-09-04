"""Run the baselines and report whether the instance distribution has headroom.

Owner: Caleb

Run this after changing anything in configs/env_default.yaml, and before
training. Two failure modes it catches:

  * makespan close to n_jobs / arrival_rate -- the arrival process is setting
    the finish time, so the scheduler is irrelevant and there is nothing for the
    agent to learn.
  * baselines within noise of each other, or a miss rate near 0 or near 1 --
    the tardiness term carries no signal.

    python scripts/check_load.py --config configs/env_default.yaml
"""

from __future__ import annotations

import argparse
import statistics as st

from duel2.baselines import FCFS, SJF, RoundRobin
from dataclasses import replace

from duel2.env import DynamicJobShopEnv, EnvConfig
from duel2.jobs import EVAL_SEED_START
from duel2.metrics import compute_metrics


def run(env, policy, n_episodes):
    rows = []
    for i in range(n_episodes):
        obs, info = env.reset(options={"instance_seed": EVAL_SEED_START + i})
        policy.reset()
        total = 0.0
        while True:
            obs, reward, terminated, truncated, info = env.step(
                policy.act(obs, info["action_mask"], info)
            )
            total += reward
            if terminated or truncated:
                break
        rows.append(compute_metrics(env.completed, env.machine_busy_time, total))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/env_default.yaml")
    ap.add_argument("--episodes", type=int, default=30)
    args = ap.parse_args()

    cfg = EnvConfig.from_yaml(args.config)
    # The dispatch rules emit (slot, machine) action indices, so this check runs
    # in a direct-assignment environment whatever action_mode the config carries.
    env = DynamicJobShopEnv(replace(cfg, action_mode="direct"))
    arrival_bound = cfg.n_jobs / cfg.arrival_rate

    print(f"capacity {cfg.capacity:.3f} jobs/time-unit   rho {cfg.arrival_rate / cfg.capacity:.2f}")
    print(f"arrival-bound makespan (n_jobs / lambda) = {arrival_bound:.1f}\n")
    print(f"{'policy':<11} {'makespan':>9} {'wait':>7} {'util':>6} {'missed':>7} {'tard':>8} {'return':>8}")

    results = {}
    for policy in (FCFS(), SJF(), RoundRobin(cfg.n_machines)):
        rows = run(env, policy, args.episodes)
        results[policy.name] = {
            field: st.mean(getattr(r, field) for r in rows)
            for field in (
                "makespan", "avg_waiting_time", "machine_utilisation",
                "missed_deadlines", "weighted_tardiness", "cumulative_reward",
            )
        }
        r = results[policy.name]
        print(
            f"{policy.name:<11} {r['makespan']:9.2f} {r['avg_waiting_time']:7.2f} "
            f"{r['machine_utilisation']:6.3f} {r['missed_deadlines']:7.3f} "
            f"{r['weighted_tardiness']:8.2f} {r['cumulative_reward']:8.3f}"
        )

    print()
    warnings = []
    if results["FCFS"]["makespan"] < arrival_bound * 1.15:
        warnings.append(
            "makespan is close to the arrival bound -- the arrival process sets the "
            "finish time and the scheduler barely matters"
        )
    spread = abs(results["FCFS"]["avg_waiting_time"] - results["SJF"]["avg_waiting_time"])
    if spread < 0.5:
        warnings.append(f"FCFS and SJF differ by only {spread:.2f} in waiting time -- no headroom")
    miss = results["FCFS"]["missed_deadlines"]
    if miss < 0.05 or miss > 0.9:
        warnings.append(f"FCFS miss rate {miss:.3f} is saturated -- retune deadline tightness")

    for w in warnings:
        print(f"WARNING: {w}")
    if not warnings:
        print(f"OK: makespan {results['FCFS']['makespan']:.1f} against arrival bound "
              f"{arrival_bound:.1f}; FCFS-SJF waiting-time gap {spread:.2f}; "
              f"FCFS miss rate {miss:.3f}")


if __name__ == "__main__":
    main()
