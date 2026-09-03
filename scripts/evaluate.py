"""Evaluate the agent and every baseline under identical conditions.

Owner: Caleb

    python scripts/evaluate.py --config configs/eval.yaml

Every policy named in the config runs through harness.run_policy on the same
held-out instances with exploration disabled, and per-episode records go to
logs/eval/*.jsonl. Aggregation across seeds happens here; make_figures.py only
reads what this produces, so no figure can exist without a log behind it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from duel2.agent import GreedyAgentPolicy
from duel2.baselines import FCFS, SJF, RandomMasked, RoundRobin
from duel2.env import DynamicJobShopEnv, EnvConfig
from duel2.harness import METRIC_FIELDS, aggregate_across_seeds, exceeds_seed_variation, run_policy
from duel2.reward import RewardWeights
from duel2.runtime import write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--env-config", default="configs/env_default.yaml")
    ap.add_argument("--reward-config", default="configs/reward.yaml")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--out", default="logs/eval")
    ap.add_argument("--include-random", action="store_true",
                    help="add the diagnostic random floor (not a required baseline)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    env_cfg = EnvConfig.from_yaml(args.env_config)
    weights = RewardWeights(**yaml.safe_load(Path(args.reward_config).read_text()))
    env = DynamicJobShopEnv(env_cfg, weights, strict_actions=True)
    n_ep, seeds = int(cfg["n_episodes"]), list(cfg["seeds"])
    out = Path(args.out)

    results: dict[str, dict] = {}

    # --- baselines: deterministic, so one pass is replicated across seed slots.
    # Their only source of variation is the instance set, which is fixed, so the
    # spread across seeds is genuinely zero. The report says so rather than
    # leaving the reader to wonder why the error bars vanished.
    baselines = [FCFS(), SJF(), RoundRobin(env_cfg.n_machines)]
    if args.include_random:
        baselines.append(RandomMasked(0))
    for pol in baselines:
        rows, _ = run_policy(pol, env, n_episodes=n_ep)
        write_jsonl(out / f"{pol.name}.jsonl", rows)
        results[pol.name] = aggregate_across_seeds({s: rows for s in seeds})

    # --- agent: one trained network per seed, same instances
    per_seed, missing = {}, []
    for s in seeds:
        ckpt = Path(args.models_dir) / f"dueling_dqn_seed{s}.pt"
        if not ckpt.exists():
            missing.append(str(ckpt))
            continue
        pol = GreedyAgentPolicy.load(ckpt, env.observation_space.shape[0],
                                     int(env.action_space.n))
        rows, rollout = run_policy(pol, env, n_episodes=n_ep, record_rollout=(s == seeds[0]))
        write_jsonl(out / f"DuelingDQN_seed{s}.jsonl", rows)
        if s == seeds[0]:
            write_jsonl(out / "rollout_seed0.jsonl", rollout)
        per_seed[s] = rows
    if missing:
        print("WARNING: missing checkpoints: " + ", ".join(missing))
    if per_seed:
        results["DuelingDQN"] = aggregate_across_seeds(per_seed)

    (out / "aggregate.json").write_text(json.dumps(results, indent=1))

    # --- report table
    print(f"{'policy':<12}" + "".join(f"{f[:15]:>17}" for f in METRIC_FIELDS))
    for name, agg in results.items():
        print(f"{name:<12}" + "".join(
            f"{agg[f]['mean']:>10.3f} ±{agg[f]['std']:<5.3f}" for f in METRIC_FIELDS))

    if "DuelingDQN" in results:
        print("\nDifference against the seed spread (3 seeds -- a comparison, not a test):")
        for base in ("FCFS", "SJF", "RoundRobin"):
            if base not in results:
                continue
            for f in ("avg_waiting_time", "missed_deadlines", "weighted_tardiness"):
                a, b = results["DuelingDQN"][f], results[base][f]
                verdict = "exceeds" if exceeds_seed_variation(a, b) else "within"
                better = "better" if a["mean"] < b["mean"] else "worse"
                print(f"  vs {base:<11} {f:<20} agent {better:<6} by "
                      f"{abs(a['mean']-b['mean']):8.3f}  ({verdict} seed spread)")


if __name__ == "__main__":
    main()
