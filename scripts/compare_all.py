"""Score every policy and every agent variant through one code path.

Owner: Caleb

    python scripts/compare_all.py --out logs/eval/comparison.json

Produces the results table for the report. Every policy -- the eight fixed
dispatching rules, the three required baselines, the random floor, and each
trained agent variant -- runs through harness.run_policy on the same 30 held-out
instances with exploration disabled and the same metric code.

Two comparisons are reported, and the distinction matters:

  * against the REQUIRED baselines (FCFS, SJF, Round Robin), which the brief asks
    for;
  * against the BEST SINGLE RULE, which is the bar Han and Yang (2020) set and
    the honest bar under the rule action space, because a rule inside the action
    set is reachable by a policy that always selects it.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from dataclasses import replace
from pathlib import Path

from duel2.agent import GreedyAgentPolicy
from duel2.baselines import FCFS, SJF, FixedRule, RandomMasked, RoundRobin
from duel2.env import DynamicJobShopEnv, EnvConfig
from duel2.harness import run_policy
from duel2.rules import N_RULES, RULE_NAMES

MET = ("avg_waiting_time", "missed_deadlines", "weighted_tardiness",
       "makespan", "machine_utilisation", "cumulative_reward")
LOWER_BETTER = {"avg_waiting_time", "missed_deadlines", "weighted_tardiness", "makespan"}


def agg(rows_per_seed):
    out = {}
    for m in MET:
        per = [st.mean(r[m] for r in rows) for rows in rows_per_seed]
        out[m] = {"mean": st.mean(per), "std": st.pstdev(per) if len(per) > 1 else 0.0,
                  "per_seed": per}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-config", default="configs/env_default.yaml")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--out", default="logs/eval/comparison.json")
    ap.add_argument("--variant", action="append", default=[],
                    metavar="NAME=DIR", help="trained rule-action variant, repeatable")
    ap.add_argument("--direct-models", default="models",
                    help="directory of direct-action checkpoints")
    args = ap.parse_args()

    base = EnvConfig.from_yaml(args.env_config)
    direct_env = DynamicJobShopEnv(base)
    rules_env = DynamicJobShopEnv(replace(base, action_mode="rules"))
    results: dict[str, dict] = {}

    # --- required baselines and the diagnostic floor, direct environment
    for pol in (RandomMasked(0), FCFS(), SJF(), RoundRobin(base.n_machines)):
        rows, _ = run_policy(pol, direct_env, n_episodes=args.episodes)
        results[pol.name] = agg([rows])

    # --- every fixed dispatching rule, rules environment
    for i, name in enumerate(RULE_NAMES):
        rows, _ = run_policy(FixedRule(i), rules_env, n_episodes=args.episodes)
        results[f"rule:{name}"] = agg([rows])

    # --- direct-action agent
    per_seed = []
    for s in (0, 1, 2):
        ck = Path(args.direct_models) / f"dueling_dqn_seed{s}.pt"
        if ck.exists():
            pol = GreedyAgentPolicy.load(ck, direct_env.observation_space.shape[0],
                                         int(direct_env.action_space.n))
            per_seed.append(run_policy(pol, direct_env, n_episodes=args.episodes)[0])
    if per_seed:
        results["agent:direct"] = agg(per_seed)

    # --- rule-action agent variants
    for spec in args.variant:
        name, _, d = spec.partition("=")
        per_seed = []
        for s in (0, 1, 2):
            ck = Path(d) / f"models/dueling_dqn_seed{s}.pt"
            if ck.exists():
                pol = GreedyAgentPolicy.load(ck, rules_env.observation_space.shape[0], N_RULES)
                per_seed.append(run_policy(pol, rules_env, n_episodes=args.episodes)[0])
        if per_seed:
            results[f"agent:{name}"] = agg(per_seed)
        else:
            print(f"WARNING: no checkpoints found for variant {name} in {d}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1))

    # ---------------------------------------------------------------- report
    print(f"{'policy':<22}" + "".join(f"{m[:13]:>16}" for m in MET))
    for k, v in results.items():
        print(f"{k:<22}" + "".join(
            f"{v[m]['mean']:>10.3f}±{v[m]['std']:<5.3f}" for m in MET))

    rules_only = {k: v for k, v in results.items() if k.startswith("rule:")}
    bar = max(rules_only, key=lambda k: rules_only[k]["cumulative_reward"]["mean"])
    print(f"\nBAR (best single rule by return): {bar} "
          f"{rules_only[bar]['cumulative_reward']['mean']:+.3f}")

    print("\nAgent variants against the bar and against the required baselines:")
    for k, v in results.items():
        if not k.startswith("agent:"):
            continue
        r, sd = v["cumulative_reward"]["mean"], v["cumulative_reward"]["std"]
        d = r - rules_only[bar]["cumulative_reward"]["mean"]
        verdict = "BEATS" if d > 0 else "below"
        conf = "exceeds seed spread" if abs(d) > sd else "within seed spread"
        print(f"  {k:<20} return {r:+.3f}±{sd:.3f}  {verdict} {bar} by {abs(d):.3f} ({conf})")
        for req in ("FCFS", "SJF", "RoundRobin"):
            dr = r - results[req]["cumulative_reward"]["mean"]
            print(f"      vs {req:<11} {'beats' if dr > 0 else 'loses to':<9} by {abs(dr):.3f}")


if __name__ == "__main__":
    main()
