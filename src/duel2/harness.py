"""Evaluation harness and cross-seed aggregation.

Owner: Caleb

One entry point runs any Policy -- the trained agent or any baseline -- over the
held-out instance set. Everything that could differ between policies is fixed
here rather than at the call site, so a policy cannot accidentally be evaluated
under conditions its competitor did not face.

Fixed by this module:
  * instance seeds 9000-9029, identical for every policy and every agent seed
  * exploration disabled: policies expose a deterministic ``act``
  * the same metrics.compute_metrics call
  * per-episode records written to logs/eval/ as JSONL, so every figure traces
    back to a committed log
"""

from __future__ import annotations

import statistics as st

from .jobs import EVAL_SEED_END, EVAL_SEED_START
from .metrics import compute_metrics

METRIC_FIELDS = ("makespan", "avg_waiting_time", "machine_utilisation",
                 "missed_deadlines", "weighted_tardiness", "cumulative_reward")


def eval_instance_seeds(n_episodes: int = 30) -> list[int]:
    """The held-out evaluation instances. The same list for every policy."""
    if EVAL_SEED_START + n_episodes > EVAL_SEED_END:
        raise ValueError(
            f"requested {n_episodes} evaluation episodes but the held-out range "
            f"[{EVAL_SEED_START}, {EVAL_SEED_END}) holds only "
            f"{EVAL_SEED_END - EVAL_SEED_START}"
        )
    return list(range(EVAL_SEED_START, EVAL_SEED_START + n_episodes))


def run_policy(policy, env, n_episodes: int = 30, record_rollout: bool = False):
    """Run one policy over the held-out set.

    Returns ``(rows, rollout)`` where rows are per-episode metric dicts and
    rollout is the decision trace of the first episode when requested.
    """
    rows, rollout = [], []
    for ep, instance_seed in enumerate(eval_instance_seeds(n_episodes)):
        obs, info = env.reset(options={"instance_seed": instance_seed})
        policy.reset()
        total = 0.0
        while True:
            obs, reward, terminated, truncated, info = env.step(
                policy.act(obs, info["action_mask"], info)
            )
            total += reward
            if terminated or truncated:
                break

        m = compute_metrics(env.completed, env.machine_busy_time, total)
        row = m.as_dict()
        row["instance_seed"] = instance_seed
        row["policy"] = policy.name
        rows.append(row)

        if record_rollout and ep == 0:
            rollout = [
                {"job": c.job.job_id, "machine": c.machine, "start": round(c.start, 3),
                 "finish": round(c.finish, 3), "deadline": round(c.job.deadline, 3),
                 "weight": c.job.weight, "processing_time": c.job.processing_time}
                for c in env.completed
            ]
    return rows, rollout


def aggregate_across_seeds(per_seed_rows: dict) -> dict:
    """Mean and standard deviation of each metric across seeds.

    Rubric: "Report the mean and the variation across seeds for every metric. A
    single number without a measure of spread is not accepted." And: "Do not
    report the best seed as the headline result."

    There is deliberately no argument that selects a seed. Making that
    impossible in code is more reliable than remembering not to do it at 2am on
    the fourth of September.

    Args:
        per_seed_rows: ``{seed: [episode metric dicts]}``
    """
    out = {}
    for field in METRIC_FIELDS:
        per_seed = [st.mean(r[field] for r in rows) for _, rows in sorted(per_seed_rows.items())]
        out[field] = {
            "mean": st.mean(per_seed),
            "std": st.pstdev(per_seed) if len(per_seed) > 1 else 0.0,
            "per_seed": per_seed,
            "n_seeds": len(per_seed),
            "n_episodes_per_seed": len(next(iter(per_seed_rows.values()))),
        }
    return out


def exceeds_seed_variation(agent_stat: dict, baseline_stat: dict) -> bool:
    """Whether a difference is larger than the seed-to-seed variation.

    With three seeds this is a comparison against the spread, not a significance
    test. Three samples do not support one, and claiming otherwise will be marked
    down. Phrase the finding in the report as "the difference exceeds / does not
    exceed the variation across seeds".
    """
    diff = abs(agent_stat["mean"] - baseline_stat["mean"])
    spread = agent_stat["std"] + baseline_stat["std"]
    return diff > spread
