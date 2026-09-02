"""Evaluation harness.

Owner: Caleb

One entry point runs any Policy -- the trained agent or any baseline -- over the
held-out instance set. Everything that could differ between policies is fixed
here rather than at the call site, so a policy cannot accidentally be evaluated
under different conditions from its competitor.

Fixed by this module:
  * instance seeds 9000-9029, identical for every policy and every agent seed
  * exploration disabled (epsilon = 0, deterministic greedy over the mask)
  * the same metrics.compute_metrics call
  * per-episode records written to logs/eval/ as JSONL so every figure traces
    back to a committed log
"""

from __future__ import annotations

from .jobs import EVAL_SEED_END, EVAL_SEED_START


def eval_instance_seeds(n_episodes: int = 30) -> list[int]:
    """The held-out evaluation instances. Same list for every policy."""
    seeds = list(range(EVAL_SEED_START, EVAL_SEED_START + n_episodes))
    if seeds[-1] >= EVAL_SEED_END:
        raise ValueError(
            f"requested {n_episodes} evaluation episodes but the held-out range "
            f"[{EVAL_SEED_START}, {EVAL_SEED_END}) holds only "
            f"{EVAL_SEED_END - EVAL_SEED_START}"
        )
    return seeds


def run_policy(policy, env_config, n_episodes: int = 30, log_path=None, record_rollout=False):
    """Run one policy over the held-out set and return per-episode metrics.

    Set ``record_rollout`` for one policy to dump a full decision trace for the
    Gantt figure and the demonstration video.
    """
    raise NotImplementedError("TODO")


# ----------------------------------------------------------------- aggregation
#
# Rubric: "Report the mean and the variation across seeds for every metric. A
# single number without a measure of spread is not accepted." And: "Do not
# report the best seed as the headline result."
#
# aggregate_across_seeds therefore returns mean and standard deviation and has
# no option to select a seed. Making that impossible in code is more reliable
# than remembering not to do it at 2am on the fourth of September.


def aggregate_across_seeds(per_seed_metrics: dict) -> dict:
    """Mean and standard deviation of each metric across seeds.

    Args:
        per_seed_metrics: ``{seed: [EpisodeMetrics, ...]}``

    Returns:
        ``{metric_name: {"mean": float, "std": float, "per_seed": [...]}}``
    """
    raise NotImplementedError("TODO")


def exceeds_seed_variation(agent_stat: dict, baseline_stat: dict) -> bool:
    """Whether a difference is larger than the seed-to-seed variation.

    With three seeds this is a comparison against the spread, not a significance
    test. Three samples do not support one, and claiming otherwise will be
    marked down. Phrase the finding in the report as "the difference exceeds /
    does not exceed the variation across seeds".
    """
    raise NotImplementedError("TODO")
