"""Cross-seed aggregation.

Owner: Caleb

Rubric: "Report the mean and the variation across seeds for every metric. A
single number without a measure of spread is not accepted." And: "Do not report
the best seed as the headline result."

``aggregate_across_seeds`` therefore returns mean and standard deviation and has
no option to select a seed. Making that impossible in code is more reliable than
remembering not to do it at 2am on the fourth of September.
"""

from __future__ import annotations


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
    test. Three samples do not support one, and claiming otherwise will be marked
    down. Phrase the finding in the report as "the difference exceeds / does not
    exceed the variation across seeds".
    """
    raise NotImplementedError("TODO")
