"""Regenerate every figure in the report from committed logs.

Owner: Caleb

    python scripts/make_figures.py --logs logs/ --out figures/

This script reads logs only. It never steps the environment and never loads a
model. That is deliberate: "figures that cannot be traced to committed logs are
not credited", and the cleanest way to guarantee traceability is to make it
structurally impossible to plot anything that is not in a committed log file.

Figures:
    training_curve  mean and spread of episode return across seeds vs. steps
    baseline_bars   agent vs. FCFS/SJF/RoundRobin on every metric, with spread
    rollout_gantt   machine lanes over time for one trained-policy rollout
"""

from __future__ import annotations

import argparse

FIGURES = ("training_curve", "baseline_bars", "rollout_gantt")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--logs", default="logs/")
    p.add_argument("--out", default="figures/")
    p.add_argument("--fig", choices=FIGURES, default=None, help="default: all")
    return p.parse_args()


def main() -> None:
    raise NotImplementedError("TODO")


if __name__ == "__main__":
    main()
