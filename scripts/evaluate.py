"""Evaluate the agent and every baseline under identical conditions.

Owner: Caleb

    python scripts/evaluate.py --config configs/eval.yaml

Runs every policy named in the config through evaluation.harness.run_policy on
the same held-out instances, with exploration disabled, and writes per-episode
records to logs/eval/*.jsonl. Aggregation across seeds happens here; the figure
script only reads what this produces.
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/eval.yaml")
    p.add_argument("--models-dir", default="models/")
    p.add_argument("--out", default="logs/eval/")
    return p.parse_args()


def main() -> None:
    raise NotImplementedError("TODO")


if __name__ == "__main__":
    main()
