"""Validate that the final experiment artifacts are complete and consistent.

This check is intentionally lightweight: it uses only the Python standard
library, does not train an agent, and never modifies an artifact.

    python scripts/validate_submission_artifacts.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


TRAINING_SEEDS = (0, 1, 2)
EVALUATION_EPISODES = 30
FINAL_STEP = 1_000_000
POLICIES = ("FCFS", "SJF", "RoundRobin")
METRICS = (
    "makespan",
    "avg_waiting_time",
    "machine_utilisation",
    "missed_deadlines",
    "weighted_tardiness",
    "cumulative_reward",
)
FIGURES = ("training_curve.png", "baseline_bars.png", "rollout_gantt.png")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def validate(root: Path) -> list[str]:
    errors: list[str] = []

    for seed in TRAINING_SEEDS:
        path = root / f"logs/train/seed_{seed}/progress.csv"
        require(path.is_file(), f"missing training log: {path}", errors)
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        require(bool(rows), f"empty training log: {path}", errors)
        if rows:
            require(
                int(rows[-1]["global_step"]) == FINAL_STEP,
                f"{path} ends at step {rows[-1]['global_step']}, expected {FINAL_STEP}",
                errors,
            )

    evaluation_rows: dict[str, list[dict]] = {}
    for policy in POLICIES:
        path = root / f"logs/eval/{policy}.jsonl"
        require(path.is_file(), f"missing baseline evaluation log: {path}", errors)
        if path.is_file():
            evaluation_rows[policy] = read_jsonl(path)

    for seed in TRAINING_SEEDS:
        name = f"DuelingDQN_seed{seed}"
        path = root / f"logs/eval/{name}.jsonl"
        require(path.is_file(), f"missing agent evaluation log: {path}", errors)
        if path.is_file():
            evaluation_rows[name] = read_jsonl(path)

    expected_instance_seeds = list(range(9000, 9000 + EVALUATION_EPISODES))
    for name, rows in evaluation_rows.items():
        require(
            len(rows) == EVALUATION_EPISODES,
            f"{name} has {len(rows)} episodes, expected {EVALUATION_EPISODES}",
            errors,
        )
        actual = [int(row["instance_seed"]) for row in rows]
        require(
            actual == expected_instance_seeds,
            f"{name} does not use held-out seeds 9000-9029 in order",
            errors,
        )
        for index, row in enumerate(rows):
            missing = [metric for metric in METRICS if metric not in row]
            require(not missing, f"{name} episode {index} misses metrics: {missing}", errors)

    aggregate_path = root / "logs/eval/aggregate.json"
    require(aggregate_path.is_file(), f"missing aggregate: {aggregate_path}", errors)
    if aggregate_path.is_file():
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
        headline_policies = {"FCFS", "SJF", "RoundRobin", "DuelingDQN"}
        missing_policies = headline_policies - set(aggregate)
        require(
            not missing_policies,
            f"aggregate.json misses headline policies: {sorted(missing_policies)}",
            errors,
        )
        for policy, stats in aggregate.items():
            for metric in METRICS:
                require(metric in stats, f"aggregate {policy} misses {metric}", errors)
                if metric in stats:
                    require(
                        math.isfinite(float(stats[metric]["mean"])),
                        f"aggregate {policy}/{metric} has a non-finite mean",
                        errors,
                    )

    rollout = root / "logs/eval/rollout_seed0.jsonl"
    require(rollout.is_file(), f"missing rollout trace: {rollout}", errors)
    if rollout.is_file():
        require(len(read_jsonl(rollout)) == 50, "rollout trace must contain 50 jobs", errors)

    for seed in TRAINING_SEEDS:
        model = root / f"models/dueling_dqn_seed{seed}.pt"
        require(model.is_file() and model.stat().st_size > 0, f"missing model: {model}", errors)

    for name in FIGURES:
        figure = root / "figures" / name
        require(figure.is_file() and figure.stat().st_size > 0, f"missing figure: {figure}", errors)

    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    errors = validate(args.root)
    if errors:
        print("Submission artifact validation FAILED:")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)

    print("Submission artifact validation passed.")
    print("  3 complete training logs ending at 1,000,000 steps")
    print("  6 evaluation logs using held-out seeds 9000-9029")
    print("  aggregate results, rollout trace, 3 models, and 3 figures present")


if __name__ == "__main__":
    main()
