"""Regenerate every figure in the report from committed logs.

Owner: Caleb

    python scripts/make_figures.py --logs logs/ --out figures/

This script reads logs only. It never steps the environment and never loads a
model. That is deliberate: "figures that cannot be traced to committed logs are
not credited", and the cleanest guarantee is to make plotting anything else
structurally impossible.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from duel2.runtime import ablation_bars, baseline_bars, read_jsonl, rollout_gantt, training_curve

FIGURES = ("training_curve", "baseline_bars", "rollout_gantt", "ablation")
BAR_METRICS = [("avg_waiting_time", True), ("missed_deadlines", True),
               ("weighted_tardiness", True), ("machine_utilisation", False)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--fig", choices=FIGURES, default=None, help="default: all")
    args = ap.parse_args()

    logs, out = Path(args.logs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    want = (args.fig,) if args.fig else FIGURES
    made = []

    if "training_curve" in want:
        per_seed = {}
        for d in sorted((logs / "train").glob("seed_*")):
            f = d / "progress.csv"
            if f.exists():
                per_seed[int(d.name.split("_")[1])] = list(csv.DictReader(f.open()))
        if per_seed:
            fig, ax = plt.subplots(figsize=(7, 4))
            training_curve(per_seed, ax=ax)
            ax.set_title(f"Training return, {len(per_seed)} seeds", fontsize=10, loc="left")
            fig.tight_layout(); fig.savefig(out / "training_curve.png", dpi=180)
            plt.close(fig); made.append("training_curve.png")

    agg_path = logs / "eval" / "aggregate.json"
    if "baseline_bars" in want and agg_path.exists():
        agg = json.loads(agg_path.read_text())
        fig, axes = plt.subplots(1, len(BAR_METRICS), figsize=(4 * len(BAR_METRICS), 3.4))
        for ax, (metric, lower) in zip(axes, BAR_METRICS):
            baseline_bars(agg, metric, ax=ax, lower_is_better=lower)
            ax.tick_params(axis="x", rotation=20)
        fig.tight_layout(); fig.savefig(out / "baseline_bars.png", dpi=180)
        plt.close(fig); made.append("baseline_bars.png")

    roll = logs / "eval" / "rollout_seed0.jsonl"
    if "rollout_gantt" in want and roll.exists():
        fig, ax = plt.subplots(figsize=(9, 3.4))
        rollout_gantt(read_jsonl(roll), ax=ax)
        ax.set_title("Trained agent, one held-out episode (red = missed deadline)",
                     fontsize=10, loc="left")
        fig.tight_layout(); fig.savefig(out / "rollout_gantt.png", dpi=180)
        plt.close(fig); made.append("rollout_gantt.png")

    comp_path = logs / "eval" / "comparison.json"
    if "ablation" in want and comp_path.exists():
        comp = json.loads(comp_path.read_text())
        fig, ax = plt.subplots(figsize=(7.6, 4))
        ablation_bars(comp, ax=ax)
        ax.set_title("Agent variants against the best single dispatching rule",
                     fontsize=10, loc="left")
        fig.tight_layout(); fig.savefig(out / "ablation.png", dpi=180)
        plt.close(fig); made.append("ablation.png")

    print("wrote: " + (", ".join(made) if made else "nothing -- no logs found"))


if __name__ == "__main__":
    main()
