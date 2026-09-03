"""Seeding, run logging and plotting.

Owners: Daniel (seeding, logging), Caleb (plotting)

SEEDING
Three independent streams per run, so agent stochasticity and instance
difficulty never become confounded:

    seed              -> network init, exploration
    seed's own band   -> training job instances, inside [0, 9000)
    9000 + i          -> evaluation instances, fixed across seeds AND policies

Training instances are partitioned into disjoint bands of 3000, one per seed, so
seed 0 draws from [0, 3000), seed 1 from [3000, 6000) and seed 2 from [6000,
9000). Every band is provably below EVAL_SEED_START, so no training episode can
ever land on a held-out instance -- and the seeds do not train on each other's
instances either, which keeps the three runs genuinely independent.

The third stream is what makes the agent-vs-baseline comparison paired.

LOGGING
Everything plotted in the report comes out of these files. make_figures.py reads
logs only and never steps the environment, so no figure can exist that is not
traceable to a committed log.

PLOTTING
Training curves show the mean across seeds with a shaded spread -- never a single
seed, and never seeds overplotted without an aggregate.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np

EVAL_SEED_START = 9000


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch. Record the value in the report."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


TRAIN_BAND = 3000     # instances per training seed


def training_instance_seed(seed: int, episode: int) -> int:
    """Instance seed for one training episode. Always below the held-out range.

    Seed s owns the band [s*3000, (s+1)*3000) modulo 9000; the episode index
    wraps inside that band. The assertion is not decoration -- an off-by-one here
    silently trains on evaluation instances and invalidates every number in the
    report.
    """
    band_start = (seed * TRAIN_BAND) % EVAL_SEED_START
    s = band_start + (episode % TRAIN_BAND)
    assert 0 <= s < EVAL_SEED_START, f"training instance seed {s} entered the held-out range"
    return s


class RunLogger:
    """Append-only CSV writer with a fixed header."""

    def __init__(self, path, fields):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fields = list(fields)
        self._f = self.path.open("w", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=self.fields)
        self._w.writeheader()

    def log(self, **kwargs) -> None:
        self._w.writerow({k: kwargs.get(k, "") for k in self.fields})
        self._f.flush()

    def close(self) -> None:
        self._f.close()

    def __enter__(self): return self
    def __exit__(self, *a): self.close()


TRAIN_FIELDS = ["global_step", "episode", "episode_return", "episode_length", "epsilon",
                "loss", "mean_q", "makespan", "avg_waiting_time", "missed_deadlines",
                "weighted_tardiness"]


def write_jsonl(path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path) -> list[dict]:
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


# ------------------------------------------------------------------ plotting

PALETTE = {"DuelingDQN": "#c2521a", "FCFS": "#3c5c74", "SJF": "#166b58", "RoundRobin": "#8a6a3f"}


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=.25, linewidth=.6)
    ax.set_axisbelow(True)


def training_curve(per_seed_rows, ax=None, smooth: int = 25):
    """Mean episode return vs. environment steps, shaded by spread across seeds.

    Args:
        per_seed_rows: ``{seed: [row dicts from progress.csv]}``
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    grids, ref = [], None
    for seed, rows in sorted(per_seed_rows.items()):
        steps = np.array([float(r["global_step"]) for r in rows])
        rets = np.array([float(r["episode_return"]) for r in rows])
        if smooth > 1 and len(rets) > smooth:
            kern = np.ones(smooth) / smooth
            rets = np.convolve(rets, kern, mode="valid")
            steps = steps[smooth - 1:]
        ref = steps if ref is None or len(steps) < len(ref) else ref
        grids.append((steps, rets))

    common = np.linspace(max(g[0][0] for g in grids), min(g[0][-1] for g in grids), 200)
    stack = np.vstack([np.interp(common, s, r) for s, r in grids])
    mean, std = stack.mean(axis=0), stack.std(axis=0)

    ax.plot(common, mean, color=PALETTE["DuelingDQN"], lw=1.8, label="Dueling DQN (mean of seeds)")
    ax.fill_between(common, mean - std, mean + std, color=PALETTE["DuelingDQN"], alpha=.18,
                    label="± 1 s.d. across seeds")
    ax.set_xlabel("environment steps")
    ax.set_ylabel("episode return")
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    return ax


def baseline_bars(aggregated, metric: str, ax=None, lower_is_better: bool = True):
    """Agent vs. baselines on one metric, error bars from the spread across seeds."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(5.4, 3.6))

    names = list(aggregated.keys())
    means = [aggregated[n][metric]["mean"] for n in names]
    stds = [aggregated[n][metric]["std"] for n in names]
    colors = [PALETTE.get(n, "#777") for n in names]

    ax.bar(names, means, yerr=stds, capsize=4, color=colors, edgecolor="none", width=.62)
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(("lower is better" if lower_is_better else "higher is better"),
                 fontsize=9, color="#666", loc="right")
    _style(ax)
    return ax


def rollout_gantt(records, ax=None):
    """Machine lanes over simulated time for a single episode.

    The demonstration must show a rollout of the trained agent. This environment
    has no graphical renderer, so this figure is what satisfies that requirement.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3.4))

    for r in records:
        late = r["finish"] > r["deadline"]
        ax.barh(r["machine"], r["finish"] - r["start"], left=r["start"], height=.62,
                color="#f1dedd" if late else "#dbe9e4",
                edgecolor="#93302c" if late else "#166b58", linewidth=.9)
    n_m = max(r["machine"] for r in records) + 1
    ax.set_yticks(range(n_m), [f"m{m}" for m in range(n_m)])
    ax.invert_yaxis()
    ax.set_xlabel("simulated time")
    _style(ax)
    return ax


def ablation_bars(comparison: dict, metric: str = "cumulative_reward", ax=None,
                  bar_key: str = "rule:ATC"):
    """Agent variants against the best single dispatching rule.

    The horizontal line is the bar: the best fixed rule inside the action set.
    Any variant below it has not learned to select better than a policy that
    always picks one rule, which is the comparison Han and Yang (2020) make and
    the only one that means anything under a rule action space.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(7.6, 4))

    order = [k for k in comparison if k.startswith("agent:")]
    labels = [k.replace("agent:", "") for k in order]
    means = [comparison[k][metric]["mean"] for k in order]
    stds = [comparison[k][metric]["std"] for k in order]
    bar = comparison[bar_key][metric]["mean"]

    # Returns are negative and the axis does not include zero, so bars are drawn
    # from the axis floor upward rather than from zero downward.
    floor = min(means) - 0.14
    colours = ["#c2521a" if m == max(means) else "#8a8f88" for m in means]
    ax.bar(labels, [m - floor for m in means], bottom=floor, yerr=stds, capsize=4,
           color=colours, edgecolor="none", width=.6)
    ax.axhline(bar, color="#166b58", lw=1.6, ls="--",
               label=f"best single rule ({bar_key.split(':')[1]}) {bar:+.3f}")

    for x, (m, sd) in enumerate(zip(means, stds)):
        ax.text(x, m + sd + 0.012, f"{m:+.3f}", ha="center", va="bottom", fontsize=8.5,
                family="monospace")

    ax.set_ylabel(metric.replace("_", " "))
    ax.set_ylim(floor, max(max(means), bar) + 0.08)
    ax.tick_params(axis="x", rotation=18)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _style(ax)
    return ax
