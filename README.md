# Dynamic Job Scheduling with a Dueling DQN

A Dueling Deep Q-Network learns to dispatch stochastically arriving jobs onto a bank of
parallel machines, and is compared against First-Come-First-Served, Shortest-Job-First and
Round Robin under an identical evaluation protocol.

Coursework project for DSCD 614 (Reinforcement Learning), University of Ghana.

| | |
|---|---|
| Algorithm | Dueling DQN, n-step 3 returns |
| Environment | `DynamicJobShop-v0` — custom Gymnasium environment, synthetic job stream |
| Action space | Eight dispatching rules (headline); direct job/machine assignment (ablation) |
| Baselines | FCFS, SJF, Round Robin (required); all eight rules; uniform-random floor |
| Headline result | Beats all three required baselines on every metric and seed; **below the best single rule (ATC) by 0.109** |
| Metrics | makespan · average waiting time · machine utilisation · missed deadlines · cumulative reward |
| Protocol | 3 training seeds, 30 held-out evaluation episodes per seed |

## Install

Python 3.10 or later. From a clean environment:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Reproduce the headline result

One entry point trains across all seeds, evaluates the agent and all three baselines through
the same code path, and regenerates every figure:

```bash
bash scripts/run_all.sh
```

Before training, confirm the instance distribution still has headroom for the agent:

```bash
python scripts/check_load.py
```

It runs all three baselines and warns if the arrival process is setting the makespan, if the
baselines sit within noise of each other, or if the deadline miss rate has saturated.

Individual stages:

```bash
python scripts/train.py       --config configs/dueling_dqn.yaml --seed 0
python scripts/compare_all.py --out logs/eval/comparison.json
python scripts/make_figures.py --logs logs/ --out figures/

# the direct-action ablation
python scripts/train.py --config configs/dueling_dqn.yaml \
    --env-config configs/env_direct.yaml --seed 0 --out-dir runs/direct
```

## Figure provenance

Every figure regenerates from logs committed under `logs/`. `make_figures.py` reads only
committed logs and never steps the environment, so no figure can exist without a log behind it.

| Figure | Command | Source log |
|---|---|---|
| Training return vs. steps, mean and spread across seeds | `make_figures.py --fig training_curve` | `logs/train/seed_*/progress.csv` |
| Agent vs. baselines, all metrics | `make_figures.py --fig baseline_bars` | `logs/eval/*.jsonl` |
| Gantt rollout of the trained policy | `make_figures.py --fig rollout_gantt` | `logs/eval/rollout_seed0.jsonl` |

## Layout

```
src/duel2/
  env.py           Gymnasium environment: decision epochs, step, termination
  jobs.py          synthetic job instance generator, held-out seed guard
  observation.py   the 69-dimensional observation vector
  action_mask.py   action validity mask, build and apply
  reward.py        the four-term reward equation and its weights
  network.py       dueling Q-network, value and advantage heads
  agent.py         masked DQN training loop
  baselines.py     Policy interface, FCFS, SJF, Round Robin
  metrics.py       episode metrics
  harness.py       evaluation harness and cross-seed aggregation
  runtime.py       seeding, structured logging, plotting
configs/       env, agent and evaluation configuration
docs/          MDP specification, protocol, hyperparameters
scripts/       train, evaluate, make_figures, run_all
tests/         Gymnasium API conformance, mask correctness, reward and metric tests
```

`run_all.sh` creates `logs/`, `figures/`, and `models/` on first run. Raw run logs are
committed once they exist — every figure traces back to one.

## Documents

- [`docs/mdp_spec.md`](docs/mdp_spec.md) — state space, action space, reward equation, termination, discount, Markov analysis
- [`docs/experimental_protocol.md`](docs/experimental_protocol.md) — seeds, held-out instances, evaluation rules
- [`docs/hyperparameters.md`](docs/hyperparameters.md) — full configuration table and deviations from library defaults
- [`docs/attribution.md`](docs/attribution.md) — third-party code adapted into this repository
- [`docs/ai_use_declaration.md`](docs/ai_use_declaration.md) — declaration of generative AI use

## Module ownership

| Area | Owner | GitHub |
|---|---|---|
| Environment, observation construction, action masking, termination | Faithful | [@SteadyHands01](https://github.com/SteadyHands01) |
| Dueling network, masked training loop, logging, hyperparameters | Daniel | [@Gaanaman](https://github.com/Gaanaman) |
| Baselines, metrics, evaluation harness, aggregation, figures | Caleb | [@Caleb-Abakah](https://github.com/Caleb-Abakah) |

`src/duel2/reward.py` is written jointly — it is the interface between the
environment and the metrics.

## Working on this

`main` is protected. Branch, open a pull request, and have another member review it.

```bash
git checkout -b env/action-mask
```

Set your git identity to the email registered on your GitHub account before your first commit,
or your commits will not attribute to you:

```bash
git config user.email "your-github-email@example.com"
```

Commit prefixes: `feat:`, `fix:`, `test:`, `docs:`, `chore:`, and `exp:` for commits that carry
run logs.

## Attribution

Third-party code adapted into this repository is attributed at the point of use in the source
file and listed in [`docs/attribution.md`](docs/attribution.md).

### Validate final experiment artifacts

Before building the submission package, verify that the committed training logs,
evaluation records, model checkpoints and figures are complete:

```bash
python scripts/validate_submission_artifacts.py