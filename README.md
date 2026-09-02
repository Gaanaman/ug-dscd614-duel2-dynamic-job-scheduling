# Dynamic Job Scheduling with a Dueling DQN

A Dueling Deep Q-Network learns to dispatch stochastically arriving jobs onto a bank of
parallel machines, and is compared against First-Come-First-Served, Shortest-Job-First and
Round Robin under an identical evaluation protocol.

Coursework project for DSCD 614 (Reinforcement Learning), University of Ghana.

| | |
|---|---|
| Algorithm | Dueling DQN |
| Environment | `DynamicJobShop-v0` — custom Gymnasium environment, synthetic job stream |
| Baselines | FCFS, SJF, Round Robin |
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

Individual stages:

```bash
python scripts/train.py    --config configs/dueling_dqn.yaml --seed 0
python scripts/evaluate.py --config configs/eval.yaml
python scripts/make_figures.py --logs logs/ --out figures/
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
  envs/        job generator, Gymnasium environment, observation, action mask
  rewards/     reward function and component weights
  agents/      dueling Q-network, masked epsilon-greedy, training loop
  baselines/   FCFS, SJF, Round Robin behind one Policy interface
  evaluation/  harness, metrics, cross-seed aggregation
  utils/       seeding, structured logging, plotting
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

| Area | Owner |
|---|---|
| Environment, observation construction, action masking, termination | Faithful |
| Dueling network, masked training loop, logging, hyperparameters | Daniel |
| Baselines, metrics, evaluation harness, aggregation, figures | Caleb |

`src/duel2/rewards/reward_fn.py` is written jointly — it is the interface between the
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
