# DUEL-2 — Dynamic Job Scheduling with a Dueling DQN

DSCD 614 Reinforcement Learning, University of Ghana · Group 11 · Option DUEL-2

A Dueling Deep Q-Network learns to dispatch stochastically arriving jobs onto a bank of
parallel machines, and is compared against First-Come-First-Served, Shortest-Job-First and
Round Robin under an identical evaluation protocol.

| | |
|---|---|
| Algorithm (binding) | Dueling DQN |
| Environment | `DynamicJobShop-v0` — custom Gymnasium environment, synthetic job stream |
| Required baseline | FCFS / SJF / Round Robin (all three implemented) |
| Headline metrics | makespan · average waiting time · machine utilisation · missed deadlines · cumulative reward |
| Seeds | 3 training seeds, 30 held-out evaluation episodes per seed |

> **Visibility.** This repository is private during the exam window and is made public on
> 4 September 2026 before submission, as required. See
> [`docs/github_plan.md`](docs/github_plan.md) for the freeze-and-publish procedure.

## Members

| Member | ID | Owns |
|---|---|---|
| Kyeremeh Faithful | 22424515 | Environment, observation construction, action masking, termination logic |
| Daniel K. Adotey | 22424924 | Dueling network, masked training loop, logging, hyperparameter configuration |
| Caleb Abakah Mensah | 22424188 | Baselines, metrics, evaluation harness, aggregation, figures |

## Install

Python 3.10+ required. From a clean environment:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Reproduce the headline result

One entry point runs training across all seeds, evaluates the agent and all three baselines
through the same code path, and regenerates every figure in the report:

```bash
bash scripts/run_all.sh
```

Individual stages:

```bash
python scripts/train.py    --config configs/dueling_dqn.yaml --seed 0
python scripts/evaluate.py --config configs/eval.yaml --policy models/dueling_dqn_seed0.pt
python scripts/make_figures.py --logs logs/ --out figures/
```

## Figure provenance

Every figure in the report regenerates from logs committed under `logs/`. `make_figures.py`
reads only committed logs — it never re-runs the environment — so a figure that cannot be
traced to a log file cannot be produced.

| Figure | Script | Source log |
|---|---|---|
| Training return vs. steps (mean ± spread over seeds) | `make_figures.py --fig training_curve` | `logs/train/seed_*/progress.csv` |
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
configs/       env, agent and evaluation configuration (YAML, version controlled)
docs/          MDP specification, protocol, hyperparameters, AI-use declaration
scripts/       train, evaluate, make_figures, run_all
tests/         Gymnasium API conformance, mask correctness, reward and metric unit tests
logs/          raw run logs — committed, figures trace back here
```

## Documents

- [`docs/mdp_spec.md`](docs/mdp_spec.md) — state space, action space, reward equation, termination, discount, Markov analysis
- [`docs/experimental_protocol.md`](docs/experimental_protocol.md) — seeds, held-out instances, evaluation rules
- [`docs/hyperparameters.md`](docs/hyperparameters.md) — full configuration table and deviations from library defaults
- [`docs/roles_and_split.md`](docs/roles_and_split.md) — ownership, report sections, demonstration script
- [`docs/ai_use_declaration.md`](docs/ai_use_declaration.md) — declaration of generative AI use

## Attribution

Third-party code adapted into this repository is attributed at the point of use in the source
file and listed in the report. See `docs/attribution.md`.
