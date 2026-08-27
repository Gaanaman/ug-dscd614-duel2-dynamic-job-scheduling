# Ownership, Report Split and Demonstration Script

The group mark is moderated by up to ±20% per member, informed by **the GitHub commit history**
and **the recorded demonstration**. A member whose contribution cannot be evidenced from either
source may be marked substantially below the group, including zero. Ownership below is designed
so that each member's contribution is visible in both places.

## Module ownership

Each member is the primary author of distinct modules and reviews the others' pull requests.

### Kyeremeh Faithful (22424515) — Environment & MDP

- `src/duel2/envs/job_generator.py`
- `src/duel2/envs/scheduling_env.py`
- `src/duel2/envs/observation.py`
- `src/duel2/envs/action_mask.py`
- `tests/test_env_api.py`, `tests/test_mask.py`
- `docs/mdp_spec.md`

### Daniel K. Adotey (22424924) — Agent & Training

- `src/duel2/agents/dueling_network.py`
- `src/duel2/agents/masked_dqn.py`
- `src/duel2/agents/replay.py`
- `src/duel2/utils/logging.py`, `src/duel2/utils/seeding.py`
- `scripts/train.py`, `configs/dueling_dqn.yaml`
- `docs/hyperparameters.md`

### Caleb Abakah Mensah (22424188) — Baselines & Evaluation

- `src/duel2/baselines/*`
- `src/duel2/evaluation/harness.py`, `metrics.py`, `aggregate.py`
- `src/duel2/utils/plotting.py`
- `scripts/evaluate.py`, `scripts/make_figures.py`
- `tests/test_metrics.py`
- `docs/experimental_protocol.md`

The reward function (`src/duel2/rewards/reward_fn.py`) is written jointly — it is the interface
between the environment and the metrics, and both owners need to agree on it.

## Report sections (4,000 words max, excluding references and appendices)

| Section | Target words | Author |
|---|---|---|
| 1. Introduction — problem, significance, aims | 400 | Faithful |
| 2. Background — Dueling DQN, why it suits scheduling, prior work | 600 | Daniel |
| 3. Problem formulation — the full MDP from `mdp_spec.md` | 900 | Faithful |
| 4. Methodology — environment, architecture, training, baseline design, protocol | 800 | Daniel + Caleb |
| 5. Results — training curves, baseline comparison, all metrics with spread | 600 | Caleb |
| 6. Discussion — convergence, stability, exploration, reward design | 400 | Daniel |
| 7. Limitations and deployment considerations | 200 | Caleb |
| 8. Conclusion and further work | 100 | Faithful |

## Recorded demonstration (20 minutes maximum, YouTube public or unlisted)

Every member must present a substantive technical portion. Reading a summary written by another
member is not participation and scores zero for that component.

| Minutes | Content | Presenter |
|---|---|---|
| 0–2 | Problem and why dynamic scheduling is an RL problem | Faithful |
| 2–7 | MDP formulation on slides: state, action, mask, reward equation, termination, Markov analysis | Faithful |
| 7–12 | Dueling architecture, the masked training loop, walk through *our* code | Daniel |
| 12–14 | Training curves across seeds | Daniel |
| 14–18 | Baselines, evaluation harness, results table, agent-vs-baseline comparison | Caleb |
| 18–20 | Gantt rollout of the trained policy, and the principal limitation of the work | Caleb |

Recordings beyond 20 minutes are marked on the first 20 minutes only — rehearse against a timer.
Only code the group wrote is walked through; library internals are not.

## Any member may be asked to explain any part of the submission

Rule 11 of the examination. Before recording, each member should be able to explain, without
notes: the reward equation and why each term is there; why the mask must be applied to the
bootstrap target; and how the evaluation instances are held out.
