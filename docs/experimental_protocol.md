# Experimental Protocol

This protocol is fixed before any result exists, so that no rule can be chosen after seeing the
numbers it would govern.

## Seeds

Three training seeds: **0, 1, 2**. Recorded here and in the report.

Hyperparameters are held **identical** across seeds. A configuration tuned per seed invalidates
the comparison and will be treated as such.

Each seed controls three independent streams, seeded separately so they do not interfere:

| Stream | Seeded by | Purpose |
|---|---|---|
| network init + exploration | `seed` | agent stochasticity |
| training job instances | `seed + 1000` | the instances the agent trains on |
| evaluation job instances | `9000 + episode_index` | **fixed across seeds and policies** |

The third row is the important one. Every policy — the agent at each seed, FCFS, SJF and Round
Robin — is evaluated on the *same 30 job instances*. This turns the comparison into a paired
one and removes instance difficulty as a confound.

## Held-out instances

Evaluation instance seeds `9000–9029` are never generated during training. Training draws from
`seed + 1000` and increments per episode within a range that terminates below 9000. Assert this
in `harness.py` so an accidental overlap fails loudly rather than silently inflating
the result.

## Evaluation settings

- Exploration disabled: `ε = 0`, deterministic greedy action selection over the valid mask.
- 30 evaluation episodes per seed.
- The agent and all baselines run through **the same `harness.run_policy()` call path** and the
  same `metrics.py` code. Baselines are `Policy` objects, not a separate script.

## Metrics

Reported for every policy as **mean ± standard deviation across the three seeds**. A single number
carries no information about run-to-run variation, so every metric is reported with its spread.

| Metric | Definition | Direction |
|---|---|---|
| makespan | `max_j C_j − min_j a_j` | lower better |
| average waiting time | `mean_j (start_j − a_j)` | lower better |
| machine utilisation | busy machine-time / (`M` × makespan) | higher better |
| missed deadlines | `|{ j : C_j > d_j }| / N` | lower better |
| weighted tardiness | `Σ_j w_j · max(0, C_j − d_j)` | lower better |
| cumulative reward | episode return | higher better |

Baselines are deterministic, so their variation across seeds comes only from the instance set —
which is fixed. Expect near-zero spread for baselines and say why in the report rather than
leaving the reader to wonder.

## Claiming a difference

State whether an observed difference exceeds the variation across seeds. With three seeds, the
honest phrasing is a comparison of the difference against the seed spread, not a p-value —
three samples do not support a significance test.

Where the agent does not beat a baseline, **that is a reportable result**, and the diagnosis is
reported with the evidence supporting it. Candidate diagnoses to test:

- Reward weights favour a term the metric does not measure (compare reward ranking to metric ranking)
- Insufficient training steps — check whether the training curve has flattened
- The mask is wrong somewhere — `tests/test_mask.py` should catch it
- The instance distribution is too easy, so SJF is near-optimal and there is no headroom

## Reporting rules

- Never report the best seed as the headline. Report the aggregate.
- Any hyperparameter search is declared: what was searched, over what range, on which seed.
  Search on seed 0 only, and never on the evaluation instances.
- If compute forces a cut, reduce **training steps**, not the number of seeds, and state the
  constraint in the report.
