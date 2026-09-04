# MDP Specification — DUEL-2 Dynamic Job Scheduling

This document is frozen before training begins, so that the formulation is fixed independently of
any result it later explains.

Notation is used consistently across this file, the code and the report.

---

## 1. The scheduling problem

`M` machines run in parallel. Jobs arrive over time and are not all known at the start — this is
what makes the problem *dynamic* rather than a static job-shop instance. Each job `j` carries:

| Symbol | Meaning |
|---|---|
| `a_j` | arrival time |
| `p_j` | processing time (machine-independent base duration) |
| `w_j` | priority weight |
| `d_j` | deadline |
| `C_j` | completion time (realised) |

Machine `m` has a speed factor `s_m`, so job `j` on machine `m` occupies `p_j / s_m` time units.
A machine processes one job at a time and is non-preemptive: once started, a job runs to
completion.

Arrivals follow a Poisson process with rate `λ`. An episode generates `N` jobs in total.

## 2. Decision epochs

The agent is not queried at every simulated time unit. A **decision epoch** occurs when at least
one machine is idle and at least one job is pending. Between epochs the simulator advances to the
next event (a job completion or a job arrival). This keeps the episode length proportional to the
number of jobs rather than to simulated clock time.

`Δt_i = t_{i+1} − t_i` denotes the simulated time elapsed between epoch `i` and epoch `i+1`.

## 3. State space

Fixed dimension `d = 5K + 3M + 4`. With `K = 10` visible job slots and `M = 5` machines,
`d = 50 + 15 + 4 = 69`.

The pending queue is sorted by a fixed key (earliest deadline, ties broken by shortest processing
time) and the first `K` jobs occupy the observation window.

**Job window** — `K` slots × 5 features:

| Feature | Definition | Normalisation |
|---|---|---|
| processing time | `p_j` | `/ p_max` |
| priority weight | `w_j` | `/ w_max` |
| slack | `d_j − t_i` | `/ H`, clipped to `[-1, 1]` |
| waiting time | `t_i − a_j` | `/ H`, clipped to `[0, 1]` |
| occupancy flag | 1 if slot holds a job, else 0 | — |

**Machine bank** — `M` machines × 3 features:

| Feature | Definition | Normalisation |
|---|---|---|
| time until free | `max(0, f_m − t_i)` | `/ p_max` |
| speed factor | `s_m` | `/ s_max` |
| utilisation so far | busy time / `t_i` | already in `[0, 1]` |

**Global block** — 4 features: queue length `/ N`, simulated time `/ H`, jobs not yet
completed `/ N`, arrival rate `λ` normalised by service capacity.

`H` is the horizon constant used for all time normalisations; fix it once in
`configs/env_default.yaml` and record the value in the report.

> **Justification to write up.** The observation is deliberately *relational* rather than
> absolute: slack and waiting time are measured against the current clock, and machine state is
> expressed as time-until-free rather than an absolute release time. A policy learned at one point
> in the episode is therefore reusable at another, which is what allows a single network to serve
> a whole episode of varying congestion.

## 4. Action space

`Discrete(K · M + 1)`.

- Action `a = k · M + m` for `k ∈ [0, K)`, `m ∈ [0, M)` — assign the job in slot `k` to machine `m`.
- Action `a = K · M` — **no-op**: commit no assignment and advance the simulator to the next event.

With `K = 10, M = 5` the space has 51 actions.

**Action masking.** A mask `μ ∈ {0,1}^{KM+1}` is emitted with every observation.
`μ[kM + m] = 1` iff slot `k` is occupied and machine `m` is idle.

The no-op is valid only when a future event exists — a running job that will complete, or a job
that has not yet arrived. Without that condition the agent could select the no-op forever in a
state where nothing else can happen, and simulated time would never advance. Because a decision
epoch requires an idle machine and a pending job, at least one assignment is always available, so
restricting the no-op can never leave the action set empty.

### The no-op is masked out in the headline configuration

`allow_noop: false` in `configs/env_default.yaml`. The action space stays `Discrete(K·M + 1)` so
that the specification, the saved networks and the ablation configuration remain compatible, but
the final entry is permanently masked.

This is an empirical decision, not a preference. Trained under otherwise identical settings for
300,000 steps on seed 0 and evaluated on the 30 held-out instances:

| Configuration | Avg. waiting | Makespan | Missed | Weighted tardiness | Return |
|---|---|---|---|---|---|
| Agent, no-op available | 7.95 | 75.58 | 0.589 | 348.4 | −3.440 |
| Agent, no-op masked | 4.74 | 69.16 | 0.401 | 154.8 | −1.582 |
| Uniform-random legal policy | 6.61 | 71.82 | 0.470 | 301.7 | −2.885 |

With the no-op available the agent is **worse than choosing uniformly at random among legal
moves**, which is the diagnostic that separates a broken agent from an undertrained one. Instrumenting
the learned policy showed why: it selected the no-op on **46.3%** of decision epochs, against
**0.7%** for an untrained network of the same architecture. Training was actively teaching the
agent to stall.

The mechanism is that dispatching at a decision epoch that remains a decision epoch advances the
clock by `Δt = 0` and therefore returns reward 0, while every path that advances time returns a
negative reward. Under a discount, an agent facing a stream of negative rewards can reduce its
discounted return by deferring them. The no-op is the action that makes deferral available, and
with `γ = 0.99` and a value function that barely discriminated between actions — the measured
spread of Q across legal actions was 0.11 against episode returns near −3 — the bias was enough to
dominate the policy.

Masking the no-op removes the deferral action. The catalogue specifies the action as *"assign a
selected job to a selected machine"*, so a dispatch-only action space is faithful to the brief;
the no-op was our addition and it is reported here with the evidence that removed it.

**For Limitations:** the no-op is the mechanism by which an agent could hold a fast machine idle
for an urgent job about to arrive. Masking it forecloses that behaviour. A reward formulation with
a potential-based shaping term, or a positive per-completion baseline that makes progress
intrinsically rewarding, would likely let the no-op be retained. We did not have time to test it.

The mask is applied in three places and all three are required for correctness:

1. ε-greedy exploration samples uniformly from valid actions only.
2. Greedy action selection takes `argmax` over `Q(s,·)` with invalid entries set to `−∞`.
3. **The bootstrap target** uses `max_{a'} Q_target(s', a')` restricted to `μ'`. Omitting the mask
   here lets the target back up the value of an action the agent can never take, and is the most
   common silent bug in masked value-based RL. The next-state mask must therefore be stored in the
   replay buffer alongside the next observation.

### 4.6 Two action formulations, and why the rule formulation is the headline

The catalogue lists *"assign a selected job to a selected machine"* as a **candidate** action, and
the brief states that "a formulation that departs from them with good reason is credited". Both
formulations are implemented, both are evaluated, and the departure is evidence-based.

**Formulation A — direct assignment.** `Discrete(K·M + 1) = Discrete(51)`, as specified in §4.
The agent names the job slot and the machine.

**Formulation B — dispatching-rule selection (headline).** `Discrete(8)`. The agent selects one of
eight priority rules; the rule then selects the job, and the machine is always the fastest idle
machine so the choice isolates job selection.

| # | Rule | Priority |
|---|---|---|
| 0 | SPT | shortest processing time |
| 1 | LPT | longest processing time |
| 2 | EDD | earliest due date |
| 3 | FCFS | earliest arrival |
| 4 | WSPT | weighted shortest processing time, `w_j / p_j` |
| 5 | MS | minimum slack, `d_j − t − p_j` |
| 6 | CR | critical ratio, `(d_j − t) / p_j` |
| 7 | ATC | apparent tardiness cost, `(w_j/p_j)·exp(−max(0, d_j − t − p_j)/(k·p̄))`, `k = 2` |

**Why B.** Han and Yang (2020, *IEEE Access* 8:186474–186495) address adaptive job-shop scheduling
with a dueling double DQN and state that "various heuristic rules are used as available actions",
reporting that the result "performs better than any single heuristic rule" on 85 OR-Library
instances. Their design is the closest published precedent to this project, and the rule action
space is the part of it we can adopt without departing from the bound algorithm. The wider review
literature reports the same ordering: rule selection over direct operation choice. See
`docs/report/literature_review.md`, where every citation is verified against Crossref, OpenAlex or
arXiv.

**The mechanism, in our own terms.** SPT implements a comparison — select the queued job minimising
processing time. Under Formulation A the network must learn that comparison from a flat
69-dimensional vector in which each slot occupies a fixed, arbitrary offset, and must rediscover it
for every pairing of positions. Under Formulation B the rule performs the comparison and the network
learns only *when* each rule is appropriate. Our measured Q-spread across legal actions of 0.11,
against episode returns near −2, is the direct evidence that Formulation A was failing at exactly
this.

**Equivalence.** The two formulations must produce comparable schedules or the comparison is not
like-for-like. `FixedRule(SPT)` in Formulation B reproduces the Shortest-Job-First baseline in
Formulation A to within 1e-9 on every metric, and `FixedRule(FCFS)` reproduces First-Come-First-Served.
Asserted in `tests/test_rules.py`.

**The bar this raises, stated plainly.** Two rules inside the action set already beat all three
required baselines on the held-out set: ATC (return −1.125, weighted tardiness 95.1) and WSPT
(−1.187, 104.9), against Shortest-Job-First at −1.352. Because ATC is reachable by a policy that
always selects it, **beating Shortest-Job-First is not evidence of learning under Formulation B.**
The meaningful bar is the best single rule, which is also the bar Han and Yang set. Results are
reported against it.

## 5. Reward function

At decision epoch `i`, let `Q_i` be the set of pending jobs, `I_i` the set of idle machines and
`F_i` the set of jobs that complete during `[t_i, t_{i+1})`.

```
r_i  =  − (α · Δt_i · |Q_i|  +  β · Δt_i · |I_i|) / Z
        + γ_c · Σ_{j ∈ F_i} w_j / Z
        − δ · Σ_{j ∈ F_i} w_j · max(0, C_j − d_j) / Z
```

with normaliser `Z = N · p̄` where `p̄` is the mean processing time, chosen so that episode
returns are O(1) across instance sizes.

| Term | Weight | What it penalises or rewards |
|---|---|---|
| `Δt_i · |Q_i|` | `α` | queue-waiting cost |
| `Δt_i · |I_i|` | `β` | machine idleness |
| `Σ w_j` | `γ_c` | weighted job completion |
| `Σ w_j · max(0, C_j − d_j)` | `δ` | weighted tardiness |

**The property worth stating in the report:** `Σ_i Δt_i · |Q_i|` telescopes to the total waiting
time accumulated over all jobs in the episode. The first term is therefore not a heuristic shaping
signal — it is the true objective decomposed over decision epochs, which is why the agent receives
dense feedback without the reward becoming inconsistent with the metric it is scored on.

Starting weights: `α = 1.0`, `β = 0.3`, `γ_c = 1.0`, `δ = 2.0`. Every change and its reason are
recorded in `docs/hyperparameters.md`. An ablation over `β` and `δ` would isolate the contribution
of the idleness and tardiness terms.

## 6. Termination and truncation

Stated separately, as the instructions require.

- **Termination** (`terminated = True`): all `N` jobs generated for the episode have completed.
  This is a genuine absorbing state — no further reward is reachable.
- **Truncation** (`truncated = True`): the epoch counter reaches `T_max = 4N`, or simulated time
  exceeds `H`. This is a time limit imposed by the harness, not a property of the task, and the
  value at the truncation state is bootstrapped rather than treated as terminal.

## 7. Discount factor

`γ = 0.99`.

Measured over the 30 held-out instances, an episode runs **50 decision epochs** under every
baseline — one per job, with no no-ops taken, since a dispatch rule always dispatches when it can.
An agent that uses the no-op will run longer, bounded by `T_max = 200`.

The effective horizon `1/(1−γ) = 100` epochs therefore covers a full episode twice over: the agent
can see the tardiness consequence of an early dispatch decision at the end of the run. A shorter
`γ = 0.95` (horizon 20) would truncate that credit assignment, and with 50-epoch episodes it would
hide the back half of the run from the front half. `γ = 0.999` adds variance without extending
reach beyond the episode.

## 8. Does the Markov property hold?

**No — and the report must say so plainly and say what compensates.**

Two sources of violation:

1. **Queue truncation.** Only the `K` head-of-queue jobs are observable. When `|Q_i| > K` the
   remaining jobs affect future dynamics but are invisible.
   *Compensation:* the queue is sorted by a fixed deterministic key, so the window always holds the
   `K` most urgent jobs rather than an arbitrary `K`; and `|Q_i|` itself is in the global block, so
   the agent observes the size of what it cannot see.

2. **Unobserved future arrivals.** The realised arrival times of jobs not yet released are not in
   the state.
   *Compensation:* the arrival rate `λ` is included in the global block. This makes the process
   Markov in *distribution* — the conditional distribution of future arrivals given the state is
   fixed — even though the individual realisations are not predictable. The task is more precisely
   a Markov decision process over the observed sufficient statistics of a partially observed
   arrival process.

The second point means the environment is a POMDP under any finite state representation. The chosen
representation captures the sufficient statistics rather than the realisation, and a recurrent or
history-stacked encoder is the natural extension.

## 9. Why Dueling DQN suits this problem

The argument to make in Background, tied to this specific task:

In scheduling states, the value of the state is dominated by system congestion — how much work is
backed up and how much capacity is free — while the *advantage* of one assignment over another is
often small and sometimes zero. When two machines are equally idle and two queued jobs have equal
processing times, every action is equivalent, but the state itself may be worth a great deal or
very little depending on backlog.

Vanilla DQN must learn `Q(s,a)` for all 51 actions independently and, in states where the action
barely matters, spends capacity re-learning the same state value 51 times. The dueling
decomposition

```
Q(s,a) = V(s) + ( A(s,a) − (1/|A_valid|) Σ_{a' ∈ A_valid} A(s,a') )
```

learns congestion once in `V(s)` and lets the advantage stream model only the differences.

**Implementation note that matters here:** the mean subtracted in the aggregation must be taken
over *valid* actions only. Averaging over all `KM+1` entries, including the masked ones, leaks
arbitrary values from unreachable actions into `V(s)`. Test this in `tests/test_mask.py`.

## 10. Resolved parameters

Frozen on 27 August after a load sweep. `scripts/check_load.py` re-runs the checks that
justified them; run it after changing any value here.

| Parameter | Value | Basis |
|---|---|---|
| machines `M` | 5, speeds `[1.0, 1.0, 1.25, 0.8, 0.8]` | heterogeneous, so *which* machine is a real choice and Round Robin has a way to lose |
| jobs `N` | 50 | gives 60-90 decision epochs, which the discount in §7 is sized against |
| queue window `K` | 10 | observation 69, actions 51 |
| arrival rate `λ` | 1.0 | see below |
| processing time `p_j` | discrete uniform [2, 10] | mean 6, so `p_max = 10` |
| weight `w_j` | `{1: 0.6, 2: 0.3, 5: 0.1}` | mostly routine work, occasional urgent job, so the weighted terms carry signal |
| deadline `d_j` | `a_j + p_j · U(1.3, 2.5)` | proportional to the job's own size, so long jobs are not penalised by construction |
| horizon `H` | 120 | above the observed makespan of ~69 |
| `T_max` | `4N = 200` | truncation limit; episodes terminate well inside it |

**Why `λ = 1.0`.** Service capacity is `Σ s_m / p̄ = 4.85 / 6 = 0.808` jobs per time unit, so
`λ = 1.0` deliberately over-subscribes the bank at `ρ = 1.24`. A backlog builds during the
arrival phase and then drains, and that transient overload is what makes dispatch order
determine the outcome.

The first attempt used `λ = 0.55` (`ρ = 0.68`) and failed the check: makespan came out at 98
against an arrival bound `N/λ` of 91, meaning the arrival process was setting the finish time and
the scheduler was close to irrelevant. Average waiting time was 0.5 time units and the three
baselines were within noise of each other.

At `λ = 1.0`, makespan is 69 against an arrival bound of 50, and the baselines separate over 30
held-out instances:

| Policy | Makespan | Avg. waiting | Utilisation | Missed deadlines | Weighted tardiness |
|---|---|---|---|---|---|
| FCFS | 69.06 | 5.54 | 0.895 | 0.465 | 212.29 |
| SJF | 69.65 | 4.08 | 0.887 | 0.256 | 136.36 |
| Round Robin | 69.91 | 4.59 | 0.886 | 0.361 | 130.49 |

**No baseline dominates.** SJF has the lowest miss rate; Round Robin has the lowest weighted
tardiness. The agent therefore has a genuine trade-off to find rather than a single rule to copy,
and the report can say which corner of that frontier the learned policy lands in. Above `λ = 1.5`
the system saturates past a 0.7 miss rate and that structure flattens out.

## 11. Still open

- [ ] Total training timesteps — set from a smoke run, not in advance
- [ ] Whether double-Q sits on top of dueling (they are independent; state which is used)
- [ ] Whether to add a public job-shop benchmark instance as a secondary evaluation
