# MDP Specification — DUEL-2 Dynamic Job Scheduling

Freeze this document before training begins. Section 4 of the examination instructions requires
the formulation to precede implementation, and 16 of the 100 marks are awarded here.

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
`μ[kM + m] = 1` iff slot `k` is occupied and machine `m` is idle. The no-op is always valid.
The mask is applied in three places and all three are required for correctness:

1. ε-greedy exploration samples uniformly from valid actions only.
2. Greedy action selection takes `argmax` over `Q(s,·)` with invalid entries set to `−∞`.
3. **The bootstrap target** uses `max_{a'} Q_target(s', a')` restricted to `μ'`. Omitting the mask
   here lets the target back up the value of an action the agent can never take, and is the most
   common silent bug in masked value-based RL. The next-state mask must therefore be stored in the
   replay buffer alongside the next observation.

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

Starting weights: `α = 1.0`, `β = 0.3`, `γ_c = 1.0`, `δ = 2.0`. Record every change and the reason
in `docs/hyperparameters.md`. An ablation over `β` and `δ` is the cheapest source of Discussion
material (10 marks) if time allows.

## 6. Termination and truncation

Stated separately, as the instructions require.

- **Termination** (`terminated = True`): all `N` jobs generated for the episode have completed.
  This is a genuine absorbing state — no further reward is reachable.
- **Truncation** (`truncated = True`): the epoch counter reaches `T_max = 4N`, or simulated time
  exceeds `H`. This is a time limit imposed by the harness, not a property of the task, and the
  value at the truncation state is bootstrapped rather than treated as terminal.

## 7. Discount factor

`γ = 0.99`.

An episode with `N = 50` jobs and `M = 5` machines runs roughly 60–90 decision epochs including
no-ops. The effective horizon `1/(1−γ) = 100` epochs therefore covers a full episode: the agent
can see the tardiness consequence of an early dispatch decision at the end of the run. A shorter
`γ = 0.95` (horizon 20) would truncate that credit assignment; `γ = 0.999` adds variance without
extending reach beyond the episode. Measure the actual mean episode length in epochs before
committing this number, and quote the measurement in the report.

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

State honestly that the second point means the environment is a POMDP under any finite state
representation, that the chosen representation captures the sufficient statistics rather than the
realisation, and that a recurrent or history-stacked encoder is the natural extension. This is
Limitations material, and examiners reward the diagnosis.

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

## 10. Open decisions

Resolve these before the environment is frozen, and record the resolution here.

- [ ] `M`, `N`, `K`, `λ` for the headline configuration
- [ ] Are machines identical (`s_m = 1` for all `m`) or heterogeneous? Heterogeneous makes the
      assignment decision genuinely non-trivial and strengthens the case against Round Robin
- [ ] Deadline generation rule — a tightness factor over `p_j` is the usual choice
- [ ] `H` and `p_max`, `w_max` normalisation constants
- [ ] Whether to include a public job-shop benchmark instance as a secondary evaluation
