# Dynamic Job Scheduling with a Dueling Deep Q-Network

DSCD 614 Reinforcement Learning · Group 11 · Option DUEL-2

---

## 1. Introduction

A manufacturing cell or a compute cluster faces the same question many times an hour: a machine has
gone free, several jobs are waiting, one must be chosen. The choice is irreversible, its cost is
invisible until much later, and the waiting set changes continuously as new work arrives. This is
*dynamic* job scheduling: unlike the classical job-shop problem, the full workload is never known
in advance.

Industry answers with dispatching rules: First-Come-First-Served, Shortest-Job-First (optimal for
mean flow time on a single machine), and Round Robin. Each applies one greedy criterion identically
in every state, which is why they are trusted and where they can be improved: none can trade a
small loss now against a larger saving later, because none models later at all.

This project asks whether reinforcement learning can learn that trade-off. We formulate dynamic
scheduling on parallel non-identical machines as an MDP, train a Dueling DQN on it, and compare
against all three rules under a protocol fixed before any result was seen.

The aims are to specify the problem as an MDP with a fixed-dimension state, a masked discrete
action space, and a reward traceable to the evaluation metrics; to implement and train a Dueling
DQN with correct action masking; to evaluate against all three dispatching rules on held-out
instances under identical conditions with variation reported across seeds; and to diagnose rather
than conceal whatever the comparison shows.

The fourth aim did substantial work. Three faults surfaced during development, none producing an
error message, one a formulation problem rather than a coding error.

## 2. Background

### 2.1 Value-based deep reinforcement learning

Deep Q-Networks (Mnih et al., 2015) fit `Q(s, a)` to the target
`y = r + γ · max_{a'} Q_target(s', a')`, using replay to decorrelate samples and a periodically
synchronised target network to stabilise the regression.

The dueling architecture (Wang et al., 2016) modifies only the head: a shared trunk feeds a scalar
state-value `V(s)` and an advantage vector `A(s, a)`, recombined as

```
Q(s, a) = V(s) + ( A(s, a) − mean_{a'} A(s, a') )
```

Subtracting the mean resolves identifiability — `V` and `A` are otherwise determined only up to a
constant — while preserving action ordering, so the greedy policy is unchanged.

### 2.2 Why dueling suits scheduling specifically

Dueling is motivated by states where the action choice barely matters, so estimating a separate `Q`
per action wastes capacity re-deriving a shared state value. Scheduling is a clean instance.

Value here is dominated by congestion — work backed up against capacity free — which is identical
for every action in the state. The advantage of one assignment over another is often small and
sometimes exactly zero: two idle machines of equal speed and two jobs of equal processing time and
weight make several actions interchangeable. A vanilla network learns the congestion signal once
per output, across all 51. Dueling learns it once in `V(s)` and leaves the advantage stream the
differences.

Our measurements support the premise: on the trained network the Q-spread across legal actions
averaged 0.11 against episode returns near −2, so the between-action signal is roughly two orders
of magnitude smaller than the between-state signal.

### 2.3 Prior work in the domain

Han and Yang (2020) is the closest precedent: a dueling double DQN for adaptive job-shop
scheduling, including the output-layer masking mechanic we adopt. Work since has converged on the
D3QN combination, with applications to flexible job-shop scheduling with automated guided vehicles
reporting lower tardiness than composite dispatching rules. A 2025 review in the *Journal of
Intelligent Manufacturing* notes that Rainbow-style combinations converge more reliably on large
dynamic instances at higher cost. Masking under random arrivals is itself active: a 2025 study
applies masked double Q-learning to semiconductor ion-implantation scheduling.

**Where we depart.** The literature standard is D3QN. This project is bound to Dueling DQN, so the
headline configuration uses dueling alone and exposes `double_q` as a configuration flag rather
than silently adopting the stronger method. We state this explicitly because a reader familiar with
the field would otherwise wonder why the obvious enhancement is missing.

## 3. Problem formulation

### 3.1 The scheduling problem

`M` machines run in parallel. Machine `m` has speed factor `s_m`, so job `j` occupies `p_j / s_m`
time units on it. Machines are non-preemptive: a started job runs to completion. Jobs arrive as a
Poisson process with rate `λ`; an episode generates `N` jobs. Job `j` carries arrival time `a_j`,
processing time `p_j`, priority weight `w_j`, and deadline `d_j`; `C_j` denotes its realised
completion time.

### 3.2 Decision epochs

The agent is not queried on a fixed clock. A **decision epoch** occurs when at least one machine is
idle *and* at least one job is pending. Between epochs the simulator jumps to the next event — a
completion or an arrival — so episode length is proportional to the number of jobs rather than to
simulated time. Measured across the held-out set, every dispatching rule completes an episode in
exactly 50 decision epochs, one per job.

`Δt_i = t_{i+1} − t_i` denotes elapsed simulated time between consecutive epochs.

### 3.3 State space

Fixed dimension `d = 5K + 3M + 4`. With `K = 10` visible job slots and `M = 5` machines, `d = 69`.

The pending queue is sorted by `(deadline, processing time)` and the first `K` jobs occupy the
observation window.

**Job window**, `K` slots × 5 features: processing time `/ p_max`; priority weight `/ w_max`; slack
`(d_j − t_i) / H` clipped to `[−1, 1]`; waiting time `(t_i − a_j) / H` clipped to `[0, 1]`; and an
occupancy flag.

**Machine bank**, `M` machines × 3 features: time until free `/ p_max` clipped to `[0, 1]`; speed
factor `/ s_max`; utilisation so far.

**Global block**, 4 features: queue length `/ N`; simulated time `/ H`; jobs not yet completed
`/ N`; arrival rate normalised by service capacity.

**Justification.** Every feature is *relational* rather than absolute: slack and waiting time are
measured against the current clock, machine state is time-until-free rather than an absolute
release time. A policy learned at one point in an episode is therefore applicable at another, which
lets one network serve a whole episode across varying congestion. An absolute encoding would force
the network to relearn the same dispatching logic for each region of the clock.

### 3.4 Action space and masking

`Discrete(K · M + 1) = Discrete(51)`. Action `a = k·M + m` assigns the job in slot `k` to machine
`m`. Action `a = K·M` is a no-op that commits no assignment and advances the clock.

A mask `μ ∈ {0,1}^51` accompanies every observation: `μ[kM + m] = 1` iff slot `k` is occupied and
machine `m` is idle. Because a decision epoch requires both an idle machine and a pending job, at
least one assignment is always legal, so the mask is never empty.

The mask is applied in five places, and all five are required:

1. ε-greedy exploration samples uniformly from legal actions only; sampling from all 51 would spend
   most of the budget on rejected moves.
2. Greedy selection argmaxes with illegal entries driven to a large negative value at the output.
3. **The bootstrap target** restricts `max_{a'} Q_target(s', a')` to actions legal in `s'`, which
   requires storing the next-state mask in the replay buffer.
4. Specific to dueling: the mean subtracted in the aggregation must be over *legal* actions only,
   or values from unreachable outputs leak into `V(s)`.
5. Following from (4), the current-state mask must be supplied during the update, or `Q(s,a)` in
   the loss differs from what the behaviour policy evaluates. Section 6.2 reports what happened
   when it was not.

**The no-op is masked out in the headline configuration.** This is an empirical decision reported
with its evidence in Section 6.3.

### 3.5 Reward

At epoch `i`, with `Q_i` the pending set, `I_i` the idle machines and `F_i` the jobs completing in
`[t_i, t_{i+1})`:

```
r_i = − ( α · Δt_i · |Q_i| + β · Δt_i · |I_i| ) / Z
      + γ_c · Σ_{j ∈ F_i} w_j / Z
      − δ · Σ_{j ∈ F_i} w_j · max(0, C_j − d_j) / Z
```

with `Z = N · p̄`, the number of jobs times mean processing time, chosen so episode returns are of
order 1 across instance sizes. Weights are `α = 1.0`, `β = 0.3`, `γ_c = 1.0`, `δ = 2.0`.

**The first term is not shaping.** Summed over an episode, `Σ_i Δt_i · |Q_i|` is the area under the
queue-length curve, which equals the total time all jobs spend waiting — precisely the quantity
reported as `avg_waiting_time × N`. It is the evaluation objective decomposed over decision epochs.
This gives the agent a dense per-step signal that cannot drift away from the metric it is scored
on. The identity is asserted numerically in `tests/test_reward.py` on a full episode, so the claim
is verified rather than argued.

### 3.6 Termination, truncation and discount

**Termination** occurs when all `N` jobs have completed: a genuine absorbing state with no further
reachable reward. **Truncation** occurs at `T_max = 4N = 200` epochs or when simulated time exceeds
`H`; it is a harness-imposed limit, not a property of the task, so the value at a truncated state
is bootstrapped rather than zeroed. The replay buffer stores `terminated` alone for this reason.

`γ = 0.99`. Episodes run 50 decision epochs under every dispatching rule, so the effective horizon
`1/(1−γ) = 100` covers a full episode twice over, and the tardiness consequence of an early
dispatch remains visible to the value function at the moment of that dispatch. `γ = 0.95` (horizon
20) would hide the back half of an episode from the front half; `γ = 0.999` adds variance without
extending reach beyond the episode.

### 3.7 Does the Markov property hold?

**No.** Two violations, both stated plainly because both are consequences of choices we made.

*Queue truncation.* Only the `K = 10` head-of-queue jobs are observable; beyond that, pending jobs
affect future dynamics invisibly. The queue is sorted by a fixed key so the window holds the ten
most urgent rather than an arbitrary ten, and `|Q_i|` is in the global block, so the agent observes
the size of what it cannot see.

*Unobserved future arrivals.* Realised arrival times of unreleased jobs are absent. The rate `λ` is
included instead, making the process Markov *in distribution* while individual realisations remain
unpredictable.

The environment is therefore a POMDP under any finite state representation, and what we specify
captures sufficient statistics rather than realisations.

## 4. Methodology

### 4.1 Environment construction

The environment is a custom Gymnasium environment implementing the MDP of Section 3, written by the
group. It is an event-driven discrete-event simulator: state advances only to the next completion
or arrival, and the agent is queried only at decision epochs.

Instance generation is separated from agent stochasticity: each episode's job stream comes from a
dedicated RNG whose seed is independent of the one controlling network initialisation and
exploration. Training instance seeds occupy disjoint bands of 3000 below 9000; evaluation uses
9000–9029 exclusively, guarded by an in-function assertion and two tests.

`check_env` passes, run in the environment's permissive action mode because the checker samples
uniformly and cannot respect a mask. Training and evaluation always run strict, where a masked
action raises rather than being silently rewritten.

**Environment constants were fixed by measurement.** The first configuration (`λ = 0.55`) produced
a makespan of 97.95 against an arrival-bound makespan `N/λ` of 90.9: the arrival process, not the
scheduler, set the finish time, and the rules were indistinguishable. Sweeping `λ` on baseline
separation gave the final value of 1.0 (`ρ = 1.24`), where makespan is 69.06 against a bound of
50.0. Detail in Appendix A.3; `scripts/check_load.py` retains the check as a permanent gate.

### 4.2 Network architecture

Shared trunk of two 256-unit ReLU layers feeding a scalar value head and a 51-unit advantage head.
Recombination follows Section 3.4, with the mean over legal actions only and illegal entries driven
to a large finite negative value — finite rather than `−inf` keeps the loss well defined when a
batch holds states with few legal actions. Approximately 85,000 parameters; capacity was never the
binding constraint.

### 4.3 Training procedure

Standard DQN with replay and a target network, structured after the CleanRL single-file reference
and written out rather than imported so masking could be threaded everywhere it is required. Adam
at `1 × 10⁻⁴`; batch 128; replay 200,000; learning starts at 5,000; one gradient step per 4
environment steps; target sync every 1,000; gradient norm clipped at 10; Huber loss. ε decays
linearly from 1.0 to 0.05 over the first 30% of training. One million steps per seed.

The replay buffer stores seven columns rather than the usual five: observation, **current mask**,
action, reward, next observation, **next mask**, and terminated. Both masks are load-bearing, for
the reasons given in Section 3.4.

`double_q` is exposed as a configuration flag and is **off** in the headline run, so the reported
algorithm is Dueling DQN as the brief requires.

### 4.4 Baseline design

Three dispatching rules, all implemented by the group behind a common `Policy` interface:

- **First-Come-First-Served** dispatches the earliest-arriving visible job.
- **Shortest-Job-First** dispatches the visible job with the smallest processing time.
- **Round Robin** takes the head of the queue, cycling the machine cursor.

Every rule chooses a job and then takes the fastest idle machine, so the comparison isolates the
job-selection rule rather than mixing in a machine preference. Critically, **every rule sees the
same ten-slot window the agent sees**. A baseline with full queue visibility would not be competing
under the agent's constraints, and the comparison would be unfair in the baseline's favour.

Two further policies are diagnostics, not required baselines. `RandomMasked` selects uniformly
among legal actions and provides a floor: a policy that cannot beat it is broken rather than
undertrained. `NeverWait` dispatches at random but never waits, isolating the cost of waiting from
the cost of choosing badly. Both changed decisions, so both are reported.

### 4.5 Experimental protocol

The protocol was written into `docs/experimental_protocol.md` before any result existed.

Three training seeds (0, 1, 2) with hyperparameters held identical. Evaluation uses 30 held-out
instances, seeds 9000–9029, **identical for every policy and every agent seed**, making the
comparison paired and removing instance difficulty as a confound. Exploration is disabled: action
selection is a deterministic masked argmax. The agent and all baselines are executed by the same
`harness.run_policy` call through the same metric implementation; baselines are `Policy` objects,
not a separate script.

Six metrics are reported. Aggregation returns mean and standard deviation across seeds and exposes
no argument that selects a seed, so reporting a single best run is not possible through the
interface. With three seeds, differences are compared against the seed spread rather than
significance-tested; three samples do not support a test and we do not claim one.

## 5. Results

Three seeds, one million steps each (~23 min/seed on CPU), evaluated on 30 held-out instances with
exploration disabled. Every figure regenerates from committed logs.

### 5.1 Training

Figure 1 shows episode return against steps, meaned across seeds with a band at one standard
deviation. Return rises from −2.58 at 100,000 steps to −1.92 at 700,000 and is flat thereafter
(−1.93, −1.94, −1.92 over the final three deciles), so the budget was sufficient. Spread across
seeds stays at or below 0.062, indicating stability under reseeding. Training returns include
ε = 0.05 and are not comparable to the evaluation numbers.

### 5.2 Evaluation against the baselines

Means over 30 held-out instances; the agent row is meaned across three seeds with the standard
deviation across them. The dispatching rules are deterministic on a fixed instance set, so their
spread is exactly zero — a property of the design, not a missing measurement.

| Policy | Makespan | Avg. waiting | Utilisation | Missed | Weighted tardiness | Return |
|---|---|---|---|---|---|---|
| Random (diagnostic) | 69.060 | 5.594 | 0.899 | 0.399 | 258.362 | −2.414 |
| First-Come-First-Served | 69.059 | 5.539 | 0.895 | 0.465 | 212.286 | −2.099 |
| Round Robin | 69.914 | 4.589 | 0.886 | 0.361 | 130.486 | −1.398 |
| Shortest-Job-First | 69.652 | 4.077 | 0.887 | 0.256 | 136.361 | −1.352 |
| **Dueling DQN** | 69.550 ± 0.085 | 4.704 ± 0.034 | 0.891 ± 0.001 | 0.397 ± 0.014 | 149.619 ± 2.188 | −1.543 ± 0.015 |

**The agent beats First-Come-First-Served on every metric, and the differences exceed the seed
spread.** Waiting time falls from 5.539 to 4.704, missed deadlines from 0.465 to 0.397, weighted
tardiness from 212.3 to 149.6. It also beats the random legal policy on every metric, establishing
that it learned something rather than defaulting to chance.

**It loses to Shortest-Job-First and Round Robin, and those differences also exceed the seed
spread** — worse than Shortest-Job-First by 0.627 in waiting time and 0.141 in missed-deadline
fraction; against Round Robin the gap narrows to 0.115 and 19.1 in weighted tardiness.

Utilisation (0.886–0.899) and makespan are effectively identical across all five policies: both are
dominated by the arrival process at this load and discriminate poorly between schedulers. We report
them because the brief requires them.

### 5.3 The result that constrains the diagnosis

Shortest-Job-First achieves a cumulative reward of −1.352 against the agent's −1.543. **A
hand-written dispatching rule scores better on the agent's own objective than the agent does.**

This separates two explanations that would otherwise be confounded. Were the reward misaligned with
the metrics, the policy winning on metrics would score poorly on reward. It does not: both rankings
agree and Shortest-Job-First tops each. The reward is therefore not the problem. The agent fails to
reach a return demonstrably reachable within its own objective, locating the shortfall in
optimisation and representation. Section 6.4 takes that up.

## 6. Discussion

### 6.1 The no-op taught the agent to stall

The most consequential finding was a formulation error, not a coding one.

The action space originally included a no-op, letting the agent decline to dispatch so the clock
could advance — intended to permit holding a fast machine for an urgent job about to arrive. At
300,000 steps on seed 0 the resulting policy reached an average waiting time of 7.95, **worse than
selecting uniformly at random among legal actions**, which scores 6.61.

That comparison made the diagnosis possible. A policy that loses to random is broken rather than
undertrained, and the distinction decides whether the response is more compute or an investigation.
Instrumenting the policy showed it choosing the no-op at 46.3% of decision epochs against 0.7% for
an untrained network of the same architecture.

The mechanism follows from the reward. Dispatching at an epoch that remains a decision epoch
advances the clock by `Δt = 0` and returns reward 0, while every action that advances time returns
a negative reward. Under discounting, an agent facing negative rewards improves its discounted
return by deferring them, and the no-op makes deferral available. With a value function barely
discriminating between actions, that bias dominated.

Masking the no-op removed the deferral action: waiting time fell from 7.95 to 4.74 and weighted
tardiness from 348.4 to 154.8 under the same budget. The catalogue specifies the action as
*"assign a selected job to a selected machine"*, so a dispatch-only space is faithful to the brief.
Full figures in `docs/mdp_spec.md` §4.

### 6.2 Two silent implementation faults

Neither produced an error message; both are in Appendix A with the tests now covering them.

The first placed training instance seeds for seeds 1 and 2 inside the held-out evaluation range.
Two of three seeds would have been evaluated on instances they trained on, inflating every headline
number.

The second supplied an all-ones current-state mask inside the loss. Because the dueling aggregation
subtracts the mean advantage over legal actions, `Q(s,a)` during the update was a different function
from the one the behaviour policy evaluated. The symptom was an agent that degraded with training:
average waiting time rose from 7.30 to 17.28 across a 20,000-step run.

### 6.3 On exploration and convergence

Exploration is uniform over legal actions, a meaningful constraint here: typically fewer than
fifteen of the fifty-one actions are legal, so unmasked ε-greedy would waste most of its budget.
The ε schedule was not tuned; no hyperparameter search was conducted and none is claimed.

Return plateaued from 700,000 steps onward and spread across seeds stayed under 0.07 throughout,
so the run is stable and the budget was adequate. Nothing here is limited by compute.

### 6.4 Why the agent stops short of Shortest-Job-First

Section 5.3 rules out reward misalignment. Two candidate explanations remain, and the evidence
distinguishes them.

Capacity is not supported: training plateaus with seed spread under 0.07, the signature of a
converged optimisation rather than an under-parameterised one.

The **action representation** fits the evidence. Shortest-Job-First implements a comparison —
select the queued job minimising processing time. The agent must learn that from a flat
69-dimensional vector where each slot sits at a fixed, arbitrary offset. Nothing in a multilayer
perceptron over concatenated slots makes "compare feature 1 across the ten blocks and take the
smallest" natural; the comparison must be rediscovered for every pairing of positions. The measured
Q-spread of 0.11 against returns near −2 is consistent with a network that learned the state value
well and the between-action ordering poorly.

That predicts the fix: a permutation-invariant encoder over the queue — shared per-slot embedding
with pooling, or attention over slots — makes cross-slot comparison native rather than positional.

## 7. Limitations and deployment considerations

**The state is partially observed.** Only the ten most urgent queued jobs are visible, and arrival
times of unreleased jobs are absent. A recurrent encoder, or stacking a short observation history,
is the natural response and was not attempted.

**Masking the no-op forecloses a real capability.** An operator sometimes should hold a fast
machine for an urgent job arriving imminently. The headline agent cannot express that. The
underlying cause is a reward whose per-step signal is almost always negative, making deferral
attractive under discounting; a potential-based shaping term, or a positive completion baseline
that makes progress intrinsically rewarding, would likely allow the no-op to be retained. Testing
that was out of scope for the time available.

**The instance distribution is synthetic and single-operation.** Real job shops have
multi-operation jobs with precedence constraints, sequence-dependent setups, breakdowns, and
non-stationary arrivals. Nothing here demonstrates transfer, and the load parameters were chosen
partly to make the comparison informative — a legitimate experimental choice, not a claim about any
real workload.

**Three seeds is a small sample.** Differences are compared against the seed-to-seed spread. No
significance test is performed and none would be supported.

**For deployment**, three properties matter more than the headline metric. Inference is one forward
pass through an 85,000-parameter network, fast enough for a dispatch loop. The mask is enforced by
the environment, so an out-of-distribution observation cannot produce an illegal schedule — the
failure mode is a poor legal choice, not an invalid one. But the policy inherits its training
arrival distribution; a load shift would require retraining, so a deployed system needs the
`check_load.py` monitor running continuously with the dispatching rules retained as fallback.

## 8. Conclusion and further work

We formulated dynamic job scheduling on heterogeneous parallel machines as a masked-action MDP and
trained a Dueling DQN on it, comparing against three dispatching rules on held-out instances under
a protocol fixed in advance.

The most useful result was methodological. Three of the four substantive problems — an
instance-seed overlap with the held-out set, a mask omitted from the loss, and a reward structure
that rewarded stalling — produced no error message and would have survived into the report unmeasured.
A uniform-random legal policy and a load-separation check, neither required by the brief, surfaced them.

Further work, in order of expected value: replace the fixed ten-slot window with a
permutation-invariant encoder over the queue; retain the no-op under potential-based shaping;
evaluate the `double_q` flag as the D3QN combination the literature favours; and test transfer
across untrained arrival rates.

---

## Appendix A — Fault forensics

### A.1 Training instances overlapped the held-out range

`training_instance_seed` used `1000 + seed × 100000` as a band start, placing seeds 1 and 2 at
101,000 and 201,000, both above the held-out range beginning at 9000; the intended wraparound never
triggered. Instances are now partitioned into disjoint bands of 3000 below `EVAL_SEED_START`, with
an in-function assertion. Covered by `test_training_instances_never_enter_the_held_out_range` and
`test_each_training_seed_gets_its_own_instances`.

### A.2 The loss used an all-ones current-state mask

`compute_loss` passed `torch.ones_like(next_mask)` as the current-state mask. Because the dueling
aggregation subtracts the mean advantage over legal actions, this optimised a different function
from the one `select_action` evaluated. The replay buffer now stores the current mask. Covered by
`test_training_uses_the_current_state_mask_for_predicted_q`.

### A.3 The environment had no headroom

At `λ = 0.55` the makespan of 97.95 was within 8% of the arrival-bound makespan of 90.9, and
First-Come-First-Served and Shortest-Job-First differed by 0.06 in average waiting time. The load
sweep and the resulting choice of `λ = 1.0` are in `docs/mdp_spec.md` §10; the check is retained as
`scripts/check_load.py`.

## Appendix B — Reproduction

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
bash scripts/run_all.sh
```

Seeds, exact dependency versions and the full hyperparameter table are in
`docs/hyperparameters.md`. Raw logs backing every figure are committed under `logs/`.
