# Dynamic Job Scheduling with a Dueling Deep Q-Network

DSCD 614 Reinforcement Learning · Group 11 · Option DUEL-2

---

## 1. Introduction

A manufacturing cell or a compute cluster faces the same question many times an hour: a machine
has just gone free, several jobs are waiting, and one of them has to be chosen. The choice is
irreversible, its cost is not visible until much later, and the set of waiting jobs changes
continuously because new work keeps arriving. This is *dynamic* job scheduling, and it differs
from the classical job-shop problem in that the full workload is never known in advance.

Industry answers the question with dispatching rules. First-Come-First-Served is fair and
predictable. Shortest-Job-First is provably optimal for mean flow time on a single machine and is
strong in practice. Round Robin spreads load evenly. Each rule is a single greedy criterion
applied identically in every state, which is both why they are trusted and why they can be
improved on: none of them can trade a small loss now against a larger saving later, because none
of them models later at all.

This project asks whether a reinforcement learning agent can learn that trade-off. We formulate
dynamic scheduling on parallel non-identical machines as a Markov decision process, train a
Dueling Deep Q-Network on it, and compare the learned policy against First-Come-First-Served,
Shortest-Job-First and Round Robin under a protocol fixed before any result was seen.

The aims are:

1. To specify the problem as an MDP with a fixed-dimension state, a masked discrete action space,
   and a reward whose components are traceable to the metrics used for evaluation.
2. To implement and train a Dueling DQN agent with correct action masking.
3. To evaluate the agent against all three dispatching rules on held-out instances under identical
   conditions, reporting variation across seeds for every metric.
4. To diagnose, rather than conceal, whatever the comparison shows.

The fourth aim did substantial work. Three separate faults were found during development, none of
which produced an error message, and one of which was a formulation problem rather than a coding
error. They are reported in Section 6 because the process of finding them is the most transferable
result the project produced.

## 2. Background

### 2.1 Value-based deep reinforcement learning

Deep Q-Networks (Mnih et al., 2015) approximate the action-value function `Q(s, a)` with a neural
network trained on the temporal-difference target `y = r + γ · max_{a'} Q_target(s', a')`, using an
experience replay buffer to decorrelate consecutive samples and a periodically synchronised target
network to stabilise the regression.

The dueling architecture (Wang et al., 2016) modifies only the head. A shared trunk feeds two
streams: a scalar state-value `V(s)` and a vector of advantages `A(s, a)`, recombined as

```
Q(s, a) = V(s) + ( A(s, a) − mean_{a'} A(s, a') )
```

Subtracting the mean resolves the identifiability problem — `V` and `A` are otherwise determined
only up to a constant — while preserving the ordering of actions, so the greedy policy is
unchanged.

### 2.2 Why dueling suits scheduling specifically

The published motivation for dueling is that in many states the choice of action barely matters,
and estimating a separate `Q` for every action wastes capacity re-deriving a shared state value.
Scheduling is an unusually clean instance of that condition.

In a scheduling state, value is dominated by congestion: how much work is backed up against how
much capacity is free. That quantity is identical for every action available in the state. The
advantage of one assignment over another is frequently small and sometimes exactly zero — when two
idle machines have the same speed and two queued jobs have the same processing time and weight,
several actions are genuinely interchangeable. A vanilla network must learn the congestion signal
once per output, across all 51 of them. The dueling decomposition learns it once in `V(s)` and
leaves the advantage stream to model only the differences.

Our own measurements support the premise: on the trained network, the spread of Q-values across
legal actions within a state averaged 0.11 against episode returns near −2, so the between-action
signal is roughly two orders of magnitude smaller than the between-state signal.

### 2.3 Prior work in the domain

Han and Yang (2020) is the closest precedent: a dueling double DQN for adaptive job-shop
scheduling, including the output-layer action-masking mechanic we adopt. Subsequent work has
largely converged on the D3QN combination — dueling plus double Q-learning — with recent
applications to flexible job-shop scheduling with automated guided vehicles reporting lower
tardiness than composite dispatching rules. A 2025 review in the *Journal of Intelligent
Manufacturing* surveys the field and notes that Rainbow-style combinations converge more reliably
on large dynamic instances at higher computational cost.

Action masking under random arrivals is itself an active topic. A 2025 study applies double
Q-learning with invalid action masking to semiconductor ion-implantation scheduling, and recent
work addresses masking for dynamic job-shop scheduling with random arrivals and machine failures.

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

**Justification.** Every feature is *relational* rather than absolute. Slack and waiting time are
measured against the current clock; machine state is time-until-free rather than an absolute
release time. A policy learned at one point in an episode is therefore applicable at another, which
is what allows a single network to serve a whole episode across widely varying congestion. An
absolute encoding would force the network to learn the same dispatching logic separately for each
region of the clock.

### 3.4 Action space and masking

`Discrete(K · M + 1) = Discrete(51)`. Action `a = k·M + m` assigns the job in slot `k` to machine
`m`. Action `a = K·M` is a no-op that commits no assignment and advances the clock.

A mask `μ ∈ {0,1}^51` accompanies every observation: `μ[kM + m] = 1` iff slot `k` is occupied and
machine `m` is idle. Because a decision epoch requires both an idle machine and a pending job, at
least one assignment is always legal, so the mask is never empty.

The mask is applied in three places, and all three are required:

1. ε-greedy exploration samples uniformly from legal actions only. Sampling from all 51 would spend
   most of the exploration budget on rejected moves.
2. Greedy selection takes the argmax with illegal entries driven to a large negative value at the
   output layer.
3. **The bootstrap target** restricts `max_{a'} Q_target(s', a')` to actions legal in `s'`. This
   requires storing the next-state mask in the replay buffer.

A fourth requirement is specific to the dueling architecture and is easy to miss: the mean
subtracted in the aggregation must be taken over *legal* actions only, otherwise arbitrary values
from unreachable outputs leak into `V(s)`. A fifth follows from it — the current-state mask must
also be supplied during the update, or `Q(s,a)` in the loss is a different quantity from the one
the behaviour policy evaluates. Section 6 reports what happened when it was not.

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

*Queue truncation.* Only the `K = 10` head-of-queue jobs are observable. When more than ten jobs
are pending, the remainder affect future dynamics but are invisible. The queue is sorted by a fixed
deterministic key so the window always holds the ten most urgent jobs rather than an arbitrary ten,
and `|Q_i|` appears in the global block so the agent at least observes the size of what it cannot
see.

*Unobserved future arrivals.* The realised arrival times of unreleased jobs are not in the state.
The arrival rate `λ` is included instead, which makes the process Markov *in distribution* — the
conditional law of future arrivals given the state is fixed — while individual realisations remain
unpredictable.

The environment is therefore a partially observed MDP under any finite state representation, and
what we have specified captures sufficient statistics rather than the realisation. A recurrent or
history-stacked encoder is the natural extension.

## 4. Methodology

### 4.1 Environment construction

The environment is a custom Gymnasium environment implementing the MDP of Section 3, written by the
group. It is an event-driven discrete-event simulator: state advances only to the next completion
or arrival, and the agent is queried only at decision epochs.

Instance generation is deliberately separated from agent stochasticity. Each episode's job stream
is drawn from a dedicated RNG seeded by an instance seed independent of the seed controlling
network initialisation and exploration. Training instance seeds are partitioned into disjoint bands
of 3000 below 9000; evaluation uses 9000–9029 exclusively. An assertion inside the seed function
fails loudly on overlap, and two tests cover it.

`gymnasium.utils.env_checker.check_env` passes. It is run with the environment's permissive action
mode, because the checker samples uniformly from the full action space and cannot respect a mask;
training and evaluation always run in strict mode, where a masked action raises rather than being
silently rewritten.

**Environment constants were fixed by measurement, not assumption.** The first configuration used
`λ = 0.55`, giving utilisation 0.68. Under it, makespan was 97.95 against an arrival-bound makespan
`N/λ` of 90.9 — the arrival process, not the scheduler, was setting the finish time. Average
waiting time was 0.58 for First-Come-First-Served against 0.52 for Shortest-Job-First, a difference
too small for three seeds to resolve. No agent could have demonstrated anything on that
distribution. Sweeping `λ` and measuring baseline separation produced the final value of 1.0
(`ρ = 1.24`), where makespan is 69.06 against an arrival bound of 50.0 and the rules separate
clearly. `scripts/check_load.py` retains this as a permanent gate.

### 4.2 Network architecture

Shared trunk of two 256-unit ReLU layers, feeding a scalar value head and a 51-unit advantage head.
Recombination follows Section 3.4, with the mean taken over legal actions only, and illegal entries
driven to a large finite negative value at the output. A finite value rather than `−inf` keeps the
loss and its gradients well defined when a batch contains states with few legal actions.

Total parameters: approximately 85,000. The network is small because the observation is 69
dimensional and hand-engineered; capacity was never the binding constraint in any experiment.

### 4.3 Training procedure

Standard DQN with replay and a target network, structured after the CleanRL single-file reference
implementation and written out rather than imported so that masking could be threaded through every
place it is required. Adam at `1 × 10⁻⁴`; batch size 128; replay capacity 200,000; learning starts
at 5,000 steps; one gradient step every 4 environment steps; target network synchronised every
1,000 steps; gradient norm clipped at 10; Huber loss. Exploration decays linearly from `ε = 1.0` to
`ε = 0.05` over the first 30% of training. One million environment steps per seed, approximately
eleven minutes on CPU.

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

Two further policies are implemented as diagnostics rather than as required baselines.
`RandomMasked` selects uniformly among legal actions and provides a floor: a learned policy that
cannot beat it is broken rather than undertrained. `NeverWait` dispatches at random but never takes
the no-op, isolating the cost of waiting from the cost of choosing badly. Both are reported because
both changed decisions.

### 4.5 Experimental protocol

The protocol was written into `docs/experimental_protocol.md` before any result existed.

Three training seeds: 0, 1, 2, with hyperparameters held identical across them. Evaluation uses 30
held-out instances, seeds 9000–9029, **identical for every policy and every agent seed**, which
makes the comparison paired and removes instance difficulty as a confound. Exploration is disabled
at evaluation: action selection is a deterministic masked argmax.

The agent and all baselines are executed by the same `harness.run_policy` call on the same episodes
through the same metric implementation. Baselines are `Policy` objects, not a separate script.

Six metrics are reported: makespan, average waiting time, machine utilisation, missed-deadline
fraction, weighted tardiness, and cumulative reward. Aggregation returns mean and standard
deviation across seeds and provides no argument that selects a seed, so reporting a single best run
is not possible through the supplied interface.

With three seeds, a difference is compared against the seed-to-seed spread rather than subjected to
a significance test. Three samples do not support one, and we do not claim otherwise.

## 5. Results

*(Filled from `logs/eval/aggregate.json` once the three-seed run completes. Every figure in this
section regenerates from committed logs via `scripts/make_figures.py`, which reads logs only and
never steps the environment.)*

## 6. Discussion

### 6.1 The no-op taught the agent to stall

The most consequential finding was a formulation error, not a coding one.

The action space originally included a no-op, letting the agent decline to dispatch and allow the
clock to advance. The intent was to permit a genuinely useful behaviour: holding a fast machine
idle for an urgent job about to arrive. Trained under otherwise identical settings for 300,000
steps on seed 0, the resulting policy reached an average waiting time of 7.95 on the held-out set —
**worse than selecting uniformly at random among legal actions**, which scores 6.61.

That comparison is what made the diagnosis possible. A learned policy that loses to random is
broken rather than undertrained, and the distinction determines whether the response is more
compute or an investigation. Instrumenting the policy showed it selecting the no-op at 46.3% of
decision epochs against 0.7% for an untrained network of the same architecture. Training was
actively teaching the agent to wait.

The mechanism follows from the reward structure. Dispatching at an epoch that remains a decision
epoch advances the clock by `Δt = 0` and returns reward 0, whereas every action that advances time
returns a negative reward. Under discounting, an agent facing a stream of negative rewards can
improve its discounted return by deferring them, and the no-op is the action that makes deferral
available. With a value function that barely discriminated between actions — measured Q-spread
across legal actions of 0.11, against episode returns near −2 — that bias was sufficient to
dominate the policy.

Masking the no-op removed the deferral action. Under the same configuration and budget, average
waiting time fell from 7.95 to 4.74 and weighted tardiness from 348.4 to 154.8. The catalogue
specifies the action as *"assign a selected job to a selected machine"*, so a dispatch-only space
is faithful to the brief; the no-op was our addition, and it is reported here with the evidence
that removed it.

### 6.2 Two silent implementation faults

Neither produced an error message; both are recorded in Appendix A with the tests that now cover
them.

The first placed training instance seeds for seeds 1 and 2 above 9000, inside the held-out
evaluation range. Had it shipped, two of three seeds would have been evaluated on instances they
trained on and every headline number would have been inflated.

The second supplied an all-ones mask for the current state inside the loss. Because the dueling
aggregation subtracts the mean advantage over legal actions, `Q(s,a)` during the update was a
different function from the one the behaviour policy evaluated. The observable symptom was an agent
that degraded with training: average waiting time rose from 7.30 to 17.28 across a 20,000-step run.

### 6.3 On exploration and convergence

Exploration is uniform over legal actions, which is a meaningful constraint in this environment:
typically fewer than fifteen of the fifty-one actions are legal, so unmasked ε-greedy would spend
most of its budget on rejected moves. The linear ε decay over the first 30% of training was not
tuned; no hyperparameter search was conducted, and none is claimed.

The training curves are reported in Section 5. Where the agent's return was still improving at the
end of the budget, that is stated rather than presented as convergence.

## 7. Limitations and deployment considerations

**The state is partially observed.** Only the ten most urgent queued jobs are visible, and the
realised arrival times of unreleased jobs are not in the state at all. The representation captures
sufficient statistics rather than realisations. A recurrent encoder, or stacking a short history of
observations, is the natural response and was not attempted.

**Masking the no-op forecloses a real capability.** An operator sometimes should hold a fast
machine for an urgent job arriving imminently. The headline agent cannot express that. The
underlying cause is a reward whose per-step signal is almost always negative, making deferral
attractive under discounting; a potential-based shaping term, or a positive completion baseline
that makes progress intrinsically rewarding, would likely allow the no-op to be retained. Testing
that was out of scope for the time available.

**The instance distribution is synthetic and single-operation.** Each job requires one operation on
one machine. Real job shops have multi-operation jobs with precedence constraints, sequence-
dependent setup times, machine breakdowns, and non-stationary arrival rates. Nothing here
demonstrates transfer to those settings, and the load parameters were chosen partly to make the
comparison informative, which is a legitimate experimental choice but not a claim about any real
workload.

**Three seeds is a small sample.** Differences are compared against the seed-to-seed spread. No
significance test is performed and none would be supported.

**For deployment**, three properties matter more than the headline metric. Inference is a single
forward pass through an 85,000-parameter network, so the policy is fast enough to sit inside a
dispatch loop. The action mask is enforced by the environment, so an out-of-distribution
observation cannot produce an illegal schedule — the failure mode is a poor legal choice, not an
invalid one. But the policy inherits the arrival distribution it was trained on, and a shift in
load would require retraining; a deployed system would need the load monitor of
`scripts/check_load.py` running continuously, with the dispatching rules retained as a fallback.

## 8. Conclusion and further work

We formulated dynamic job scheduling on heterogeneous parallel machines as a masked-action MDP and
trained a Dueling DQN on it, comparing against three dispatching rules on held-out instances under
a protocol fixed in advance.

The most useful result was methodological. Three of the four substantive problems encountered —
an instance-seed overlap with the held-out set, a mask omitted from the loss, and a reward
structure that rewarded stalling — produced no error message and would have survived into the
report had they not been measured for. A uniform-random legal policy and a load-separation check,
neither of them required by the brief, are what surfaced them.

Further work, in order of expected value: retain the no-op with potential-based reward shaping;
replace the fixed ten-slot window with a permutation-invariant encoder over the whole queue;
evaluate the `double_q` flag as the D3QN combination the literature favours; and test transfer
across arrival rates the agent was not trained on.

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
