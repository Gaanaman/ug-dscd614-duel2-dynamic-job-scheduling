# Dynamic Job Scheduling with a Dueling Deep Q-Network

DSCD 614 · Group 11 · Option DUEL-2

---

## 1. Introduction

A manufacturing cell or a compute cluster faces the same question many times an hour: a machine has
gone free, several jobs wait, one must be chosen. The choice is irreversible, its cost is invisible
until much later, and the waiting set changes as new work arrives. Unlike the classical job-shop
problem, the full workload is never known in advance.

Industry answers with dispatching rules: First-Come-First-Served, Shortest-Job-First and Round
Robin. Each applies one greedy criterion identically in every state. That is why they are trusted and
where they can be improved: none can trade a small loss now against a larger saving later, because
none models later at all.

This project asks whether reinforcement learning can learn that trade-off. We formulate dynamic
scheduling on parallel non-identical machines as an MDP, train a Dueling DQN, and compare against
all three rules under a protocol fixed before any result was seen.

We set four aims. Specify the problem as an MDP with a fixed-dimension state, a masked discrete
action space and a reward traceable to the evaluation metrics. Implement and train a Dueling DQN
with correct action masking. Evaluate against the dispatching rules on held-out instances, with
variation reported across seeds. Then diagnose whatever the comparison shows. The fourth aim did
substantial work.

## 2. Background

### 2.1 Value-based deep reinforcement learning

Deep Q-Networks (Mnih et al., 2015) fit `Q(s, a)` to `y = r + γ·max_{a'} Q_target(s', a')`, with
replay to decorrelate samples and a target network to stabilise the regression. The dueling
architecture (Wang et al., 2016) modifies only the head: a shared trunk feeds a scalar state-value
`V(s)` and an advantage vector `A(s, a)`, recombined as

$$Q(s,a) \;=\; V(s) \;+\; \Bigl( A(s,a) \;-\; \tfrac{1}{|\mathcal{A}|}\sum_{a'} A(s,a') \Bigr)$$

Subtracting the mean resolves identifiability, since `V` and `A` are otherwise determined only up to
a constant. Action ordering is preserved, so the greedy policy is unchanged.

### 2.2 Why dueling suits scheduling specifically

Dueling is motivated by states where the action choice barely matters, so estimating a separate `Q`
per action wastes capacity re-deriving a shared state value. Scheduling is a clean instance: value
is dominated by congestion, which is identical for every action in the state, while the advantage of
one assignment over another is often small and sometimes exactly zero. Two idle machines of equal
speed and two identical jobs make several actions interchangeable. Dueling learns congestion once in
`V(s)` and leaves the advantage stream the differences. Our measurements support the premise: the
Q-spread across legal actions averaged 0.11 against episode returns near −2, roughly two orders of
magnitude smaller than the between-state signal.

### 2.3 Prior work in the domain

Han and Yang (2020) is the closest published precedent and it settles three design questions.
Working on adaptive job-shop scheduling, they encode "manufacturing states as multi-channel images"
into a CNN, use "various heuristic rules as available actions", and train a dueling double DQN
with prioritised replay. On 85 OR-Library instances the method "performs better than any single
heuristic rule", with performance comparable to a genetic algorithm.

Three points follow for this project. Their action space is dispatching-rule selection, not direct
operation assignment. Their state carries structure over jobs rather than a flat concatenation.
Their benchmark is any single heuristic rule. This is the strong form of the comparison, and much
of the surrounding literature does not adopt it.

The structural-state finding is corroborated: Zhang et al. (2020) embed the disjunctive graph with a
GNN to obtain a size-agnostic policy, and Smit et al. (2024) survey the GNN literature for
scheduling. Separately, two studies apply reward shaping to dynamic flexible job-shop
scheduling with random arrivals, which is the setting here (Zhang et al., 2024; Zhang et al., 2025),
the second combining it with a dueling architecture, both motivated by sparse and delayed
scheduling rewards. Lv et al. (2025) survey the wider field.

Where we depart. The literature standard is D3QN. This project is bound to Dueling DQN, so the
headline configuration uses dueling alone and exposes `double_q` as a configuration flag rather
than silently adopting the stronger method. We state this explicitly because a reader familiar with
the field would otherwise wonder why the obvious enhancement is missing.

## 3. Problem formulation

### 3.1 The scheduling problem

`M` non-preemptive parallel machines. Machine `m` has speed `s_m`, so job `j` occupies `p_j/s_m`
time units. Jobs arrive as a Poisson process with rate `λ`. An episode generates `N`. Job `j` carries
arrival `a_j`, processing time `p_j`, weight `w_j`, deadline `d_j`, and realised completion `C_j`.

### 3.2 Decision epochs

A decision epoch occurs when at least one machine is idle *and* at least one job is pending;
between epochs the simulator jumps to the next completion or arrival, so episode length scales with
the number of jobs rather than simulated time. Every dispatching rule completes an episode in exactly
50 epochs, one per job. `Δt_i = t_{i+1} − t_i` is the elapsed time between consecutive epochs.

### 3.3 State space

Fixed dimension `d = 5K + 3M + 4`. With `K = 10` visible job slots and `M = 5` machines, `d = 69`.

The pending queue is sorted by `(deadline, processing time)` and the first `K` jobs occupy the
observation window.

Job window, `K` × 5: processing time `/ p_max`; weight `/ w_max`; slack `(d_j − t_i)/H` clipped
to `[−1,1]`; waiting time `(t_i − a_j)/H` clipped to `[0,1]`; occupancy flag.
Machine bank, `M` × 3: time until free `/ p_max`; speed `/ s_max`; utilisation so far.
Global, 4: queue length `/ N`; time `/ H`; jobs outstanding `/ N`; arrival rate over capacity.

Justification. Every feature is *relational* rather than absolute: slack and waiting measured
against the current clock, machine state as time-until-free. A policy learned at one point in an
episode is therefore applicable at another, which lets one network serve a whole episode across
varying congestion. An absolute encoding would force the network to relearn the same logic for each
region of the clock.

### 3.4 Action space and masking

`Discrete(K · M + 1) = Discrete(51)`. Action `a = k·M + m` assigns the job in slot `k` to machine
`m`. Action `a = K·M` is a no-op that commits no assignment and advances the clock.

A mask `μ ∈ {0,1}^51` accompanies every observation: `μ[kM + m] = 1` iff slot `k` is occupied and
machine `m` is idle. Because a decision epoch requires both an idle machine and a pending job, at
least one assignment is always legal, so the mask is never empty.

The mask is applied in five places, and all five are required. ε-greedy exploration samples only
legal actions; greedy selection argmaxes with illegal entries at a large negative value; the
bootstrap target restricts `max_{a'} Q_target(s',a')` to actions legal in `s'`, requiring the
next-state mask in the buffer. The dueling mean is taken over legal actions only, or values from
unreachable outputs leak into `V(s)`. Finally, the current-state mask must be supplied during the update,
or `Q(s,a)` in the loss differs from what the behaviour policy evaluates. Appendix A.5 reports what
happened when it was not.

Two formulations are implemented and both reported. Formulation A is the direct assignment just
described. Formulation B, the headline, is dispatching-rule selection: `Discrete(8)` over SPT,
LPT, EDD, FCFS, WSPT, minimum slack, critical ratio and apparent tardiness cost. The rule selects the
job; the machine is the fastest idle one for every rule, so the action isolates job selection.

The catalogue lists direct assignment as a *candidate* and the brief credits a justified departure.
The justification is Han and Yang (2020). The mechanism is that SPT implements a comparison across
queued jobs which, under A, the network must rediscover for every pairing of slot positions in a flat
vector, while under B the rule performs it. `FixedRule(SPT)` reproduces Shortest-Job-First to within
1e-9 on every metric and `FixedRule(FCFS)` reproduces First-Come-First-Served, so the formulations
are shown comparable rather than assumed so. The no-op is masked out in both, evidenced in §6.3.

### 3.5 Reward

At epoch `i`, with `Q_i` the pending set, `I_i` the idle machines and `F_i` the jobs completing in
`[t_i, t_{i+1})`:

$$
r_i \;=\; -\frac{\alpha\,\Delta t_i\,|Q_i| \;+\; \beta\,\Delta t_i\,|I_i|}{Z}
\;+\; \frac{\gamma_c \sum_{j \in F_i} w_j}{Z}
\;-\; \frac{\delta \sum_{j \in F_i} w_j \max\!\left(0,\, C_j - d_j\right)}{Z}
$$

with $Z = N\bar{p}$, the number of jobs times mean processing time, chosen so episode returns are of
order 1 across instance sizes. Weights are $\alpha = 1.0$, $\beta = 0.3$, $\gamma_c = 1.0$, $\delta = 2.0$.

The first term is not shaping. Summed over an episode, $\sum_i \Delta t_i |Q_i|$ is the area under the
queue-length curve. That equals the total time all jobs spend waiting, the quantity reported as
`avg_waiting_time × N`. It is the evaluation objective decomposed over decision epochs, giving a
dense per-step signal that cannot drift from the metric it is scored on. The identity is asserted
numerically on a full episode in `tests/test_reward.py`.

### 3.6 Termination, truncation and discount

Termination occurs when all `N` jobs complete: a genuine absorbing state. Truncation occurs
at `T_max = 4N = 200` epochs or when time exceeds `H`. It is a harness limit, not a property of the
task, so a truncated state is bootstrapped rather than zeroed, and the buffer stores `terminated`
alone for this reason.

`γ = 0.99`. Episodes run 50 decision epochs under every rule, so the effective horizon
`1/(1−γ) = 100` covers an episode twice over and the tardiness consequence of an early dispatch
stays visible at the moment of that dispatch. `γ = 0.95` would hide the back half of an episode from
the front half; `γ = 0.999` adds variance without extending reach.

### 3.7 Does the Markov property hold?

No, and both violations follow from choices we made. *Queue truncation:* only the `K = 10`
head-of-queue jobs are observable. The queue is sorted by a fixed key so the window holds the ten
most urgent, and `|Q_i|` is in the global block so the agent observes the size of what it cannot see.
*Unobserved future arrivals:* realised arrival times of unreleased jobs are absent and `λ` is
included instead, making the process Markov *in distribution*. The environment is therefore a POMDP
under any finite state representation.

## 4. Methodology

### 4.1 Environment construction

The environment is a custom Gymnasium (Towers et al., 2023) environment implementing the MDP of Section 3, written by the
group: an event-driven simulator advancing only to the next completion or arrival, querying the
agent only at decision epochs.

Instance generation is separated from agent stochasticity: each episode's job stream comes from a
dedicated RNG independent of the seed controlling network initialisation and exploration. Training
instance seeds occupy disjoint bands of 3000 below 9000. Evaluation uses 9000–9029 exclusively,
guarded by an assertion and two tests. `check_env` passes, run in permissive action mode because the
checker cannot respect a mask. Training and evaluation always run strict.

Environment constants were fixed by measurement. At `λ = 0.55` makespan was 97.95 against an
arrival bound `N/λ` of 90.9. The arrival process, not the scheduler, set the finish time, and the
rules were indistinguishable. Sweeping `λ` gave 1.0 (`ρ = 1.24`), where makespan is 69.06 against a
bound of 50.0. Appendix A.3; `scripts/check_load.py` retains the check.

### 4.2 Network architecture

Shared trunk of two 256-unit ReLU layers feeding a scalar value head and an advantage head.
Recombination follows §3.4, with the mean over legal actions only and illegal entries at a large
finite negative value. A finite value rather than `−inf` keeps the loss defined when a batch holds states
with few legal actions. About 85,000 parameters. Capacity was never the binding constraint.

### 4.3 Training procedure

Standard DQN with replay and a target network, structured after the CleanRL single-file reference
(Huang et al., 2022)
and written out rather than imported so masking could be threaded everywhere it is required. Adam at
`1 × 10⁻⁴`; batch 128; replay 200,000; learning starts at 5,000; one gradient step per 4 environment
steps; target sync every 1,000; gradient clipped at 10; Huber loss; ε decaying 1.0 → 0.05 over the
first 30% of training; one million steps per seed. The buffer stores both the current and the next
mask, for the reasons in §3.4.

Two enhancements are evaluated by ablation and neither changes the algorithm family. Prioritised
experience replay (Schaul et al., 2016) samples in proportion to the last temporal-difference error
with importance-sampling weights annealed to 1, following Han and Yang (2020) and Liu et al.
(2025). It alters which transitions are drawn, not the
learning rule. n-step returns propagate a delayed consequence to the causing action in one update
rather than n (Hessel et al., 2018), which matters because a dispatch returns reward 0 when committed. The buffer stores
the discount actually applied, so a window flushed at an episode boundary carries `γ^k` for the `k`
rewards accumulated. `double_q` is exposed as a flag and is off throughout, so the reported
algorithm is Dueling DQN as the brief requires.

### 4.4 Baseline design

The three required rules, First-Come-First-Served, Shortest-Job-First and Round Robin, are
implemented by the group behind a common `Policy` interface. Each chooses a job and then the fastest
idle machine, so the comparison isolates job selection, and every rule sees the same ten-slot
window the agent sees. A baseline with full queue visibility would not compete under the agent's
constraints.

`RandomMasked` is a diagnostic floor, not a required baseline: a policy that cannot beat uniform
selection among legal actions is broken rather than undertrained. It changed decisions, so it is
reported.

A second, harder bar. Under Formulation B the eight rules are themselves policies and two beat
all three required baselines. Any rule in the action set is reachable by a policy that always selects
it, so beating Shortest-Job-First is not evidence of learning. Results are reported against the
best single rule, the benchmark Han and Yang use.

### 4.5 Experimental protocol

The protocol was written into `docs/experimental_protocol.md` before any result existed.

Three training seeds (0, 1, 2) with hyperparameters held identical. Evaluation uses 30 held-out
instances, seeds 9000–9029, identical for every policy and every agent seed, making the
comparison paired. Exploration is disabled. Agent and baselines are executed by the same
`harness.run_policy` call through the same metric implementation. Baselines are `Policy` objects,
not a separate script. Aggregation returns mean and standard deviation and exposes no argument that
selects a seed, so reporting a single best run is not possible through the interface. Differences
are compared against the seed spread. Three samples do not support a significance test.

## 5. Results

Three seeds, one million steps each, on the 30 held-out instances with exploration disabled. Every
policy, covering eight fixed rules, three required baselines, the random floor and every agent variant, runs
through the same `harness.run_policy` call on the same instances through the same metric code.

### 5.1 Training

Return rises from −2.58 at 100,000 steps to −1.92 at 700,000 and is flat thereafter, so the budget
sufficed; seed spread stays at or below 0.062. Training returns include ε = 0.05 and are not
comparable to evaluation numbers.

### 5.2 The dispatching rules, and the bar they set

| Rule | Avg. waiting | Missed | Weighted tardiness | Return |
|---|---|---|---|---|
| SPT (= SJF) | 4.077 | 0.256 | 136.4 | −1.352 |
| WSPT | 4.344 | 0.276 | 104.9 | −1.187 |
| ATC | 4.372 | 0.277 | 95.1 | −1.125 |
| EDD | 4.521 | 0.363 | 124.0 | −1.345 |
| CR | 4.851 | 0.445 | 133.8 | −1.465 |
| MS | 4.947 | 0.421 | 144.8 | −1.554 |
| FCFS | 5.539 | 0.465 | 212.3 | −2.099 |
| LPT | 7.744 | 0.421 | 437.9 | −3.958 |

Two rules inside the action set beat all three required baselines. ATC is the bar: any policy
in Formulation B can reach −1.125 by always selecting it, so beating Shortest-Job-First is not
evidence of learning. This is the benchmark Han and Yang use.

### 5.3 Ablation

| Variant | Per-seed return | Mean | s.d. | Gap to ATC |
|---|---|---|---|---|
| Formulation A, direct assignment | −1.525, −1.544, −1.561 | −1.543 | 0.015 | −0.418 |
| Formulation B, uniform replay | −1.334, −1.314, −1.247 | −1.298 | 0.037 | −0.173 |
| B + prioritised replay | −1.309, −1.333, −1.294 | −1.312 | 0.016 | −0.187 |
| B + n-step 3 | −1.222, −1.212, −1.269 | −1.235 | 0.025 | −0.109 |
| B + PER + n-step 3 | −1.315, −1.282, −1.263 | −1.287 | 0.021 | −0.161 |

Differences are compared against the combined seed spread, with per-seed dominance checked.

- Action space is the largest of the three effects tested. Formulation B improves on A by 0.245,
  far beyond any spread. Every B variant outperforms all three required baselines, whereas A
  outperforms only First-Come-First-Served.
- n-step returns help: +0.064 against a combined spread of 0.062, winning on every seed.
- Prioritised replay does not: −0.014 against a spread of 0.053, no per-seed dominance, and a
  0.052 degradation when added on top of n-step.

### 5.4 Headline result

The best configuration is Formulation B with n-step 3 returns, at −1.235 ± 0.025. It wins on
every metric and every seed against the required baselines: better than First-Come-First-Served by
0.864 in return, Shortest-Job-First by 0.117 and Round Robin by 0.164, all beyond the seed spread.

Against the bar it loses. ATC scores −1.125 against the agent's −1.235, a shortfall of 0.109
exceeding the seed spread. The agent selects rules better than any weak rule and better than both
required dispatching rules, and does not select them better than always choosing the strongest.
§6.5 takes up why.

Machine utilisation (0.884–0.927) and makespan vary by under 5% across all thirteen policies. Both
are dominated by the arrival process and are reported because the brief requires them.

## 6. Discussion

### 6.1 The action space was the binding constraint

Formulation B improves on A by 0.245 in return, an order of magnitude larger than any other
change tested and far beyond the seed spread. Under A the agent outperformed only
First-Come-First-Served, whereas under B every variant outperforms all three required baselines.

The mechanism is specific. SPT implements a comparison across queued jobs. Under A the network must
learn that from a flat 69-dimensional vector where each slot occupies a fixed, arbitrary offset, so
the comparison has to be rediscovered for every pairing of positions. Under B the rule performs the
comparison and the network learns only *when* each rule applies. The measured Q-spread across legal
actions of 0.11 against episode returns near −2 is the direct evidence: the network learned state
value well and between-action ordering poorly.

### 6.2 Prioritised replay did not help, contrary to our own prediction

Our literature review recommended prioritised replay as the highest-value change, because it sits in
the recipe of the closest precedent. The ablation contradicts that: −0.014 against a combined spread
of 0.053 with no per-seed dominance, and a *degradation* of 0.052 when added on top of n-step.

This does not refute Han and Yang. Their CNN-over-images state, 85 static OR-Library instances and
dueling double DQN all differ from our setting, and any could change the value of prioritising
by temporal-difference error. The result applies to this problem only: with this state and action
space, prioritised replay contributes nothing measurable, while n-step returns contribute a real
improvement. The review found no scheduling-domain ablation isolating prioritised replay, so this
addresses a gap it identified.

The mechanism fits the diagnosis: reward is 0 when a dispatch is committed and its cost lands tens
of decisions later. n-step propagates that consequence to the causing action. Prioritised replay
changes which transitions are replayed, not how far credit travels.

### 6.3 Three faults found by measuring rather than assuming

Three problems surfaced during development, none producing an error message. The action space
originally included a no-op. The policy chose it at 46.3% of decision epochs and scored worse than
a uniform-random legal policy, the diagnostic separating a broken agent from an undertrained one.
Training instance seeds for two of three seeds fell inside the held-out evaluation range, which would
have inflated every headline number. And the loss supplied an all-ones current-state mask, so
`Q(s,a)` during the update differed from what the behaviour policy evaluated, producing an agent that
degraded as it trained.

Each was found by a cheap measurement taken before the result was trusted. Full forensics, with the
tests now covering each, are in Appendix A.

### 6.5 Why the agent stops short of the best single rule

Two explanations are ruled out by measurement. Compute is not the constraint: return plateaus
from 700,000 steps with seed spread under 0.07. Reward misalignment is not either: the reward
ranking and the metric ranking agree and ATC tops both, so a better score on this reward is a better
schedule.

The state representation is the remaining constraint. Under Formulation B the network decides
*which rule suits the current queue* while seeing that queue as a flat concatenation of ten slots. The literature's answer is
a structured encoder, either a graph network over the disjunctive representation or attention over
the queue. Such an encoder makes cross-job comparison native and yields a size-agnostic policy. That is the
leading item in further work.

Exploration is uniform over legal actions. No hyperparameter search was conducted.

## 7. Limitations and deployment considerations

The state is partially observed. Only the ten most urgent queued jobs are visible, and arrival
times of unreleased jobs are absent. A recurrent encoder, or stacking a short observation history,
is the natural response and was not attempted.

Masking the no-op forecloses a real capability. An operator should sometimes hold a fast machine
for an urgent job arriving imminently, and the agent cannot express that. The cause is a reward
whose per-step signal is almost always negative, making deferral attractive under discounting;
potential-based shaping would likely let the no-op be retained. Testing that was out of scope.

The instance distribution is synthetic and single-operation. Real job shops have multi-operation
jobs with precedence constraints, sequence-dependent setups, breakdowns, and non-stationary
arrivals. Nothing here demonstrates transfer, and the load parameters were chosen partly to make the
comparison informative.

Three seeds is a small sample. Differences are compared against the seed spread; no significance
test is supported.

For deployment, three properties matter more than the headline metric. Inference is one forward
pass through a small network, fast enough for a dispatch loop. The mask is enforced by the
environment, so an out-of-distribution observation cannot produce an illegal schedule. The failure
mode is a poor legal choice, not an invalid one. Under Formulation B every action is a named
dispatching rule, so an operator can audit any decision. However, the policy inherits its training
arrival distribution, so a load shift requires retraining and the rules stay as fallback.

## 8. Conclusion and further work

We formulated dynamic job scheduling on heterogeneous parallel machines as a masked-action MDP and
trained a Dueling DQN on it under two action formulations, comparing against three required
dispatching rules and against the best of eight rules, on held-out instances under a protocol fixed
in advance.

The most useful result was methodological. Three substantive problems produced no error message: an instance-seed overlap with
the held-out set, a mask omitted from the loss, and a reward structure that rewarded stalling. None
of them and would have survived unmeasured. A uniform-random legal policy and a
load-separation check, neither required by the brief, surfaced them.

Further work, in order of expected value: replace the fixed ten-slot window with a
permutation-invariant encoder over the queue; retain the no-op under potential-based shaping;
evaluate the `double_q` flag as the D3QN combination the literature favours; and test transfer
across untrained arrival rates.

---

## 9. References

Han, B.-A., & Yang, J.-J. (2020). Research on adaptive job shop scheduling problems based on
dueling double DQN. *IEEE Access*, 8, 186474–186495. https://doi.org/10.1109/ACCESS.2020.3029868

Hessel, M., Modayil, J., van Hasselt, H., Schaul, T., Ostrovski, G., Dabney, W., Horgan, D., Piot,
B., Azar, M., & Silver, D. (2018). Rainbow: Combining improvements in deep reinforcement learning.
*Proceedings of the AAAI Conference on Artificial Intelligence*, 32(1).
https://doi.org/10.1609/aaai.v32i1.11796

Huang, S., Dossa, R. F. J., Ye, C., Braga, J., Chakraborty, D., Mehta, K., & Araújo, J. G. (2022).
CleanRL: High-quality single-file implementations of deep reinforcement learning algorithms.
*Journal of Machine Learning Research*, 23(274), 1–18.

Liu, C., Chen, K., Wang, H., Yang, B., & Leng, J. (2025). Job shop scheduling by deep dual-Q network
with prioritized experience replay for resilient production control in flexible manufacturing
system. *Computers & Operations Research*, 183, 107190. https://doi.org/10.1016/j.cor.2025.107190

Lv, L., Zhang, C., Fan, J., & Shen, W. (2025). Deep reinforcement learning for job shop scheduling
problems: A comprehensive literature review. *Knowledge-Based Systems*, 321, 113633.
https://doi.org/10.1016/j.knosys.2025.113633

Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., Graves, A.,
Riedmiller, M., Fidjeland, A. K., Ostrovski, G., Petersen, S., Beattie, C., Sadik, A., Antonoglou,
I., King, H., Kumaran, D., Wierstra, D., Legg, S., & Hassabis, D. (2015). Human-level control
through deep reinforcement learning. *Nature*, 518(7540), 529–533.
https://doi.org/10.1038/nature14236

Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2016). Prioritized experience replay.
*International Conference on Learning Representations*. https://arxiv.org/abs/1511.05952

Smit, I. G., Zhou, J., Reijnen, R., Wu, Y., Chen, J., Zhang, C., Bukhsh, Z., Zhang, Y., & Nuijten,
W. (2024). Graph neural networks for job shop scheduling problems: A survey.
https://arxiv.org/abs/2406.14096

Towers, M., Terry, J. K., Kwiatkowski, A., Balis, J. U., de Cola, G., Deleu, T., Goulão, M.,
Kallinteris, A., KG, A., Krimmel, M., Perez-Vicente, R., Pierré, A., Schulhoff, S., Tai, J. J.,
Shen, A. T. J., & Younis, O. G. (2023). *Gymnasium*. https://gymnasium.farama.org

van Hasselt, H., Guez, A., & Silver, D. (2016). Deep reinforcement learning with double Q-learning.
*Proceedings of the AAAI Conference on Artificial Intelligence*, 30(1).
https://doi.org/10.1609/aaai.v30i1.10295

Wang, Z., Schaul, T., Hessel, M., van Hasselt, H., Lanctot, M., & de Freitas, N. (2016). Dueling
network architectures for deep reinforcement learning. *Proceedings of the 33rd International
Conference on Machine Learning*, 48, 1995–2003. https://arxiv.org/abs/1511.06581

Zhang, C., Song, W., Cao, Z., Zhang, J., Tan, P. S., & Xu, C. (2020). Learning to dispatch for job
shop scheduling via deep reinforcement learning. *Advances in Neural Information Processing
Systems*, 33. https://arxiv.org/abs/2010.12367

Zhang, L., Yan, Y., Yang, C., & Hu, Y. (2024). Dynamic flexible job-shop scheduling by multi-agent
reinforcement learning with reward-shaping. *Advanced Engineering Informatics*, 62, 102872.
https://doi.org/10.1016/j.aei.2024.102872

Zhang, Z.-Q., Wu, Z.-M., Qian, B., & Hu, R. (2025). A reward-shaping dueling distributed multi-agent
deep reinforcement learning framework for dynamic flexible job shop scheduling with random job
arrivals. *Expert Systems with Applications*, 297, 128951.
https://doi.org/10.1016/j.eswa.2025.128951

Zhang, W., Kong, M., Zhang, Y., Fathollahi-Fard, A. M., & Tian, G. (2025). A revised deep
reinforcement learning algorithm for parallel machine scheduling problem under multi-scenario due
date constraints. *Swarm and Evolutionary Computation*, 92, 101808.
https://doi.org/10.1016/j.swevo.2024.101808

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

### A.4 The no-op taught the agent to stall

The action space originally included a no-op. At 300,000 steps the resulting policy reached an
average waiting time of 7.95, **worse than selecting uniformly at random among legal actions**
(6.61), and chose the no-op at 46.3% of decision epochs against 0.7% for an untrained network.

A policy losing to random is broken rather than undertrained, and the distinction decides whether
the response is more compute or an investigation. Dispatching at an epoch that remains a decision
epoch advances the clock by `Δt = 0` and returns reward 0, while every action advancing time returns
a negative reward. Under discounting an agent facing negative rewards improves its return by
deferring them, and the no-op makes deferral available. Masking it dropped waiting time to 4.74.

### A.5 Two silent implementation faults

Neither produced an error message; both are in Appendix A with the tests now covering them. The
first placed training instance seeds for two of three seeds inside the held-out evaluation range,
which would have inflated every headline number. The second supplied an all-ones current-state mask
inside the loss, so `Q(s,a)` during the update was a different function from the one the behaviour
policy evaluated; the symptom was an agent degrading with training, waiting time rising from 7.30 to
17.28 over a 20,000-step run.


## Appendix B — Reproduction

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
bash scripts/run_all.sh
```

Seeds, exact dependency versions and the full hyperparameter table are in
`docs/hyperparameters.md`. Raw logs backing every figure are committed under `logs/`.
