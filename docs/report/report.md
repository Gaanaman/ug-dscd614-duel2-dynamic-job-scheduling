# Dynamic Job Scheduling with a Dueling Deep Q-Network

DSCD 614 · Group 11 · Option DUEL-2

Reading copy exported from `paper/main.tex`, `paper/body.tex` and `paper/appendix.tex`, which are the source. Citation keys refer to `paper/references.bib`.

---

## Abstract

# Introduction

A manufacturing cell or a compute cluster faces the same question many times an hour: a machine has gone free, several jobs wait, and one must be chosen. The choice cannot be reversed, its cost is not visible until much later, and the set of waiting jobs changes as new work arrives. Unlike the classical job-shop problem, the full workload is not known in advance.

Dispatching rules are the standard industrial answer, and the three the brief requires are First-Come-First-Served, Shortest-Job-First and Round Robin. Each applies one greedy criterion identically in every state, so none can trade a small loss now against a larger saving later, because none represents the later state.

To test whether reinforcement learning can learn that trade-off, we formulate dynamic scheduling on parallel non-identical machines as an MDP, train a Dueling DQN on it, and compare the result against all three rules under a protocol fixed before any result was seen. The problem is specified as an MDP with a fixed-dimension state, a masked discrete action space and a reward traceable to the evaluation metrics. A Dueling DQN with action masking is trained on it and evaluated on held-out instances with variation reported across seeds, and the result is explained.

# Background

## Value-based methods

Deep Q-Networks fit `Q(s, a)` to `y = r + `$`\gamma`$$`\cdot`$`max_{a'} Q_target(s', a')`, using a replay buffer to decorrelate samples and a target network to stabilise the regression. The dueling architecture modifies only the head, following the shared trunk with two streams: one estimates a scalar state value `V(s)`, the other an advantage vector `A(s, a)`. The two are recombined as

``` math
Q(s,a) = V(s) + \Bigl( A(s,a) - \tfrac{1}{|\mathcal{A}|}\textstyle\sum_{a'} A(s,a') \Bigr)
```

Subtracting the mean resolves identifiability, since `V` and `A` are otherwise determined only up to a constant. The ordering of actions is preserved, so the greedy policy is unchanged.

## Dueling and scheduling

Dueling is motivated by states in which the choice of action barely matters, so that estimating a separate `Q` per action spends capacity re-deriving a shared state value. Scheduling has that structure. Value is dominated by congestion, which is identical for every action available in a state. The advantage of one assignment over another is often small and sometimes zero, since two idle machines of equal speed and two identical jobs make several actions interchangeable. Measurements on the trained network are consistent with this. The Q-spread across legal actions averaged 0.11 against episode returns near -2, two orders of magnitude smaller than the between-state signal.

## Prior work

The closest published precedent is , and three of our design decisions follow from it. Working on adaptive job-shop scheduling, they encode “manufacturing states as multi-channel images” into a CNN, use “various heuristic rules as available actions”, and train a dueling double DQN with prioritised replay. On 85 OR-Library instances the method reaches optimal solutions on the small instances and “performs better than any single heuristic rule for large scale problems”. Their action space is therefore dispatching-rule selection rather than direct operation assignment, and their state representation is structured over jobs rather than flat. Their benchmark is any single heuristic rule.

likewise use a structured state, embedding the disjunctive graph with a GNN to obtain a size-agnostic policy, and survey the GNN literature for scheduling. Two further studies apply reward shaping to dynamic flexible job-shop scheduling with random arrivals, the setting here (Zhang et al., 2024; Zhang et al., 2025). survey the wider field.

The standard in that literature is D3QN, but the brief specifies Dueling DQN, so the headline configuration uses dueling alone and exposes `double_q` as a flag.

# Problem formulation

## Scheduling problem

The system has `M` non-preemptive parallel machines, where machine `m` has speed `s_m`, so job `j` occupies `p_j/s_m` time units. Jobs arrive as a Poisson process with rate $`\lambda`$, and an episode generates `N` of them. Job `j` carries arrival `a_j`, processing time `p_j`, weight `w_j`, deadline `d_j`, and realised completion `C_j`.

## Decision epochs

A decision epoch occurs when at least one machine is idle *and* at least one job is pending. Between epochs the simulator jumps to the next completion or arrival, so episode length scales with the number of jobs rather than simulated time. Every dispatching rule completes an episode in exactly 50 epochs, one per job, and $`\Delta`$`t_i = t_{i+1} - t_i` is the elapsed time between them.

## State space

The state has fixed dimension `d = 5K + 3M + 4`, so with `K = 10` visible job slots and `M = 5` machines, `d = 69`. The pending queue is sorted by `(deadline, processing time)` and the first `K` jobs occupy the observation window. The job window contributes `K` $`\times`$ 5 features: processing time `/ p_max`; weight `/ w_max`; slack `(d_j - t_i)/H` clipped to `[``-1,1``]`; waiting time `(t_i - a_j)/H` clipped to `[``0,1``]`; and an occupancy flag. The machine bank contributes `M` $`\times`$ 3: time until free `/ p_max`; speed `/ s_max`; utilisation so far. Four global features complete the vector: queue length `/ N`; time `/ H`; jobs outstanding `/ N`; arrival rate over capacity.

Every feature is *relational* rather than absolute, with slack and waiting measured against the current clock and machine state given as time-until-free. A policy learned at one point in an episode applies at another, so one network serves a whole episode across varying congestion. An absolute encoding would force it to relearn the same logic for each region of the clock.

## Action space and masking

The direct action space is `Discrete(K `$`\cdot`$` M + 1) = Discrete(51)`. Action `a = k`$`\cdot`$`M + m` assigns the job in slot `k` to machine `m`, and action `a = K`$`\cdot`$`M` is a no-op that commits no assignment and advances the clock.

A mask $`\mu`$` `$`\in`$` {0,1}^51` accompanies every observation, with $`\mu`$`[``kM + m``]`` = 1` iff slot `k` is occupied and machine `m` is idle. Because a decision epoch requires both an idle machine and a pending job, at least one assignment is legal and the mask is non-empty.

The mask is applied at five points: $`\epsilon`$-greedy exploration samples only legal actions, and greedy selection argmaxes with illegal entries at a large negative value. The bootstrap target restricts `max_{a'} Q_target(s',a')` to actions legal in `s'`, which requires the next-state mask in the buffer. The dueling mean is taken over legal actions only, since otherwise values from unreachable outputs leak into `V(s)`. Finally, the current-state mask must be supplied during the update, or `Q(s,a)` in the loss differs from what the behaviour policy evaluates, as Appendix A.2 reports.

Formulation A is the direct assignment just described, and Formulation B, the headline, is dispatching-rule selection: `Discrete(8)` over SPT, LPT, EDD, FCFS, WSPT, minimum slack, critical ratio and apparent tardiness cost. The rule selects the job and the machine is the fastest idle one, so the action isolates job selection.

The catalogue lists direct assignment as a *candidate* and the brief credits a justified departure. Our justification is , and §6.1 gives the mechanism. `FixedRule(SPT)` reproduces Shortest-Job-First to within 1e-9 on every metric and `FixedRule(FCFS)` reproduces First-Come-First-Served, so the two formulations are comparable. The no-op is masked out in both, for the evidence in §6.3.

## Reward

At epoch `i`, with `Q_i` the pending set, `I_i` the idle machines and `F_i` the jobs completing in `[``t_i, t_{i+1})`:

``` math
\begin{aligned}
r_i = \;& -\frac{\alpha\,\Delta t_i\,|Q_i| + \beta\,\Delta t_i\,|I_i|}{Z} \\
        & + \frac{\gamma_c \sum_{j \in F_i} w_j}{Z} \\
        & - \frac{\delta \sum_{j \in F_i} w_j \max(0,\, C_j - d_j)}{Z}
\end{aligned}
```

with $`Z = N\bar{p}`$, the number of jobs times mean processing time, chosen so episode returns are of order 1 across instance sizes. Weights are $`\alpha = 1.0`$, $`\beta = 0.3`$, $`\gamma_c = 1.0`$, $`\delta = 2.0`$.

Summed over an episode, $`\sum_i \Delta t_i |Q_i|`$ is the area under the queue-length curve, which equals the total time all jobs spend waiting, the quantity we report as `avg_waiting_time `$`\times`$` N`. The term is the evaluation objective decomposed over decision epochs, so the per-step signal and the evaluation metric agree. `tests/test_reward.py` asserts the identity numerically on a full episode.

## Termination and discount

Termination occurs when all `N` jobs complete, an absorbing state; truncation occurs at `T_max = 4N = 200` epochs or when time exceeds `H`. Truncation is a harness limit rather than a property of the task, so a truncated state is bootstrapped rather than zeroed, which is why the buffer stores `terminated` alone.

We set $`\gamma`$` = 0.99`. Episodes run 50 decision epochs under every rule, so the effective horizon `1/(1-`$`\gamma`$`) = 100` covers an episode twice over and the tardiness consequence of an early dispatch stays visible at the moment of that dispatch. At 0.95 the effective horizon of 20 covers less than half an episode; at 0.999 the added variance brings no further reach.

## Markov property

The state is not Markov, for two reasons that follow from the design. Under *queue truncation*, only the `K = 10` head-of-queue jobs are observable. The queue is sorted by a fixed key so the window holds the ten most urgent, and `Q_i` sits in the global block, so the agent observes the number of jobs outside the window. Under *unobserved future arrivals*, realised arrival times of unreleased jobs are absent and $`\lambda`$ is included instead, which makes the process Markov *in distribution*. The environment is therefore a POMDP under any encoding of this form.

# Methodology

## Environment construction

The group wrote a custom Gymnasium environment implementing the MDP of Section 3. It is an event-driven simulator that advances only to the next completion or arrival, and queries the agent only at decision epochs.

Instance generation is separated from agent stochasticity, so each episode’s job stream comes from a dedicated RNG independent of the seed controlling network initialisation and exploration. Training instance seeds occupy disjoint bands of 3000 below 9000, and evaluation uses 9000–9029 exclusively, guarded by an assertion and two tests. `check_env` passes, run in permissive action mode because the checker cannot respect a mask; training and evaluation run strict.

A load sweep set the arrival rate: at $`\lambda`$` = 0.55` the makespan was 97.95 against an arrival bound `N/`$`\lambda`$ of 90.9, so the arrival process rather than the scheduler set the finish time and the rules differed by 0.06 in average waiting time. Sweeping $`\lambda`$ gave 1.0 ($`\rho`$` = 1.24`), where the makespan is 69.06 against a bound of 50.0. Appendix A.3 gives the sweep, and `scripts/check_load.py` retains the check.

## Network architecture

The network is a shared trunk of two 256-unit ReLU layers followed by a scalar value head and an advantage head. Recombination follows §3.4, with the mean over legal actions only and illegal entries at a large finite negative value. A finite value rather than `-inf` keeps the loss defined when a batch holds states with few legal actions. The network has about 85,000 parameters.

## Training procedure

Training is standard DQN with replay and a target network, structured after the CleanRL single-file reference . We reimplemented it so that the mask could be applied at every point listed in §3.4. The settings are Adam at `1 `$`\times`$` 10-4`, batch 128, replay 200,000, learning starts at 10,000, one gradient step per 4 environment steps, target sync every 1,000, gradient clipped at 10, and Huber loss. $`\epsilon`$ decays 1.0 $`\rightarrow`$ 0.05 over the first 30% of training, and each seed runs one million steps. The buffer stores both masks, for the reasons in §3.4.

Prioritised experience replay , the first of two enhancements evaluated by ablation, samples in proportion to the last temporal-difference error with importance-sampling weights annealed to 1, following and ; it alters which transitions are drawn, not the learning rule. n-step returns, the second, propagate a delayed consequence to the causing action in one update rather than n , which matters here because a dispatch returns reward 0 at the moment it is committed. The buffer stores the discount actually applied, so a window flushed at an episode boundary uses $`\gamma`$`^k` for the `k` rewards accumulated. `double_q` is off throughout, so the algorithm we report is Dueling DQN as the brief requires.

## Baseline design

The group implemented the three required rules, First-Come-First-Served, Shortest-Job-First and Round Robin, behind a common `Policy` interface. Each chooses a job and then the fastest idle machine, so the comparison isolates job selection, and every rule receives the same ten-slot window as the agent. A baseline with full queue visibility would not face the agent’s constraints.

`RandomMasked`, uniform selection among legal actions, is a diagnostic floor rather than a required baseline, and is reported because it informed the action-space design.

Under Formulation B the eight rules are themselves policies, and two of them beat all three required baselines, which sets a second, harder comparison. Any rule in the action set is reachable by a policy that always selects it, so beating Shortest-Job-First alone would not distinguish learning from constant selection of one rule. We therefore report against the best single rule, the benchmark Han and Yang use.

## Experimental protocol

The protocol, written into `docs/experimental_protocol.md` before any result existed, fixes three training seeds (0, 1, 2) with hyperparameters held identical, and evaluation on 30 held-out instances, seeds 9000–9029, identical for every policy and every agent seed, which makes the comparison paired. Exploration is disabled at evaluation. The aggregation code has no option to select a seed, so every reported number is a mean over the three, and differences are compared against the seed spread.

# Results

Results are reported over three seeds, one million steps each, on the 30 held-out instances with exploration disabled. All thirteen policies run through the same `harness.run_policy` call on the same instances through the same metric code.

<figure id="fig:training" data-latex-placement="t">
<img src="figures/training_curve.png" style="width:72.0%" />
<figcaption>Episode return against environment steps, averaged across three seeds with a band at one standard deviation.</figcaption>
</figure>

## Training

Figure <a href="#fig:training" data-reference-type="ref" data-reference="fig:training">1</a> plots episode return against environment steps, averaged across seeds. Return rises from -1.79 at 100,000 steps to -1.52 at 400,000 and is flat thereafter, so the budget sufficed, and the seed spread stays at or below 0.041. Training returns include $`\epsilon`$ = 0.05 exploration and are therefore lower than the evaluation numbers.

## Dispatching rules

<div class="table*">

| Rule            | Avg. waiting | Missed | Weighted tardiness |     Return |
|:----------------|-------------:|-------:|-------------------:|-----------:|
| SPT ($`=`$ SJF) |        4.077 |  0.256 |              136.4 | $`-1.352`$ |
| WSPT            |        4.344 |  0.276 |              104.9 | $`-1.187`$ |
| ATC             |        4.372 |  0.277 |               95.1 | $`-1.125`$ |
| EDD             |        4.521 |  0.363 |              124.0 | $`-1.345`$ |
| CR              |        4.851 |  0.445 |              133.8 | $`-1.465`$ |
| MS              |        4.947 |  0.421 |              144.8 | $`-1.554`$ |
| FCFS            |        5.539 |  0.465 |              212.3 | $`-2.099`$ |
| LPT             |        7.744 |  0.421 |              437.9 | $`-3.958`$ |

</div>

Table <a href="#tab:rules" data-reference-type="ref" data-reference="tab:rules">[tab:rules]</a> lists the eight rules on the held-out instances, ordered by return; two rules inside the action set beat all three required baselines. ATC at -1.125 is the reference, since a Formulation B policy that always selects it reaches that score.

<figure id="fig:ablation" data-latex-placement="t">
<img src="figures/ablation.png" style="width:72.0%" />
<figcaption>Agent variants against the best single dispatching rule. The dashed line is ATC at <span class="math inline">−1.125</span>. Every variant on the rule action space outperforms all three required baselines; none reaches ATC.</figcaption>
</figure>

## Ablation

<div class="table*">

| Variant | Per-seed return | Mean | s.d. | Gap to ATC |
|:---|:--:|:--:|:--:|:--:|
| Formulation A, direct assignment | $`-1.525`$, $`-1.544`$, $`-1.561`$ | $`-1.543`$ | 0.015 | $`-0.418`$ |
| Formulation B, uniform replay | $`-1.334`$, $`-1.314`$, $`-1.247`$ | $`-1.298`$ | 0.037 | $`-0.173`$ |
| B $`+`$ prioritised replay | $`-1.309`$, $`-1.333`$, $`-1.294`$ | $`-1.312`$ | 0.016 | $`-0.187`$ |
| B $`+`$ n-step 3 | $`-1.222`$, $`-1.212`$, $`-1.269`$ | $`-1.235`$ | 0.025 | $`-0.109`$ |
| B $`+`$ PER $`+`$ n-step 3 | $`-1.315`$, $`-1.282`$, $`-1.263`$ | $`-1.287`$ | 0.021 | $`-0.161`$ |

</div>

Table <a href="#tab:ablation" data-reference-type="ref" data-reference="tab:ablation">[tab:ablation]</a> gives per-seed and mean return for each variant, and Figure <a href="#fig:ablation" data-reference-type="ref" data-reference="fig:ablation">2</a> plots the means against ATC. We compare differences against the combined seed spread, with per-seed dominance checked.

- The action space is the largest of the three effects tested. Formulation B improves on A by 0.245, far beyond any spread, and every B variant outperforms all three required baselines, whereas A outperforms only First-Come-First-Served.

- n-step returns help: +0.064 against a combined spread of 0.062, winning on every seed.

- Prioritised replay leaves the return within the spread: -0.014 against 0.053, with no per-seed dominance, and -0.052 when combined with n-step.

## Headline result

The best configuration is Formulation B with n-step 3 returns, at -1.235 $`\pm`$ 0.025. It beats First-Come-First-Served and Round Robin on every metric and every seed, and Shortest-Job-First on return, weighted tardiness (107.6 against 136.4) and makespan. Shortest-Job-First retains the lower average waiting time (4.077 against 4.525) and fewer missed deadlines (0.256 against 0.354). In return the agent improves on First-Come-First-Served by 0.864, on Shortest-Job-First by 0.117 and on Round Robin by 0.164, all beyond the seed spread.

ATC scores -1.125 against the agent’s -1.235, a difference of 0.109 that exceeds the seed spread; §6.4 discusses the cause.

Figure <a href="#fig:bars" data-reference-type="ref" data-reference="fig:bars">3</a> shows every policy on the four metrics. Machine utilisation (0.884–0.927) and makespan vary by under 5% across all thirteen policies; both are dominated by the arrival process, and are reported for completeness.

<figure id="fig:bars" data-latex-placement="t">
<img src="figures/baseline_bars.png" />
<figcaption>Every policy on four metrics, error bars from the spread across seeds. Utilisation and makespan vary by under 5% across policies at this load.</figcaption>
</figure>

# Discussion

## Action space

Formulation B improves on A by 0.245 in return, an order of magnitude larger than any other change tested. Under A the agent outperformed only First-Come-First-Served, whereas under B every variant outperforms all three required baselines.

The mechanism lies in the state encoding: SPT implements a comparison across queued jobs, which under A the network must learn from a flat 69-dimensional vector in which each slot occupies a fixed, arbitrary offset. Under B the rule performs the comparison and the network need only represent *when* each rule applies. The measured Q-spread of 0.11 across legal actions, against episode returns near -2, supports this: the between-state signal is the larger of the two by two orders of magnitude.

## Prioritised replay

Our literature review recommended prioritised replay as the highest-value change, because it is part of the recipe of the closest precedent. In the ablation the change is -0.014 against a combined spread of 0.053, with no per-seed dominance.

The setting differs from in three respects. Their CNN-over-images state, their 85 static OR-Library instances and their dueling double DQN all differ from our setting, and any of those could change the value of prioritising by temporal-difference error. Our review found no scheduling-domain ablation isolating prioritised replay.

The mechanism fits the diagnosis in §6.1. Reward is 0 when a dispatch is committed and its cost appears tens of decisions later, and n-step propagates that consequence to the causing action. Prioritised replay changes which transitions are replayed, not how far credit is propagated.

## Validation checks

Three checks were run before any result was accepted, and each led to a design change that the test suite now enforces. A comparison against the uniform-random legal policy showed the no-op selected at 46.3% of decision epochs, and the no-op was masked out. An assertion on instance seeds confirmed that training and held-out bands are disjoint for every seed. A test on the loss confirmed that `Q(s,a)` in the update uses the same current-state mask as the behaviour policy. Appendix A gives the measurements and the tests.

## ATC gap

Return plateaus from 400,000 steps with a seed spread under 0.05, so more compute does not close the gap, and the reward ranking and the metric ranking agree, with ATC top of both, so a better score on this reward is a better schedule.

The remaining constraint is the state representation: under Formulation B the network selects a rule for the current queue from a flat concatenation of ten slots. Prior work addresses this with a structured encoder, either a graph network over the disjunctive representation or attention over the queue, which makes cross-job comparison native. That is the leading item in further work.

## Exploration

Exploration is $`\epsilon`$-greedy and uniform over the legal actions, with $`\epsilon`$ decaying from 1.0 to 0.05 over the first 30% of training and held there. Sampling from the legal set rather than all 51 outputs matters at this action size. Under a typical mask most of the full space is illegal, and uniform sampling would spend the budget on actions the environment rejects. We conducted no hyperparameter search, so the schedule is the standard DQN default. The evidence that it sufficed is the return curve in §5.1, flat over the last 600,000 steps at $`\epsilon`$ = 0.05.

# Limitations and deployment

The state is only partially observed: just the ten most urgent queued jobs are visible, and the arrival times of unreleased jobs are absent. A recurrent encoder, or a short observation history, is the extension for this.

The masked no-op means the policy cannot hold a fast machine for an urgent job arriving imminently. Potential-based shaping would let the no-op be retained under the present reward, and is listed under further work.

The instance distribution is synthetic and single-operation, whereas real job shops have multi-operation jobs with precedence constraints, sequence-dependent setups, breakdowns and non-stationary arrivals. Transfer to those settings is outside the scope of this study, and the load parameters were chosen so that the rules separate. With three seeds, differences are compared against the seed spread.

For deployment, inference is one forward pass through a small network, fast enough for a dispatch loop. The mask is enforced by the environment, so an out-of-distribution observation cannot produce an illegal schedule, and the worst case is a suboptimal legal choice rather than an invalid one. Under Formulation B every action is a named dispatching rule, so an operator can audit each decision. However, the policy is specific to its training arrival distribution, so a shift in load requires retraining.

# Conclusion and further work

We formulated dynamic job scheduling on heterogeneous parallel machines as a masked-action MDP and trained a Dueling DQN on it under two action formulations. The comparison ran on held-out instances under a protocol fixed in advance, against three required dispatching rules and the best of eight. The rule formulation with three-step returns beats all three required baselines in return on every seed, and scores 0.109 below the strongest rule in its own action set.

Three validation checks, a uniform-random legal policy, a seed-band assertion and a loss-mask test, fixed the action space, the instance split and the update before any result was accepted, and each is retained in the test suite.

Further work, in order of expected value: replace the fixed ten-slot window with a permutation-invariant encoder over the queue; retain the no-op under potential-based shaping; evaluate `double_q` as the D3QN combination standard in the literature; and test transfer across untrained arrival rates.

<div class="center">

------------------------------------------------------------------------

</div>

# Validation record

Each of the three checks in §6.3 is set out here with the measurement and the test that now covers it. A fourth entry records the load sweep that set the environment constants.

## Instance overlap

`training_instance_seed` used `1000 + seed `$`\times`$` 100000` as a band start, which placed seeds 1 and 2 at 101,000 and 201,000, both above the held-out range beginning at 9000, because the intended wraparound never triggered. We now partition instances into disjoint bands of 3000 below `EVAL_SEED_START`, with an in-function assertion. Two tests in `tests/test_env_api.py` cover it: one asserts that no training instance enters the held-out range, the other that each seed draws its own instances.

## Loss mask

`compute_loss` passed `torch.ones_like(next_mask)` as the current-state mask. Because the dueling aggregation subtracts the mean advantage over legal actions, this optimised a different function from the one `select_action` evaluated, and waiting time rose from 7.30 to 17.28 over a 20,000-step run. The replay buffer now stores the current mask, and a test in `tests/test_mask.py` fails if the predicted `Q(s,a)` is computed with any mask other than the current one.

## Load sweep

At $`\lambda`$` = 0.55` the makespan of 97.95 was within 8% of the arrival-bound makespan of 90.9, and First-Come-First-Served and Shortest-Job-First differed by 0.06 in average waiting time. The load sweep and the resulting choice of $`\lambda`$` = 1.0` are in `docs/mdp_spec.md` §10, and the check is retained as `scripts/check_load.py`.

## No-op stalling

The action space originally included a no-op. At 300,000 steps the resulting policy reached an average waiting time of 7.95, against 6.61 for uniform selection among legal actions, and chose the no-op at 46.3% of decision epochs against 0.7% for an untrained network.

Dispatching at an epoch that remains a decision epoch advances the clock by $`\Delta`$`t = 0` and returns reward 0, while every action that advances time returns a negative reward. Under discounting, an agent facing negative rewards improves its return by deferring them, and the no-op makes deferral available. Masking it reduced waiting time to 4.74.

# Reproduction

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    pip install -e .
    bash scripts/run_all.sh

Seeds, exact dependency versions and the full hyperparameter table are in `docs/hyperparameters.md`. Raw logs backing every figure are committed under `logs/`.
