# Findings log

Written as the work happened, so Discussion, Limitations and the AI-use declaration can be
grounded in what actually occurred rather than reconstructed at the end.

## Three failures found before they reached a result

### 1. Training instances overlapped the held-out evaluation range

`training_instance_seed(seed, episode)` used `1000 + seed * 100000` as a band start. For seeds 1
and 2 that is 101,000 and 201,000, both far above the held-out range at 9000, and the wraparound
that was supposed to keep it below never triggered correctly.

**Consequence had it shipped:** seeds 1 and 2 would have trained on the same instances they were
evaluated on. Every headline number would have been inflated, and the protocol section of the
report would have been false.

**Fix:** training instances are partitioned into disjoint bands of 3000, one per seed, all below
`EVAL_SEED_START`. An assertion fires inside the function, and two tests cover it
(`test_training_instances_never_enter_the_held_out_range`, `test_each_training_seed_gets_its_own_instances`).

**Found by:** printing the function's output for seed 2 rather than trusting it.

### 2. The loss used an all-ones mask for the current state

The dueling aggregation subtracts the mean advantage **over valid actions**. `compute_loss`
supplied `torch.ones_like(next_mask)` as the current-state mask, so `Q(s,a)` during the update was
computed with a mean over all 51 actions while `select_action` computed it with the real mask.

**Consequence:** the network was optimised for a different function from the one it acted with.
Nothing raised. Average waiting time degraded from 7.30 to 17.28 over a 20,000-step run — the
agent got measurably worse the longer it trained.

**Fix:** the replay buffer stores the current-state mask alongside the next-state mask, and both
are used. `test_training_uses_the_current_state_mask_for_predicted_q` fails if the mask is dropped.

**Found by:** the smoke run getting worse instead of better, and asking why.

### 3. The agent learned to stall

Documented in full in `docs/mdp_spec.md` section 4. The trained policy chose the no-op on 46.3% of
decision epochs against 0.7% for an untrained network, and scored worse than a uniform-random legal
policy.

**Found by:** adding a `RandomMasked` diagnostic floor. A learned policy that cannot beat random is
broken rather than undertrained, and that distinction decided whether the next step was more
compute or a bug hunt. It was neither — it was a formulation problem.

## What this says for the report

The Discussion section should lead with the random floor. It is cheap, it is not a required
baseline, and it converted an ambiguous "the agent is bad" into a specific, testable claim within
minutes. The same applies to the load check in `scripts/check_load.py`, which caught an instance
distribution where no scheduler could have mattered.

Three of the four substantive problems in this project were found by measuring something before
trusting it. None of them would have produced an error message.
