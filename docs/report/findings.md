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

## 4. A checkpoint stopped matching its own log

`aggregate.json` records per-seed held-out waiting times of [4.716, 4.658, 4.738] and the training
log for seed 1 shows a final-500-episode return of −2.003, in line with seeds 0 and 2. Both are
consistent, and the reported headline of 4.704 ± 0.034 is traceable to them.

The saved weights for seed 1 were later overwritten by a training run launched without `--out-dir`,
which defaults to the repository root. Evaluating the checkpoint on disk gives 24.595 — nothing like
the log it is supposed to accompany.

**Consequence:** the reported numbers are correct, but the submitted artefacts were internally
inconsistent. A marker re-running `evaluate.py` against the committed checkpoints would not have
reproduced the committed `aggregate.json`, which is exactly the failure the reproducibility
criterion is testing for.

**Found by:** evaluating each seed separately after an aggregate came out with an implausible spread
(±9.3 where the committed value was ±0.03). A single aggregated number would have hidden it.

**Fix:** seeds retrained so the committed weights, logs and aggregate all agree, and
`scripts/train.py` now refuses to write into a directory that already holds a committed run unless
`--overwrite` is passed.

## 5. Prioritised replay did not help, against the literature's recommendation

`docs/report/literature_review.md` recommended prioritised experience replay as the highest-value
next change, on the grounds that it appears in the recipe of the closest published precedent
(Han & Yang, 2020, dueling double DQN **with prioritised replay**) and in Liu et al. (2025). The
review also flagged that **no paper in the verified corpus isolates PER's contribution on a
scheduling problem** — the canonical ablation (Hessel et al., 2018) is on Atari.

The ablation says PER does not help here. Three seeds, one million steps, rule action space,
held-out evaluation, cumulative reward:

| Variant | Per-seed return | Mean | s.d. |
|---|---|---|---|
| rules, uniform replay | −1.334, −1.314, −1.247 | −1.298 | 0.037 |
| rules + PER | −1.309, −1.333, −1.294 | −1.312 | 0.016 |
| rules + n-step 3 | −1.222, −1.212, −1.269 | **−1.235** | 0.025 |
| rules + PER + n-step 3 | −1.315, −1.282, −1.263 | −1.287 | 0.021 |

- **PER against uniform replay:** difference −0.014 against a combined spread of 0.053. Within the
  spread, and no per-seed dominance. **No measurable effect.**
- **n-step 3 against uniform replay:** difference +0.064 against a combined spread of 0.062.
  Exceeds the spread, and n-step wins on **every** seed. A real improvement.
- **PER added on top of n-step:** difference −0.052 against a combined spread of 0.046. Adding PER
  to n-step *degrades* the result.

### What this does and does not license us to say

It does **not** refute Han and Yang. Their setting differs on three axes that plausibly matter:
they use a CNN over multi-channel image states, they solve 85 static OR-Library instances rather
than a stochastic-arrival stream, and their PER sits inside a dueling *double* DQN. Any of those
could change the value of prioritising by TD error.

It does license a narrow, defensible claim: **on this problem, with this state representation and
this action space, prioritised replay contributes nothing measurable and n-step returns contribute
a real improvement.** That is a genuine, if small, contribution precisely because the review found
no scheduling-domain ablation isolating PER.

The mechanism is consistent with the diagnosis. Our reward is 0 at the instant a dispatch is
committed and its cost appears tens of decisions later. n-step returns attack that directly by
propagating the delayed consequence to the causing action in one update. PER changes *which*
transitions are replayed, not how far credit travels, so it does not address the binding constraint.
Prioritising by TD error may even concentrate replay on high-variance transitions near episode
boundaries, which would explain the degradation when combined with n-step.

**Method note for the report:** with three seeds we compare a difference against the combined
seed spread and check per-seed dominance. We do not run a significance test; three samples do not
support one.
