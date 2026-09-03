# Declaration of Generative Artificial Intelligence Use

Required by the examination instructions (Part C, `AI_Use_Declaration.pdf`; Part F rule 9).
Generative AI is permitted for code assistance, debugging, literature searching and language
editing. Its use must be declared, stating which tools were used, for which parts of the work, and
how the output was verified. The group remains responsible for the correctness of everything
submitted; an error introduced by a tool is the group's error.

**This draft was prepared from the working record. Each member must check the rows covering their
own modules, correct anything inaccurate, and sign before submission.**

## Tools used

| Tool | Version / access date | Used by | Used for |
|---|---|---|---|
| Claude (Anthropic), via Claude Code | 27 August – 2 September 2026 | Daniel | Code assistance across the environment, agent, harness and analysis scripts; debugging; literature search; drafting and editing prose |
| | | | *(add any tool used by Faithful or Caleb)* |

Full names and student identification numbers appear on the signed PDF submitted through Sakai.
This repository is public and carries first names only.

## Parts of the work

| Component | AI involvement | How the output was verified |
|---|---|---|
| MDP formulation | Drafted with AI assistance; state design, reward decomposition and the Markov analysis were reviewed and amended by the group | The reward's waiting term is asserted numerically against total waiting time on a full episode (`test_waiting_term_telescopes_to_total_waiting_time`). The spec was corrected against the implementation when the two disagreed on no-op validity. |
| Environment implementation | Written with AI assistance | `gymnasium.utils.env_checker.check_env` passes. Eight further tests cover observation bounds, seed reproducibility, termination against truncation, and monotonic simulated time. |
| Reward function | Written with AI assistance | Five unit tests, including the telescoping identity and a hand-computed interval. |
| Action masking | Written with AI assistance | Nine tests, including three that fail if the mask is removed from the bootstrap target, from the current-state loss input, or from the dueling mean. |
| Dueling network and training loop | Written with AI assistance, structured after the CleanRL reference (attributed in `docs/attribution.md` and in the source file) | Masked forward pass verified against a hand-set advantage bias. Training verified end to end against a uniform-random legal policy as a floor. |
| Baselines | Written with AI assistance | Run through the same harness and metric code as the agent; results cross-checked against expectation (Shortest-Job-First lowest waiting time, Round Robin lowest weighted tardiness). |
| Evaluation harness and metrics | Written with AI assistance | Metrics verified against a three-job, two-machine schedule computed by hand in the test docstring. Held-out seed range asserted in code. |
| Plotting and analysis | Written with AI assistance | Figures regenerate from committed logs only; the figure script cannot step the environment. |
| Literature search | AI-assisted search for prior work on dueling DQN in job-shop scheduling and on action masking in value-based RL | **Every citation in `docs/report/references.md` must be opened and confirmed by a group member before submission.** Search results are a starting point, not a verified bibliography. |
| Report prose | Drafted with AI assistance from the group's own results and working notes | All numbers traced to `logs/`; no figure or number appears that is not in a committed log. |

## Verification statement

Verification was not a review pass at the end. It was continuous, and it caught four faults that
produced no error message:

1. **Training instance seeds overlapped the held-out evaluation range** for two of three seeds.
   Found by printing the seed function's output rather than trusting it. Now covered by two tests
   and an in-function assertion.
2. **The training loss used an all-ones current-state mask**, optimising a different function from
   the one the behaviour policy evaluated. Found because a smoke run got worse with training, not
   better. Now covered by a test.
3. **The agent learned to stall**, selecting the no-op at 46.3% of decision epochs. Found by adding
   a uniform-random legal policy as a diagnostic floor and observing that the trained agent lost to
   it. Resolved by a documented two-arm ablation.
4. **The first environment configuration had no headroom** — makespan was set by the arrival
   process, not the scheduler. Found by a load-separation check run before any training. Retained
   as `scripts/check_load.py`.

Every claim in the report that rests on a number is traceable to a file under `logs/`, and every
figure is regenerated from those logs by a script that cannot access the environment.

## Statement

*(To be signed by all three members once the table above has been checked.)*

We confirm that the work submitted is our own, that generative AI was used as declared above, that
we have verified the outputs as described, and that we accept responsibility for the correctness of
everything submitted.

Signed by all three members on the submitted PDF. Names, student identification numbers and
signatures are on that document, not in this repository.
