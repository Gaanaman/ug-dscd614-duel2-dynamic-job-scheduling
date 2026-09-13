# Declaration of Generative Artificial Intelligence Use

Group 11 (Daniel K. Adotey, Kyeremeh Faithful, Caleb Abakah Mensah), Option DUEL-2, DSCD 614
Reinforcement Learning, University of Ghana.

## Scope of the work

The group designed the problem formulation, chose the algorithm and the baselines, wrote the
experimental protocol before any result was produced, ran every experiment, read every result
and wrote the report. Each member owned the modules listed in the repository README and reviewed
the other members' pull requests.

## Tools used

| Tool | Dates | Used by | Purpose |
|---|---|---|---|
| Claude (Anthropic), Claude Code | 27 August to 13 September 2026 | Daniel | Code completion and debugging in the environment, agent and evaluation modules; literature search; language editing of the report |
| ChatGPT (OpenAI), Codex | 1 to 4 September 2026 | Faithful | Repository audit, a submission-artifact validator, Git conflict resolution and PDF build diagnosis |

## Verification

The group checked every tool output before use. Code was accepted only after the test suite
passed (43 tests), including tests that fail if action masking is removed from the loss or the
bootstrap target. The waiting-time term of the reward is asserted numerically against the
evaluation metric on a full episode. Every number in the report traces to a committed log under
`logs/`, and every figure regenerates from those logs. Cited works were read in full where the
text was available; the remainder are cited on the bibliographic record.

## Statement

Generative AI was used as set out here and nowhere else. The submitted work is the group's own.
The group accepts responsibility for the correctness of everything in it, including anything a
tool contributed.

Daniel K. Adotey · Kyeremeh Faithful · Caleb Abakah Mensah
13 September 2026
