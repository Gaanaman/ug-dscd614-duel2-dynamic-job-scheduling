# GitHub Plan

12 marks for repository and reproducibility, plus the commit history is one of only two
sources the examiner uses to moderate individual marks by up to ±20%. The repository is
therefore not a delivery mechanism — it is assessed evidence.

## Hard constraints from the instructions

| Rule | Consequence of breaking it |
|---|---|
| Public GitHub repository, no other hosting | non-submission |
| Public and **unmodified** until marks are released | a repo made private, deleted or **force-pushed** after the deadline is treated as non-submission |
| Final commit SHA recorded in `Submission_Links.txt` | later commits are ignored; the recorded SHA is what is marked |
| Commit throughout the fourteen days | a repository with a single commit does not evidence individual contribution |
| No large datasets committed | provide a generator script or documented instructions instead |
| Raw logs used to produce the figures committed | untraceable figures are not credited |

## Visibility: private now, public on submission day

The rules require the repository to be public **at the point of submission** and to stay public
and unmodified until marks are released. They do not require it to be public while the work is in
progress. It is therefore created **private** and flipped to public on 4 September.

The reason is that collusion between groups and submission of another group's code are examination
offences, and a public repository during the exam window puts the environment design, reward
weights and evaluation harness in reach of the other groups for eight days. Nothing is gained by
that exposure; the marks come from the repository being public when it is read, not from it having
always been so.

The cost is one command on submission day, which is on the checklist and in the freeze procedure
below. **If it is missed, the submission does not count.**

## Setup

```bash
cd duel2-dynamic-job-scheduling
git init -b main
git add .
git commit -m "chore: project scaffold, MDP spec and experimental protocol"
gh repo create ug-dscd614-duel2-dynamic-job-scheduling --private --source=. --remote=origin --push
```

Then, in the repository settings:

- **Add both teammates as collaborators** with write access. On a private repository this is the
  only way they get access at all, so do it first.
- **Protect `main`**: require a pull request before merging, and **block force pushes**. A force
  push after the deadline is treated as non-submission, so the setting removes the possibility
  rather than relying on nobody doing it.
- Leave Actions enabled — `.github/workflows/ci.yml` runs the tests on every push and gives the
  examiner visible evidence that the repository is exercised. Private repositories on a free
  account include Actions minutes; this workflow uses a negligible number of them.

## Every member commits under their own identity

Moderation reads the commit history. A commit authored under the wrong email attributes someone
else's work. Each member runs this once, in the repository:

```bash
git config user.name "Full Name"
git config user.email "the email attached to your GitHub account"
```

Verify with `git log --format='%an <%ae>'` before the deadline. Commits whose email is not linked
to a GitHub account do not appear in the contributor graph at all.

## Branching

`main` stays green. Work happens on short-lived branches named for the module, which maps
one-to-one onto the ownership table in `roles_and_split.md`:

```
env/observation-vector        env/action-mask        env/job-generator
agent/dueling-network         agent/masked-target    agent/training-loop
eval/metrics                  eval/harness           eval/baselines
docs/mdp-spec                 docs/report-section-3
```

Open a pull request, have **another member review it**, then merge. The review trail is a second
independent record of participation, and it catches the mask bugs.

## Commit messages

Conventional prefixes — `feat:`, `fix:`, `test:`, `docs:`, `chore:`, `exp:` — with a body that
says *why* when the change is a design decision. `exp:` is reserved for committing run logs, so
the experimental history is greppable:

```
exp: training run, 3 seeds, 500k steps, reward weights a=1.0 b=0.3 g=1.0 d=2.0
```

## Cadence

Commit at least once per working day, per member. Small and frequent beats one large commit at
the end — the graph is what is being read, and eight days of daily activity from three people is
the picture the moderation is looking for. Push run logs the same day the run finishes.

## Issues and board

Create one issue per row of the eight-day plan, assigned to its owner, with the day as a label.
It costs twenty minutes and makes the plan visible to the examiner as well as to the group.

## Freeze and submit

On 4 September, before 23:59 GMT:

```bash
pytest                                    # green
bash scripts/run_all.sh                   # reproduces from a clean checkout
git add logs/ figures/ models/
git commit -m "exp: final run logs, figures and model weights"
git tag -a v1.0-submission -m "DSCD 614 final submission, Group 11, DUEL-2"
git push origin main --tags
git rev-parse HEAD                        # <- this SHA goes in Submission_Links.txt

# THE STEP THAT ENDS THE PROJECT IF IT IS MISSED
gh repo edit --visibility public --accept-visibility-change-consequences
```

Then verify in a **signed-out browser** that the repository URL opens, and that the YouTube link
plays. Both are explicit checklist items and both fail silently for the person who is logged in.
The signed-out check is what actually confirms the visibility flip took effect — do not take the
command's exit status as proof.

After that commit: no pushes, no force pushes, no deletions, no visibility changes, until marks
are released.
