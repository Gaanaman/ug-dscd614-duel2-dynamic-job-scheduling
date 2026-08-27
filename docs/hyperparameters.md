# Hyperparameters and Seeds

This file is the source for `Hyperparameters_and_Seeds.pdf` in the submission ZIP, and for the
configuration table in the report. Every value that differs from the library default is marked
and justified — the instructions require it explicitly.

## Library and algorithm

| | |
|---|---|
| Library | *(record: CleanRL adaptation / Stable-Baselines3 / RLlib)* |
| Version | *(exact, from `pip freeze`)* |
| Algorithm class | Dueling DQN with action masking |
| Framework | PyTorch *(exact version)* |

## Environment configuration

| Parameter | Value | Note |
|---|---|---|
| machines `M` | 5 | |
| jobs per episode `N` | 50 | |
| visible job slots `K` | 10 | queue window; drives observation and action size |
| arrival rate `λ` | *(tbd)* | set so the system is loaded but stable |
| processing time distribution | *(tbd)* | |
| deadline tightness | *(tbd)* | |
| machine speeds `s_m` | *(tbd)* | identical or heterogeneous — see `mdp_spec.md` §10 |
| horizon `H` | *(tbd)* | time normalisation constant |
| observation dimension | 69 | `5K + 3M + 4` |
| action space | `Discrete(51)` | `K·M + 1` |
| `T_max` | `4N` | truncation limit |

## Reward weights

| Weight | Value | Term |
|---|---|---|
| `α` | 1.0 | queue waiting |
| `β` | 0.3 | machine idleness |
| `γ_c` | 1.0 | weighted completion |
| `δ` | 2.0 | weighted tardiness |

Log every change here with the date and the reason.

## Agent configuration

| Parameter | Value | Differs from default? |
|---|---|---|
| network | 2 × 256 hidden, ReLU trunk; value head 256→1; advantage head 256→51 | yes — dueling heads |
| optimiser | Adam | |
| learning rate | 1e-4 | |
| batch size | 128 | |
| replay capacity | 200,000 | |
| learning starts | 10,000 steps | |
| train frequency | every 4 steps | |
| target update interval | 1,000 steps | |
| `γ` | 0.99 | justified in `mdp_spec.md` §7 |
| `ε` schedule | 1.0 → 0.05 linear over first 30% of training | |
| total timesteps | *(tbd)* | reduce this before reducing seeds if compute is short |
| gradient clipping | 10.0 | |
| double Q target | *(decide)* | dueling and double are independent; state which is used |

## Seeds

| Purpose | Values |
|---|---|
| training seeds | 0, 1, 2 |
| training instance stream | `seed + 1000` |
| evaluation instance stream | 9000–9029 (held out, identical for every policy) |

## Hyperparameter search declaration

Record any search here: what was searched, over what range, on which seed, and whether it touched
the evaluation set. If no search was run, say so — an undeclared search is a reporting offence,
an absent one is simply a fact.
