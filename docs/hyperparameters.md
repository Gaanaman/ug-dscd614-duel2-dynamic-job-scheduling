# Hyperparameters and Seeds

This file is the source for `Hyperparameters_and_Seeds.pdf` in the submission ZIP, and for the
configuration table in the report. Every value that differs from the library default is marked
and justified — the instructions require it explicitly.

## Library and algorithm

| | |
|---|---|
| Library | None for the algorithm — the training loop is written out, structured after the CleanRL `dqn.py` reference (MIT). See `docs/attribution.md`. |
| Version | gymnasium 1.0.0 · torch 2.5.1 · numpy 2.1.3 (pinned in `requirements.txt`) |
| Algorithm class | Dueling DQN with action masking |
| Framework | PyTorch 2.5.1, CPU |

## Environment configuration

| Parameter | Value | Note |
|---|---|---|
| machines `M` | 5 | |
| machine speeds `s_m` | `[1.0, 1.0, 1.25, 0.8, 0.8]` | heterogeneous |
| jobs per episode `N` | 50 | |
| visible job slots `K` | 10 | queue window; drives observation and action size |
| arrival rate `λ` | 1.0 | `ρ = 1.24`; chosen by load sweep, see `mdp_spec.md` §10 |
| processing time | discrete uniform [2, 10] | mean 6 |
| priority weights | `{1: 0.6, 2: 0.3, 5: 0.1}` | |
| deadline | `a_j + p_j · U(1.3, 2.5)` | |
| horizon `H` | 120 | time normalisation constant |
| observation dimension | 69 | `5K + 3M + 4` |
| action space | `Discrete(51)` | `K·M + 1` |
| `T_max` | `4N = 200` | truncation limit |
| `allow_noop` | **false** | headline configuration; ablation in `mdp_spec.md` §4 |

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
| learning starts | 5,000 steps | |
| train frequency | every 4 steps | |
| target update interval | 1,000 steps | |
| `γ` | 0.99 | justified in `mdp_spec.md` §7 |
| `ε` schedule | 1.0 → 0.05 linear over first 30% of training | |
| total timesteps | 1,000,000 | ~11 min/seed on CPU; the 300k curve was still improving |
| gradient clipping | 10.0 | |
| double Q target | **off** | the brief binds the algorithm to Dueling DQN; exposed as a flag for the ablation |

## Seeds

| Purpose | Values |
|---|---|
| training seeds | 0, 1, 2 |
| training instance stream | `seed + 1000` |
| evaluation instance stream | 9000–9029 (held out, identical for every policy) |

## Hyperparameter search declaration

**No hyperparameter search was conducted.** Agent hyperparameters are standard DQN defaults and
were not tuned. Nothing was searched on the evaluation set.

Two configuration decisions *were* made by measurement, both on quantities that are not agent
hyperparameters, and both are reported with their evidence:

1. **Arrival rate `λ`**, swept over {0.55, 0.75, 1.0, 1.5, 2.5, 4.0} and selected by baseline
   separation, before any agent existed. Measured on held-out instances but with **no agent
   involved**, so no agent was tuned on the evaluation set. See `mdp_spec.md` §10.
2. **`allow_noop`**, decided by a two-arm ablation at 300,000 steps on seed 0. This *did* use
   held-out instances to compare two agents, and it is declared as such: it is a formulation
   decision informed by evaluation data, and the honest reading is that the headline configuration
   was selected with knowledge of held-out performance. Both arms are reported.
