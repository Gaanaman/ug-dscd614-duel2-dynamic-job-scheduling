# References — annotated

The fourteen works cited in the report, in the order of its reference list, with what each is
cited for and the basis on which it was checked. The reference list itself is §9 of
`docs/report/report.md`; the BibTeX records are `paper/references.bib`. Full texts held locally are
in `corpus/`, which is not committed.

| Key | Reference | Cited for | Checked against |
|---|---|---|---|
| `hanyang2020` | Han, B.-A., & Yang, J.-J. (2020). Research on adaptive job shop scheduling problems based on dueling double DQN. *IEEE Access*, 8, 186474–186495. https://doi.org/10.1109/ACCESS.2020.3029868 | Closest precedent. Dispatching rules as the action space; multi-channel image state into a CNN; dueling double DQN with prioritised replay; 85 OR-Library instances; benchmark against any single heuristic rule, for large-scale problems, and comparable to a genetic algorithm. | Full text, 22 pages, open access CC-BY. Every quoted phrase located in the body. |
| `hessel2018rainbow` | Hessel, M., et al. (2018). Rainbow: Combining improvements in deep reinforcement learning. *AAAI*, 32(1). https://doi.org/10.1609/aaai.v32i1.11796 | n-step returns as one of the Rainbow components; the mechanism by which a delayed consequence reaches the causing action in one update. | Full text (arXiv 1710.02298). |
| `huang2022cleanrl` | Huang, S., et al. (2022). CleanRL: High-quality single-file implementations of deep reinforcement learning algorithms. *JMLR*, 23(274), 1–18. | Structural reference for the training loop, written out rather than imported. Attributed at the point of use in `src/duel2/agent.py` and in `docs/attribution.md`. | Bibliographic record (JMLR). The adapted file is `cleanrl/dqn.py`, MIT licence. |
| `liu2025ddqnper` | Liu, C., Chen, K., Wang, H., Yang, B., & Leng, J. (2025). Job shop scheduling by deep dual-Q network with prioritized experience replay for resilient production control in flexible manufacturing system. *Computers & Operations Research*, 183, 107190. https://doi.org/10.1016/j.cor.2025.107190 | A second scheduling-domain use of prioritised replay in a Q-learning recipe. | Bibliographic record (Crossref). The title carries the claim; the body was not read. |
| `lv2025review` | Lv, L., Zhang, C., Fan, J., & Shen, W. (2025). Deep reinforcement learning for job shop scheduling problems: A comprehensive literature review. *Knowledge-Based Systems*, 321, 113633. https://doi.org/10.1016/j.knosys.2025.113633 | Survey of the wider field. | Bibliographic record (Crossref). |
| `mnih2015` | Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529–533. https://doi.org/10.1038/nature14236 | The DQN recipe: replay buffer, target network, ε-greedy. | Bibliographic record. The description in §2.1 is the canonical one. |
| `schaul2016per` | Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2016). Prioritized experience replay. *ICLR*. https://arxiv.org/abs/1511.05952 | Sampling in proportion to TD error with importance-sampling weights annealed to 1. | Full text (arXiv 1511.05952). |
| `smit2024gnn` | Smit, I. G., et al. (2024). Graph neural networks for job shop scheduling problems: A survey. https://arxiv.org/abs/2406.14096 | Survey of structured (graph) state encoders for scheduling. | Full text (arXiv 2406.14096). |
| `towers2023gymnasium` | Towers, M., et al. (2023). *Gymnasium*. https://gymnasium.farama.org | The environment API and `check_env`. | Software; version 1.0.0 pinned in `requirements.txt`. |
| `vanhasselt2016` | van Hasselt, H., Guez, A., & Silver, D. (2016). Deep reinforcement learning with double Q-learning. *AAAI*, 30(1). https://doi.org/10.1609/aaai.v30i1.10295 | The double-Q target behind the `double_q` flag, off in every reported run. | Full text (arXiv 1509.06461). |
| `wang2016dueling` | Wang, Z., Schaul, T., Hessel, M., van Hasselt, H., Lanctot, M., & de Freitas, N. (2016). Dueling network architectures for deep reinforcement learning. *ICML*, 48, 1995–2003. https://arxiv.org/abs/1511.06581 | The `Q = V + (A − mean A)` decomposition and the identifiability argument for subtracting the mean. The paper defines the head over the full action set; restricting the mean to legal actions is this project's adaptation and is not attributed to them. | Full text (arXiv 1511.06581). |
| `zhang2020l2d` | Zhang, C., Song, W., Cao, Z., Zhang, J., Tan, P. S., & Xu, C. (2020). Learning to dispatch for job shop scheduling via deep reinforcement learning. *NeurIPS*, 33. https://arxiv.org/abs/2010.12367 | GNN over the disjunctive graph giving a size-agnostic policy. | Full text (arXiv 2010.12367). |
| `zhang2024shaping` | Zhang, L., Yan, Y., Yang, C., & Hu, Y. (2024). Dynamic flexible job-shop scheduling by multi-agent reinforcement learning with reward-shaping. *Advanced Engineering Informatics*, 62, 102872. https://doi.org/10.1016/j.aei.2024.102872 | Reward shaping for dynamic flexible job-shop scheduling with random arrivals. | Bibliographic record (Crossref). Cited for what the title states. |
| `zhang2025dueling` | Zhang, Z.-Q., Wu, Z.-M., Qian, B., & Hu, R. (2025). A reward-shaping dueling distributed multi-agent deep reinforcement learning framework for dynamic flexible job shop scheduling with random job arrivals. *Expert Systems with Applications*, 297, 128951. https://doi.org/10.1016/j.eswa.2025.128951 | Reward shaping combined with a dueling architecture in the random-arrival setting. | Bibliographic record (Crossref). Cited for what the title states. |

## Basis of the checks

Seven works are held in full text and every claim attributed to them was located in the body:
Han and Yang (2020), Hessel et al. (2018), Schaul et al. (2016), Smit et al. (2024), van Hasselt
et al. (2016), Wang et al. (2016) and Zhang et al. (2020). Han and Yang is gold open access under
CC-BY; the earlier note that it sat behind a paywall was wrong, and its full text was obtained on
11 September 2026.

Seven are cited on the strength of a verified bibliographic record and, where the report attributes
a specific claim, on the claim the title itself states: Huang et al. (2022), Liu et al. (2025), Lv
et al. (2025), Mnih et al. (2015), Towers et al. (2023), and the two reward-shaping papers. No
number in the report is taken from any of them.

One correction followed from reading Han and Yang in full. Their headline claim is that the
method "performs better than any single heuristic rule *for large scale problems*", with optimal
solutions on the small instances; the qualifier was missing from an earlier draft and is now in
§2.3 and in `docs/mdp_spec.md`. The paper does not describe action masking, and an earlier
docstring that credited it with output-layer masking has been corrected.

## Departure from the literature

Han and Yang (2020) and most recent work combine dueling with double Q-learning (D3QN). The brief
binds this project to Dueling DQN, so the headline run uses dueling alone and `double_q` is exposed
as a configuration flag. §2.3 of the report states this.
