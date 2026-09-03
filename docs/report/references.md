# References

Found via literature search on 2 September 2026. **Verify every one before submission** —
open the DOI, confirm authors, year, venue and page numbers. An examiner who finds a
fabricated or garbled citation will discount the rest. Rule 8 also requires every adapted
implementation to be attributed at the point of use.

## Algorithm foundations

- **Mnih, V., et al. (2015).** Human-level control through deep reinforcement learning.
  *Nature*, 518(7540), 529–533. — replay buffer, target network, ε-greedy. The recipe our
  training loop follows.
- **Wang, Z., Schaul, T., Hessel, M., van Hasselt, H., Lanctot, M., & de Freitas, N. (2016).**
  Dueling network architectures for deep reinforcement learning. *ICML*. — the
  `Q = V + (A − mean A)` decomposition and the identifiability argument for subtracting the
  mean. **The paper the project option is named after.**
- **van Hasselt, H., Guez, A., & Silver, D. (2016).** Deep reinforcement learning with double
  Q-learning. *AAAI*. — cited for the `double_q` flag, which is off in the headline run.

## Deep RL for job-shop scheduling

- **Han, B.-A., & Yang, J.-J. (2020).** Research on adaptive job shop scheduling problems
  based on dueling double DQN. *IEEE Access*, 8, 186474–186495. — closest prior work:
  dueling + double DQN for adaptive job-shop scheduling, and the output-layer action-masking
  mechanic. Cite in Background as the direct precedent.
- **Reinforcement learning in dynamic job shop scheduling: a comprehensive review of AI-driven
  approaches in modern manufacturing.** *Journal of Intelligent Manufacturing* (2025).
  <https://link.springer.com/article/10.1007/s10845-025-02585-6> — recent survey; use for the
  "prior work in the domain" requirement.
- **Deep Reinforcement Learning for Dynamic Flexible Job-Shop Scheduling with Automated Guided
  Vehicles** (2024). <https://link.springer.com/chapter/10.1007/978-981-97-0194-0_11> — D3QN
  reported to beat composite dispatching rules on tardiness. Supports our choice of dispatch
  rules as the baseline family.

## Action masking

- **A novel double Q-learning with invalid action masking for semiconductor ion implantation
  scheduling.** *Computers & Industrial Engineering* (2025).
  <https://www.sciencedirect.com/science/article/abs/pii/S0360835225007661> — double Q-learning
  with invalid action masking on a scheduling problem. Directly supports our design.
- **Policy-Based Reinforcement Learning with Action Masking for Dynamic Job Shop Scheduling
  under Uncertainty: Handling Random Arrivals and Machine Failures.** arXiv:2601.09293. —
  masking under random arrivals, the same dynamic setting as ours.
- **Gymnasium — Action Masking tutorial.**
  <https://gymnasium.farama.org/tutorials/training_agents/action_masking_taxi/> — the standard
  reference for the mechanic: assign large negative values to invalid actions at the output.

## Tooling

- **Towers, M., et al. (2023).** Gymnasium. <https://gymnasium.farama.org> — environment API.
- **Huang, S., et al. (2022).** CleanRL: High-quality single-file implementations of deep
  reinforcement learning algorithms. *JMLR*, 23(274). — structural reference for the training
  loop. **Attribute in `docs/attribution.md` and at the point of use.**
- **Paszke, A., et al. (2019).** PyTorch. *NeurIPS*.

## What the literature says that we depart from

Han and Yang (2020) and most recent work combine dueling with double Q-learning (D3QN). The
brief binds this project to **Dueling DQN**, so the headline run uses dueling alone and
`double_q` is exposed as a configuration flag for the ablation. State this explicitly in
Methodology rather than leaving a reader to wonder why the obvious enhancement is absent.
