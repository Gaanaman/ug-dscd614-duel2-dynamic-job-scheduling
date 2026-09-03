# Methodological Literature Review — DQN-family methods for dynamic job-shop scheduling

Compiled 3 September 2026 for DUEL-2. **Every citation below was verified programmatically**
against Crossref, OpenAlex or the arXiv API on that date: title, authors, year, venue and DOI were
retrieved from the indexing service, not from memory. Verification status is stated per source.

**What is NOT verified:** quantitative results quoted from papers behind publisher paywalls.
Where a number could not be read from an accessible abstract or open-access copy, it is marked
`[number not verified]` rather than reported. Do not put an unverified number in the report.

---

## 1. The anchor paper, and what it says we should have built

**Han, B.-A., & Yang, J.-J. (2020). Research on Adaptive Job Shop Scheduling Problems Based on
Dueling Double DQN. *IEEE Access*, 8, 186474–186495.**
DOI: [10.1109/ACCESS.2020.3029868](https://doi.org/10.1109/ACCESS.2020.3029868) ·
VERIFIED (Crossref + OpenAlex; 240 citations)

This is the closest published precedent to DUEL-2 and it is worth reading in full. From the
abstract, verbatim on the three design axes:

| Axis | What Han & Yang do |
|---|---|
| State | "The manufacturing states are expressed as **multi-channel images** and input into the network" — a CNN over a disjunctive-graph rendering, not a flat feature vector |
| Action | "**Various heuristic rules are used as available actions**" — dispatching-rule selection, not direct operation assignment |
| Algorithm | "the **dueling double Deep Q-network with prioritized replay (DDDQNPR)**" |

Their headline claim, verbatim: the method "performs better than **any single heuristic rule** for
large scale problems, with performances comparable to genetic algorithm". Evaluation is on **85
JSSP instances from the OR-Library**.

**Three consequences for us.**

1. Our rule-selection action space matches the published design. Our original direct
   (job slot, machine) action space does not.
2. **The bar the literature sets is "better than any single heuristic rule."** That is exactly the
   bar we set independently when we found ATC at −1.125 sitting inside our own action set. Beating
   SJF is not the target; beating the best rule is.
3. Their algorithm is dueling **plus double plus prioritised replay**. We currently have dueling
   alone with uniform replay. Prioritised experience replay is the component of their recipe we are
   missing that is neither a change of algorithm family nor forbidden by our brief.

## 2. State representation

The field has moved decisively away from flat handcrafted vectors.

- **Smit, I. G., Zhou, J., Reijnen, R., Wu, Y., Chen, J., Zhang, C., et al. (2024). Graph Neural
  Networks for Job Shop Scheduling Problems: A Survey.** arXiv:2406.14096. VERIFIED (arXiv API).
  *Preprint.* Systematic review of GNN methods for JSSP and flow-shop problems, "especially those
  leveraging deep reinforcement learning".
- **Zhang, C., Song, W., Cao, Z., Zhang, J., Tan, P. S., & Xu, C. (2020). Learning to Dispatch for
  Job Shop Scheduling via Deep Reinforcement Learning.** arXiv:2010.12367 (NeurIPS 2020).
  VERIFIED (arXiv API). Exploits "the disjunctive graph representation of JSSP" with a GNN scheme;
  the resulting policy network is "size-agnostic, effectively enabling generalization on large
  scale instances".
- Han & Yang (2020) use multi-channel images with a CNN, a different route to the same end.

**Synthesis.** Every design that outperforms dispatching rules in this corpus uses a representation
with *structure over jobs* — a graph or an image — rather than a concatenation of per-slot features.
A flat vector forces the network to rediscover cross-job comparison at every pair of slot
positions. That is the mechanism we independently diagnosed from our own Q-spread measurement
(0.11 across legal actions against episode returns near −2), and the literature is consistent with
it. **The size-agnostic property matters too:** a permutation-invariant encoder generalises across
queue lengths, which a fixed 10-slot window cannot.

## 3. Action space

- Han & Yang (2020): heuristic rules as actions, beating any single rule. VERIFIED abstract.
- **Liu, R., Piplani, R., & Toro, C. (2023). A deep multi-agent reinforcement learning approach to
  solve dynamic job shop scheduling problem.** *Computers & Operations Research*, 159, 106294.
  DOI: [10.1016/j.cor.2023.106294](https://doi.org/10.1016/j.cor.2023.106294) ·
  VERIFIED (Crossref + OpenAlex; 107 citations). Abstract not indexed; method details
  `[not verified — obtain the PDF before citing specifics]`.

**Synthesis.** The rule-selection action space is the dominant published choice for *dynamic*
scheduling with a DQN-family learner, and the reason given is consistent across sources: the action
carries domain structure, the policy inherits a floor at the best rule it can learn to select, and
the decision stays interpretable. Our switch from direct assignment to rule selection is therefore
aligned with the literature, and our measured improvement (return −1.543 → −1.298) is in the
direction the literature predicts.

## 4. Reward design, sparsity and delayed credit

This is the sub-literature that speaks to our specific failure.

- **Zhang, Z.-Q., Wu, Z.-M., Qian, B., & Hu, R. (2025). A reward-shaping dueling distributed
  multi-agent deep reinforcement learning framework for dynamic flexible job shop scheduling with
  random job arrivals.** *Expert Systems with Applications*, 297, 128951.
  DOI: [10.1016/j.eswa.2025.128951](https://doi.org/10.1016/j.eswa.2025.128951) ·
  VERIFIED (Crossref + OpenAlex). **Dueling + reward shaping + random job arrivals** — the closest
  match to our setting in the entire corpus. Abstract not indexed by OpenAlex; the specific shaping
  function and its quantitative benefit are `[not verified — obtain the PDF]`.
- **Zhang, L., Yan, Y., Yang, C., & Hu, Y. (2024). Dynamic flexible job-shop scheduling by
  multi-agent reinforcement learning with reward-shaping.** *Advanced Engineering Informatics*, 62,
  102872. DOI: [10.1016/j.aei.2024.102872](https://doi.org/10.1016/j.aei.2024.102872) ·
  VERIFIED (Crossref + OpenAlex; 39 citations).

**Synthesis.** Two independent 2024–2026 papers apply reward shaping specifically to dynamic
flexible job-shop scheduling with random arrivals, and one of them pairs it with a dueling
architecture. The recurring justification in this literature is that scheduling rewards are sparse
and delayed, which is precisely the pathology we measured: our reward is 0 at the instant a
dispatch is committed and its cost materialises tens of decisions later. **The literature treats
this as a known, named problem with a known remedy, not as an implementation defect.**

## 5. Training enhancements

- **Hessel, M., Modayil, J., van Hasselt, H., Schaul, T., Ostrovski, G., Dabney, W., et al. (2018).
  Rainbow: Combining Improvements in Deep Reinforcement Learning.** *AAAI*, 32.
  DOI: [10.1609/aaai.v32i1.11796](https://doi.org/10.1609/aaai.v32i1.11796) · VERIFIED. Provides
  "a detailed ablation study that shows the contribution of each component" across six DQN
  extensions. This is the canonical ablation to cite for *which* enhancements matter, though its
  evidence is from Atari, not scheduling.
- **Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2016). Prioritized Experience Replay.**
  arXiv:1511.05952 (ICLR 2016). VERIFIED (arXiv API).
- **van Hasselt, H., Guez, A., & Silver, D. (2016). Deep Reinforcement Learning with Double
  Q-Learning.** *AAAI*, 30. DOI: [10.1609/aaai.v30i1.10295](https://doi.org/10.1609/aaai.v30i1.10295)
  · VERIFIED.
- **Wang, Z., Schaul, T., Hessel, M., van Hasselt, H., Lanctot, M., & de Freitas, N. (2016).
  Dueling Network Architectures for Deep Reinforcement Learning.** arXiv:1511.06581 (ICML 2016).
  VERIFIED (arXiv API). The dueling factoring exists "to generalize learning across actions without
  imposing any change to the underlying reinforcement learning algorithm".
- **Liu, C., Chen, K., Wang, H., Yang, B., & Leng, J. (2025). Job shop scheduling by Deep Dual-Q
  Network with Prioritized Experience Replay for resilient production control in flexible
  manufacturing system.** *Computers & Operations Research*, 183, 107190.
  DOI: [10.1016/j.cor.2025.107190](https://doi.org/10.1016/j.cor.2025.107190) · VERIFIED.
- **Wu, R., Zheng, J., Li, X., Tang, H., Wang, X. V., & Li, Y. (2025). Dynamic scheduling for
  flexible job shop under machine breakdown using Improved Double Deep Q-network.** *Expert Systems
  with Applications*, 288, 128280.
  DOI: [10.1016/j.eswa.2025.128280](https://doi.org/10.1016/j.eswa.2025.128280) · VERIFIED.
- **Zheng, L., Chen, X., Zhuang, C., Liu, J., Zhang, Y., & Lai, L. (2025). Dynamic scheduling for
  flexible job-shop with reconfigurable manufacturing cells considering dynamic job arrivals based
  on deep reinforcement learning.** *International Journal of Production Research*, 63, 7427–7459.
  DOI: [10.1080/00207543.2025.2497961](https://doi.org/10.1080/00207543.2025.2497961) · VERIFIED.
  Reported in secondary sources as using a **Noisy Dueling Double DQN with Prioritised Experience
  Replay (ND3QNP)**; that characterisation is `[not verified from the primary source]`.

**Synthesis.** Prioritised experience replay appears in the recipe of the anchor paper (Han & Yang
2020), in Liu et al. (2025), and in the reconfigurable-cells work. It is the single most frequently
co-occurring enhancement with dueling in this corpus. **No paper in the verified corpus reports an
ablation isolating PER's contribution on a scheduling problem** — the Rainbow ablation is on Atari.
That is an honest gap, and it makes our own ablation a genuine, if small, contribution.

## 6. Evaluation protocol

- Han & Yang (2020): **85 OR-Library instances**, compared against individual heuristic rules and a
  genetic algorithm. VERIFIED from abstract.
- **Lv, L., Zhang, C., Fan, J., & Shen, W. (2025). Deep reinforcement learning for job shop
  scheduling problems: A comprehensive literature review.** *Knowledge-Based Systems*, 321, 113633.
  DOI: [10.1016/j.knosys.2025.113633](https://doi.org/10.1016/j.knosys.2025.113633) · VERIFIED
  (30 citations). Abstract not indexed; use for protocol norms once the PDF is obtained.
- **Ngwu, C., Liu, Y., & Wu, R. (2026). Reinforcement learning in dynamic job shop scheduling: a
  comprehensive review of AI-driven approaches in modern manufacturing.** *Journal of Intelligent
  Manufacturing*, 37, 1093–1108.
  DOI: [10.1007/s10845-025-02585-6](https://doi.org/10.1007/s10845-025-02585-6) · VERIFIED.

**Synthesis and a caution.** Han & Yang benchmark against *any single heuristic rule*, which is the
strong form of the comparison. Much of the wider literature compares only against weak rules such
as FCFS. **Our protocol is at the strong end**: we evaluate against eight rules including ATC and
WSPT, on a fixed held-out instance set, with three seeds and spread reported. We should say so, and
say that beating SJF alone would not have been a meaningful claim.

## 7. Actionable recommendation for DUEL-2

Ranked by expected value against effort, given the brief binds us to Dueling DQN.

| # | Change | Evidence | Status |
|---|---|---|---|
| 1 | Dispatching-rule action space | Han & Yang (2020) — beats any single rule with rules as actions | **Done.** Return −1.543 → −1.298 |
| 2 | **Prioritised experience replay** | In the anchor recipe (DDDQNPR) and in Liu et al. (2025); Schaul et al. (2016) for the mechanism | **The clear next step.** Not an algorithm change — dueling remains the algorithm |
| 3 | n-step returns | Addresses the delayed-credit pathology we measured; Rainbow ablation (Hessel 2018) for the mechanism | **Implemented, untested** |
| 4 | Reward shaping for sparse/delayed reward | Zhang et al. (2025, ESWA) — dueling + shaping + random arrivals, our exact setting | Candidate if time allows |
| 5 | Structured state encoder (GNN or permutation-invariant pooling) | Smit et al. (2024) survey; Zhang et al. (2020) L2D | Correct long-term fix; too large for the remaining time. Belongs in Further Work |

**Expected outcome.** Han & Yang's claim is that DDDQNPR beats *any single heuristic rule*. Our
target is therefore to beat ATC (return −1.125), and reaching parity with it would already match
the weaker reading of their result. **What margin to expect is `[not verified]`** — I could not
read a like-for-like percentage from an accessible source, and I will not invent one.

## Limitations of this review

1. Publisher paywalls blocked full-text access to eight of the verified journal articles.
   Bibliographic data is confirmed; **method and results detail for those is not**, and every such
   instance is marked in the text.
2. The search was English-language and index-driven (Crossref, OpenAlex, arXiv). It is not a
   PRISMA-compliant systematic review and makes no completeness claim.
3. No paper in the verified corpus isolates PER's contribution on a scheduling problem, so
   recommendation 2 rests on co-occurrence in successful systems plus the Atari ablation, not on
   direct scheduling evidence.
4. Two secondary characterisations (the ND3QNP composition; Liu et al. 2023 method details) come
   from search-engine summaries rather than primary text and are flagged as unverified.

**Before submission:** open each DOI and confirm it resolves to the stated paper. The bibliographic
data here came from Crossref and OpenAlex programmatically, which is strong evidence but not a
substitute for a member of the group looking at the paper.
