# Citation Audit — Project_Report.pdf

First pass 4 September 2026; closed 11 September 2026. Checked against the full texts held in
`corpus/` (nine papers) and the bibliographic records from Crossref, OpenAlex and the arXiv API.

## Verification summary

| | |
|---|---|
| Distinct in-text citations | 14 |
| Reference-list entries | 14 |
| Listed but never cited | 0 |
| Cited but not listed | 0 |
| Full three-way verification (text ↔ bibliographic record ↔ local full text) | 7 |
| Bibliographic record verified, claim carried by the title | 7 |

Seven citations carry a local full text and were checked phrase by phrase: Han and Yang (2020),
Hessel et al. (2018), Schaul et al. (2016), Smit et al. (2024), van Hasselt et al. (2016), Wang et
al. (2016), and Zhang et al. (2020). Every claim attributed to them was located in the source.

Seven have verified bibliographic records and are cited for what their titles state or for
canonical content: Huang et al. (2022), Liu et al. (2025), Lv et al. (2025), Mnih et al. (2015),
Towers et al. (2023), Zhang et al. (2024), and Zhang et al. (2025). No number in the report is
taken from any of them.

## Findings and their disposition

| Claim | Citation | Finding | Disposition |
|---|---|---|---|
| "performs better than any single heuristic rule" on 85 OR-Library instances | Han and Yang (2020) | The first pass could read only the abstract and recorded the paper as paywalled. It is gold open access (CC-BY); the full text was obtained on 11 September. The quoted phrase is accurate but the source qualifies it: "for large scale problems", with optimal solutions on the small instances. The report had dropped the qualifier. | Corrected. §2.3 and `docs/mdp_spec.md` now carry the qualifier. |
| Output-layer action masking "described by Wang et al. (2016) ... and applied to scheduling with masking by Han and Yang (2020)" | Wang et al. (2016); Han and Yang (2020) | Neither paper describes action masking. The term does not occur in either full text. The attribution was in a docstring (`src/duel2/network.py`) and in the working references file, not in the report. | Corrected. The docstring now states that restricting the dueling mean to legal actions is this project's adaptation. |
| `double_q` flag | van Hasselt et al. (2016) | Listed in §9 but not cited in the body: an uncited reference-list entry. | Corrected. Cited at the point in §4.3 where the flag is described. |
| — | Zhang, W., et al. (2025), *Swarm and Evolutionary Computation* | Listed in §9, never cited, not in `references.bib`. | Removed from §9. |
| Three design decisions follow from Han and Yang (2020): rule action space, structured state, single-rule benchmark | Han and Yang (2020) | Confirmed in the body. Action definition: "the action space consists of a variety of different heuristic rules". State: "the scheduling state was creatively represented as a multi-channel image". Comparison against individual rules and a genetic algorithm in Section VI. | No change. |
| Prioritised replay "following Han and Yang (2020) and Liu et al. (2025)" | Both | Han and Yang: DDDQNPR, confirmed in the body. Liu et al.: the title states prioritized experience replay; body not read. | No change. |
| DQN recipe | Mnih et al. (2015) | Canonical description, record verified. | No change. |
| Two reward-shaping studies in the random-arrival setting | Zhang et al. (2024); Zhang et al. (2025) | Both titles state reward shaping and dynamic flexible job-shop scheduling; the 2025 title states random job arrivals. The report claims no more than the titles support. | No change. |

## Checks that pass

- Every in-text key resolves to a reference-list entry, and every entry is cited.
- Author names and years match the retrieved records.
- All DOIs and URLs are well formed and use the `https://doi.org/` form.
- The two direct quotations from Han and Yang are marked as quotations, attributed in the
  sentence, and located verbatim in the full text (abstract, and Sections I and V).
- No statistic in the report is attributed to an external source. Every number in Results comes
  from this project's committed logs.

## Assessment

The one exposure the first pass identified is closed. Han and Yang carry the report's central
design justification and its benchmark; both hold in the full text, with a scope qualifier the
report now states. Two attribution defects the first pass did not catch were found while reading
the body — a masking claim neither cited paper makes, and an uncited reference-list entry — and
both are corrected.
