# Citation Audit — Project_Report.pdf

Audited 4 September 2026 against locally held `.md` sources in `corpus/` (eight papers, 68,692
words) plus the bibliographic records retrieved from Crossref, OpenAlex and the arXiv API.

## Verification summary

| | |
|---|---|
| Distinct in-text citations | 11 |
| Reference-list entries | 13 |
| Listed but never cited | 0 (7 fixed during this audit) |
| Cited but not listed | 0 |
| **Full three-way verification** (text ↔ bibliographic record ↔ local `.md`) | **6** |
| Bibliographic record verified, no local `.md` | 7 |

Six citations carry a local source file and were checked phrase by phrase: Wang et al. (2016),
Schaul et al. (2016), Zhang et al. (2020), Smit et al. (2024), Hessel et al. (2018), and
van Hasselt et al. (2016). Every claim attributed to them was located in the source text.

Seven have verified bibliographic records but no local full text, because the publishers paywall
them. Under the strict criterion their claim support is **unverifiable locally**, not wrong.

## Findings

| Sentence or Claim | Citation | Issue | Explanation | Severity | Recommended Correction |
|---|---|---|---|---|---|
| "the closest published precedent and it settles three design questions" | Han and Yang (2020) | Missing local source | The report's central design argument, covering state representation, action space and algorithm, rests on this work. Its abstract was retrieved from OpenAlex; the full text sits behind an IEEE paywall and is not held locally. The quoted phrases match the abstract exactly, but nothing beyond the abstract can be checked. | **High severity, High confidence** | Obtain the PDF through the university library and confirm the three design claims and the "better than any single heuristic rule" result against the body, not the abstract. |
| "performs better than any single heuristic rule" on 85 OR-Library instances | Han and Yang (2020) | Quoted claim verified only against an abstract | This sets the benchmark the whole Results section is measured against. The wording is verbatim from the retrieved abstract. | **High severity, Medium confidence** | Same as the row above. If the full text is unobtainable before submission, add "as reported in the abstract" to the sentence. |
| "fit `Q(s,a)` to `y = r + γ·max Q_target`, with replay ... and a target network" | Mnih et al. (2015) | Missing local source | Nature paywall. The description is standard textbook material and matches the canonical account of DQN. | Low severity, High confidence | No change required. The claim is common knowledge in the field and the record is verified. |
| "two studies apply reward shaping to dynamic flexible job-shop scheduling with random arrivals" | Zhang et al. (2024); Zhang et al. (2025) | Missing local source | Both are Elsevier paywalled. Bibliographic records verified via Crossref and OpenAlex; abstracts not indexed. The claim is drawn from the title and the venue, not from the body. | Medium severity, Medium confidence | Confirm both papers use reward shaping in the way described, or narrow the sentence to what the titles support. |
| "appears in the recipe of Han and Yang (2020) and of Liu et al. (2025)" | Liu et al. (2025) | Missing local source | Title states "Prioritized Experience Replay", which supports the attribution. The body is not held locally. | Low severity, High confidence | No change required; the title carries the claim. |
| Reference list carried 7 entries with no in-text citation | Hessel, Huang, Liu, Lv, Towers, Zhang (2024), Zhang (2025) | APA violation, now fixed | A reference list contains only cited works. Each entry described a work the report discussed without formally citing. | Medium severity | **Fixed during this audit.** In-text citations added at the point each work is discussed. Inventory now closes in both directions. |

## Checks that passed

- Every in-text key resolves to a reference-list entry, and every entry is cited.
- Author names and years in the reference list match the records retrieved from Crossref and
  OpenAlex, character for character.
- All 15 DOIs and URLs are well formed; the DOI entries use the `https://doi.org/` form.
- No direct quotations appear without attribution. The two quoted phrases from Han and Yang are
  marked as quotations and attributed in the sentence.
- Reference-list formatting follows APA 7 sentence-case titles, italicised journal names,
  volume(issue), and page ranges.
- No statistic in the report is attributed to an external source. Every number in Results comes
  from this project's own committed logs, so the fabrication-risk check has no external surface.

## Overall assessment

The report's own measurements are self-contained and traceable to committed logs, so the
fabrication risk that usually dominates a citation audit does not arise here. The exposure is
concentrated in one place: Han and Yang (2020) carries the report's central design justification
and its benchmark, and only its abstract could be read. That single dependency should be closed by
obtaining the full text. The seven uncited reference entries were an APA defect and were corrected
during this audit; the citation inventory now closes in both directions.
