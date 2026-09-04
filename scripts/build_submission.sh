#!/usr/bin/env bash
# Build the submission PDFs from the markdown sources.
#
#   bash scripts/build_submission.sh
#
# Requires pandoc and xelatex. xelatex rather than pdflatex because the reward
# equation and the load analysis use Greek letters, which pdflatex rejects.
set -euo pipefail
export PATH="/Library/TeX/texbin:$PATH"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p submission

python3 - <<'PY'
import pathlib
s = pathlib.Path("docs/report/report.md").read_text()
s = s.replace("### 5.1 Training\n",
 "### 5.1 Training\n\n![Training return against environment steps, mean across three seeds with a band at one standard deviation.](../../figures/training_curve.png){width=85%}\n\n")
anchor = "### 5.4 The result that constrains the diagnosis" if "### 5.4" in s else "## 6. Discussion"
s = s.replace(anchor,
 "![Agent variants against the best single dispatching rule. The dashed line is ATC at -1.125.](../../figures/ablation.png){width=90%}\n\n"
 "![Every policy on four metrics, error bars from the spread across seeds.](../../figures/baseline_bars.png){width=100%}\n\n" + anchor, 1)
pathlib.Path("/tmp/report_build.md").write_text(s)
PY

build() {
  pandoc "$1" -o "$2" --pdf-engine=xelatex --resource-path=.:docs:docs/report:figures \
    -V geometry:margin=1in -V fontsize=11pt \
    -V mainfont="Georgia" -V monofont="Menlo" -V monofontoptions="Scale=0.85" \
    -V colorlinks=true -V linkcolor=RoyalBlue -V urlcolor=RoyalBlue \
    -V title="$3" -V author="Group 11 — DSCD 614 Reinforcement Learning, University of Ghana" \
    -V date="4 September 2026" --toc --toc-depth=2 2>&1 | grep -iv "Missing character" || true
  echo "  $(basename "$2")"
}
build /tmp/report_build.md             submission/Project_Report.pdf            "Dynamic Job Scheduling with a Dueling Deep Q-Network"
build docs/hyperparameters.md          submission/Hyperparameters_and_Seeds.pdf "Hyperparameters and Seeds — DUEL-2"
build docs/ai_use_declaration.md       submission/AI_Use_Declaration.pdf        "Declaration of Generative AI Use — DUEL-2"
build docs/report/literature_review.md submission/Literature_Review.pdf         "Methodological Literature Review — DUEL-2"
echo "submission/ built"
