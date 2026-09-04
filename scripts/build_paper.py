"""Render docs/report/report.md into the two-column LaTeX paper.

    python3 scripts/build_paper.py

Markdown is the single source. This script converts it, rewrites the narrative
citations into apacite commands, substitutes the tables and figures, normalises
the characters pdflatex cannot set, and runs the LaTeX toolchain.
"""
from __future__ import annotations

import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "docs/report/report.md"
OUT = ROOT / "paper"

CITES = [
    (r"Han and Yang \(2020\)", r"\\citet{hanyang2020}"),
    (r"\(Mnih et al\., 2015\)", r"\\citep{mnih2015}"),
    (r"\(Wang et al\., 2016\)", r"\\citep{wang2016dueling}"),
    (r"\(Schaul et al\., 2016\)", r"\\citep{schaul2016per}"),
    (r"\(Hessel et al\., 2018\)", r"\\citep{hessel2018rainbow}"),
    (r"Zhang et al\. \(2020\)", r"\\citet{zhang2020l2d}"),
    (r"Smit et al\. \(2024\)", r"\\citet{smit2024gnn}"),
    (r"Lv et al\. \(2025\)", r"\\citet{lv2025review}"),
    (r"Liu et al\.\s*\n?\(2025\)", r"\\citet{liu2025ddqnper}"),
    (r"\(Zhang et al\., 2024; Zhang et al\., 2025\)",
     r"\\citep{zhang2024shaping,zhang2025dueling}"),
    (r"\(Towers et al\., 2023\)", r"\\citep{towers2023gymnasium}"),
    (r"\(Huang et al\., 2022\)", r"\\citep{huang2022cleanrl}"),
]

# pdflatex cannot set these; apacite and cvpr.sty give no better route than substitution
GLYPH = {
    "γ": r"$\gamma$", "λ": r"$\lambda$", "Δ": r"$\Delta$", "α": r"$\alpha$",
    "β": r"$\beta$", "δ": r"$\delta$", "ρ": r"$\rho$", "μ": r"$\mu$",
    "Σ": r"$\Sigma$", "ε": r"$\epsilon$", "∈": r"$\in$", "−": "-",
    "×": r"$\times$", "≤": r"$\leq$", "≥": r"$\geq$", "→": r"$\rightarrow$",
    "±": r"$\pm$", "·": r"$\cdot$", "…": "...", "∞": r"$\infty$",
    "’": "'", "‘": "'", "“": "``", "”": "''", "–": "--", "—": "---",
    "p̄": r"$\bar{p}$", "⁻": "-", "⁴": "4", "¯": "",
}

TABLES = [
    r"""\begin{table*}[t]
\centering\small
\begin{tabular}{lrrrr}
\toprule
Rule & Avg.\ waiting & Missed & Weighted tardiness & Return \\
\midrule
SPT ($=$ SJF) & 4.077 & 0.256 & 136.4 & $-1.352$ \\
WSPT & 4.344 & 0.276 & 104.9 & $-1.187$ \\
ATC  & 4.372 & 0.277 & 95.1  & $-1.125$ \\
EDD  & 4.521 & 0.363 & 124.0 & $-1.345$ \\
CR   & 4.851 & 0.445 & 133.8 & $-1.465$ \\
MS   & 4.947 & 0.421 & 144.8 & $-1.554$ \\
FCFS & 5.539 & 0.465 & 212.3 & $-2.099$ \\
LPT  & 7.744 & 0.421 & 437.9 & $-3.958$ \\
\bottomrule
\end{tabular}
\caption{The eight dispatching rules on the thirty held-out instances. ATC sets the bar.}
\label{tab:rules}
\end{table*}
""",
    r"""\begin{table*}[t]
\centering\small
\begin{tabular}{lcccc}
\toprule
Variant & Per-seed return & Mean & s.d. & Gap to ATC \\
\midrule
Formulation A, direct assignment & $-1.525$, $-1.544$, $-1.561$ & $-1.543$ & 0.015 & $-0.418$ \\
Formulation B, uniform replay    & $-1.334$, $-1.314$, $-1.247$ & $-1.298$ & 0.037 & $-0.173$ \\
B $+$ prioritised replay         & $-1.309$, $-1.333$, $-1.294$ & $-1.312$ & 0.016 & $-0.187$ \\
B $+$ n-step 3                   & $-1.222$, $-1.212$, $-1.269$ & $-1.235$ & 0.025 & $-0.109$ \\
B $+$ PER $+$ n-step 3           & $-1.315$, $-1.282$, $-1.263$ & $-1.287$ & 0.021 & $-0.161$ \\
\bottomrule
\end{tabular}
\caption{Ablation across three seeds at one million steps per seed.}
\label{tab:ablation}
\end{table*}
""",
]

FIGURES = [
    ("\\subsection{Training}", r"""
\begin{figure*}[t]
\centering
\includegraphics[width=0.72\textwidth]{figures/training_curve.png}
\caption{Episode return against environment steps, meaned across three seeds with a band at one
standard deviation. Return improves from $-1.79$ to $-1.51$ and is flat from roughly step
400{,}000, so the budget was sufficient and the plateau is not an artefact of stopping early.}
\label{fig:training}
\end{figure*}
"""),
    ("\\subsection{Ablation}", r"""
\begin{figure*}[t]
\centering
\includegraphics[width=0.72\textwidth]{figures/ablation.png}
\caption{Agent variants against the best single dispatching rule. The dashed line is ATC at
$-1.125$. Every variant on the rule action space outperforms all three required baselines; none
reaches the bar.}
\label{fig:ablation}
\end{figure*}
"""),
    ("\\section{Discussion}", r"""
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/baseline_bars.png}
\caption{Every policy on four metrics, error bars from the spread across seeds. Utilisation and
makespan barely separate the policies at this load.}
\label{fig:bars}
\end{figure*}
"""),
]


def substitute_glyphs(text: str) -> str:
    """Replace unsettable characters, leaving existing maths untouched."""
    out, in_math, i = [], False, 0
    while i < len(text):
        if text[i] == "$":
            run = 2 if text[i:i + 2] == "$$" else 1
            in_math = not in_math
            out.append(text[i:i + run])
            i += run
            continue
        ch = text[i]
        out.append(ch if (in_math or ch not in GLYPH) else GLYPH[ch])
        i += 1
    return "".join(out).replace("1 $\\times$ 10-4", "$1\\times10^{-4}$")


def main() -> None:
    md = SRC.read_text()
    body = md.split("## 9. References")[0]     # bibtex renders the list
    body = re.sub(r"^# .*?\n", "", body, count=1)
    for pat, rep in CITES:
        body = re.sub(pat, rep, body)

    tex = subprocess.run(["pandoc", "-f", "markdown", "-t", "latex", "--wrap=preserve"],
                         input=body, capture_output=True, text=True, check=True).stdout

    tex = re.sub(r"^DSCD 614.*?\n", "", tex, count=1)
    tex = tex.replace(r"\begin{center}\rule{0.5\linewidth}{0.5pt}\end{center}", "", 1)
    tex = re.sub(r"\\subsection\{(\d+)\. ([^}]*)\}", r"\\section{\2}", tex)
    tex = re.sub(r"\\subsubsection\{[\d.]+ ([^}]*)\}", r"\\subsection{\1}", tex)
    tex = re.sub(r"\\subsection\{Appendix [AB][^}]*\}", "", tex)
    tex = re.sub(r"\\tightlist\n", "", tex)
    tex = re.sub(r"\\textbf\{([^}]*)\}", r"\1", tex)   # convention: no bold in body prose

    pending = list(TABLES)
    tex = re.sub(r"\\begin\{longtable\}.*?\\end\{longtable\}",
                 lambda m: pending.pop(0) if pending else "", tex, flags=re.S)

    for anchor, fig in FIGURES:
        if anchor in tex:
            tex = tex.replace(anchor, fig + "\n" + anchor, 1)

    tex = substitute_glyphs(tex)
    residual = sorted({c for c in tex if ord(c) > 0x2000})
    if residual:
        print(f"  warning: unsettable characters remain: {residual}")

    (OUT / "body.tex").write_text(tex.strip() + "\n")

    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "main"],
                       cwd=OUT, capture_output=True)
        subprocess.run(["bibtex", "main"], cwd=OUT, capture_output=True)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "main"], cwd=OUT, capture_output=True)
    log = (OUT / "main.log").read_text(errors="replace")
    pages = re.search(r"Output written on main\.pdf \((\d+) pages", log)
    fatal = len(re.findall(r"(?m)^! ", log))
    print(f"  main.pdf: {pages.group(1) if pages else '?'} pages, {fatal} fatal LaTeX errors")


if __name__ == "__main__":
    main()
