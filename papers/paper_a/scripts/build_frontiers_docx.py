#!/usr/bin/env python3
"""Build the Frontiers .docx from manuscript_frontiers/main.tex with pandoc.

Pandoc drops tables wrapped in \\resizebox, does not expand the custom \\code
macro, and leaves \\ref labels unresolved. This script rewrites those
constructs, numbers floats in order of appearance, uses PNG figures, and runs
pandoc with citeproc.

Usage: python papers/paper_a/scripts/build_frontiers_docx.py [out.docx]
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
MS = HERE / "manuscript_frontiers"
FIG = HERE / "figures"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "frontiers_upload" / "PaperA_Frontiers_manuscript.docx"

src = (MS / "main.tex").read_text(encoding="utf-8")
body = src.split(r"\begin{document}", 1)[1].split(r"\end{document}", 1)[0]

title = re.search(r"\\title\[[^\]]*\]\{(.*?)\}\n\n", body, re.S).group(1)
title = " ".join(title.split())
front = re.search(r"\\maketitle\n(.*?)\\begin\{abstract\}", body, re.S).group(1)
abstract = body.split("%ABSTRACT-START", 1)[1].split("%ABSTRACT-END", 1)[0]
keywords = re.search(r"\\section\{Keywords:\}(.*?)\}\n\\end\{abstract\}", body, re.S).group(1)
main = body.split("%BODY-START", 1)[1].split("%BODY-END", 1)[0]
floats = body.split("% Figures and tables, in order of first mention.", 1)[1]

# ---- number floats in order of appearance
labels = {}
counts = {"fig": 0, "tab": 0}
for env, lab in re.findall(r"\\begin\{(figure|table)\}.*?\\label\{([^}]*)\}", floats, re.S):
    k = "fig" if env == "figure" else "tab"
    counts[k] += 1
    labels[lab] = counts[k]


def fix(t: str) -> str:
    t = re.sub(r"\\ref\{([^}]*)\}", lambda m: str(labels[m.group(1)]), t)
    t = re.sub(r"\\code\{([^}]*)\}", lambda m: r"\texttt{" + m.group(1).replace("_", r"\_") + "}", t)
    t = t.replace(r"Table~", "Table ").replace(r"Tables~", "Tables ").replace(r"Figure~", "Figure ")
    return t


def float_fix(env_text: str) -> str:
    lab = re.search(r"\\label\{([^}]*)\}", env_text).group(1)
    n = labels[lab]
    kind = "Figure" if env_text.lstrip().startswith(r"\begin{figure}") else "Table"
    env_text = env_text.replace(r"\caption{", r"\caption{" + rf"\textbf{{{kind} {n}.}} ", 1)
    env_text = re.sub(r"\\resizebox\{\\linewidth\}\{!\}\{%\s*", "", env_text)
    env_text = env_text.replace(r"\end{tabular}}", r"\end{tabular}")
    env_text = re.sub(r"\\multicolumn\{(\d+)\}\{l\}\{(.*?)\} \\\\",
                      lambda m: m.group(2) + " &" * (int(m.group(1)) - 1) + r" \\", env_text)
    env_text = re.sub(r"\\includegraphics\[[^\]]*\]\{([^}]*)\.pdf\}",
                      lambda m: r"\includegraphics[width=6in]{" + str(FIG / (m.group(1) + ".png")) + "}", env_text)
    return env_text


floats_fixed = re.sub(r"\\begin\{(figure|table)\}.*?\\end\{\1\}", lambda m: float_fix(m.group(0)), floats, flags=re.S)
floats_fixed = floats_fixed.replace(r"\end{document}", "")

doc = rf"""\documentclass{{article}}
\usepackage{{booktabs,graphicx,amsmath}}
\title{{{title}}}
\author{{Muthu Kumar M\textsuperscript{{1}} and Selva Rani B\textsuperscript{{1,*}}}}
\begin{{document}}
\maketitle

\textsuperscript{{1}}School of Computer Science Engineering and Information Systems, Vellore Institute of Technology, Vellore, India

\textsuperscript{{*}}Correspondence: Selva Rani B, bselvarani@vit.ac.in

{fix(front)}

\section*{{Abstract}}
{fix(abstract)}

\textbf{{Keywords:}} {" ".join(keywords.split())}

{fix(main)}

\section*{{Figures and tables}}
{fix(floats_fixed)}

\section*{{References}}
\end{{document}}
"""
(MS / "_pandoc_input.tex").write_text(doc, encoding="utf-8")
try:
    import pypandoc  # bundles pandoc >= 3 (needed for --citeproc)
    PANDOC = pypandoc.get_pandoc_path()
except Exception:
    PANDOC = "pandoc"
cmd = [PANDOC, str(MS / "_pandoc_input.tex"), "-f", "latex", "-o", str(OUT),
       "--citeproc", "--bibliography", str(MS / "references.bib"),
       "--resource-path", f"{MS}:{FIG}", "--metadata", "link-citations=false"]
subprocess.run(cmd, check=True)
print("labels:", labels)
print("Wrote", OUT)
