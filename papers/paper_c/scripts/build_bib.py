#!/usr/bin/env python3
"""Assemble dlReferences.bib from the Crossref-verified entries.

Reads the section comments and ordering from candidates.txt so the shipped
bibliography stays grouped by theme, and fails loudly if a candidate that
survived verification is missing from the generated BibTeX.

Usage:
    python build_bib.py    # writes ../dlReferences.bib
"""

from __future__ import annotations

import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
CANDIDATES = HERE / "candidates.txt"
VERIFIED = HERE / "crossref_verified.bib"
OUTPUT = HERE.parent / "dlReferences.bib"

HEADER = """% dlReferences.bib -- Paper C (deep-learning IDC histopathology)
%
% Scope: breast cancer detection from the Kaggle IDC breast histopathology
% dataset (277,524 patches, 279 public case identifiers) and the BCSS external
% cohort, with modern pathology foundation models under case-ID-grouped and
% institution-held-out evaluation.
%
% Admission policy: peer-reviewed journal articles only. No preprints, no
%   conference or workshop proceedings, no LNCS chapters, no dataset cards.
%
% Every entry below was retrieved from the Crossref REST API and required to
% report type=journal-article with a journal title, a volume, and a page range
% or article number. Regenerate and re-check with:
%
%   python scripts/verify_crossref.py scripts/candidates.txt --bibtex
%   python scripts/build_bib.py
%
% Two records carried a manually supplied volume because Crossref had not yet
% populated the field for a recent issue; both are marked inline below.
"""

RULER = "% " + "-" * 76


def parse_entries(text: str) -> dict[str, str]:
    entries: dict[str, str] = {}
    for block in re.split(r"\n(?=@)", text.strip()):
        match = re.match(r"@\w+\{([^,]+),", block)
        if match:
            entries[match.group(1)] = block.rstrip()
    return entries


def main() -> int:
    entries = parse_entries(VERIFIED.read_text(encoding="utf-8"))
    out: list[str] = [HEADER]
    emitted: set[str] = set()

    for raw in CANDIDATES.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            comment = line.lstrip("#").strip()
            if comment.startswith("---") and comment.endswith("---"):
                out.append(f"\n{RULER}\n% {comment.strip('- ').strip()}\n{RULER}")
            continue
        key = line.split()[0]
        if key in entries:
            note = ""
            if "volume=" in line:
                note = "% volume supplied manually: absent from the Crossref record\n"
            out.append("\n" + note + entries[key])
            emitted.add(key)

    missing = set(entries) - emitted
    if missing:
        print(f"ERROR: verified but not emitted: {sorted(missing)}", file=sys.stderr)
        return 1

    OUTPUT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT} with {len(emitted)} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
