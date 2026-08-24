#!/usr/bin/env python3
"""Verify candidate DOIs against Crossref and emit BibTeX for admissible entries.

Admissible = Crossref ``type == journal-article`` with a container title and a
volume plus page range or article number. Anything else is reported so it can be
dropped or replaced before it reaches dlReferences.bib.

Crossref occasionally omits the volume for very recent articles. A candidate line
may therefore carry a trailing ``volume=NN`` override, which is applied only after
the record is confirmed to be a journal article.

Candidate file format, one entry per line:
    <bibkey> <DOI> [volume=NN]

Usage:
    python verify_crossref.py candidates.txt            # report only
    python verify_crossref.py candidates.txt --bibtex   # report + emit .bib
"""

from __future__ import annotations

import html
import json
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

MAILTO = "muthu@example.org"
API = "https://api.crossref.org/works/"
OUTPUT = Path(__file__).resolve().parent / "crossref_verified.bib"


@dataclass
class Record:
    doi: str
    key: str
    volume_override: str = ""
    ok: bool = False
    reason: str = ""
    data: dict = field(default_factory=dict)


def fetch(doi: str) -> dict | None:
    url = API + urllib.parse.quote(doi, safe="") + "?mailto=" + MAILTO
    req = urllib.request.Request(url, headers={"User-Agent": f"paper-c-refs (mailto:{MAILTO})"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.load(resp)["message"]
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def clean(text: str) -> str:
    """Crossref serves titles as HTML fragments; strip tags and decode entities."""
    text = re.sub(r"<[^>]+>", "", text)
    return " ".join(html.unescape(text).split())


def latex_escape(text: str) -> str:
    for src, dst in (("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
        text = text.replace(src, dst)
    return text


CONSORTIUM_NOISE = re.compile(
    r"^(on behalf of|for the|and the)\b|consorti|initiative|working group|update panel|challenge",
    re.IGNORECASE,
)


MAX_AUTHORS = 15


def authors(msg: dict) -> str:
    """Crossref lists challenge consortia as pseudo-authors; drop them, and cap
    the very long challenge-paper author lists with a BibTeX ``others``."""
    names = []
    for a in msg.get("author", []):
        family, given = a.get("family"), a.get("given")
        if family and given:
            names.append(f"{clean(family)}, {clean(given)}")
            continue
        solo = clean(family or a.get("name") or "")
        if solo and not CONSORTIUM_NOISE.search(solo):
            names.append(solo)
    if len(names) > MAX_AUTHORS:
        names = names[:MAX_AUTHORS] + ["others"]
    return " and ".join(names)


def year(msg: dict) -> str:
    for field_name in ("published-print", "published-online", "issued", "created"):
        part = msg.get(field_name, {}).get("date-parts", [[None]])[0][0]
        if part:
            return str(part)
    return ""


def check(msg: dict) -> tuple[bool, str]:
    if msg.get("type") != "journal-article":
        return False, f"type={msg.get('type')}"
    if not msg.get("container-title"):
        return False, "no journal title"
    if not msg.get("volume"):
        return False, "no volume"
    if not (msg.get("page") or msg.get("article-number")):
        return False, "no pages/article-number"
    return True, ""


def to_bibtex(key: str, msg: dict, doi: str) -> str:
    lines = [f"@article{{{key},"]
    lines.append(f"  author  = {{{authors(msg)}}},")
    lines.append(f"  title   = {{{latex_escape(clean(msg.get('title', [''])[0]))}}},")
    lines.append(f"  journal = {{{latex_escape(clean(msg['container-title'][0]))}}},")
    if msg.get("volume"):
        lines.append(f"  volume  = {{{msg['volume']}}},")
    if msg.get("issue"):
        lines.append(f"  number  = {{{msg['issue']}}},")
    pages = msg.get("page") or msg.get("article-number")
    if pages:
        lines.append(f"  pages   = {{{pages.replace('-', '--')}}},")
    lines.append(f"  year    = {{{year(msg)}}},")
    lines.append(f"  doi     = {{{doi}}},")
    lines.append(f"  url     = {{https://doi.org/{doi}}}")
    lines.append("}")
    return "\n".join(lines)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    emit = "--bibtex" in sys.argv
    records: list[Record] = []
    with open(sys.argv[1], encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            parts = raw.split()
            key, doi = parts[0], parts[1]
            override = ""
            for extra in parts[2:]:
                if extra.startswith("volume="):
                    override = extra.split("=", 1)[1]
            records.append(Record(doi=doi, key=key, volume_override=override))

    for rec in records:
        msg = fetch(rec.doi)
        time.sleep(0.12)
        if msg is None:
            rec.reason = "not in Crossref"
            continue
        rec.data = msg
        if rec.volume_override and not msg.get("volume"):
            msg["volume"] = rec.volume_override
            if not (msg.get("page") or msg.get("article-number")):
                alt = msg.get("alternative-id") or []
                if alt:
                    msg["article-number"] = alt[0]
        rec.ok, rec.reason = check(msg)

    print("=" * 78)
    print("REJECTED / NEEDS ATTENTION")
    print("=" * 78)
    for rec in records:
        if not rec.ok:
            title = clean(rec.data.get("title", [""])[0])[:70] if rec.data else ""
            print(f"  {rec.key:28s} {rec.doi:38s} {rec.reason:24s} {title}")

    print()
    print("=" * 78)
    print(f"ACCEPTED ({sum(r.ok for r in records)}/{len(records)})")
    print("=" * 78)
    for rec in records:
        if rec.ok:
            m = rec.data
            print(f"  {rec.key:28s} {year(m):4s} {clean(m['container-title'][0])[:44]:46s} {rec.doi}")

    if emit:
        out = "\n\n".join(to_bibtex(r.key, r.data, r.doi) for r in records if r.ok)
        with OUTPUT.open("w", encoding="utf-8") as fh:
            fh.write(out + "\n")
        print(f"\nWrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
