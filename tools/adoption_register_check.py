#!/usr/bin/env python3
"""Check the adoption register in the ledger (§0.3) against the actual build.

Why this exists: §81 was formally adopted on 2026-10-24, with an adoption
receipt and a frozen artifact identity. Somewhere after 10-27 a rebuild from
canonical dropped the feature and nothing recorded it -- the table still said
"adopted", which reads identically to "adopted and still here".

So every row now carries a status, and this script enforces what each status
promises:

  在产 in production    any MYDS_* it names must exist in
                        include/myds_knobs_generated.h -- the authoritative list
                        of what this build actually reads. A row that claims to
                        be live while pointing at a knob no code reads is
                        exactly the §81 failure.
  已取代 superseded     must name the commit or section that replaced it
  已丢失 lost           must name where the source can still be found
  已证否 disproven      nothing extra

Usage:
    python3 tools/adoption_register_check.py
    python3 tools/adoption_register_check.py --strict
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "docs" / "2026-10-20-prefill-decode-pipeline-and-optimization-ledger.md"
KNOBS = ROOT / "include" / "myds_knobs_generated.h"

STATUSES = ("在产", "已取代", "已丢失", "已证否")
KNOB_RE = re.compile(r"MYDS_[A-Z0-9_]+")
# A successor pointer: a backticked short sha, or a section reference.
SUCCESSOR_RE = re.compile(r"`[0-9a-f]{7,40}`|§\d")
# A recovery pointer: a repository path or a section reference.
RECOVERY_RE = re.compile(r"`[^`]*/[^`]*`|§\d")


def load_knobs() -> set[str]:
    if not KNOBS.exists():
        return set()
    txt = KNOBS.read_text(errors="replace")
    return set(KNOB_RE.findall(txt))


def register_rows(text: str) -> list[tuple[int, str]]:
    """Return (line_no, raw_line) for the §0.3 table's data rows."""
    lines = text.splitlines()
    start = end = None
    for i, ln in enumerate(lines):
        if ln.startswith("### 0.3 "):
            start = i
        elif start is not None and (ln.startswith("### 0.4") or
                                    ln.startswith("#### 0.3.1")):
            # 0.3.1 is the rule text with its own small table; it is not part of
            # the register. Stopping here also stops the check from grading the
            # status vocabulary against itself.
            end = i
            break
    if start is None:
        return []
    end = end if end is not None else len(lines)
    rows = []
    for i in range(start, end):
        ln = lines[i]
        if not ln.startswith("|"):
            continue
        if set(ln.replace("|", "").strip()) <= set("-: "):
            continue
        if "commit / 配置" in ln:
            continue
        rows.append((i + 1, ln))
    return rows


def last_cell(line: str) -> str:
    cells = [c for c in line.strip().strip("|").split("|")]
    return cells[-1] if cells else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    if not LEDGER.exists():
        print(f"FAIL: ledger not found at {LEDGER}")
        return 1
    if not KNOBS.exists():
        print(f"FAIL: knob header not found at {KNOBS}")
        return 1

    text = LEDGER.read_text(errors="replace")
    knobs = load_knobs()
    rows = register_rows(text)
    if not rows:
        print("FAIL: no rows found under §0.3")
        return 1

    problems: list[tuple[int, str, str]] = []   # line, kind, detail
    counts = {s: 0 for s in STATUSES}

    for lineno, line in rows:
        cell = last_cell(line)
        hit = [s for s in STATUSES if s in cell]
        if not hit:
            problems.append((lineno, "no-status",
                             "现状 column must be one of " + " / ".join(STATUSES)))
            continue
        if len(hit) > 1:
            problems.append((lineno, "ambiguous-status", " / ".join(hit)))
            continue
        status = hit[0]
        counts[status] += 1

        # Promises are kept or broken inside the status cell. Checking the whole
        # row is wrong: a superseded row contains its own commit id, which would
        # satisfy "must name the successor" by accident.
        if status == "在产":
            for knob in sorted(set(KNOB_RE.findall(line))):
                if knob not in knobs:
                    problems.append((lineno, "knob-not-in-build",
                                     f"{knob} is claimed live but no code reads it "
                                     f"(absent from include/myds_knobs_generated.h)"))
        elif status == "已取代":
            if not SUCCESSOR_RE.search(cell):
                problems.append((lineno, "superseded-without-pointer",
                                 "must name the commit or § that replaced it"))
        elif status == "已丢失":
            if not RECOVERY_RE.search(cell):
                problems.append((lineno, "lost-without-location",
                                 "must name where the source still exists"))

    print(f"register rows : {len(rows)}")
    print("status        : " + ", ".join(f"{s} {counts[s]}" for s in STATUSES))
    print(f"knobs known   : {len(knobs)}")
    print()

    if not problems:
        print("PASS -- every adoption row carries a status and keeps its promise")
        return 0

    for lineno, kind, detail in problems:
        print(f"  ledger:{lineno}  {kind}")
        print(f"      {detail}")
    print()
    print(f"{len(problems)} problem(s) in the adoption register")
    return 1 if args.strict else 0


if __name__ == "__main__":
    sys.exit(main())
