#!/usr/bin/env python3
"""Audit the docs: every path and make target a reader is told to use must exist.

Why this exists: REPRODUCE-register-only.md told readers to run
`~/Projects/myfleet/sync_myds.sh` for weeks after that script was deleted. A doc
that lies about a command costs someone an afternoon and is invisible to
reading -- you only find it by running the command.

Scope discipline matters more than coverage here. A checker that flags "GB/s"
or "A/B" as broken paths gets ignored, and an ignored checker is worse than
none. So this only claims a token when it is unambiguous:

  * the token starts with a known repo top-level directory (src/, tools/, ...),
    or with ~/Projects/myds, or is an absolute path inside this repo;
  * it contains no shell expansion, placeholder or glob;
  * it is not one of the documented "produced at run time" outputs.

make targets are matched as the whole first word after `make`, so `make bin/myds`
is a path, not a target named "bin".

Usage:
    python3 tools/doc_link_audit.py
    python3 tools/doc_link_audit.py --strict     # exit 1 on any break
    python3 tools/doc_link_audit.py --verbose
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent.parent

# Real, tracked top-level directories. A token must start with one of these to
# be treated as a repo path claim.
TOP = {"benchmarks", "build", "docs", "eval", "include", "metal",
       "scratch", "src", "tests", "third_party", "tools", "bin"}

# Directories that legitimately hold generated output. Their absence says
# nothing about the doc.
GENERATED = {"bin", "build", "scratch"}

# Known-absent by record, not by accident: the ledger states at its own
# line 13430 that this evidence tree was never committed ("git 里 0 个文件"),
# and that the two Python suites SKIP when it is missing. Reporting these as
# fresh breakage would be noise; reporting them as nothing would hide the fact
# that the archives do not exist anywhere on this machine.
KNOWN_ABSENT_PREFIX = "docs/benchmarks/"
KNOWN_ABSENT_NOTE = (
    "evidence tree recorded as never committed (ledger line 13430); "
    "the files are absent on this host too")

# Arguments whose value is produced by the command itself.
OUTPUT_FLAGS = ("--out", "--output", "--log", "-o", ">>", ">")

FENCE_RE = re.compile(r"^```([\w+-]*)")

# Languages whose fenced blocks are commands the reader is meant to run.
SHELL_LANGS = {"", "bash", "sh", "zsh", "shell", "console", "shell-session"}
# One path-ish token: no whitespace, no shell metacharacters.
TOKEN_RE = re.compile(r"[^\s'\"`|;&()<>]+")
MAKE_RE = re.compile(r"(?:^|[;&|]\s*)\s*(?:cd\s+[^\s;&|]+\s*&&\s*)*make\s+((?:-\S+\s+)*)([^\s;&|]+)")


def make_targets() -> set[str]:
    targets: set[str] = set()
    mk = ROOT / "Makefile"
    if not mk.exists():
        return targets
    for line in mk.read_text(errors="replace").splitlines():
        m = re.match(r"^([A-Za-z][\w.%-]*)\s*:(?!=)", line)
        if m:
            targets.add(m.group(1))
    return targets


def has_expansion(tok: str) -> bool:
    return any(c in tok for c in "$<>{}*?[]\\") or "..." in tok


def classify(tok: str, cwd: Path | None = None) -> tuple[str, Path] | None:
    """Return (kind, candidate) for a repo-path claim, else None.

    `cwd` is the directory the block has cd'd into, if any. Without it,
    `cd ~/Projects/myfleet && ./start_fleet.sh` looks like a missing repo-root
    script -- a false positive that would train the reader to ignore this tool.
    """
    t = tok
    if "/runs/" in t:
        return None            # benchmark output directory (see .gitignore)
    # `./x` resolves against the block's cd target when we know it.
    if cwd is not None and t.startswith("./") and not has_expansion(t):
        return ("path", cwd / t[2:])
    # file.c:123 / file.m:790）—— 行号是合法引用，剥掉后再判存在性。
    t = re.sub(r":\d+\b.*$", "", t)
    for sep in (",", ";", ":", ")", "]", "}"):
        t = t.rstrip(sep)
    if not t or has_expansion(t):
        return None

    # Normalise the prefixes a reader would actually type.
    rel: str | None = None
    if t.startswith("~/Projects/myds/"):
        rel = t[len("~/Projects/myds/"):]
    elif t.startswith("/Users/") and "/myds/" in t:
        rel = t.split("/myds/", 1)[1]
    elif t.startswith("./"):
        rel = t[2:]
    elif "/" in t:
        rel = t
    if not rel:
        return None

    # Explicitly repo-relative (`./x`, `~/Projects/myds/x`, `/Users/.../myds/x`)
    # is a claim no matter where it points: that is how ./sync_myds.sh slipped
    # past an earlier version of this check.
    explicit = tok.startswith("./") or tok.startswith("~/Projects/myds/") or (
        tok.startswith("/Users/") and "/myds/" in tok)

    head = rel.split("/", 1)[0]
    if not explicit and head not in TOP and rel not in ("Makefile", "README.md", "LICENSE"):
        return None
    if head in GENERATED:
        return None
    return ("path", ROOT / rel)


SIBLING_RE = re.compile(r"^~/Projects/([\w.-]+)(?:/([\w./-]+))?")


def classify_sibling(tok: str) -> tuple[str, Path] | None:
    """`~/Projects/<other-repo>/...` -- a real path claim about this host.

    Reported separately from repo paths because a host may legitimately not
    have that sibling checked out. But if the script the doc tells you to run
    lives there, a broken reference is still a broken reference (that is
    exactly the sync_myds.sh case).
    """
    m = SIBLING_RE.match(tok)
    if not m:
        return None
    if has_expansion(tok):
        return None
    tail = tok[len("~/"):]
    if not m.group(2):
        return None              # the repo root itself; presence is enough
    return ("sibling", Path.home() / tail)


CD_RE = re.compile(r"\bcd\s+(\S+)")


def iter_code_blocks(path: Path):
    """Yield (line_no, line, cwd) where cwd is the block's current directory.

    cwd is a Path when the block has cd'd somewhere resolvable, and False when
    it cd'd somewhere with a placeholder (so `./x` becomes unknowable rather
    than a false hit).
    """
    inside = False
    lang = ""
    cwd: Path | bool | None = None
    for n, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        m = FENCE_RE.match(line)
        if m:
            inside = not inside
            if inside:
                lang = m.group(1).lower()
                cwd = None
            continue
        if not inside:
            continue
        # Only shell-ish blocks are instructions to run. A block tagged `text`
        # or `c` may quote a command as an example (the audit's own negative
        # test does exactly that) -- auditing it would be a false alarm.
        if lang not in SHELL_LANGS:
            continue
        for m in CD_RE.finditer(line):
            target = m.group(1).strip("'\"")
            if has_expansion(target):
                cwd = False
                continue
            if target.startswith("~/"):
                cwd = Path.home() / target[2:]
            elif target.startswith("/"):
                cwd = Path(target)
            elif target.startswith("./") or "/" in target:
                cwd = ROOT / target
            else:
                cwd = False
        yield n, line, cwd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    targets = make_targets()
    md_files = sorted(
        p for p in list(ROOT.glob("docs/*.md"))
        + list(ROOT.glob("benchmarks/**/*.md"))
        + list(ROOT.glob("*.md"))
        if p.exists()
    )

    missing: list[tuple[str, int, str, str]] = []
    sibling_missing: list[tuple[str, int, str, str]] = []
    known: set[str] = set()
    n_paths = n_make = n_unverifiable = 0

    for md in md_files:
        rel_md = str(md.relative_to(ROOT))
        for lineno, line, cwd in iter_code_blocks(md):
            for m in MAKE_RE.finditer(line):
                target = m.group(2)
                if "/" in target or has_expansion(target):
                    continue          # it is a path argument, not a target
                if not re.match(r"^[A-Za-z][\w.%-]*$", target):
                    continue          # '#' and friends: prose, not a target
                n_make += 1
                if target not in targets:
                    missing.append((rel_md, lineno, "make-target", target))
            # Values after an output flag are created by the command, not read
            # from the repo; skip them so `--out path/to/new.json` is not a bug.
            out_spans: list[tuple[int, int]] = []
            for flag in OUTPUT_FLAGS:
                for fm in re.finditer(re.escape(flag) + r"\s+(\S+)", line):
                    out_spans.append(fm.span(1))

            def is_output(pos: int) -> bool:
                return any(a <= pos < b for a, b in out_spans)

            for m in TOKEN_RE.finditer(line):
                if is_output(m.start()):
                    continue
                # `./x` with no resolvable cd in this block: we cannot know
                # which directory the reader is in (`cd <部署目录> && ./x` is a
                # real pattern here). Count it, do not guess.
                if m.group(0).startswith("./") and not isinstance(cwd, Path):
                    n_unverifiable += 1
                    continue
                sib = classify_sibling(m.group(0))
                if sib:
                    kind, cand = sib
                    n_paths += 1
                    if not cand.exists():
                        sibling_missing.append((rel_md, lineno, kind, m.group(0)))
                    continue
                got = classify(m.group(0), cwd if isinstance(cwd, Path) else None)
                if not got:
                    continue
                kind, cand = got
                n_paths += 1
                if cand.exists():
                    continue
                # A resolved `cd` can point outside this repo (the deployment
                # dirs live in ~/Projects/myfleet). Those are real claims too;
                # just not repo-relative.
                in_repo = ROOT == cand or ROOT in cand.parents
                if in_repo and str(cand.relative_to(ROOT)).startswith(KNOWN_ABSENT_PREFIX):
                    known.add(f"{rel_md}:{lineno}  {m.group(0)}")
                    continue
                missing.append((rel_md, lineno, kind, m.group(0)))

    print(f"doc files     : {len(md_files)}")
    print(f"path claims   : {n_paths}")
    print(f"make refs     : {n_make}  ({len(targets)} rules in Makefile)")
    if n_unverifiable:
        print(f"unverifiable  : {n_unverifiable}  (./x with no resolvable cd -- not guessed)")
    print()

    if known and args.verbose:
        print(f"known-absent ({len(known)}) -- {KNOWN_ABSENT_NOTE}:")
        for k in sorted(known):
            print(f"  {k}")
        print()

    if not missing:
        print("PASS -- every path and make target referenced from the docs resolves")
        if known:
            print(f"       ({len(known)} reference(s) point at the known-absent evidence tree)")
        return 0

    by_file: dict[str, list[tuple[int, str, str]]] = {}
    for f, ln, kind, tok in missing:
        by_file.setdefault(f, []).append((ln, kind, tok))

    for f in sorted(by_file):
        print(f)
        seen: set = set()
        for ln, kind, tok in sorted(by_file[f]):
            key = (kind, tok)
            if not args.verbose and key in seen:
                continue
            seen.add(key)
            print(f"  {ln:>7}  {kind:<12} {tok}")
        print()

    if sibling_missing:
        print("sibling-repo references (outside this checkout):")
        seen2: set = set()
        for f, ln, kind, tok in sorted(sibling_missing):
            key = (kind, tok)
            if not args.verbose and key in seen2:
                continue
            seen2.add(key)
            print(f"  {f}:{ln}  {tok}")
        print()

    total = len(missing) + len(sibling_missing)
    print(f"{total} broken reference(s) "
          f"({len(missing)} in-repo, {len(sibling_missing)} sibling-repo)")
    return 1 if (args.strict and total) else 0


if __name__ == "__main__":
    sys.exit(main())
