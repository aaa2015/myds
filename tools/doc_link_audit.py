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
KNOWN_ABSENT_PREFIXES = (
    "docs/benchmarks/",
    # Eight benchmark directories that are referenced by the ledger but exist
    # nowhere: not here, not in the mydsdoc repository, not in git history
    # (1520 commits), not anywhere on this host. Each has a 0.x "evidence
    # auditability" notice in the section that cites it.
    "benchmarks/2026-10-23-iq2-signed4/",
    "benchmarks/2026-10-23-qkv-dual/",
    "benchmarks/2026-10-23-attention-multihead/",
    "benchmarks/2026-10-23-next-directions/",
    "benchmarks/2026-10-24-moe-workspace/",
    "benchmarks/2026-10-24-moe-localgrow/",
    "benchmarks/2026-10-24-moe-phys-cause/",
    "benchmarks/2026-10-24-prefill-latency-v4/",
)
KNOWN_ABSENT_NOTE = (
    "evidence referenced by the ledger but present nowhere: not this repo, not "
    "mydsdoc, not git history, not this host. Each citing section carries a "
    "0.x evidence-auditability notice")

# Arguments whose value is produced by the command itself.
OUTPUT_FLAGS = ("--out", "--output", "--log", "-o", ">>", ">")

FENCE_RE = re.compile(r"^```([\w+-]*)")

# Languages whose fenced blocks are commands the reader is meant to run.
SHELL_LANGS = {"", "bash", "sh", "zsh", "shell", "console", "shell-session"}
# One path-ish token: no whitespace, no shell metacharacters.
# Full-width punctuation must end a token too: Chinese prose writes
# `include/myds_knobs_generated.h（说明）` with no space before the bracket.
TOKEN_RE = re.compile(r"[^\s'\"`|;&()<>（）：，。、「」【】]+")
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


# A benchmark evidence directory, wherever it appears (prose or code).
# Trailing `-` and `-*` are stripped: `benchmarks/2026-10-23-*` is a glob in prose,
# not a claim about one directory.
BENCH_DIR_RE = re.compile(r"benchmarks/[0-9]{4}-[0-9]{2}-[0-9]{2}[a-z0-9-]*")

# Evidence moved out of this repository into the documentation repo
# (commit 754edd5 "keep only the pipeline ledger in this repo, move everything
# else to the documentation repository"). Still exists, just elsewhere -- so we
# verify it there instead of calling it broken.
DOCS_REPO = Path.home() / "Projects" / "mydsdoc"


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
    n_paths = n_make = n_unverifiable = n_bench = 0
    bench_missing: set[str] = set()
    moved: set[str] = set()
    unverifiable_dirs: set[str] = set()

    for md in md_files:
        rel_md = str(md.relative_to(ROOT))
        # Every benchmark directory named anywhere in the file (prose included)
        # must exist. The 93 references to the eight lost 2026-10-23/24 trees are
        # mostly prose, so the code-block pass alone never sees them.
        text = md.read_text(errors="replace")
        for m in BENCH_DIR_RE.finditer(text):
            tok = m.group(0).rstrip("-/")
            while tok.endswith("-*"):
                tok = tok[:-2].rstrip("-")
            if "/" in tok.split("/", 1)[1] and not tok.split("/", 1)[1].count("/"):
                pass
            cand = ROOT / tok
            if cand.is_dir():
                continue
            n_bench += 1
            # A family reference (`benchmarks/2026-10-23` after stripping `-*`)
            # is a prefix of known-absent entries, not a directory that should
            # exist on its own.
            is_family = any(pref.startswith(tok + "-") or pref.startswith(tok + "/")
                            for pref in KNOWN_ABSENT_PREFIXES)
            if is_family or any((tok + "/").startswith(pref) or tok == pref.rstrip("/")
                                for pref in KNOWN_ABSENT_PREFIXES):
                known.add(f"{rel_md}  {tok}")
                continue
            if DOCS_REPO.is_dir():
                if (DOCS_REPO / tok).is_dir():
                    moved.add(f"{rel_md}  {tok}")
                    continue
            elif (tok + "/").startswith("benchmarks/2026-10-05") or tok == \
                    "benchmarks/2026-10-24-indexscore-mask-survey":
                # The documentation repo simply is not checked out on this host.
                # "I cannot look there" is not "it is missing" -- treating the two
                # the same is how a gate becomes machine-dependent and gets
                # disabled. Counted as unverifiable instead.
                unverifiable_dirs.add(f"{rel_md}  {tok}")
                continue
            bench_missing.add(f"{rel_md}  {tok}")
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
                # Normalise with a trailing slash: a bare directory name must
                # match a prefix that ends in "/", otherwise the known-absent
                # list silently fails for exactly the entries it lists.
                rel_cand = (str(cand.relative_to(ROOT)) if in_repo else "") + "/"
                if in_repo and rel_cand.startswith(KNOWN_ABSENT_PREFIXES):
                    known.add(f"{rel_md}:{lineno}  {m.group(0)}")
                    continue
                missing.append((rel_md, lineno, kind, m.group(0)))

    print(f"doc files     : {len(md_files)}")
    print(f"path claims   : {n_paths}")
    print(f"make refs     : {n_make}  ({len(targets)} rules in Makefile)")
    if n_bench:
        extra = f", {len(unverifiable_dirs)} unverifiable (mydsdoc not checked out)" \
            if unverifiable_dirs else ""
        print(f"benchmark dirs: {n_bench} named -- {len(known)} known-absent, "
              f"{len(moved)} moved to mydsdoc{extra}")
    if n_unverifiable:
        print(f"unverifiable  : {n_unverifiable}  (./x with no resolvable cd -- not guessed)")
    print()

    if known and args.verbose:
        print(f"known-absent ({len(known)}) -- {KNOWN_ABSENT_NOTE}:")
        for k in sorted(known):
            print(f"  {k}")
        print()

    if not missing and not bench_missing:
        print("PASS -- every path, benchmark directory and make target "
              "referenced from the docs resolves")
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

    if moved and args.verbose:
        print("benchmark directories that now live in the documentation repo:")
        for mv in sorted(moved):
            print(f"  {mv}")
        print()

    if bench_missing:
        print("benchmark directories named but missing here, in mydsdoc, "
              "and not in the known-absent list:")
        for b in sorted(bench_missing):
            print(f"  {b}")
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

    total = len(missing) + len(sibling_missing) + len(bench_missing)
    print(f"{total} broken reference(s) "
          f"({len(missing)} in-repo, {len(sibling_missing)} sibling-repo, "
          f"{len(bench_missing)} benchmark-dir)")
    return 1 if (args.strict and total) else 0


if __name__ == "__main__":
    sys.exit(main())
