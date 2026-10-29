#!/usr/bin/env python3
"""Inventory every MYDS_* environment knob the source actually reads.

Two read paths must both be covered, which is why a plain grep for
getenv("MYDS_...") is not enough:

  1. direct      getenv("MYDS_X")
  2. wrapper     some_env_helper("MYDS_X")  where the helper itself calls
                 getenv(<parameter>)

A helper is anything whose body calls getenv() with something that is not a
string literal. Its call sites are then scanned for MYDS_* arguments. Missing
path 2 silently classifies real production keys as unknown, which is exactly the
failure this tool exists to prevent.

Usage:
  knob_inventory.py --check                     text report, exit 1 on drift
  knob_inventory.py --write                     rewrite the generated header
  knob_inventory.py --conf path/to/myds.conf    also verify a live config file
"""
import argparse
import hashlib
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "include" / "myds_knobs_generated.h"
SOURCES = [ROOT / "src", ROOT / "include"]
# .m matters: src/myds_metal.m reads MYDS_METAL_CONCURRENT, MYDS_MOE_NSG,
# MYDS_INDEXER_HEAD_SLOTS and friends. Scanning only .c/.h silently reports the
# Metal host knobs as unknown. .inc matters for the same reason: the real body of
# decode_candidate_parse (which reads six production keys) lives in
# src/myds_decode_candidates.inc, not in the .m that calls it.
SUFFIXES = {".c", ".h", ".m", ".mm", ".inc"}

KNOB = re.compile(r'"MYDS_[A-Z0-9_]+"')
CALL = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*(\"MYDS_[A-Z0-9_]+\")")
# getenv( <identifier> ) -- the helper's own name argument, not a literal key.
GETENV_VAR = re.compile(r"\bgetenv\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)")
FUNC_DEF = re.compile(r"^[A-Za-z_][A-Za-z0-9_ \*]*\b([A-Za-z_][A-Za-z0-9_]*)\s*\([^;{]*\)\s*\{", re.M)
# An array initializer that may be iterated into getenv().
ARRAY_DEF = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\[[^\]]*\]\s*=\s*\{")


def strip_comments(text):
    """Blank out // and /* */ comments while keeping byte offsets and newlines.

    Comments are the main source of false positives: a doc block that names a
    knob ("policy MYDS_INDEXER_TOPK_THRESHOLD=absent/0/1") must not be mistaken
    for a reading site, or a dead switch would look alive. String literals are
    preserved so quoted keys survive.
    """
    out = list(text)
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == '"':
            i += 1
            while i < n:
                if text[i] == "\\":
                    i += 2
                    continue
                if text[i] == '"':
                    i += 1
                    break
                i += 1
            continue
        if c == "'":
            i += 1
            while i < n:
                if text[i] == "\\":
                    i += 2
                    continue
                if text[i] == "'":
                    i += 1
                    break
                i += 1
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                out[i] = " "
                i += 1
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            out[i] = out[i + 1] = " "
            i += 2
            while i < n and not (text[i] == "*" and i + 1 < n and text[i + 1] == "/"):
                if text[i] != "\n":
                    out[i] = " "
                i += 1
            if i < n:
                out[i] = " "
                if i + 1 < n:
                    out[i + 1] = " "
                i += 2
            continue
        i += 1
    return "".join(out)


def source_files():
    for base in SOURCES:
        for path in sorted(base.rglob("*")):
            if path.suffix in SUFFIXES and path.is_file():
                yield path


def scan():
    """Return (direct, wrapped, helpers) with (file, line) provenance.

    Only three patterns count as a reading site:
      1. getenv("MYDS_*")
      2. helper("MYDS_*") where the helper body calls getenv(<parameter>)
      3. an element of an array whose name is later passed to getenv()
    Anything else that merely mentions a key -- comments, doc strings,
    record_effective() bookkeeping -- is not a reader. Treating those as readers
    would hide exactly the dead-switch defect this tool exists to catch.
    """
    direct, wrapped, helpers = {}, {}, {}
    texts = {}
    for path in source_files():
        texts[path] = strip_comments(path.read_text(errors="replace"))

    # Pass 1: which functions read env through a parameter?
    for path, text in texts.items():
        for match in FUNC_DEF.finditer(text):
            name = match.group(1)
            start = match.start()
            depth, end = 0, start
            for i in range(match.end() - 1, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            body = text[match.end():end]
            if GETENV_VAR.search(body) and name != "getenv":
                helpers[name] = path

    # Pass 2: direct literals.
    for path, text in texts.items():
        for match in KNOB.finditer(text):
            prefix = text[max(0, match.start() - 40):match.start()]
            if re.search(r"getenv\s*\(\s*$", prefix) is None:
                continue
            line = text.count("\n", 0, match.start()) + 1
            key = match.group(0)[1:-1]
            direct.setdefault(key, []).append((str(path.relative_to(ROOT)), line))

    # Pass 3: helper call sites carrying a literal key.
    for path, text in texts.items():
        for match in CALL.finditer(text):
            if match.group(1) not in helpers:
                continue
            line = text.count("\n", 0, match.start()) + 1
            key = match.group(2)[1:-1]
            wrapped.setdefault(key, []).append((str(path.relative_to(ROOT)), line))

    # Pass 4: array initializers whose elements reach getenv( array[i] ).
    for path, text in texts.items():
        iterated = set(GETENV_VAR.findall(text))
        if not iterated:
            continue
        for match in ARRAY_DEF.finditer(text):
            var = match.group(1)
            if var not in iterated:
                continue
            depth, end = 0, match.end() - 1
            for i in range(match.end() - 1, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            body = text[match.end():end]
            for literal in KNOB.finditer(body):
                offset = match.end() + literal.start()
                line = text.count("\n", 0, offset) + 1
                key = literal.group(0)[1:-1]
                wrapped.setdefault(key, []).append((str(path.relative_to(ROOT)), line))

    return direct, wrapped, helpers


def render(direct, wrapped, helpers):
    keys = sorted(set(direct) | set(wrapped))
    lines = [
        "/* Generated by tools/knob_inventory.py -- do not edit by hand.",
        " * Regenerate with: python3 tools/knob_inventory.py --write",
        " *",
        " * direct  = read through getenv(\"MYDS_*\") at the call site",
        " * indirect = the key is an argument to an env helper, an element of an",
        " *           array iterated by getenv(), or otherwise not adjacent to a",
        " *           getenv() call; a plain getenv() grep cannot see these.",
        " * Requires include/myds_knobs.h for myds_knob_t; do not include alone.",
        " */",
        "#ifndef MYDS_KNOBS_GENERATED_H",
        "#define MYDS_KNOBS_GENERATED_H",
        "",
        "static const myds_knob_t myds_knobs_generated[] = {",
    ]
    for key in keys:
        if key in direct:
            path, line = direct[key][0]
            lines.append(f'    {{"{key}", "{path}", {line}, 1}},')
        else:
            path, line = wrapped[key][0]
            lines.append(f'    {{"{key}", "{path}", {line}, 0}},')
    lines += [
        "};",
        "",
        f"#define MYDS_KNOB_COUNT {len(keys)}u",
        f"#define MYDS_KNOB_DIRECT_COUNT {len(direct)}u",
        f"#define MYDS_KNOB_WRAPPED_COUNT {len(set(wrapped) - set(direct))}u",
        f'#define MYDS_KNOB_HELPERS "{",".join(sorted(helpers))}"',
        "",
        "#endif",
        "",
    ]
    return "\n".join(lines)


def parse_conf(path):
    """MYDS_* keys set by a fleet config (KEY=value lines, # comments)."""
    keys, problems = set(), []
    for number, raw in enumerate(path.read_text(errors="replace").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key = line.split("=", 1)[0].strip()
        if key.startswith("MYDS_"):
            keys.add(key)
    return keys, problems


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="verify the committed header is current")
    ap.add_argument("--write", action="store_true", help="rewrite the generated header")
    ap.add_argument("--conf", type=Path, action="append", default=[],
                    help="also verify a real config file (repeatable)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    if not args.check and not args.write:
        args.check = True

    direct, wrapped, helpers = scan()
    known = set(direct) | set(wrapped)
    generated = render(direct, wrapped, helpers)
    status = 0

    print(f"helpers reading env by parameter: {len(helpers)}")
    for name in sorted(helpers):
        print(f"  {name:28s} {helpers[name].relative_to(ROOT)}")
    print(f"direct getenv('MYDS_*') keys : {len(direct)}")
    print(f"indirect keys (helper arg or array): {len(set(wrapped) - set(direct))}"
          f"  <-- invisible to a getenv() grep")
    print(f"total distinct keys          : {len(known)}")
    if args.verbose:
        for key in sorted(set(wrapped) - set(direct)):
            path, line = wrapped[key][0]
            print(f"  wrapped  {key:36s} {path}:{line}")

    for conf in args.conf:
        keys, _ = parse_conf(conf)
        unknown = sorted(keys - known)
        print(f"\nconfig {conf}: {len(keys)} MYDS_* keys")
        if unknown:
            status = 1
            print(f"  UNKNOWN ({len(unknown)}) -- set in config, read by nothing on record:")
            for key in unknown:
                print(f"    {key}")
        else:
            print("  every key resolves to a source read path")
        if args.verbose:
            unset = sorted(known - keys)
            print(f"  known but not in this config: {len(unset)} (informational)")

    if args.write:
        HEADER.parent.mkdir(parents=True, exist_ok=True)
        HEADER.write_text(generated)
        print(f"\nwrote {HEADER.relative_to(ROOT)} ({len(generated)} bytes)")
    elif args.check:
        if not HEADER.exists():
            print(f"\nMISSING {HEADER.relative_to(ROOT)}; run --write")
            status = 1
        elif HEADER.read_text() != generated:
            print(f"\nSTALE {HEADER.relative_to(ROOT)}; run --write")
            status = 1
        else:
            print(f"\ngenerated header is current "
                  f"(sha256 {hashlib.sha256(generated.encode()).hexdigest()[:16]})")
    return status


if __name__ == "__main__":
    sys.exit(main())
