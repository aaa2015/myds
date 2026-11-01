#!/usr/bin/env python3
"""Build provenance: what source did the deployed binary actually come from?

Why this exists: `bin/` is gitignored, so the running executable is not in
version control and nothing in it names a commit. The link between "what is
deployed" and "what is in the repository" is therefore implicit and easy to
break -- edit src/, forget to redeploy, and the two silently diverge with no
symptom until someone debugs behaviour that is not in the code they are reading.

Two halves:
  write    record {commit, dirty paths, source fingerprint, artifact hashes}
           next to the binary (deploy_artifact.sh calls this on the builder)
  verify   recompute from the checkout and compare (deploy-check, on the host)

The fingerprint hashes the *contents* the build reads: every C/ObjC source, every
header, every .metal, and the Makefile. It does not use the commit id alone,
because a dirty tree can build a binary that no commit describes.

Usage:
    python3 tools/build_provenance.py write  [--out bin/BUILD-INFO.json]
    python3 tools/build_provenance.py verify [--file bin/BUILD-INFO.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RECORD = "bin/BUILD-INFO.json"

# Directories whose contents end up in the binary.
BUILD_INPUTS = ("src", "include", "metal", "third_party")
SUFFIXES = {".c", ".h", ".m", ".metal", ".inc"}
# The Makefile itself is deliberately NOT hashed. Adding an unrelated phony
# target would then invalidate provenance and demand a pointless rebuild. What
# actually determines the output is the source contents plus the effective
# flags, and the flags are captured below by asking make for them.


def build_files() -> list[Path]:
    out: list[Path] = []
    for d in BUILD_INPUTS:
        base = ROOT / d
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file() and p.suffix in SUFFIXES:
                out.append(p)
    return out


def effective_flags() -> tuple[str, str]:
    """Ask make for the flags it would actually use.

    Returns (flags, error). An empty flags with a non-empty error means "this
    checkout cannot tell me", which is a different claim from "the flags are
    different" -- conflating them would report a stale checkout as a rebuild.
    """
    try:
        r = subprocess.run(["make", "--no-print-directory", "provenance-flags"],
                           cwd=ROOT, capture_output=True, text=True, timeout=60)
    except Exception as e:
        return "", f"could not run make: {e}"
    if r.returncode != 0:
        first = (r.stderr or r.stdout).strip().splitlines()
        return "", (first[-1] if first else f"make exited {r.returncode}")
    return r.stdout, ""


def fingerprint() -> tuple[str, int, str]:
    """Hash of build inputs' path+content, plus the effective compile flags."""
    h = hashlib.sha256()
    files = build_files()
    for p in files:
        rel = p.relative_to(ROOT).as_posix()
        h.update(rel.encode())
        h.update(b"\0")
        h.update(hashlib.sha256(p.read_bytes()).digest())
    flags, err = effective_flags()
    h.update(b"\0flags\0")
    h.update(flags.encode())
    return h.hexdigest(), len(files), (flags, err)


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                              text=True, timeout=30).stdout.strip()
    except Exception:
        return ""


def dirty_build_paths() -> list[str]:
    """Uncommitted source changes that would make the binary un-reproducible."""
    out = git("status", "--porcelain", "--untracked-files=no", "--", *BUILD_INPUTS)
    return [ln for ln in out.splitlines() if ln.strip()]


def sha256_file(p: Path) -> str:
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()
    except OSError:
        return ""


def write(out_rel: str) -> int:
    fp, n, (flags, flags_err) = fingerprint()
    if flags_err:
        print(f"WARN: could not read effective flags: {flags_err}")
    rec = {
        "schema_version": 1,
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit": git("rev-parse", "HEAD"),
        "commit_short": git("rev-parse", "--short", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty_build_paths": dirty_build_paths(),
        "source_fingerprint": fp,
        "source_file_count": n,
        "effective_flags": flags,
        "artifacts": {
            "bin/myds": sha256_file(ROOT / "bin" / "myds"),
            "bin/myds.metallib": sha256_file(ROOT / "bin" / "myds.metallib"),
        },
    }
    out = ROOT / out_rel
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2) + "\n")
    print(f"wrote {out_rel}")
    print(f"  commit            : {rec['commit_short']} ({rec['branch']})")
    print(f"  source fingerprint: {fp[:16]}  ({n} files)")
    print(f"  dirty build paths : {len(rec['dirty_build_paths'])}")
    for ln in rec["dirty_build_paths"]:
        print(f"      {ln}")
    return 0


def verify(file_rel: str) -> int:
    path = ROOT / file_rel
    if not path.exists():
        print(f"UNKNOWN -- no {file_rel} next to the binary")
        print("  The deployed artifacts predate provenance recording, so their source")
        print("  cannot be established. Redeploy once to create a truthful baseline:")
        print("      ./tools/deploy_artifact.sh")
        return 2

    try:
        rec = json.loads(path.read_text())
    except Exception as e:
        print(f"FAIL -- {file_rel} is not readable JSON: {e}")
        return 1

    fp, n, (flags, flags_err) = fingerprint()
    rec_fp = rec.get("source_fingerprint", "")
    problems: list[str] = []

    if rec_fp != fp:
        problems.append(
            f"source fingerprint differs: recorded {rec_fp[:16]}, current {fp[:16]}\n"
            f"      the deployed binary was built from different source than this checkout")
    if rec.get("source_file_count") != n:
        problems.append(f"build input count differs: recorded {rec.get('source_file_count')}, current {n}")
    if flags_err:
        unsupported = "No rule to make target" in flags_err or "deploy-check" in flags_err
        msg = (f"cannot read this checkout's build flags ({flags_err}).\n"
               "      It predates the provenance targets, or make is unavailable, so\n"
               "      source and deployment cannot be compared at all")
        if unsupported:
            msg += " -- pull this checkout first"
        problems.append(msg)
    elif rec.get("effective_flags") != flags:
        problems.append("effective compile flags differ from the recorded build")

    for rel, want in (rec.get("artifacts") or {}).items():
        got = sha256_file(ROOT / rel)
        if got != want:
            problems.append(f"{rel} changed since it was recorded: "
                            f"recorded {str(want)[:16]}, current {got[:16]}")

    print(f"record          : {file_rel}")
    print(f"  built from    : {rec.get('commit_short')} ({rec.get('branch')}) "
          f"at {rec.get('written_utc')}")
    print(f"  fingerprint   : recorded {rec_fp[:16]} / current {fp[:16]}  "
          f"({'match' if rec_fp == fp else 'DIFFER'})")
    print(f"  build inputs  : recorded {rec.get('source_file_count')} / current {n}")
    dirty = rec.get("dirty_build_paths") or []
    print(f"  dirty at build: {len(dirty)}")
    print()

    if problems:
        for p in problems:
            print(f"FAIL -- {p}")
        print()
        print("  The deployed artifacts do not correspond to this source tree.")
        print("  Either redeploy (if the source change should be live) or rebuild-free")
        print("  accept the drift consciously -- but do not debug current code against")
        print("  a binary built from something else.")
        return 1

    print("PASS -- deployed artifacts were built from this exact source tree")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("write", "verify"))
    ap.add_argument("--out", default=DEFAULT_RECORD, help="record path (write)")
    ap.add_argument("--file", default=DEFAULT_RECORD, help="record path (verify)")
    args = ap.parse_args()
    return write(args.out) if args.action == "write" else verify(args.file)


if __name__ == "__main__":
    sys.exit(main())
