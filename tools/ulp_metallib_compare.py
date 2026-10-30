#!/usr/bin/env python3
"""0-ULP comparison between two metallib builds, using the same binary.

Why this can be false and still look fine: two shader libraries can differ in
0.44% of their bytes and still produce identical results (different instruction
scheduling, same arithmetic), or they can differ in one ULP at position 137 and
agree everywhere a human would look. So the assertion is not "output text matches"
-- it is "every dumped logit is bit-identical, and the same arm twice is also
bit-identical", because a non-deterministic arm makes any comparison meaningless
(ledger §C.13 Q9: the first assertion is same-arm-twice, not arm-vs-arm).

Both arms share one executable; only <dir>/myds.metallib differs, which is the
first path myds_metal_init searches. That keeps the compiler, the binary and the
model identical and leaves the shader as the single variable.

Must run inside an exclusive window: the model is ~150 GB wired and the box has
192 GB.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

DEFAULTS = dict(
    workdir=str(Path.home() / "Projects/myfleet"),
    model=str(Path.home() / ".lmstudio/models/antirez/deepseek-v4.1-flash-1017/"
              "DeepSeek-V4.1-Flash-1017-Q2-dense-q4-imx-cliecy.gguf"),
    mem_floor_gb=140,
)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def emit(path, record):
    with open(path, "a") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def mem_gb():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    values = {}
    for line in out.splitlines():
        match = re.match(r"Pages (free|inactive|speculative):\s+(\d+)", line)
        if match:
            values[match.group(1)] = int(match.group(2))
    return sum(values.values()) * 16384 / 1e9


def wait_memory(floor_gb, timeout=400.0):
    deadline, seen = time.time() + timeout, []
    while time.time() < deadline:
        available = mem_gb()
        seen.append(round(available, 1))
        if available >= floor_gb:
            return available
        time.sleep(2.0)
    raise RuntimeError(f"memory below {floor_gb} GB: {seen[-5:]}")


def sha256(path):
    handle = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            handle.update(block)
    return handle.hexdigest()


def run_score(binary_dir, model, text, dump_dir, tag, out, timeout=3600):
    """One scoring pass; returns the digest of every dumped logit vector.

    Runs with cwd = the arm directory on purpose. myds_metal_init searches
    `./bin/myds.metallib` BEFORE `<exec_dir>/myds.metallib`, and that path is
    relative to the working directory -- run from the repository root, every arm
    silently loads the deployed library and the comparison is meaningless. That
    happened once here and produced a false "identical" verdict.
    """
    shutil.rmtree(dump_dir, ignore_errors=True)
    dump_dir.mkdir(parents=True, exist_ok=True)
    binary_dir = Path(binary_dir)
    argv = [str(binary_dir / "score"), model, "--text", text, "--dump", str(dump_dir)]
    started = time.time()
    completed = subprocess.run(argv, capture_output=True, text=True,
                               timeout=timeout, cwd=str(binary_dir))
    elapsed = round(time.time() - started, 2)
    files = sorted(dump_dir.glob("*.f32"))
    # Digest covers position count, per-file sizes and per-file bytes, so a
    # truncated or reordered dump cannot accidentally match.
    digest = hashlib.sha256()
    sizes = {}
    for path in files:
        raw = path.read_bytes()
        sizes[path.name] = len(raw)
        digest.update(path.name.encode())
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    # Prove which library was actually loaded; without this the whole comparison
    # can pass while both arms used the same file.
    loaded = None
    for line in completed.stderr.splitlines():
        match = re.search(r"loaded precompiled library from (\S+)", line)
        if match:
            loaded = match.group(1)
    expected = str((binary_dir / "myds.metallib").resolve())
    loaded_ok = loaded is not None and Path(loaded).resolve() == Path(expected)
    record = dict(kind="run", tag=tag, exit=completed.returncode, seconds=elapsed,
                  loaded_library=loaded, loaded_expected=expected, loaded_ok=loaded_ok,
                  positions=len(files), digest=digest.hexdigest(),
                  position_bytes=sorted(set(sizes.values())),
                  stderr_tail=completed.stderr.strip().splitlines()[-3:]
                  if completed.stderr.strip() else [])
    emit(out, record)
    log(f"{tag}: exit={completed.returncode} positions={len(files)} "
        f"digest={record['digest'][:16]} lib={loaded_ok} ({elapsed}s)")
    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--old-dir", required=True, help="score + deployed metallib")
    ap.add_argument("--new-dir", required=True, help="score + rebuilt metallib")
    ap.add_argument("--text", required=True, help="plain-text prompt file")
    ap.add_argument("--repeats", type=int, default=3,
                    help="per-arm repeats; the first check is same-arm determinism")
    ap.add_argument("--warmup", type=int, default=1,
                    help="discarded runs per arm before measuring. The very first "
                         "process on this machine produced different logits from all "
                         "later ones (cold caches), which is a warm-up artifact, not a "
                         "library difference -- measured 2026-10-30.")
    for key, value in DEFAULTS.items():
        ap.add_argument("--" + key.replace("_", "-"), default=value, type=type(value))
    a = ap.parse_args()

    out = Path(a.out)
    if out.exists():
        ap.error("--out already exists")
    old_dir, new_dir = Path(a.old_dir), Path(a.new_dir)
    for label, path in (("old", old_dir), ("new", new_dir)):
        for name in ("score", "myds.metallib"):
            if not (path / name).is_file():
                ap.error(f"{label} arm missing {path / name}")
    old_lib = sha256(old_dir / "myds.metallib")
    new_lib = sha256(new_dir / "myds.metallib")
    if old_lib == new_lib:
        ap.error("the two arms have the same metallib; nothing to compare")
    if sha256(old_dir / "score") != sha256(new_dir / "score"):
        ap.error("the two arms use different executables; only the library may differ")

    emit(out, dict(kind="plan", model=a.model, text=a.text, warmup=a.warmup,
                   old_lib=old_lib, new_lib=new_lib,
                   score_sha256=sha256(old_dir / "score"),
                   repeats=a.repeats,
                   assertion="same-arm twice identical (Q9), then arm-vs-arm identical"))
    available = wait_memory(a.mem_floor_gb)
    log(f"available {available:.1f} GB")

    # Warm-up first, discarded. Without this the very first process on the
    # machine differs from every later run and the comparison reports a false
    # difference -- observed here as old#0 = fe98df1d versus a993fd95 for the
    # following three runs.
    for index in range(a.warmup):
        for arm, directory in (("old", old_dir), ("new", new_dir)):
            dump = Path(f"/tmp/ulp-warm-{arm}-{index}")
            record = run_score(directory, a.model, a.text, dump,
                               f"{arm} warmup#{index}", out)
            if not record["loaded_ok"]:
                emit(out, dict(kind="abort",
                               error=f"{arm} loaded {record['loaded_library']}, "
                                     f"expected {record['loaded_expected']}"))
                return 1
            if record["exit"] != 0 or record["positions"] == 0:
                emit(out, dict(kind="abort", error=f"warm-up run failed for {arm}"))
                return 1
            shutil.rmtree(dump, ignore_errors=True)
        log("warm-up complete")

    # Interleave: old, new, old, new. Same-arm determinism is checked on the
    # repeats, arm equality on the first measured pair.
    runs = {"old": [], "new": []}
    for index in range(a.repeats):
        for arm, directory in (("old", old_dir), ("new", new_dir)):
            dump = Path(f"/tmp/ulp-dump-{arm}-{index}")
            record = run_score(directory, a.model, a.text, dump, f"{arm}#{index}", out)
            if not record["loaded_ok"]:
                emit(out, dict(kind="abort",
                               error=f"{arm} loaded {record['loaded_library']}, "
                                     f"expected {record['loaded_expected']}"))
                return 1
            runs[arm].append(record)
            shutil.rmtree(dump, ignore_errors=True)

    failures = []
    for arm in ("old", "new"):
        digests = {r["digest"] for r in runs[arm]}
        if len(digests) != 1:
            failures.append(f"{arm} is not deterministic across repeats: {sorted(digests)}")
        if any(r["exit"] != 0 for r in runs[arm]):
            failures.append(f"{arm} had a non-zero exit")
        if any(r["positions"] == 0 for r in runs[arm]):
            failures.append(f"{arm} dumped no positions")
    if not failures:
        if runs["old"][0]["digest"] != runs["new"][0]["digest"]:
            failures.append("old and new logits differ")
        elif runs["old"][0]["positions"] != runs["new"][0]["positions"]:
            failures.append("old and new dumped different position counts")

    verdict = dict(kind="verdict", pass_=not failures, failures=failures,
                   old_digest=runs["old"][0]["digest"] if runs["old"] else None,
                   new_digest=runs["new"][0]["digest"] if runs["new"] else None,
                   positions=runs["old"][0]["positions"] if runs["old"] else 0,
                   old_lib=old_lib[:16], new_lib=new_lib[:16])
    emit(out, verdict)
    for item in failures:
        log(f"FAIL {item}")
    log(f"VERDICT: {'PASS - bit-identical' if not failures else 'FAIL'}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
