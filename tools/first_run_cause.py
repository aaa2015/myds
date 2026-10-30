#!/usr/bin/env python3
"""Separate the three candidate causes of the first-run anomaly.

Ledger §125 measured 30 cold starts of one library, all identical, so the baseline
is deterministic -- yet three earlier runs disagreed with every neighbour, each on
the first run of its session. This runs the experiments that can tell the
hypotheses apart instead of guessing between them:

  H1 fresh library   a library file that has never been loaded behaves differently
                     on first use
  H2 cold cache      an empty Metal compile cache changes the first result
  H3 base rate       it is simply rare, and 30 clean runs did not rule it out

Design (single window, ~2-4 minutes with a small input, not 29 like §125):

  phase A  library X, warm cache        N runs  -> replicates §125
  phase B  library Y (fresh copy), warm cache, no prior warm-up  N runs -> H1
  phase C  library X, cache cleared     N runs  -> H2
  phase D  fresh copy of Z, cache cleared                        N runs -> H1*H2

Every run records the digest, the library path actually loaded, and whether the
Metal cache was empty immediately beforehand, so the phases are verifiable from the
record instead of from the plan.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ulp_metallib_compare as cmp  # noqa: E402

# The Metal compile cache lives in the per-user temp dir; the exact path includes a
# random component, so it is discovered rather than hard-coded.
CACHE_GLOB = "/var/folders/*/*/C/com.apple.metal"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def emit(path, record):
    with open(path, "a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()


def cache_dirs():
    import glob
    return [Path(p) for p in glob.glob(CACHE_GLOB)]


def cache_state():
    dirs = cache_dirs()
    files = sum(len(list(d.rglob("*"))) for d in dirs if d.is_dir())
    size = sum(f.stat().st_size for d in dirs if d.is_dir()
               for f in d.rglob("*") if f.is_file())
    return dict(dirs=[str(d) for d in dirs], entries=files, bytes=size)


def clear_cache():
    """Remove the Metal compile cache. Only meaningful when no myds process runs."""
    removed = 0
    for directory in cache_dirs():
        if directory.is_dir():
            shutil.rmtree(directory, ignore_errors=True)
            removed += 1
    return removed


def run_once(arm_dir, model, text, index, out, tag):
    dump = Path(f"/tmp/firstrun-{tag}-{index}")
    record = cmp.run_score(arm_dir, model, text, dump, f"{tag}#{index}", out)
    shutil.rmtree(dump, ignore_errors=True)
    if not record["loaded_ok"]:
        emit(out, dict(kind="abort", tag=f"{tag}#{index}",
                       error=f"loaded {record['loaded_library']}"))
        raise SystemExit("wrong library loaded")
    if record["exit"] != 0 or record["positions"] == 0:
        emit(out, dict(kind="abort", tag=f"{tag}#{index}", error="failed run"))
        raise SystemExit("run failed")
    return record["digest"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--source-arm", required=True,
                    help="directory holding score + myds.metallib to copy and vary")
    ap.add_argument("--text", required=True)
    ap.add_argument("--runs", type=int, default=8, help="runs per phase")
    for key, value in cmp.DEFAULTS.items():
        ap.add_argument("--" + key.replace("_", "-"), default=value, type=type(value))
    a = ap.parse_args()

    out = Path(a.out)
    if out.exists():
        ap.error("--out already exists")
    source = Path(a.source_arm).resolve()
    for name in ("score", "myds.metallib"):
        if not (source / name).is_file():
            ap.error(f"{source} is missing {name}")

    work = Path("/tmp/firstrun-arms")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)

    # X: the library §125 already characterised as deterministic.
    # Y: a byte-identical copy under a new path, never loaded before.
    x = work / "X"
    x.mkdir()
    shutil.copy2(source / "score", x / "score")
    shutil.copy2(source / "myds.metallib", x / "myds.metallib")
    y = work / "Y"
    y.mkdir()
    shutil.copy2(source / "score", y / "score")
    shutil.copy2(source / "myds.metallib", y / "myds.metallib")

    lib_x = cmp.sha256(x / "myds.metallib")
    lib_y = cmp.sha256(y / "myds.metallib")
    if lib_x != lib_y:
        ap.error("X and Y must be byte-identical libraries")
    if not cache_dirs():
        ap.error("Metal compile cache directory not found; cannot test H2")

    emit(out, dict(kind="plan", runs=a.runs, library=lib_x, text=a.text,
                   machine=os.uname().nodename,
                   phases=["A X warm", "B Y warm (fresh file)", "C X cold cache",
                           "D Y cold cache (fresh file + empty cache)"],
                   note="X and Y are byte-identical; only the path and prior use differ"))
    log(f"library {lib_x[:16]}, {a.runs} runs per phase")

    initial_cache = cache_state()
    emit(out, dict(kind="cache", phase="initial", **initial_cache))
    log(f"cache before: {initial_cache['entries']} files, "
        f"{initial_cache['bytes'] / 1e6:.0f} MB")

    results = {}

    def phase(tag, arm, clear, runs):
        if clear:
            removed = clear_cache()
            state = cache_state()
            log(f"{tag}: cleared {removed} cache dir(s), now {state['entries']} files")
            emit(out, dict(kind="cache", phase=tag, cleared=removed, **state))
        digests = []
        for index in range(runs):
            digests.append(run_once(arm, a.model, a.text, index, out, tag))
        results[tag] = digests
        distinct = sorted(set(digests))
        log(f"{tag}: {len(distinct)} distinct of {runs}"
            f" ({', '.join(d[:8] for d in distinct)})")

    phase("A_X_warm", x, False, a.runs)
    phase("B_Y_warm_fresh_file", y, False, a.runs)
    phase("C_X_cold_cache", x, True, a.runs)
    phase("D_Y_cold_cache_fresh_file", y, True, a.runs)

    summary = dict(kind="summary",
                   library=lib_x[:16],
                   runs_per_phase=a.runs,
                   phases={tag: dict(distinct=len(set(v)),
                                     counts={d[:16]: v.count(d) for d in set(v)})
                           for tag, v in results.items()})
    # Each hypothesis predicts a specific pattern; state them before looking so the
    # record cannot be read backwards into a preferred story.
    summary["predictions"] = {
        "H1_fresh_library": "B and D show a first-run anomaly, A and C do not",
        "H2_cold_cache": "C and D show a first-run anomaly, A and B do not",
        "H3_base_rate": "anomalies appear in any phase, not concentrated at index 0",
        "none": "no anomaly in any phase",
    }
    first_run_deviations = {}
    for tag, digests in results.items():
        majority = max(set(digests), key=digests.count)
        first_run_deviations[tag] = digests[0] != majority
    summary["first_run_deviation"] = first_run_deviations
    any_deviation = any(len(set(v)) > 1 for v in results.values())
    summary["any_anomaly"] = any_deviation
    summary["verdict"] = ("no anomaly reproduced" if not any_deviation
                          else "anomaly observed -- see first_run_deviation")
    emit(out, summary)

    log("---")
    log(f"first-run deviation per phase: {first_run_deviations}")
    log(f"VERDICT: {summary['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
