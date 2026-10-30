#!/usr/bin/env python3
"""Sample the digest distribution of one library across many cold processes.

Why: three times now a single run disagreed with all its neighbours (fe98df1d,
8093312c, 2c3ba4c9) while repeated runs agreed. Ledger §124 records that as
unexplained. Before choosing a judgement rule for A/B comparisons we need to know
whether the machine is deterministic at all: if 5% of cold starts produce different
logits, then "all samples equal" is the wrong test and every past comparison that
relied on it needs restating.

This does not try to explain the mechanism. It measures the distribution, records
the order in which each value appeared, and says nothing about causes.

One library, N separate processes, no warm-up discarding -- the warm-up is exactly
what we are trying to observe.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ulp_metallib_compare as cmp  # noqa: E402


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def emit(path, record):
    with open(path, "a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm-dir", required=True, help="directory with score + myds.metallib")
    ap.add_argument("--text", required=True)
    ap.add_argument("--runs", type=int, default=30)
    for key, value in cmp.DEFAULTS.items():
        ap.add_argument("--" + key.replace("_", "-"), default=value, type=type(value))
    a = ap.parse_args()

    out = Path(a.out)
    if out.exists():
        ap.error("--out already exists")
    arm = Path(a.arm_dir).resolve()
    for name in ("score", "myds.metallib"):
        if not (arm / name).is_file():
            ap.error(f"{arm} is missing {name}")

    library = cmp.sha256(arm / "myds.metallib")
    emit(out, dict(kind="plan", arm=str(arm), library=library, runs=a.runs,
                   text=a.text, machine=os.uname().nodename,
                   note="one library, N cold processes; no warm-up discarded"))
    log(f"library {library[:16]}, {a.runs} cold runs")
    log(f"available {cmp.wait_memory(a.mem_floor_gb):.1f} GB")

    results = []
    for index in range(a.runs):
        dump = Path(f"/tmp/coldrepeat-{index}")
        record = cmp.run_score(arm, a.model, a.text, dump, f"cold#{index}", out)
        shutil.rmtree(dump, ignore_errors=True)
        if not record["loaded_ok"]:
            emit(out, dict(kind="abort", tag=f"cold#{index}",
                           error=f"loaded {record['loaded_library']}"))
            raise SystemExit("wrong library loaded; the distribution would be meaningless")
        if record["exit"] != 0 or record["positions"] == 0:
            emit(out, dict(kind="abort", tag=f"cold#{index}", error="failed run or empty dump"))
            raise SystemExit("run failed; distribution incomplete")
        results.append(dict(index=index, digest=record["digest"],
                            seconds=record["seconds"],
                            positions=record["positions"]))

    counts = {}
    first_seen = {}
    for item in results:
        digest = item["digest"]
        counts[digest] = counts.get(digest, 0) + 1
        first_seen.setdefault(digest, item["index"])

    total = len(results)
    majority = max(counts, key=counts.get)
    summary = dict(
        kind="distribution",
        library=library,
        runs=total,
        distinct=len(counts),
        counts={d[:16]: c for d, c in counts.items()},
        first_seen={d[:16]: i for d, i in first_seen.items()},
        majority=majority[:16],
        majority_fraction=counts[majority] / total,
        deterministic=(len(counts) == 1),
        outliers=[dict(index=item["index"], digest=item["digest"][:16])
                  for item in results if item["digest"] != majority],
        positions=results[0]["positions"],
        seconds_median=sorted(item["seconds"] for item in results)[total // 2],
    )
    emit(out, summary)
    log(f"distinct digests: {len(counts)} over {total} runs")
    for digest, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        log(f"  {digest[:16]}  x{count}  first at #{first_seen[digest]}")
    if summary["deterministic"]:
        log("VERDICT: deterministic -- every cold start produced the same digest")
    else:
        log(f"VERDICT: NOT deterministic -- {100*(1-summary['majority_fraction']):.1f}% of "
            f"runs deviated from the majority")
    return 0


if __name__ == "__main__":
    sys.exit(main())
