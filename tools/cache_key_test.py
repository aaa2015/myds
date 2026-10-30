#!/usr/bin/env python3
"""Does the Metal cache key on library content, or only on path?

Ledger §126 established that a cold cache changes the first run's numerics. The
dangerous follow-up is the opposite failure: replacing the library in place while
leaving the cache populated. If the cache is keyed by path rather than content, a
new library could be served artefacts built for the old one -- silently, and with
the old library's results.

Semantics under test at one fixed path:
  1. load library A twice        -> digest_A (second run is the stable value)
  2. swap in library B, run once -> which digest?
       equals digest_A  => cache reused across differing content: contamination
       a new value      => cache keyed on content (or at least invalidated)
  3. run B again                 -> digest_B, stable
  4. swap A back, run once       -> should be digest_A again if keyed on content

No model weights are modified; this only permutes two metallib files under one path.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ulp_metallib_compare as cmp  # noqa: E402


def log(msg):
    print(f"[{msg}]", flush=True)


def emit(path, record):
    with open(path, "a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lib-a", required=True)
    ap.add_argument("--lib-b", required=True)
    ap.add_argument("--score", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--runs", type=int, default=2, help="runs per state")
    for key, value in cmp.DEFAULTS.items():
        ap.add_argument("--" + key.replace("_", "-"), default=value, type=type(value))
    a = ap.parse_args()

    out = Path(a.out)
    if out.exists():
        ap.error("--out already exists")
    lib_a, lib_b = Path(a.lib_a), Path(a.lib_b)
    sha_a, sha_b = cmp.sha256(lib_a), cmp.sha256(lib_b)
    if sha_a == sha_b:
        ap.error("the two libraries are identical; nothing to test")

    arm = Path("/tmp/cachekey-arm")
    shutil.rmtree(arm, ignore_errors=True)
    arm.mkdir()
    shutil.copy2(a.score, arm / "score")

    emit(out, dict(kind="plan", lib_a=sha_a[:16], lib_b=sha_b[:16],
                   path=str(arm / "myds.metallib"), runs=a.runs,
                   question="does swapping the library at a fixed path reuse cached "
                            "artefacts built for the previous content?"))
    log(f"A={sha_a[:16]} B={sha_b[:16]}")
    log(f"available {cmp.wait_memory(a.mem_floor_gb):.1f} GB")

    def run(tag):
        dump = Path(f"/tmp/cachekey-{tag}")
        record = cmp.run_score(arm, a.model, a.text, dump, tag, out)
        shutil.rmtree(dump, ignore_errors=True)
        if not record["loaded_ok"] or record["exit"] != 0 or record["positions"] == 0:
            emit(out, dict(kind="abort", tag=tag, error="bad run"))
            raise SystemExit(f"bad run at {tag}")
        return record["digest"]

    results = {}

    shutil.copy2(lib_a, arm / "myds.metallib")
    results["A_first"] = run("A_first")
    results["A_second"] = run("A_second")

    # swap content at the same path, cache untouched
    shutil.copy2(lib_b, arm / "myds.metallib")
    results["B_first_after_swap"] = run("B_first_after_swap")
    results["B_second"] = run("B_second")

    shutil.copy2(lib_a, arm / "myds.metallib")
    results["A_again_after_B"] = run("A_again_after_B")

    verdict = dict(kind="verdict", digests={k: v[:16] for k, v in results.items()},
                   a_stable=results["A_first"] == results["A_second"],
                   b_stable=results["B_second"] == results["B_first_after_swap"])
    # The failure this looks for: B's first run reproducing A's value.
    verdict["contaminated"] = results["B_first_after_swap"] == results["A_second"]
    verdict["a_restored"] = results["A_again_after_B"] == results["A_second"]
    emit(out, verdict)
    for key, value in verdict["digests"].items():
        log(f"  {key:22s} {value}")
    log(f"contaminated (B first run == A digest): {verdict['contaminated']}")
    log(f"A restored after B: {verdict['a_restored']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
