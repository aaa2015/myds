#!/usr/bin/env python3
"""Quantify the difference between the JIT path and the cache-load path.

Ledger §126/§127 established that the first run after a cache invalidation produces
different logits than every run after it, reproducibly and independently of which
library is loaded. That says the two code paths disagree; it does not say by how
much, and without the magnitude the question "which one is correct" cannot even be
framed:

  - a one-ULP difference means both paths round differently but are equally valid,
    and the correct response is to pick one and pin it;
  - a large difference means one path is computing something else, which is a
    correctness problem rather than a rounding one.

Method: clear the cache, run once keeping the dump (JIT), run again keeping the dump
(cache), then compare element by element and in ULPs at each value's own exponent.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ulp_metallib_compare as cmp  # noqa: E402
import first_run_cause as frc  # noqa: E402


def log(msg):
    print(f"[{msg}]", flush=True)


def emit(path, record):
    with open(path, "a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()


def ulp_distance(a, b):
    """Signed-magnitude integer distance between two float32 values.

    Comparing bit patterns as ordered integers is the standard ULP metric: it
    counts representable steps, so a difference of 1 is the smallest change the
    format can express at that magnitude.
    """
    def key(x):
        bits = struct.unpack("<I", struct.pack("<f", x))[0]
        return bits - 0x80000000 if bits & 0x80000000 else bits + 0x80000000

    if a != a or b != b:            # NaN: ULP distance is undefined
        return None
    return abs(key(a) - key(b))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--source-arm", required=True, help="dir with score + myds.metallib")
    ap.add_argument("--text", required=True)
    ap.add_argument("--keep-dumps", default=None,
                    help="directory to keep the two dumps in (default: discard)")
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

    work = Path("/tmp/jitcache-arm")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir()
    shutil.copy2(source / "score", work / "score")
    shutil.copy2(source / "myds.metallib", work / "myds.metallib")
    library = cmp.sha256(work / "myds.metallib")

    emit(out, dict(kind="plan", library=library[:16], text=a.text,
                   note="run 1 after cache clear = JIT path; run 2 = cache path"))
    log(f"library {library[:16]}")
    log(f"available {cmp.wait_memory(a.mem_floor_gb):.1f} GB")

    if not frc.cache_dirs():
        ap.error("Metal cache directory not found")

    keep = Path(a.keep_dumps) if a.keep_dumps else None
    if keep:
        shutil.rmtree(keep, ignore_errors=True)
        keep.mkdir(parents=True)

    dumps = {}
    for tag, clear in (("jit", True), ("cache", False)):
        if clear:
            removed = frc.clear_cache()
            state = frc.cache_state()
            emit(out, dict(kind="cache", phase=tag, cleared=removed, **state))
            log(f"{tag}: cleared {removed} dir(s), {state['entries']} entries")
        dump = (keep / tag) if keep else Path(f"/tmp/jitcache-{tag}")
        record = cmp.run_score(work, a.model, a.text, dump, tag, out)
        if not record["loaded_ok"] or record["exit"] != 0 or record["positions"] == 0:
            emit(out, dict(kind="abort", tag=tag, error="bad run"))
            raise SystemExit(f"bad run at {tag}")
        dumps[tag] = dump
        log(f"{tag}: {record['positions']} positions, digest {record['digest'][:16]}")
        if not keep:
            pass    # keep for comparison; removed at the end

    # compare every float of every position
    files_jit = sorted(dumps["jit"].glob("*.f32"))
    files_cache = sorted(dumps["cache"].glob("*.f32"))
    if len(files_jit) != len(files_cache):
        emit(out, dict(kind="abort", error="different position counts"))
        raise SystemExit("differing position counts")
    if len(files_jit) == 0:
        emit(out, dict(kind="abort", error="no dumps"))
        raise SystemExit("no dumps")

    total = 0
    differing = 0
    max_ulp = 0
    max_abs = 0.0
    first_position = None
    per_position = []
    for fa, fb in zip(files_jit, files_cache):
        ra, rb = fa.read_bytes(), fb.read_bytes()
        if len(ra) != len(rb) or len(ra) % 4:
            emit(out, dict(kind="abort", error="dump size mismatch"))
            raise SystemExit("dump size mismatch")
        n = len(ra) // 4
        va = struct.unpack(f"<{n}f", ra)
        vb = struct.unpack(f"<{n}f", rb)
        local = 0
        for x, y in zip(va, vb):
            total += 1
            if x == y:
                continue
            differing += 1
            local += 1
            d = ulp_distance(x, y)
            if d is not None:
                max_ulp = max(max_ulp, d)
            max_abs = max(max_abs, abs(x - y))
            if first_position is None:
                first_position = fa.name
        if local:
            per_position.append(dict(file=fa.name, differing=local, of=n))

    summary = dict(kind="diff", library=library[:16],
                   positions=len(files_jit),
                   elements=total,
                   differing_elements=differing,
                   differing_fraction=differing / total if total else 0.0,
                   max_ulp=max_ulp,
                   max_abs_delta=max_abs,
                   first_differing_position=first_position,
                   differing_positions=len(per_position),
                   first_few=per_position[:5])
    emit(out, summary)
    log(f"{differing}/{total} elements differ across {len(per_position)} of "
        f"{len(files_jit)} positions; max ULP {max_ulp}, max |delta| {max_abs:.6g}")
    log(f"first differing position: {first_position}")
    if differing == 0:
        log("VERDICT: JIT and cache paths agree bit-for-bit on this input "
            "(the earlier difference must come from elsewhere)")
    elif max_ulp <= 1:
        log("VERDICT: 1 ULP -- same arithmetic, different rounding; neither is more "
            "correct, pick one and pin it")
    else:
        log(f"VERDICT: up to {max_ulp} ULP -- a real numerical difference, not rounding")
    if not keep:
        for dump in dumps.values():
            shutil.rmtree(dump, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
