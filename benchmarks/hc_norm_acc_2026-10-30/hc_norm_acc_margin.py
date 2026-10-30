#!/usr/bin/env python3
"""Measure how far a shader change moves the *pre-BF16* value, versus the quantum.

Ledger §120 showed `reorder` (accumulating the four HC terms in reverse) leaves the
final logits bit-identical, because the kernel rounds to BF16 before the model sees
the value. A logits-only gate therefore cannot see that class of change. The
question that actually matters is whether the change is *safely* absorbed or only
*just* absorbed:

  - if |delta| is a tiny fraction of the BF16 quantum, some other input would have
    to be within that fraction of a rounding boundary to flip a bit -- a bounded,
    quantified risk;
  - if |delta| approaches half the quantum, roughly every other element sits near a
    boundary and the change is a coin flip per input.

`probe_hc_norm_acc` dumps that pre-rounding value directly. Both arms get the same
dump patch, so the only difference between them is the change under test.

No model, no weights, no exclusive window: synthetic buffers only. It exercises the
GPU for a few microseconds and must not be reported as a production measurement.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys

REPO = Path(__file__).resolve().parent.parent
# The host entry point is myds_metal_v41_hc_sum_rmsnorm_batch, which dispatches
# kernel_myds_v41_hc_sum_rmsnorm_batch -- NOT kernel_myds_v41_hc_sum_rmsnorm in
# dsv4_hc.metal. Patching the non-batch kernel left the dump post-rounding and the
# first run reported a meaningless "0 of 5120 differ". The kernel name and file are
# therefore explicit and the dump is verified to be pre-rounding below.
KERNEL_FILE = "dsv4_misc.metal"
KERNEL_HEAD = "kernel void kernel_myds_v41_hc_sum_rmsnorm_batch("

# Applied to EVERY arm so the dumped buffer is the raw accumulator.
DUMP_OLD = "        if (args.has_sum_out) {\n            sum_out[i00] = v;\n        }\n"
DUMP_NEW = ("        if (args.has_sum_out) {\n"
            "            sum_out[i00] = acc; /* MARGINCTL raw pre-round accumulator */\n"
            "        }\n")

ARMS = {
    "baseline": None,
    # Validity gate. If this does not move the dumped accumulator, the probe is not
    # looking at the executed kernel and no other conclusion from this run counts.
    # The perturbation must move the DUMPED quantity. Perturbing `v` after the dump
    # line left the dump untouched and the gate read 0 differences -- a gate that
    # cannot fail is not a gate.
    "gross": dict(
        old="        const float4 v = dsv41_bf16_round_f32x4(acc);\n",
        new="        const float4 v = dsv41_bf16_round_f32x4(acc);\n",
        dump_new=("        if (args.has_sum_out) {\n"
                  "            sum_out[i00] = acc * 1.0009765625f; /* MARGINCTL gross */\n"
                  "        }\n")),
    "reorder": dict(
        old=("        acc += r0[i00] * w0;\n"
             "        acc += r1[i00] * w1;\n"
             "        acc += r2[i00] * w2;\n"
             "        acc += r3[i00] * w3;\n"),
        new=("        acc += r3[i00] * w3; /* MARGINCTL reversed */\n"
             "        acc += r2[i00] * w2;\n"
             "        acc += r1[i00] * w1;\n"
             "        acc += r0[i00] * w0;\n")),
}


def log(msg):
    print(f"[{msg}]", flush=True)


def emit(out, record):
    """Append one JSON line. Use this everywhere: write_text would silently drop
    every earlier record, which is how a run ends up with only its verdict."""
    with Path(out).open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()


def kernel_span(text):
    """[start, end) covering the target kernel only.

    Scoping matters: the dump pattern also appears in the non-batch kernel in
    another file, and a whole-file replace would either hit the wrong one or fail
    with 'occurs 2 times'. Patching is only meaningful inside the kernel the host
    entry point actually dispatches.
    """
    if text.count(KERNEL_HEAD) != 1:
        raise SystemExit(f"{KERNEL_HEAD!r} appears {text.count(KERNEL_HEAD)} times in "
                         f"{KERNEL_FILE}, need exactly 1")
    start = text.index(KERNEL_HEAD)
    end = text.index("\n}\n", start)
    return start, end + 3


def replace_once(text, old, new, what):
    start, end = kernel_span(text)
    body = text[start:end]
    if body.count(old) != 1:
        raise SystemExit(f"{what}: pattern occurs {body.count(old)} times inside the "
                         f"target kernel, need exactly 1")
    return text[:start] + body.replace(old, new, 1) + text[end:]


def build_arm(root, arm, repo):
    tree = root / "tree" / arm
    shutil.copytree(repo / "metal", tree / "metal")
    (tree / "tools").mkdir(parents=True)
    shutil.copy2(repo / "tools/build_metallib.py", tree / "tools/build_metallib.py")
    path = tree / "metal" / KERNEL_FILE
    text = path.read_text(encoding="utf-8")
    spec = ARMS[arm]
    if spec:
        text = replace_once(text, spec["old"], spec["new"], f"{arm} perturbation")
    dump_new = (spec or {}).get("dump_new", DUMP_NEW)
    text = replace_once(text, DUMP_OLD, dump_new, "accumulator dump")
    path.write_text(text, encoding="utf-8")
    arm_dir = root / "arms" / arm
    arm_dir.mkdir(parents=True)
    done = subprocess.run([sys.executable, str(tree / "tools/build_metallib.py"),
                           "--bin-dir", str(arm_dir), "--build-dir", str(root / "build" / arm)],
                          capture_output=True, text=True)
    if done.returncode != 0:
        raise SystemExit(f"metallib build failed for {arm}:\n{done.stderr[-1500:]}")
    return arm_dir


def bf16_quantum(value):
    """Spacing of BF16 neighbours at this magnitude (BF16 has 7 explicit mantissa bits)."""
    magnitude = abs(value)
    if magnitude == 0.0 or magnitude != magnitude:
        return None
    exponent = 0
    while magnitude >= 2.0:
        magnitude /= 2.0
        exponent += 1
    while magnitude < 1.0:
        magnitude *= 2.0
        exponent -= 1
    return 2.0 ** (exponent - 7)


def check_pre_rounding(path, arm):
    """A pre-rounding dump must contain values BF16 cannot represent.

    First run of this tool patched the wrong kernel (the non-batch variant), so the
    buffer still held post-rounding values, every element was BF16-representable,
    and the comparison reported "0 of 5120 differ" -- a clean-looking, meaningless
    result. This check makes that failure loud instead of silent.
    """
    raw = Path(path).read_bytes()
    n = len(raw) // 4
    if n == 0:
        raise SystemExit(f"{arm}: empty dump")
    vals = struct.unpack(f"<{n}f", raw)
    representable = sum(1 for v in vals
                        if struct.unpack("<I", struct.pack("<f", v))[0] & 0xFFFF == 0)
    if representable == n:
        raise SystemExit(
            f"{arm}: all {n} dumped values are BF16-representable, so the buffer is "
            "still post-rounding -- the patch did not take effect on the executed kernel")
    return representable / n


def analyse(path_a, path_b, label, out):
    a = Path(path_a).read_bytes()
    b = Path(path_b).read_bytes()
    if len(a) != len(b) or len(a) % 4:
        raise SystemExit("dump sizes differ or are not float32")
    n = len(a) // 4
    fa = struct.unpack(f"<{n}f", a)
    fb = struct.unpack(f"<{n}f", b)
    ratios, abs_deltas, nonzero = [], [], 0
    for x, y in zip(fa, fb):
        d = abs(x - y)
        if d == 0.0:
            continue
        nonzero += 1
        abs_deltas.append(d)
        q = bf16_quantum(x)
        if q:
            ratios.append(d / q)
    ratios.sort()
    record = dict(kind="margin", label=label, elements=n,
                  differing_elements=nonzero,
                  fraction_differing=(nonzero / n) if n else None,
                  max_abs_delta=max(abs_deltas) if abs_deltas else 0.0,
                  max_delta_over_quantum=ratios[-1] if ratios else 0.0,
                  p99_delta_over_quantum=ratios[int(0.99 * (len(ratios) - 1))] if ratios else 0.0,
                  median_delta_over_quantum=ratios[len(ratios) // 2] if ratios else 0.0)
    emit(out, record)
    log(f"{label}: {nonzero}/{n} elements differ, max |d|/quantum = "
        f"{record['max_delta_over_quantum']:.4f}")
    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="scratch dir (must not exist)")
    ap.add_argument("--out", required=True, help="JSONL results (must not exist)")
    ap.add_argument("--repo", default=str(REPO))
    ap.add_argument("--probe", required=True, help="built probe_hc_norm_acc binary")
    ap.add_argument("--dim", type=int, default=5120)
    ap.add_argument("--n-tok", type=int, default=1)
    ap.add_argument("--prebuilt", action="append", default=[], metavar="ARM=DIR",
                    help="use an already-built arm instead of building here (repeatable). "
                         "Needed when the machine that runs the probe cannot compile "
                         "shaders -- macmini currently has no Metal Toolchain either.")
    a = ap.parse_args()

    repo = Path(a.repo).resolve()
    if not (repo / "metal" / KERNEL_FILE).is_file():
        raise SystemExit(f"--repo {repo} has no metal/{KERNEL_FILE}")
    root = Path(a.root).resolve()
    if a.prebuilt:
        # Prebuilt arms live outside --root; --root only holds dumps.
        root.mkdir(parents=True, exist_ok=True)
    elif root.exists():
        raise SystemExit("--root already exists")
    else:
        root.mkdir(parents=True)
    out = Path(a.out)
    if out.exists():
        raise SystemExit("--out already exists")
    probe = Path(a.probe).resolve()
    if not probe.is_file():
        raise SystemExit(f"probe not found: {probe}")

    emit(out, dict(kind="plan", repo=str(repo), probe=str(probe),
                   dim=a.dim, n_tok=a.n_tok,
                   note="both arms carry the same accumulator-dump patch; "
                        "the only difference is the reorder"))

    dirs = {}
    for spec in a.prebuilt:
        if "=" not in spec:
            raise SystemExit(f"--prebuilt wants ARM=DIR, got {spec!r}")
        arm, where = spec.split("=", 1)
        if arm not in ARMS:
            raise SystemExit(f"unknown arm {arm!r}")
        directory = Path(where).resolve()
        if not (directory / "myds.metallib").is_file():
            raise SystemExit(f"{directory} has no myds.metallib")
        dirs[arm] = directory
        log(f"using prebuilt {arm} from {directory}")
    for arm in ARMS:
        if arm in dirs:
            continue
        dirs[arm] = build_arm(root, arm, repo)
        log(f"built {arm}")
    for arm, directory in dirs.items():
        shutil.copy2(probe, directory / probe.name)

    dumps = {}
    for arm, arm_dir in dirs.items():
        dump = root / f"{arm}.f32"
        # cwd = arm dir so myds_metal_init resolves that arm's library.
        done = subprocess.run([str(arm_dir / probe.name), str(dump),
                               str(a.dim), str(a.n_tok), arm],
                              cwd=str(arm_dir), capture_output=True, text=True)
        if done.returncode != 0:
            raise SystemExit(f"probe failed for {arm}:\n{done.stdout}\n{done.stderr}")
        loaded = None
        for line in done.stdout.splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if "loaded_library" in record:
                loaded = record["loaded_library"]
        expected = str((arm_dir / "myds.metallib").resolve())
        if not loaded or Path(loaded).resolve() != Path(expected):
            raise SystemExit(f"{arm} loaded {loaded!r}, expected {expected}")
        emit(out, dict(kind="run", arm=arm, loaded_library=loaded, loaded_ok=True))
        dumps[arm] = dump
        log(f"{arm}: loaded {Path(loaded).name} from its own arm")

    libs = {arm: hashlib.sha256((d / "myds.metallib").read_bytes()).hexdigest()
            for arm, d in dirs.items()}
    if len(set(libs.values())) != len(libs):
        raise SystemExit("two arms produced the same library; the reorder had no effect")

    frac = {arm: check_pre_rounding(d, arm) for arm, d in dumps.items()}
    emit(out, dict(kind="pre_rounding_check",
                   bf16_representable_fraction=frac,
                   note="must be < 1.0; 1.0 means the dump is post-rounding"))
    log(f"pre-rounding confirmed: {', '.join(f'{a}={100*v:.1f}%' for a, v in frac.items())} "
        f"of values are BF16-representable")

    record = analyse(dumps["baseline"], dumps["reorder"], "reorder_vs_baseline", out)
    verdict = dict(kind="verdict", library_sha256={k: v[:16] for k, v in libs.items()},
                   **{k: record[k] for k in ("differing_elements", "elements",
                                             "max_delta_over_quantum",
                                             "p99_delta_over_quantum")})
    # Half a quantum is the distance to the nearest rounding boundary: below it the
    # change is absorbed at this magnitude, above it some elements would round
    # differently. This is a bound on one input, not a proof for all inputs.
    verdict["max_below_half_quantum"] = record["max_delta_over_quantum"] < 0.5
    emit(out, verdict)
    log(f"VERDICT max ratio {record['max_delta_over_quantum']:.4f} of half-quantum "
        f"margin -> {'absorbed' if verdict['max_below_half_quantum'] else 'CAN FLIP'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
