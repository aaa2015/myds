#!/usr/bin/env python3
"""Negative control for the 0-ULP logits comparison (ledger §118 follow-up).

§118 showed two metallibs agree on one input, but none of its negative controls
managed to produce a failure, so "this comparison can detect a shader-level
numeric change" was never demonstrated. This tool builds libraries that are
KNOWN to change arithmetic on the path `score` executes, and checks whether the
comparison sees them.

Nothing here edits metal/. `prepare` copies metal/ and tools/build_metallib.py
into an isolated tree, applies one textual perturbation per arm (asserting the
target occurs exactly once inside the target kernel), and builds every arm --
including an unpatched baseline -- with the same toolchain in the same session,
so the patch is the only variable between baseline and arm.

Target: kernel_myds_v41_hc_sum_rmsnorm (metal/dsv4_hc.metal). The per-token
forward that `score` uses calls it before every layer's attention/FFN and for the
final output norm (src/myds_v41.c v41_hc_sum_rmsnorm_scoped, fused path is the
default unless MYDS_FUSE_HC_SUM_RMSNORM=0). Its output is BF16-rounded on that
path (round_bf16=true), which matters: a one-f32-ULP perturbation before the
rounding is usually absorbed, and is a plausible reason earlier controls showed
nothing. Arm b therefore perturbs by one BF16 ULP AFTER rounding.

Arms (dose ladder):
  gross     output * (1 + 2^-6)                      expected: DIFFER (validity gate)
  bf16ulp   one element, +1 BF16 ULP after rounding  expected: DIFFER (sensitivity)
  f32ulp    one element, +1 f32 ULP before rounding  informational (shows masking)
  reorder   HC weighted-sum terms accumulated c3..c0 informational (real bug class)

Validity rules (all evaluated by `run`):
  * every run loads its own arm's library (loaded_ok)            -- else abort
  * `gross` must differ from every baseline digest               -- else the whole
    run is INVALID: the kernel is not on the executed path or the dump is blind
  * baseline determinism is measured, not assumed: N cold processes; the set of
    baseline digests is reported, and an arm "differs" only if its digest is
    outside that set and the arm repeats itself
  * coverage: `score` is the per-token path only; batched prefill is not covered

`prepare` can run anywhere with the Metal toolchain. `run` needs the model and an
exclusive window (~150 GB wired on a 192 GB machine).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ulp_metallib_compare as cmp  # noqa: E402  (run_score, wait_memory, sha256)

# Resolved from --repo, not from __file__: this tool gets copied next to the
# window harness in /tmp, where parent.parent is not the repository and every
# relative path silently points at nothing. That exact mistake has now happened
# three times on this project (prompt corpus, witness staging, this), so the
# repository root is an explicit argument rather than an inference.
_DEFAULT_REPO = Path(__file__).resolve().parent.parent
KERNEL_FILE = "dsv4_hc.metal"
KERNEL_HEAD = "kernel void kernel_myds_v41_hc_sum_rmsnorm("

NORM_LINE = "        const float4 norm_v = (v * scale) * norm_w[i00];\n"
STORE_LINE = "        dst[i00] = round_bf16 ? dsv41_bf16_round_f32x4(norm_v) : norm_v;\n"
ACC_BLOCK = ("        acc += c0 * w0;\n"
             "        acc += c1 * w1;\n"
             "        acc += c2 * w2;\n"
             "        acc += c3 * w3;\n")

ARMS = {
    "gross": dict(
        expect="differ", gate=True,
        old=NORM_LINE,
        new="        const float4 norm_v = ((v * scale) * norm_w[i00]) * 1.015625f; /* NEGCTL gross */\n"),
    "bf16ulp": dict(
        expect="differ", gate=False,
        old=STORE_LINE,
        new=("        float4 negctl_o = round_bf16 ? dsv41_bf16_round_f32x4(norm_v) : norm_v;\n"
             "        if (i00 == 0) { negctl_o.x = as_type<float>(as_type<uint>(negctl_o.x) + 0x10000u); } /* NEGCTL +1 bf16 ulp */\n"
             "        dst[i00] = negctl_o;\n")),
    "f32ulp": dict(
        expect="informational", gate=False,
        old=STORE_LINE,
        new=("        float4 negctl_v = norm_v;\n"
             "        if (i00 == 0) { negctl_v.x = as_type<float>(as_type<uint>(negctl_v.x) + 1u); } /* NEGCTL +1 f32 ulp pre-round */\n"
             "        dst[i00] = round_bf16 ? dsv41_bf16_round_f32x4(negctl_v) : negctl_v;\n")),
    "reorder": dict(
        expect="informational", gate=False,
        old=ACC_BLOCK,
        new=("        acc += c3 * w3; /* NEGCTL reversed accumulation */\n"
             "        acc += c2 * w2;\n"
             "        acc += c1 * w1;\n"
             "        acc += c0 * w0;\n")),
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def kernel_span(text):
    """[start, end) of the target kernel body; end is just past its closing brace."""
    start = text.find(KERNEL_HEAD)
    if start < 0 or text.find(KERNEL_HEAD, start + 1) >= 0:
        raise SystemExit(f"{KERNEL_HEAD!r} must occur exactly once in {KERNEL_FILE}")
    end = text.find("\n}\n", start)
    if end < 0:
        raise SystemExit("could not find the end of the target kernel")
    return start, end + 3


def patch(text, old, new):
    start, end = kernel_span(text)
    body = text[start:end]
    if body.count(old) != 1:
        raise SystemExit(f"patch target occurs {body.count(old)} times in the kernel, need 1:\n{old}")
    return text[:start] + body.replace(old, new) + text[end:]


def build(tree, bin_dir, build_dir):
    bin_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for key in ("MYDS_METALLIB_DEFINES", "MYDS_METAL_STD", "MYDS_METAL_MINVER"):
        if env.get(key):
            raise SystemExit(f"{key} is set; unset it so every arm uses identical flags")
    done = subprocess.run([sys.executable, str(tree / "tools/build_metallib.py"),
                           "--bin-dir", str(bin_dir), "--build-dir", str(build_dir)],
                          capture_output=True, text=True, env=env)
    if done.returncode != 0:
        raise SystemExit(f"metallib build failed in {tree}:\n{done.stderr[-2000:]}")
    return bin_dir / "myds.metallib"


def prepare(a):
    repo = Path(a.repo).resolve()
    if not (repo / "metal" / KERNEL_FILE).is_file():
        raise SystemExit(f"--repo {repo} does not contain metal/{KERNEL_FILE}; "
                         "pass the repository root explicitly when running from a copy")
    root = Path(a.root).resolve()
    if root.exists():
        raise SystemExit("--root already exists")
    score = Path(a.score).resolve()
    if not score.is_file():
        raise SystemExit(f"score binary not found: {score} (build it with `make bin/score`)")
    root.mkdir(parents=True)
    manifest = dict(kind="manifest", created=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    repo_root=str(repo),
                    repo_head=subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                             capture_output=True, text=True).stdout.strip(),
                    score_sha256=cmp.sha256(score), kernel=KERNEL_HEAD, arms={})
    pristine = (repo / "metal" / KERNEL_FILE).read_text(encoding="utf-8")
    kernel_span(pristine)  # fail early if the kernel moved
    for arm in ["baseline"] + list(ARMS):
        tree = root / "tree" / arm
        shutil.copytree(repo / "metal", tree / "metal")
        (tree / "tools").mkdir(parents=True)
        shutil.copy2(repo / "tools/build_metallib.py", tree / "tools/build_metallib.py")
        if arm != "baseline":
            spec = ARMS[arm]
            patched = patch(pristine, spec["old"], spec["new"])
            (tree / "metal" / KERNEL_FILE).write_text(patched, encoding="utf-8")
            diff = subprocess.run(["diff", "-u", str(repo / "metal" / KERNEL_FILE),
                                   str(tree / "metal" / KERNEL_FILE)],
                                  capture_output=True, text=True).stdout
            (root / f"{arm}.patch").write_text(diff, encoding="utf-8")
        arm_dir = root / "arms" / arm
        lib = build(tree, arm_dir, root / "build" / arm)
        shutil.copy2(score, arm_dir / "score")
        manifest["arms"][arm] = dict(metallib_sha256=cmp.sha256(lib),
                                     expect="identical" if arm == "baseline" else ARMS[arm]["expect"],
                                     gate=arm == "gross")
        log(f"built {arm}: {manifest['arms'][arm]['metallib_sha256'][:16]}")
    libs = [v["metallib_sha256"] for v in manifest["arms"].values()]
    if len(set(libs)) != len(libs):
        raise SystemExit("two arms produced the same metallib: a patch had no effect on the "
                         "library bytes, so that arm cannot test anything")
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    log(f"prepared {len(libs)} libraries under {root}")
    return 0


def run(a):
    root = Path(a.root).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    out = Path(a.out)
    if out.exists():
        raise SystemExit("--out already exists")
    if os.environ.get("MYDS_FUSE_HC_SUM_RMSNORM", "1") == "0":
        raise SystemExit("MYDS_FUSE_HC_SUM_RMSNORM=0 routes around the patched kernel")
    arms_dir = root / "arms"
    for arm, info in manifest["arms"].items():
        if cmp.sha256(arms_dir / arm / "myds.metallib") != info["metallib_sha256"]:
            raise SystemExit(f"{arm} library changed since prepare")
        if cmp.sha256(arms_dir / arm / "score") != manifest["score_sha256"]:
            raise SystemExit(f"{arm} score binary differs from the manifest")
    cmp.emit(out, dict(manifest, kind="plan", text=a.text, model=a.model,
                       baseline_repeats=a.baseline_repeats, arm_repeats=a.arm_repeats))
    log(f"available {cmp.wait_memory(a.mem_floor_gb):.1f} GB")

    def one(arm, tag):
        rec = cmp.run_score(arms_dir / arm, a.model, a.text, Path(f"/tmp/ulpneg-{arm}"),
                            tag, out)
        shutil.rmtree(f"/tmp/ulpneg-{arm}", ignore_errors=True)
        if not rec["loaded_ok"] or rec["exit"] != 0 or rec["positions"] == 0:
            cmp.emit(out, dict(kind="abort", tag=tag, error="wrong library, failed run or empty dump"))
            raise SystemExit(f"abort at {tag}")
        return rec["digest"]

    one("baseline", "baseline warmup#0")  # discarded: §118 cold-start artifact
    base = [one("baseline", f"baseline#{i}") for i in range(a.baseline_repeats)]
    results = {}
    for arm in ARMS:
        digests = []
        for i in range(a.arm_repeats):
            digests.append(one(arm, f"{arm}#{i}"))
            base.append(one("baseline", f"baseline-interleave-{arm}#{i}"))
        results[arm] = digests

    base_set = sorted(set(base))
    verdicts, invalid = {}, []
    for arm, digests in results.items():
        self_consistent = len(set(digests)) == 1
        outside = all(d not in base_set for d in digests)
        observed = "differ" if (self_consistent and outside) else (
            "identical" if all(d in base_set for d in digests) else "unstable")
        verdicts[arm] = dict(observed=observed, expect=ARMS[arm]["expect"], digests=digests)
    if verdicts["gross"]["observed"] != "differ":
        invalid.append("gross perturbation not detected: kernel off the executed path or dump blind")
    summary = dict(
        kind="verdict", valid=not invalid, invalid_reasons=invalid,
        baseline_runs=len(base), baseline_distinct=len(base_set),
        baseline_digests={d: base.count(d) for d in base_set},
        baseline_deterministic=len(base_set) == 1,
        detects_bf16_ulp=verdicts["bf16ulp"]["observed"] == "differ",
        arms=verdicts,
        coverage="score per-token forward only; batched prefill not covered")
    cmp.emit(out, summary)
    log(json.dumps({k: summary[k] for k in ("valid", "baseline_distinct", "detects_bf16_ulp")}))
    for arm, v in verdicts.items():
        log(f"{arm:8s} expect={v['expect']:13s} observed={v['observed']}")
    return 0 if not invalid else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="build baseline + perturbed libraries (no model needed)")
    p.add_argument("--root", required=True)
    p.add_argument("--repo", default=str(_DEFAULT_REPO),
                   help="repository root; required when this script is run from a copy")
    p.add_argument("--score", default=None,
                   help="score binary (default: <repo>/bin/score)")
    r = sub.add_parser("run", help="score every arm (exclusive window only)")
    r.add_argument("--root", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--text", required=True)
    r.add_argument("--baseline-repeats", type=int, default=10)
    r.add_argument("--arm-repeats", type=int, default=2)
    for key, value in cmp.DEFAULTS.items():
        r.add_argument("--" + key.replace("_", "-"), default=value, type=type(value))
    a = ap.parse_args()
    if a.cmd == "prepare" and a.score is None:
        a.score = str(Path(a.repo) / "bin" / "score")
    return prepare(a) if a.cmd == "prepare" else run(a)


if __name__ == "__main__":
    sys.exit(main())
