#!/usr/bin/env python3
"""Build and run a same-process, CPU-only tokenizer A/B benchmark.

Example: python3 tools/bench_tokenizer.py --model /path/model.gguf \
  --prompt prompt.txt --prompt code.txt --output /tmp/tokenizer-ab.jsonl
Only GGUF metadata is copied/read. Model tensor payloads and GPU are never loaded.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import struct
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ("load", "free", "encode", "encode_bytes", "decode_bytes", "decode_piece")
SCALAR_BYTES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def metadata_copy(source, dest):
    """Copy the GGUF header and metadata into a tensor-free, aligned GGUF."""
    size = source.stat().st_size
    with source.open("rb") as f:
        def read(n):
            data = f.read(n)
            if len(data) != n:
                raise ValueError("truncated GGUF metadata")
            return data

        def u32():
            return struct.unpack("<I", read(4))[0]

        def u64():
            return struct.unpack("<Q", read(8))[0]

        def string(keep=False):
            length = u64()
            if length > size - f.tell():
                raise ValueError("GGUF string exceeds file")
            if keep:
                return read(length)
            f.seek(length, 1)

        def skip(kind):
            if kind in SCALAR_BYTES:
                read(SCALAR_BYTES[kind])
            elif kind == 8:
                string()
            elif kind == 9:
                elem, count = u32(), u64()
                if elem in SCALAR_BYTES:
                    amount = count * SCALAR_BYTES[elem]
                    if amount > size - f.tell():
                        raise ValueError("GGUF array exceeds file")
                    f.seek(amount, 1)
                elif elem == 8:
                    if count > (size - f.tell()) // 8:
                        raise ValueError("GGUF string array exceeds file")
                    for _ in range(count):
                        string()
                else:
                    raise ValueError(f"unsupported GGUF array type {elem}")
            else:
                raise ValueError(f"unsupported GGUF metadata type {kind}")

        if read(4) != b"GGUF":
            raise ValueError("not a GGUF file; pass the first physical GGUF shard")
        version, tensors, count = u32(), u64(), u64()
        if version not in (2, 3) or count > (size - 24) // 12:
            raise ValueError("invalid GGUF header")
        alignment = 32
        for _ in range(count):
            key, kind = string(True), u32()
            if key == b"general.alignment" and kind == 4:
                alignment = u32() or 32
            else:
                skip(kind)
        end = f.tell()
        if alignment > 1048576 or end > 512 * 1024 * 1024:
            raise ValueError("unreasonable metadata size/alignment")
        f.seek(24)
        metadata = read(end - 24)
    raw = b"GGUF" + struct.pack("<IQQ", version, 0, count) + metadata
    raw += b"\0" * (-len(raw) % alignment)
    dest.write_bytes(raw)
    return {"source_path": str(source), "source_bytes": size,
            "source_mtime_ns": source.stat().st_mtime_ns, "source_tensor_count": tensors,
            "metadata_bytes": end, "metadata_sha256": sha(metadata),
            "tensor_free_gguf_sha256": sha(raw), "payload_read": False}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, help="real GGUF or its first physical shard (metadata only)")
    ap.add_argument("--prompt", type=Path, action="append", default=[], help="binary-safe prompt file; repeatable")
    ap.add_argument("--output", type=Path, required=True, help="new raw JSONL output (never overwritten)")
    ap.add_argument("--baseline-ref", default="HEAD", help="baseline git revision (default HEAD)")
    ap.add_argument("--baseline-source", type=Path, help="explicit baseline tokenizer C source, instead of git export")
    ap.add_argument("--candidate-source", type=Path, default=ROOT / "src/myds_tokenizer.c", help="candidate tokenizer C source (default src/myds_tokenizer.c)")
    ap.add_argument("--samples", type=int, default=9, help="paired samples per prompt, minimum 9")
    ap.add_argument("--min-batch-ms", type=float, default=100, help="target warm-calibrated batch duration")
    ap.add_argument("--random-cases", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20261028)
    ap.add_argument("--cc", default="clang")
    ap.add_argument("--deploy-target", default="26.0", help="macOS deployment target on Darwin (default 26.0, as in Makefile)")
    ap.add_argument("--build-only", action="store_true", help="compile only; model and prompts not required")
    args = ap.parse_args()
    if args.samples < 9 or not 0 < args.min_batch_ms <= 10000 or not 0 <= args.random_cases <= 100000:
        ap.error("samples must be >=9, batch ms in (0,10000], random cases in [0,100000]")
    if not 0 <= args.seed <= 0xffffffff:
        ap.error("seed must be uint32")
    if not args.build_only and (not args.model or not args.prompt):
        ap.error("--model and at least one --prompt are required")
    if args.output.exists():
        ap.error("--output already exists; use a fresh filename")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    baseline_sha = None
    if args.baseline_source:
        baseline = args.baseline_source.read_bytes()
    else:
        baseline_sha = subprocess.check_output(["git", "rev-parse", "--verify", args.baseline_ref + "^{commit}"], cwd=ROOT, text=True).strip()
        baseline = subprocess.check_output(["git", "show", baseline_sha + ":src/myds_tokenizer.c"], cwd=ROOT)
    with tempfile.TemporaryDirectory(prefix="myds-tokenizer-ab-") as td, args.output.open("x") as log:
        work = Path(td)

        def emit(row):
            log.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            log.flush()

        # Freeze both translation units and their shared headers before compile.
        (work / "baseline.c").write_bytes(baseline)
        for name in ("myds_tokenizer.h", "myds_gguf.h"):
            shutil.copyfile(ROOT / "include" / name, work / name)
        shutil.copyfile(ROOT / "src/myds_tokenizer_unicode.h", work / "myds_tokenizer_unicode.h")
        shutil.copyfile(args.candidate_source, work / "candidate.c")
        for rel, name in (("src/myds_gguf.c", "gguf.c"), ("tools/bench_tokenizer.c", "worker.c")):
            shutil.copyfile(ROOT / rel, work / name)
        sources = {p.name: sha(p.read_bytes()) for p in work.iterdir()}
        deployment = [f"-mmacosx-version-min={args.deploy_target}"] if platform.system() == "Darwin" else []
        common = [args.cc, "-O3", "-std=c11", "-Wall", "-Wextra", *deployment, "-I", str(work)]
        commands = []
        for source in ("baseline", "candidate", "gguf", "worker"):
            command = common + ([f"-Dmyds_tokenizer_{name}=baseline_tokenizer_{name}" for name in PUBLIC] if source == "baseline" else [])
            commands.append(command + ["-c", str(work / (source + ".c")), "-o", str(work / (source + ".o"))])
        binary = work / "bench_tokenizer"
        commands.append([args.cc, *deployment] + [str(work / (name + ".o")) for name in ("baseline", "candidate", "gguf", "worker")] + ["-o", str(binary)])
        emit({"kind": "provenance", "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "host": platform.node(), "platform": platform.platform(), "machine": platform.machine(),
              "baseline_revision": baseline_sha, "baseline_ref": None if args.baseline_source else args.baseline_ref,
              "baseline_source": str(args.baseline_source.resolve()) if args.baseline_source else None,
              "candidate_source": str(args.candidate_source.resolve()),
              "sources_sha256": sources, "compile_commands": commands,
              "compiler": subprocess.check_output([args.cc, "--version"], text=True).splitlines()[0],
              "samples_per_arm": args.samples, "min_batch_ms": args.min_batch_ms,
              "timing_scope": "encode_bytes only; output buffer preallocated; add_bos=false; CPU only; no end-to-end inference claim"})
        for command in commands:
            subprocess.run(command, check=True)
        emit({"kind": "build", "binary_sha256": sha(binary.read_bytes()), "success": True})
        if args.build_only:
            emit({"kind": "summary", "build_only": True, "measured": False})
            return
        emit({"kind": "model_metadata", **metadata_copy(args.model.resolve(), work / "metadata.gguf")})
        prompt_paths = []
        for i, path in enumerate(args.prompt):
            raw = path.read_bytes()
            target = work / f"prompt-{i}.bin"
            target.write_bytes(raw)
            prompt_paths.append(str(target))
            emit({"kind": "input", "prompt_index": i, "path": str(path.resolve()), "bytes": len(raw), "sha256": sha(raw)})
        command = [str(binary), str(work / "metadata.gguf"), str(args.samples), str(args.min_batch_ms), str(args.random_cases), str(args.seed), *prompt_paths]
        emit({"kind": "run", "command": command})
        rows = []
        with subprocess.Popen(command, stdout=subprocess.PIPE, text=True) as proc:
            for line in proc.stdout:
                row = json.loads(line)
                emit(row)
                rows.append(row)
            code = proc.wait()
        if code:
            emit({"kind": "summary", "pass": False, "exit_code": code})
            raise RuntimeError(f"worker failed with exit {code}; inspect stderr and {args.output}")
        summaries = []
        for i in range(len(args.prompt)):
            groups = {arm: [r["ns_per_encode"] for r in rows if r["kind"] == "sample" and r["prompt_index"] == i and r["arm"] == arm] for arm in ("baseline", "candidate")}
            if any(len(values) != args.samples for values in groups.values()):
                raise RuntimeError("incomplete sample set")
            stats = {arm: {"n": len(v), "median_ms": statistics.median(v)/1e6, "min_ms": min(v)/1e6,
                           "max_ms": max(v)/1e6, "stdev_ms": statistics.stdev(v)/1e6} for arm, v in groups.items()}
            a, b = stats["baseline"]["median_ms"], stats["candidate"]["median_ms"]
            summaries.append({"prompt_index": i, **stats, "speedup_ratio": a/b,
                              "latency_reduction_percent": 100*(a-b)/a})
        emit({"kind": "summary", "pass": True, "prompts": summaries,
              "scope": "CPU tokenizer only; no model inference or GPU performance measured"})
        print(json.dumps({"output": str(args.output.resolve()), "prompts": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"bench_tokenizer: {exc}", file=sys.stderr)
        sys.exit(1)
