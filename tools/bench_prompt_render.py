#!/usr/bin/env python3
"""Reproducible CPU-only A/B of the actual parse_chat_prompt_ex path.

The synthetic corpus includes tool results, tool schemas and ordinary user
messages. The process never opens model weights or initializes Metal. The
baseline defaults to a pinned git commit; --baseline-source supports an
isolated machine whose checkout has different history. Output is new JSONL.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "f82959151f1f29026eb991e096dc4741339f72bd"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def request(kind, text):
    if kind == "tool":
        return {"input": [{"type": "function_call_output", "output": text}]}
    if kind == "user":
        return {"messages": [{"role": "user", "content": text}]}
    return {"input": "hello", "tools": [{"type": "function", "function": {
        "name": "inspect", "description": text,
        "parameters": {"type": "object", "properties": {"x": {"type": "string"}}}}}]}


def corpus():
    unit = "Alpha xyz 123 中文🙂 \"quoted\" \\path\n<node> &value "
    cases = []
    for length in (8192, 65536, 524288):
        text = (unit * (length // len(unit) + 1))[:length]
        text += "</tool_result>&lt;/tool_result>&amp;amp;lt;/tool_result>"
        for kind in ("tool", "schema", "user"):
            cases.append((f"{kind}_{length}", request(kind, text)))
    return cases


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True, help="new JSONL path")
    p.add_argument("--baseline-ref", default=BASELINE)
    p.add_argument("--baseline-source", type=Path)
    p.add_argument("--candidate-source", type=Path, default=ROOT / "src/myds_server.c")
    p.add_argument("--extra-text", type=Path, action="append", default=[], help="additional tool-result text (repeatable)")
    p.add_argument("--samples", type=int, default=9)
    p.add_argument("--min-batch-ms", type=float, default=50)
    p.add_argument("--cc", default="clang")
    a = p.parse_args()
    if a.output.exists() or a.samples < 9 or not 0 < a.min_batch_ms <= 1000:
        p.error("output must be new, samples >= 9 and batch target in (0,1000] ms")
    if a.baseline_source:
        base = a.baseline_source.read_bytes()
        rev = None
    else:
        rev = subprocess.check_output(["git", "rev-parse", "--verify", a.baseline_ref + "^{commit}"], cwd=ROOT, text=True).strip()
        base = subprocess.check_output(["git", "show", rev + ":src/myds_server.c"], cwd=ROOT)
    cases = corpus()
    extra_sources = []
    for item in a.extra_text:
        raw = item.read_bytes()
        extra_sources.append({"path": str(item.resolve()), "bytes": len(raw), "sha256": sha(raw)})
        cases.append(("extra_" + item.name, request("tool", raw.decode("utf-8"))))
    fuzz_seed = 20261028
    rng = random.Random(fuzz_seed)
    pieces = ["a", "<", "</tool_result>", "&", "&lt;/tool_result>",
              "&amp;lt;/tool_result>", "&amp;amp;lt;/tool_result>",
              "é", "中", "🙂", "\n", "\r", "\t", '"', "\\", "\b", "\x01"]
    fuzz = [(f"fuzz_{i}", request(("tool", "schema", "user")[i % 3],
             "".join(rng.choice(pieces) for _ in range(rng.randrange(0, 50))))) for i in range(48)]
    # The renderer parses bytes supplied by HTTP. Invalid encodings and JSON
    # must continue to fail with the same reason, output bytes and guards.
    invalid = [
        ("invalid_overlong_utf8", b'{"input":"bad\xc0\xaf"}'),
        ("invalid_truncated_utf8", b'{"input":"bad\xe4\xb8"}'),
        ("invalid_surrogate_utf8", b'{"input":"bad\xed\xa0\x80"}'),
        ("invalid_out_of_range_utf8", b'{"input":"bad\xf4\x90\x80\x80"}'),
        ("invalid_nul_byte", b'{"input":"bad\x00tail"}'),
        ("invalid_nul_escape", b'{"input":"bad\\u0000tail"}'),
        ("invalid_high_surrogate", b'{"input":"bad\\uD800"}'),
        ("invalid_low_surrogate", b'{"input":"bad\\uDC00"}'),
        ("invalid_escape", b'{"input":"bad\\q"}'),
        ("invalid_unterminated", b'{"input":"unfinished'),
        ("invalid_trailing_comma", b'{"input":"bad",}'),
    ]
    invalid_names = {name for name, _ in invalid}
    names = [name for name, _ in cases + fuzz + invalid]
    if len(set(names)) != len(names):
        p.error("corpus names collide; extra text basenames must be distinct")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="myds-prompt-ab-") as temp, a.output.open("x") as log:
        d = Path(temp)
        shutil.copytree(ROOT / "include", d / "include")
        shutil.copyfile(ROOT / "src/myds_log.c", d / "myds_log.c")
        shutil.copyfile(ROOT / "tools/bench_prompt_render.c", d / "worker.c")
        (d / "baseline.c").write_bytes(base)
        (d / "candidate.c").write_bytes(a.candidate_source.read_bytes())
        sources = {p.name: sha(p.read_bytes()) for p in d.glob("*.c")}
        sources["headers"] = {p.name: sha(p.read_bytes()) for p in (d / "include").glob("*.h")}
        flags = [a.cc, "-O3", "-Wall", "-Wextra", "-std=c11", "-I", str(d / "include")]
        if sys.platform == "darwin":
            flags.append("-mmacosx-version-min=26.0")
        commands = [flags + [str(d / "worker.c"), str(d / (arm + ".c")), str(d / "myds_log.c"),
                    "-Wl,-dead_strip", "-lpthread", "-lm", "-o", str(d / arm)]
                    for arm in ("baseline", "candidate")]

        def emit(row):
            log.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            log.flush()

        emit({"kind": "provenance", "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "host": platform.node(), "platform": platform.platform(), "machine": platform.machine(),
              "baseline_revision": rev, "baseline_source": str(a.baseline_source.resolve()) if a.baseline_source else None,
              "candidate_source": str(a.candidate_source.resolve()), "sources_sha256": sources,
              "script_sha256": sha(Path(__file__).resolve().read_bytes()),
              "fuzz_seed": fuzz_seed, "extra_sources": extra_sources,
              "compiler": subprocess.check_output([a.cc, "--version"], text=True).splitlines()[0],
              "compile_commands": commands, "samples_per_arm": a.samples,
              "scope": "CPU parse_chat_prompt_ex only; no HTTP, tokenizer, model or GPU"})
        for cmd in commands:
            subprocess.run(cmd, check=True)
        emit({"kind": "build", "baseline_binary_sha256": sha((d / "baseline").read_bytes()),
              "candidate_binary_sha256": sha((d / "candidate").read_bytes())})

        def run(arm, path, cap, mode, reps=1):
            return subprocess.run([str(d / arm), str(path), str(cap), mode, str(reps)],
                                  check=True, capture_output=True).stdout

        outputs = {}
        for name, body in cases + fuzz + invalid:
            data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()
            path = d / (name + ".json")
            path.write_bytes(data)
            cap = max(65536, len(data) * 4 + 8192)
            full = run("baseline", path, cap, "check")
            header, _, raw = full.partition(b"\n")
            status, why_len, guards = map(int, header.split())
            expected_status = 0 if name in invalid_names else 1
            if status != expected_status or guards != 1 or len(raw) != why_len + cap:
                raise RuntimeError(f"baseline returned an unexpected status or buffer for {name}")
            output = raw[why_len:].split(b"\0", 1)[0]
            if status and not output:
                raise RuntimeError(f"empty prompt for {name}")
            # Include complete raw output capacity (initialized canaries),
            # failure reason, and boundary guards, not only a text digest.
            caps = sorted({0, 1, 2, 7, 31, 64, max(0, len(output)-1),
                           len(output), len(output)+1, cap})
            for limit in caps:
                old = full if limit == cap else run("baseline", path, limit, "check")
                new = run("candidate", path, limit, "check")
                if old != new or int(old.split(b"\n", 1)[0].split()[2]) != 1:
                    emit({"kind": "correctness", "case": name, "capacity": limit, "pass": False})
                    raise RuntimeError(f"prompt/output/reason/canary mismatch: {name} capacity {limit}")
            outputs[name] = (path, cap, len(output))
            emit({"kind": "correctness", "case": name, "input_bytes": len(data), "input_sha256": sha(data),
                  "prompt_bytes": len(output), "prompt_sha256": sha(output), "capacities_checked": caps,
                  "expected_failure": name in invalid_names,
                  "full_capacity_and_failure_bytes_equal": True, "pass": True})

        summaries = []
        for name, body in cases:
            path, cap, output_len = outputs[name]
            warm = [int(run(arm, path, cap, "time").split()[0]) for arm in ("baseline", "candidate")]
            fastest = max(1, min(warm))
            reps = min(10000, max(1, int(a.min_batch_ms * 1e6 / fastest)))
            # Bound the slower arm to roughly one second per sample.
            reps = min(reps, max(1, int(1e9 / max(warm))))
            emit({"kind": "warmup", "case": name, "baseline_ns": warm[0], "candidate_ns": warm[1], "repetitions": reps})
            groups = {"baseline": [], "candidate": []}
            for sample in range(a.samples):
                arms = ("baseline", "candidate") if sample % 2 == 0 else ("candidate", "baseline")
                for order, arm in enumerate(arms):
                    elapsed, rendered_len = map(int, run(arm, path, cap, "time", reps).split())
                    if elapsed <= 0 or rendered_len != output_len:
                        raise RuntimeError("timed parse failed or rendered length changed")
                    groups[arm].append(elapsed / reps)
                    emit({"kind": "sample", "case": name, "sample": sample, "order": order, "arm": arm,
                          "repetitions": reps, "elapsed_ns": elapsed, "ns_per_parse": elapsed / reps})
            old, new = (statistics.median(groups[arm]) for arm in ("baseline", "candidate"))
            summaries.append({"case": name, "baseline_median_ns": old, "candidate_median_ns": new,
                              "latency_reduction_percent": 100 * (old - new) / old,
                              "baseline_range_ns": [min(groups['baseline']), max(groups['baseline'])],
                              "candidate_range_ns": [min(groups['candidate']), max(groups['candidate'])]})
        emit({"kind": "summary", "pass": True, "fuzz_cases": len(fuzz), "invalid_cases": len(invalid), "cases": summaries,
              "scope": "CPU prompt rendering only; no inference or HTTP speedup measured"})
        print(json.dumps({"output": str(a.output.resolve()), "cases": summaries}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as e:
        print(f"bench_prompt_render: {e}", file=sys.stderr)
        sys.exit(1)
