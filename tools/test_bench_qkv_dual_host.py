#!/usr/bin/env python3
"""CPU-only QKV capture/GGUF/alias rejection controls; no Metal device or GPU.

Usage: python3 tools/test_bench_qkv_dual_host.py [path/to/bench_qkv_dual]
The synthetic sparse GGUF is host validation only, never model evidence.
"""
import copy
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


def string(value):
    data = value.encode()
    return struct.pack("<Q", len(data)) + data


def main():
    binary = Path(sys.argv[1] if len(sys.argv) > 1 else "bin/bench_qkv_dual").resolve()
    result = subprocess.run([str(binary), "--cpu-self-test"], text=True, capture_output=True, check=True)
    assert json.loads(result.stdout)["rejected_invalid_bindings"] == 8
    print(result.stdout.strip())
    with tempfile.TemporaryDirectory(prefix="myds-qkv-dual-host-") as temp:
        root = Path(temp)
        header = struct.pack("<IIQQ", 0x46554747, 3, 2, 1)
        header += string("general.architecture") + struct.pack("<I", 8) + string("deepseek41")
        for name, rows, offset in (("attn_q_a.weight", 1280, 0), ("attn_kv.weight", 512, 3686400)):
            header += string("blk.0." + name) + struct.pack("<IQQIQ", 2, 5120, rows, 12, offset)
        header += b"\0" * (-len(header) % 32)
        model = root / "sparse-cpu-fixture.gguf"
        model_size = len(header) + 3686400 + 1474560
        with model.open("wb") as stream:
            stream.write(header)
            stream.truncate(model_size)

        def payload(name, data):
            (root / name).write_bytes(data)
            return {"path": name, "sha256": hashlib.sha256(data).hexdigest()}

        capture = {
            "schema_version": 1, "capture_kind": "myds_decode_qkv",
            "model_bytes": model.stat().st_size, "model_header_bytes": len(header),
            "model_header_sha256": hashlib.sha256(header).hexdigest(),
            "source": {"kind": "synthetic_cpu_only_negative_control_not_inference"},
            "cases": [{"layer": 0, "position": 0,
                       "input_f32": payload("input.f32", struct.pack("<5120f", *([1.0] * 5120)))}],
        }
        (root / "probe.metallib").write_bytes(b"CPU-only-not-a-library")
        (root / "production.metallib").write_bytes(b"CPU-only-not-production")

        def run(label, value, expected):
            capture_path = root / "capture.json"
            capture_path.write_text(json.dumps(value))
            output = root / label
            result = subprocess.run([
                str(binary), "--model", str(model), "--capture-json", str(capture_path),
                "--probe-metallib", str(root / "probe.metallib"),
                "--production-metallib", str(root / "production.metallib"),
                "--out-dir", str(output), "--host-only",
            ], text=True, capture_output=True)
            assert (result.returncode == 0) == expected, (label, result.returncode, result.stdout, result.stderr)
            record = json.loads((output / "artifact-manifest.json").read_text())
            assert record["status"] == ("host_only" if expected else "failed")
            assert "device" not in record
            assert record["host_invalid_bindings_rejected"] == 8
            if expected:
                complete = json.loads(result.stdout.splitlines()[-1])
                assert complete["host_only"] and not complete["metal_device_created"]
            print(json.dumps({"case": label, "passed": True, "expected_success": expected,
                              "exit_code": result.returncode, "gpu_executed": False}))

        run("valid-host-only", capture, True)
        bad = copy.deepcopy(capture); bad["capture_kind"] = "myds_decode_routed_moe"
        run("capture-kind-rejected", bad, False)
        bad = copy.deepcopy(capture); bad["cases"][0]["input_f32"]["sha256"] = "0" * 64
        run("payload-sha-rejected", bad, False)
        bad = copy.deepcopy(capture); bad["model_header_sha256"] = "0" * 64
        run("model-header-sha-rejected", bad, False)
        bad = copy.deepcopy(capture); bad["model_bytes"] += 32
        run("model-size-rejected", bad, False)
        bad = copy.deepcopy(capture); bad["cases"][0]["layer"] = 40
        run("layer-range-rejected", bad, False)
        bad = copy.deepcopy(capture)
        bad["cases"][0]["input_f32"] = payload("nan.f32", struct.pack("<5120f", float("nan"), *([0.0] * 5119)))
        run("nonfinite-rejected", bad, False)
        bad = copy.deepcopy(capture); bad["cases"][0]["input_f32"] = payload("short.f32", b"\0" * 20)
        run("payload-size-rejected", bad, False)
        # A self-consistent header and captured file length cannot authorize a
        # tensor descriptor whose declared payload extends beyond real EOF.
        with model.open("r+b") as stream:
            stream.truncate(model_size - 32)
        bad = copy.deepcopy(capture); bad["model_bytes"] -= 32
        run("tensor-eof-rejected", bad, False)


if __name__ == "__main__":
    main()
