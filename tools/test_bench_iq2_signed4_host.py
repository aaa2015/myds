#!/usr/bin/env python3
"""CPU-only capture and GGUF identity rejection controls; creates no Metal device.

The sparse GGUF is a synthetic host fixture, never GPU or model-quality evidence.
Usage: python3 tools/test_bench_iq2_signed4_host.py [path/to/bench_iq2_signed4]
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
    binary = Path(sys.argv[1] if len(sys.argv) > 1 else "bin/bench_iq2_signed4").resolve()
    with tempfile.TemporaryDirectory(prefix="myds-iq2-signed4-host-") as temp:
        root = Path(temp)
        metadata = string("general.architecture") + struct.pack("<I", 8) + string("deepseek4")
        metadata += string("deepseek4.swiglu_limit") + struct.pack("<If", 6, 10.0)
        header = struct.pack("<IIQQ", 0x46554747, 3, 2, 2) + metadata
        tensor_bytes = 1167851520
        for name, offset in (("ffn_gate_exps.weight", 0), ("ffn_up_exps.weight", tensor_bytes)):
            header += string("blk.0." + name) + struct.pack("<IQQQIQ", 3, 5120, 2304, 384, 16, offset)
        header += b"\0" * (-len(header) % 32)
        model = root / "sparse-cpu-fixture.gguf"
        with model.open("wb") as stream:
            stream.write(header)
            stream.truncate(len(header) + 2 * tensor_bytes)

        def payload(name, data):
            (root / name).write_bytes(data)
            return {"path": name, "sha256": hashlib.sha256(data).hexdigest()}

        capture = {
            "schema_version": 1,
            "capture_kind": "myds_decode_routed_moe",
            "model_bytes": model.stat().st_size,
            "model_header_bytes": len(header),
            "model_header_sha256": hashlib.sha256(header).hexdigest(),
            "source": {"kind": "synthetic_cpu_only_negative_control_not_inference"},
            "cases": [{
                "layer": 0, "position": 0, "clamp": 10.0,
                "input_f32": payload("input.f32", struct.pack("<5120f", *([1.0] * 5120))),
                "ids_i32": payload("ids.i32", struct.pack("<6i", *range(6))),
                "route_weights_f32": payload("weights.f32", struct.pack("<6f", *([1 / 6] * 6))),
            }],
        }
        # Host-only mode fingerprints these files, but cannot create/load Metal libraries.
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
            if expected:
                complete = json.loads(result.stdout.splitlines()[-1])
                assert complete["host_only"] and not complete["metal_device_created"]
            print(json.dumps({"case": label, "passed": True, "expected_success": expected,
                              "exit_code": result.returncode, "gpu_executed": False}))

        run("valid-host-only", capture, True)
        bad = copy.deepcopy(capture); bad["cases"][0]["clamp"] = 0.0
        run("clamp-rejected", bad, False)
        bad = copy.deepcopy(capture); bad["cases"][0]["input_f32"]["sha256"] = "0" * 64
        run("payload-sha-rejected", bad, False)
        bad = copy.deepcopy(capture); bad["model_header_sha256"] = "0" * 64
        run("model-header-sha-rejected", bad, False)
        bad = copy.deepcopy(capture); bad["model_bytes"] += 32
        run("model-size-rejected", bad, False)
        bad = copy.deepcopy(capture)
        bad["cases"][0]["ids_i32"] = payload("badids.i32", struct.pack("<6i", 0, 1, 2, 3, 4, 384))
        run("expert-range-rejected", bad, False)
        bad = copy.deepcopy(capture)
        bad["cases"][0]["ids_i32"] = payload("dupids.i32", struct.pack("<6i", 0, 1, 2, 3, 4, 4))
        run("expert-duplicate-rejected", bad, False)
        bad = copy.deepcopy(capture)
        bad["cases"][0]["route_weights_f32"] = payload("nanweights.f32", struct.pack("<6f", float("nan"), 0, 0, 0, 0, 0))
        run("nonfinite-rejected", bad, False)
        bad = copy.deepcopy(capture); bad["cases"][0]["input_f32"] = payload("short.f32", b"\0" * 20)
        run("payload-size-rejected", bad, False)


if __name__ == "__main__":
    main()
