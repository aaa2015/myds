#!/usr/bin/env python3
"""CPU-only attention ABI, lossless KV dictionary, mask and identity controls.

The tiny synthetic inputs only validate host rejection and storage contracts;
they are never GPU, model correctness, or performance evidence.
"""
import copy
import ctypes as ct
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


class FlashArgs(ct.LittleEndianStructure):
    _fields_ = [(name, kind) for names, kind in (
        (("ne01", "ne02", "ne03"), ct.c_int32), (("nb01", "nb02", "nb03"), ct.c_uint64),
        (("ne11", "ne_12_2", "ne_12_3", "ns10"), ct.c_int32), (("nb11", "nb12", "nb13"), ct.c_uint64),
        (("ns20",), ct.c_int32), (("nb21", "nb22", "nb23"), ct.c_uint64),
        (("ne31", "ne32", "ne33"), ct.c_int32), (("nb31", "nb32", "nb33"), ct.c_uint64),
        (("ne1", "ne2", "ne3"), ct.c_int32), (("scale", "max_bias", "m0", "m1"), ct.c_float),
        (("n_head_log2",), ct.c_int32), (("logit_softcap",), ct.c_float), (("nb_kvq",), ct.c_uint64),
    ) for name in names]


def string(value):
    data = value.encode()
    return struct.pack("<Q", len(data)) + data


def main():
    binary = Path(sys.argv[1] if len(sys.argv) > 1 else "bin/bench_attention_multihead").resolve()
    result = subprocess.run([str(binary), "--cpu-self-test"], text=True, capture_output=True, check=True)
    assert json.loads(result.stdout)["invalid_args_rejected"] == 6
    print(result.stdout.strip())
    assert ct.sizeof(FlashArgs) == 200 and FlashArgs.nb_kvq.offset == 192
    args = FlashArgs(ne01=1, ne02=64, ne03=1, nb01=131072, nb02=2048, nb03=131072,
                     ne11=640, ne_12_2=1, ne_12_3=1, ns10=512, nb11=1024, nb12=655360, nb13=655360,
                     ns20=512, nb21=1024, nb22=655360, nb23=655360, ne31=1, ne32=1, ne33=1,
                     nb31=2048, nb32=2048, nb33=2048, ne1=64, ne2=1, ne3=1,
                     scale=1 / math.sqrt(512), nb_kvq=655360)
    with tempfile.TemporaryDirectory(prefix="myds-attention-multihead-host-") as temp:
        root = Path(temp)
        header = struct.pack("<IIQQ", 0x46554747, 3, 1, 1)
        header += string("general.architecture") + struct.pack("<I", 8) + string("deepseek41")
        header += string("blk.2.attn_sinks.weight") + struct.pack("<IQIQ", 1, 64, 0, 0)
        header += b"\0" * (-len(header) % 32)
        model = root / "synthetic-host-only.gguf"
        model.write_bytes(header + b"\0" * 256)

        def payload(name, data):
            (root / name).write_bytes(data)
            return {"path": name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}

        unique = struct.pack("<512H", *([0x3800] * 512))
        case = {"layer": 2, "base_pos": 0, "c_start": 2048, "rows": 1, "keys": 640,
                "args200": payload("args200.bin", bytes(args)),
                "q_f32": payload("q.f32", struct.pack("<32768f", *([0.5] * 32768))),
                "kv_unique_f16": payload("unique.f16", unique),
                "kv_row_ids_u32": payload("rows.u32", b"\0" * (640 * 4)),
                "kv_expanded_sha256": hashlib.sha256(unique * 640).hexdigest(),
                "mask_f16": payload("mask.f16", b"\0" * 1280),
                "sinks_f32": payload("sinks.f32", b"\0" * 256)}
        capture = {"schema_version": 1, "capture_kind": "myds_prefill_attention_multihead",
                   "model_bytes": model.stat().st_size, "model_header_bytes": len(header),
                   "model_header_sha256": hashlib.sha256(header).hexdigest(),
                   "source": {"kind": "synthetic_host_rejection_control_not_real_capture"},
                   "cases": [case, copy.deepcopy(case)]}
        (root / "probe.metallib").write_bytes(b"CPU-only-not-a-library")
        (root / "production.metallib").write_bytes(b"CPU-only-not-production")

        def run(label, value, expected):
            capture_path = root / "capture.json"
            capture_path.write_text(json.dumps(value))
            output = root / label
            result = subprocess.run([
                str(binary), "--model", str(model), "--capture-json", str(capture_path),
                "--probe-metallib", str(root / "probe.metallib"), "--production-metallib", str(root / "production.metallib"),
                "--out-dir", str(output), "--host-only",
            ], text=True, capture_output=True)
            assert (result.returncode == 0) == expected, (label, result.returncode, result.stdout, result.stderr)
            record = json.loads((output / "artifact-manifest.json").read_text())
            assert record["status"] == ("host_only" if expected else "failed") and "device" not in record
            if expected:
                complete = json.loads(result.stdout.splitlines()[-1])
                assert complete["host_only"] and not complete["metal_device_created"]
                first, second = output / "capture-case-000-q.f32", output / "capture-case-001-q.f32"
                assert first.stat().st_ino == second.stat().st_ino
                assert any(f.get("storage_alias_of") == first.name for f in record["files"] if f["path"] == second.name)
            print(json.dumps({"case": label, "passed": True, "expected_success": expected, "gpu_executed": False}))

        run("valid-lossless-and-hardlink", capture, True)
        masked = copy.deepcopy(capture)
        masked["cases"][0]["mask_f16"] = payload("valid-masked.f16", struct.pack("<H", 0xFC00) + b"\0" * 1278)
        run("valid-negative-infinity-mask", masked, True)
        bad = copy.deepcopy(capture); bad["capture_kind"] = "myds_decode_attention_multihead"
        run("wrong-scope-rejected", bad, False)
        bad = copy.deepcopy(capture); bad["cases"][0]["q_f32"]["sha256"] = "0" * 64
        run("payload-sha-rejected", bad, False)
        bad = copy.deepcopy(capture); bad["cases"][0]["q_f32"]["bytes"] += 4
        run("payload-declared-length-rejected", bad, False)
        bad = copy.deepcopy(capture); bad["cases"][0]["kv_expanded_sha256"] = "0" * 64
        run("expanded-kv-sha-rejected", bad, False)
        bad = copy.deepcopy(capture)
        bad["cases"][0]["kv_row_ids_u32"] = payload("bad-rows.u32", struct.pack("<I", 1) + b"\0" * (639 * 4))
        run("dictionary-range-rejected", bad, False)
        bad_args = FlashArgs.from_buffer_copy(bytes(args)); bad_args.nb_kvq = 0
        bad = copy.deepcopy(capture); bad["cases"][0]["args200"] = payload("bad-args200.bin", bytes(bad_args))
        run("args-query-stride-rejected", bad, False)
        bad = copy.deepcopy(capture); bad["cases"][0]["mask_f16"] = payload("bad-mask.f16", struct.pack("<H", 0x7C00) + b"\0" * 1278)
        run("invalid-mask-rejected", bad, False)
        bad = copy.deepcopy(capture); bad["cases"][0]["sinks_f32"] = payload("bad-sinks.f32", struct.pack("<f", 0.5) + b"\0" * 252)
        run("model-sinks-mismatch-rejected", bad, False)
        bad = copy.deepcopy(capture); bad["model_bytes"] += 32
        run("model-file-size-rejected", bad, False)
        bad = copy.deepcopy(capture); bad["model_header_sha256"] = "0" * 64
        run("model-prefix-sha-rejected", bad, False)


if __name__ == "__main__":
    main()
