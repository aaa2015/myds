#!/usr/bin/env python3
"""CPU-only fixture checks for the actual-prepare measurement and log audit.

Compiles and runs the production Engram code against a small valid GGUF.
Storage counter records in summary tests are deliberately mocked. These
checks prove correctness/log rejection, never physical coldness or GPU gain.
"""
import hashlib
import importlib.util
import contextlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("engram_summary", ROOT / "tools/summarize_engram_io.py")
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)
RUNNER_SPEC = importlib.util.spec_from_file_location("engram_runner", ROOT / "tools/run_engram_io.py")
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(RUNNER)
FNV_INIT = 14695981039346656037
TOKENS = [1, 2, 3, 4, 3, 2, 1]
TOKEN_MAP = [0, 3, 5, 7, 2]
MULTIPLIERS = [[1, 3, 5, 7], [11, 13, 17, 19]]


def fnv(data, state=FNV_INIT):
    for b in data:
        state = ((state ^ b) * 1099511628211) & ((1 << 64) - 1)
    return state


def string(value):
    data = value.encode()
    return struct.pack("<Q", len(data)) + data


def kv(name, kind, data):
    return string(name) + struct.pack("<I", kind) + data


def array(kind, values, fmt):
    return struct.pack("<IQ", kind, len(values)) + struct.pack("<" + fmt * len(values), *values)


def raw_row(rid):
    return bytes([0x20 + rid % 20]) * 256 + bytes([127 + rid % 2]) * 8


def fixture(path):
    metadata = [
        kv("general.alignment", 4, struct.pack("<I", 16384)),
        kv("deepseek41.engram.encoding", 8, string("e4m3_e8m0_32_row264")),
        kv("deepseek41.engram.layer_ids", 9, array(4, [1, 14], "I")),
        kv("deepseek41.engram.rows", 9, array(4, [264, 264], "I")),
        kv("deepseek41.engram.primes", 9, array(4, [11] * 48, "I")),
        kv("deepseek41.engram.multipliers", 9, array(10, sum(MULTIPLIERS, []), "Q")),
        kv("deepseek41.engram.token_map", 9, array(4, TOKEN_MAP, "I")),
        kv("deepseek41.engram.compressed_vocab_size", 4, struct.pack("<I", 8)),
        kv("deepseek41.engram.pad_id", 4, struct.pack("<I", 1)),
    ]
    tensors = []
    payload = bytearray()
    for layer in (1, 14):
        for name, dims, kind, nbytes in [
            ("engram_kv.weight", [6144, 5], 1, 61440),
            ("engram_q_norm.weight", [1, 4], 0, 16),
            ("engram_k_norm.weight", [1, 4], 0, 16),
        ]:
            offset = len(payload)
            tensors.append((f"blk.{layer}.{name}", dims, kind, offset))
            payload.extend(bytes(nbytes))
    payload.extend(bytes((-len(payload)) % 16384))
    for layer in (1, 14):
        tensors.append((f"blk.{layer}.engram_embd.weight", [264, 264], 24, len(payload)))
        for rid in range(264):
            payload.extend(raw_row(rid))
    header = struct.pack("<IIQQ", 0x46554747, 3, len(tensors), len(metadata)) + b"".join(metadata)
    for name, dims, kind, offset in tensors:
        header += string(name) + struct.pack("<I", len(dims)) + struct.pack("<" + "Q" * len(dims), *dims)
        header += struct.pack("<IQ", kind, offset)
    header += bytes((-len(header)) % 16384)
    path.write_bytes(header + payload)


def expected():
    history = [-1, -1, -1]
    outputs = []
    for token in TOKENS:
        current = TOKEN_MAP[token]
        ids = [current] + history
        blocked = False
        for j, value in enumerate(ids):
            blocked = blocked or value == -1
            ids[j] = 1 if blocked else value
        rows = []
        for multipliers in MULTIPLIERS:
            state = ids[0] * multipliers[0]
            offset = 0
            for order in range(1, 4):
                state ^= ids[order] * multipliers[order]
                for _ in range(8):
                    rows.append(state % 11 + offset)
                    offset += 11
        # Independent scalar E4M3 -> scale -> BF16 conversion.
        data = bytearray()
        for rid in rows:
            raw = raw_row(rid)
            code, scale = raw[0], raw[256]
            exponent, fraction = (code >> 3) & 15, code & 7
            value = math.ldexp(float(8 + fraction), exponent - 10) if exponent else math.ldexp(float(fraction), -9)
            if code & 128:
                value = -value
            value = math.ldexp(value, scale - 127)
            bits = struct.unpack("<I", struct.pack("<f", value))[0]
            bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
            data.extend(struct.pack("<I", bits) * 256)
        history = [current] + history[:2]
        outputs.append((rows, list(history), format(fnv(data), "016x")))
    return outputs


def driver(byte_count):
    return ('+-o IOBlockStorageDriver <class IOBlockStorageDriver, id 0x123, registered>\n'
            '  "Statistics" = {"Bytes (Read)"=%d,"Operations (Read)"=%d,"Total Time (Read)"=%d}\n'
            % (byte_count, byte_count // 4096, byte_count * 2))


class EngramIOToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="myds-engram-io-tools-")
        cls.work = Path(cls.temp.name)
        cls.model = cls.work / "fixture.gguf"
        fixture(cls.model)
        cls.model.chmod(0o444)
        cls.source_sha = hashlib.sha256(cls.model.read_bytes()).hexdigest()
        cls.tokens = cls.work / "tokens.bin"
        cls.tokens.write_bytes(struct.pack("<" + "i" * len(TOKENS), *TOKENS))
        cls.bench = cls.work / "bench"
        subprocess.run(["cc", "-O3", "-Wall", "-Wextra", "-Werror", "-std=c11",
                        "-I" + str(ROOT / "include"), str(ROOT / "tools/bench_engram_io.c"),
                        str(ROOT / "src/myds_engram.c"), str(ROOT / "src/myds_gguf.c"),
                        str(ROOT / "src/myds_tokenizer.c"), "-lpthread", "-lm", "-o", str(cls.bench)],
                       check=True, capture_output=True, text=True, timeout=60)
        cls.records = {}
        cls.clones = {}
        for worker in (1, 4, 8):
            clone = cls.work / f"clone-{worker}.gguf"
            # Small fixture copy is intentional; a real model runner uses APFS
            # clonefile and does not copy a several-hundred-GB payload.
            shutil.copyfile(cls.model, clone)
            cls.clones[worker] = clone
            result = subprocess.run([str(cls.bench), "--model", str(clone),
                                     "--tokens-bin", str(cls.tokens), "--workers", str(worker),
                                     "--runs", "2"], capture_output=True, text=True, timeout=30)
            if result.returncode:
                raise RuntimeError(result.stderr)
            cls.records[worker] = [json.loads(line) for line in result.stdout.splitlines()]

    @classmethod
    def tearDownClass(cls):
        try:
            if hashlib.sha256(cls.model.read_bytes()).hexdigest() != cls.source_sha:
                raise AssertionError("source fixture changed")
        finally:
            cls.temp.cleanup()

    def test_serial_parallel_actual_prepare_against_scalar_oracle(self):
        oracle = expected()
        for worker, records in self.records.items():
            self.assertEqual(records[-1], {"kind": "engram_io_complete", "valid": True})
            self.assertEqual(records[0]["workers"], worker)
            for token in (r for r in records if r["kind"] == "engram_io_token"):
                rows, history, value_hash = oracle[token["pos"]]
                self.assertEqual(token["row_ids"], rows)
                self.assertEqual(token["history"], history)
                self.assertEqual(token["values_fnv1a64"], value_hash)
                self.assertEqual(token["rows_fnv1a64"], format(fnv(struct.pack("<" + "I" * 48, *rows)), "016x"))

    def test_invalid_tokens_and_worker_rejected(self):
        for data, workers in [(b"", "1"), (b"\x01", "1"), (struct.pack("<i", -1), "1"),
                              (struct.pack("<i", 5), "1"), (struct.pack("<i", 1), "2")]:
            bad = self.work / "bad.bin"
            bad.write_bytes(data)
            result = subprocess.run([str(self.bench), "--model", str(self.model), "--tokens-bin",
                                     str(bad), "--workers", workers], capture_output=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)

    def make_window(self):
        root = self.work / ("window-" + os.urandom(6).hex())
        root.mkdir()
        stat = self.model.stat()
        window = [dict(kind="window_created", order=[1, 4, 8], runs=2, mode="cached",
                       model_stat=dict(inode=stat.st_ino, bytes=stat.st_size))]
        for worker in (1, 4, 8):
            name = f"sample-workers-{worker}"
            ready = self.records[worker][0]
            window.append(dict(kind="clone_created", name=name, workers=worker,
                               inode=ready["inode"], device=ready["device"], file_bytes=ready["file_bytes"]))
            for run in (0, 1):
                window.append(dict(kind="storage_before", name=name, run=run, ioreg=driver(100000)))
                window.append(dict(kind="storage_after", name=name, run=run, ioreg=driver(300000 if run == 0 else 100000)))
            window.append(dict(kind="measurement_end", name=name, returncode=0))
            (root / (name + ".jsonl")).write_text("\n".join(json.dumps(r) for r in self.records[worker]) + "\n")
        window.append(dict(kind="window_complete", error=None, source_stat_unchanged=True, clones_removed=True))
        (root / "window.jsonl").write_text("\n".join(json.dumps(r) for r in window) + "\n")
        return root

    def test_valid_summary_and_warm_cold_separation(self):
        result = SUMMARY.summarize(self.make_window())
        self.assertEqual(set(result["by_workers"]), {"1", "4", "8"})
        for worker in result["by_workers"].values():
            self.assertEqual(worker["fresh_clone_samples"], 1)
            self.assertTrue(worker["all_fresh_clone_phases_have_device_reads"])
            self.assertEqual(len(worker["warm_prepare_ms_per_token_values"]), 1)
        self.assertFalse(result["samples"][0]["runs"][1]["device_reads_observed"])

    def test_zero_device_reads_preserved_without_claiming_cold(self):
        root = self.make_window()
        rows = SUMMARY.read_jsonl(root / "window.jsonl")
        for row in rows:
            if row["kind"] == "storage_after":
                row["ioreg"] = driver(100000)
        (root / "window.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        result = SUMMARY.summarize(root)
        self.assertTrue(all(not s["all_fresh_clone_phases_have_device_reads"] for s in result["by_workers"].values()))

    def test_corrupted_value_hash_rejected(self):
        root = self.make_window()
        path = root / "sample-workers-4.jsonl"
        rows = SUMMARY.read_jsonl(path)
        next(r for r in rows if r["kind"] == "engram_io_token")["values_fnv1a64"] = "deadbeef"
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        with self.assertRaises(ValueError):
            SUMMARY.summarize(root)

    def test_corrupted_aggregate_hash_rejected(self):
        for field in ("rows_fnv1a64", "values_fnv1a64"):
            root = self.make_window()
            path = root / "sample-workers-4.jsonl"
            rows = SUMMARY.read_jsonl(path)
            next(r for r in rows if r["kind"] == "engram_io_lookup_done")[field] = "0" * 16
            path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            with self.assertRaises(ValueError):
                SUMMARY.summarize(root)

    def test_counter_reset_and_incomplete_window_rejected(self):
        for mode in ("counter", "window"):
            root = self.make_window()
            rows = SUMMARY.read_jsonl(root / "window.jsonl")
            if mode == "counter":
                next(r for r in rows if r["kind"] == "storage_after")["ioreg"] = driver(0)
            else:
                rows[-1]["error"] = "interrupted"
            (root / "window.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            with self.assertRaises(ValueError):
                SUMMARY.summarize(root)

    @unittest.skipUnless(sys.platform == "darwin", "clonefile runner is macOS-only")
    def test_clone_runner_gates_and_cleans_small_fixture(self):
        root = self.work / ("runner-" + os.urandom(6).hex())
        clones = self.work / ("clones-" + os.urandom(6).hex())
        args = ["run_engram_io.py", "--bench", str(self.bench), "--model", str(self.model),
                "--tokens-bin", str(self.tokens), "--output", str(root),
                "--clone-directory", str(clones), "--order", "1,4,8", "--warm-runs", "1"]
        old_handlers = {signum: RUNNER.signal.getsignal(signum)
                        for signum in (RUNNER.signal.SIGINT, RUNNER.signal.SIGTERM)}
        try:
            with patch.object(sys, "argv", args), patch.object(RUNNER, "storage", return_value=driver(100000)), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(RUNNER.main(), 0)
        finally:
            for signum, handler in old_handlers.items():
                RUNNER.signal.signal(signum, handler)
        self.assertFalse(clones.exists())
        result = SUMMARY.summarize(root)
        self.assertEqual(set(result["by_workers"]), {"1", "4", "8"})
        self.assertTrue(all(not s["all_fresh_clone_phases_have_device_reads"]
                            for s in result["by_workers"].values()))


if __name__ == "__main__":
    unittest.main()
