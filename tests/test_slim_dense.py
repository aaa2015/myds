#!/usr/bin/env python3
"""Unit tests for tools/slim_dense_q4.py.

Validates end-to-end slimming of dense Q8_0 weights to Q4_K on a synthetic GGUF:
- Verifies target dense tensors are converted to Q4_K.
- Verifies non-target tensors are preserved bit-for-bit.
- Verifies alignment and header recalculation.
- Verifies source GGUF is completely untouched and preserved.
"""

import hashlib
import os
import struct
import subprocess
import sys
import unittest

GGUF_MAGIC = b"GGUF"
GGUF_ALIGNMENT = 16384

TYPE_F32 = 0
TYPE_Q8_0 = 8
TYPE_Q4_K = 12


def align(val, a=GGUF_ALIGNMENT):
    return (val + a - 1) // a * a


def write_str(f, s):
    data = s.encode("utf-8")
    f.write(struct.pack("<Q", len(data)) + data)


def create_synthetic_model(path):
    with open(path, "wb") as f:
        # Header
        n_tensors = 3
        n_kv = 2
        f.write(GGUF_MAGIC + struct.pack("<IQQ", 3, n_tensors, n_kv))

        # KV 0: general.architecture
        write_str(f, "general.architecture")
        f.write(struct.pack("<I", 8))  # string
        write_str(f, "deepseek41")

        # KV 1: general.alignment
        write_str(f, "general.alignment")
        f.write(struct.pack("<II", 4, GGUF_ALIGNMENT))

        # Tensor 0: blk.0.ffn_gate_exps.weight (F32, non-target)
        # dims: [256, 2], size: 256 * 2 * 4 = 2048 bytes
        t0_data = b"\x11" * 2048
        t0_size = len(t0_data)

        # Tensor 1: blk.0.ffn_gate_shexp.weight (Q8_0, target)
        # dims: [256, 4] -> 4 rows of 256 elements
        # 1 row = 256 / 32 * 34 = 272 bytes; 4 rows = 1088 bytes
        t1_data = b"\x22" * 1088
        t1_size = len(t1_data)

        # Tensor 2: blk.0.attn_output_a.weight (Q8_0, target)
        # dims: [512, 2] -> 2 rows of 512 elements
        # 1 row = 512 / 32 * 34 = 544 bytes; 2 rows = 1088 bytes
        t2_data = b"\x33" * 1088
        t2_size = len(t2_data)

        t0_off = 0
        t1_off = align(t0_off + t0_size, GGUF_ALIGNMENT)
        t2_off = align(t1_off + t1_size, GGUF_ALIGNMENT)

        # Tensor descriptors
        # Tensor 0
        write_str(f, "blk.0.ffn_gate_exps.weight")
        f.write(struct.pack("<I2QIQ", 2, 256, 2, TYPE_F32, t0_off))

        # Tensor 1
        write_str(f, "blk.0.ffn_gate_shexp.weight")
        f.write(struct.pack("<I2QIQ", 2, 256, 4, TYPE_Q8_0, t1_off))

        # Tensor 2
        write_str(f, "blk.0.attn_output_a.weight")
        f.write(struct.pack("<I2QIQ", 2, 512, 2, TYPE_Q8_0, t2_off))

        hdr_end = f.tell()
        data_start = align(hdr_end, GGUF_ALIGNMENT)
        f.write(bytes(data_start - hdr_end))

        # Data payloads
        assert f.tell() == data_start + t0_off
        f.write(t0_data)
        f.write(bytes(align(t0_size, GGUF_ALIGNMENT) - t0_size))

        assert f.tell() == data_start + t1_off
        f.write(t1_data)
        f.write(bytes(align(t1_size, GGUF_ALIGNMENT) - t1_size))

        assert f.tell() == data_start + t2_off
        f.write(t2_data)
        f.write(bytes(align(t2_size, GGUF_ALIGNMENT) - t2_size))


class SlimDenseTests(unittest.TestCase):
    def setUp(self):
        self.src_path = "/tmp/test_mini_dense_q8.gguf"
        self.dst_path = "/tmp/test_mini_dense_q4.gguf"
        if os.path.exists(self.src_path):
            os.unlink(self.src_path)
        if os.path.exists(self.dst_path):
            os.unlink(self.dst_path)
        create_synthetic_model(self.src_path)
        with open(self.src_path, "rb") as f:
            self.orig_sha256 = hashlib.sha256(f.read()).hexdigest()

    def tearDown(self):
        for p in (self.src_path, self.dst_path, self.dst_path + ".partial", self.dst_path + ".journal.json"):
            if os.path.exists(p):
                os.unlink(p)

    def test_dry_run_does_not_modify_disk(self):
        cmd = [
            sys.executable,
            "tools/slim_dense_q4.py",
            "--in", self.src_path,
            "--out", self.dst_path,
            "--dry-run",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"dry-run failed: {res.stderr}")
        self.assertIn("[DRY-RUN] Verification complete", res.stdout)
        self.assertFalse(os.path.exists(self.dst_path))

    def test_slim_conversion_and_preservation(self):
        cmd = [
            sys.executable,
            "tools/slim_dense_q4.py",
            "--in", self.src_path,
            "--out", self.dst_path,
            "--threads", "2",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Conversion failed:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}")
        self.assertTrue(os.path.exists(self.dst_path))

        # 1. Source file must remain completely unchanged
        with open(self.src_path, "rb") as f:
            new_src_sha256 = hashlib.sha256(f.read()).hexdigest()
        self.assertEqual(self.orig_sha256, new_src_sha256, "Original baseline file was modified!")

        # 2. Verify destination GGUF structure
        with open(self.dst_path, "rb") as f:
            magic = f.read(4)
            self.assertEqual(magic, GGUF_MAGIC)
            version, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))
            self.assertEqual(version, 3)
            self.assertEqual(n_tensors, 3)
            self.assertEqual(n_kv, 2)

            # Skip KVs
            def read_s():
                l = struct.unpack("<Q", f.read(8))[0]
                return f.read(l).decode("utf-8")

            read_s()
            f.seek(4, os.SEEK_CUR)
            read_s()
            read_s()
            f.seek(8, os.SEEK_CUR)

            # Read tensors
            tensors = []
            for _ in range(n_tensors):
                name = read_s()
                ndims = struct.unpack("<I", f.read(4))[0]
                dims = struct.unpack(f"<{ndims}Q", f.read(8 * ndims))
                ttype, offset = struct.unpack("<IQ", f.read(12))
                tensors.append((name, dims, ttype, offset))

            data_start = align(f.tell(), GGUF_ALIGNMENT)

        # Tensor 0 (non-target)
        self.assertEqual(tensors[0][0], "blk.0.ffn_gate_exps.weight")
        self.assertEqual(tensors[0][2], TYPE_F32)
        self.assertEqual(tensors[0][3] % GGUF_ALIGNMENT, 0)

        # Verify Tensor 0 payload is bit-exact to original
        with open(self.src_path, "rb") as f_src, open(self.dst_path, "rb") as f_dst:
            f_src.seek(data_start + 0)
            f_dst.seek(data_start + tensors[0][3])
            self.assertEqual(f_src.read(2048), f_dst.read(2048))

        # Tensor 1 (target, converted to Q4_K)
        self.assertEqual(tensors[1][0], "blk.0.ffn_gate_shexp.weight")
        self.assertEqual(tensors[1][2], TYPE_Q4_K)
        self.assertEqual(tensors[1][3] % GGUF_ALIGNMENT, 0)
        # Expected Q4_K size for [256, 4]: 4 rows * (256/256 * 144) = 576 bytes (vs 1088 Q8_0)
        expected_t1_size = 576

        # Tensor 2 (target, converted to Q4_K)
        self.assertEqual(tensors[2][0], "blk.0.attn_output_a.weight")
        self.assertEqual(tensors[2][2], TYPE_Q4_K)
        self.assertEqual(tensors[2][3] % GGUF_ALIGNMENT, 0)
        # Expected Q4_K size for [512, 2]: 2 rows * (512/256 * 144) = 576 bytes (vs 1088 Q8_0)
        expected_t2_size = 576

        # Check offsets are compact and aligned
        self.assertEqual(tensors[1][3], align(tensors[0][3] + 2048, GGUF_ALIGNMENT))
        self.assertEqual(tensors[2][3], align(tensors[1][3] + expected_t1_size, GGUF_ALIGNMENT))


if __name__ == "__main__":
    unittest.main()
