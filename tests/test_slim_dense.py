#!/usr/bin/env python3
"""Unit tests for tools/slim_dense_q4.py.

Validates end-to-end slimming of dense Q8_0 weights to Q4_K on a synthetic GGUF:
- Verifies target dense tensors are converted to Q4_K.
- Verifies non-target tensors are preserved bit-for-bit.
- Verifies alignment and header recalculation.
- Verifies source GGUF is completely untouched and preserved.
"""

import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

    def test_collision_symlink_parent_rejected(self):
        # F1: output resolves to same real file through symlinked parent directory
        import tempfile
        tmp_dir = tempfile.mkdtemp()
        try:
            real_dir = os.path.join(tmp_dir, "real")
            link_dir = os.path.join(tmp_dir, "link")
            os.makedirs(real_dir)
            src_file = os.path.join(real_dir, "model.gguf")
            create_synthetic_model(src_file)
            with open(src_file, "rb") as f:
                orig_sha = hashlib.sha256(f.read()).hexdigest()
            os.symlink(real_dir, link_dir, target_is_directory=True)
            out_file = os.path.join(link_dir, "model.gguf")

            cmd = [sys.executable, "tools/slim_dense_q4.py", "--in", src_file, "--out", out_file, "--force"]
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertNotEqual(res.returncode, 0)
            self.assertIn("resolves to the same real path", res.stderr)

            # Verify source was untouched
            with open(src_file, "rb") as f:
                after_sha = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(orig_sha, after_sha)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_collision_partial_input_rejected(self):
        # F1: input happens to be named x.partial, output named x -> would collide with x.partial
        partial_src = self.src_path + ".partial"
        os.rename(self.src_path, partial_src)
        with open(partial_src, "rb") as f:
            orig_sha = hashlib.sha256(f.read()).hexdigest()

        cmd = [sys.executable, "tools/slim_dense_q4.py", "--in", partial_src, "--out", self.src_path]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("resolves to the same real path", res.stderr)

        # Verify source was not truncated
        with open(partial_src, "rb") as f:
            after_sha = hashlib.sha256(f.read()).hexdigest()
        self.assertEqual(orig_sha, after_sha)
        os.rename(partial_src, self.src_path)

    def test_resume_source_mismatch_rejected(self):
        # F2: resume with modified source header/size is rejected
        journal_path = self.dst_path + ".journal.json"
        partial_path = self.dst_path + ".partial"

        # Create dummy partial and journal with fake header sha
        with open(partial_path, "wb") as f:
            f.write(b"0" * 4096)
        with open(journal_path, "w") as f:
            json.dump({
                "completed_tensor_idx": 1,
                "tensor_name": "blk.0.ffn_gate_shexp.weight",
                "source_header_sha256": "0" * 64,
                "source_size": 999999,
                "n_tensors": 3,
            }, f)

        cmd = [sys.executable, "tools/slim_dense_q4.py", "--in", self.src_path, "--out", self.dst_path, "--resume"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("Resume rejected! Source model, metadata, alignment, or quantizer library has changed", res.stderr)

    def test_resume_all_tensors_completed_finalizes(self):
        # F2: when journal completed_tensor_idx == n_tensors, cleanly finalize without IndexError
        # First do a complete run
        cmd = [sys.executable, "tools/slim_dense_q4.py", "--in", self.src_path, "--out", self.dst_path, "--threads", "2"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)

        # Rename out to partial, write journal with completed_tensor_idx == 3
        partial_path = self.dst_path + ".partial"
        journal_path = self.dst_path + ".journal.json"
        os.rename(self.dst_path, partial_path)

        with open(self.src_path, "rb") as f:
            # Read header up to data_start
            f.seek(0)
            f.read(4)
            _, nt, nk = struct.unpack("<IQQ", f.read(20))
            def rs():
                l = struct.unpack("<Q", f.read(8))[0]
                return f.read(l)
            for _ in range(nk):
                rs()
                vt = struct.unpack("<I", f.read(4))[0]
                if vt == 4: f.seek(4, os.SEEK_CUR)
                elif vt == 8: rs()
            for _ in range(nt):
                rs()
                nd = struct.unpack("<I", f.read(4))[0]
                f.seek(8 * nd + 12, os.SEEK_CUR)
            hdr_end = f.tell()
            f.seek(0)
            hdr_sha = hashlib.sha256(f.read(hdr_end)).hexdigest()

        # Load quants library to get sha
        from tools.slim_dense_q4 import load_quants_library
        _, _, lib_sha = load_quants_library(None)

        with open(journal_path, "w") as jf:
            json.dump({
                "completed_tensor_idx": 3,
                "tensor_name": "blk.0.attn_output_a.weight",
                "source_header_sha256": hdr_sha,
                "source_size": os.path.getsize(self.src_path),
                "n_tensors": 3,
                "quants_lib_sha256": lib_sha,
                "alignment": GGUF_ALIGNMENT,
            }, jf)

        # Resume should detect all completed and rename cleanly without IndexError
        cmd = [sys.executable, "tools/slim_dense_q4.py", "--in", self.src_path, "--out", self.dst_path, "--resume"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Resume failed: {res.stderr}")
        self.assertIn("All 3 tensors already completed. Finalizing output file", res.stdout)
        self.assertTrue(os.path.exists(self.dst_path))
        self.assertFalse(os.path.exists(journal_path))

    def test_zero_target_dry_run(self):
        # F8: zero targets in dry-run should not divide by zero
        import tempfile
        zero_src = "/tmp/test_zero_target.gguf"
        zero_out = "/tmp/test_zero_out.gguf"
        try:
            with open(zero_src, "wb") as f:
                f.write(GGUF_MAGIC + struct.pack("<IQQ", 3, 1, 1))
                write_str(f, "general.architecture")
                f.write(struct.pack("<I", 8))
                write_str(f, "deepseek41")
                # Single non-target tensor
                write_str(f, "token_embd.weight")
                f.write(struct.pack("<I2QIQ", 2, 256, 2, TYPE_F32, 0))
                hdr_end = f.tell()
                data_start = (hdr_end + 31) // 32 * 32
                f.write(bytes(data_start - hdr_end))
                f.write(b"\x00" * (256 * 2 * 4))

            cmd = [sys.executable, "tools/slim_dense_q4.py", "--in", zero_src, "--out", zero_out, "--dry-run"]
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"Zero target dry run failed: {res.stderr}")
            self.assertIn("0.0% reduction", res.stdout)
        finally:
            if os.path.exists(zero_src): os.unlink(zero_src)
            if os.path.exists(zero_out): os.unlink(zero_out)

    def test_explicit_missing_quants_lib_fails_fast(self):
        # F9: missing explicit library must fail fast
        cmd = [sys.executable, "tools/slim_dense_q4.py", "--in", self.src_path, "--out", self.dst_path,
               "--quants-lib", "/nonexistent/path/libds4quants.dylib"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("Explicit quants library '/nonexistent/path/libds4quants.dylib' not found", res.stderr)

    def test_strict_whitelist(self):
        # F7: strict whitelist
        from tools.slim_dense_q4 import is_dense_target
        # Allowed targets (Q8_0)
        self.assertTrue(is_dense_target("output.weight", TYPE_Q8_0))
        self.assertTrue(is_dense_target("blk.0.attn_q_a.weight", TYPE_Q8_0))
        self.assertTrue(is_dense_target("blk.39.ffn_down_shexp.weight", TYPE_Q8_0))
        self.assertTrue(is_dense_target("blk.12.attn_output_b.weight", TYPE_Q8_0))

        # Rejected (non-target or invalid pattern)
        self.assertFalse(is_dense_target("output.weight", TYPE_F32)) # not Q8_0
        self.assertFalse(is_dense_target("blk.0.indexer.attn_q_b.weight", TYPE_Q8_0)) # nested indexer
        self.assertFalse(is_dense_target("blk.not_a_layer.attn_q_a.weight", TYPE_Q8_0)) # non-numeric layer
        self.assertFalse(is_dense_target("blk.0.unknown_shexp.weight", TYPE_Q8_0)) # unknown suffix
        self.assertFalse(is_dense_target("blk.0.ffn_gate_exps.weight", TYPE_Q8_0)) # routed expert


if __name__ == "__main__":
    unittest.main()

