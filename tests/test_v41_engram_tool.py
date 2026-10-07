#!/usr/bin/env python3
"""Unit tests for tools/v41_engram_tool.py"""
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.v41_engram_tool import (
    MYDS_ENGRAM_COLS,
    MYDS_ENGRAM_DIM,
    MYDS_ENGRAM_LAYERS,
    MYDS_ENGRAM_NGRAM,
    MYDS_ENGRAM_ROW_BYTES,
    build_compressed_token_map,
    generate_layer_multipliers,
    generate_prime_buckets,
    normalize_token_text,
    quantize_row_fp32,
    quantize_table_stream,
    validate_engram_metadata,
)


class TestV41EngramTool(unittest.TestCase):
    def test_token_normalization(self):
        cases = [
            ("Hello", "hello"),
            ("Café", "cafe"),
            ("naïve", "naive"),
            ("   multiple   spaces  \t\n", " multiple spaces "),
            ("DeepSeek-V4.1", "deepseek-v4.1"),
            ("你好🙂", "你好🙂"),
        ]
        for inp, expected in cases:
            self.assertEqual(normalize_token_text(inp), expected)

    def test_compressed_token_map(self):
        vocab = ["Hello", "hello", "HELLO", "world", "world!", "café", "cafe"]
        token_map, comp_size = build_compressed_token_map(vocab, max_compressed_size=100)
        self.assertEqual(len(token_map), len(vocab))
        # "Hello", "hello", "HELLO" all normalize to "hello" and share token ID 0
        self.assertEqual(token_map[0], token_map[1])
        self.assertEqual(token_map[1], token_map[2])
        # "café" and "cafe" normalize to "cafe" and share token ID
        self.assertEqual(token_map[5], token_map[6])
        self.assertLess(max(token_map), comp_size)

    def test_multipliers(self):
        for layer in [1, 14]:
            mults = generate_layer_multipliers(layer, MYDS_ENGRAM_NGRAM)
            self.assertEqual(len(mults), MYDS_ENGRAM_NGRAM)
            for m in mults:
                self.assertEqual(m & 1, 1, "Multiplier must be odd")
                self.assertGreater(m, 0)
                self.assertLessEqual(m, 0x7FFFFFFFFFFFFFFF)

    def test_prime_buckets(self):
        total_rows = 100_000
        primes = generate_prime_buckets(total_rows, MYDS_ENGRAM_COLS)
        self.assertEqual(len(primes), MYDS_ENGRAM_COLS)
        self.assertEqual(sum(primes), total_rows)
        for p in primes:
            self.assertGreaterEqual(p, 2)

    def test_quantization_and_chunked_streaming(self):
        num_rows = 10
        floats = [float(i % 17) * 0.125 - 1.0 for i in range(num_rows * MYDS_ENGRAM_DIM)]

        with tempfile.TemporaryDirectory() as tmpdir:
            in_file = Path(tmpdir) / "in.bin"
            out_file = Path(tmpdir) / "out.bin"

            in_file.write_bytes(struct.pack(f"<{len(floats)}f", *floats))

            written = quantize_table_stream(in_file, out_file, num_rows, chunk_size=4)
            self.assertEqual(written, num_rows)
            self.assertEqual(out_file.stat().st_size, num_rows * MYDS_ENGRAM_ROW_BYTES)

    def test_metadata_validator(self):
        rows = [10000, 10000]
        valid_meta = {
            "deepseek41.engram.layer_ids": [1, 14],
            "deepseek41.engram.rows": rows,
            "deepseek41.engram.primes": [generate_prime_buckets(r) for r in rows],
            "deepseek41.engram.multipliers": [
                generate_layer_multipliers(1),
                generate_layer_multipliers(14),
            ],
            "deepseek41.engram.pad_id": 0,
            "deepseek41.engram.token_map": [0, 1, 2, 3, 0],
        }
        ok, errs = validate_engram_metadata(valid_meta)
        self.assertTrue(ok, f"Expected valid metadata, got errors: {errs}")

        # Test invalid layer_ids
        bad_meta = dict(valid_meta, **{"deepseek41.engram.layer_ids": [0, 1]})
        ok, errs = validate_engram_metadata(bad_meta)
        self.assertFalse(ok)

        # Test even multiplier
        bad_mults = [[2, 3, 5, 7], [11, 13, 17, 19]]
        bad_meta = dict(valid_meta, **{"deepseek41.engram.multipliers": bad_mults})
        ok, errs = validate_engram_metadata(bad_meta)
        self.assertFalse(ok)

        # Test prime sum mismatch
        bad_primes = [list(valid_meta["deepseek41.engram.primes"][0]), list(valid_meta["deepseek41.engram.primes"][1])]
        bad_primes[0][0] += 10
        bad_meta = dict(valid_meta, **{"deepseek41.engram.primes": bad_primes})
        ok, errs = validate_engram_metadata(bad_meta)
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
