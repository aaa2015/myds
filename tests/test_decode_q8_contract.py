#!/usr/bin/env python3
"""Compile the actual wrapper without Metal/model; MYDS_TEST_TSAN=1 enables TSan."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class DecodeQ8Contract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="myds-decode-q8-host-")
        cls.binary = Path(cls.temp.name) / "test_decode_q8_contract"
        sanitizer = ["-fsanitize=thread", "-g"] if os.environ.get("MYDS_TEST_TSAN") == "1" else []
        compilation = subprocess.run([
            "clang", "-O2", "-Wall", "-Wextra", "-Werror", "-std=c11",
            *sanitizer,
            "-I", str(ROOT / "include"),
            str(ROOT / "tests/test_decode_q8_contract.c"),
            str(ROOT / "src/myds_v41_decode.c"), "-lpthread", "-o", str(cls.binary),
        ], capture_output=True, text=True)
        if compilation.returncode:
            raise RuntimeError(f"Host contract compilation failed:\n{compilation.stderr}")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def run_contract(self, value, expected):
        env = os.environ.copy()
        env.pop("MYDS_DECODE_Q8_FUSE_BF16", None)
        if value is not None:
            env["MYDS_DECODE_Q8_FUSE_BF16"] = value
        if os.environ.get("MYDS_TEST_TSAN") == "1":
            env["TSAN_OPTIONS"] = "halt_on_error=1:exitcode=66"
        result = subprocess.run([str(self.binary), str(expected)], env=env,
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("WARNING: ThreadSanitizer", result.stderr)
        self.assertIn("PASS decode Q8 wrapper", result.stdout)

    def test_default_enabled(self):
        self.run_contract(None, 1)

    def test_disabled_zero(self):
        self.run_contract("0", 0)

    def test_enabled_one(self):
        self.run_contract("1", 1)

    def test_empty_uses_default(self):
        self.run_contract("", 1)

    def test_false_text_uses_default(self):
        self.run_contract("false", 1)

    def test_other_text_uses_default(self):
        self.run_contract("invalid", 1)


if __name__ == "__main__":
    unittest.main()
