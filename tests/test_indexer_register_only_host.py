#!/usr/bin/env python3
"""CPU-only regression for the actual register-only selector and publisher.

Compiles source-including fixtures with stubbed Metal calls, so the selector's
scope and failure propagation are exercised without a device, model or shader
library. Also checks the offline writer preserves the qualified native prefix.
"""
import contextlib
import hashlib
import importlib.util
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
APPROVED_SHADER_SHA = "a62922c6f5937236bd0e354dd06943b39bfa5f788419c16a8700da793769d616"


class RegisterOnlyHostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="myds-register-only-host-")
        cls.out = Path(cls.temporary.name)
        compiler = shutil.which("cc")
        if not compiler:
            raise RuntimeError("A C compiler is required for the source-including fixtures")
        for kind in ("runtime", "publisher"):
            name = "test_indexer_register_only_" + kind + "_host"
            command = [compiler, "-std=c11", "-O2", "-ffunction-sections", "-fdata-sections",
                       "-Iinclude", "-Ithird_party", "-Wl,-dead_strip"]
            if kind == "publisher":
                command += ["-Wno-unused-function", "-Wno-unused-variable",
                            "-Wno-unused-but-set-variable", "-Wno-misleading-indentation"]
            command += ["tests/" + name + ".c", "-o", str(cls.out / name), "-lm", "-lpthread"]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(f"Fixture compilation failed: {command!r}\n{result.stderr}")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def run_selector(self, value, case, extra=None):
        env = {k: v for k, v in os.environ.items()
               if not k.startswith("MYDS_") and k != "EXPECT_MODE"}
        if value is not None:
            env["MYDS_INDEXER_REGISTER_ONLY"] = value
        if case == "parse":
            env["EXPECT_MODE"] = "1" if value in (None, "1") else "0" if value == "0" else "-1"
        env.update(extra or {})
        result = subprocess.run([str(self.out / "test_indexer_register_only_runtime_host"), case],
                                env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def test_actual_selector_scope_and_fail_closed_contract(self):
        for value in (None, "0", "1", "", "2", "01", "true", " 1", "1 "):
            with self.subTest(parse=value):
                self.run_selector(value, "parse")
        for value in (None, "0", "1", "2"):
            with self.subTest(matrix=value):
                output = self.run_selector(value, "matrix")
                if value in (None, "1"):
                    self.assertIn('"controls":50', output)
        self.run_selector("1", "matrix", {"MYDS_INDEXER_KEY_WINDOW": "16"})

    def test_actual_publisher_stops_after_scorer_failure(self):
        result = subprocess.run([str(self.out / "test_indexer_register_only_publisher_host")],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('"no_repeat_rope_or_consumers":true', result.stdout)

    def test_offline_writer_preserves_native_prefix_and_qualified_fragment(self):
        spec = importlib.util.spec_from_file_location("register_only_production_builder",
                                                     ROOT / "tools/build_metallib.py")
        build = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(build)
        fragment = (ROOT / "metal" / build.REGISTER_ONLY_SHADER).read_bytes()
        self.assertEqual(hashlib.sha256(fragment).hexdigest(), APPROVED_SHADER_SHA)
        reference = bytearray(build.PREAMBLE.encode() + b"\n")
        for name in build.SHADER_FILES:
            reference.extend(f'\n#line 1 "{name}"\n'.encode())
            reference.extend((ROOT / "metal" / name).read_bytes())
        reference.extend(build.COPY_KERNELS.encode() + b"\n")
        native = bytes(reference)
        output = self.out / "writer"
        library = output / "bin/myds.metallib"

        def compiler_stub(command, **kwargs):
            # The writer reaches both compile steps, but this test cannot run
            # a compiler, create a Metal device, or claim a compiled library.
            if "-c" in command:
                source = Path(command[command.index("-c") + 1])
                self.assertEqual(source, output / "build/all.metal")
                self.assertEqual(source.read_bytes(), (output / "build/myds_all.metal").read_bytes())
                self.assertEqual(command[command.index("-o") + 1], str(output / "build/myds_all.air"))
            library.write_bytes(b"CPU assembly fixture")
            return subprocess.CompletedProcess(command, 0)

        argv = ["build_metallib", "--bin-dir", str(library.parent),
                "--build-dir", str(output / "build")]
        with mock.patch.object(sys, "argv", argv), \
                mock.patch.object(build.subprocess, "run", side_effect=compiler_stub) as calls, \
                contextlib.redirect_stdout(io.StringIO()):
            build.main()
        combined = (output / "build/myds_all.metal").read_bytes()
        self.assertEqual((output / "build/all.metal").read_bytes(), combined)
        self.assertEqual(combined, native + build.REGISTER_ONLY_MARKER.encode() + fragment)
        self.assertEqual(combined[:len(native)], native)
        self.assertEqual(calls.call_count, 2)
        self.assertEqual(combined.count(b"kernel void kernel_myds_indexer_scores_register_only_bf16("), 1)


if __name__ == "__main__":
    unittest.main()
