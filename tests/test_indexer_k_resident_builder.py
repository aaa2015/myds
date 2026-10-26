#!/usr/bin/env python3
"""CPU-only checks for lossless complete-library probe preparation."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("indexer_builder", ROOT / "tools/build_indexer_k_resident_probe.py")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)
READER_SPEC = importlib.util.spec_from_file_location("indexer_reader", ROOT / "tools/review_indexer_k_resident.py")
reader = importlib.util.module_from_spec(READER_SPEC)
READER_SPEC.loader.exec_module(reader)


class FullLibraryPreparationTests(unittest.TestCase):
    def setUp(self):
        self.build, self.script = builder.load_frozen_build(ROOT)

    def test_full_native_matches_production_assembly_and_every_original_byte(self):
        actual, records, payloads = builder.assemble_full_native(ROOT, self.build)
        # Reproduce the production writer independently, using its text reads
        # and exact #line/newline order rather than builder assembly helpers.
        reference = io.StringIO()
        reference.write(self.build.PREAMBLE + "\n")
        for filename in self.build.SHADER_FILES:
            reference.write(f'\n#line 1 "{filename}"\n')
            reference.write((ROOT / "metal" / filename).read_text())
        reference.write(self.build.COPY_KERNELS + "\n")
        self.assertEqual(actual, reference.getvalue().encode())
        self.assertEqual([r["path"][6:] for r in records], self.build.SHADER_FILES)
        for record in records:
            raw = (ROOT / record["path"]).read_bytes()
            self.assertEqual(actual[record["source_begin"]:record["source_end"]], raw)
            self.assertEqual(record["sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(record["bytes"], len(raw))
            self.assertEqual(payloads[record["path"][6:]], raw)

    def test_candidate_only_append_has_no_native_redeclarations_and_exact_inverse(self):
        source = (ROOT / "metal/dsv4_misc.metal").read_text()
        fragment, proof = builder.candidate_fragment(source)
        self.assertNotIn("struct ds4_metal_args_dsv4_indexer_scores_fused {", fragment)
        self.assertNotIn("kernel void kernel_dsv4_indexer_scores_tiled_storage(", fragment)
        self.assertNotIn('host_name("kernel_dsv4_indexer_scores_tiled_', fragment)
        self.assertEqual(fragment.count('host_name("kernel_myds_indexer_scores_k_resident_'), 2)
        candidate = builder.extract(fragment, "template<bool BF16, bool WARP_ONLY = false>\nkernel void probe_indexer_scores_k_resident")
        for edit in reversed(proof["exact_edits"]):
            self.assertEqual(candidate.count(edit["after"]), 1)
            candidate = candidate.replace(edit["after"], edit["before"], 1)
        native = builder.extract(source, "template<bool BF16, bool WARP_ONLY = false>\nkernel void kernel_dsv4_indexer_scores_tiled_storage")
        self.assertEqual(candidate, native)
        self.assertEqual(proof["native_template_sha256"], hashlib.sha256(native.encode()).hexdigest())

    def test_emit_only_freezes_full_source_flags_and_invokes_no_compiler(self):
        with tempfile.TemporaryDirectory(prefix="myds-indexer-builder-host-") as temporary:
            output = Path(temporary) / "full"
            argv = ["builder", "--source-root", str(ROOT), "--out-dir", str(output), "--full-library", "--emit-only"]
            environment = {"MYDS_METAL_MINVER": "15.0", "MYDS_METAL_STD": "metal3.2",
                           "MYDS_METALLIB_DEFINES": "-DTEST_HOST_FREEZE=1 -DTEST_SECOND=2"}
            with mock.patch.object(sys, "argv", argv), mock.patch.dict(os.environ, environment), \
                    mock.patch.object(builder.subprocess, "run", side_effect=AssertionError("compiler invoked")), \
                    contextlib.redirect_stdout(io.StringIO()):
                builder.main()
            identity = json.loads((output / "source-identity.json").read_text())
            self.assertEqual(identity["build_mode"], "full-library")
            self.assertTrue(identity["compilation_skipped"])
            self.assertFalse(identity["gpu_work_submitted"])
            self.assertFalse(identity["production_registered"])
            self.assertFalse((output / "probe.metallib").exists())
            self.assertFalse((output / "native.metallib").exists())
            native = (output / "native.metal").read_bytes()
            append = identity["candidate_append_marker"].encode() + (output / "probe.metal").read_bytes()
            self.assertEqual((output / "all.metal").read_bytes(), native + append)
            self.assertEqual(identity["native_source_sha256"], hashlib.sha256(native).hexdigest())
            for record in identity["original_shaders"]:
                self.assertEqual((output / "native-source" / record["path"]).read_bytes(), (ROOT / record["path"]).read_bytes())
            self.assertEqual((output / "native-source/tools/build_metallib.py").read_bytes(), self.script)
            metal = [command for command in identity["commands"] if command[3] == "metal"]
            self.assertEqual(len(metal), 2)
            for command in metal:
                self.assertIn("-std=metal3.2", command)
                self.assertIn("-DTEST_HOST_FREEZE=1", command)
                self.assertIn("-DTEST_SECOND=2", command)
                self.assertEqual(command[4:command.index("-c")], identity["compile_options"])

    def test_legacy_small_source_layout_remains_available(self):
        source = (ROOT / "metal/dsv4_misc.metal").read_text()
        small, proof = builder.generate(source)
        self.assertIn("struct ds4_metal_args_dsv4_indexer_scores_fused {", small)
        self.assertIn('host_name("kernel_dsv4_indexer_scores_tiled_f32")', small)
        self.assertIn('host_name("kernel_dsv4_indexer_scores_tiled_bf16")', small)
        self.assertTrue(proof["reverse_edits_restore_native_byte_for_byte"])
        # This exact source generated the recorded historical small-mode run.
        if hashlib.sha256(source.encode()).hexdigest() == "7fe23a7707e97758c3127caadfb2bc3b6818c1c6ba9d30a73f632563b82f3d57":
            self.assertEqual(hashlib.sha256(small.encode()).hexdigest(), "3bf735c0ac4e7a009813e26bd1f61864f04dd939e548c513de70176b002be57d")

    def test_independent_reader_replays_full_source_and_rejects_changed_shader(self):
        with tempfile.TemporaryDirectory(prefix="myds-indexer-reader-host-") as temporary:
            output = Path(temporary) / "full"
            argv = ["builder", "--source-root", str(ROOT), "--out-dir", str(output), "--full-library", "--emit-only"]
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(builder.subprocess, "run", side_effect=AssertionError("compiler invoked")), \
                    contextlib.redirect_stdout(io.StringIO()):
                builder.main()
            identity = json.loads((output / "source-identity.json").read_text())
            generated, combined = (output / "probe.metal").read_text(), (output / "all.metal").read_text()
            native = reader.verify_full_native_source(output, identity, generated, combined)
            self.assertEqual(native.encode(), (output / "native.metal").read_bytes())
            first = output / "native-source/metal" / self.build.SHADER_FILES[0]
            first.write_bytes(first.read_bytes() + b"\n// deliberate host corruption\n")
            with self.assertRaises(ValueError):
                reader.verify_full_native_source(output, identity, generated, combined)

    def test_missing_shader_and_duplicate_shader_fail_closed(self):
        clone = type("FrozenBuild", (), dict(PREAMBLE=self.build.PREAMBLE, COPY_KERNELS=self.build.COPY_KERNELS,
                                              SHADER_FILES=self.build.SHADER_FILES + ["absent-indexer-probe.metal"]))
        with self.assertRaises(FileNotFoundError):
            builder.assemble_full_native(ROOT, clone)
        clone.SHADER_FILES = self.build.SHADER_FILES + [self.build.SHADER_FILES[0]]
        with self.assertRaises(ValueError):
            builder.assemble_full_native(ROOT, clone)


if __name__ == "__main__":
    unittest.main()
