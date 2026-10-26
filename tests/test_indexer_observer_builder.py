#!/usr/bin/env python3
"""CPU-only mutation/identity fixtures for the literal observer builder."""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("indexer_observer_builder", ROOT / "tools/build_indexer_observer_probe.py")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class LiteralObserverBuilderTests(unittest.TestCase):
    def setUp(self):
        self.original = (ROOT / builder.ORIGINAL).read_bytes()

    def test_complete_inverse_and_original_helpers_phase1_observer_unchanged(self):
        generated, proof = builder.generate(self.original)
        builder.verify(self.original, generated, proof)
        self.assertEqual(builder.inverse(generated, proof["exact_edits"]), self.original)
        self.assertEqual(proof["edit_count"], 6)
        main = b"int main(int argc,char **argv){@autoreleasepool {"
        self.assertEqual(self.original.split(main)[0], generated.split(main)[0])
        begin = b"    // Phase 1: every layer's complete scores and actual ordered Top512 consumer"
        end = b"    real_timing_phase=YES;"
        self.assertEqual(self.original.split(begin)[1].split(end)[0], generated.split(begin)[1].split(end)[0])
        observer_begin = b"                if(!real_index_validate(f,provider,NO)) {"
        observer_end = b'                emit(@{@"kind":@"real_timing"'
        self.assertEqual(self.original.split(observer_begin)[1].split(observer_end)[0],
                         generated.split(observer_begin)[1].split(observer_end)[0])
        self.assertIn(b"real_index_run(queue,f,provider,psos[ci][actual_provider_index])", generated)
        self.assertIn(b"sham_tiled_b&&control==1u&&arm==1u", generated)
        self.assertIn(b'@"actual_provider_index":@(actual_provider_index),@"tiled_b_sham":@(tiled_b_sham)', generated)
        self.assertNotIn(b"stability_run", generated)
        self.assertNotIn(b"host_snapshot", generated)
        self.assertIn(b'finish(@"diagnostic_complete_not_adoption")', generated)

    def test_only_sham_middle_tiled_binding_differs_and_boolean_option_needs_no_value(self):
        generated, _ = builder.generate(self.original)
        cli = generated.split(b"    for(int i=1;i<argc;i++) {")[1].split(b"    need(capture_path")[0]
        self.assertLess(cli.index(b'"--sham-tiled-b"'), cli.index(b'need(i+1<argc,"option value")'))
        # Freeze the complete logical-provider/actual-PSO contract, including
        # exact-control B and every real-mode arm, independently of GPU output.
        for sham in (False, True):
            actual = []
            logical = []
            for control in (0, 1):
                for arm, provider in enumerate((control, 2, control)):
                    logical.append(provider)
                    actual.append(1 if sham and control == 1 and arm == 1 else provider)
            self.assertEqual(logical, [0, 2, 0, 1, 2, 1])
            self.assertEqual(actual, [0, 2, 0, 1, 1, 1] if sham else logical)
        self.assertIn(b'else [e setBytes:&f->args length:72 atIndex:0];', generated)
        self.assertIn(b'[e setBuffer:f->out[provider]offset:GUARD atIndex:4];', generated)

    def test_missing_or_duplicate_anchor_and_unlisted_mutation_are_rejected(self):
        anchor = builder.EDITS[3][1]
        for bad in (self.original.replace(anchor, b"", 1), self.original + anchor):
            with self.assertRaises(ValueError):
                builder.generate(bad)
        generated, proof = builder.generate(self.original)
        with self.assertRaises(ValueError):
            builder.verify(self.original, generated + b"\n// extra observer edit\n", proof)
        changed = copy.deepcopy(proof)
        changed["exact_edits"][3]["after"] = changed["exact_edits"][3]["after"].replace("provider,psos", "1,psos")
        with self.assertRaises(ValueError):
            builder.verify(self.original, generated, changed)

    def run_builder(self, out, *extra):
        argv = ["builder", "--source-root", str(ROOT), "--out-dir", str(out), *extra]
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            builder.main()

    def test_emit_only_freezes_whole_seed_diff_commands_and_executes_nothing(self):
        with tempfile.TemporaryDirectory(prefix="myds-indexer-observer-test-") as temporary:
            out = Path(temporary) / "probe"
            with mock.patch.object(builder.subprocess, "run", side_effect=AssertionError("execution forbidden")) as run:
                self.run_builder(out, "--emit-only")
            run.assert_not_called()
            identity = json.loads((out / "build-manifest.json").read_text())
            seed, generated = (Path(identity[key]).read_bytes() for key in ("original_seed", "generated_source"))
            proof = json.loads((out / "source-transform.json").read_text())
            self.assertEqual(seed, self.original)
            builder.verify(seed, generated, proof)
            for rel, record in identity["inputs"].items():
                frozen = (out / "source" / rel).read_bytes()
                self.assertEqual(record, dict(bytes=len(frozen), sha256=hashlib.sha256(frozen).hexdigest()))
            for filename, key in (("observer-source.diff", "source_diff_sha256"),
                                  ("source-transform.json", "transformation_proof_sha256"),
                                  ("build-commands.json", "commands_sha256")):
                self.assertEqual(hashlib.sha256((out / filename).read_bytes()).hexdigest(), identity[key])
            self.assertEqual(len(identity["commands"]), 7)
            for command in identity["commands"]:
                self.assertEqual(command[:2], ["xcrun", "clang"])
            self.assertIn("-O3", identity["commands"][0])
            self.assertTrue(identity["diagnostic_only"])
            for key in ("compile_only_pass", "gpu_work_submitted", "binary_executed", "adoption_cohort",
                        "production_adoption", "runtime_registered", "fullmodel_eligible"):
                self.assertFalse(identity[key], key)
            self.assertTrue(identity["compilation_skipped"])
            self.assertFalse((out / "bench_indexer_observer_probe").exists())
            self.assertEqual((ROOT / builder.ORIGINAL).read_bytes(), self.original)

    def test_existing_output_and_checkout_output_refused(self):
        with tempfile.TemporaryDirectory(prefix="myds-indexer-observer-test-") as temporary:
            with self.assertRaises(ValueError):
                self.run_builder(Path(temporary), "--emit-only")
        with self.assertRaises(ValueError):
            self.run_builder(ROOT / "build" / "forbidden-observer-fixture", "--emit-only")
        self.assertFalse((ROOT / "build" / "forbidden-observer-fixture").exists())

    def test_compile_failure_keeps_frozen_proof_and_never_claims_success(self):
        with tempfile.TemporaryDirectory(prefix="myds-indexer-observer-test-") as temporary:
            out = Path(temporary) / "probe"
            with mock.patch.object(builder.subprocess, "run", side_effect=subprocess.CalledProcessError(9, ["clang"])):
                with self.assertRaises(subprocess.CalledProcessError):
                    self.run_builder(out)
            identity = json.loads((out / "build-manifest.json").read_text())
            self.assertFalse(identity["compile_only_pass"])
            self.assertFalse(identity["gpu_work_submitted"])
            self.assertFalse(identity["binary_executed"])
            self.assertIn("build_error", identity)
            self.assertTrue((out / "source-transform.json").exists())
            self.assertEqual((ROOT / builder.ORIGINAL).read_bytes(), self.original)


if __name__ == "__main__":
    unittest.main()
