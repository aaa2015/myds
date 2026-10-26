#!/usr/bin/env python3
"""CPU-only lossless-generation and scope fixtures for matched replay."""
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
SPEC = importlib.util.spec_from_file_location("indexer_matched_builder", ROOT / "tools/build_indexer_matched_replay.py")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class MatchedReplayBuilderTests(unittest.TestCase):
    def setUp(self):
        self.original = (ROOT / builder.ORIGINAL).read_bytes()
        self.observer = builder.load_observer(ROOT)

    def test_complete_inverse_helpers_phase1_and_full_observer_preserved(self):
        generated, proof = builder.generate(self.original, self.observer)
        builder.verify(self.original, generated, proof, self.observer)
        self.assertEqual(self.observer.inverse(generated, proof["exact_edits"]), self.original)
        self.assertEqual((proof["observer_base_edit_count"], proof["matched_edit_count"], proof["edit_count"]), (6, 5, 11))
        main = b"int main(int argc,char **argv){@autoreleasepool {"
        self.assertEqual(self.original.split(main)[0], generated.split(main)[0])
        begin = b"    // Phase 1: every layer's complete scores and actual ordered Top512 consumer"
        end = b"    real_timing_phase=YES;"
        self.assertEqual(self.original.split(begin)[1].split(end)[0], generated.split(begin)[1].split(end)[0])
        validation = b"                if(!real_index_validate(f,provider,NO)) {"
        timestamps = b"                double us="
        self.assertEqual(self.original.split(validation)[1].split(timestamps)[0],
                         generated.split(validation)[1].split(timestamps)[0])
        self.assertIn(b'@"full_scores_sha256":sha(body(f->out[provider]),f->captured.length)', generated)
        self.assertNotIn(b"host_snapshot", generated)
        self.assertNotIn(b"stability_run", generated)

    def test_generated_loop_executes_cond_then_arms_and_excludes_conditioner(self):
        generated, _ = builder.generate(self.original, self.observer)
        self.assertIn(b"for(unsigned step=0;step<4;step++)", generated)
        self.assertIn(b"const BOOL is_conditioner=step==0u;", generated)
        self.assertIn(b"const unsigned arm=is_conditioner?1u:step-1u;", generated)
        self.assertIn(b"if(sample&&!is_conditioner)times[arm][sample-1]=us;", generated)
        self.assertNotIn(b"if(sample)times[arm][sample-1]=us;", generated)
        self.assertIn(b'@"arm":is_conditioner?@"conditioner":@[@"A1",@"B",@"A2"][arm]', generated)
        self.assertIn(b'@"performance_sample":@(sample>0u&&!is_conditioner)', generated)
        self.assertIn(b"real_index_run(queue,f,provider,psos[ci][actual_provider_index])", generated)
        # The constructor must pass logical provider through to the unchanged
        # encoder; substituting actual_provider_index would silently use out1.
        self.assertNotIn(b"real_index_run(queue,f,actual_provider_index,", generated)

    def test_all_formal_controls_and_B_have_two_dispatch_output_reuse(self):
        for control in (0, 1):
            providers = [p for _ in range(10) for p in (2, control, 2, control)]
            for sample in range(1, 10):
                for step in (1, 2, 3):
                    index = sample * 4 + step
                    prior = max(i for i in range(index) if providers[i] == providers[index])
                    self.assertEqual(index - prior, 2)
                self.assertEqual(providers[sample*4], providers[sample*4+2])
                self.assertEqual(providers[sample*4+1-1], 2)
                self.assertEqual(providers[sample*4+3-1], 2)
        self.assertEqual(3 * 2 * 10 * 4, 240)
        self.assertEqual(3 * 2 * 10, 60)
        self.assertEqual(3 * 2 * 1 * 3, 18)
        self.assertEqual(3 * 2 * 9 * 3, 162)

    def test_missing_duplicate_anchor_and_unlisted_observer_edit_refused(self):
        for bad in (self.original.replace(b"for(unsigned arm=0;arm<3;arm++)", b"for(unsigned arm=0;arm<2;arm++)"),
                    self.original + builder.MATCHED_EDITS[1][1]):
            with self.assertRaises(ValueError):
                builder.generate(bad, self.observer)
        generated, proof = builder.generate(self.original, self.observer)
        with self.assertRaises(ValueError):
            builder.verify(self.original, generated.replace(b"if(sample&&!is_conditioner)", b"if(sample)"), proof, self.observer)
        changed = copy.deepcopy(proof)
        changed["matched_edit_count"] = 4
        with self.assertRaises(ValueError):
            builder.verify(self.original, generated, changed, self.observer)

    def run_builder(self, out, *extra):
        argv = ["builder", "--source-root", str(ROOT), "--out-dir", str(out), *extra]
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            builder.main()

    def test_emit_only_freezes_whole_seed_both_builders_and_exact_commands_without_execution(self):
        with tempfile.TemporaryDirectory(prefix="myds-indexer-matched-test-") as temporary:
            out = Path(temporary) / "probe"
            with mock.patch.object(builder.subprocess, "run", side_effect=AssertionError("no execution")) as run:
                self.run_builder(out, "--emit-only")
            run.assert_not_called()
            identity = json.loads((out / "build-manifest.json").read_text())
            seed, generated = (Path(identity[key]).read_bytes() for key in ("original_seed", "generated_source"))
            proof = json.loads((out / "source-transform.json").read_text())
            frozen_observer = builder.load_observer(out / "source")
            builder.verify(seed, generated, proof, frozen_observer)
            self.assertEqual(seed, self.original)
            for rel, record in identity["inputs"].items():
                data = (out / "source" / rel).read_bytes()
                self.assertEqual(record, dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
            self.assertIn(builder.OBSERVER_BUILDER, identity["inputs"])
            self.assertIn("tools/build_indexer_matched_replay.py", identity["inputs"])
            self.assertIn("tests/test_indexer_matched_builder.py", identity["inputs"])
            for filename, key in (("matched-source.diff", "source_diff_sha256"),
                                  ("source-transform.json", "transformation_proof_sha256"),
                                  ("build-commands.json", "commands_sha256")):
                self.assertEqual(hashlib.sha256((out / filename).read_bytes()).hexdigest(), identity[key])
            self.assertEqual(len(identity["commands"]), 7)
            self.assertTrue(all(command[:2] == ["xcrun", "clang"] for command in identity["commands"]))
            self.assertEqual(identity["timing_records_expected"], 240)
            for key in ("compile_only_pass", "stage_gates_authorized_by_builder", "gpu_work_submitted",
                        "binary_executed", "production_adoption", "runtime_registered"):
                self.assertFalse(identity[key], key)
            self.assertFalse((out / "bench_indexer_matched_replay").exists())
            self.assertEqual((ROOT / builder.ORIGINAL).read_bytes(), self.original)

    def test_existing_or_checkout_output_and_compile_failure_refused(self):
        with tempfile.TemporaryDirectory(prefix="myds-indexer-matched-test-") as temporary:
            with self.assertRaises(ValueError):
                self.run_builder(Path(temporary), "--emit-only")
            out = Path(temporary) / "failed"
            with mock.patch.object(builder.subprocess, "run", side_effect=subprocess.CalledProcessError(9, ["clang"])):
                with self.assertRaises(subprocess.CalledProcessError):
                    self.run_builder(out)
            identity = json.loads((out / "build-manifest.json").read_text())
            self.assertFalse(identity["compile_only_pass"])
            self.assertFalse(identity["gpu_work_submitted"])
            self.assertIn("build_error", identity)
        forbidden = ROOT / "build" / "forbidden-matched-fixture"
        with self.assertRaises(ValueError):
            self.run_builder(forbidden, "--emit-only")
        self.assertFalse(forbidden.exists())


if __name__ == "__main__":
    unittest.main()
