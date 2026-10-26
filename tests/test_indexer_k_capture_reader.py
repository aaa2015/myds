#!/usr/bin/env python3
"""CPU-only failure-classification and immutable-gate checks for the raw reader."""
import argparse
import importlib.util
from pathlib import Path
import statistics
import struct
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("real_indexer_reader", ROOT / "tools/review_indexer_k_capture.py")
reader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reader)


class Fixture:
    def __init__(self, root, fault=None, timing=False):
        self.root = Path(root)
        self.review = v = reader.Review(argparse.Namespace(path_map=[], capture_root=None))
        v.protocol = {"performance": {"max_abs_control_drift_percent": 2.0,
                                     "min_conservative_time_saved_percent_strict": 2.0}}
        v.cases, v.records = {}, []
        v.manifest = {"status": "passed_component_and_topk_only" if timing else "failed",
                      "all_layers_numerical_gate_passed": fault is None}
        v.window = {"exit_code": 0 if timing else 1}
        v.window_gate = {"passed": True}
        self.input_rows = []
        counters = dict(prefix_requests=1, prefix_encoded=1, prefix_unavailable=0,
                        nan_fuse_encoded=1, nan_shuffle_encoded=0)
        capture = struct.pack("<1024f", *([512.0 - k for k in range(512)] * 2))
        ids = struct.pack("<1024i", *(list(range(512)) * 2))
        for layer in reader.LAYERS:
            metadata = {"cache_bf16": True, "provider": "kernel_myds_indexer_score_chunk_exact",
                        "invisible_minus_inf_count": 0, "actual_topk_consumer": counters}
            payloads = {"scores-f32": capture, "selected-i32": ids,
                        "q-f32": b"q123", "weights-f32": b"w123", "cache": b"k123"}
            v.cases[layer] = dict(metadata=metadata, payloads=payloads, rows=2, keys=512, pos=30720, ratio=2)
            for provider in range(3):
                scores = capture
                if fault == "candidate_bits" and layer == 8 and provider == 2 or fault == "actual_bits" and layer == 8 and provider == 0:
                    changed = bytearray(scores)
                    changed[:4] = struct.pack("<I", struct.unpack_from("<I", changed)[0] + 1)
                    scores = bytes(changed)
                guard = reader.GUARD * 2
                if fault == "candidate_guard" and layer == 8 and provider == 2:
                    guard = b"\0" + guard[1:]
                prefix = f"layer-{layer}-provider-{provider}"
                self.put(prefix + "-scores.f32", scores)
                self.put(prefix + "-guards.bin", guard)
                bad = 1 if scores != capture else 0
                ok = guard == reader.GUARD * 2 and (provider != 0 or bad == 0) and (provider != 2 or bad == 0)
                v.records.append(dict(kind="real_parity", layer=layer, provider=provider, tokens=2, keys=512, bf16=True,
                    captured_actual_provider=metadata["provider"], mismatches_to_actual_capture=bad,
                    candidate_mismatches_to_tiled=bad if provider == 2 else None, invisible_minus_inf_count=0,
                    full_scores=True, guards_intact=guard == reader.GUARD * 2, all_written=True, visible_finite=True,
                    causal_tail_valid=True, passed=ok, topk_mask_validated=False, fullmodel_speedup_proven=False))
            for provider in range(3):
                prefix = f"layer-{layer}-provider-{provider}"
                self.put(prefix + "-selected.i32", ids)
                self.put(prefix + "-topk-guards.bin", reader.GUARD * 4)
                v.records.append(dict(kind="real_topk", layer=layer, provider=provider, top=512, rows=2,
                    ordered_ids_complete=True, actual_capture_mismatches=0, candidate_mismatches_to_tiled=0 if provider == 2 else None,
                    host_encoded=1, GPU_completed=1, guards_intact=True, score_input_unchanged=True,
                    same_consumer_counter_path_as_capture=True, **counters, passed=True,
                    actual_mask_filter_for_these_layers=False, fullmodel_speedup_proven=False))
            self.inputs(layer, "phase1_post_topk", fault)
        v.records.append(dict(kind="all_layers_numerical_gate", passed=fault is None, layers=list(reader.LAYERS),
                              complete_scores=True, ordered_top512_all_providers_equal_capture=fault is None, timing_started=False))
        if timing:
            self.timings()

    def put(self, name, data):
        path = self.root / name
        path.write_bytes(data)
        self.review.raw[name] = path

    def inputs(self, layer, phase, fault=None):
        v = self.review
        for name in reader.INPUTS:
            stem = f"layer-{layer}-{phase}-{name}"
            data = v.cases[layer]["payloads"][name]
            prefix, suffix = reader.GUARD, reader.GUARD
            if fault == "input_guard" and layer == 8 and name == "cache":
                suffix = suffix[:-1] + b"\0"
            self.put(stem + "-post.bin", data)
            self.put(stem + "-guard-prefix.bin", prefix)
            self.put(stem + "-guard-suffix.bin", suffix)
            intact = prefix == suffix == reader.GUARD
            v.records.append(dict(kind="real_inputs", layer=layer, phase=phase, input=name, bytes=len(data), available=True,
                capture_sha256=reader.sha(data), post_sha256=reader.sha(data), guards_intact=intact, bytes_equal_capture=True,
                post_raw=stem + "-post.bin", guard_prefix_raw=stem + "-guard-prefix.bin", guard_suffix_raw=stem + "-guard-suffix.bin",
                guard_bytes_per_side=64, guard_prefix_sha256=reader.sha(prefix), guard_suffix_sha256=reader.sha(suffix), passed=intact))

    def timings(self):
        v = self.review
        for layer in reader.LAYERS:
            for control in (0, 1):
                values = {arm: [] for arm in ("A1", "B", "A2")}
                for sample in range(10):
                    for arm in ("A1", "B", "A2"):
                        duration = 0.0008 if arm == "B" else 0.001
                        if layer == 2 and control == 1 and arm == "A2":
                            duration = 0.001 * (1 - 0.020846074729885644)
                        begin = 100.0 + len(v.records) * 0.02
                        end = begin + duration
                        us = (end - begin) * 1e6
                        if sample:
                            values[arm].append(us)
                        v.records.append(dict(kind="real_timing", layer=layer, control=control, provider=2 if arm == "B" else control,
                            arm=arm, sample=sample, warmup=sample == 0, gpu_start_s=begin, gpu_end_s=end, gpu_us=us,
                            provider_full_scores_bit_exact_to_tiled=True, full_scores_sha256=reader.sha(v.cases[layer]["payloads"]["scores-f32"]),
                            candidate_full_scores_to_tiled_gate_passed=True, all_layers_numerical_gate_passed=True, fullmodel_speedup_proven=False))
                a1, b, a2 = [statistics.median(values[arm]) for arm in ("A1", "B", "A2")]
                v.records.append(dict(kind="real_summary", layer=layer, control=control, a1_us=a1, b_us=b, a2_us=a2,
                    conservative_time_saved_percent=(1 - b / min(a1, a2)) * 100, control_drift_percent=(a2 / a1 - 1) * 100,
                    ordered_top512_validated=True, fullmodel_speedup_proven=False))
            self.inputs(layer, "phase2_post_timing")

    def run(self):
        self.review.check_numerical()
        self.review.check_timing()
        return self.review.finish()


class RawReaderTests(unittest.TestCase):
    def test_all_correct_with_one_drift_failure_is_not_an_accepted_experiment(self):
        with tempfile.TemporaryDirectory() as root:
            report = Fixture(root, timing=True).run()
        self.assertTrue(report["review_completed"])
        self.assertTrue(report["all_layers_numerical_gate_passed"])
        self.assertEqual(report["timing_records"], 180)
        self.assertEqual(report["outcome"], "frozen_component_performance_gate_failed_correctness_passed")
        self.assertFalse(report["fullmodel_eligible"])
        failed = [group for group in report["performance_groups"] if not group["passed"]]
        self.assertEqual([(g["layer"], g["control"]) for g in failed], [(2, 1)])

    def test_candidate_numeric_failure_retains_structured_evidence(self):
        with tempfile.TemporaryDirectory() as root:
            report = Fixture(root, "candidate_bits").run()
        self.assertTrue(report["review_completed"])
        self.assertFalse(report["all_layers_numerical_gate_passed"])
        self.assertEqual(report["outcome"], "frozen_component_numerical_gate_failed")
        failed = [r for r in report["scores"] if not r["passed"]]
        self.assertEqual([(r["layer"], r["provider"]) for r in failed], [(8, 2)])
        self.assertEqual(failed[0]["candidate_mismatches_to_tiled"]["mismatches"], 1)
        self.assertEqual(report["timing_records"], 0)

    def test_actual_native_failure_does_not_blame_candidate(self):
        with tempfile.TemporaryDirectory() as root:
            report = Fixture(root, "actual_bits").run()
        failed = [r for r in report["scores"] if not r["passed"]]
        self.assertEqual([(r["category"], r["provider"]) for r in failed], [("actual_capture_replay", 0)])
        self.assertTrue(all(r["passed"] for r in report["scores"] if r["provider"] == 2))

    def test_guards_are_separate_from_arithmetic_differences(self):
        for fault, category in (("candidate_guard", "candidate_scores"), ("input_guard", "input_integrity")):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as root:
                report = Fixture(root, fault).run()
            self.assertEqual(report["outcome"], "frozen_component_numerical_gate_failed")
            self.assertTrue(any(r["category"] == category for r in report["failures"]))
            self.assertTrue(all(r["candidate_mismatches_to_tiled"]["mismatches"] == 0 for r in report["scores"] if r["provider"] == 2))

    def test_false_pass_boolean_is_evidence_contradiction(self):
        with tempfile.TemporaryDirectory() as root:
            fixture = Fixture(root, "candidate_bits")
            bad = next(r for r in fixture.review.records if r["kind"] == "real_parity" and r["layer"] == 8 and r["provider"] == 2)
            bad["passed"] = True
            with self.assertRaises(reader.EvidenceError):
                fixture.run()

    def test_raw_timestamps_override_summary_and_phase_order(self):
        with tempfile.TemporaryDirectory() as root:
            fixture = Fixture(root, timing=True)
            next(r for r in fixture.review.records if r["kind"] == "real_summary")["b_us"] += 1
            with self.assertRaises(reader.EvidenceError):
                fixture.run()
        with tempfile.TemporaryDirectory() as root:
            fixture = Fixture(root, timing=True)
            timing = next(r for r in fixture.review.records if r["kind"] == "real_timing")
            fixture.review.records.remove(timing)
            fixture.review.records.insert(0, timing)
            with self.assertRaises(reader.EvidenceError):
                fixture.run()

    def test_exact_frozen_thresholds_and_native_integer_boolean(self):
        gates = {"max_abs_control_drift_percent": 2.0, "min_conservative_time_saved_percent_strict": 2.0}
        self.assertEqual(reader.frozen_performance_gate(2.0, 2.0, gates), (True, False))
        self.assertEqual(reader.frozen_performance_gate(2.001, -2.0, gates), (True, True))
        self.assertEqual(reader.frozen_performance_gate(50.0, -2.00001, gates), (False, True))
        self.assertTrue(reader.harness_flag(1))
        with self.assertRaises(reader.EvidenceError):
            reader.harness_flag(2)


if __name__ == "__main__":
    unittest.main()
