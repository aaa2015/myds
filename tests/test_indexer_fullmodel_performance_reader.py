#!/usr/bin/env python3
"""CPU-only P boundary/counterexample tests; no model, device or subprocess."""
from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import review_indexer_fullmodel_performance as reader
import test_indexer_fullmodel_reader as fixtures


def samples(arm):
    rows = []
    for run in range(4):
        primary = ("99999", "97", "97.5", "96") if arm == "B" else ("0.000000001", "100", "101", "99")
        rows.append(dict(run=run, warmup=run == 0, exact_native_metrics=dict(prefill_s=primary[run],
            ttft_s="1.005" if arm == "B" else "1", decode_ms_per_token="100.5" if arm == "B" else "100")))
    return rows


def raw_fixture(root):
    (root / "state").mkdir(); (root / "logits").mkdir()
    metadata = dict(kind="metadata", subject="prefill_indexer_k_resident", protocol_version=3, repeats=3, warmup_runs=1,
        max_new_tokens=2, max_seq_len=8192, temperature=0, top_p=1, top_k=0, min_p=0, repetition_penalty=1, seed=42,
        qos="user_interactive", qos_result=0, prompt_renderer="parse_chat_prompt", clock="CLOCK_MONOTONIC",
        transport="direct_byte_callback", decode_basis="first_to_last_callback/(tokens-1)", dspark_model=None)
    loaded = dict(kind="load", native_v41=True, layers=40, dspark_loaded=False, fresh_context_per_sample=True,
        cache_configuration_preserved=True, vocab_size=3)
    records = [dict(kind="indexer_metadata"), metadata, loaded]
    for run in range(4):
        _, sample, logits = fixtures.fixture(root / "state" / f"run-{run}", run)
        logits.rename(root / "logits" / f"run-{run}.f32")
        sample.update(kind="sample", warmup=run == 0, fresh_context=True, prompt_tokens=9, cached_tokens=0,
            prefill_tokens=8, decode_intervals=1, prefill_s=1.000000001, wall_s=2, ttft_s=1.1, first_callback_s=1.1,
            last_callback_s=1.2, decode_s=.1, decode_ms_per_token=100, callback_intervals_ms=[100],
            generation_ok=True, stop_reason="length", stop_reason_code=1, sample_valid=True, output_equal_to_warmup=True,
            gpu_indexer_provider=dict(fixtures.witness(1), run=run))
        records.append(sample)
    stdout = root / "stdout.jsonl"; stdout.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return stdout, records


class PerformanceReaderTests(unittest.TestCase):
    def test_primary_two_percent_is_strict_and_drift_one_percent_inclusive_exactly(self):
        self.assertFalse(reader.metric_gate("100.000000000", "98.000000000", "100.000000000", True)["passed"])
        self.assertTrue(reader.metric_gate("100", "97.999999999", "100", True)["passed"])
        for a2, expected in (("101", True), ("99", True), ("101.000000001", False), ("98.999999999", False)):
            with self.subTest(a2=a2):
                result = reader.metric_gate("100", "96", a2, True)
                self.assertEqual(result["passed"], expected)
        exact = reader.metric_gate("100", "96", "101", True)
        self.assertEqual(exact["exact_rationals"]["signed_control_drift_percent"], dict(numerator=1, denominator=1))

    def test_nonregression_half_percent_and_drift_one_percent_inclusive_without_epsilon(self):
        self.assertTrue(reader.metric_gate("100", "100.5", "101", False)["passed"])
        self.assertFalse(reader.metric_gate("100", "100.500000001", "101", False)["passed"])
        self.assertFalse(reader.metric_gate("100", "100", "101.000000001", False)["passed"])
        exact = reader.metric_gate("1.000000000", "1.005000000", "1.000000000", False)
        self.assertEqual(exact["exact_rationals"]["regression_percent"], dict(numerator=1, denominator=2))
        self.assertTrue(exact["passed"])
        with self.assertRaises(reader.EvidenceError): reader.metric_gate(100.0, "97", "100", True)

    def test_all_three_formals_and_warmup_exclusion_no_pooling(self):
        rows = {arm: samples(arm) for arm, _ in reader.PROVIDERS}
        result = reader.performance_summary(rows)
        self.assertTrue(result["passed"])
        self.assertEqual(result["metrics"]["prefill_s"]["a1_median"], 100)
        self.assertEqual(result["metrics"]["prefill_s"]["b_median"], 97)
        self.assertEqual(result["metrics"]["prefill_s"]["conservative_saved_percent"], 3)
        self.assertTrue(result["warmup_excluded_from_medians"])
        self.assertEqual(len(result["native_decimal_samples_including_warmup"]["B"]["prefill_s"]), 4)
        for mutate in ("missing3", "replaced2", "warmup3"):
            bad = deepcopy(rows)
            if mutate == "missing3": bad["B"].pop()
            elif mutate == "replaced2": bad["B"][2]["run"] = 1
            else: bad["B"][3]["warmup"] = True
            with self.subTest(mutate=mutate), self.assertRaises(reader.EvidenceError): reader.performance_summary(bad)

    def test_both_workloads_and_numerical_gate_required_before_any_perf_acceptance(self):
        groups = [dict(workload=name, passed=True) for name, _ in reader.WORKLOADS]
        good = reader.performance_decision(True, groups)
        self.assertTrue(good["native_fullmodel_gate_passed"])
        self.assertFalse(good["production_acceptance"])
        for numeric, rows in ((False, groups), (True, groups[:1]), (True, [groups[0], groups[0]]),
                              (True, [groups[0], dict(groups[1], passed=False)])):
            result = reader.performance_decision(numeric, rows)
            self.assertFalse(result["native_fullmodel_gate_passed"])
            self.assertFalse(result["performance_acceptance"])
            if not numeric: self.assertFalse(result["performance_evaluated"])

    def test_P_reads_all_four_exports_including_run3_without_fixed_Q0_helper(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); stdout, _ = raw_fixture(root)
            with patch.object(reader.core, "read_run_exports", side_effect=AssertionError("fixed Q0 helper forbidden for P")):
                result = reader.read_run_exports(root, stdout, 1, 9, 2)
            self.assertEqual(set(result["exports"]), {0, 1, 2, 3})
            self.assertEqual([r["warmup"] for r in result["samples"]], [True, False, False, False])
            self.assertEqual(result["samples"][3]["exact_native_metrics"]["prefill_s"], "1.000000001")

    def test_last_formal_counter_leak_or_missing_sample_cannot_be_ignored(self):
        for mutation in ("decode3", "missing3", "duplicate2"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as d:
                root = Path(d); stdout, records = raw_fixture(root)
                if mutation == "decode3":
                    counter = records[-1]["gpu_indexer_provider"]
                    for field in ("encoded", "bf16_dispatches"): counter[field] += 1; counter["decode_" + field] += 1
                elif mutation == "missing3": records.pop()
                else: records[-1]["run"] = 2
                stdout.write_text("\n".join(json.dumps(r) for r in records) + "\n")
                with self.assertRaises(reader.EvidenceError): reader.read_run_exports(root, stdout, 1, 9, 2)

    def test_all_twelve_raw_payload_sets_per_workload_include_last_B_formal(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "positive-zero", Path(d) / "negative-zero"
            a.write_bytes(struct.pack("<f", 0)); b.write_bytes(struct.pack("<f", -0.0))
            export = lambda path: dict(files={"final-logits.f32": path}, metadata={}, finite_logits=True)
            exports = {arm: {run: export(a) for run in range(4)} for arm, _ in reader.PROVIDERS}
            self.assertTrue(reader.cross_arm_comparisons(exports)["passed"])
            exports["B"][3] = export(b)
            result = reader.cross_arm_comparisons(exports)
            self.assertFalse(result["passed"])
            self.assertEqual(len(result["comparisons"]), 11)
            self.assertFalse(next(r for r in result["comparisons"] if (r["provider"], r["run"]) == ("B", 3))["passed"])

    def test_qualified_long_report_and_exact_inputs_source_reader_are_mandatory(self):
        protocol = dict(artifacts=dict(qualification_protocol=dict(sha256="ql-protocol"), private_build_manifest=dict(sha256="private"),
                                      target_build_manifest=dict(sha256="target")), workloads=["inputs"], model=dict(prefix_sha256="model"))
        spec = dict(report_path="/ql.json", report_sha_locked_in_execution_plan=True, protocol_sha256="ql-protocol", reader_sha256=reader.QUALIFICATION_READER_SHA)
        descriptor = dict(path="/ql.json", bytes=1, sha256="report")
        report = dict(review_completed=True, qualification_passed=True, long_workload_qualification_completed=True, P_eligible=True,
            formal_performance_cohort_eligible=True, execution_complete=True, native_generation_count=12, providers=[{}] * 6,
            protocol_sha256="ql-protocol", reader_sha256=reader.QUALIFICATION_READER_SHA, core_reader_sha256=reader.CORE_SHA,
            binary_sha256="binary", private_build_manifest_sha256="private", target_build_manifest_sha256="target",
            workload_identities=["inputs"], model_identity_scope=dict(prefix_sha256="model"), production_acceptance=False,
            workload_comparisons=[dict(workload=name, qualification_passed=True, gold_library_bridge_passed=True,
                candidate_reference_passed=True, within_provider_repeatability_passed=True) for name, _ in reader.WORKLOADS])
        reader.prerequisite_gate(report, descriptor, spec, protocol, "binary")
        for key, value in (("qualification_passed", False), ("long_workload_qualification_completed", False), ("protocol_sha256", "other"),
                           ("reader_sha256", "other"), ("workload_identities", ["other"]), ("binary_sha256", "other")):
            with self.subTest(key=key), self.assertRaises(reader.EvidenceError):
                reader.prerequisite_gate(dict(report, **{key: value}), descriptor, spec, protocol, "binary")
        bad = deepcopy(report); bad["workload_comparisons"][1]["candidate_reference_passed"] = False
        with self.assertRaises(reader.EvidenceError): reader.prerequisite_gate(bad, descriptor, spec, protocol, "binary")

    def test_evidence_failure_never_leaks_prior_speed_or_HTTP_eligibility(self):
        result = reader.evidence_failure(dict(native_fullmodel_gate_passed=True, native_prefill_speedup_proven=True,
            eligible_to_prepare_HTTP_validation=True, performance_acceptance=True, raw_logical_bytes=123), reader.EvidenceError("changed report SHA"))
        self.assertEqual(result["raw_logical_bytes"], 123)
        for key in ("review_completed", "native_fullmodel_gate_passed", "native_prefill_speedup_proven", "eligible_to_prepare_HTTP_validation",
                    "performance_acceptance", "production_acceptance"):
            self.assertIs(result[key], False)

    def test_protocol_gate_dictionary_cannot_contradict_frozen_artifact_formulas_or_limits(self):
        gate = dict(primary_saved_percent_formula="100*(1-B/min(A1,A2))", primary_min_saved_percent_strict=2.0,
                    max_abs_control_drift_percent_inclusive=1.0, max_nonregression_percent_inclusive=.5)
        reader.protocol_gate_consistency(gate, deepcopy(gate))
        for key, value in (("primary_saved_percent_formula", "100*(1-B/mean(A1,A2))"),
                           ("primary_min_saved_percent_strict", 1.0), ("max_abs_control_drift_percent_inclusive", 2.0)):
            with self.subTest(key=key), self.assertRaises(reader.EvidenceError):
                reader.protocol_gate_consistency(dict(gate, **{key: value}), gate)


if __name__ == "__main__":
    unittest.main()
