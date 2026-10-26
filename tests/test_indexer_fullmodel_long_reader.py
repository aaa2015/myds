#!/usr/bin/env python3
"""CPU-only counterexamples for the separately frozen two-long-input stage."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import review_indexer_fullmodel_long as reader


class LongReaderTests(unittest.TestCase):
    def test_six_ordered_processes_twelve_generations_no_replacement(self):
        rows = reader.schedule()
        self.assertEqual(len(rows), 6)
        self.assertEqual([x["workload"] for x in rows], ["business96"] * 3 + ["code38404"] * 3)
        self.assertEqual(reader.checked_order(rows), rows)
        self.assertEqual(reader.checked_order(rows[:2]), rows[:2])
        for bad in ([], rows + [rows[0]], [rows[1], rows[0]] + rows[2:], rows[:3] + rows[:3]):
            with self.assertRaises(reader.EvidenceError): reader.checked_order(bad)

    def test_one_failed_or_missing_long_workload_prohibits_P(self):
        good = [dict(workload=name, qualification_passed=True) for name, _ in reader.WORKLOADS]
        result = reader.qualification_decision(good, True, True)
        self.assertTrue(result["qualification_passed"])
        self.assertTrue(result["long_workload_qualification_completed"])
        self.assertTrue(result["P_eligible"])
        self.assertFalse(result["performance_acceptance"])
        self.assertFalse(result["production_acceptance"])
        for rows, complete, native in ((good[:1], True, True), ([good[0], good[0]], True, True),
                ([good[0], dict(good[1], qualification_passed=False)], True, True), (good, False, True), (good, True, False)):
            with self.subTest(rows=rows, complete=complete, native=native):
                decision = reader.qualification_decision(rows, complete, native)
                self.assertFalse(decision["qualification_passed"])
                self.assertFalse(decision["P_eligible"])

    def test_Q0_report_must_be_true_frozen_identity_and_only_Q0_scope(self):
        descriptor = dict(path="/prior.json", bytes=20, sha256="prior")
        spec = dict(report=descriptor, protocol_sha256="protocol", reader_sha256=reader.CORE_SHA)
        report = dict(review_completed=True, qualification_passed=True, gold_library_bridge_passed=True,
            candidate_reference_passed=True, within_provider_repeatability_passed=True, protocol_sha256="protocol",
            reader_sha256=reader.CORE_SHA, binary_sha256="binary", long_workload_qualification_completed=False, production_acceptance=False)
        reader.prerequisite_gate(report, descriptor, spec, "binary")
        for key, value in (("qualification_passed", False), ("gold_library_bridge_passed", False), ("candidate_reference_passed", False),
                ("reader_sha256", "other"), ("protocol_sha256", "other"), ("binary_sha256", "other")):
            with self.subTest(key=key), self.assertRaises(reader.EvidenceError):
                reader.prerequisite_gate(dict(report, **{key: value}), descriptor, spec, "binary")
        with self.assertRaises(reader.EvidenceError): reader.prerequisite_gate(report, dict(descriptor, sha256="other"), spec, "binary")

    def test_new_stage_rejects_P_protocol_instead_of_misreading_runs2_3(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "protocol.json"
            path.write_text(json.dumps(dict(schema_version=1, kind="indexer_fullmodel_p_performance_protocol", stage="P")))
            review = reader.Review(SimpleNamespace(path_map=[], protocol=path))
            with self.assertRaises(reader.EvidenceError): review.initialize()

    def test_frozen_core_identity_is_explicit_and_original_Q0_not_mutated(self):
        self.assertEqual(reader.core.file_sha(reader.core.__file__), reader.CORE_SHA)
        original = Path(reader.core.__file__).with_name("review_indexer_fullmodel.py")
        self.assertEqual(reader.core.file_sha(original), reader.core.ORIGINAL_Q0_READER_SHA)

    def test_gold_library_or_candidate_byte_difference_prevents_long_eligibility(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "a", Path(d) / "b"; a.write_bytes(b"all exact"); b.write_bytes(b"one error")
            export = lambda path: dict(files={"final-logits.f32": path}, metadata=dict(rng_state="same"), finite_logits=True)
            for gold, off, on, field in ((a, b, b, "gold_library_bridge_passed"), (a, a, b, "candidate_reference_passed")):
                exports = {name: {0: export(path), 1: export(path)} for name, path in (("G_lib", gold), ("off", off), ("on", on))}
                failed = reader.core.qualification_comparisons(exports)
                self.assertFalse(failed[field])
                rows = [dict(workload="business96", **failed), dict(workload="code38404", qualification_passed=True)]
                self.assertFalse(reader.qualification_decision(rows, True, True)["P_eligible"])

    def test_evidence_failure_clears_all_long_eligibility_retains_failure_payload(self):
        report = dict(P_eligible=True, qualification_passed=True, long_workload_qualification_completed=True,
                      formal_performance_cohort_eligible=True, raw_logical_bytes=123)
        result = reader.evidence_failure(report, reader.EvidenceError("wrong child workload or prerequisite SHA"))
        self.assertEqual(result["raw_logical_bytes"], 123)
        for key in ("P_eligible", "qualification_passed", "long_workload_qualification_completed", "formal_performance_cohort_eligible", "production_acceptance"):
            self.assertIs(result[key], False)
        self.assertEqual(result["outcome"], "Qlong_evidence_invalid_P_prohibited")


if __name__ == "__main__":
    unittest.main()
