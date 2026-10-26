#!/usr/bin/env python3
"""CPU counterexamples for conditioners, S calibration and prebound R gates."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import review_indexer_matched as reader

PROTOCOL = dict(calibration=dict(max_sham_span_percent_inclusive=0.5),
                formal=dict(max_abs_control_drift_percent_inclusive=2.0, strict_min_savings_percent=2.0))


def fixture(sham):
    rows = []
    for layer, control, sample, step in reader.matched_schedule():
        arm = ("conditioner", "A1", "B", "A2")[step]
        logical = 2 if step in (0, 2) else control
        actual = 1 if sham and control == 1 and step in (0, 2) else logical
        us = 10000.0 if step == 0 else 1000.0
        if step == 2:
            us = 1000.0 if sham and control == 1 else 600.0 if control == 0 else 880.0
        begin = 1 + len(rows) * .02
        end = begin + us * 1e-6
        rows.append(dict(layer=layer, control=control, sample=sample, step=step, arm=arm,
            provider=logical, actual_provider_index=actual, tiled_b_sham=sham and control == 1 and step in (0, 2),
            conditioner=step == 0, performance_sample=sample > 0 and step != 0, warmup=sample == 0,
            gpu_start_s=begin, gpu_end_s=end, gpu_us=(end - begin) * 1e6, full_scores_sha256="identical-complete-output"))
    return rows


class MatchedReaderTests(unittest.TestCase):
    def test_240_labels_and_conditioner_is_never_a_median_sample(self):
        rows, groups = reader.annotate_matched(fixture(True), True)
        self.assertEqual(len(rows), 240)
        self.assertEqual(sum(r["conditioner"] for r in rows), 60)
        self.assertEqual(sum(r["sample"] == 0 and not r["conditioner"] for r in rows), 18)
        self.assertEqual(sum(r["performance_sample"] for r in rows), 162)
        for group in groups:
            self.assertAlmostEqual(group["a1_us"], 1000.0)
            self.assertGreater(group["conditioner_us_descriptive_only"]["median"], 9000.0)
            self.assertTrue(group["conditioner_excluded_from_all_medians"])
            self.assertEqual(len(group["paired_samples"]), 9)

    def test_conditioner_and_middle_actual_pso_output_and_readback_must_match(self):
        for key, changed in (("provider", 1), ("actual_provider_index", 2), ("performance_sample", True),
                             ("full_scores_sha256", "different-output")):
            with self.subTest(key=key):
                rows = fixture(True)
                conditioner = next(r for r in rows if r["control"] == 1 and r["step"] == 0)
                conditioner[key] = changed
                with self.assertRaises(reader.core.EvidenceError):
                    reader.annotate_matched(rows, True)

    def test_s_exact_groups_are_not_null_and_one_bad_tiled_layer_stops(self):
        _, groups = reader.annotate_matched(fixture(True), True)
        self.assertGreater(next(g for g in groups if g["control"] == 0)["three_measured_slot_span_percent"], 10.0)
        decision = reader.stage_decision("calibration", groups, True, PROTOCOL)
        self.assertTrue(decision["calibration_passed"])
        self.assertTrue(decision["R_component_eligible"])
        self.assertFalse(decision["fullmodel_eligible"])
        bad = deepcopy(groups)
        next(g for g in bad if g["layer"] == 8 and g["control"] == 1)["three_measured_slot_span_percent"] = .500001
        self.assertFalse(reader.stage_decision("calibration", bad, True, PROTOCOL)["calibration_passed"])

    def test_formal_all_six_strict_savings_and_inclusive_drift_unchanged(self):
        _, groups = reader.annotate_matched(fixture(False), False)
        self.assertTrue(reader.stage_decision("formal", groups, True, PROTOCOL)["fullmodel_eligible"])
        for drift, saved, expected in ((2.0, 2.00001, True), (-2.0, 2.00001, True), (2.00001, 50.0, False), (0.0, 2.0, False)):
            bad = deepcopy(groups)
            bad[0]["signed_control_drift_percent"], bad[0]["conservative_savings_percent"] = drift, saved
            self.assertEqual(reader.stage_decision("formal", bad, True, PROTOCOL)["formal_component_passed"], expected)
        self.assertFalse(reader.stage_decision("formal", groups, False, PROTOCOL)["fullmodel_eligible"])
        duplicate = deepcopy(groups)
        duplicate[-1] = deepcopy(duplicate[0])
        self.assertFalse(reader.stage_decision("formal", duplicate, True, PROTOCOL)["formal_component_passed"])

    def test_formal_requires_prior_pass_report_hash_same_binary_source_reader_protocol(self):
        report = dict(review_completed=True, stage="calibration", calibration_passed=True,
            complete_correctness_passed=True, complete_run=True, R_component_eligible=True, fullmodel_eligible=False,
            matched_protocol_sha256="protocol", reader_sha256="reader", binary_sha256="binary",
            matched_build_manifest_sha256="build", generated_source_sha256="source", finished_epoch=10)
        plan = dict(build_manifest_sha256="build", calibration_report_sha256="report", started_epoch=11)
        build = dict(binary_sha256="binary", generated_source_sha256="source")
        reader.calibration_prerequisite(report, "report", plan, build, "protocol", "reader")
        for key, value in (("calibration_passed", False), ("binary_sha256", "other"), ("generated_source_sha256", "other"),
                           ("reader_sha256", "other"), ("matched_protocol_sha256", "other")):
            with self.subTest(key=key):
                bad = dict(report); bad[key] = value
                with self.assertRaises(reader.core.EvidenceError):
                    reader.calibration_prerequisite(bad, "report", plan, build, "protocol", "reader")
        with self.assertRaises(reader.core.EvidenceError):
            reader.calibration_prerequisite(report, "wrong-report", plan, build, "protocol", "reader")
        with self.assertRaises(reader.core.EvidenceError):
            reader.calibration_prerequisite(report, "report", dict(plan, started_epoch=9), build, "protocol", "reader")

    def test_bad_step_order_or_timestamp_is_not_a_replaceable_sample(self):
        for field, value in (("step", 2), ("gpu_us", -1)):
            rows = fixture(False)
            rows[0][field] = value
            with self.assertRaises(reader.core.EvidenceError):
                reader.annotate_matched(rows, False)

    def test_evidence_failure_retains_details_and_cannot_leak_prior_pass_flags(self):
        prior = dict(stage="formal", raw_files_verified=90, calibration_passed=True,
                     formal_component_passed=True, fullmodel_eligible=True, R_component_eligible=True)
        result = reader.evidence_failure(prior, "formal", reader.core.EvidenceError("bad retained raw SHA"))
        self.assertEqual(result["raw_files_verified"], 90)
        self.assertEqual(result["evidence_error"], "bad retained raw SHA")
        for key in ("review_completed", "calibration_passed", "formal_component_passed", "fullmodel_eligible", "R_component_eligible", "production_adoption"):
            self.assertIs(result[key], False)
        early = reader.evidence_failure(None, "calibration", ValueError("bad protocol"))
        self.assertEqual(early["stage"], "calibration")
        self.assertIs(early["calibration_passed"], False)


if __name__ == "__main__":
    unittest.main()
