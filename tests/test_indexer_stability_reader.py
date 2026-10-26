#!/usr/bin/env python3
"""CPU counterexamples for the schema3 diagnostic decision and byte reader."""
import argparse
from pathlib import Path
import struct
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import review_indexer_stability as reader

RULE = dict(sham_stability_max_abs_median_slot_drift_percent=0.5,
            balanced_inline_max_abs_median_slot_drift_percent=0.5,
            legacy_reproduction_min_A2_faster_percent_inclusive=0.5,
            within_round_conditioning_reduction_min_median_percentage_points_inclusive=0.5,
            within_round_conditioning_reduction_positive_pairs_min=10)


def fixture(change=None):
    rows = []
    for round_index, order, pi, step in reader.expected_schedule():
        role = ["conditioner", "A1", "B", "A2"][step]
        us = 1000.0
        if pi >= 3 and role == "B":
            us = 880.0
        if pi == 3 and role == "A2":
            us = 980.0
        if pi >= 5 and role == "A2":
            us = 1500.0  # Descriptive phases cannot veto or rescue a decision.
        row = dict(round=round_index, phase_order=order, phase_index=pi, role=role, gpu_us=us)
        if change:
            change(row)
        rows.append(row)
    return rows


class StabilityDecisionTests(unittest.TestCase):
    def test_finite_forward_reverse_schedule_and_conditioner_roles(self):
        rows = reader.expected_schedule()
        self.assertEqual(len(rows), 468)
        self.assertEqual([pi for r, o, pi, s in rows if r == 0 and s == 0], list(range(9)))
        self.assertEqual([pi for r, o, pi, s in rows if r == 1 and s == 0], list(reversed(range(9))))
        self.assertEqual([pi for r, o, pi, s in rows if r == 2 and s == 0], list(range(9)))
        for pi in range(9):
            positions = [o for r, o, p, s in rows if r > 0 and p == pi and s == 0]
            self.assertEqual(positions.count(pi), 12 if pi == 4 else 6)

    def test_five_frozen_phases_pass_without_using_four_descriptive_phases(self):
        summaries, decision = reader.summary_and_decision(fixture(), RULE)
        self.assertTrue(decision["measurement_correction_supported"])
        self.assertEqual(decision["strictly_positive_pairs"], 12)
        self.assertTrue(all(row["descriptive_only"] for row in summaries[5:]))
        self.assertFalse(decision["formal_component_adoption_eligible"])
        self.assertFalse(decision["fullmodel_eligible"])

    def test_sham_b_slot_is_checked_even_if_a1_a2_are_equal(self):
        def change(r):
            if r["phase_index"] == 2 and r["role"] == "B":
                r["gpu_us"] = 1010.0
        _, decision = reader.summary_and_decision(fixture(change), RULE)
        self.assertFalse(decision["sham_stability"]["sham_candidate_inline"])
        self.assertFalse(decision["measurement_correction_supported"])

    def test_legacy_nonreproduction_blocks_next_stage_despite_healthy_shams(self):
        def change(r):
            if r["phase_index"] == 3 and r["role"] == "A2":
                r["gpu_us"] = 1000.0
        _, decision = reader.summary_and_decision(fixture(change), RULE)
        self.assertTrue(decision["all_three_shams_passed"])
        self.assertFalse(decision["legacy_reproduction_passed"])
        self.assertFalse(decision["measurement_correction_supported"])

    def test_nine_positive_pairs_do_not_pass_ten_of_twelve_rule(self):
        def change(r):
            if r["round"] >= 10 and r["role"] == "A2" and r["phase_index"] in (3, 4):
                r["gpu_us"] = 1000.0 if r["phase_index"] == 3 else 980.0
        _, decision = reader.summary_and_decision(fixture(change), RULE)
        self.assertEqual(decision["strictly_positive_pairs"], 9)
        self.assertGreater(decision["median_reduction_percentage_points"], 0.5)
        self.assertTrue(decision["legacy_reproduction_passed"])
        self.assertTrue(decision["balanced_inline_stability_passed"])
        self.assertFalse(decision["conditioning_reduction_passed"])

    def test_paired_reduction_uses_absolute_drift(self):
        def change(r):
            if r["role"] == "A2" and r["phase_index"] in (3, 4):
                r["gpu_us"] = 990.0 if r["phase_index"] == 3 else 1004.0
        _, decision = reader.summary_and_decision(fixture(change), RULE)
        self.assertAlmostEqual(decision["median_reduction_percentage_points"], 0.6)
        self.assertTrue(decision["measurement_correction_supported"])

    def test_raw_bits_poison_and_guards_are_not_replaced_by_declared_hashes(self):
        review = reader.StabilityReview(argparse.Namespace(path_map=[], capture_root=None))
        original = struct.pack("<2f", 2.0, 1.0)
        review.case = dict(rows=1, keys=2, pos=100, ratio=1, payloads={"scores-f32": original})
        review.tiled = original
        healthy = review.facts(original, reader.core.GUARD * 2)
        self.assertEqual(healthy["tiled"]["mismatches"], 0)
        poison = review.facts(struct.pack("<2I", 0xA5A5A5A5, 0x3F800000), reader.core.GUARD * 2)
        self.assertFalse(poison["stats"]["all_written"])
        corrupt_guard = review.facts(original, b"\0" + (reader.core.GUARD * 2)[1:])
        self.assertFalse(corrupt_guard["guards"])
        self.assertEqual(corrupt_guard["tiled"]["mismatches"], 0)


if __name__ == "__main__":
    unittest.main()
