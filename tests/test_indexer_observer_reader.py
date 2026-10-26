#!/usr/bin/env python3
"""CPU counterexamples for actual-PSO labels and literal-observer provenance."""
import difflib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import review_indexer_observer as reader
import build_indexer_observer_probe as producer


def timing_fixture(sham):
    records = []
    for layer in (2, 8, 14):
        for control in (0, 1):
            for sample in range(10):
                for arm in ("A1", "B", "A2"):
                    logical = 2 if arm == "B" else control
                    actual = 1 if sham and control == 1 and arm == "B" else logical
                    us = 1000.0
                    if arm == "B":
                        us = 1000.0 if sham and control == 1 else 600.0 if control == 0 else 880.0
                    if arm == "A2" and not (sham and control == 1):
                        us = 980.0
                    begin = 100 + len(records) * 0.01
                    end = begin + us * 1e-6
                    records.append(dict(layer=layer, control=control, sample=sample, arm=arm, provider=logical,
                        actual_provider_index=actual, tiled_b_sham=sham and control == 1 and arm == "B",
                        warmup=sample == 0, gpu_start_s=begin, gpu_end_s=end, gpu_us=(end - begin) * 1e6))
    return records


def build_fixture(directory):
    directory = Path(directory)
    original = (ROOT / "tools/bench_indexer_k_capture.m").read_bytes()
    generated, proof = producer.generate(original)
    payloads = {"tools/bench_indexer_k_capture.m": original, "src/reference-unit.c": b"int fixture(void){return 1;}\n"}
    inputs = {}
    for name, payload in payloads.items():
        path = directory / "source" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        inputs[name] = dict(bytes=len(payload), sha256=reader.core.sha(payload))
    (directory / "source/tools/bench_indexer_observer_probe.m").write_bytes(generated)
    proof_bytes = (json.dumps(proof, indent=2) + "\n").encode()
    (directory / "source-transform.json").write_bytes(proof_bytes)
    difference = "".join(difflib.unified_diff(original.decode().splitlines(keepends=True), generated.decode().splitlines(keepends=True),
        fromfile="tools/bench_indexer_k_capture.m", tofile="tools/bench_indexer_observer_probe.m")).encode()
    (directory / "observer-source.diff").write_bytes(difference)
    commands = []
    command_bytes = (json.dumps(commands, indent=2) + "\n").encode()
    (directory / "build-commands.json").write_bytes(command_bytes)
    binary = directory / "bench_indexer_observer_probe"
    binary.write_bytes(b"CPU fixture executable identity; never executed")
    manifest = dict(kind="indexer_literal_observer_diagnostic_build", compile_only_pass=True,
        diagnostic_only=True, adoption_cohort=False, fullmodel_eligible=False, production_adoption=False,
        runtime_registered=False, gpu_work_submitted=False, binary_executed=False,
        original_shared_output_observer_preserved=True, additional_cpu_timestamps_or_output_snapshots=False,
        inputs=inputs, original_replay_sha256=reader.core.sha(original), generated_source_sha256=reader.core.sha(generated),
        transformation_proof_sha256=reader.core.sha(proof_bytes), inverse_restores_original_byte_for_byte=True,
        inverse_restored_sha256=reader.core.sha(original), source_diff_sha256=reader.core.sha(difference),
        commands=commands, commands_sha256=reader.core.sha(command_bytes), binary=str(binary), binary_sha256=reader.core.file_sha(binary))
    path = directory / "build-manifest.json"
    path.write_text(json.dumps(manifest))
    return path, dict(inputs=inputs), manifest


class ObserverReaderTests(unittest.TestCase):
    def test_sham_is_only_tiled_control_middle_not_actual_exact_group(self):
        annotated, groups = reader.annotate_and_summarize(timing_fixture(True), True)
        for row in annotated:
            expected = 1 if row["control"] == 1 else row["provider"]
            self.assertEqual(row["actual_scorer"], expected)
        for group in groups:
            if group["control"] == 1:
                self.assertEqual(group["comparison_kind"], "same_TILED_PSO_sham_not_candidate_gain")
                self.assertAlmostEqual(group["middle_vs_min_control_difference_percent"], 0)
            else:
                self.assertEqual(group["actual_scorers_by_arm"]["B"], [2])
            self.assertFalse(group["fullmodel_eligible"])

    def test_real_and_sham_actual_predecessors_are_distinct_from_output_labels(self):
        real, _ = reader.annotate_and_summarize(timing_fixture(False), False)
        sham, _ = reader.annotate_and_summarize(timing_fixture(True), True)
        a2 = next(row for row in real if row["layer"] == 2 and row["control"] == 1 and row["sample"] == 1 and row["arm"] == "A2")
        b2 = next(row for row in sham if row["layer"] == 2 and row["control"] == 1 and row["sample"] == 1 and row["arm"] == "A2")
        self.assertEqual(a2["previous"]["logical_output_provider"], 2)
        self.assertEqual(b2["previous"]["logical_output_provider"], 2)
        self.assertEqual(a2["previous"]["actual_scorer"], 2)
        self.assertEqual(b2["previous"]["actual_scorer"], 1)

    def test_false_actual_pso_or_sham_flag_is_rejected(self):
        for control in (0, 1):
            with self.subTest(control=control):
                rows = timing_fixture(True)
                bad = next(row for row in rows if row["control"] == control and row["arm"] == "B")
                bad["actual_provider_index"] = 1 if control == 0 else 2
                with self.assertRaises(reader.core.EvidenceError):
                    reader.annotate_and_summarize(rows, True)
        rows = timing_fixture(False)
        next(row for row in rows if row["arm"] == "B")["tiled_b_sham"] = True
        with self.assertRaises(reader.core.EvidenceError):
            reader.annotate_and_summarize(rows, False)

    def test_no_sample_replacement_and_timestamp_arithmetic(self):
        rows = timing_fixture(False)
        rows[5] = dict(rows[4])
        with self.assertRaises(reader.core.EvidenceError):
            reader.annotate_and_summarize(rows, False)
        rows = timing_fixture(False)
        rows[7]["gpu_us"] += 1
        with self.assertRaises(reader.core.EvidenceError):
            reader.annotate_and_summarize(rows, False)

    def test_per_run_layer_paired_signs_and_gaps_are_retained(self):
        annotated, groups = reader.annotate_and_summarize(timing_fixture(False), False)
        self.assertEqual(len(annotated), 180)
        self.assertEqual(len(groups), 6)
        for group in groups:
            self.assertEqual(len(group["paired_samples"]), 9)
            self.assertEqual(group["paired_a2_faster_count"], 9)
            self.assertLess(group["original_ratio_of_medians_a2_vs_a1_percent"], 0)
            self.assertEqual(group["per_arm_gap_before_us"]["A1"]["n"], 9)

    def test_fixed_inverse_proof_and_common_compiler_source_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path, reference, manifest = build_fixture(directory)
            reader.verify_observer_build(path, reference, Path)
            generated = Path(directory) / "source/tools/bench_indexer_observer_probe.m"
            generated.write_bytes(generated.read_bytes() + b"\n// outside the six allowed edits\n")
            with self.assertRaises(reader.core.EvidenceError):
                reader.verify_observer_build(path, reference, Path)
        with tempfile.TemporaryDirectory() as directory:
            path, reference, manifest = build_fixture(directory)
            unit = Path(directory) / "source/src/reference-unit.c"
            unit.write_bytes(b"int fixture(void){return 2;}\n")
            manifest["inputs"] = {key: dict(value) for key, value in manifest["inputs"].items()}
            manifest["inputs"]["src/reference-unit.c"] = dict(bytes=unit.stat().st_size, sha256=reader.core.file_sha(unit))
            path.write_text(json.dumps(manifest))
            with self.assertRaises(reader.core.EvidenceError):
                reader.verify_observer_build(path, reference, Path)


if __name__ == "__main__":
    unittest.main()
