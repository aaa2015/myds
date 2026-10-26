#!/usr/bin/env python3
"""Independent CPU gate for separately frozen matched S and R stages.

S is a one-cohort measurement calibration, not a candidate speed result.
Formal R requires the passing S report SHA in its prelaunch plan, identical
source/binary/protocol, all numerical gates, and all six original 2% gates.
Every C/B actual scorer and logical output2 binding is checked. Conditioners
are retained but never enter arm medians. Production adoption is always false.
"""
import argparse
import difflib
import json
import math
from pathlib import Path
import re
import statistics

import review_indexer_k_capture as core
import review_indexer_observer as observer

require = core.require
GENERATED_SHA = "9e1ce21d6d5deca836898f8a0e21243e677e3ad7d01eb9af889eabd00f67269c"
OBSERVER_READER_SHA = "c61ddc99de5c1c53d21e8713b70e35a02fe8cc6a74cb1ec8524b1d123688678b"


def matched_schedule():
    return [(layer, control, sample, step) for layer in core.LAYERS for control in (0, 1)
            for sample in range(10) for step in range(4)]


def annotate_matched(records, sham):
    actual_order = [(r["layer"], r["control"], r["sample"], r["step"]) for r in records]
    expected = matched_schedule()
    require(len(records) <= 240 and actual_order == expected[:len(records)], "one bounded C/A1/B/A2 cohort without replacement")
    result = []
    for index, source in enumerate(records):
        r = dict(source)
        step, control, sample = r["step"], r["control"], r["sample"]
        role = ("conditioner", "A1", "B", "A2")[step]
        logical = 2 if step in (0, 2) else control
        actual = 1 if sham and control == 1 and step in (0, 2) else logical
        is_sham = sham and control == 1 and step in (0, 2)
        require(r["arm"] == role and r["provider"] == logical and r["actual_provider_index"] == actual and
                core.harness_flag(r["tiled_b_sham"]) == is_sham and core.harness_flag(r["conditioner"]) == (step == 0) and
                core.harness_flag(r["performance_sample"]) == (sample > 0 and step != 0) and
                core.harness_flag(r["warmup"]) == (sample == 0), "actual PSO/logical output2/conditioner and performance labels")
        begin, end = r["gpu_start_s"], r["gpu_end_s"]
        require(math.isfinite(begin) and math.isfinite(end) and end > begin > 0 and
                abs((end - begin) * 1e6 - r["gpu_us"]) < 1e-6, "GPU timestamp arithmetic")
        if index:
            require(begin >= result[-1]["gpu_end_s"], "serialized matched CBs")
        r.update(index=index, actual_scorer=actual, logical_output_slot=logical,
                 actual_scorer_name=("captured_actual_default", "complete_TU_TILED", "complete_TU_K_resident")[actual],
                 gap_before_us=(begin - result[-1]["gpu_end_s"]) * 1e6 if index else None,
                 previous=dict(actual_scorer=result[-1]["actual_scorer"], logical_output_slot=result[-1]["logical_output_slot"],
                               arm=result[-1]["arm"]) if index else None)
        result.append(r)
    groups = []
    for layer in core.LAYERS:
        for control in (0, 1):
            rows = [r for r in result if (r["layer"], r["control"]) == (layer, control)]
            if len(rows) != 40:
                continue
            for sample in range(10):
                quartet = [r for r in rows if r["sample"] == sample]
                c, a1, b, a2 = quartet
                require(c["actual_scorer"] == b["actual_scorer"] and c["logical_output_slot"] == b["logical_output_slot"] == 2 and
                        c["provider"] == b["provider"] == 2, "conditioner and B share actual PSO/args72/geometry/output2")
                require(a1["previous"]["actual_scorer"] == a2["previous"]["actual_scorer"] == b["actual_scorer"] and
                        a1["previous"]["logical_output_slot"] == a2["previous"]["logical_output_slot"] == 2,
                        "both measured controls follow the same scorer/output slot")
                if "full_scores_sha256" in c and "full_scores_sha256" in b:
                    require(c["full_scores_sha256"] == b["full_scores_sha256"], "same-sample C/B complete readback identity")
            arms = {arm: [r for r in rows if r["sample"] > 0 and r["arm"] == arm] for arm in ("A1", "B", "A2")}
            require(all(len(v) == 9 for v in arms.values()), "nine original measured samples per arm")
            a1, b, a2 = [statistics.median(r["gpu_us"] for r in arms[arm]) for arm in ("A1", "B", "A2")]
            paired = []
            for sample in range(1, 10):
                q = {r["arm"]: r for r in rows if r["sample"] == sample}
                av, bv, az = [q[arm]["gpu_us"] for arm in ("A1", "B", "A2")]
                paired.append(dict(sample=sample, conditioner_us=q["conditioner"]["gpu_us"], a1_us=av, b_us=bv, a2_us=az,
                    a2_vs_a1_percent=100 * (az / av - 1), middle_vs_min_control_difference_percent=100 * (1 - bv / min(av, az)),
                    gaps_before_us={role: q[role]["gap_before_us"] for role in ("conditioner", "A1", "B", "A2")}))
            groups.append(dict(layer=layer, control=control, a1_us=a1, b_us=b, a2_us=a2,
                calibration_null=sham and control == 1,
                comparison_kind="same_TILED_PSO_null" if sham and control == 1 else "different_scorer_comparison",
                signed_control_drift_percent=100 * (a2 / a1 - 1),
                conservative_savings_percent=100 * (1 - b / min(a1, a2)),
                three_measured_slot_span_percent=100 * (max(a1, b, a2) / min(a1, b, a2) - 1),
                conditioner_us_descriptive_only=observer.describe(r["gpu_us"] for r in rows if r["sample"] > 0 and r["conditioner"]),
                conditioner_excluded_from_all_medians=True, paired_samples=paired,
                paired_a2_faster_count=sum(r["a2_us"] < r["a1_us"] for r in paired),
                per_arm_gap_before_us={arm: observer.describe(r["gap_before_us"] for r in arms[arm]) for arm in arms}))
    return result, groups


def stage_decision(stage, groups, complete_numeric, protocol):
    complete = len(groups) == 6 and {(g["layer"], g["control"]) for g in groups} == {(layer, control) for layer in core.LAYERS for control in (0, 1)}
    if stage == "calibration":
        null = [g for g in groups if g["control"] == 1]
        threshold = protocol["calibration"]["max_sham_span_percent_inclusive"]
        gates = [dict(layer=g["layer"], control=1, three_slot_span_percent=g["three_measured_slot_span_percent"],
                      passed=g["calibration_null"] and g["three_measured_slot_span_percent"] <= threshold) for g in null]
        passed = complete_numeric and complete and len(gates) == 3 and all(g["passed"] for g in gates)
        return dict(calibration_passed=passed, calibration_null_gates=gates, R_component_eligible=passed,
                    formal_component_passed=False, fullmodel_eligible=False, production_adoption=False)
    max_drift = protocol["formal"]["max_abs_control_drift_percent_inclusive"]
    min_saved = protocol["formal"]["strict_min_savings_percent"]
    gates = [dict(layer=g["layer"], control=g["control"], signed_control_drift_percent=g["signed_control_drift_percent"],
                  conservative_savings_percent=g["conservative_savings_percent"],
                  drift_passed=abs(g["signed_control_drift_percent"]) <= max_drift,
                  strict_savings_passed=g["conservative_savings_percent"] > min_saved,
                  passed=abs(g["signed_control_drift_percent"]) <= max_drift and g["conservative_savings_percent"] > min_saved) for g in groups]
    passed = complete_numeric and complete and all(g["passed"] for g in gates)
    return dict(calibration_passed=None, formal_group_gates=gates, formal_component_passed=passed,
                fullmodel_eligible=passed, production_adoption=False)


def calibration_prerequisite(report, report_sha, plan, build, protocol_sha, reader_sha):
    require(report["review_completed"] is True and report["stage"] == "calibration" and report["calibration_passed"] is True and
            report["complete_correctness_passed"] is True and report["complete_run"] is True and report["R_component_eligible"] is True and
            report["fullmodel_eligible"] is False and report["matched_protocol_sha256"] == protocol_sha and report["reader_sha256"] == reader_sha and
            report["binary_sha256"] == build["binary_sha256"] and report["matched_build_manifest_sha256"] == plan["build_manifest_sha256"] and
            report["generated_source_sha256"] == build["generated_source_sha256"] and
            plan["calibration_report_sha256"] == report_sha, "passing independent S report prebound to formal plan, same executable/source/protocol/reader")
    require(plan["started_epoch"] >= report["finished_epoch"], "formal plan follows completed S calibration")


def verify_matched_build(path, reference, mapped):
    b = core.read_json(path)
    directory = Path(path).resolve().parent
    require(b["kind"] == "indexer_matched_conditioner_replay_build" and b["compile_only_pass"] is True and
            b["production_adoption"] is False and b["fullmodel_eligible_from_binary_completion"] is False and
            b["stage_gates_authorized_by_builder"] is False and b["runtime_registered"] is False and
            b["gpu_work_submitted"] is False and b["binary_executed"] is False and
            b["original_shared_output_observer_preserved"] is True and b["additional_cpu_timestamps_or_output_snapshots"] is False and
            b["timing_records_expected"] == 240 and b["conditioner_records_expected"] == 60 and
            b["warmup_arm_records_expected"] == 18 and b["formal_arm_records_expected"] == 162, "separately frozen matched replay build")
    for rel, item in b["inputs"].items():
        source = core.safe_child(directory / "source", rel)
        require(source.stat().st_size == item["bytes"] and core.file_sha(source) == item["sha256"], "frozen matched compiler input: " + rel)
    require(set(reference["inputs"]) <= set(b["inputs"]) and all(b["inputs"][k] == v for k, v in reference["inputs"].items()),
            "all original r2 compiler-visible sources retained exactly")
    original = (directory / "source/tools/bench_indexer_k_capture.m").read_bytes()
    generated = (directory / "source/tools/bench_indexer_matched_replay.m").read_bytes()
    proof_path = directory / "source-transform.json"
    proof = core.read_json(proof_path)
    require(core.sha(original) == b["original_replay_sha256"] == proof["original_replay_sha256"] == observer.ORIGINAL_SHA and
            core.sha(generated) == b["generated_source_sha256"] == proof["generated_source_sha256"] == GENERATED_SHA and
            core.file_sha(proof_path) == b["transformation_proof_sha256"] and proof["edit_count"] == 11 and
            proof["observer_base_edit_count"] == 6 and proof["matched_edit_count"] == 5, "fixed eleven minimal source edits")
    restored = generated
    for index, edit in enumerate(reversed(proof["exact_edits"])):
        before, after = edit["before"].encode(), edit["after"].encode()
        require(restored.count(after) == 1, "unique matched inverse source anchor")
        restored = restored.replace(after, before, 1)
        if index == 4:
            require(core.sha(restored) == proof["observer_base_generated_source_sha256"] == observer.GENERATED_SHA, "first five inverse edits restore unchanged literal observer probe")
    require(restored == original and core.sha(restored) == b["inverse_restored_sha256"] == proof["inverse_restored_sha256"], "complete original replay recovered without observer changes")
    difference = "".join(difflib.unified_diff(original.decode().splitlines(keepends=True), generated.decode().splitlines(keepends=True),
        fromfile="tools/bench_indexer_k_capture.m", tofile="tools/bench_indexer_matched_replay.m")).encode()
    require((directory / "matched-source.diff").read_bytes() == difference and core.sha(difference) == b["source_diff_sha256"], "exact retained matched source diff")
    command_bytes = (json.dumps(b["commands"], indent=2) + "\n").encode()
    require((directory / "build-commands.json").read_bytes() == command_bytes and core.sha(command_bytes) == b["commands_sha256"] and
            core.file_sha(mapped(b["binary"])) == b["binary_sha256"], "frozen matched executable and compile/link commands")
    return b


class MatchedRun(observer.ObserverRun):
    def __init__(self, args, protocol):
        super().__init__(args, protocol, 0 if args.stage == "calibration" else 1, "sham" if args.stage == "calibration" else "real")
        self.matched_protocol = protocol
        self.report.update(reader="independent_matched_stage_reader", reader_sha256=core.file_sha(__file__), stage=args.stage,
                           observer_cpu_dependency_sha256=core.file_sha(observer.__file__))

    def initialize_identity(self):
        require(core.file_sha(core.__file__) == observer.CORE_SHA and core.file_sha(observer.__file__) == OBSERVER_READER_SHA, "unchanged independent CPU dependencies")
        self.protocol = core.read_json(self.args.r2_protocol)
        self.capture = core.read_json(self.args.capture_verified)
        self.build = self.reference_build = core.read_json(self.args.r2_build_manifest)
        self.metal = core.read_json(self.args.metal_manifest)
        self.check_protocol(); self.check_build(); self.check_capture()
        self.report["r2_input_protocol_sha256"] = self.report.pop("protocol_sha256")
        self.report["r2_reference_build_sha256"] = self.report.pop("build_manifest_sha256")
        self.build = verify_matched_build(self.args.matched_build_manifest, self.reference_build, self.mapped)
        self.report.update(matched_build_manifest_sha256=core.file_sha(self.args.matched_build_manifest),
            build_manifest_sha256=core.file_sha(self.args.matched_build_manifest),
            matched_protocol_sha256=core.file_sha(self.args.matched_protocol), binary_sha256=self.build["binary_sha256"],
            generated_source_sha256=self.build["generated_source_sha256"], protocol_sha256=core.file_sha(self.args.matched_protocol))

    def load_window_and_raw(self):
        wd = self.args.window_dir
        plan, window, monitor = (core.read_json(wd / name) for name in ("execution-plan.json", "window.json", "monitor.json"))
        p = self.matched_protocol
        require(plan["stage"] == window["stage"] == self.args.stage and
                plan["protocol_sha256"] == window["protocol_sha256"] == core.file_sha(self.args.matched_protocol) and
                plan["reader_sha256"] == core.file_sha(__file__) and core.file_sha(self.args.window_wrapper) == plan["wrapper_sha256"] and
                plan["build_manifest_sha256"] == core.file_sha(self.args.matched_build_manifest) and plan["binary_sha256"] == self.build["binary_sha256"] and
                plan["capture_json_sha256"] == p["capture_json_sha256"] and plan["probe_sha256"] == p["probe_sha256"] and
                plan["native_sha256"] == p["native_sha256"] and plan["config_sha256"] == self.capture["config_sha256"], "prelaunch matched stage protocol/reader/build/input binding")
        require(plan["MYDS_inherited"] == {} and plan["exclusive_recognized_process_check"] is True, "clean inherited environment/exclusive stage")
        argv = plan["argv"]
        require(argv.count("--sham-tiled-b") == int(self.sham), "S calibration or real R mode")
        valued = [word for word in argv if word != "--sham-tiled-b"]
        require(len(valued) == 13 and valued[1::2] == ["--capture-json", "--metallib", "--native-metallib", "--config", "--out-dir", "--samples"], "unchanged valued operands")
        options = dict(zip(valued[1::2], valued[2::2]))
        require(core.file_sha(self.mapped(valued[0])) == self.build["binary_sha256"] and options["--samples"] == "9" and
                core.file_sha(self.mapped(options["--capture-json"])) == p["capture_json_sha256"] and
                core.file_sha(self.mapped(options["--metallib"])) == p["probe_sha256"] and
                core.file_sha(self.mapped(options["--native-metallib"])) == p["native_sha256"] and
                core.file_sha(self.mapped(options["--config"])) == self.capture["config_sha256"], "actual matched child argv artifact identities")
        self.result_dir = self.args.result_dir or wd / "outputs"
        require(self.mapped(options["--out-dir"]).resolve() == self.result_dir.resolve(), "actual matched output path")
        baseline = core.read_json(self.args.baseline_identity)
        before, after = window["canonical_before"], window["canonical_after"]
        require(before == after and window["canonical_unchanged"] is True and len(before) == 3 and
                {Path(k).name for k in before} == {"myds", "myds.metallib", "myds.conf"} and
                all(k in baseline and baseline[k]["sha256"] == v for k, v in before.items()), "canonical baseline/before/after identity")
        child = window["child_pid"]
        require(isinstance(child, int) and child > 0 and monitor and not window["recognized_before"] and not window["recognized_after"] and
                window["abort_reason"] is None and all(all(row["pid"] == child for row in entry["recognized"]) for entry in monitor) and
                any(any(row["pid"] == child and Path(row["comm"]).name == "bench_indexer_matched_replay" for row in entry["recognized"]) for entry in monitor), "fresh matched child and no recognized competitors")
        require(plan["started_epoch"] <= window["finished_epoch"] and 0 <= window["elapsed_s"] <= plan["timeout_s"] + 20, "bounded stage execution")
        self.plan, self.window = plan, window
        self.report.update(child_pid=child, started_epoch=plan["started_epoch"], finished_epoch=window["finished_epoch"],
            window_gates_passed=True, window_evidence_sha256={name: core.file_sha(wd / name) for name in ("execution-plan.json", "window.json", "monitor.json")})
        if self.args.stage == "formal":
            require(self.args.calibration_report is not None, "independent S report required before R")
            calibration = core.read_json(self.args.calibration_report)
            calibration_prerequisite(calibration, core.file_sha(self.args.calibration_report), plan, self.build,
                                     core.file_sha(self.args.matched_protocol), core.file_sha(__file__))
            require(child != calibration["child_pid"], "R is a fresh process after S")
            self.report["calibration_report_sha256"] = core.file_sha(self.args.calibration_report)
        m = self.manifest = core.read_json(self.result_dir / "artifact-manifest.json")
        require(m["probe"] == "indexer_matched_conditioner_replay" and m["matched_conditioner_per_sample"] is True and
                m["conditioner_output_slot"] == 2 and m["diagnostic_only"] is True and m["production_adoption"] is False and
                m["fullmodel_eligible"] is False and m["binary_sha256"] == self.build["binary_sha256"] and
                m["capture_json_sha256"] == p["capture_json_sha256"] and m["probe_sha256"] == p["probe_sha256"] and
                m["native_sha256"] == p["native_sha256"] and m["config_sha256"] == self.capture["config_sha256"] and
                core.harness_flag(m["tiled_b_sham_enabled"]) == self.sham and m["selected_mode"] == ("sham_tiled_b" if self.sham else "real_candidate") and
                m["samples"] == 9 and m["warmups"] == 1 and m["mode"] == "concurrent" and m["gpu_work_submitted"] is True and
                m["timing_record_count_expected"] == 240 and m["conditioner_record_count_expected"] == 60 and
                m["warmup_arm_record_count_expected"] == 18 and m["formal_arm_record_count_expected"] == 162, "matched runtime is evidence, never stage-gate authority")
        require(m["topk_loaded_library_sha256"] == p["native_sha256"] and
                self.mapped(m["topk_loaded_library_path"]).resolve() == self.mapped(options["--native-metallib"]).resolve() and
                core.file_sha(self.mapped(m["topk_loaded_library_path"])) == p["native_sha256"], "actual loaded native TopK library")
        for item in m["files"]:
            name = item["path"]
            require(name not in self.raw, "unique matched raw path")
            path = core.safe_child(self.result_dir, name, single=True)
            require(path.stat().st_size == item["bytes"] and core.file_sha(path) == item["sha256"], "full retained matched raw SHA/bytes")
            self.raw[name] = path
        self.records, info = [], []
        jsonl = self.args.jsonl or wd / "stdout.jsonl"
        for line in jsonl.read_text().splitlines():
            if not line.strip():
                continue
            if line.startswith("{"):
                self.records.append(json.loads(line, object_pairs_hook=core.unique_object))
            else:
                require(not self.records, "initial native log prefix only")
                match = re.fullmatch(r"\[\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d{3}\] \[INFO\] (.+)", line)
                require(match is not None, "known native INFO format")
                info.append(match.group(1))
        reference_info = [re.fullmatch(r"\[.+\] \[INFO\] (.+)", line).group(1) for line in self.args.reference_stdout.read_text().splitlines() if line and not line.startswith("{")]
        require(len(info) == len(reference_info) == 21 and info[2:] == reference_info[2:], "same frozen complete native initialization")
        config = re.fullmatch(r"config: 已加载 (.+) —— 生效 (\d+) 条", info[0])
        lib = re.fullmatch(r"myds_metal: loaded precompiled library from (.+) \(all required V4\.1 kernels verified\)", info[1])
        require(config and lib and int(config.group(2)) == self.capture["config_declared_unique_count"] and
                core.file_sha(self.mapped(config.group(1))) == self.capture["config_sha256"] and
                core.file_sha(self.mapped(lib.group(1))) == p["native_sha256"], "initialization paths and same effective config")
        require(all(r["kind"] in {"real_inputs", "real_parity", "real_topk", "all_layers_numerical_gate", "real_timing", "real_summary"} for r in self.records), "only original record kinds")
        self.report.update(raw_files_verified=len(self.raw), raw_logical_bytes=sum(path.stat().st_size for path in self.raw.values()),
            artifact_manifest_sha256=core.file_sha(self.result_dir / "artifact-manifest.json"), stdout_sha256=core.file_sha(jsonl))

    def check_diagnostic_timing(self):
        rows = [(index, r) for index, r in enumerate(self.records) if r["kind"] == "real_timing"]
        if rows:
            require(self.numerical_passed and self.phase1_gate_index is not None and all(index > self.phase1_gate_index for index, _ in rows), "all layers numerical/TopK before conditioner or timing")
        annotated, groups = annotate_matched([r for _, r in rows], self.sham)
        for r in annotated:
            reference = self.initial_scores[(r["layer"], r["provider"])]
            tiled = self.initial_scores[(r["layer"], 1)]
            require(r["full_scores_sha256"] == core.sha(reference) and r["provider_full_scores_bit_exact_to_tiled"] == (reference == tiled) and
                    r["candidate_full_scores_to_tiled_gate_passed"] is True and r["all_layers_numerical_gate_passed"] is True and
                    r["fullmodel_speedup_proven"] is False, "all conditioner and arm complete readbacks")
        summaries = [r for r in self.records if r["kind"] == "real_summary"]
        require(len(summaries) == len(groups), "one summary per complete group")
        for g in groups:
            matches = [r for r in summaries if (r["layer"], r["control"]) == (g["layer"], g["control"])]
            require(len(matches) == 1, "unique matched group summary")
            s = matches[0]
            for name, actual in (("a1_us", g["a1_us"]), ("b_us", g["b_us"]), ("a2_us", g["a2_us"]),
                    ("control_drift_percent", g["signed_control_drift_percent"]), ("conservative_time_saved_percent", g["conservative_savings_percent"])):
                require(abs(s[name] - actual) < 1e-6, "original arm-only median, conditioners never counted")
        conditioner_count = sum(r["conditioner"] for r in annotated)
        warm_arms = sum(r["sample"] == 0 and not r["conditioner"] for r in annotated)
        formal_arms = sum(r["performance_sample"] for r in annotated)
        complete = len(annotated) == 240 and len(groups) == 6 and conditioner_count == 60 and warm_arms == 18 and formal_arms == 162 and \
            all((layer, name, "phase2_post_timing") in self.input_seen for layer in core.LAYERS for name in core.INPUTS)
        numeric = self.all_numerical_passed and all(r["passed"] for r in self.input_reports)
        if self.manifest["status"] == "diagnostic_complete_not_adoption":
            require(complete and numeric and self.window["exit_code"] == 0, "stage execution complete with full input/numeric evidence")
        else:
            require(self.manifest["status"] == "failed" and self.window["exit_code"] != 0, "known numerical/execution stage failure")
        require(self.consumed == set(self.raw), "all manifest raw files consumed independently")
        decision = stage_decision(self.args.stage, groups, complete and numeric, self.matched_protocol)
        self.report.update(timing_records=len(annotated), conditioner_records=conditioner_count,
            warmup_arm_records=warm_arms, formal_arm_records=formal_arms, complete_run=complete,
            complete_correctness_passed=numeric, annotated_timing_records=annotated, groups=groups, **decision)
        if self.args.stage == "formal":
            self.report["calibration_passed"] = True  # Independently bound before reading formal samples.

    def run(self):
        self.initialize_identity(); self.load_window_and_raw(); self.check_numerical(); self.check_diagnostic_timing()
        if not self.report["complete_correctness_passed"]:
            outcome = "matched_numerical_gate_failed_no_performance_acceptance"
        elif self.args.stage == "calibration":
            outcome = "matched_calibration_passed_separate_R_eligible" if self.report["calibration_passed"] else "matched_calibration_failed_stop"
        else:
            outcome = "matched_formal_component_passed_fullmodel_eligible" if self.report["formal_component_passed"] else "matched_formal_component_gate_failed_stop"
        self.report.update(review_completed=True, outcome=outcome, production_adoption=False,
            old_R2_joint_gate_remains_failed=True, schema3_468_primary_rule_remains_failed=True,
            literal720_remains_descriptive=True, binary_completion_does_not_authorize_gates=True)
        return self.report


def check_protocol(p):
    require(p["schema_version"] == 1 and p["kind"] == "indexer_matched_output_reuse_protocol" and
            p["old_R2_joint_gate_remains_failed"] is True and p["schema3_diagnostic_rule_remains_failed"] is True and
            p["literal720_remains_descriptive_no_adoption"] is True and p["fresh_process_per_stage"] is True and
            p["same_binary_and_source_both_stages"] is True and p["original_replay_sha256"] == observer.ORIGINAL_SHA and
            p["layers"] == list(core.LAYERS) and p["controls"] == [0, 1] and p["formal_samples_per_arm"] == 9 and
            p["warmup_samples_per_arm"] == 1 and p["sequence_per_sample"] == ["conditioner", "A1", "B", "A2"] and
            p["output_slots_per_sample"] == [2, "control", 2, "control"] and p["conditioner_performance_sample"] is False and
            p["conditioner_is_same_actual_pso_args_output_as_middle_B"] is True and
            p["timing_records_per_stage"] == 240 and p["conditioner_records_per_stage"] == 60 and
            p["warmup_arm_records_per_stage"] == 18 and p["formal_arm_records_per_stage"] == 162 and
            p["calibration"]["max_sham_span_percent_inclusive"] == 0.5 and p["calibration"]["all_three_null_groups_required"] is True and
            p["formal"]["max_abs_control_drift_percent_inclusive"] == 2.0 and p["formal"]["strict_min_savings_percent"] == 2.0 and
            p["formal"]["all_six_groups_required"] is True and p["formal"]["production_adoption"] is False and
            p["no_sample_replacement_no_retry_no_threshold_relaxation"] is True, "immutable separate calibration/formal stage protocol")


def evidence_failure(report, stage, error):
    report = report if report is not None else dict(schema_version=1, stage=stage)
    report.update(review_completed=False, outcome="matched_evidence_invalid", evidence_error=str(error),
                  calibration_passed=False, formal_component_passed=False, fullmodel_eligible=False,
                  R_component_eligible=False, production_adoption=False)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("calibration", "formal"), required=True)
    for name in ("matched-protocol", "matched-build-manifest", "r2-protocol", "r2-build-manifest", "metal-manifest",
                 "capture-verified", "input-root", "baseline-identity", "reference-stdout", "window-dir", "window-wrapper", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--calibration-report", type=Path)
    p.add_argument("--capture-root", type=Path)
    p.add_argument("--result-dir", type=Path)
    p.add_argument("--jsonl", type=Path)
    p.add_argument("--path-map", action="append", default=[])
    args = p.parse_args()
    require(not args.out.exists(), "new independent stage report")
    args.protocol, args.build_manifest = args.r2_protocol, args.r2_build_manifest
    run, code = None, 0
    try:
        protocol = core.read_json(args.matched_protocol); check_protocol(protocol)
        run = MatchedRun(args, protocol); report = run.run()
    except (core.EvidenceError, KeyError, OSError, ValueError, TypeError, IndexError, StopIteration) as error:
        report = evidence_failure(run.report if run else None, args.stage, error)
        code = 2
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report.get(key) for key in ("review_completed", "stage", "outcome", "calibration_passed", "formal_component_passed", "fullmodel_eligible", "evidence_error")}, indent=2))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
