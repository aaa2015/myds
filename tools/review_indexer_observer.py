#!/usr/bin/env python3
"""Independent CPU reader for four finite literal-observer probe processes.

R/S share one minimally transformed executable. S changes only control1/B's
actual score PSO; logical provider2, output2 and the old observer stay intact.
The reader retains both labels and never treats the sham as candidate gain or
grants adoption eligibility. Frozen r2 CPU byte/ABI/ranking helpers are reused;
no benchmark implementation or prior experimental verdict is imported.
"""
import argparse
import difflib
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

import review_indexer_k_capture as core

require = core.require
CORE_SHA = "c403c38f2d254d0d4e113afad0e27e32ddf6684d0faeee2d70cac9d9001303f5"
ORIGINAL_SHA = "31f63df2a6fcbe3bd0d12a76723828c7e8f1a51cd981fa4f58da2941b669cefe"
GENERATED_SHA = "fa4de3ba8d401d58dbe249c2d85653a8a6937d4aa6d3d7a8ebfc41c1abc0cdf5"


def actual_scorer(sham, control, arm, logical_provider):
    return 1 if sham and control == 1 and arm == "B" else logical_provider


def checked_actual_label(record, sham):
    logical = 2 if record["arm"] == "B" else record["control"]
    actual = actual_scorer(sham, record["control"], record["arm"], logical)
    is_sham = sham and record["control"] == 1 and record["arm"] == "B"
    require(record["provider"] == logical and record["actual_provider_index"] == actual and
            core.harness_flag(record["tiled_b_sham"]) == is_sham, "logical/output provider and actual score-PSO witness agree")
    return actual


def describe(values):
    values = list(values)
    if not values:
        return dict(n=0)
    ordered = sorted(values)
    def percentile(fraction):
        index = (len(ordered) - 1) * fraction
        lo, hi = math.floor(index), math.ceil(index)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)
    return dict(n=len(values), min=min(values), max=max(values), mean=statistics.mean(values),
                median=statistics.median(values), p25=percentile(.25), p75=percentile(.75),
                population_stdev=statistics.pstdev(values))


def annotate_and_summarize(records, sham):
    expected = [(layer, control, sample, arm) for layer in core.LAYERS for control in (0, 1)
                for sample in range(10) for arm in ("A1", "B", "A2")]
    actual_order = [(r["layer"], r["control"], r["sample"], r["arm"]) for r in records]
    require(actual_order == expected[:len(records)] and len(records) <= 180, "one complete or prefix-only original 180-CB order")
    annotated = []
    for index, source in enumerate(records):
        r = dict(source)
        actual = checked_actual_label(r, sham)
        begin, end = r["gpu_start_s"], r["gpu_end_s"]
        require(math.isfinite(begin) and math.isfinite(end) and end > begin > 0 and
                abs((end - begin) * 1e6 - r["gpu_us"]) < 1e-6, "original GPU timestamp arithmetic")
        require(core.harness_flag(r["warmup"]) == (r["sample"] == 0), "original warmup sample labels")
        r.update(index=index, logical_output_provider=r["provider"], actual_scorer=actual,
                 actual_scorer_name=("captured_actual_default", "complete_TU_TILED", "complete_TU_K_resident")[actual],
                 role_is_TILED_sham=sham and r["control"] == 1,
                 gap_before_us=(begin - records[index - 1]["gpu_end_s"]) * 1e6 if index else None)
        if index:
            previous = annotated[-1]
            require(begin >= previous["gpu_end_s"], "serialized original command buffers")
            r["previous"] = dict(index=index - 1, layer=previous["layer"], control=previous["control"],
                arm=previous["arm"], logical_output_provider=previous["provider"], actual_scorer=previous["actual_scorer"],
                same_group=(previous["layer"], previous["control"]) == (r["layer"], r["control"]))
        else:
            r["previous"] = None
        annotated.append(r)
    groups = []
    for layer in core.LAYERS:
        for control in (0, 1):
            subset = [r for r in annotated if (r["layer"], r["control"]) == (layer, control)]
            if len(subset) != 30:
                continue
            formal = [r for r in subset if r["sample"] > 0]
            by_arm = {arm: [r for r in formal if r["arm"] == arm] for arm in ("A1", "B", "A2")}
            a1, b, a2 = [statistics.median(r["gpu_us"] for r in by_arm[arm]) for arm in ("A1", "B", "A2")]
            paired = []
            for sample in range(1, 10):
                triple = {r["arm"]: r for r in formal if r["sample"] == sample}
                av, bv, az = [triple[arm]["gpu_us"] for arm in ("A1", "B", "A2")]
                paired.append(dict(sample=sample, a1_us=av, b_us=bv, a2_us=az, a2_minus_a1_us=az - av,
                    a2_vs_a1_percent=100 * (az / av - 1), middle_vs_min_control_difference_percent=100 * (1 - bv / min(av, az)),
                    gaps_before_us={arm: triple[arm]["gap_before_us"] for arm in ("A1", "B", "A2")}))
            groups.append(dict(layer=layer, control=control,
                comparison_kind="same_TILED_PSO_sham_not_candidate_gain" if sham and control == 1 else "different_scorer_diagnostic_not_adoption",
                actual_scorers_by_arm={arm: sorted({r["actual_scorer"] for r in by_arm[arm]}) for arm in ("A1", "B", "A2")},
                a1_us=a1, b_us=b, a2_us=a2, original_ratio_of_medians_a2_vs_a1_percent=100 * (a2 / a1 - 1),
                middle_vs_min_control_difference_percent=100 * (1 - b / min(a1, a2)),
                paired_samples=paired, paired_a2_vs_a1_percent=describe(p["a2_vs_a1_percent"] for p in paired),
                paired_a2_faster_count=sum(p["a2_us"] < p["a1_us"] for p in paired),
                paired_a2_slower_count=sum(p["a2_us"] > p["a1_us"] for p in paired),
                paired_equal_count=sum(p["a2_us"] == p["a1_us"] for p in paired),
                per_arm_gpu_us={arm: describe(r["gpu_us"] for r in by_arm[arm]) for arm in ("A1", "B", "A2")},
                per_arm_gap_before_us={arm: describe(r["gap_before_us"] for r in by_arm[arm]) for arm in ("A1", "B", "A2")},
                original_thresholds_are_not_applied_as_new_diagnostic_acceptance=True,
                fullmodel_eligible=False, production_adoption=False))
    return annotated, groups


def verify_observer_build(path, reference, mapped):
    b = core.read_json(path)
    directory = Path(path).resolve().parent
    require(b["kind"] == "indexer_literal_observer_diagnostic_build" and b["compile_only_pass"] is True and
            b["diagnostic_only"] is True and b["adoption_cohort"] is False and b["fullmodel_eligible"] is False and
            b["production_adoption"] is False and b["runtime_registered"] is False and b["gpu_work_submitted"] is False and
            b["binary_executed"] is False and b["original_shared_output_observer_preserved"] is True and
            b["additional_cpu_timestamps_or_output_snapshots"] is False, "minimal diagnostic build scope")
    for rel, descriptor in b["inputs"].items():
        source = core.safe_child(directory / "source", rel)
        require(source.stat().st_size == descriptor["bytes"] and core.file_sha(source) == descriptor["sha256"], "frozen observer compiler input: " + rel)
    require(set(reference["inputs"]) <= set(b["inputs"]) and
            all(b["inputs"][rel] == descriptor for rel, descriptor in reference["inputs"].items()),
            "every original r2 compiler-visible source is retained byte-exact")
    original = (directory / "source/tools/bench_indexer_k_capture.m").read_bytes()
    generated = (directory / "source/tools/bench_indexer_observer_probe.m").read_bytes()
    proof_path = directory / "source-transform.json"
    proof = core.read_json(proof_path)
    require(core.sha(original) == b["original_replay_sha256"] == proof["original_replay_sha256"] == ORIGINAL_SHA and
            core.sha(generated) == b["generated_source_sha256"] == proof["generated_source_sha256"] == GENERATED_SHA,
            "exact independently reviewed original and minimal transformed source")
    require(core.file_sha(proof_path) == b["transformation_proof_sha256"] and proof["edit_count"] == 6 and
            proof["inverse_restores_original_byte_for_byte"] is True and b["inverse_restores_original_byte_for_byte"] is True,
            "six reversible source edits")
    restored = generated
    for edit in reversed(proof["exact_edits"]):
        before, after = edit["before"].encode(), edit["after"].encode()
        require(restored.count(after) == 1, "unique independent inverse source anchor")
        restored = restored.replace(after, before, 1)
    require(restored == original and core.sha(restored) == proof["inverse_restored_sha256"] == b["inverse_restored_sha256"], "entire original source restored, observer not rewritten")
    diff = "".join(difflib.unified_diff(original.decode().splitlines(keepends=True), generated.decode().splitlines(keepends=True),
                   fromfile="tools/bench_indexer_k_capture.m", tofile="tools/bench_indexer_observer_probe.m")).encode()
    require((directory / "observer-source.diff").read_bytes() == diff and core.sha(diff) == b["source_diff_sha256"], "frozen minimal source diff")
    command_data = (json.dumps(b["commands"], indent=2) + "\n").encode()
    require((directory / "build-commands.json").read_bytes() == command_data and core.sha(command_data) == b["commands_sha256"], "frozen compile/link commands")
    require(core.file_sha(mapped(b["binary"])) == b["binary_sha256"], "one executable for all four R/S processes")
    return b


class ObserverRun(core.Review):
    def __init__(self, args, observer_protocol, index, mode):
        super().__init__(args)
        self.observer_protocol, self.run_index, self.mode = observer_protocol, index, mode
        self.sham = mode == "sham"
        self.report.update(reader="independent_literal_observer_reader", reader_sha256=core.file_sha(__file__),
            cpu_helper_sha256=core.file_sha(core.__file__), run_index=index, mode=mode, adoption_cohort=False,
            old_R2_joint_gate_remains_failed=True, schema3_468_primary_rule_remains_failed=True)

    def initialize_identity(self):
        require(core.file_sha(core.__file__) == CORE_SHA, "unchanged frozen CPU byte/ranking helper")
        self.protocol = core.read_json(self.args.protocol)  # Original r2 input/ABI contract, not a new acceptance rule.
        self.capture = core.read_json(self.args.capture_verified)
        self.build = self.reference_build = core.read_json(self.args.build_manifest)
        self.metal = core.read_json(self.args.metal_manifest)
        self.check_protocol()
        self.check_build()
        self.check_capture()
        self.report["r2_input_protocol_sha256"] = self.report.pop("protocol_sha256")
        self.report["r2_reference_build_sha256"] = self.report.pop("build_manifest_sha256")
        self.build = verify_observer_build(self.args.observer_build_manifest, self.reference_build, self.mapped)
        self.report["build_manifest_sha256"] = core.file_sha(self.args.observer_build_manifest)
        self.report["observer_protocol_sha256"] = core.file_sha(self.args.observer_protocol)

    def load_window_and_raw(self):
        directory = self.args.window_dir
        plan = core.read_json(directory / "execution-plan.json")
        window = core.read_json(directory / "window.json")
        monitor = core.read_json(directory / "monitor.json")
        p = self.observer_protocol
        require(plan["protocol_sha256"] == window["protocol_sha256"] == core.file_sha(self.args.observer_protocol) and
                plan["build_manifest_sha256"] == core.file_sha(self.args.observer_build_manifest) and
                plan["binary_sha256"] == self.build["binary_sha256"] and plan["capture_json_sha256"] == p["capture_json_sha256"] and
                plan["probe_sha256"] == p["probe_sha256"] and plan["native_sha256"] == p["native_sha256"] and
                plan["config_sha256"] == self.capture["config_sha256"], "per-process frozen observer/input binding")
        require(plan["run_index"] == window["run_index"] == self.run_index and
                plan["mode"] == window["mode"] == self.mode and plan["adoption_cohort"] is False,
                "R/S schedule ordinal and mode belong to this fresh process")
        require(core.file_sha(self.args.window_wrapper) == plan["wrapper_sha256"], "exact frozen four-process wrapper")
        require(plan["MYDS_inherited"] == {} and plan["exclusive_recognized_process_check"] is True, "same clean inherited environment/exclusivity")
        argv = plan["argv"]
        flag_count = argv.count("--sham-tiled-b")
        require(flag_count == int(self.sham), "R/S physical CLI mode")
        valued = [word for word in argv if word != "--sham-tiled-b"]
        require(len(valued) == 13 and valued[1::2] == ["--capture-json", "--metallib", "--native-metallib", "--config", "--out-dir", "--samples"], "original observer argv and one optional boolean flag")
        options = dict(zip(valued[1::2], valued[2::2]))
        require(core.file_sha(self.mapped(valued[0])) == self.build["binary_sha256"] and
                core.file_sha(self.mapped(options["--capture-json"])) == p["capture_json_sha256"] and
                core.file_sha(self.mapped(options["--metallib"])) == p["probe_sha256"] and
                core.file_sha(self.mapped(options["--native-metallib"])) == p["native_sha256"] and
                core.file_sha(self.mapped(options["--config"])) == self.capture["config_sha256"] and options["--samples"] == "9",
                "actual process operands")
        self.result_dir = self.args.result_dir or directory / "outputs"
        require(self.mapped(options["--out-dir"]).resolve() == self.result_dir.resolve(), "actual output path")
        baseline = core.read_json(self.args.baseline_identity)
        before, after = window["canonical_before"], window["canonical_after"]
        require(before == after and window["canonical_unchanged"] is True and len(before) == 3 and
                {Path(key).name for key in before} == {"myds", "myds.metallib", "myds.conf"} and
                all(key in baseline and baseline[key]["sha256"] == digest for key, digest in before.items()), "canonical frozen before/after identities")
        child = window["child_pid"]
        require(isinstance(child, int) and child > 0 and monitor and not window["recognized_before"] and not window["recognized_after"] and
                window["abort_reason"] is None and all(all(row["pid"] == child for row in entry["recognized"]) for entry in monitor) and
                any(any(row["pid"] == child and Path(row["comm"]).name == "bench_indexer_observer_probe" for row in entry["recognized"]) for entry in monitor), "fresh child/no recognized competing inference process")
        self.window, self.plan = window, plan
        require(plan["started_epoch"] <= window["finished_epoch"] and 0 <= window["elapsed_s"] <= plan["timeout_s"] + 20,
                "bounded process wall window")
        self.report["started_epoch"], self.report["finished_epoch"] = plan["started_epoch"], window["finished_epoch"]
        m = self.manifest = core.read_json(self.result_dir / "artifact-manifest.json")
        require(m["probe"] == "indexer_literal_observer_probe" and m["fixture_kind"] == "actual_fullmodel_prefill" and
                m["diagnostic_only"] is True and m["adoption_cohort"] is False and m["production_adoption"] is False and
                m["fullmodel_eligible"] is False and m["fullmodel"] is False and m["gpu_work_submitted"] is True and
                m["binary_sha256"] == self.build["binary_sha256"] and m["capture_json_sha256"] == p["capture_json_sha256"] and
                m["probe_sha256"] == p["probe_sha256"] and m["native_sha256"] == p["native_sha256"] and
                m["config_sha256"] == self.capture["config_sha256"] and core.harness_flag(m["tiled_b_sham_enabled"]) == self.sham and
                m["selected_mode"] == ("sham_tiled_b" if self.sham else "real_candidate") and
                m["samples"] == 9 and m["warmups"] == 1 and m["mode"] == "concurrent", "diagnostic manifest identity and R/S binding")
        require(m["topk_loaded_library_sha256"] == p["native_sha256"] and
                core.file_sha(self.mapped(m["topk_loaded_library_path"])) == p["native_sha256"] and
                self.mapped(m["topk_loaded_library_path"]).resolve() == self.mapped(options["--native-metallib"]).resolve(),
                "actual phase1 TopK loaded native library")
        for item in m["files"]:
            name = item["path"]
            require(name not in self.raw, "unique raw manifest filename")
            path = core.safe_child(self.result_dir, name, single=True)
            require(path.stat().st_size == item["bytes"] and core.file_sha(path) == item["sha256"], "full raw SHA/bytes: " + name)
            self.raw[name] = path
        self.records, info = [], []
        jsonl = self.args.jsonl or directory / "stdout.jsonl"
        for line in jsonl.read_text().splitlines():
            if not line.strip():
                continue
            if line.startswith("{"):
                self.records.append(json.loads(line, object_pairs_hook=core.unique_object))
            else:
                require(not self.records, "only native initialization precedes JSONL")
                match = re.fullmatch(r"\[\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d{3}\] \[INFO\] (.+)", line)
                require(match is not None, "known INFO initialization format")
                info.append(match.group(1))
        old_info = [re.fullmatch(r"\[.+\] \[INFO\] (.+)", line).group(1) for line in self.args.reference_stdout.read_text().splitlines() if line and not line.startswith("{")]
        require(len(info) == len(old_info) == 21 and info[2:] == old_info[2:], "same complete frozen native initialization contract")
        config = re.fullmatch(r"config: 已加载 (.+) —— 生效 (\d+) 条", info[0])
        library = re.fullmatch(r"myds_metal: loaded precompiled library from (.+) \(all required V4\.1 kernels verified\)", info[1])
        require(config and library and int(config.group(2)) == self.capture["config_declared_unique_count"] and
                core.file_sha(self.mapped(config.group(1))) == self.capture["config_sha256"] and
                core.file_sha(self.mapped(library.group(1))) == p["native_sha256"], "initialization config/library bytes")
        require(all(r["kind"] in {"real_inputs", "real_parity", "real_topk", "all_layers_numerical_gate", "real_timing", "real_summary"} for r in self.records), "only original record kinds")
        self.report.update(child_pid=child, raw_files_verified=len(self.raw), raw_logical_bytes=sum(path.stat().st_size for path in self.raw.values()),
            artifact_manifest_sha256=core.file_sha(self.result_dir / "artifact-manifest.json"), stdout_sha256=core.file_sha(jsonl),
            window_evidence_sha256={name: core.file_sha(directory / name) for name in ("execution-plan.json", "window.json", "monitor.json")})

    def check_diagnostic_timing(self):
        indexed = [(index, r) for index, r in enumerate(self.records) if r["kind"] == "real_timing"]
        if indexed:
            require(self.numerical_passed and self.phase1_gate_index is not None and
                    all(index > self.phase1_gate_index for index, _ in indexed), "all three layers numerical/TopK gate before first timing")
        annotated, groups = annotate_and_summarize([r for _, r in indexed], self.sham)
        for r in annotated:
            initial = self.initial_scores[(r["layer"], r["provider"])]
            tiled = self.initial_scores[(r["layer"], 1)]
            require(r["full_scores_sha256"] == core.sha(initial) and r["provider_full_scores_bit_exact_to_tiled"] == (initial == tiled) and
                    r["candidate_full_scores_to_tiled_gate_passed"] is True and r["all_layers_numerical_gate_passed"] is True and
                    r["fullmodel_speedup_proven"] is False, "same-provider full-score readback per timing, distinct from actual PSO label")
        summaries = [r for r in self.records if r["kind"] == "real_summary"]
        require(len(summaries) == len(groups), "complete original summary coverage")
        for g in groups:
            matches = [r for r in summaries if (r["layer"], r["control"]) == (g["layer"], g["control"])]
            require(len(matches) == 1, "one original summary per layer/control")
            r = matches[0]
            for name, actual in (("a1_us", g["a1_us"]), ("b_us", g["b_us"]), ("a2_us", g["a2_us"]),
                    ("control_drift_percent", g["original_ratio_of_medians_a2_vs_a1_percent"]),
                    ("conservative_time_saved_percent", g["middle_vs_min_control_difference_percent"])):
                require(abs(r[name] - actual) < 1e-6, "recomputed descriptive original summary: " + name)
        complete = len(annotated) == 180 and len(groups) == 6 and all((layer, name, "phase2_post_timing") in self.input_seen for layer in core.LAYERS for name in core.INPUTS)
        numeric = self.all_numerical_passed and all(r["passed"] for r in self.input_reports)
        if self.manifest["status"] == "diagnostic_complete_not_adoption":
            require(complete and numeric and self.window["exit_code"] == 0, "complete independent process requires all raw/input/timing records")
        else:
            require(self.manifest["status"] == "failed" and self.window["exit_code"] != 0, "known partial diagnostic failure")
        require(self.consumed == set(self.raw), "all manifest raw files independently consumed")
        self.report.update(timing_records=len(annotated), warmup_records=sum(r["sample"] == 0 for r in annotated),
            formal_records=sum(r["sample"] > 0 for r in annotated), complete_run=complete, complete_correctness_passed=numeric,
            annotated_timing_records=annotated, descriptive_groups=groups, fullmodel_eligible=False, production_adoption=False,
            original_summary_savings_are_not_diagnostic_acceptance_gates=True)

    def run(self):
        self.initialize_identity()
        self.load_window_and_raw()
        self.check_numerical()
        self.check_diagnostic_timing()
        self.report.update(review_completed=True,
            outcome="literal_observer_process_complete_descriptive_only" if self.report["complete_run"] and self.report["complete_correctness_passed"] else "literal_observer_process_failed_no_timing_interpretation")
        return self.report


def check_protocol(p):
    require(p["schema_version"] == 1 and p["kind"] == "indexer_literal_observer_fidelity_diagnostic" and
            p["schedule"] == ["real", "sham", "sham", "real"] and p["fresh_process_per_run"] is True and
            p["same_binary_all_runs"] is True and p["samples"] == 9 and p["warmups"] == 1 and
            p["timing_records_per_run"] == 180 and p["timing_records_total"] == 720 and
            p["warmup_timing_records_total"] == 72 and p["formal_timing_records_total"] == 648 and
            p["observation_resolution_percent"] == 0.5 and p["adoption_cohort"] is False and
            p["no_automatic_next_stage"] is True and p["fullmodel_eligible"] is False and p["production_adoption"] is False and
            p["old_R2_joint_gate_remains_failed"] is True and p["schema3_468_primary_rule_remains_failed"] is True and
            p["one_finite_cohort_no_replacement_or_retry"] is True and p["original_replay_sha256"] == ORIGINAL_SHA,
            "one finite predeclared four-process R/S/S/R descriptive diagnostic")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("observer-protocol", "r2-protocol", "observer-build-manifest", "r2-build-manifest", "metal-manifest",
                 "capture-verified", "input-root", "baseline-identity", "reference-stdout", "window-wrapper", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--run-dir", type=Path, action="append", required=True, help="Four directories in frozen real/sham/sham/real order.")
    p.add_argument("--capture-root", type=Path)
    p.add_argument("--path-map", action="append", default=[])
    args = p.parse_args()
    require(not args.out.exists() and len(args.run_dir) == 4, "new report and exactly four frozen process directories")
    report = dict(schema_version=1, reader="independent_literal_observer_four_process_reader", reader_sha256=core.file_sha(__file__),
                  review_completed=False, adoption_cohort=False, fullmodel_eligible=False, production_adoption=False,
                  old_R2_joint_gate_remains_failed=True, schema3_468_primary_rule_remains_failed=True, runs=[])
    code = 0
    try:
        protocol = core.read_json(args.observer_protocol)
        check_protocol(protocol)
        report["observer_protocol_sha256"] = core.file_sha(args.observer_protocol)
        for index, (mode, directory) in enumerate(zip(protocol["schedule"], args.run_dir)):
            run_args = argparse.Namespace(**vars(args))
            run_args.protocol, run_args.build_manifest = args.r2_protocol, args.r2_build_manifest
            run_args.window_dir, run_args.result_dir, run_args.jsonl = directory, None, None
            run = ObserverRun(run_args, protocol, index, mode)
            report["runs"].append(run.report)
            run.run()
            if not run.report["complete_run"] or not run.report["complete_correctness_passed"]:
                report.update(review_completed=True, outcome="literal_observer_numerical_or_execution_failure_no_timing_interpretation",
                              no_automatic_next_stage=True)
                args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                print(json.dumps({key: report[key] for key in ("review_completed", "outcome", "fullmodel_eligible")}, indent=2))
                return
        require(len({r["child_pid"] for r in report["runs"]}) == 4, "four independently launched processes")
        require(len({r["build_manifest_sha256"] for r in report["runs"]}) == 1, "same frozen executable/build all modes")
        require(all(a["finished_epoch"] <= b["started_epoch"] for a, b in zip(report["runs"], report["runs"][1:])), "four independent nonoverlapping process windows")
        parents = {directory.resolve().parent for directory in args.run_dir}
        require(len(parents) == 1, "one frozen four-process window directory")
        outer_path = next(iter(parents)) / "window.json"
        outer = core.read_json(outer_path)
        require(outer["protocol_sha256"] == core.file_sha(args.observer_protocol) and outer["schedule"] == protocol["schedule"] and
                outer["all_four_runs_completed"] is True and len(outer["runs"]) == 4 and not outer["recognized_after"] and
                outer["canonical_before"] == outer["canonical_after"] and outer["fullmodel_eligible"] is False and outer["production_adoption"] is False and
                all(outer["runs"][index] == core.read_json(args.run_dir[index] / "window.json") for index in range(4)), "outer window binds all four original child receipts")
        report["outer_window_sha256"] = core.file_sha(outer_path)
        require(sum(r["timing_records"] for r in report["runs"]) == 720 and sum(r["warmup_records"] for r in report["runs"]) == 72 and
                sum(r["formal_records"] for r in report["runs"]) == 648, "all four complete original timing cohorts retained")
        comparisons = []
        for layer in core.LAYERS:
            for control in (0, 1):
                rows = [next(g for g in r["descriptive_groups"] if (g["layer"], g["control"]) == (layer, control)) for r in report["runs"]]
                comparisons.append(dict(layer=layer, control=control,
                    modes=[r["mode"] for r in report["runs"]],
                    original_ratio_of_medians_a2_vs_a1_percent=[g["original_ratio_of_medians_a2_vs_a1_percent"] for g in rows],
                    paired_a2_faster_counts=[g["paired_a2_faster_count"] for g in rows],
                    middle_vs_min_control_difference_percent=[g["middle_vs_min_control_difference_percent"] for g in rows],
                    sham_is_null_only_for_control1=control == 1,
                    observation_resolution_percent=protocol["observation_resolution_percent"],
                    categorical_causal_interpretation_not_automated=True))
        report.update(review_completed=True, outcome="four_process_literal_observer_diagnostic_descriptive_only",
                      comparisons=comparisons, no_automatic_next_stage=True,
                      observer_fidelity="Original shared-output validation/SHA/memcmp/JSON/fflush retained, but CLI/manifest/two timing fields added in both R/S; not the original binary byte-for-byte.",
                      causal_limitations="Conditioner/provider/output-slot, process state and host observation are not unique hardware frequency/cache measurements.")
    except (core.EvidenceError, KeyError, OSError, ValueError, TypeError, IndexError, StopIteration) as error:
        code = 2
        report.update(outcome="observer_evidence_invalid_or_incomplete", evidence_error=str(error))
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report.get(k) for k in ("review_completed", "outcome", "fullmodel_eligible", "evidence_error")}, indent=2))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
