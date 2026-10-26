#!/usr/bin/env python3
"""CPU-only reader for the finite schema-3 predecessor diagnostic.

The frozen independent r2 reader supplies only byte/hash/ABI/ranking helpers;
no benchmark code or prior acceptance verdict is imported. This reader binds
the new build/window, checks every retained output, and computes the new
diagnostic rules. It never grants performance-adoption eligibility.
"""
import argparse
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
PHASES = [
    ("sham_exact_inline", 0, [0, 0, 0], False, False),
    ("sham_tiled_inline", 1, [1, 1, 1], False, False),
    ("sham_candidate_inline", 2, [2, 2, 2], False, False),
    ("legacy_tiled_inline", 1, [1, 2, 1], False, False),
    ("balanced_tiled_inline", 2, [1, 2, 1], False, False),
    ("legacy_tiled_deferred", 1, [1, 2, 1], True, False),
    ("balanced_tiled_deferred", 2, [1, 2, 1], True, False),
    ("legacy_tiled_distinct", 1, [1, 2, 1], False, True),
    ("balanced_tiled_distinct", 2, [1, 2, 1], False, True),
]


def phase_index(round_index, order):
    return 8 - order if round_index and round_index % 2 else order


def expected_schedule():
    return [(round_index, order, phase_index(round_index, order), step)
            for round_index in range(13) for order in range(9) for step in range(4)]


def summary_and_decision(records, rule):
    """Pure CPU statistics; no sample selection or alternate decision phases."""
    summaries, by_name = [], {}
    for pi, (name, conditioner, arms, deferred, distinct) in enumerate(PHASES):
        subset = [r for r in records if r["phase_index"] == pi and r["round"] > 0]
        require(len(subset) == 48, "each diagnostic phase has 12 complete four-step formal rounds")
        med = {role: statistics.median(r["gpu_us"] for r in subset if r["role"] == role)
               for role in ("conditioner", "A1", "B", "A2")}
        drift = 100 * (med["A2"] / med["A1"] - 1)
        slot_range = 100 * (max(med[x] for x in ("A1", "B", "A2")) / min(med[x] for x in ("A1", "B", "A2")) - 1)
        paired = []
        for round_index in range(1, 13):
            group = {r["role"]: r for r in subset if r["round"] == round_index}
            paired.append(dict(round=round_index, phase_order=group["A1"]["phase_order"],
                a1_us=group["A1"]["gpu_us"], b_us=group["B"]["gpu_us"], a2_us=group["A2"]["gpu_us"],
                signed_a2_vs_a1_percent=100 * (group["A2"]["gpu_us"] / group["A1"]["gpu_us"] - 1)))
        row = dict(name=name, phase_index=pi, medians_us=med, signed_median_a2_vs_a1_percent=drift,
                   sham_all_three_slot_span_percent=slot_range, a2_faster_percent=-drift,
                   paired_rounds=paired, descriptive_only=pi >= 5)
        summaries.append(row)
        by_name[name] = row
    shams = {name: by_name[name]["sham_all_three_slot_span_percent"] <= rule["sham_stability_max_abs_median_slot_drift_percent"]
             for name in ("sham_exact_inline", "sham_tiled_inline", "sham_candidate_inline")}
    legacy = by_name["legacy_tiled_inline"]
    balanced = by_name["balanced_tiled_inline"]
    reproduction = legacy["a2_faster_percent"] >= rule["legacy_reproduction_min_A2_faster_percent_inclusive"]
    balanced_pass = abs(balanced["signed_median_a2_vs_a1_percent"]) <= rule["balanced_inline_max_abs_median_slot_drift_percent"]
    reduction = [dict(round=a["round"], legacy_signed_drift_percent=a["signed_a2_vs_a1_percent"],
                      balanced_signed_drift_percent=b["signed_a2_vs_a1_percent"],
                      reduction_percentage_points=abs(a["signed_a2_vs_a1_percent"]) - abs(b["signed_a2_vs_a1_percent"]))
                 for a, b in zip(legacy["paired_rounds"], balanced["paired_rounds"])]
    median_reduction = statistics.median(r["reduction_percentage_points"] for r in reduction)
    positive = sum(r["reduction_percentage_points"] > 0 for r in reduction)
    reduction_pass = median_reduction >= rule["within_round_conditioning_reduction_min_median_percentage_points_inclusive"] and \
        positive >= rule["within_round_conditioning_reduction_positive_pairs_min"]
    decision = dict(sham_stability=shams, all_three_shams_passed=all(shams.values()),
                    legacy_reproduction_passed=reproduction, balanced_inline_stability_passed=balanced_pass,
                    paired_conditioning_reductions=reduction, median_reduction_percentage_points=median_reduction,
                    strictly_positive_pairs=positive, conditioning_reduction_passed=reduction_pass,
                    measurement_correction_supported=all(shams.values()) and reproduction and balanced_pass and reduction_pass,
                    formal_component_adoption_eligible=False, fullmodel_eligible=False, production_adoption=False)
    return summaries, decision


class StabilityReview(core.Review):
    def __init__(self, args):
        super().__init__(args)
        self.report.update(reader="independent_predecessor_stability_diagnostic_reader",
                           reader_sha256=core.file_sha(__file__), cpu_helper_sha256=core.file_sha(core.__file__),
                           adoption_cohort=False, old_joint_gate_remains_failed=True)
        self.cache = {}
        self.bytes_cache = {}
        self.stats_cache = {}

    def artifact(self, name):
        require(name in self.raw, "manifest-listed raw artifact: " + name)
        self.consumed.add(name)
        path = self.raw[name]
        stat = path.stat()
        key = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if key not in self.bytes_cache:
            self.bytes_cache[key] = path.read_bytes()
        return self.bytes_cache[key]

    def check_frozen(self):
        require(core.file_sha(core.__file__) == CORE_SHA, "unchanged frozen independent CPU helper")
        self.protocol = p = core.read_json(self.args.protocol)
        require(p["schema_version"] == 3 and p["kind"] == "indexer_predecessor_stability_diagnostic" and
                p["adoption_cohort"] is False and p["layer"] == 2 and p["shape"] == [128, 15424] and
                p["warmup_rounds"] == 1 and p["formal_rounds"] == 12 and p["timing_records"] == 468 and
                p["steps"] == ["conditioner", "A1", "B", "A2"] and p["old_joint_gate_remains_failed"] is True and
                p["fullmodel_run"] is False and p["production_adoption"] is False, "finite schema3 diagnostic scope")
        require(len(p["phases"]) == 9, "nine frozen phases")
        for actual, expected in zip(p["phases"], PHASES):
            name, conditioner, arms, deferred, distinct = expected
            require(actual == dict(name=name, conditioner=conditioner, arms=arms,
                    deferred_validation=deferred, distinct_outputs=distinct), "exact frozen phase matrix")
        rule = p["decision_rule"]
        for key in ("sham_stability_max_abs_median_slot_drift_percent", "balanced_inline_max_abs_median_slot_drift_percent",
                    "legacy_reproduction_min_A2_faster_percent_inclusive", "within_round_conditioning_reduction_min_median_percentage_points_inclusive"):
            require(rule[key] == 0.5, "immutable diagnostic threshold: " + key)
        require(rule["within_round_conditioning_reduction_positive_pairs_min"] == 10 and
                rule["decision_phases"] == [phase[0] for phase in PHASES[:5]] and
                rule["descriptive_only_phases"] == [phase[0] for phase in PHASES[5:]] and
                rule["formal_component_adoption_eligible_from_diagnostic"] is False, "decision uses only frozen five phases")
        self.capture = c = core.read_json(self.args.capture_verified)
        require(core.file_sha(self.args.capture_verified) == p["capture_json_sha256"] and
                c["status"] == "verified_cpu_only" and c["capture_kind"] == "myds_real_prefill_indexer", "actual verified capture binding")
        require(core.file_sha(self.args.config) == c["config_sha256"], "same production config bytes")
        caproot = self.args.capture_root or self.mapped(c["capture_directory"])
        selected = next(case for case in c["cases"] if case["layer"] == 2)
        payloads = {key: self.payload(descriptor, caproot) for key, descriptor in selected["payloads"].items()}
        require(selected["n_tokens"] == 128 and selected["n_comp"] == 15424 and selected["pos0"] == 30720 and
                selected["ratio"] == 2 and selected["cache_bf16"] is True and selected["synthetic"] is False,
                "exact failed r2 real subgroup selected before diagnostic")
        self.case = dict(metadata=selected, payloads=payloads, rows=128, keys=15424, pos=30720, ratio=2)
        self.build = b = core.read_json(self.args.build_manifest)
        require(b["kind"] == "indexer_stability_diagnostic_build" and b["compile_only_pass"] is True and
                b["gpu_work_submitted"] is False and b["production_registered"] is False and b["adoption_cohort"] is False,
                "CPU-built diagnostic-only executable")
        buildroot = self.args.build_manifest.resolve().parent
        for rel, item in b["inputs"].items():
            path = core.safe_child(buildroot / "source", rel)
            require(path.stat().st_size == item["bytes"] and core.file_sha(path) == item["sha256"], "complete frozen diagnostic source: " + rel)
        original = (buildroot / "source/tools/bench_indexer_k_capture.m").read_bytes()
        marker = b'int main(int argc,char **argv){@autoreleasepool {'
        require(core.sha(original) == "31f63df2a6fcbe3bd0d12a76723828c7e8f1a51cd981fa4f58da2941b669cefe" and original.count(marker) == 1,
                "original frozen r2 replay is unmodified")
        prefix, suffix = original.split(marker, 1)
        generated = (buildroot / "source/tools/indexer_replay_helpers.inc").read_bytes()
        require(generated == prefix and len(generated) == b["helpers_bytes"] and core.sha(generated) == b["helpers_sha256"] and
                core.sha(marker + suffix) == b["removed_main_suffix_sha256"], "helper extraction exact and lossless")
        require(core.file_sha(self.mapped(b["binary"])) == b["binary_sha256"], "actual diagnostic executable identity")
        require(core.file_sha(self.args.probe_library) == p["probe_sha256"] and core.file_sha(self.args.native_library) == p["native_sha256"] == c["loaded_metallib_sha256"],
                "unchanged frozen score/TopK libraries")
        self.report.update(protocol_sha256=core.file_sha(self.args.protocol), build_manifest_sha256=core.file_sha(self.args.build_manifest),
                           source_files_verified=len(b["inputs"]), capture_json_sha256=core.file_sha(self.args.capture_verified))

    def check_window_and_load(self):
        wd = self.args.window_dir
        plan = core.read_json(wd / "execution-plan.json")
        window = core.read_json(wd / "window.json")
        monitor = core.read_json(wd / "monitor.json")
        require(plan["protocol_sha256"] == window["protocol_sha256"] == core.file_sha(self.args.protocol) and
                plan["build_manifest_sha256"] == core.file_sha(self.args.build_manifest) and plan["binary_sha256"] == self.build["binary_sha256"] and
                plan["probe_sha256"] == self.protocol["probe_sha256"] and plan["native_sha256"] == self.protocol["native_sha256"] and
                plan["capture_json_sha256"] == self.protocol["capture_json_sha256"] and plan["config_sha256"] == core.file_sha(self.args.config),
                "actual launch binds complete frozen diagnostic inputs")
        require(plan["MYDS_inherited"] == {} and plan["exclusive_recognized_process_check"] is True, "clean child environment and recognized exclusivity")
        argv = plan["argv"]
        require(len(argv) == 13, "exact diagnostic argv size")
        options = dict(zip(argv[1::2], argv[2::2]))
        require(set(options) == {"--capture-json", "--metallib", "--native-metallib", "--config", "--protocol", "--out-dir"}, "exact six diagnostic operands")
        hashes = {"--capture-json": core.file_sha(self.args.capture_verified), "--metallib": self.protocol["probe_sha256"],
                  "--native-metallib": self.protocol["native_sha256"], "--config": core.file_sha(self.args.config), "--protocol": core.file_sha(self.args.protocol)}
        require(core.file_sha(self.mapped(argv[0])) == self.build["binary_sha256"] and all(core.file_sha(self.mapped(options[k])) == digest for k, digest in hashes.items()), "actual child operands hash matched")
        self.result_dir = self.args.result_dir or wd / "outputs"
        require(self.mapped(options["--out-dir"]).resolve() == self.result_dir.resolve(), "actual output path")
        before, after = window["canonical_before"], window["canonical_after"]
        baseline = core.read_json(self.args.baseline_identity)
        require(before == after and window["canonical_unchanged"] is True and len(before) == 3 and
                {Path(k).name for k in before} == {"myds", "myds.metallib", "myds.conf"} and
                all(k in baseline and baseline[k]["sha256"] == v for k, v in before.items()), "canonical before/after/baseline bytes")
        child = window["child_pid"]
        require(isinstance(child, int) and child > 0 and monitor and not window["recognized_before"] and not window["recognized_after"] and
                window["abort_reason"] is None and all(all(row["pid"] == child for row in sample["recognized"]) for sample in monitor) and
                any(any(row["pid"] == child and Path(row["comm"]).name == "bench_indexer_stability" for row in sample["recognized"]) for sample in monitor), "actual diagnostic child and no recognized competitors")
        if self.args.window_wrapper:
            require(core.file_sha(self.args.window_wrapper) == plan["wrapper_sha256"], "frozen monitoring wrapper")
        self.window = window
        self.manifest = m = core.read_json(self.result_dir / "artifact-manifest.json")
        require(m["probe"] == "indexer_predecessor_stability_diagnostic" and m["adoption_cohort"] is False and
                m["fullmodel"] is False and m["production_adoption"] is False and m["gpu_work_submitted"] is True and
                m["protocol_sha256"] == core.file_sha(self.args.protocol) and m["capture_json_sha256"] == self.protocol["capture_json_sha256"] and
                m["probe_sha256"] == self.protocol["probe_sha256"] and m["native_sha256"] == self.protocol["native_sha256"] and
                m["binary_sha256"] == self.build["binary_sha256"] and m["config_sha256"] == core.file_sha(self.args.config), "diagnostic artifact identity and scope")
        require(m["formal_rounds"] == 12 and m["warmup_rounds"] == 1 and m["phase_count"] == 9 and
                m["timing_record_count_expected"] == 468 and m["layer"] == 2 and m["tokens"] == 128 and m["keys"] == 15424,
                "finite diagnostic output extent")
        require(self.mapped(m["loaded_native_library_path"]).resolve() == self.args.native_library.resolve() and
                core.file_sha(self.mapped(m["loaded_native_library_path"])) == self.protocol["native_sha256"], "actual native TopK loaded library")
        for item in m["files"]:
            name = item["path"]
            require(name not in self.raw, "unique raw manifest name")
            path = core.safe_child(self.result_dir, name, single=True)
            st = path.stat(); key = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
            if key not in self.cache:
                self.cache[key] = core.file_sha(path)
            require(st.st_size == item["bytes"] and self.cache[key] == item["sha256"], "raw full payload identity: " + name)
            self.raw[name] = path
        self.records = []
        info = []
        for line in (self.args.jsonl or wd / "stdout.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            if line.startswith("{"):
                self.records.append(json.loads(line, object_pairs_hook=core.unique_object))
            else:
                require(not self.records, "initialization messages precede structured diagnostic records")
                match = re.fullmatch(r"\[\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d{3}\] \[INFO\] (.+)", line)
                require(match is not None, "known native INFO prefix")
                info.append(match.group(1))
        # Bind the full prior native INFO contract, normalizing only the two
        # exact path-bearing messages; all flags and their order stay exact.
        reference_lines = [line for line in self.args.reference_stdout.read_text().splitlines() if line and not line.startswith("{")]
        expected_info = [re.fullmatch(r"\[.+\] \[INFO\] (.+)", line).group(1) for line in reference_lines]
        require(len(info) == len(expected_info) == 21 and info[2:] == expected_info[2:], "exact frozen 21-message native initialization contract")
        config = re.fullmatch(r"config: 已加载 (.+) —— 生效 (\d+) 条", info[0])
        library = re.fullmatch(r"myds_metal: loaded precompiled library from (.+) \(all required V4\.1 kernels verified\)", info[1])
        require(config and library and int(config.group(2)) == self.capture["config_declared_unique_count"] and
                core.file_sha(self.mapped(config.group(1))) == core.file_sha(self.args.config) and
                core.file_sha(self.mapped(library.group(1))) == self.protocol["native_sha256"], "known initialization paths bind config/library")
        self.report.update(raw_files_verified=len(self.raw), raw_logical_bytes=sum(p.stat().st_size for p in self.raw.values()),
                           raw_unique_physical_payloads=len(self.cache), reference_stdout_sha256=core.file_sha(self.args.reference_stdout),
                           artifact_manifest_sha256=core.file_sha(self.result_dir / "artifact-manifest.json"),
                           stdout_sha256=core.file_sha(self.args.jsonl or wd / "stdout.jsonl"), child_pid=child,
                           window_evidence_sha256={name: core.file_sha(wd / name) for name in ("execution-plan.json", "window.json", "monitor.json")})

    def facts(self, data, guards):
        key = (core.sha(data), core.sha(guards))
        if key not in self.stats_cache:
            stats = self.score_stats(data, self.case)
            actual = core.diff_words(self.case["payloads"]["scores-f32"], data, float_metrics=True)
            tiled = core.diff_words(self.tiled, data, float_metrics=True)
            self.stats_cache[key] = dict(stats=stats, actual=actual, tiled=tiled, guards=guards == core.GUARD * 2)
        return self.stats_cache[key]

    def check_initial_and_inputs(self):
        initial_scores, initial_ids = {}, {}
        numerical = True
        score_records = [r for r in self.records if r["kind"] == "real_parity"]
        require([r["provider"] for r in score_records] == [0, 1, 2] and all(r["layer"] == 2 for r in score_records), "initial three complete scorers")
        self.tiled = self.artifact("layer-2-provider-1-scores.f32")
        for r in score_records:
            provider = r["provider"]
            data = self.artifact(f"layer-2-provider-{provider}-scores.f32")
            guards = self.artifact(f"layer-2-provider-{provider}-guards.bin")
            initial_scores[provider] = data
            facts = self.facts(data, guards)
            ok = facts["guards"] and all(facts["stats"][k] for k in ("visible_finite", "causal_tail_valid", "all_written")) and \
                (facts["actual"] if provider == 0 else facts["tiled"])["mismatches"] == 0
            require(r["mismatches_to_actual_capture"] == facts["actual"]["mismatches"] and r["passed"] == ok and
                    (provider != 2 or r["candidate_mismatches_to_tiled"] == facts["tiled"]["mismatches"]), "initial score record versus complete bytes")
            numerical &= ok
        topk_records = [r for r in self.records if r["kind"] == "real_topk"]
        require([r["provider"] for r in topk_records] == [0, 1, 2], "initial ordered TopK for all providers")
        for r in topk_records:
            provider = r["provider"]
            ids = self.artifact(f"layer-2-provider-{provider}-selected.i32")
            guards = self.artifact(f"layer-2-provider-{provider}-topk-guards.bin")
            initial_ids[provider] = ids
            actual = core.diff_words(self.case["payloads"]["selected-i32"], ids)
            counters = self.case["metadata"]["actual_topk_consumer"]
            counter_path = all(r[k] == counters[k] for k in core.COUNTERS)
            ok = actual["mismatches"] == 0 and guards == core.GUARD * 4 and counter_path and \
                core.harness_flag(r["host_encoded"]) and core.harness_flag(r["GPU_completed"]) and r["score_input_unchanged"] is True
            require(r["actual_capture_mismatches"] == actual["mismatches"] and r["passed"] == ok, "initial ordered ID record versus raw")
            numerical &= ok and self.ordered_ids_stats(initial_scores[provider], ids, self.case)["passed"]
        numerical &= initial_scores[2] == initial_scores[1] and initial_ids[2] == initial_ids[1]
        inputs = []
        for r in self.records:
            if r["kind"] != "real_inputs":
                continue
            require(r["layer"] == 2 and r["input"] in core.INPUTS and r["phase"] in
                    ("diagnostic_before", "diagnostic_after", "diagnostic_score_failure", "diagnostic_gpu_failure"), "known diagnostic input phase")
            require(r["available"] is True, "allocated diagnostic input body")
            data, prefix, suffix = self.artifact(r["post_raw"]), self.artifact(r["guard_prefix_raw"]), self.artifact(r["guard_suffix_raw"])
            reference = self.case["payloads"][r["input"]]
            ok = data == reference and prefix == suffix == core.GUARD
            require(len(data) == r["bytes"] == len(reference) and r["post_sha256"] == core.sha(data) and
                    r["capture_sha256"] == core.sha(reference) and r["guard_prefix_sha256"] == core.sha(prefix) and
                    r["guard_suffix_sha256"] == core.sha(suffix) and r["passed"] == ok and r["guards_intact"] == (prefix == suffix == core.GUARD) and
                    r["bytes_equal_capture"] == (data == reference) and r["guard_bytes_per_side"] == 64,
                    "input record versus full post/pre/guard bytes")
            inputs.append(dict(phase=r["phase"], input=r["input"], passed=ok, bytes=len(data)))
        before = [r for r in inputs if r["phase"] == "diagnostic_before"]
        require(len(before) == 3 and {r["input"] for r in before} == set(core.INPUTS), "all three prediagnostic input witnesses")
        initial_ok = numerical and all(r["passed"] for r in before)
        require(self.manifest["initial_numerical_gate_passed"] == initial_ok, "declared initial gate versus raw")
        self.initial_ok, self.inputs, self.initial_scores = initial_ok, inputs, initial_scores
        self.report.update(initial_numerical_gate_passed=initial_ok, inputs=inputs)

    def check_diagnostic(self):
        timing = [r for r in self.records if r["kind"] == "stability_timing"]
        schedule = expected_schedule()
        require(len(timing) <= 468, "bounded diagnostic cohort")
        numerical = self.initial_ok
        predecessor, previous_slot = -1, -1
        groups = {}
        host = []
        for sequence, r in enumerate(timing):
            round_index, order, pi, step = schedule[sequence]
            name, conditioner, arms, deferred, distinct = PHASES[pi]
            provider = conditioner if step == 0 else arms[step - 1]
            slot = step if distinct else provider
            require((r["sequence"], r["round"], r["phase_order"], r["phase_index"], r["step"]) ==
                    (sequence, round_index, order, pi, step) and r["phase"] == name and
                    r["role"] == (["conditioner", "A1", "B", "A2"][step]) and r["provider"] == provider and
                    r["output_slot"] == slot and r["previous_provider"] == predecessor and r["previous_output_slot"] == previous_slot and
                    core.harness_flag(r["warmup"]) == (round_index == 0) and r["deferred_validation"] == deferred and
                    r["distinct_outputs"] == distinct and r["adoption_cohort"] is False and r["fullmodel_speedup_proven"] is False,
                    "exact schema3 schedule/provider/slot/predecessor")
            predecessor, previous_slot = provider, slot
            stem = f"round-{round_index:02d}-phase-{pi}-step-{step}"
            require(r["scores_raw"] == stem + "-scores.f32" and r["guards_raw"] == stem + "-guards.bin", "one complete snapshot per CB")
            data, guard = self.artifact(r["scores_raw"]), self.artifact(r["guards_raw"])
            facts = self.facts(data, guard)
            ok = data == self.initial_scores[provider] and facts["guards"] and all(facts["stats"][k] for k in ("visible_finite", "causal_tail_valid", "all_written"))
            require(r["scores_sha256"] == core.sha(data) and r["guards_sha256"] == core.sha(guard) and
                    r["mismatches_to_actual_capture"] == facts["actual"]["mismatches"] and
                    r["mismatches_to_tiled"] == facts["tiled"]["mismatches"] and r["validation_passed"] == ok and
                    r["guards_intact"] == facts["guards"] and all(r[k] == facts["stats"][k] for k in ("visible_finite", "causal_tail_valid", "all_written")), "all CB outputs and validation records versus raw")
            numerical &= ok
            begin, end = r["gpu_start_s"], r["gpu_end_s"]
            require(math.isfinite(begin) and end > begin > 0 and abs((end - begin) * 1e6 - r["gpu_us"]) < 1e-6, "GPU timestamp arithmetic")
            keys = ("host_poison_begin_s", "host_poison_end_s", "host_commit_s", "host_wait_completed_s", "host_snapshot_completed_s")
            clocks = [r[k] for k in keys]
            require(all(math.isfinite(value) for value in clocks) and clocks == sorted(clocks), "ordered monotonic host poison/commit/wait/snapshot events")
            vb, ve = r["validation_host_begin_s"], r["validation_host_end_s"]
            require(math.isfinite(vb) and math.isfinite(ve) and ve >= vb >= clocks[-1], "validation starts after retained snapshot")
            if sequence:
                require(begin >= timing[sequence - 1]["gpu_end_s"] and clocks[0] >= timing[sequence - 1]["host_snapshot_completed_s"], "serialized CB/host sequence")
            groups.setdefault((round_index, pi), []).append(r)
            host.append(dict(sequence=sequence, round=round_index, phase=name, step=step, provider=provider,
                gpu_us=r["gpu_us"], gap_before_us=(begin - timing[sequence - 1]["gpu_end_s"]) * 1e6 if sequence else None,
                poison_us=(clocks[1] - clocks[0]) * 1e6, encoding_us=(clocks[2] - clocks[1]) * 1e6,
                commit_wait_us=(clocks[3] - clocks[2]) * 1e6, snapshot_us=(clocks[4] - clocks[3]) * 1e6,
                validation_us=(ve - vb) * 1e6))
        for (round_index, pi), group in groups.items():
            if len(group) != 4:
                continue
            if PHASES[pi][3]:
                require(all(r["validation_host_begin_s"] >= group[3]["host_snapshot_completed_s"] for r in group), "deferred validation after all four snapshots")
                require(all(a["validation_host_end_s"] <= b["validation_host_begin_s"] for a, b in zip(group, group[1:])), "deferred validation step order")
            else:
                require(all(a["validation_host_end_s"] <= b["host_poison_begin_s"] for a, b in zip(group, group[1:])), "inline full validation before next CB")
        after = [r for r in self.inputs if r["phase"] == "diagnostic_after"]
        complete = len(timing) == 468 and len(after) == 3 and {r["input"] for r in after} == set(core.INPUTS)
        numerical &= all(r["passed"] for r in self.inputs)
        self.report.update(timing_records=len(timing), complete_numerical_snapshots_passed=numerical, complete_finite_cohort=complete,
                           host_phase_records=host, scores_logical_bytes_checked=sum(self.raw[r["scores_raw"]].stat().st_size for r in timing))
        if self.manifest["status"] == "diagnostic_complete_not_adoption":
            require(complete and numerical and self.window["exit_code"] == 0, "successful diagnostic execution requires full raw/count/input gates")
        else:
            require(self.manifest["status"] == "failed" and self.window["exit_code"] != 0, "known failed diagnostic outcome")
        require(self.consumed == set(self.raw), "every retained raw file consumed by independent reader")
        if not numerical:
            self.report.update(outcome="diagnostic_numerical_gate_failed_no_timing_interpretation", measurement_correction_supported=False)
        elif not complete:
            self.report.update(outcome="diagnostic_execution_incomplete_no_timing_interpretation", measurement_correction_supported=False)
        else:
            summaries, decision = summary_and_decision(timing, self.protocol["decision_rule"])
            self.report.update(phase_summaries=summaries, diagnostic_decision=decision,
                measurement_correction_supported=decision["measurement_correction_supported"],
                outcome="diagnostic_supports_new_measurement_cohort_not_adoption" if decision["measurement_correction_supported"] else "diagnostic_rule_not_satisfied_no_next_stage")

    def run(self):
        self.check_frozen()
        self.check_window_and_load()
        self.check_initial_and_inputs()
        self.check_diagnostic()
        self.report.update(review_completed=True, fullmodel_eligible=False, production_adoption=False,
                           old_joint_gate_remains_failed=True,
                           scope="controlled predecessor plus host-observer intervention; no unique frequency/cache attribution")
        return self.report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "capture-verified", "build-manifest", "config", "probe-library", "native-library", "window-dir", "baseline-identity", "reference-stdout", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--capture-root", type=Path)
    p.add_argument("--result-dir", type=Path)
    p.add_argument("--jsonl", type=Path)
    p.add_argument("--window-wrapper", type=Path)
    p.add_argument("--path-map", action="append", default=[])
    args = p.parse_args()
    require(not args.out.exists(), "new diagnostic review output")
    review = None
    code = 0
    try:
        review = StabilityReview(args)
        report = review.run()
    except (core.EvidenceError, KeyError, OSError, ValueError, TypeError, IndexError, StopIteration) as error:
        report = review.report if review else dict(schema_version=1)
        report.update(review_completed=False, outcome="evidence_invalid", evidence_error=str(error), fullmodel_eligible=False,
                      production_adoption=False, old_joint_gate_remains_failed=True)
        code = 2
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report.get(k) for k in ("review_completed", "outcome", "measurement_correction_supported", "fullmodel_eligible", "evidence_error")}, indent=2))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
