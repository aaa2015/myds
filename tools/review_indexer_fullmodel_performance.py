#!/usr/bin/env python3
"""Independent CPU P reader, after both long inputs qualify; no GPU actions.

This separate file leaves the frozen Q0 and Qlong readers untouched. All raw
state/logits/tokens/output and provider/window gates precede performance gates.
Exact rational arithmetic uses the native JSON decimal timings, with no epsilon,
trimming, retries, pooled workloads or replacement of any formal sample.
"""
import argparse
from decimal import Decimal
from fractions import Fraction
import json
from pathlib import Path
import struct

import review_indexer_fullmodel_v2 as core
import review_indexer_fullmodel_long as qualification

CORE_SHA = "f20b70993ca97adaaf84529122732f71abce1cdc58a087b2bad8ef1c2f2d4dcd"
QUALIFICATION_READER_SHA = "2a803c71deb0d0e6cad98bc302cf03b6a07f79918e428326c6fd60e5f359a806"
PREDECLARED_GATES_SHA = "80c60a99919051aab9d39dc411c9923bf899af305b5552d7343c93cd6c344e7b"
WORKLOADS = qualification.WORKLOADS
PROVIDERS = (("A1", 0), ("B", 1), ("A2", 0))
METRICS = ("prefill_s", "ttft_s", "decode_ms_per_token")
require, EvidenceError = core.require, core.EvidenceError
read_json, file_sha, fnv1a = core.read_json, core.file_sha, core.fnv1a
integer, loaded_library_identity = core.integer, core.loaded_library_identity
effective_configuration = core.effective_configuration


def schedule():
    return [dict(workload=w, provider=p) for w, _ in WORKLOADS for p, _ in PROVIDERS]


def checked_order(rows):
    order = [dict(workload=r["workload"], provider=r["provider"]) for r in rows]
    require(order and len(order) <= 6 and order == schedule()[:len(order)], "one fixed two-workload A1/B/A2 cohort without retry/replacement")
    return order


def exact_fraction(value):
    require(type(value) in (int, Decimal, str), "native decimal timing, not an already rounded binary float")
    decimal = Decimal(value)
    require(decimal.is_finite() and decimal > 0, "finite positive exact native decimal timing")
    return Fraction(decimal)


def metric_gate(a1, b, a2, primary):
    a1, b, a2 = (exact_fraction(v) for v in (a1, b, a2))
    drift = 100 * (a2 / a1 - 1)
    saved = 100 * (1 - b / min(a1, a2))
    regression = -saved
    passed = abs(drift) <= 1 and (saved > 2 if primary else regression <= Fraction(1, 2))
    rationals = {k: dict(numerator=v.numerator, denominator=v.denominator) for k, v in
                 (("signed_control_drift_percent", drift), ("conservative_saved_percent", saved), ("regression_percent", regression))}
    return dict(passed=bool(passed), control_drift_passed=abs(drift) <= 1,
        strict_primary_savings_passed=saved > 2 if primary else None,
        nonregression_passed=regression <= Fraction(1, 2) if not primary else None,
        a1_median=float(a1), b_median=float(b), a2_median=float(a2), signed_control_drift_percent=float(drift),
        conservative_saved_percent=float(saved), regression_percent=float(regression), exact_rationals=rationals)


def performance_summary(samples_by_arm):
    require(set(samples_by_arm) == {name for name, _ in PROVIDERS}, "all three original P arms")
    medians, raw = {}, {}
    for arm, samples in samples_by_arm.items():
        require([s["run"] for s in samples] == [0, 1, 2, 3] and [s["warmup"] for s in samples] == [True, False, False, False],
                "exact warmup0 and formal1/2/3, no silently substituted sample")
        medians[arm], raw[arm] = {}, {}
        for metric in METRICS:
            values = [(exact_fraction(s["exact_native_metrics"][metric]), s["exact_native_metrics"][metric]) for s in samples[1:]]
            medians[arm][metric] = sorted(values)[1][1]
            raw[arm][metric] = [s["exact_native_metrics"][metric] for s in samples]
    metrics = {}
    for metric in METRICS:
        values = [medians[arm][metric] for arm in ("A1", "B", "A2")]
        metrics[metric] = metric_gate(*values, primary=metric == "prefill_s")
    return dict(passed=all(row["passed"] for row in metrics.values()), metrics=metrics,
                native_decimal_samples_including_warmup=raw, warmup_excluded_from_medians=True, formal_runs=[1, 2, 3],
                gate_arithmetic="Exact Fraction of original JSON decimal timings; no epsilon or rounded percent comparisons")


def performance_decision(numerical, groups):
    complete = len(groups) == 2 and {g["workload"] for g in groups} == {name for name, _ in WORKLOADS}
    passed = bool(numerical and complete and all(g["passed"] is True for g in groups))
    return dict(numerical_gates_passed=bool(numerical), qualification_passed=bool(numerical), performance_evaluated=bool(numerical),
        native_fullmodel_gate_passed=passed, native_prefill_speedup_proven=passed, performance_acceptance=passed,
        native_performance_scope="ordinary native engine prefill_s; TTFT/decode nonregression; two frozen long inputs, same private executable/library",
        eligible_to_prepare_HTTP_validation=passed, next_stage_automatic_authorization=False, stage_next_requires_separate_freeze=True,
        production_acceptance=False, production_adoption=False, deployed_executable_wire_bridge_completed=False,
        fullmodel_speedup_proven=False, HTTP_speedup_proven=False)


def protocol_gate_consistency(protocol_gate, frozen_gate):
    require(protocol_gate == frozen_gate, "P protocol gate dictionary including formulas equals predeclared frozen artifact")


def prerequisite_gate(report, descriptor, spec, protocol, binary_sha):
    require(spec["report_sha_locked_in_execution_plan"] is True and descriptor["path"] == spec["report_path"] and
            report["review_completed"] is True and report["qualification_passed"] is True and
            report["long_workload_qualification_completed"] is True and report["P_eligible"] is True and
            report["formal_performance_cohort_eligible"] is True and report["execution_complete"] is True and
            report["native_generation_count"] == 12 and len(report["providers"]) == 6 and
            report["protocol_sha256"] == spec["protocol_sha256"] == protocol["artifacts"]["qualification_protocol"]["sha256"] and
            report["reader_sha256"] == spec["reader_sha256"] == QUALIFICATION_READER_SHA and
            report["core_reader_sha256"] == CORE_SHA and report["binary_sha256"] == binary_sha and
            report["private_build_manifest_sha256"] == protocol["artifacts"]["private_build_manifest"]["sha256"] and
            report["target_build_manifest_sha256"] == protocol["artifacts"]["target_build_manifest"]["sha256"] and
            report["workload_identities"] == protocol["workloads"] and report["model_identity_scope"] == protocol["model"] and
            report["production_acceptance"] is False, "completed both-long-input qualification from exact same inputs/source/binary/model/core")
    rows = report["workload_comparisons"]
    require(len(rows) == 2 and {r["workload"] for r in rows} == {name for name, _ in WORKLOADS} and
            all(all(r[name] is True for name in ("qualification_passed", "gold_library_bridge_passed", "candidate_reference_passed", "within_provider_repeatability_passed")) for r in rows),
            "every Qlong workload numerical/G_lib/candidate/repeatability gate passed before P")


def read_run_stdout(path, expected_prompt_tokens, expected_output_tokens=64):
    lines = Path(path).read_text().splitlines()
    require(lines and all(line.strip() and line.startswith("{") for line in lines), "strict original P JSONL only")
    records = [json.loads(line, object_pairs_hook=core.unique_object) for line in lines]
    require([row["kind"] for row in records] == ["indexer_metadata", "metadata", "load"] + ["sample"] * 4,
            "exact original metadata/load/warmup0/formal1/2/3 with no omission or replacement")
    indexer, metadata, loaded, *samples = records
    require(metadata["subject"] == "prefill_indexer_k_resident" and metadata["protocol_version"] == 3 and
            metadata["repeats"] == 3 and metadata["warmup_runs"] == 1 and metadata["max_new_tokens"] == expected_output_tokens and
            metadata["temperature"] == 0 and metadata["top_p"] == 1 and metadata["top_k"] == 0 and metadata["min_p"] == 0 and
            metadata["repetition_penalty"] == 1 and metadata["seed"] == 42 and metadata["qos"] == "user_interactive" and
            metadata["qos_result"] == 0 and metadata["prompt_renderer"] == "parse_chat_prompt" and metadata["clock"] == "CLOCK_MONOTONIC" and
            metadata["transport"] == "direct_byte_callback" and metadata["decode_basis"] == "first_to_last_callback/(tokens-1)" and
            metadata["dspark_model"] is None, "ordinary unchanged direct greedy P driver")
    require(loaded["native_v41"] is True and loaded["layers"] == 40 and loaded["dspark_loaded"] is False and
            loaded["fresh_context_per_sample"] is True and loaded["cache_configuration_preserved"] is True and
            integer(loaded["vocab_size"]) and loaded["vocab_size"] > 0, "native V4.1 full vocabulary/fresh-context P model")
    exact_records = [json.loads(line, object_pairs_hook=core.unique_object, parse_float=Decimal) for line in lines]
    for run, (sample, exact) in enumerate(zip(samples, exact_records[3:])):
        core.sample_contract(sample, run, expected_prompt_tokens, expected_output_tokens)
        require(type(sample["sample_valid"]) is bool and type(sample["output_equal_to_warmup"]) is bool and sample["gpu_indexer_provider"]["run"] == run,
                "native P declarations and exact provider run identity")
        sample["exact_native_metrics"] = {name: str(exact[name]) for name in METRICS}
        for value in sample["exact_native_metrics"].values():
            exact_fraction(value)
    return indexer, metadata, loaded, samples


def read_run_exports(directory, stdout_path, mode, prompt_tokens, output_tokens, payload_reader=None):
    directory = Path(directory)
    indexer, metadata, loaded, samples = read_run_stdout(stdout_path, prompt_tokens, output_tokens)
    logits, state = directory / "logits", directory / "state"
    require(not logits.is_symlink() and not state.is_symlink() and logits.is_dir() and state.is_dir() and
            {p.name for p in logits.iterdir()} == {f"run-{i}.f32" for i in range(4)} and
            {p.name for p in state.iterdir()} == {f"run-{i}" for i in range(4)}, "complete raw run0/1/2/3 roots")
    payload_reader = payload_reader or core.PayloadReader()
    exports, reports = {}, []
    for sample in samples:
        run = sample["run"]
        witness = core.provider_witness(sample["gpu_indexer_provider"], mode)
        export = payload_reader.read_export(state / f"run-{run}", logits / f"run-{run}.f32", sample, loaded["vocab_size"])
        require(export["decoded_scalars"]["max_seq_len"] == metadata["max_seq_len"] and
                export["decoded_scalars"]["max_new_tokens"] == metadata["max_new_tokens"] and
                export["decoded_scalars"]["seed_u32"] == metadata["seed"] and export["metadata"]["next_position"] <= metadata["max_seq_len"],
                "P final state matches fixed context/output/seed")
        exports[run] = export
        reports.append(dict(run=run, warmup=sample["warmup"], native_sample_valid=sample["sample_valid"],
            native_output_equal_to_warmup=sample["output_equal_to_warmup"], provider=witness,
            raw_files=export["evidence"], manifest_sha256=export["manifest_sha256"], raw_bytes=export["payload_bytes"],
            final_logits_finite=export["finite_logits"], exact_native_metrics=sample["exact_native_metrics"],
            native_timing_descriptive_only={name: sample[name] for name in METRICS + ("decode_s", "wall_s")}))
    return dict(indexer_metadata=indexer, metadata=metadata, loaded=loaded, samples=reports, exports=exports,
        native_declared_valid=all(s["sample_valid"] is True and s["output_equal_to_warmup"] is True for s in samples),
        stdout_sha256=file_sha(stdout_path), sample_contracts_passed=True, provider_adoption_passed=True)


def cross_arm_comparisons(exports):
    require(set(exports) == {name for name, _ in PROVIDERS} and all(set(runs) == set(range(4)) for runs in exports.values()),
            "all twelve full P payload sets per workload")
    reference = exports["A1"][0]
    rows = [dict(provider=arm, run=run, **core.compare_exports(reference, value)) for arm, runs in exports.items()
            for run, value in runs.items() if (arm, run) != ("A1", 0)]
    return dict(passed=all(row["passed"] for row in rows), reference_provider="A1", reference_run=0,
                full_payload_sets=12, comparisons=rows)


class Review(qualification.Review):
    def __init__(self, args):
        super().__init__(args)
        self.execution_reader_sha = file_sha(__file__)
        self.report.update(reader="independent_indexer_fullmodel_performance_reader", reader_sha256=file_sha(__file__),
            reader_revision="1_P_only", qualification_reader_sha256=file_sha(qualification.__file__), stage="P",
            scope="two_long_inputs_same_binary_library_A1_B_A2_warmup1_formal3_native_prefill", P_eligible=False,
            native_fullmodel_gate_passed=False, performance_evaluated=False)

    def check_predeclared_gates(self, p):
        path = self.artifacts["predeclared_p_gates"]
        require(file_sha(path) == p["artifacts"]["predeclared_p_gates"]["sha256"] == PREDECLARED_GATES_SHA,
                "P gates frozen before any Qlong/P GPU data")
        g = read_json(path); gate = g["performance_gate"]
        protocol_gate_consistency(p["performance_gate"], gate)
        require(g["kind"] == "indexer_P_gates_frozen_before_any_P_and_QLong_GPU_data" and g["root_approved_before_data"] is True and
                g["no_retuning_from_QLong_or_P_results"] is True and g["request_contract"] == p["request_contract"] and
                g["workload_artifacts"] == {w["name"]: w["artifacts"] for w in p["workloads"]} and
                all(r["binary"] == g["binary"] and r["library"] == g["library"] for r in p["providers"]), "all P frozen inputs/sampling/bin/lib match prospective gates artifact")
        require(gate["all_workloads_and_all_nonregression_gates_required"] is True and gate["formal_median_samples_per_arm"] == 3 and
                gate["warmup_run"] == 0 and gate["formal_runs"] == [1, 2, 3] and gate["primary_metric"] == "prefill_s" and
                gate["primary_min_saved_percent_strict"] == 2 and gate["max_abs_control_drift_percent_inclusive"] == 1 and
                gate["nonregression_metrics"] == ["ttft_s", "decode_ms_per_token"] and gate["max_nonregression_percent_inclusive"] == .5 and
                gate["total_native_generations"] == 24 and gate["total_warmup_generations"] == 6 and gate["total_formal_generations"] == 18,
                "unchanged strict prefill>2, absdrift<=1, TTFT/decode regression<=.5 and all24 generations")
        self.report["predeclared_performance_gate"] = gate
        self.report["predeclared_gates_sha256"] = file_sha(path)
    def initialize(self):
        p = self.full_protocol = self.protocol = read_json(self.args.protocol)
        require(p["schema_version"] == 1 and p["kind"] == "indexer_fullmodel_p_performance_protocol" and p["stage"] == "P" and
                [(w["name"], w["prompt_tokens"]) for w in p["workloads"]] == list(WORKLOADS) and
                [(r["name"], r["mode"]) for r in p["providers"]] == list(PROVIDERS), "fixed P-only two-input/six-process performance cohort")
        c = p["request_contract"]
        expected = dict(output_tokens=64, max_seq_len=327680, warmup_runs=1, temperature=0, top_p=1, top_k=0,
            min_p=0, repetition_penalty=1, seed=42, token_format="little_endian_int32_raw", renderer="parse_chat_prompt",
            tokenizer="target_gguf_metadata", repeats=3, formal_runs=3, performance_generation_count=24)
        require(all(c[k] == v for k, v in expected.items()) and p["performance_acceptance_by_collector"] is False and
                p["production_acceptance"] is False and p["fullmodel_speedup_proven"] is False and p["no_retry_or_replacement"] is True and
                p["no_automatic_next_stage"] is True and p["classification"]["G_lib_is_deployed_executable_or_source"] is False and
                p["classification"]["deployed_executable_wire_bridge_completed"] is False and
                p["correctness"]["whole_model_payload_sha256_verified"] is False and p["correctness"]["selected_symbols"] == list(core.SYMBOLS),
                "one warm/three formal/64 outputs; collector cannot confer performance or production acceptance")
        require(p["output_layout"]["provider_order"] == [name for name, _ in PROVIDERS] and
                p["output_layout"]["stdout_kinds"] == ["indexer_metadata", "metadata", "load"] + ["sample"] * 4,
                "P protocol explicitly describes all four native sample records and original arm order")
        self.artifacts = {name: self.descriptor(item) for name, item in p["artifacts"].items()}
        require(file_sha(core.__file__) == p["artifacts"]["core_reader"]["sha256"] == CORE_SHA and
                file_sha(__file__) == p["artifacts"]["reader"]["sha256"] and
                file_sha(qualification.__file__) == p["artifacts"]["qualification_reader"]["sha256"] == QUALIFICATION_READER_SHA and
                self.artifacts["private_build_manifest"].resolve() == self.args.private_build_manifest.resolve() and
                self.artifacts["target_build_manifest"].resolve() == self.args.target_build_manifest.resolve() and
                self.mapped(p["window_root"]).resolve() == self.args.window_root.resolve(), "actual frozen core/long reader/source/window arguments")
        proof = core.verify_private_build(self.args.private_build_manifest, self.mapped)
        self.private = proof.pop("private_build")
        self.check_target_build()
        for row in p["providers"]:
            self.descriptor(row["binary"]); self.descriptor(row["library"])
        require(all(r["binary"]["sha256"] == self.private["binary"]["sha256"] for r in p["providers"]) and
                p["providers"][1]["library"]["sha256"] == p["providers"][2]["library"]["sha256"] == self.private["library"]["sha256"],
                "identical private executable and unchanged complete-TU off/on library")
        baseline = read_json(self.artifacts["baseline_identity"])
        require(p["model"] == baseline["model"] and all(name in baseline and all(item[k] == baseline[name][k] for k in ("bytes", "sha256"))
                for name, item in p["canonical_files"].items()), "preserved canonical/model identity")
        require(all(r["library"]["sha256"] == self.private["library"]["sha256"] for r in p["providers"]), "all P arms use same approved private library")
        qp = read_json(self.artifacts["qualification_protocol"])
        require(qp["kind"] == "indexer_fullmodel_qlong_qualification_protocol" and qp["workloads"] == p["workloads"] and
                qp["model"] == p["model"] and qp["wrapper_sha256"] == p["wrapper_sha256"], "P uses qualified inputs/model and same uninstrumented collection mechanics")
        self.check_predeclared_gates(p)
        self.workload_inputs = {}
        for w in p["workloads"]:
            artifacts = {name: self.descriptor(item) for name, item in w["artifacts"].items()}
            request, prompt, tokens = (artifacts[key].read_bytes() for key in ("request", "rendered_prompt", "rendered_tokens"))
            tokenizer = read_json(artifacts["tokenizer"])
            require(b"\0" not in request and b"\0" not in prompt and len(tokens) == w["prompt_tokens"] * 4 and
                    tokenizer["prompt_tokens"] == w["prompt_tokens"] and tokenizer["prompt_bytes"] == len(prompt) and
                    tokenizer["request_bytes"] == len(request) and tokenizer["prompt_renderer"] == c["renderer"] and
                    tokenizer["tokenizer"] == c["tokenizer"] and tokenizer["token_encoding"] == "little_endian_int32" and
                    tokenizer["add_bos"] is True and struct.unpack("<i", tokens[:4])[0] == tokenizer["bos_id"] and
                    tokenizer["model_loader_called"] is False and tokenizer["metal_initialized"] is False, "complete long request/rendered prompt/token identities")
            self.workload_inputs[w["name"]] = dict(artifacts=artifacts, request=request, prompt=prompt)
        prior_path = self.mapped(p["prerequisite"]["report_path"])
        self.prerequisite_descriptor = dict(path=p["prerequisite"]["report_path"], bytes=prior_path.stat().st_size, sha256=file_sha(prior_path))
        self.descriptor(self.prerequisite_descriptor)
        self.prerequisite = read_json(prior_path)
        prerequisite_gate(self.prerequisite, self.prerequisite_descriptor, p["prerequisite"], p, self.private["binary"]["sha256"])
        self.report.update(protocol_sha256=file_sha(self.args.protocol), private_build=proof, binary_sha256=self.private["binary"]["sha256"],
            private_build_manifest_sha256=file_sha(self.args.private_build_manifest), target_build_manifest_sha256=file_sha(self.args.target_build_manifest),
            prerequisite_report_sha256=file_sha(prior_path), prerequisite_protocol_sha256=p["prerequisite"]["protocol_sha256"],
            workload_identities=p["workloads"], provider_library_sha256={r["name"]: r["library"]["sha256"] for r in p["providers"]})

    def check_window_common(self, window, allowed_abort=None):
        p = self.protocol
        require(window["protocol_sha256"] == file_sha(self.args.protocol) and window["wrapper_sha256"] == p["wrapper_sha256"] and
                window["stage"] == "P" and window["abort_reason"] in (None, allowed_abort) and window["canonical_unchanged"] is True and
                window["canonical_before"] == window["canonical_after"] and window["model_unchanged"] is True and
                window["model_before"] == window["model_after"] and not window["recognized_before"] and not window["recognized_after"],
                "unchanged exclusive P canonical/model window")
        require(set(window["canonical_before"]) == set(p["canonical_files"]), "all frozen canonical artifacts observed")
        for name, item in p["canonical_files"].items():
            self.check_snapshot(window["canonical_before"][name], item)
        self.check_snapshot(window["model_before"], p["model"], model=True)

    def review_provider(self, provider):
        p = self.protocol
        directory = self.args.window_root / provider["name"]
        require({path.name for path in directory.iterdir()} == set(p["output_layout"]["provider_files"]) |
                set(p["output_layout"]["provider_directories"]), "complete original provider evidence directory")
        plan, window, monitor = (read_json(directory / name) for name in ("execution-plan.json", "window.json", "monitor.json"))
        self.check_window_common(window, allowed_abort="native nonzero exit")
        expected_argv = [provider["binary"]["path"], "--config", p["artifacts"]["config"]["path"], "--run", p["model"]["path"],
                         p["artifacts"]["request"]["path"], "3", "64", "327680"]
        expected_root = Path(p["window_root"]) / provider["name"]
        expected_env = dict(MYDS_INDEXER_K_RESIDENT=str(provider["mode"]), MYDS_BENCH_LOGITS_DIR=str(expected_root / "logits"),
                            MYDS_BENCH_STATE_DIR=str(expected_root / "state"))
        require(plan["stage"] == "P" and plan["workload"] == window["workload"] == self.workload["name"] and plan["provider"] == window["provider"] == provider["name"] and
                plan["mode"] == window["mode"] == provider["mode"] and plan["argv"] == expected_argv and
                plan["cwd"] == str(expected_root / "cwd") and plan["MYDS_inherited"] == expected_env and
                plan["protocol_sha256"] == file_sha(self.args.protocol) and plan["wrapper_sha256"] == p["wrapper_sha256"] and
                plan["reader_sha256"] == window["reader_sha256"] == self.execution_reader_sha and
                plan["prerequisite_report_sha256"] == window["prerequisite_report_sha256"] == self.report["prerequisite_report_sha256"] and
                plan["expected_loaded_library_path"] == window["expected_loaded_library_path"] == provider["expected_loaded_library_path"] and
                plan["binary_sha256"] == provider["binary"]["sha256"] and plan["library_sha256"] == provider["library"]["sha256"] and
                plan["target_build_manifest_sha256"] == file_sha(self.args.target_build_manifest) and
                plan["private_build_manifest_sha256"] == file_sha(self.args.private_build_manifest) and
                plan["request_sha256"] == p["artifacts"]["request"]["sha256"] and plan["config_sha256"] == p["artifacts"]["config"]["sha256"] and
                plan["config_loaded_in_child_override_env"] is True and plan["effective_MYDS_count_expected"] == 65 and
                plan["adoption_cohort"] is True and plan["performance_acceptance"] is False,
                "actual argv/mode/environment/source/bin/lib/config/request and prelaunch reader binding")
        require(plan["timeout_s"] == p["timeout_s"] and plan["monitor_interval_s"] == p["monitor_interval_s"] and
                plan["shared_lock_path"] == p["lock_path"] and plan["exclusive_recognized_process_check"] is True and
                plan["started_epoch"] == window["started_epoch"] <= window["finished_epoch"] and
                0 <= window["elapsed_s"] <= p["timeout_s"] + 20, "bounded exclusive native process window")
        child = window["child_pid"]
        require(integer(child) and child > 0 and monitor and
                all(all(row["pid"] == child for row in item["recognized"]) for item in monitor) and
                any(any(row["pid"] == child and Path(row["comm"]).name == "bench_indexer_fullmodel" for row in item["recognized"]) for item in monitor),
                "actual child observed and no recognized inference competitors")
        require((window["exit_code"] == 0 and window["abort_reason"] is None) or
                (type(window["exit_code"]) is int and window["exit_code"] != 0 and window["abort_reason"] == "native nonzero exit"),
                "successful child or preserved native execution failure")
        inventory = self.check_inventory(directory, window)
        result = dict(workload=self.workload["name"], provider=provider["name"], mode=provider["mode"], child_pid=child, exit_code=window["exit_code"],
            started_epoch=window["started_epoch"], finished_epoch=window["finished_epoch"], window_gates_passed=True,
            raw_inventory=inventory, execution_complete=window["exit_code"] == 0,
            evidence_sha256={name: file_sha(directory / name) for name in ("execution-plan.json", "window.json", "monitor.json")},
            actual_argv=plan["argv"], window=window)
        try:
            parsed = read_run_exports(directory, directory / "stdout.jsonl", provider["mode"], self.workload["prompt_tokens"], 64, self.payload_reader)
            configuration = effective_configuration(parsed["indexer_metadata"], parsed["metadata"], self.artifacts["config"].read_bytes(), provider["mode"],
                expected_root / "logits", expected_root / "state", p["artifacts"]["config"]["path"], p["config_contract"]["file_entries"])
            metadata, loaded = parsed["metadata"], parsed["loaded"]
            require(metadata["model"] == p["model"]["path"] and metadata["request_file"] == p["artifacts"]["request"]["path"] and
                    metadata["max_seq_len"] == p["request_contract"]["max_seq_len"] and metadata["request_bytes"] == len(self.request) and
                    metadata["prompt_bytes"] == len(self.prompt) and metadata["request_fnv1a64"] == fnv1a(self.request) and
                    metadata["prompt_fnv1a64"] == fnv1a(self.prompt), "actual native request/model/rendered prompt identity")
            actual_library = loaded["actual_metallib_path"]
            loaded_library_identity(actual_library, provider["expected_loaded_library_path"], provider["library"]["path"],
                                    provider["library"]["sha256"], self.mapped)
            result.update(samples=parsed["samples"], configuration=configuration, metadata=metadata, loaded=loaded,
                native_declared_valid=parsed["native_declared_valid"], provider_adoption_passed=True, loaded_library_verified=True,
                numerical_evidence_complete=True)
            return result, parsed["exports"]
        except (EvidenceError, KeyError, OSError, ValueError, TypeError, IndexError) as error:
            if window["exit_code"] == 0:
                raise
            result.update(numerical_evidence_complete=False, native_execution_or_applicability_failure=True,
                          native_failure_evidence_error=str(error), provider_adoption_passed=False, loaded_library_verified=False)
            return result, None

    def run(self):
        self.initialize()
        p, root = self.full_protocol, self.args.window_root
        require(file_sha(self.mapped(p["wrapper_path"])) == p["wrapper_sha256"], "frozen collection-only long wrapper")
        plan, window = (read_json(root / name) for name in ("execution-plan.json", "window.json"))
        order = checked_order(window["providers"])
        require(plan["stage"] == window["stage"] == "P" and plan["run_order"] == window["run_order"] == schedule() and
                plan["protocol_sha256"] == file_sha(self.args.protocol) and plan["wrapper_sha256"] == p["wrapper_sha256"] and
                plan["reader_sha256"] == window["reader_sha256"] == file_sha(__file__) and
                plan["shared_lock_path"] == p["lock_path"] and plan["timeout_s"] == p["timeout_s"] and
                plan["performance_acceptance"] is False and plan["prerequisite_report"] == window["prerequisite_report"] == self.prerequisite_descriptor and
                plan["started_epoch"] >= self.prerequisite["finished_epoch"], "prelaunch fixed long order, prerequisite report, reader, protocol, lock binding")
        expected_workloads = {row["workload"] for row in order}
        require({path.name for path in root.iterdir()} == {"execution-plan.json", "window.json", "model-prefix.bin"} | expected_workloads,
                "one outer evidence root without unregistered retries")
        for name in expected_workloads:
            require({path.name for path in (root / name).iterdir()} == {row["provider"] for row in order if row["workload"] == name},
                    "one provider directory per frozen workload/role")
        last = order[-1]
        self.check_window_common(window, "provider collection stopped: " + last["workload"] + "/" + last["provider"])
        require(window["lock_held_through_final_checks"] is True and window["service_or_production_mutation_performed"] is False and
                window["external_reader_required"] is True and window["qualification_passed"] is None and window["fullmodel_speedup_proven"] is False and
                window["production_acceptance"] is False and window["performance_acceptance"] is False and window["no_automatic_next_stage"] is True,
                "collector completion is not numerical/performance/production acceptance")
        prefix = self.descriptor(window["model_prefix"])
        require(window["model_prefix"]["path"] == p["model_prefix_path"] and prefix.stat().st_size == p["model"]["prefix_bytes"] == 1048576 and
                file_sha(prefix) == p["model"]["prefix_sha256"], "complete retained model identity prefix")
        results, exports = [], {name: {} for name, _ in WORKLOADS}
        self.report["providers"] = results
        outer_args_root = self.args.window_root
        try:
            for item in order:
                w = next(w for w in p["workloads"] if w["name"] == item["workload"])
                provider = next(r for r in p["providers"] if r["name"] == item["provider"])
                inputs = self.workload_inputs[w["name"]]
                self.workload = w
                self.protocol = dict(p, artifacts=dict(p["artifacts"], **w["artifacts"]),
                    request_contract=dict(p["request_contract"], prompt_tokens=w["prompt_tokens"]),
                    window_root=str(Path(p["window_root"]) / w["name"]))
                self.artifacts = dict(self.artifacts, **inputs["artifacts"])
                self.request, self.prompt = inputs["request"], inputs["prompt"]
                self.args.window_root = outer_args_root / w["name"]
                result, payloads = self.review_provider(provider)
                require(result["window"] == window["providers"][len(results)], "outer/child long window records identical")
                require(plan["started_epoch"] <= result["started_epoch"] <= result["finished_epoch"] <= window["finished_epoch"], "bounded long-stage child chronology")
                if results:
                    require(result["child_pid"] not in {r["child_pid"] for r in results} and result["started_epoch"] >= results[-1]["finished_epoch"],
                            "six distinct serial fresh native processes")
                results.append(result)
                if payloads is not None:
                    exports[w["name"]][provider["name"]] = payloads
        finally:
            self.args.window_root, self.protocol = outer_args_root, p
        complete = len(results) == 6 and all(r["execution_complete"] for r in results) and window["abort_reason"] is None
        require(window["execution_complete"] is complete, "outer/child exact six-process completion")
        comparisons = []
        for name, _ in WORKLOADS:
            if len(exports[name]) == 3:
                comparisons.append(dict(workload=name, **cross_arm_comparisons(exports[name])))
        native_valid = all(r.get("native_declared_valid") is True for r in results)
        numerical = complete and native_valid and len(comparisons) == 2 and all(r["passed"] for r in comparisons)
        groups = []
        if numerical:
            for name, _ in WORKLOADS:
                samples = {row["provider"]: row["samples"] for row in results if row["workload"] == name}
                groups.append(dict(workload=name, **performance_summary(samples)))
        decision = performance_decision(numerical, groups)
        self.report.update(**decision, review_completed=True,
            outcome=("P_native_prefill_gate_passed_no_production_acceptance" if decision["native_fullmodel_gate_passed"] else
                     "P_performance_gate_failed_no_retry" if numerical else "P_numerical_or_execution_gate_failed_performance_not_evaluated"),
            execution_complete=complete, workload_numerical_comparisons=comparisons, performance_workloads=groups,
            native_generation_count=sum(len(r.get("samples", [])) for r in results),
            warmup_generations=sum(s["warmup"] for r in results for s in r.get("samples", [])),
            formal_generations=sum(not s["warmup"] for r in results for s in r.get("samples", [])),
            raw_inventory_files=sum(r["raw_inventory"]["files"] for r in results), raw_logical_bytes=sum(r["raw_inventory"]["logical_bytes"] for r in results),
            outer_window_gates_passed=True, model_prefix_sha256=file_sha(prefix), model_identity_scope=p["model"], whole_model_payload_sha256_verified=False,
            window_evidence_sha256={name: file_sha(root / name) for name in ("execution-plan.json", "window.json")},
            prerequisite_qualification_passed=True, old_R2_joint_gate_remains_failed=True,
            schema3_468_primary_rule_remains_failed=True, literal720_remains_descriptive=True,
            finished_epoch=window["finished_epoch"])
        return self.report


def evidence_failure(report, error):
    report = core.evidence_failure(report, error)
    report.update(outcome="P_evidence_invalid_no_performance_acceptance", native_fullmodel_gate_passed=False,
        native_prefill_speedup_proven=False, performance_evaluated=False, eligible_to_prepare_HTTP_validation=False,
        P_eligible=False, HTTP_speedup_proven=False)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "window-root", "private-build-manifest", "target-build-manifest", "out"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--path-map", action="append", default=[])
    args = p.parse_args()
    require(not args.out.exists(), "new independent P report")
    run, code = None, 0
    try:
        run = Review(args); report = run.run()
    except (EvidenceError, KeyError, OSError, ValueError, TypeError, IndexError, StopIteration) as error:
        report = evidence_failure(run.report if run else None, error); code = 2
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report.get(k) for k in ("review_completed", "outcome", "numerical_gates_passed", "performance_evaluated",
        "native_fullmodel_gate_passed", "native_prefill_speedup_proven", "production_acceptance", "evidence_error")}, indent=2))
    raise SystemExit(code)


if __name__ == "__main__":
    main()

