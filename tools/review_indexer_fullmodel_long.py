#!/usr/bin/env python3
"""CPU-only Qlong qualification of two frozen inputs; no P performance logic.

All numerical, schema2 state, counters, aliases and artifact helpers come from
the explicitly frozen revision2 core. Q0 data/tools remain immutable. A passing
report qualifies preparation of a separate prospective P stage, never adoption.
"""
import argparse
import json
from pathlib import Path
import struct

import review_indexer_fullmodel_v2 as core

CORE_SHA = "f20b70993ca97adaaf84529122732f71abce1cdc58a087b2bad8ef1c2f2d4dcd"
WORKLOADS = (("business96", 29609), ("code38404", 38404))
PROVIDERS = (("G_lib", 0), ("off", 0), ("on", 1))
require, EvidenceError = core.require, core.EvidenceError
read_json, file_sha, fnv1a = core.read_json, core.file_sha, core.fnv1a
integer, loaded_library_identity = core.integer, core.loaded_library_identity
read_run_exports, effective_configuration = core.read_run_exports, core.effective_configuration


def schedule():
    return [dict(workload=w, provider=p) for w, _ in WORKLOADS for p, _ in PROVIDERS]


def checked_order(rows):
    observed = [dict(workload=r["workload"], provider=r["provider"]) for r in rows]
    require(bool(observed) and len(observed) <= 6 and observed == schedule()[:len(observed)],
            "one original two-workload/six-provider prefix without retry or replacement")
    return observed


def qualification_decision(workloads, execution_complete, native_valid):
    complete = len(workloads) == 2 and {row["workload"] for row in workloads} == {name for name, _ in WORKLOADS}
    numerical = complete and all(row["qualification_passed"] is True for row in workloads)
    passed = bool(execution_complete and native_valid and numerical)
    return dict(qualification_passed=passed, long_workload_qualification_completed=passed,
        formal_performance_cohort_eligible=passed, P_eligible=passed, numerical_gates_passed=numerical,
        stage_next_requires_separate_freeze=True, next_stage_automatic_authorization=False,
        performance_acceptance=False, fullmodel_speedup_proven=False, production_acceptance=False,
        production_adoption=False, deployed_executable_wire_bridge_completed=False)


def prerequisite_gate(report, descriptor, spec, binary_sha):
    require(descriptor == spec["report"] and report["review_completed"] is True and report["qualification_passed"] is True and
            report["gold_library_bridge_passed"] is True and report["candidate_reference_passed"] is True and
            report["within_provider_repeatability_passed"] is True and report["protocol_sha256"] == spec["protocol_sha256"] and
            report["reader_sha256"] == spec["reader_sha256"] == CORE_SHA and report["binary_sha256"] == binary_sha and
            report["long_workload_qualification_completed"] is False and report["production_acceptance"] is False,
            "completed independent Q0 report, binary, reader/protocol identity prebound before long stage")


class Review(core.Review):
    def __init__(self, args):
        super().__init__(args)
        self.report.update(reader="independent_indexer_fullmodel_qlong_reader", reader_sha256=file_sha(__file__),
            reader_revision="1_Qlong_only", core_reader_sha256=file_sha(core.__file__), stage="Qlong",
            scope="two_long_inputs_same_private_binary_G_lib_off_on_final_logits_schema2_state", P_eligible=False)
        self.execution_reader_sha = file_sha(__file__)

    def initialize(self):
        p = self.full_protocol = self.protocol = read_json(self.args.protocol)
        require(p["schema_version"] == 1 and p["kind"] == "indexer_fullmodel_qlong_qualification_protocol" and p["stage"] == "Qlong" and
                [(w["name"], w["prompt_tokens"]) for w in p["workloads"]] == list(WORKLOADS) and
                [(r["name"], r["mode"]) for r in p["providers"]] == list(PROVIDERS), "fixed Qlong-only two-input/six-process qualification")
        c = p["request_contract"]
        expected = dict(output_tokens=64, max_seq_len=327680, warmup_runs=1, temperature=0, top_p=1, top_k=0,
            min_p=0, repetition_penalty=1, seed=42, token_format="little_endian_int32_raw", renderer="parse_chat_prompt",
            tokenizer="target_gguf_metadata", repeats=1, verification_runs=1, qualification_generation_count=12)
        require(all(c[k] == v for k, v in expected.items()) and p["performance_acceptance_by_collector"] is False and
                p["production_acceptance"] is False and p["fullmodel_speedup_proven"] is False and p["no_retry_or_replacement"] is True and
                p["no_automatic_next_stage"] is True and p["classification"]["G_lib_is_deployed_executable_or_source"] is False and
                p["classification"]["deployed_executable_wire_bridge_completed"] is False and
                p["correctness"]["whole_model_payload_sha256_verified"] is False and p["correctness"]["selected_symbols"] == list(core.SYMBOLS),
                "one warm/one verification/64 outputs and no performance/production claim")
        self.artifacts = {name: self.descriptor(item) for name, item in p["artifacts"].items()}
        require(file_sha(core.__file__) == p["artifacts"]["core_reader"]["sha256"] == CORE_SHA and
                file_sha(__file__) == p["artifacts"]["reader"]["sha256"] and
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
        deployed = [v for n, v in p["canonical_files"].items() if Path(n).name == "myds.metallib"]
        require(len(deployed) == 1 and deployed[0]["sha256"] == p["providers"][0]["library"]["sha256"], "G_lib deployed-library anchor")
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
        prior_path = self.descriptor(p["prerequisite"]["report"])
        self.prerequisite = read_json(prior_path)
        prerequisite_gate(self.prerequisite, p["prerequisite"]["report"], p["prerequisite"], self.private["binary"]["sha256"])
        self.report.update(protocol_sha256=file_sha(self.args.protocol), private_build=proof, binary_sha256=self.private["binary"]["sha256"],
            private_build_manifest_sha256=file_sha(self.args.private_build_manifest), target_build_manifest_sha256=file_sha(self.args.target_build_manifest),
            prerequisite_report_sha256=file_sha(prior_path), prerequisite_protocol_sha256=p["prerequisite"]["protocol_sha256"],
            workload_identities=p["workloads"], provider_library_sha256={r["name"]: r["library"]["sha256"] for r in p["providers"]})

    def check_window_common(self, window, allowed_abort=None):
        p = self.protocol
        require(window["protocol_sha256"] == file_sha(self.args.protocol) and window["wrapper_sha256"] == p["wrapper_sha256"] and
                window["stage"] == "Qlong" and window["abort_reason"] in (None, allowed_abort) and window["canonical_unchanged"] is True and
                window["canonical_before"] == window["canonical_after"] and window["model_unchanged"] is True and
                window["model_before"] == window["model_after"] and not window["recognized_before"] and not window["recognized_after"],
                "unchanged exclusive Qlong canonical/model window")
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
                         p["artifacts"]["request"]["path"], "1", "64", "327680"]
        expected_root = Path(p["window_root"]) / provider["name"]
        expected_env = dict(MYDS_INDEXER_K_RESIDENT=str(provider["mode"]), MYDS_BENCH_LOGITS_DIR=str(expected_root / "logits"),
                            MYDS_BENCH_STATE_DIR=str(expected_root / "state"))
        require(plan["stage"] == "Qlong" and plan["workload"] == window["workload"] == self.workload["name"] and plan["provider"] == window["provider"] == provider["name"] and
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
                plan["adoption_cohort"] is False and plan["performance_acceptance"] is False,
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
        require(plan["stage"] == window["stage"] == "Qlong" and plan["run_order"] == window["run_order"] == schedule() and
                plan["protocol_sha256"] == file_sha(self.args.protocol) and plan["wrapper_sha256"] == p["wrapper_sha256"] and
                plan["reader_sha256"] == window["reader_sha256"] == file_sha(__file__) and
                plan["shared_lock_path"] == p["lock_path"] and plan["timeout_s"] == p["timeout_s"] and
                plan["performance_acceptance"] is False and plan["prerequisite_report"] == window["prerequisite_report"] == p["prerequisite"]["report"] and
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
                comparisons.append(dict(workload=name, **core.qualification_comparisons(exports[name])))
        native_valid = all(r.get("native_declared_valid") is True for r in results)
        decision = qualification_decision(comparisons, complete, native_valid)
        self.report.update(**decision, review_completed=True,
            outcome="Qlong_both_inputs_qualified_P_requires_separate_freeze" if decision["qualification_passed"] else "Qlong_qualification_failed_P_prohibited",
            execution_complete=complete, workload_comparisons=comparisons, native_generation_count=sum(len(r.get("samples", [])) for r in results),
            raw_inventory_files=sum(r["raw_inventory"]["files"] for r in results), raw_logical_bytes=sum(r["raw_inventory"]["logical_bytes"] for r in results),
            outer_window_gates_passed=True, model_prefix_sha256=file_sha(prefix), model_identity_scope=p["model"], whole_model_payload_sha256_verified=False,
            window_evidence_sha256={name: file_sha(root / name) for name in ("execution-plan.json", "window.json")},
            old_R2_joint_gate_remains_failed=True, schema3_468_primary_rule_remains_failed=True, literal720_remains_descriptive=True,
            finished_epoch=window["finished_epoch"])
        return self.report


def evidence_failure(report, error):
    report = core.evidence_failure(report, error)
    report.update(outcome="Qlong_evidence_invalid_P_prohibited", long_workload_qualification_completed=False, P_eligible=False,
                  formal_performance_cohort_eligible=False, qualified_to_prepare_separate_long_protocol=False)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "window-root", "private-build-manifest", "target-build-manifest", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--path-map", action="append", default=[])
    args = p.parse_args()
    require(not args.out.exists(), "new independent Qlong report")
    run, code = None, 0
    try:
        run = Review(args); report = run.run()
    except (EvidenceError, KeyError, OSError, ValueError, TypeError, IndexError, StopIteration) as error:
        report = evidence_failure(run.report if run else None, error); code = 2
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report.get(k) for k in ("review_completed", "outcome", "qualification_passed", "long_workload_qualification_completed",
        "P_eligible", "production_acceptance", "evidence_error")}, indent=2))
    raise SystemExit(code)


if __name__ == "__main__":
    main()

