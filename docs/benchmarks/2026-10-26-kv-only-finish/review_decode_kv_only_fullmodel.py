#!/usr/bin/env python3
"""CPU-only independent decode KV-only Q/P evidence reader.

This is a new subject. Unlike prefill-indexer adoption, ordinary scalar decode
must encode the candidate at each of 32 non-index-source layers on every one
of N target forwards. N includes the last-prompt forward producing callback1.
No model, device, subprocess, service or remote job is launched by this reader.
"""
import argparse
import array
from decimal import Decimal
from fractions import Fraction
import json
import math
from pathlib import Path
import struct
import sys

try:
    import review_indexer_fullmodel_v2 as core
except ModuleNotFoundError:
    # Frozen dependency lives in the retained indexer evidence tree when this
    # reader is installed at the independent decode validation root.
    for location in ("/private/tmp/myds-indexer-r3-20261025/source/tools", "/tmp/myds-indexer-r3-local-20261025/source/tools"):
        if (Path(location)/"review_indexer_fullmodel_v2.py").is_file():
            sys.path.insert(0,location);break
    import review_indexer_fullmodel_v2 as core

CORE_SHA = "f20b70993ca97adaaf84529122732f71abce1cdc58a087b2bad8ef1c2f2d4dcd"
PRIVATE_BUILD_SHA = "6b4e04fa67914aec0bd8636b298a0aa832349e2de4dd496c6a7de0623e55a82d"
PRIVATE_SOURCE_SHA = "3c41008d2b886e61ee2641254f3504e7bc555cfdc0dda0fcb9d4f48910fbe3db"
PRIVATE_LIBRARY_SHA = "d12ddb1ec806f1daa3848e9687e468aba3ea80d1e91aa3d4081c16eb4852facd"
GOLD_LIBRARY_SHA = "3837102396703da59e0eebbc0de1fdc58b68b3a3a5ec5d07504b351c67a29ba9"
GATES_SHA = "79ebeabdfa3f51465f774153f37473079141f7c1687c2e99b9f30ffd42a758f2"
SUBJECT = "decode_kv_only_finish"
STATE_SCOPE = "fresh_context_final_state_after_prefill_and_decode"
SYMBOL = "kernel_myds_probe_kv_only_finish"
SELECTOR = "MYDS_DECODE_KV_ONLY_FINISH"
COUNTERS = ("requests", "eligible", "encoded", "references", "fallbacks", "failures")
SPEC_COUNTERS = ("batch2_calls", "batch2_hits", "batch2_misses", "batch2_failures", "batch2_fallbacks",
    "depth2_calls", "depth2_hits1", "depth2_hits2", "depth2_misses", "depth2_failures", "depth2_fallbacks",
    "draft_cycles", "draft_proposed", "draft_accepted", "draft_rejects", "confidence_rejects", "margin_rejects", "backoff_steps",
    "pld_queries", "pld_cycles", "pld_proposed", "pld_accepted")
METRICS = ("decode_ms_per_token", "ttft_s", "prefill_s")
require, EvidenceError = core.require, core.EvidenceError
read_json, file_sha, fnv1a, digest = core.read_json, core.file_sha, core.fnv1a, core.digest
integer, regular_child, state_layout = core.integer, core.regular_child, core.state_layout
SCALARS, STATE_METADATA, LIMITATIONS = core.SCALARS, core.STATE_METADATA, core.LIMITATIONS


def no_spec_telemetry(row, count):
    expected_keys = set(SPEC_COUNTERS) | {"target_forward_calls", "target_rows", "output_tokens", "cycle_count", "timing_enabled",
        "recorded_cycles", "draft_ms", "verify_ms", "rollback_ms", "snapshot_ms", "pld_ms", "cycle_ms", "cycle_output_tokens"}
    require(set(row) == expected_keys and integer(count) and count >= 2 and
            all(type(row[name]) is int and row[name] == count for name in ("target_forward_calls", "target_rows", "output_tokens", "cycle_count")) and
            all(type(row[name]) is int and row[name] == 0 for name in SPEC_COUNTERS) and
            row["timing_enabled"] is False and type(row["recorded_cycles"]) is int and row["recorded_cycles"] == 0 and
            all(row[name] is None for name in ("draft_ms", "verify_ms", "rollback_ms", "snapshot_ms", "pld_ms")) and
            row["cycle_ms"] == row["cycle_output_tokens"] == [], "real ordinary scalar no-spec telemetry for every target forward/output")
    return dict(no_spec_verified=True, target_forwards=count, target_rows=count, output_tokens=count, cycle_count=count,
                speculative_counters_all_zero=True, profiling_disabled=True)


def provider_witness(row, expected_mode, count, telemetry):
    fields={"api_available","mode","run","adoption_valid","selected_symbol","observed_callbacks","expected_calls_per_scalar_forward",
        "no_spec_source_predicate_and_telemetry_verified","first_callback_scope","decode_scope","counters_count_encoding_not_gpu_completion"}
    fields|={prefix+name for prefix in ("","through_first_callback_","decode_")for name in COUNTERS}
    require(set(row)==fields and type(row["run"])is int and row["run"]>=0 and integer(count) and count >= 2 and
            type(expected_mode)is int and expected_mode in (0, 1) and row["api_available"] is True and
            type(row["mode"]) is int and row["mode"] == expected_mode and row["adoption_valid"] is True and
            type(row["observed_callbacks"]) is int and row["observed_callbacks"] == count and
            type(row["expected_calls_per_scalar_forward"]) is int and row["expected_calls_per_scalar_forward"] == 32 and
            row["no_spec_source_predicate_and_telemetry_verified"] is True and
            row["first_callback_scope"] == "chunk_prefill_plus_last_prompt_scalar_forward_through_first_output_callback" and
            row["decode_scope"] == "N_minus_1_output_intervals_ordinary_scalar_decode" and
            row["counters_count_encoding_not_gpu_completion"] is True, "real decode API, actual callback count and ordinary scalar witness scopes")
    no_spec = no_spec_telemetry(telemetry, count)
    total, first = count * 32, 32
    expected_total = dict(requests=total, eligible=total if expected_mode else 0, encoded=total if expected_mode else 0,
                          references=0 if expected_mode else total, fallbacks=0, failures=0)
    expected_first = dict(requests=first, eligible=first if expected_mode else 0, encoded=first if expected_mode else 0,
                          references=0 if expected_mode else first, fallbacks=0, failures=0)
    for name in COUNTERS:
        observed = row[name], row["through_first_callback_" + name], row["decode_" + name]
        require(all(type(v) is int and v >= 0 for v in observed) and observed ==
                (expected_total[name], expected_first[name], expected_total[name] - expected_first[name]),
                "exact32-per-forward total/first/N-1 decode provider field: " + name)
    require(row["selected_symbol"] == (SYMBOL if expected_mode else None), "single exact candidate symbol, reference selected-symbol null")
    return dict(counters={key: row[key] for key in row}, no_spec=no_spec,
                expected_total_calls=total, expected_first_calls=first, expected_decode_calls=total-first)


def exact_fraction(value):
    require(type(value) in (int, Decimal, str), "original native decimal timing, not binary float")
    value = Decimal(value)
    require(value.is_finite() and value > 0, "positive finite native decimal time")
    return Fraction(value)


def metric_gate(a1, b, a2, primary):
    a1, b, a2 = (exact_fraction(value) for value in (a1, b, a2))
    drift, saved = 100*(a2/a1-1), 100*(1-b/min(a1,a2))
    regression = -saved
    passed = abs(drift) <= 1 and (saved > 2 if primary else regression <= Fraction(1,2))
    return dict(passed=bool(passed), control_drift_passed=abs(drift)<=1,
        strict_decode_savings_passed=saved>2 if primary else None, nonregression_passed=regression<=Fraction(1,2) if not primary else None,
        a1_median=float(a1), b_median=float(b), a2_median=float(a2), signed_control_drift_percent=float(drift),
        conservative_saved_percent=float(saved), regression_percent=float(regression),
        exact_rationals={key:dict(numerator=value.numerator,denominator=value.denominator) for key,value in
            (("signed_control_drift_percent",drift),("conservative_saved_percent",saved),("regression_percent",regression))})


def performance_summary(samples_by_arm):
    require(set(samples_by_arm)=={"A1","B","A2"}, "three original decode P arms")
    medians, raw = {}, {}
    for arm, samples in samples_by_arm.items():
        require([r["run"] for r in samples]==[0,1,2,3] and [r["warmup"] for r in samples]==[True,False,False,False],
                "original warmup0/formal1,2,3 without replacement")
        medians[arm], raw[arm] = {}, {}
        for metric in METRICS:
            values=[(exact_fraction(r["exact_native_metrics"][metric]),r["exact_native_metrics"][metric]) for r in samples[1:]]
            medians[arm][metric]=sorted(values)[1][1]
            raw[arm][metric]=[r["exact_native_metrics"][metric] for r in samples]
    metrics={metric:metric_gate(*(medians[arm][metric] for arm in ("A1","B","A2")),primary=metric=="decode_ms_per_token") for metric in METRICS}
    return dict(passed=all(r["passed"] for r in metrics.values()),metrics=metrics,native_decimal_samples_including_warmup=raw,
                warmup_excluded_from_medians=True,formal_runs=[1,2,3],gate_arithmetic="Exact Fraction of native JSON decimals; no epsilon")


class DecodePayloadReader(core.PayloadReader):
    """Checked copy of frozen read_export, changing subject and scope only."""
    def read_export(self, directory, logits_path, sample, vocab_size, snapshot_slots=44):
        root = Path(directory)
        m = read_json(root / "manifest.json")
        require(set(m) == {"schema_version", "subject", "run", "scope", "metadata", "limitations", "files", "file_count", "complete"} and
                m["schema_version"] == 2 and m["subject"] == "decode_kv_only_finish" and m["run"] == sample["run"] and
                m["scope"] == "fresh_context_final_state_after_prefill_and_decode" and m["complete"] is True and m["limitations"] == LIMITATIONS,
                "complete unchanged schema2 final live/published-snapshot state contract")
        require(set(m["metadata"]) == STATE_METADATA and all(set(d) == {"path", "bytes", "fnv1a64_bytes"} for d in m["files"]),
                "no silently ignored state metadata or payload fields")
        descriptors = {d["path"]: d for d in m["files"]}
        require(len(descriptors) == len(m["files"]) == m["file_count"], "unique complete manifest payload list")
        require({p.name for p in root.iterdir()} == set(descriptors) | {"manifest.json"}, "directory exactly matches full state manifest")
        files, evidence, arrays = {}, {}, {}
        for name, d in descriptors.items():
            path, data, item = self.payload(root, d)
            files[name], evidence[name] = path, item
            if name in ("context-scalars.u64le", "anchor-lengths.u32le", "snapshot-positions.u32le", "engram-tail.u32le", "tokens.i32le"):
                width = 8 if name == "context-scalars.u64le" else 4
                require(len(data) % width == 0, "whole little-endian scalar/token array")
                code = "Q" if width == 8 else "i" if name == "tokens.i32le" else "I"
                arrays[name] = list(struct.unpack("<" + code * (len(data) // width), data))
        require(all(name in arrays for name in ("context-scalars.u64le", "anchor-lengths.u32le", "snapshot-positions.u32le", "engram-tail.u32le", "tokens.i32le")),
                "all structural state arrays are present")
        scalars = dict(zip(SCALARS, arrays["context-scalars.u64le"]))
        expected = state_layout(arrays["context-scalars.u64le"], arrays["anchor-lengths.u32le"], arrays["snapshot-positions.u32le"],
                                sample["output_bytes"], sample["output_tokens"], snapshot_slots)
        require(set(descriptors) == set(expected) and all(descriptors[name]["bytes"] == size for name, size in expected.items()),
                "derived complete final-state/active-owner/all-published-snapshot file set and extents")
        md = m["metadata"]
        require(md["byte_order"] == "little" and md["fresh_context"] is True and md["layers"] == 40 and md["window_width"] == 512 and
                md["window_order"] == "chronological_active_rows" and md["compressed_row_bytes"] == 1024 and md["index_row_bytes"] == 256 and
                md["anchor_layout"] == "per_slot_active_tokens" and md["scalar_order"] == list(SCALARS) and
                all(md[name] == scalars[name] for name in ("ctx_pos", "next_position", "engram_position", "anchor_slots", "anchor_cap", "anchor_n")) and
                md["window_rows"] == min(scalars["next_position"], 128) and md["rng_state"] == f"{scalars['rng_state']:016x}" and
                md["engram_tail"] == arrays["engram-tail.u32le"] and scalars["max_new_tokens"] == sample["output_tokens"],
                "metadata matches independently decoded state and frozen output count")
        require(arrays["tokens.i32le"] == sample["token_ids"] and len(arrays["tokens.i32le"]) == sample["output_tokens"] and
                all(0 <= token < vocab_size for token in arrays["tokens.i32le"]), "raw full token IDs match callbacks and vocabulary")
        require(evidence["tokens.i32le"]["fnv1a64_bytes"] == sample["token_ids_fnv1a64"] and
                evidence["output.bin"]["fnv1a64_bytes"] == sample["output_bytes_fnv1a64"], "native callback byte/token FNV cross-check")
        logits_path = Path(logits_path)
        require(not logits_path.is_symlink() and logits_path.is_file(), "retained final full-vocabulary logits")
        logits = logits_path.read_bytes()
        require(len(logits) == vocab_size * 4 and sample["final_logits"]["elements"] == vocab_size, "complete final logits extent")
        values = array.array("f"); values.frombytes(logits)
        if sys.byteorder != "little":
            values.byteswap()
        finite = all(math.isfinite(v) for v in values)
        h = digest(logits); key = len(logits), h
        if key not in self.fnv_cache:
            self.fnv_cache[key] = fnv1a(logits)
        require(sample["final_logits"]["finite"] is finite and sample["final_logits"]["fnv1a64_bytes"] == self.fnv_cache[key],
                "raw final logits finite/FNV declaration")
        files["final-logits.f32"] = logits_path
        evidence["final-logits.f32"] = dict(bytes=len(logits), sha256=h, fnv1a64_bytes=self.fnv_cache[key], finite=finite)
        return dict(files=files, evidence=evidence, metadata=md, decoded_scalars=scalars, manifest_sha256=file_sha(root / "manifest.json"),
                    complete=True, finite_logits=finite, payload_bytes=sum(e["bytes"] for e in evidence.values()))


def effective_configuration(outer,metadata,raw,mode,logits_dir,state_dir,config_path,execute=True,expected_count=62):
    base,count=core.config_settings(raw)
    require(count==expected_count and SELECTOR not in base and "MYDS_INDEXER_K_RESIDENT" not in base and
            not ({"MYDS_BENCH_LOGITS_DIR","MYDS_BENCH_STATE_DIR"}&set(base)),"original complete config does not override selectors/evidence sinks")
    expected=dict(base,MYDS_DECODE_KV_ONLY_FINISH=str(mode),MYDS_INDEXER_K_RESIDENT="0",
                  MYDS_BENCH_LOGITS_DIR=str(logits_dir),MYDS_BENCH_STATE_DIR=str(state_dir))
    require(outer["kind"]=="decode_kv_metadata" and outer["schema_version"]==1 and outer["subject"]==SUBJECT and
            outer["execute_requested"] is execute and outer["provider_adapter_available"] is True and
            outer["fresh_context_per_sample"] is True and outer["cache_configuration_preserved"] is True and
            outer["config_load_override_env"] is True and outer["config_file"]==str(config_path) and
            outer["config_loaded_entries"]==count and outer["config_file_bytes"]==len(raw) and
            outer["config_file_fnv1a64"]==fnv1a(raw) and outer["config_raw"].encode()==raw and
            outer["effective_configuration_count"]==len(expected) and outer["effective_configuration"]==expected and
            type(outer["runtime_mode"]) is int and outer["runtime_mode"]==mode and
            outer["requested_symbol"]==(SYMBOL if mode else None),"all66 effective settings, exact runtime mode/single symbol and original config")
    if metadata is not None:
        require({k:v for k,v in metadata["environment"].items() if v is not None}==expected,"base and outer metadata agree on every effective MYDS key")
    require(all(expected[k]=="1" for k in ("MYDS_PREFIX_CACHE","MYDS_PLD","MYDS_METAL_CONCURRENT")) and
            expected["MYDS_BATCH2_SPEC"]==expected["MYDS_DSPARK"]=="0", "unchanged cache/PLD/concurrent and no-spec configuration")
    forbidden={key for key in expected if key.startswith(("MYDS_PROFILE","MYDS_TRACE","MYDS_SKIP_"))}
    forbidden|=set(expected)&{"MYDS_KPROF","MYDS_PHASE_PROFILE","MYDS_PHASE_PROFILE_FILE","MYDS_BATCH_TRACE_PREFIX",
        "MYDS_PROF","MYDS_DISPATCH_HIST","MYDS_DEBUG_DISPATCH","MYDS_MOE_STEP_PROF","MYDS_DISABLE_V41_PREFILL",
        "MYDS_DSPARK_CAPTURE_BYPASS","MYDS_ENGINE_CAPTURE_BYPASS","MYDS_BENCH_STORAGE_COUNTERS","MYDS_DECODE_METRICS"}
    require(not forbidden and expected.get("MYDS_BATCH2_TRACE_GPU","0")=="0" and expected.get("MYDS_SPEC_SERIAL_PROBE","0")=="0",
            "no profiler/trace/spec serial/capture/skip/storage observer")
    return dict(config_entries=count,effective_entries=len(expected),effective=expected,original_config_sha256=digest(raw),
                no_unlisted_MYDS=True,indexer_candidate_fixed_off=True,normal_async_selectors=True)


def read_run_stdout(path,prompt_tokens,count,repeats,mode):
    require(repeats in (1,3),"separately frozen Q1 verification or P3 formal samples")
    lines=Path(path).read_text().splitlines()
    require(lines and all(line.strip() and line.startswith("{") for line in lines),"strict native decode JSONL only")
    records=[json.loads(line,object_pairs_hook=core.unique_object)for line in lines]
    require([r["kind"] for r in records]==["decode_kv_metadata","metadata","load"]+["sample"]*(repeats+1),
            "all native metadata/load/warmup/verification-or-three-formal records in original order")
    outer,metadata,loaded,*samples=records
    require(metadata["subject"]==SUBJECT and metadata["protocol_version"]==3 and metadata["repeats"]==repeats and
            metadata["warmup_runs"]==1 and metadata["max_new_tokens"]==count and metadata["temperature"]==0 and metadata["top_p"]==1 and
            metadata["top_k"]==metadata["min_p"]==0 and metadata["repetition_penalty"]==1 and metadata["seed"]==42 and
            metadata["qos"]=="user_interactive" and metadata["qos_result"]==0 and metadata["prompt_renderer"]=="parse_chat_prompt" and
            metadata["clock"]=="CLOCK_MONOTONIC" and metadata["transport"]=="direct_byte_callback" and
            metadata["decode_basis"]=="first_to_last_callback/(tokens-1)" and metadata["dspark_model"] is None,
            "new-subject unchanged ordinary greedy decode driver and exact N")
    require(loaded["native_v41"] is True and loaded["layers"]==40 and loaded["dspark_loaded"] is False and
            loaded["fresh_context_per_sample"] is True and loaded["cache_configuration_preserved"] is True and
            integer(loaded["vocab_size"]) and loaded["vocab_size"]>0,"native V4.1/fresh context/no draft model")
    exact=[json.loads(line,object_pairs_hook=core.unique_object,parse_float=Decimal)for line in lines]
    for run,(sample,decimals) in enumerate(zip(samples,exact[3:])):
        core.sample_contract(sample,run,prompt_tokens,count)
        require(sample["subject"]==SUBJECT and "gpu_indexer_provider" not in sample and type(sample["run"])is int and
                type(sample["sample_valid"])is bool and type(sample["output_equal_to_warmup"])is bool and
                type(sample["gpu_decode_kv_only_provider"]["run"])is int and sample["gpu_decode_kv_only_provider"]["run"]==run,
                "actual new decode sample/provider identity and native validity declarations")
        try:
            sample["checked_decode_witness"]=dict(passed=True,**provider_witness(sample["gpu_decode_kv_only_provider"],mode,count,sample["decode_telemetry"]))
        except EvidenceError as error:
            # Truthful failed applicability/telemetry is numerical evidence, not
            # a reason to discard otherwise complete retained raw payloads.
            sample["checked_decode_witness"]=dict(passed=False,evidence_error=str(error),
                declared_provider=sample["gpu_decode_kv_only_provider"],declared_telemetry=sample["decode_telemetry"])
        sample["exact_native_metrics"]={key:str(decimals[key])for key in METRICS}
        for value in sample["exact_native_metrics"].values():exact_fraction(value)
    return outer,metadata,loaded,samples


def read_run_exports(directory,stdout_path,mode,prompt_tokens,count,repeats,payload_reader=None):
    root=Path(directory);outer,metadata,loaded,samples=read_run_stdout(stdout_path,prompt_tokens,count,repeats,mode)
    logits,state=root/"logits",root/"state"
    require(not logits.is_symlink() and not state.is_symlink() and logits.is_dir() and state.is_dir() and
            {p.name for p in logits.iterdir()}=={f"run-{i}.f32"for i in range(repeats+1)} and
            {p.name for p in state.iterdir()}=={f"run-{i}"for i in range(repeats+1)},"complete original decode raw sample roots")
    payload_reader=payload_reader or DecodePayloadReader();exports,reports={},[]
    for sample in samples:
        run=sample["run"];export=payload_reader.read_export(state/f"run-{run}",logits/f"run-{run}.f32",sample,loaded["vocab_size"])
        s=export["decoded_scalars"]
        require(s["max_seq_len"]==metadata["max_seq_len"] and s["max_new_tokens"]==count and s["seed_u32"]==metadata["seed"] and
                export["metadata"]["next_position"]<=metadata["max_seq_len"],"raw decode context/output/seed versus requested parameters")
        exports[run]=export
        reports.append(dict(run=run,warmup=sample["warmup"],native_sample_valid=sample["sample_valid"],native_output_equal_to_warmup=sample["output_equal_to_warmup"],
            provider=sample["checked_decode_witness"],decode_telemetry=sample["decode_telemetry"],raw_files=export["evidence"],
            manifest_sha256=export["manifest_sha256"],raw_bytes=export["payload_bytes"],final_logits_finite=export["finite_logits"],
            exact_native_metrics=sample["exact_native_metrics"],native_timing_descriptive_only={k:sample[k]for k in METRICS+("decode_s","wall_s")}))
    return dict(outer_metadata=outer,metadata=metadata,loaded=loaded,samples=reports,exports=exports,
                native_declared_valid=all(s["sample_valid"] is True and s["output_equal_to_warmup"] is True for s in samples),
                provider_adoption_passed=all(s["checked_decode_witness"]["passed"] is True for s in samples),
                finite_logits_passed=all(s["final_logits_finite"] is True for s in reports))


def verify_runtime_build(path):
    path=Path(path);root=path.parent;b=read_json(path);seed=read_json(root/"seed-manifest.json")
    require(file_sha(path)==PRIVATE_BUILD_SHA and b["source_inventory_sha256"]==PRIVATE_SOURCE_SHA,
            "root-approved immutable decode source/build identity; no unreviewed runtime revision")
    require(b["schema_version"]==1 and b["kind"]=="decode_kv_only_private_runtime_CPU_build" and
            b["CPU_build_only_passed"] is True and b["GPU_work_submitted"] is False and b["remote_or_deployment"] is False and
            b["worktree_production_sources_or_Metal_registration_modified"] is False and b["counters"]==list(COUNTERS) and
            b["symbol"]==SYMBOL and b["driver_subject"]==SUBJECT and b["state_schema"]==2 and b["state_scope"]==STATE_SCOPE and
            b["state_raw_layout_unchanged"] is True and b["driver_actual_callback_count_equals_requested_N_and_N_at_least2"] is True and
            b["no_spec_source_predicate_plus_actual_telemetry_required"] is True,"private decode source/driver/schema/no-spec build contract")
    require(file_sha(root/"seed-manifest.json")==b["seed_manifest_sha256"] and seed["files"]==548 and
            core.inventory_digest(seed["source_inventory"])==seed["inventory_sha256"]==b["seed_inventory_sha256"] and
            core.inventory_digest(b["source_inventory"])==b["source_inventory_sha256"] and len(b["source_inventory"])==556,
            "complete parent548 and new556 source inventory identities")
    core.verify_inventory(root/"source",b["source_inventory"]);core.verify_inventory(root/"seed",seed["source_inventory"])
    before,after=seed["source_inventory"],b["source_inventory"]
    require(set(before)<=set(after) and sorted(set(after)-set(before))==sorted(b["added_paths"]) and
            sorted(name for name in before if before[name]!=after[name])==sorted(b["modified_paths"])==
            ["include/myds_v41_gpu.h","src/myds_v41_gpu.c","src/myds_v41_moe.c"],"only named private API/scoped caller/fixture/driver edits")
    artifacts={}
    for name,item in b["artifacts"].items():
        rel=Path(name);require(not rel.is_absolute() and ".."not in rel.parts,"relative private build artifact")
        target=root/rel;require(target.is_file() and not target.is_symlink() and target.stat().st_size==item["bytes"] and file_sha(target)==item["sha256"],
                                "frozen complete decode build artifact: "+name)
        artifacts[name]=target
    require(b["artifacts"]["bin/bench_decode_kv_only_fullmodel"]["sha256"]==b["artifacts"]["gold-bin/bench_decode_kv_only_fullmodel"]["sha256"],
            "same private CPU executable for G_lib and off/on")
    proof=read_json(artifacts["patch-byteproof.json"])
    require(proof["passed"] is True and proof["forward_patch_equals_complete_derived_inventory"] is True and
            proof["reverse_patch_restores_all548_seed_files_and_removes_added_files"] is True and proof["state_exporter_only_subject_scope_strings_changed"] is True and
            set(proof["reference_functions"])=={"int myds_v41_gpu_qkv_finish(","static int indexscores_chunk("},"complete private source inverse/reference proof")
    old_gpu=(root/"seed/src/myds_v41_gpu.c").read_bytes();new_gpu=(root/"source/src/myds_v41_gpu.c").read_bytes()
    for signature,item in proof["reference_functions"].items():
        old,new=core.complete_c_function(old_gpu,signature),core.complete_c_function(new_gpu,signature)
        require(item["byte_exact"] is True and old==new and digest(old)==item["sha256"],"complete original public qkv/reference scorer unchanged")
    old_state=(root/"source/tools/indexer_fullmodel_state.inc").read_bytes();new_state=(root/"source/tools/decode_kv_only_fullmodel_state.inc").read_bytes()
    require(new_state.replace(SUBJECT.encode(),b"prefill_indexer_k_resident").replace(STATE_SCOPE.encode(),b"fresh_context_full_prefill_final_state")==old_state,
            "decode exporter differs only in subject/scope; all raw state bytes/layout preserved")
    base=(root/"source/tools/bench_output_a_fullmodel.c").read_text();begin=base.index("static bool export_state(");end=base.index("\nint main(",begin)
    changes=[(base[begin:end],'#include "decode_kv_only_fullmodel_state.inc"\n'),
        ('    fprintf(json, "{\\"kind\\":\\"metadata\\",\\"model\\":");','    fprintf(json, "{\\"kind\\":\\"metadata\\",\\"subject\\":\\"decode_kv_only_finish\\",\\"model\\":");'),
        ('        c->first = t;','        c->first = t;\n        bench_decode_kv_first_callback();'),
        ('        bench_outa_get(NULL,true);','        bench_decode_kv_before_sample();\n        bench_outa_get(NULL,true);'),
        ('                           logits_finite && logits_hash == expected_logits;','                           logits_finite && logits_hash == expected_logits && bench_decode_kv_provider_valid(ctx,c.count,reason);'),
        ('        decode_metrics(json, ctx);','        bench_decode_kv_sample_json(json,run,ctx,c.count,reason);\n        decode_metrics(json, ctx);')]
    generated=base
    for old,new in changes:
        require(generated.count(old)==1,"unique exact decode driver evidence hook")
        generated=generated.replace(old,new,1)
    actual=artifacts["final-build/bench_decode_kv_only_fullmodel_base.inc"].read_text()
    require(actual==generated,"full original generation/callback/timing lifecycle plus exactly six declared decode hooks")
    restored=actual
    for old,new in reversed(changes):
        require(restored.count(new)==1,"unique inverse decode driver hook")
        restored=restored.replace(new,old,1)
    require(restored==base,"complete decode driver inverse restores original base bytes")
    return dict(build=b,manifest_sha256=file_sha(path),source_files_verified=556,seed_files_verified=548,
                original_reference_functions_verified=2,raw_state_subject_scope_only=True,driver_inverse_exact=True)


def stage_providers(stage):
    require(stage in ("Q","P"),"independent decode Q or P stage")
    return (("G_lib",0),("off",0),("on",1)) if stage=="Q" else (("A1",0),("B",1),("A2",0))


def stage_order(stage):
    return [dict(workload=w,provider=p)for w in ("short","business2")for p,_ in stage_providers(stage)]


def checked_order(rows,stage):
    observed=[dict(workload=r["workload"],provider=r["provider"])for r in rows]
    require(observed and len(observed)<=6 and observed==stage_order(stage)[:len(observed)],"one fixed short/business2 stage prefix without retry/replacement")
    return observed


def raw_comparisons(exports,stage):
    providers=stage_providers(stage);repeats=1 if stage=="Q" else 3
    require(set(exports)=={name for name,_ in providers} and all(set(runs)==set(range(repeats+1))for runs in exports.values()),
            "all complete warm/verification-or-formal payload sets per workload")
    first=providers[0][0];reference=exports[first][0]
    rows=[dict(provider=arm,run=run,**core.compare_exports(reference,value))for arm,runs in exports.items()for run,value in runs.items()if(arm,run)!=(first,0)]
    result=dict(passed=all(r["passed"]for r in rows),reference_provider=first,reference_run=0,
                full_payload_sets=3*(repeats+1),comparisons=rows)
    if stage=="Q":
        result.update(gold_library_bridge=[core.compare_exports(exports["G_lib"][run],exports["off"][run])for run in range(2)],
                      candidate_reference=[core.compare_exports(exports["off"][run],exports["on"][run])for run in range(2)])
        result.update(gold_library_bridge_passed=all(r["passed"]for r in result["gold_library_bridge"]),
                      candidate_reference_passed=all(r["passed"]for r in result["candidate_reference"]))
    return result


def stage_decision(stage,numerical,performance_workloads):
    require(stage in ("Q","P"),"decode stage decision")
    if stage=="Q":
        return dict(qualification_passed=bool(numerical),N256_two_workload_qualification_completed=bool(numerical),
            formal_performance_cohort_eligible=bool(numerical),performance_evaluated=False,performance_acceptance=False,
            native_decode_gate_passed=False,native_decode_speedup_proven=False,production_acceptance=False,
            deployed_executable_wire_bridge_completed=False,stage_next_requires_separate_freeze=True,next_stage_automatic_authorization=False)
    complete=len(performance_workloads)==2 and {r["workload"]for r in performance_workloads}=={"short","business2"}
    passed=bool(numerical and complete and all(r["passed"] is True for r in performance_workloads))
    return dict(numerical_gates_passed=bool(numerical),performance_evaluated=bool(numerical),performance_acceptance=passed,
        native_decode_gate_passed=passed,native_decode_speedup_proven=passed,eligible_to_prepare_HTTP_validation=passed,
        production_acceptance=False,production_adoption=False,fullmodel_speedup_proven=False,HTTP_speedup_proven=False,
        deployed_executable_wire_bridge_completed=False,stage_next_requires_separate_freeze=True,next_stage_automatic_authorization=False)


def prerequisite_gate(stage,report,spec,protocol,binary_sha):
    require(spec["future_report_exact_descriptor_bound_in_prelaunch_plan"] is True and
            report["review_completed"] is True and report["protocol_sha256"]==spec["protocol_sha256"] and
            report["reader_sha256"]==spec["reader_sha256"],"exact independently completed prerequisite protocol/reader identity")
    if stage=="Q":
        require(report["correctness_passed"] is True and report["numerical_passed"] is True and report["GPU_completion_passed"] is True and
                report["raw_integrity_passed"] is True and report["complete_run"] is True and report["complete_command_buffers"]==342 and
                report["complete_cases"]==114 and report["complete_role_outputs"]==1026 and report["complete_guard_files"]==2052 and
                report["total_raw_files"]==3535 and report["device_name"]==spec["device_name"]=="Apple M2 Ultra" and
                report["timing_evaluated"] is False and report["harness_failure"] is None and
                report["manifest_status"]=="synthetic_correctness_complete_not_adoption" and
                spec["protocol_sha256"]=="76a066e601e9788b24550d0014d21acdc3e4a285d6672c9696a0e19cfb046b00" and
                spec["reader_sha256"]=="7f42684ec6f76e3921d8bb811205f2627c454bab95f06be8ef2eb4cc0f0e555f",
                "actual M2 full component correctness; M4/host-only/failure cannot authorize native Q")
    else:
        require(report["qualification_passed"] is True and report["N256_two_workload_qualification_completed"] is True and
                report["formal_performance_cohort_eligible"] is True and report["execution_complete"] is True and
                report["native_generation_count"]==12 and len(report["providers"])==6 and report["binary_sha256"]==binary_sha and
                report["private_build_manifest_sha256"]==protocol["artifacts"]["private_build_manifest"]["sha256"] and
                report["target_build_manifest_sha256"]==protocol["artifacts"]["target_build_manifest"]["sha256"] and
                report["workload_identities"]==protocol["workloads"] and report["model_identity_scope"]==protocol["model"] and
                report["production_acceptance"] is False and report["subject"]==SUBJECT and report["stage"]=="Q" and
                report["core_reader_sha256"]==CORE_SHA and spec["reader_sha256"]==protocol["artifacts"]["reader"]["sha256"] and
                spec["protocol_sha256"]==protocol["artifacts"]["qualification_protocol"]["sha256"],
                "both N256 decode inputs qualified with same source/bin/model/inputs before P")
        rows=report["workload_numerical_comparisons"]
        require(len(rows)==2 and {r["workload"]for r in rows}=={"short","business2"} and
                all(r["passed"] is True and r["gold_library_bridge_passed"] is True and r["candidate_reference_passed"] is True for r in rows),
                "both decode Q deployed-library and candidate raw bridges passed")


class Review(core.Review):
    def __init__(self,args):
        super().__init__(args)
        self.payload_reader=DecodePayloadReader()
        self.execution_reader_sha=file_sha(__file__)
        self.report.update(reader="independent_decode_kv_only_fullmodel_reader",reader_revision="1_new_decode_Q_P",
            reader_sha256=self.execution_reader_sha,core_reader_sha256=file_sha(core.__file__),subject=SUBJECT,
            scope="two_frozen_inputs_N256_ordinary_async_scalar_decode_final_logits_schema2_state",
            N256_two_workload_qualification_completed=False,native_decode_gate_passed=False,performance_evaluated=False)

    def check_gates(self,p):
        path=self.artifacts["predeclared_p_gates"];g=read_json(path);gate=g["performance_gate"]
        require(file_sha(path)==GATES_SHA and p["performance_gate"]==gate and
            g["kind"]=="decode_KV_only_P_gates_frozen_before_native_Q_or_P_data" and
            g["root_approved_before_native_data"] is True and g["subject"]==SUBJECT and
            g["no_retuning_from_Q_or_P_results"] is True and
            all(p["request_contract"][k]==v for k,v in g["request_contract"].items()) and
            g["request_body_artifacts"]=={w["name"]:w["artifacts"]["request"]for w in p["workloads"]},
            "prospective decode gates, formulas, N256 sampling and original request bodies exactly frozen")
        require(gate["primary_metric"]=="decode_ms_per_token" and gate["primary_min_saved_percent_strict"]==2 and
            gate["control_drift_metrics"]==["prefill_s","ttft_s","decode_ms_per_token"] and
            gate["max_abs_control_drift_percent_inclusive"]==1 and gate["nonregression_metrics"]==["prefill_s","ttft_s"] and
            gate["max_nonregression_percent_inclusive"]==.5 and gate["formal_runs"]==[1,2,3] and
            gate["formal_median_samples_per_arm"]==3 and gate["warmup_run"]==0 and
            gate["all_workloads_and_all_secondary_gates_required"] is True and
            gate["total_native_generations"]==24 and gate["total_warmup_generations"]==6 and gate["total_formal_generations"]==18,
            "strict decode>2, all metric drift<=1, TTFT/prefill regression<=.5; original 24 P generations")
        self.report.update(predeclared_gates_sha256=file_sha(path),predeclared_performance_gate=gate)

    def check_target_build(self):
        b=read_json(self.args.target_build_manifest)
        require(b["schema_version"]==1 and b["kind"]=="M2_private_decode_kv_only_runtime_target_build" and b["subject"]==SUBJECT and
            b["compile_only_pass"] is True and b["gpu_work_submitted"] is False and b["GPU_work_submitted"] is False and
            b["production_deployed"] is False and b["source_inventory_sha256"]==PRIVATE_SOURCE_SHA and
            b["source_build_manifest"]["sha256"]==PRIVATE_BUILD_SHA,"M2 CPU-only build of root-approved complete556 private decode sources")
        for key in ("source_build_manifest","source_check","runtime_core_patch","binary","gold_binary","preflight","binary_minos","helper"):
            self.descriptor(b[key])
        require(set(b["libraries"])=={"private","G_lib"} and set(b["build_logs"])=={"stdout","stderr"},"complete target libraries and build logs")
        for item in list(b["libraries"].values())+list(b["build_logs"].values()):self.descriptor(item)
        require(b["binary"]["sha256"]==b["gold_binary"]["sha256"] and
            b["libraries"]["private"]["sha256"]==PRIVATE_LIBRARY_SHA and b["libraries"]["G_lib"]["sha256"]==GOLD_LIBRARY_SHA and
            b["runtime_core_patch"]["sha256"]==self.private["artifacts"]["runtime-core.patch"]["sha256"],
            "same target executable with approved private/gold complete libraries and runtime patch")
        source=read_json(self.descriptor(b["source_check"]))
        require(source["source_files_verified"]==source["postbuild_source_files_verified"]==556 and
            source["source_inventory_sha256"]==PRIVATE_SOURCE_SHA and source["before_after_source_unchanged"] is True and
            source["complete_source_inventory"]==self.private["source_inventory"] and source["gpu_work_submitted"] is False,
            "complete target source inventory unchanged before/after CPU compilation and preflight")
        command=b["commands"]
        require(len(command)==1 and command[0][:3]==["make","-f","tools/decode_kv_only_fullmodel.mk"] and
            len(command[0])>=10 and command[0][3].startswith("-j") and
            "DECODE_KV_ADAPTER_HEADER=tools/decode_kv_only_runtime_adapter.inc"in command[0] and "DEPLOY_TARGET=26.0"in command[0] and
            command[0][-1]==b["binary"]["path"] and Path(b["command_cwd"]).name=="source" and
            set(b["compile_environment"])=={"DEVELOPER_DIR","SDKROOT","CFLAGS","OBJCFLAGS","LDFLAGS","PYTHONDONTWRITEBYTECODE"} and
            "M2 Ultra"in b["cpu_brand"] and "minos 26.0"in self.descriptor(b["binary_minos"]).read_text(),
            "recorded M2 build/compiler/SDK/flags/deployment target and explicit real decode adapter")
        pf=read_json(self.descriptor(b["preflight"]))
        require(pf["schema_version"]==1 and pf["subject"]==SUBJECT and pf["passed"] is True and pf["gpu_work_submitted"] is False and
            pf["config"]==self.protocol["artifacts"]["config"] and [r["mode"]for r in pf["runs"]]==[0,1],"both target zero-device preflights")
        for r in pf["runs"]:
            stdout=self.descriptor(r["stdout"]);self.descriptor(r["stderr"])
            require(r["returncode"]==0 and r["metadata"]==read_json(stdout) and r["baseline62_values_exact"] is True and
                r["effective66_exact"] is True and r["model_context_or_device_created"] is False and r["gpu_work_submitted"] is False and
                r["argv"]==[b["binary"]["path"],"--config",pf["config"]["path"],"--preflight"],"real target preflight argv/metadata, no model or device")
            root=Path(r["cwd"]).parent
            effective_configuration(r["metadata"],None,self.artifacts["config"].read_bytes(),r["mode"],root/"logits",root/"state",pf["config"]["path"],execute=False)
        self.target=b
        self.report["target_build_verified"]=dict(manifest_sha256=file_sha(self.args.target_build_manifest),source_check=source,commands=b["commands"],preflight_sha256=b["preflight"]["sha256"])

    def initialize(self):
        p=self.full_protocol=self.protocol=read_json(self.args.protocol);self.stage=p["stage"]
        require(self.stage in ("Q","P") and p["schema_version"]==1 and p["status"]=="frozen_ready_for_root_review_no_launch_by_author" and
            p["kind"]==("decode_kv_only_fullmodel_qualification_protocol"if self.stage=="Q"else"decode_kv_only_fullmodel_performance_protocol") and
            p["subject"]==SUBJECT and [w["name"]for w in p["workloads"]]==["short","business2"] and
            all(type(w["prompt_tokens"])is int and w["prompt_tokens"]>0 for w in p["workloads"]) and
            [(r["name"],r["mode"])for r in p["providers"]]==list(stage_providers(self.stage)),
            "final frozen new decode stage, actual target-tokenizer workload lengths and exact provider order")
        c=p["request_contract"];repeats=1 if self.stage=="Q"else 3
        expected=dict(output_tokens=256,max_seq_len=327680,warmup_runs=1,temperature=0,top_p=1,top_k=0,min_p=0,repetition_penalty=1,
            seed=42,stream=False,thinking=False,token_format="little_endian_int32_raw",renderer="parse_chat_prompt",tokenizer="target_gguf_metadata",
            actual_prompt_counts_from_target_not_nominal=True,no_request_length_retuning=True,repeats=repeats,
            total_native_generations=12 if self.stage=="Q"else 24)
        require(all(c[k]==v for k,v in expected.items()) and c["formal_runs"]==(0 if self.stage=="Q"else 3) and
            (self.stage!="Q"or c["verification_runs"]==1) and p["performance_acceptance_by_collector"] is False and
            p["production_acceptance"] is False and p["fullmodel_speedup_proven"] is False and p["no_retry_or_replacement"] is True and
            p["no_automatic_next_stage"] is True and p["classification"]["G_lib_is_deployed_executable_or_source"] is False and
            p["classification"]["deployed_executable_wire_bridge_completed"] is False and
            p["output_layout"]["provider_order"]==[n for n,_ in stage_providers(self.stage)] and
            p["output_layout"]["stdout_kinds"]==["decode_kv_metadata","metadata","load"]+["sample"]*(repeats+1),
            "complete warm/verification-or-formal256-output ordinary greedy stage with no collector acceptance")
        correctness=p["correctness"]
        require(correctness["subject"]==SUBJECT and correctness["state_schema"]==2 and correctness["state_scope"]==STATE_SCOPE and
            correctness["selected_symbol"]==SYMBOL and correctness["counter_names"]==list(COUNTERS) and
            correctness["N256_through_first_decode_final"]==[32,8160,8192] and correctness["whole_model_payload_sha256_verified"] is False and
            all(correctness[k] is True for k in ("all_complete_raw_bytes_required","cross_arm_final_logits_tokens_output_state_exact",
                "all_runs_finite_logits","within_provider_all_warm_and_formal_exact","ordinary_scalar_no_spec_required",
                "all_modes_fallbacks_failures_zero","stop_length_and_exact_N256_required_for_all_samples")),
            "full raw final state/logits/token/output and all32 scalar-layer calls, no speculation")
        self.artifacts={name:self.descriptor(item)for name,item in p["artifacts"].items()}
        require(file_sha(core.__file__)==p["artifacts"]["core_reader"]["sha256"]==CORE_SHA and
            file_sha(__file__)==p["artifacts"]["reader"]["sha256"] and
            self.artifacts["private_build_manifest"].resolve()==self.args.private_build_manifest.resolve() and
            self.artifacts["target_build_manifest"].resolve()==self.args.target_build_manifest.resolve() and
            self.mapped(p["window_root"]).resolve()==self.args.window_root.resolve(),"frozen reader/core/build/window CLI identities")
        proof=verify_runtime_build(self.args.private_build_manifest);self.private=proof.pop("build");self.check_target_build();self.check_gates(p)
        for row in p["providers"]:
            self.descriptor(row["binary"]);self.descriptor(row["library"])
            gold=row["name"]=="G_lib"
            require(row["binary"]==self.target["gold_binary"if gold else"binary"] and
                row["library"]==self.target["libraries"]["G_lib"if gold else"private"] and
                row["expected_loaded_library_path"]==row["library"]["path"],"all actual stage provider target executable/library identities")
        baseline=read_json(self.artifacts["baseline_identity"])
        require(p["model"]==baseline["model"] and all(n in baseline and all(d[k]==baseline[n][k]for k in ("bytes","sha256"))for n,d in p["canonical_files"].items()),
            "preserved canonical source/binary/config/model identities")
        self.workload_inputs={}
        for w in p["workloads"]:
            a={name:self.descriptor(item)for name,item in w["artifacts"].items()};request,prompt,tokens=(a[k].read_bytes()for k in ("request","rendered_prompt","rendered_tokens"));t=read_json(a["tokenizer"])
            require(set(a)=={"request","tokenizer","rendered_prompt","rendered_tokens"} and b"\0"not in request and b"\0"not in prompt and
                len(tokens)==w["prompt_tokens"]*4 and t["prompt_tokens"]==w["prompt_tokens"] and t["prompt_bytes"]==len(prompt) and
                t["request_bytes"]==len(request) and t["prompt_renderer"]==c["renderer"] and t["tokenizer"]==c["tokenizer"] and
                t["token_encoding"]=="little_endian_int32" and t["add_bos"] is True and struct.unpack("<i",tokens[:4])[0]==t["bos_id"] and
                t["model_loader_called"] is False and t["metal_initialized"] is False,"complete real target-tokenizer input identities without model/device launch")
            self.workload_inputs[w["name"]]=dict(artifacts=a,request=request,prompt=prompt)
        if self.stage=="P":
            q=read_json(self.artifacts["qualification_protocol"])
            require(q["subject"]==SUBJECT and q["stage"]=="Q" and q["status"]=="frozen_ready_for_root_review_no_launch_by_author" and
                q["workloads"]==p["workloads"] and q["model"]==p["model"] and q["artifacts"]["private_build_manifest"]==p["artifacts"]["private_build_manifest"] and
                q["artifacts"]["target_build_manifest"]==p["artifacts"]["target_build_manifest"] and q["artifacts"]["reader"]==p["artifacts"]["reader"] and
                q["wrapper_sha256"]==p["wrapper_sha256"] and q["performance_gate"]==p["performance_gate"],
                "P independently verifies frozen Q protocol and exact source/build/inputs/reader/collector/gates")
        prior=self.mapped(p["prerequisite"]["report_path"])
        self.prerequisite_descriptor=dict(path=p["prerequisite"]["report_path"],bytes=prior.stat().st_size,sha256=file_sha(prior))
        self.descriptor(self.prerequisite_descriptor);self.prerequisite=read_json(prior)
        prerequisite_gate(self.stage,self.prerequisite,p["prerequisite"],p,self.target["binary"]["sha256"])
        self.report.update(stage=self.stage,protocol_sha256=file_sha(self.args.protocol),private_build=proof,
            private_build_manifest_sha256=file_sha(self.args.private_build_manifest),target_build_manifest_sha256=file_sha(self.args.target_build_manifest),
            binary_sha256=self.target["binary"]["sha256"],prerequisite_report_sha256=file_sha(prior),workload_identities=p["workloads"],
            provider_library_sha256={r["name"]:r["library"]["sha256"]for r in p["providers"]})

    def check_window_common(self,window,allowed_abort=None):
        p=self.protocol
        require(window["protocol_sha256"]==file_sha(self.args.protocol) and window["wrapper_sha256"]==p["wrapper_sha256"] and
            window["stage"]==self.stage and window["abort_reason"]in(None,allowed_abort) and window["canonical_unchanged"] is True and
            window["canonical_before"]==window["canonical_after"] and window["model_unchanged"] is True and
            window["model_before"]==window["model_after"] and not window["recognized_before"] and not window["recognized_after"],"exclusive unchanged native decode window")
        require(set(window["canonical_before"])==set(p["canonical_files"]),"complete canonical before/after file set")
        for n,d in p["canonical_files"].items():self.check_snapshot(window["canonical_before"][n],d)
        self.check_snapshot(window["model_before"],p["model"],model=True)

    def review_provider(self,provider):
        p=self.protocol;directory=self.args.window_root/provider["name"];repeats=p["request_contract"]["repeats"]
        require({x.name for x in directory.iterdir()}==set(p["output_layout"]["provider_files"])|set(p["output_layout"]["provider_directories"]),"complete unamended provider evidence directory")
        plan,window,monitor=(read_json(directory/n)for n in ("execution-plan.json","window.json","monitor.json"));self.check_window_common(window,"native nonzero exit")
        expected_root=Path(p["window_root"])/provider["name"]
        argv=[provider["binary"]["path"],"--config",p["artifacts"]["config"]["path"],"--run",p["model"]["path"],p["artifacts"]["request"]["path"],str(repeats),"256","327680"]
        env={SELECTOR:str(provider["mode"]),"MYDS_INDEXER_K_RESIDENT":"0","MYDS_BENCH_LOGITS_DIR":str(expected_root/"logits"),"MYDS_BENCH_STATE_DIR":str(expected_root/"state")}
        require(plan["stage"]==self.stage and plan["workload"]==window["workload"]==self.workload["name"] and
            plan["provider"]==window["provider"]==provider["name"] and plan["mode"]==window["mode"]==provider["mode"] and
            plan["argv"]==argv and plan["cwd"]==str(expected_root/"cwd") and plan["MYDS_inherited"]==env and
            plan["protocol_sha256"]==file_sha(self.args.protocol) and plan["wrapper_sha256"]==p["wrapper_sha256"] and
            plan["reader_sha256"]==window["reader_sha256"]==self.execution_reader_sha and
            plan["prerequisite_report_sha256"]==window["prerequisite_report_sha256"]==self.report["prerequisite_report_sha256"] and
            plan["expected_loaded_library_path"]==window["expected_loaded_library_path"]==provider["expected_loaded_library_path"] and
            plan["binary_sha256"]==provider["binary"]["sha256"] and plan["library_sha256"]==provider["library"]["sha256"] and
            plan["target_build_manifest_sha256"]==file_sha(self.args.target_build_manifest) and
            plan["private_build_manifest_sha256"]==file_sha(self.args.private_build_manifest) and
            plan["request_sha256"]==p["artifacts"]["request"]["sha256"] and plan["config_sha256"]==p["artifacts"]["config"]["sha256"] and
            plan["config_loaded_in_child_override_env"] is True and plan["effective_MYDS_count_expected"]==66 and
            plan["adoption_cohort"] is(self.stage=="P") and plan["performance_acceptance"] is False,
            "actual new decode argv/66MYDS/source/bin/lib/request/config/reader and prelaunch prerequisite identity")
        require(plan["timeout_s"]==p["timeout_s"] and plan["monitor_interval_s"]==p["monitor_interval_s"] and
            plan["shared_lock_path"]==p["lock_path"] and plan["exclusive_recognized_process_check"] is True and
            plan["started_epoch"]==window["started_epoch"]<=window["finished_epoch"] and 0<=window["elapsed_s"]<=p["timeout_s"]+20,
            "bounded exclusive native process lifetime")
        child=window["child_pid"]
        require(type(child)is int and child>0 and monitor and all(all(r["pid"]==child for r in x["recognized"])for x in monitor) and
            any(any(r["pid"]==child and Path(r["comm"]).name=="bench_decode_kv_only_fullmodel"for r in x["recognized"])for x in monitor),
            "real fresh decode child observed without recognized competitors")
        require((window["exit_code"]==0 and window["abort_reason"] is None)or(type(window["exit_code"])is int and window["exit_code"]!=0 and window["abort_reason"]=="native nonzero exit"),
            "successful child or preserved native nonzero execution failure")
        inv=self.check_inventory(directory,window)
        result=dict(workload=self.workload["name"],provider=provider["name"],mode=provider["mode"],child_pid=child,exit_code=window["exit_code"],
            started_epoch=window["started_epoch"],finished_epoch=window["finished_epoch"],window_gates_passed=True,raw_inventory=inv,
            execution_complete=window["exit_code"]==0,evidence_sha256={n:file_sha(directory/n)for n in ("execution-plan.json","window.json","monitor.json")},actual_argv=argv,window=window)
        try:
            parsed=read_run_exports(directory,directory/"stdout.jsonl",provider["mode"],self.workload["prompt_tokens"],256,repeats,self.payload_reader)
            configuration=effective_configuration(parsed["outer_metadata"],parsed["metadata"],self.artifacts["config"].read_bytes(),provider["mode"],expected_root/"logits",expected_root/"state",p["artifacts"]["config"]["path"])
            m,loaded=parsed["metadata"],parsed["loaded"]
            require(m["model"]==p["model"]["path"] and m["request_file"]==p["artifacts"]["request"]["path"] and
                m["max_seq_len"]==327680 and m["request_bytes"]==len(self.request) and m["prompt_bytes"]==len(self.prompt) and
                m["request_fnv1a64"]==fnv1a(self.request) and m["prompt_fnv1a64"]==fnv1a(self.prompt),"actual loaded model/request/rendered-prompt identity")
            core.loaded_library_identity(loaded["actual_metallib_path"],provider["expected_loaded_library_path"],provider["library"]["path"],provider["library"]["sha256"],self.mapped)
            result.update(samples=parsed["samples"],configuration=configuration,metadata=m,loaded=loaded,
                native_declared_valid=parsed["native_declared_valid"],provider_adoption_passed=parsed["provider_adoption_passed"],
                finite_logits_passed=parsed["finite_logits_passed"],loaded_library_verified=True,numerical_evidence_complete=True)
            return result,parsed["exports"]
        except (EvidenceError,KeyError,OSError,ValueError,TypeError,IndexError)as error:
            if window["exit_code"]==0:raise
            result.update(numerical_evidence_complete=False,native_execution_or_applicability_failure=True,native_failure_evidence_error=str(error),provider_adoption_passed=False,loaded_library_verified=False)
            return result,None

    def run(self):
        self.initialize();p,root=self.full_protocol,self.args.window_root
        require(file_sha(self.mapped(p["wrapper_path"]))==p["wrapper_sha256"],"frozen collector-only decode wrapper")
        plan,window=(read_json(root/n)for n in ("execution-plan.json","window.json"));order=checked_order(window["providers"],self.stage)
        require(plan["stage"]==window["stage"]==self.stage and plan["run_order"]==window["run_order"]==stage_order(self.stage) and
            plan["protocol_sha256"]==file_sha(self.args.protocol) and plan["wrapper_sha256"]==p["wrapper_sha256"] and
            plan["reader_sha256"]==window["reader_sha256"]==self.execution_reader_sha and plan["shared_lock_path"]==p["lock_path"] and
            plan["timeout_s"]==p["timeout_s"] and plan["performance_acceptance"] is False and
            plan["prerequisite_report"]==window["prerequisite_report"]==self.prerequisite_descriptor and
            plan["started_epoch"]>=self.prerequisite.get("finished_epoch",0),"prelaunch fixed decode order, actual prerequisite descriptor and reader/protocol/lock binding")
        require({x.name for x in root.iterdir()}=={"execution-plan.json","window.json","model-prefix.bin"}|{r["workload"]for r in order},"no undeclared outer files/retries")
        for name in {r["workload"]for r in order}:
            require({x.name for x in(root/name).iterdir()}=={r["provider"]for r in order if r["workload"]==name},"exactly one directory per frozen workload/provider")
        last=order[-1];self.check_window_common(window,"provider collection stopped: "+last["workload"]+"/"+last["provider"])
        require(window["lock_held_through_final_checks"] is True and window["service_or_production_mutation_performed"] is False and
            window["external_reader_required"] is True and window["qualification_passed"] is None and window["fullmodel_speedup_proven"] is False and
            window["production_acceptance"] is False and window["performance_acceptance"] is False and window["no_automatic_next_stage"] is True,
            "collector completed is neither quality nor speed nor production verdict")
        prefix=self.descriptor(window["model_prefix"])
        require(window["model_prefix"]["path"]==p["model_prefix_path"] and prefix.stat().st_size==p["model"]["prefix_bytes"]==1048576 and file_sha(prefix)==p["model"]["prefix_sha256"],"complete retained model identity prefix")
        results,exports=[],{w["name"]:{}for w in p["workloads"]};self.report["providers"]=results;outerroot=self.args.window_root
        try:
            for item in order:
                w=next(w for w in p["workloads"]if w["name"]==item["workload"]);provider=next(r for r in p["providers"]if r["name"]==item["provider"]);inputs=self.workload_inputs[w["name"]];self.workload=w
                self.protocol=dict(p,artifacts=dict(p["artifacts"],**w["artifacts"]),window_root=str(Path(p["window_root"])/w["name"]))
                self.artifacts=dict(self.artifacts,**inputs["artifacts"]);self.request,self.prompt=inputs["request"],inputs["prompt"];self.args.window_root=outerroot/w["name"]
                result,payloads=self.review_provider(provider)
                require(result["window"]==window["providers"][len(results)] and plan["started_epoch"]<=result["started_epoch"]<=result["finished_epoch"]<=window["finished_epoch"],"same outer/child record and bounded chronology")
                if results:require(result["child_pid"]not in {r["child_pid"]for r in results} and result["started_epoch"]>=results[-1]["finished_epoch"],"six serial fresh processes, no overlapping replacement")
                results.append(result)
                if payloads is not None:exports[w["name"]][provider["name"]]=payloads
        finally:self.args.window_root,self.protocol=outerroot,p
        complete=len(results)==6 and all(r["execution_complete"]for r in results)and window["abort_reason"] is None
        require(window["execution_complete"] is complete,"exact outer/child six-process execution completion")
        comparisons=[dict(workload=n,**raw_comparisons(e,self.stage))for n,e in exports.items()if len(e)==3]
        numerical=complete and len(comparisons)==2 and all(r["passed"]for r in comparisons)and all(
            all(r.get(k)is True for k in ("native_declared_valid","provider_adoption_passed","finite_logits_passed","numerical_evidence_complete"))for r in results)
        groups=[]
        if self.stage=="P"and numerical:
            groups=[dict(workload=w["name"],**performance_summary({r["provider"]:r["samples"]for r in results if r["workload"]==w["name"]}))for w in p["workloads"]]
        decision=stage_decision(self.stage,numerical,groups);decision["numerical_gates_passed"]=bool(numerical)
        self.report.update(**decision,review_completed=True,execution_complete=complete,
            outcome=("Q_qualified_P_requires_separate_freeze"if numerical else"Q_numerical_or_execution_failure_P_prohibited")if self.stage=="Q"else(
                "P_native_decode_gate_passed_no_production_acceptance"if decision["performance_acceptance"]else"P_performance_gate_failed_no_retry"if numerical else"P_numerical_or_execution_failure_performance_not_evaluated"),
            workload_numerical_comparisons=comparisons,performance_workloads=groups,native_generation_count=sum(len(r.get("samples",[]))for r in results),
            warmup_generations=sum(s["warmup"]for r in results for s in r.get("samples",[])),verification_or_formal_generations=sum(not s["warmup"]for r in results for s in r.get("samples",[])),
            raw_inventory_files=sum(r["raw_inventory"]["files"]for r in results),raw_logical_bytes=sum(r["raw_inventory"]["logical_bytes"]for r in results),
            outer_window_gates_passed=True,model_prefix_sha256=file_sha(prefix),model_identity_scope=p["model"],whole_model_payload_sha256_verified=False,
            window_evidence_sha256={n:file_sha(root/n)for n in ("execution-plan.json","window.json")},finished_epoch=window["finished_epoch"])
        return self.report


def evidence_failure(report,error):
    report=core.evidence_failure(report,error)
    report.update(outcome="decode_evidence_invalid_no_qualification_or_performance_acceptance",N256_two_workload_qualification_completed=False,
        native_decode_gate_passed=False,native_decode_speedup_proven=False,performance_evaluated=False,eligible_to_prepare_HTTP_validation=False)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("protocol","window-root","private-build-manifest","target-build-manifest","out"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--path-map",action="append",default=[]);args=p.parse_args();require(not args.out.exists(),"new independent decode report; existing evidence never overwritten")
    run,code=None,0
    try:run=Review(args);report=run.run()
    except (EvidenceError,KeyError,OSError,ValueError,TypeError,IndexError,StopIteration)as error:
        report=evidence_failure(run.report if run else None,error);code=2
    args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:report.get(k)for k in ("review_completed","stage","outcome","numerical_gates_passed","qualification_passed","performance_evaluated","native_decode_gate_passed","production_acceptance","evidence_error")},indent=2))
    raise SystemExit(code)


if __name__=="__main__":main()

