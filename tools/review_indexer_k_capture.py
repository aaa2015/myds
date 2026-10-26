#!/usr/bin/env python3
"""Independent CPU reader for the frozen real-indexer component experiment.

No benchmark/builder modules are imported, no subprocesses are launched, and
no GPU API is used. Full score/ordered-ID/input/guard bytes are the numerical
authority. Timing medians and the immutable protocol gates are recomputed.
An experimental numerical failure produces a structured completed review;
missing, corrupted or contradictory evidence produces evidence_invalid.
"""
import argparse
import array
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import struct
import sys

LAYERS = (2, 8, 14)
INPUTS = ("q-f32", "weights-f32", "cache")
COUNTERS = ("prefix_requests", "prefix_encoded", "prefix_unavailable", "nan_fuse_encoded", "nan_shuffle_encoded")
GUARD = b"\xa5" * 64


class EvidenceError(ValueError):
    pass


def require(value, label):
    if not value:
        raise EvidenceError(label)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def harness_flag(value):
    # Objective-C @(C unary-not expression) serializes as integer 0/1.
    require(type(value) in (bool, int) and value in (0, 1), "bounded native BOOL/int flag")
    return bool(value)


def frozen_performance_gate(saved, drift, performance):
    return (abs(drift) <= performance["max_abs_control_drift_percent"],
            saved > performance["min_conservative_time_saved_percent_strict"])


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def read_json(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=unique_object)


def words(data, signed=False):
    require(len(data) % 4 == 0, "complete 32-bit payload")
    result = array.array("i" if signed else "I")
    require(result.itemsize == 4, "reader uses 32-bit array words")
    result.frombytes(data)
    if sys.byteorder != "little":
        result.byteswap()
    return result


def floats(data):
    result = array.array("f")
    require(result.itemsize == 4, "reader uses F32")
    result.frombytes(data)
    if sys.byteorder != "little":
        result.byteswap()
    return result


def diff_words(a, b, float_metrics=False):
    require(len(a) == len(b), "full equal comparison extents")
    if a == b:
        return dict(mismatches=0, first_mismatch=None, max_abs_error=0.0 if float_metrics else None,
                    max_finite_ulp_distance=0 if float_metrics else None)
    x, y = words(a), words(b)
    bad, first, maximum, ulp = 0, None, 0.0, 0
    for index, (left, right) in enumerate(zip(x, y)):
        if left == right:
            continue
        bad += 1
        if first is None:
            first = dict(index=index, reference_bits=f"0x{left:08x}", actual_bits=f"0x{right:08x}")
        if float_metrics and left & 0x7F800000 != 0x7F800000 and right & 0x7F800000 != 0x7F800000:
            lv = struct.unpack("<f", struct.pack("<I", left))[0]
            rv = struct.unpack("<f", struct.pack("<I", right))[0]
            maximum = max(maximum, abs(lv - rv))
            ordered = lambda value: (~value & 0xFFFFFFFF) if value & 0x80000000 else value | 0x80000000
            ulp = max(ulp, abs(ordered(left) - ordered(right)))
    return dict(mismatches=bad, first_mismatch=first,
                max_abs_error=maximum if float_metrics else None,
                max_finite_ulp_distance=ulp if float_metrics else None)


def safe_child(root, name, single=False):
    rel = Path(name)
    require(isinstance(name, str) and name and not rel.is_absolute() and ".." not in rel.parts,
            "relative payload path")
    if single:
        require(len(rel.parts) == 1 and name not in (".", ".."), "single local payload filename")
    path = Path(root) / rel
    require(not path.is_symlink() and path.is_file(), "regular payload file: " + str(path))
    require(path.resolve().is_relative_to(Path(root).resolve()), "payload remains inside root")
    return path


class Review:
    def __init__(self, args):
        self.args = args
        self.maps = []
        for item in args.path_map:
            require("=" in item, "--path-map OLD=NEW")
            old, new = item.split("=", 1)
            require(Path(old).is_absolute() and Path(new).is_absolute(), "absolute path map")
            self.maps.append((Path(old), Path(new)))
        self.maps.sort(key=lambda pair: len(str(pair[0])), reverse=True)
        self.issues = []
        self.report = dict(schema_version=1, reader="independent_real_indexer_component_reader",
                           review_completed=False, fullmodel_speedup_proven=False, production_adoption=False,
                           reader_sha256=file_sha(__file__),
                           path_maps=[dict(original=str(old), retained_copy=str(new)) for old, new in self.maps],
                           capture_root_override=str(args.capture_root) if args.capture_root else None)
        self.raw = {}
        self.consumed = set()

    def mapped(self, value):
        path = Path(value)
        for old, new in self.maps:
            if path == old or path.is_relative_to(old):
                return new / path.relative_to(old)
        return path

    def gate(self, category, name, ok, **details):
        result = dict(category=category, name=name, passed=bool(ok), **details)
        if not ok:
            self.issues.append(result)
        return result

    def payload(self, descriptor, root):
        path = safe_child(root, descriptor["path"], single=True)
        require(path.stat().st_size == descriptor["bytes"] and file_sha(path) == descriptor["sha256"],
                "payload size/SHA: " + str(path))
        return path.read_bytes()

    def artifact(self, name):
        require(name in self.raw, "output raw is manifest-listed: " + name)
        self.consumed.add(name)
        return self.raw[name].read_bytes()

    def check_protocol(self):
        p = self.protocol
        require(p["schema_version"] == 2 and p["kind"] == "real_indexer_component_protocol", "frozen protocol v2")
        require(p["layers"] == list(LAYERS) and p["tokens_used"] == 32768 and p["global_chunk_start"] == 30720 and
                p["batch_floats"] == 2097152 and p["require_actual_shape"] is True and
                p["revision_before_any_real_replay_gpu"] is True, "protocol scope")
        performance = p["performance"]
        require(performance["mode"] == "concurrent" and performance["warmup_groups"] == 1 and
                performance["formal_aba_groups"] == 9 and len(performance["controls"]) == 2 and
                performance["every_layer_every_control_required"] is True and performance["one_formal_cohort"] is True and
                performance["exclusive_recognized_process_check"] is True and
                performance["max_abs_control_drift_percent"] == 2.0 and
                performance["min_conservative_time_saved_percent_strict"] == 2.0, "immutable performance thresholds")
        require(p["correctness"]["all_layers_before_formal_timing"] is True and
                p["fullmodel_run"] is False and p["production_adoption"] is False, "protocol numerical/full-model scope")
        root = self.args.input_root.resolve(strict=True)
        for name, digest in p["input_sha256"].items():
            require(file_sha(safe_child(root, name)) == digest, "frozen protocol input: " + name)
        self.report["protocol_sha256"] = file_sha(self.args.protocol)
        self.report["protocol_input_files_verified"] = len(p["input_sha256"])

    def check_build(self):
        b = self.build
        require(b["kind"] == "real_indexer_replay_build" and b["compile_only_pass"] is True and
                b["gpu_work_submitted"] is False and b["production_registered"] is False and
                b["topk_api"] == "myds_v41_gpu_topk_causal" and b["normal_runtime_object_observer_absent"] is True,
                "frozen replay build scope")
        directory = self.args.build_manifest.resolve().parent
        for rel, descriptor in b["inputs"].items():
            source = safe_child(directory / "source", rel)
            require(source.stat().st_size == descriptor["bytes"] and file_sha(source) == descriptor["sha256"], "frozen replay source: " + rel)
        require(file_sha(self.mapped(b["binary"])) == b["binary_sha256"], "complete replay executable identity")
        self.report["build_manifest_sha256"] = file_sha(self.args.build_manifest)
        self.report["replay_source_files_verified"] = len(b["inputs"])
        metal = self.metal
        require(metal["build_mode"] == "full-library" and metal["compile_only_pass"] is True and
                metal["production_registered"] is False and metal["gpu_work_submitted"] is False, "complete translation unit candidate build")
        md = self.args.metal_manifest.resolve().parent
        require(file_sha(md / "probe.metallib") == metal["metallib_sha256"] and
                file_sha(md / "native.metallib") == metal["native_metallib_sha256"], "frozen complete libraries")
        native = (md / "native.metal").read_bytes()
        generated = (md / "probe.metal").read_bytes()
        combined = (md / "all.metal").read_bytes()
        require(sha(native) == metal["native_source_sha256"] and sha(generated) == metal["probe_sha256"] and
                sha(combined) == metal["combined_source_sha256"], "complete Metal sources")
        require(combined == native + metal["candidate_append_marker"].encode() + generated, "native source prefix unchanged")
        script = md / "native-source/tools/build_metallib.py"
        require(file_sha(script) == metal["build_metallib_script_sha256"], "frozen Metal builder")
        constants = {}
        for node in ast.parse(script.read_text()).body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in ("PREAMBLE", "SHADER_FILES", "COPY_KERNELS"):
                        constants[target.id] = ast.literal_eval(node.value)
        require(constants["SHADER_FILES"] == metal["original_shader_order"], "frozen shader order")
        assembled = constants["PREAMBLE"].encode() + b"\n"
        require(len(metal["original_shaders"]) == len(constants["SHADER_FILES"]), "complete shader list")
        for index, name in enumerate(constants["SHADER_FILES"]):
            source = safe_child(md / "native-source/metal", name)
            payload, r = source.read_bytes(), metal["original_shaders"][index]
            require(r["index"] == index and r["path"] == "metal/" + name and r["bytes"] == len(payload) and r["sha256"] == sha(payload), "original shader identity")
            assembled += f'\n#line 1 "{name}"\n'.encode()
            require(len(assembled) == r["source_begin"], "shader source begin")
            assembled += payload
            require(len(assembled) == r["source_end"], "shader source end")
        assembled += constants["COPY_KERNELS"].encode() + b"\n"
        require(assembled == native, "independent full native assembly")
        def body(source, name):
            at = source.index("kernel void " + name + "(")
            begin, cursor, depth = source.rfind("template<", 0, at), source.index("{", at) + 1, 1
            while depth:
                depth += (source[cursor] == "{") - (source[cursor] == "}")
                cursor += 1
            return source[begin:cursor]
        original = body(native.decode(), "kernel_dsv4_indexer_scores_tiled_storage")
        restored = body(generated.decode(), "probe_indexer_scores_k_resident")
        for edit in reversed(metal["proof"]["exact_edits"]):
            require(restored.count(edit["after"]) == 1, "unique inverse-edit source anchor")
            restored = restored.replace(edit["after"], edit["before"], 1)
        require(restored == original and sha(original.encode()) == metal["proof"]["native_template_sha256"], "exact candidate inverse edits")
        metal_commands = [command for command in metal["commands"] if command[:4] == ["xcrun", "-sdk", "macosx", "metal"]]
        require(len(metal_commands) == 2, "complete native and probe compiler invocations")
        expected_sources = {sha(native), sha(combined)}
        actual_sources = set()
        for command in metal_commands:
            at = command.index("-c")
            require(command[4:at] == metal["compile_options"], "identical frozen Metal options for native and probe")
            actual_sources.add(file_sha(self.mapped(command[at + 1])))
        require(actual_sources == expected_sources, "compiler source operands bind complete native/probe source bytes")
        self.report["metal_build_manifest_sha256"] = file_sha(self.args.metal_manifest)
        self.report["complete_metal_shader_files_verified"] = len(constants["SHADER_FILES"])

    def check_capture(self):
        c = self.capture
        require(c["schema_version"] == 1 and c["capture_kind"] == "myds_real_prefill_indexer" and
                c["status"] == "verified_cpu_only" and c["replay_gpu_submitted"] is False and
                c["performance_from_capture"] is False and c["config_values_matched"] is True, "verified actual capture scope")
        root = self.args.capture_root.resolve(strict=True) if self.args.capture_root else self.mapped(c["capture_directory"]).resolve(strict=True)
        for name, descriptor in c["files"].items():
            require(name == descriptor["path"], "capture filename descriptor")
            self.payload(descriptor, root)
        actual_build_path = self.mapped(c["build_manifest_path"])
        require(file_sha(actual_build_path) == c["build_manifest_sha256"], "capture build-manifest binding")
        capture_build = read_json(actual_build_path)
        require(capture_build["compile_only_pass"] is True and capture_build["normal_preprocessor_observer_absent"] is True and
                capture_build["binary_sha256"] == c["capture_binary_sha256"] and
                file_sha(self.mapped(capture_build["binary"])) == c["capture_binary_sha256"], "actual capture executable identity")
        for rel, descriptor in capture_build["inputs"].items():
            source = safe_child(actual_build_path.parent / "source", rel)
            require(source.stat().st_size == descriptor["bytes"] and file_sha(source) == descriptor["sha256"],
                    "complete capture compiler-visible snapshot: " + rel)
        self.report["capture_source_files_verified"] = len(capture_build["inputs"])
        metadata = c["metadata"]
        require(metadata["real_model_forward"] is True and metadata["tokens"] == 32768 and
                metadata["performance_from_capture"] is False, "real model forward witness")
        require(file_sha(self.mapped(c["loaded_metallib_path"])) == c["loaded_metallib_sha256"], "captured actual loaded library identity")
        require(c["config_sha256"] == self.protocol["input_sha256"]["deployed/myds.conf"] and
                c["tokens_file_sha256"] == self.protocol["input_sha256"]["capture-tokens.txt"], "capture/frozen request and config binding")
        tokens = (root / "input-token-ids.i32").read_bytes()
        require(len(tokens) == 32768 * 4 and sha(tokens) == self.protocol["input_sha256"]["capture-tokens.i32le.bin"], "exact supplied 32768 consumed token bytes")
        require([case["layer"] for case in c["cases"]] == list(LAYERS), "exact three actual owner layers")
        self.cases = {}
        for case in c["cases"]:
            layer = case["layer"]
            payloads = {name: self.payload(descriptor, root) for name, descriptor in case["payloads"].items()}
            args = struct.unpack("<6I5Qf4x", payloads["args72"])
            keys, rows, heads, dim, pos, ratio, qs, hs, ws, ks, ss, scale = args
            require(15000 <= keys <= 16384 and 8 <= rows <= 2048 and keys * rows <= 2097152 and
                    heads == 32 and dim == 128 and pos == 30720 and ratio == 2 and qs == 16384 and hs == 512 and
                    ws == 128 and ks == (256 if case["cache_bf16"] else 512) and ss == keys * 4 and scale == 1 / 64,
                    "bounded complete real subgroup ABI")
            require((keys, rows, pos, ratio) == (case["n_comp"], case["n_tokens"], case["pos0"], case["ratio"]) and
                    case["base_pos"] + case["c_start"] == pos and case["group_off"] == 0 and
                    case["synthetic"] is False and case["full_scores"] is True and case["performance_from_capture"] is False and
                    case["key_window"] == 0 and case["observer_commit_wait"] is True, "actual captured provider scope")
            provider = case["provider"]
            tiled = provider in ("kernel_dsv4_indexer_scores_tiled_f32", "kernel_dsv4_indexer_scores_tiled_bf16")
            exact = provider == "kernel_myds_indexer_score_chunk_exact"
            direct = provider == "kernel_glm_indexer_score_chunk_direct"
            require(tiled or exact or direct, "known actual captured native provider")
            expected_native = payloads["args72"][:68] if tiled else struct.pack("<4If4I", keys, 32, 128, 2 if case["cache_bf16"] else 0, 1 / 64, rows, pos, 2, 0)
            require((payloads["native-args"][:68] if tiled else payloads["native-args"]) == expected_native, "actual provider args")
            require(case["grid"] == ([keys, rows, 1] if direct else [(keys + (31 if tiled else 15)) // (32 if tiled else 16), (rows + 7) // 8, 1]) and
                    case["threads"] == [32, 4, 1] and case["dynamic_threadgroup_bytes"] == (21504 if tiled else 5120 if exact else 528), "actual captured dispatch geometry")
            for name, extent in (("q-f32", rows * qs), ("weights-f32", rows * ws), ("cache", keys * ks),
                                 ("scores-f32", rows * keys * 4), ("selected-i32", rows * 512 * 4)):
                require(len(payloads[name]) == extent, "full capture payload extent: " + name)
            for name in ("q-f32", "weights-f32"):
                require(all(math.isfinite(v) for v in floats(payloads[name])), "finite complete actual input: " + name)
            topk = case["actual_topk_consumer"]
            require(topk["ordered_ids_complete"] is True and topk["mask_filter_between_scores_and_topk"] is False and
                    topk["top"] == 512 and topk["n_tokens"] == rows and topk["n_comp"] == keys and
                    topk["pos0"] == pos and topk["ratio"] == ratio and topk["dispatches"], "captured complete ordered TopK provenance")
            require(all(call["host_accepted"] is True and call["kernel"].startswith("kernel_") for call in topk["dispatches"]), "accepted actual TopK dispatch witnesses")
            self.cases[layer] = dict(metadata=case, payloads=payloads, rows=rows, keys=keys, pos=pos, ratio=ratio)
        self.report["capture_verified_sha256"] = file_sha(self.args.capture_verified)
        self.report["capture_files_verified"] = len(c["files"])

    def check_window(self):
        directory = self.args.window_dir.resolve(strict=True)
        if (directory / "refused.json").exists() and not (directory / "execution-plan.json").exists():
            self.report.update(review_completed=True, outcome="preflight_refused_no_child", candidate_evaluated=False,
                               refusal=read_json(directory / "refused.json"), fullmodel_eligible=False,
                               refusal_sha256=file_sha(directory / "refused.json"))
            return False
        plan = read_json(directory / "execution-plan.json")
        window = read_json(directory / "window.json")
        monitor = read_json(directory / "monitor.json")
        require(plan["protocol_sha256"] == window["protocol_sha256"] == file_sha(self.args.protocol), "execution/frozen protocol SHA")
        require(plan["build_manifest_sha256"] == file_sha(self.args.build_manifest) and
                plan["binary_sha256"] == self.build["binary_sha256"] and
                plan["probe_sha256"] == self.metal["metallib_sha256"] and
                plan["native_sha256"] == self.capture["loaded_metallib_sha256"] and
                plan["capture_json_sha256"] == file_sha(self.args.capture_verified) and
                plan["config_sha256"] == self.capture["config_sha256"], "execution plan binds every actual input/artifact")
        require(plan["MYDS_inherited"] == {} and plan["exclusive_recognized_process_check"] is True and plan["fullmodel"] is False,
                "clean child environment and window scope")
        wrapper = self.args.window_wrapper or self.args.input_root / "real-window-v4.py"
        require(file_sha(wrapper) == plan["wrapper_sha256"], "frozen child-launch/monitor wrapper identity")
        argv = plan["argv"]
        require(len(argv) == 13 and argv[1::2] == ["--capture-json", "--metallib", "--native-metallib", "--config", "--out-dir", "--samples"], "exact replay argv shape")
        options = dict(zip(argv[1::2], argv[2::2]))
        require(file_sha(self.mapped(argv[0])) == self.build["binary_sha256"] and
                file_sha(self.mapped(options["--capture-json"])) == file_sha(self.args.capture_verified) and
                file_sha(self.mapped(options["--metallib"])) == plan["probe_sha256"] and
                file_sha(self.mapped(options["--native-metallib"])) == plan["native_sha256"] and
                file_sha(self.mapped(options["--config"])) == plan["config_sha256"] and options["--samples"] == "9", "actual child argv payload identity")
        self.result_dir = self.args.result_dir.resolve(strict=True) if self.args.result_dir else directory / "outputs"
        require(self.mapped(options["--out-dir"]).resolve() == self.result_dir.resolve(), "actual child output directory")
        baseline_path = self.args.baseline_identity or self.args.input_root / "baseline-identity.json"
        baseline = read_json(baseline_path)
        before, after = window["canonical_before"], window["canonical_after"]
        require(before == after and window["canonical_unchanged"] is True and len(before) == 3 and
                {Path(path).name for path in before} == {"myds", "myds.metallib", "myds.conf"} and set(before) <= set(baseline) and
                all(digest == baseline[path]["sha256"] for path, digest in before.items()), "canonical before/after/frozen baseline identity")
        native_source = [value for path, value in baseline.items() if path.endswith("/build/myds_all.metal")]
        require(len(native_source) == 1 and native_source[0]["sha256"] == self.metal["native_source_sha256"],
                "full native source equals frozen deployed compiler-visible complete source")
        self.report["baseline_identity_sha256"] = file_sha(baseline_path)
        child = window["child_pid"]
        require(isinstance(child, int) and child > 0 and isinstance(monitor, list) and monitor, "actual launched child PID and monitor")
        previous, child_seen = -1.0, False
        for sample in monitor:
            require(math.isfinite(sample["elapsed_s"]) and sample["elapsed_s"] >= previous, "ordered monitor samples")
            previous = sample["elapsed_s"]
            rows = sample["recognized"]
            require(all(isinstance(row["pid"], int) and isinstance(row["comm"], str) for row in rows), "monitor process records")
            child_seen |= any(row["pid"] == child and Path(row["comm"]).name == "bench_indexer_k_capture" for row in rows)
        exclusive = not window["recognized_before"] and not window["recognized_after"] and child_seen and \
            all(all(row["pid"] == child for row in sample["recognized"]) for sample in monitor) and window["abort_reason"] is None
        self.window_gate = self.gate("window", "recognized_process_exclusivity", exclusive,
                                     child_pid=child, monitor_samples=len(monitor), child_seen=child_seen,
                                     recognized_before=window["recognized_before"], recognized_after=window["recognized_after"],
                                     abort_reason=window["abort_reason"])
        require(plan["started_epoch"] <= window["finished_epoch"] and 0 <= window["elapsed_s"] <= plan["timeout_s"] + 20, "bounded window time")
        self.plan, self.window = plan, window
        self.report["window"] = self.window_gate
        self.report["window_evidence_sha256"] = {name: file_sha(directory / name) for name in ("execution-plan.json", "monitor.json", "window.json")}
        return True

    def load_outputs(self):
        manifest_path = self.result_dir / "artifact-manifest.json"
        self.manifest = m = read_json(manifest_path)
        require(m["probe"] == "indexer_k_resident_real_capture" and m["fixture_kind"] == "actual_fullmodel_prefill" and
                m["fullmodel"] is False and m["cold_dram_proven"] is False and m["performance_from_capture"] is False and
                m["host_only"] is False and m["gpu_work_submitted"] is True and
                m["mode"] == "concurrent" and m["samples"] == 9 and m["warmups"] == 1, "real replay output scope")
        require(m["capture_json_sha256"] == file_sha(self.args.capture_verified) and
                m["binary_sha256"] == self.build["binary_sha256"] and m["probe_sha256"] == self.metal["metallib_sha256"] and
                m["native_sha256"] == self.capture["loaded_metallib_sha256"] and
                m["config_sha256"] == self.capture["config_sha256"] and harness_flag(m["topk_replayed_with_actual_helper"]) and
                m["capture_default_provider_is_timing_control_0"] is True, "actual replay manifest artifact binding")
        require(m["topk_loaded_library_sha256"] == m["native_sha256"] and
                file_sha(self.mapped(m["topk_loaded_library_path"])) == m["native_sha256"] and
                self.mapped(m["topk_loaded_library_path"]).resolve() == self.mapped(self.plan["argv"][6]).resolve(), "actual TopK loaded native library")
        for descriptor in m["files"]:
            name = descriptor["path"]
            require(name not in self.raw, "unique replay raw filename")
            path = safe_child(self.result_dir, name, single=True)
            require(path.stat().st_size == descriptor["bytes"] and file_sha(path) == descriptor["sha256"], "complete replay raw bytes/SHA: " + name)
            self.raw[name] = path
        self.jsonl_path = self.args.jsonl or self.args.window_dir / "stdout.jsonl"
        self.records, info = [], []
        json_started = False
        for line in self.jsonl_path.read_text().splitlines():
            if not line.strip():
                continue
            if line.startswith("{"):
                json_started = True
                self.records.append(json.loads(line, object_pairs_hook=unique_object))
            else:
                require(not json_started, "non-JSON text occurs only in frozen initialization prefix")
                match = re.fullmatch(r"\[\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d{3}\] \[INFO\] (.+)", line)
                require(match is not None, "known INFO initialization format; arbitrary non-JSON output refused")
                info.append(match.group(1))
        require(len(info) == 21, "complete known 21-message native initialization prefix")
        config_match = re.fullmatch(r"config: 已加载 (.+) —— 生效 (\d+) 条", info[0])
        library_match = re.fullmatch(r"myds_metal: loaded precompiled library from (.+) \(all required V4\.1 kernels verified\)", info[1])
        require(config_match and library_match and int(config_match.group(2)) == self.capture["config_declared_unique_count"] and
                file_sha(self.mapped(config_match.group(1))) == m["config_sha256"] and
                file_sha(self.mapped(library_match.group(1))) == m["native_sha256"], "known initialization config/library identities")
        expected = [
            "Shared Q4 V41 NB: 0 (ordinary NB+HC GU2304x5120/Down5120x2304, NR2/NSG4)",
            "myds_metal: Initialized successfully on GPU: " + m["device"],
            "Indexer head slots: 1 (F32 candidate; explicit ordinary scalar scope)",
            "Indexer topk threshold: 0 (reference; explicit ordinary scalar scope)",
            "Indexer topk NaN shuffle: 0 (reference; reference partition geometry)",
            "Routed MoE geometry: IQ2 NSG=2, Q2 down NSG=4 (explicit)",
            "Routed MoE IQ2 LUT: threadgroup (reference)",
            "Routed MoE IQ2 gate/up sign: popcount (explicit, bit-exact)",
            "Batch2 Q8 shared weights: enabled (explicit; TAIL, SHARED, QKV)",
            "LM-head Q8 rows per threadgroup: 2 (reference; single token, large output, NSG8)",
            "Dense Q4 rows per threadgroup: 2 (reference; explicit ordinary scalar scope), NSG=4",
            "F16 GEMV BF16 fuse: 1 (fused store; explicit ordinary scalar scope)",
            "Indexer RoPE quant fuse: 1 (SIMD candidate; explicit ordinary scalar scope)",
            "Routed MoE Q2 sum cache: 0 (reference; explicit ordinary scalar scope)",
            "Routed MoE Q2 sum cache fused: 0 (reference; explicit ordinary scalar scope)",
            "Decode candidates: down overlap=1, IQ2 V4.1=0, Dense Q4 V4.1=1 (ordinary scalar only)",
            "Routed MoE Q2 V4.1: 1 (ordinary scalar F32 MID only; 5120/2304/384/6, NR2, down NSG2/4; MM, trace and unscoped paths retain reference)",
            "Attention flash-decoding: 0 (reference (single-TG in-register 14-NSG))",
            "HC mix norm pre-fusion: 0 (reference (separate norm + GEMV))",
        ]
        require(info[2:] == expected, "only the exact frozen native INFO messages in their original order")
        known = {"real_inputs", "real_parity", "real_topk", "all_layers_numerical_gate", "real_timing", "real_summary"}
        require(self.records and all(record["kind"] in known for record in self.records), "known complete JSONL records")
        self.report["artifact_manifest_sha256"] = file_sha(manifest_path)
        self.report["timing_jsonl_sha256"] = file_sha(self.jsonl_path)
        self.report["raw_files_verified"] = len(self.raw)
        self.report["known_initialization_info_lines"] = len(info)

    def score_stats(self, data, case):
        require(len(data) == case["rows"] * case["keys"] * 4, "complete score extent")
        bits = words(data)
        finite, tail, written = True, True, True
        invisible, bad_finite, bad_tail, poison = 0, 0, 0, 0
        for row in range(case["rows"]):
            visible = min(case["keys"], (case["pos"] + row + 1) // case["ratio"])
            start = row * case["keys"]
            for col in range(case["keys"]):
                b = bits[start + col]
                poison += b == 0xA5A5A5A5
                if col >= visible:
                    invisible += 1
                    bad_tail += b != 0xFF800000
                else:
                    bad_finite += b & 0x7F800000 == 0x7F800000
        finite, tail, written = not bad_finite, not bad_tail, not poison
        return dict(visible_finite=finite, causal_tail_valid=tail, all_written=written,
                    invisible_minus_inf_count=invisible, nonfinite_visible_count=bad_finite,
                    noncanonical_tail_count=bad_tail, poison_count=poison)

    def ordered_ids_stats(self, scores, ids, case):
        require(len(ids) == case["rows"] * 512 * 4, "complete ordered Top512 extent")
        selected, values = words(ids, signed=True), floats(scores)
        distinct, valid, descending, cutoff = True, True, True, True
        first = None
        for row in range(case["rows"]):
            visible = min(case["keys"], (case["pos"] + row + 1) // case["ratio"])
            row_ids = selected[row * 512:(row + 1) * 512]
            row_distinct = len(set(row_ids)) == 512
            row_valid = all(0 <= k < visible for k in row_ids)
            row_descending, row_cutoff = False, False
            if row_valid:
                chosen = [values[row * case["keys"] + k] for k in row_ids]
                row_descending = all(math.isfinite(v) for v in chosen) and all(a >= b for a, b in zip(chosen, chosen[1:]))
                if row_descending:
                    bottom = chosen[-1]
                    ids_set = set(row_ids)
                    row_cutoff = all(k in ids_set or values[row * case["keys"] + k] <= bottom for k in range(visible))
            distinct &= row_distinct
            valid &= row_valid
            descending &= row_descending
            cutoff &= row_cutoff
            if first is None and not (row_distinct and row_valid and row_descending and row_cutoff):
                first = dict(row=row, distinct=row_distinct, causally_valid=row_valid,
                             scores_descending=row_descending, no_better_omitted_key=row_cutoff)
        return dict(distinct=distinct, causally_valid=valid, scores_descending=descending,
                    no_better_omitted_key=cutoff, first_invalid_row=first,
                    passed=distinct and valid and descending and cutoff)

    def check_numerical(self):
        self.initial_scores, self.initial_ids = {}, {}
        self.score_reports, self.id_reports, self.input_reports = [], [], []
        self.score_gate_passed, self.topk_gate_passed = True, True
        failure_index, phase1_scores, phase1_ids = 0, set(), set()
        for layer, case in self.cases.items():
            captured_stats = self.score_stats(case["payloads"]["scores-f32"], case)
            require(captured_stats["all_written"] and captured_stats["visible_finite"] and captured_stats["causal_tail_valid"] and
                    captured_stats["invisible_minus_inf_count"] == case["metadata"]["invisible_minus_inf_count"], "actual capture canonical score table")
            require(self.ordered_ids_stats(case["payloads"]["scores-f32"], case["payloads"]["selected-i32"], case)["passed"], "actual captured Top512 is complete, distinct and maximal")
        for index, r in enumerate(self.records):
            if r["kind"] != "real_parity":
                continue
            layer, provider = r["layer"], r["provider"]
            require(layer in self.cases and provider in (0, 1, 2), "known score replay provider/layer")
            case = self.cases[layer]
            key = (layer, provider)
            initial = key not in phase1_scores
            if initial:
                prefix = f"layer-{layer}-provider-{provider}"
                phase1_scores.add(key)
            else:
                prefix = f"layer-{layer}-provider-{provider}-failure-{failure_index}"
                failure_index += 1
            data = self.artifact(prefix + "-scores.f32")
            guards = self.artifact(prefix + "-guards.bin")
            actual_diff = diff_words(case["payloads"]["scores-f32"], data, float_metrics=True)
            if initial:
                self.initial_scores[key] = data
            reference = self.initial_scores.get((layer, 1))
            candidate_diff = diff_words(reference, data, float_metrics=True) if provider == 2 and reference is not None else None
            stats = self.score_stats(data, case)
            intact = guards == GUARD * 2
            ok = intact and stats["all_written"] and stats["visible_finite"] and stats["causal_tail_valid"] and \
                (provider != 0 or actual_diff["mismatches"] == 0) and \
                (provider == 0 or reference is None or data == reference)
            require(r["tokens"] == case["rows"] and r["keys"] == case["keys"] and bool(r["bf16"]) == bool(case["metadata"]["cache_bf16"]) and
                    r["captured_actual_provider"] == case["metadata"]["provider"] and r["mismatches_to_actual_capture"] == actual_diff["mismatches"] and
                    r["candidate_mismatches_to_tiled"] == (candidate_diff["mismatches"] if provider == 2 else None) and
                    r["guards_intact"] == intact and all(r[name] == stats[name] for name in
                    ("all_written", "visible_finite", "causal_tail_valid", "invisible_minus_inf_count")) and
                    r["passed"] == ok and r["full_scores"] is True and r["fullmodel_speedup_proven"] is False,
                    "score record agrees with full raw bytes")
            category = "actual_capture_replay" if provider == 0 else "candidate_scores" if provider == 2 else "tiled_structure"
            entry = self.gate(category, prefix, ok, layer=layer, provider=provider, initial=initial, record_index=index,
                              mismatches_to_actual_capture=actual_diff, candidate_mismatches_to_tiled=candidate_diff,
                              guards_intact=intact, **stats)
            self.score_reports.append(entry)
            self.score_gate_passed &= ok
        topk_seen = set()
        for index, r in enumerate(self.records):
            if r["kind"] != "real_topk":
                continue
            layer, provider = r["layer"], r["provider"]
            require(layer in self.cases and provider in (0, 1, 2), "known TopK provider/layer")
            key, case = (layer, provider), self.cases[layer]
            initial = key not in topk_seen
            topk_seen.add(key)
            prefix = f"layer-{layer}-provider-{provider}" + ("" if initial else "-failure")
            ids = self.artifact(prefix + "-selected.i32")
            guards = self.artifact(prefix + "-topk-guards.bin")
            if initial:
                phase1_ids.add(key)
                self.initial_ids[key] = ids
            require(key in self.initial_scores, "TopK has a complete score source")
            score_source = self.initial_scores[key]
            # A formal failure stores an additional score for one provider;
            # unaffected providers still consume their last byte-identical table.
            if not initial:
                failures = [entry for entry in self.score_reports if entry["layer"] == layer and entry["provider"] == provider and not entry["initial"]]
                if failures:
                    score_source = self.artifact(failures[-1]["name"] + "-scores.f32")
            comparison = diff_words(case["payloads"]["selected-i32"], ids)
            tiled = self.initial_ids.get((layer, 1)) if initial else None
            if not initial:
                tiled = self.artifact(f"layer-{layer}-provider-1-failure-selected.i32")
            candidate_diff = diff_words(tiled, ids) if provider == 2 and tiled is not None else None
            intact = guards == GUARD * 4
            ranking = self.ordered_ids_stats(score_source, ids, case)
            captured_counter = case["metadata"]["actual_topk_consumer"]
            same_counter_path = all(r[name] == captured_counter[name] for name in COUNTERS)
            witnessed = bool(r["host_encoded"]) and bool(r["GPU_completed"]) and r["score_input_unchanged"] is True and same_counter_path
            ok = comparison["mismatches"] == 0 and (provider != 2 or candidate_diff is not None and candidate_diff["mismatches"] == 0) and intact and witnessed
            require(r["top"] == 512 and r["rows"] == case["rows"] and r["ordered_ids_complete"] is True and
                    r["actual_capture_mismatches"] == comparison["mismatches"] and
                    r["candidate_mismatches_to_tiled"] == (candidate_diff["mismatches"] if provider == 2 else None) and
                    r["guards_intact"] == intact and r["same_consumer_counter_path_as_capture"] == same_counter_path and
                    r["passed"] == ok and r["actual_mask_filter_for_these_layers"] is False and r["fullmodel_speedup_proven"] is False,
                    "ordered TopK record agrees with complete raw bytes/counters")
            passed = ok and ranking["passed"]
            entry = self.gate("ordered_top512", prefix, passed, layer=layer, provider=provider, initial=initial, record_index=index,
                              mismatches_to_actual_capture=comparison, candidate_mismatches_to_tiled=candidate_diff,
                              guards_intact=intact, consumer_counter_path_matches=same_counter_path,
                              host_encoded=bool(r["host_encoded"]), GPU_completed=bool(r["GPU_completed"]),
                              score_input_unchanged_runtime_witness=r["score_input_unchanged"], ranking=ranking)
            self.id_reports.append(entry)
            self.topk_gate_passed &= passed
        input_seen = set()
        for index, r in enumerate(self.records):
            if r["kind"] != "real_inputs":
                continue
            layer, name, phase = r["layer"], r["input"], r["phase"]
            require(layer in self.cases and name in INPUTS and phase in
                    ("phase1_post_topk", "phase2_post_timing", "phase2_score_failure", "phase2_timestamp_failure",
                     "phase2_gpu_failure", "phase1_gpu_failure", "phase2_input_failure"), "known input witness scope")
            require((layer, name, phase) not in input_seen, "unique complete input phase witness")
            input_seen.add((layer, name, phase))
            captured = self.cases[layer]["payloads"][name]
            require(r["bytes"] == len(captured), "full original input extent")
            if not r["available"]:
                require(r["passed"] is False and phase not in ("phase1_post_topk", "phase2_post_timing"), "unallocated input only on partial failure")
                self.input_reports.append(self.gate("input_integrity", f"layer-{layer}-{phase}-{name}", False, available=False))
                continue
            stem = f"layer-{layer}-{phase}-{name}"
            require(r["post_raw"] == stem + "-post.bin" and r["guard_prefix_raw"] == stem + "-guard-prefix.bin" and
                    r["guard_suffix_raw"] == stem + "-guard-suffix.bin" and r["guard_bytes_per_side"] == 64, "complete input raw naming and guard extent")
            post = self.artifact(r["post_raw"])
            prefix, suffix = self.artifact(r["guard_prefix_raw"]), self.artifact(r["guard_suffix_raw"])
            require(len(post) == len(captured) and len(prefix) == len(suffix) == 64, "complete input body and both guard sides")
            equal, intact = post == captured, prefix == suffix == GUARD
            require(r["capture_sha256"] == sha(captured) and r["post_sha256"] == sha(post) and
                    r["guard_prefix_sha256"] == sha(prefix) and r["guard_suffix_sha256"] == sha(suffix) and
                    r["bytes_equal_capture"] == equal and r["guards_intact"] == intact and r["passed"] == (equal and intact),
                    "input witness record versus full raw bytes")
            self.input_reports.append(self.gate("input_integrity", stem, equal and intact, layer=layer, phase=phase,
                                               input=name, available=True, body_bytes=len(post), bytes_equal_capture=equal,
                                               guards_intact=intact, capture_sha256=sha(captured), post_sha256=sha(post)))
        complete_scores = phase1_scores == {(layer, p) for layer in LAYERS for p in range(3)}
        complete_ids = phase1_ids == {(layer, p) for layer in LAYERS for p in range(3)}
        complete_inputs = all((layer, name, "phase1_post_topk") in input_seen for layer in LAYERS for name in INPUTS)
        self.gate("coverage", "all_three_layers_all_providers_scores_ids_phase1_inputs", complete_scores and complete_ids and complete_inputs)
        phase1_inputs_pass = all(r["passed"] for r in self.input_reports if r.get("phase") == "phase1_post_topk")
        initial_scores_pass = all(r["passed"] for r in self.score_reports if r["initial"])
        initial_ids_pass = all(r["passed"] for r in self.id_reports if r["initial"])
        self.numerical_passed = complete_scores and complete_ids and complete_inputs and initial_scores_pass and initial_ids_pass and phase1_inputs_pass
        self.all_numerical_passed = self.numerical_passed and self.score_gate_passed and self.topk_gate_passed and all(r["passed"] for r in self.input_reports)
        gates = [(index, r) for index, r in enumerate(self.records) if r["kind"] == "all_layers_numerical_gate"]
        require(len(gates) <= 1, "one all-layer numerical gate")
        self.phase1_gate_index = gates[0][0] if gates else None
        if gates:
            gate = gates[0][1]
            require(gate["layers"] == list(LAYERS) and gate["complete_scores"] is True and gate["timing_started"] is False and
                    gate["passed"] == self.numerical_passed and gate["ordered_top512_all_providers_equal_capture"] == self.numerical_passed and
                    self.manifest["all_layers_numerical_gate_passed"] == self.numerical_passed, "all-layer declared gate versus independent raw gate")
            phase1_indices = [r["record_index"] for r in self.score_reports + self.id_reports if r["initial"]]
            phase1_indices += [index for index, r in enumerate(self.records) if r["kind"] == "real_inputs" and r["phase"] == "phase1_post_topk"]
            require(all(index < self.phase1_gate_index for index in phase1_indices), "all phase1 numerical/input records precede gate")
        elif self.manifest["status"] != "failed":
            raise EvidenceError("completed run lacks frozen all-layer numerical gate")
        self.input_seen = input_seen
        self.report.update(scores=self.score_reports, ordered_top512=self.id_reports, inputs=self.input_reports,
                           all_layers_numerical_gate_passed=self.numerical_passed,
                           all_phase1_and_retained_formal_numerical_witnesses_passed=self.all_numerical_passed,
                           complete_score_bytes_checked=sum(len(data) for data in self.initial_scores.values()),
                           complete_ordered_id_bytes_checked=sum(len(data) for data in self.initial_ids.values()),
                           input_raw_bytes_checked=sum(r.get("body_bytes", 0) + 128 for r in self.input_reports if r.get("available")))

    def check_timing(self):
        rows = [(index, r) for index, r in enumerate(self.records) if r["kind"] == "real_timing"]
        summaries = [r for r in self.records if r["kind"] == "real_summary"]
        if rows:
            require(self.numerical_passed and self.phase1_gate_index is not None and
                    all(index > self.phase1_gate_index for index, _ in rows), "all layers pass before first warmup/formal timing")
        expected = [(layer, control, sample, arm) for layer in LAYERS for control in (0, 1)
                    for sample in range(10) for arm in ("A1", "B", "A2")]
        actual = [(r["layer"], r["control"], r["sample"], r["arm"]) for _, r in rows]
        require(actual == expected[:len(actual)], "single immutable cohort order without replacement/duplicate samples")
        for index, r in rows:
            require(r["provider"] == (2 if r["arm"] == "B" else r["control"]) and
                    bool(r["warmup"]) == (r["sample"] == 0), "actual timing provider/arm/warmup")
            begin, end = r["gpu_start_s"], r["gpu_end_s"]
            require(math.isfinite(begin) and math.isfinite(end) and end > begin > 0 and
                    abs((end - begin) * 1e6 - r["gpu_us"]) < 1e-6, "raw GPU timestamp arithmetic")
            data = self.initial_scores[(r["layer"], r["provider"])]
            tiled = self.initial_scores[(r["layer"], 1)]
            require(r["full_scores_sha256"] == sha(data) and r["provider_full_scores_bit_exact_to_tiled"] == (data == tiled) and
                    r["candidate_full_scores_to_tiled_gate_passed"] is True and r["all_layers_numerical_gate_passed"] is True and
                    r["fullmodel_speedup_proven"] is False, "per-command complete score SHA/role-specific parity")
        groups, all_pass = [], len(rows) == 180 and len(summaries) == 6
        for layer in LAYERS:
            for control in (0, 1):
                subset = [r for _, r in rows if r["layer"] == layer and r["control"] == control]
                supplied = [r for r in summaries if r["layer"] == layer and r["control"] == control]
                require(len(supplied) <= 1, "one summary per layer/control")
                if len(subset) != 30:
                    require(not supplied, "summary requires all nine original formal groups")
                    all_pass = False
                    continue
                require(len(supplied) == 1, "complete group has summary")
                medians = {arm: statistics.median([r["gpu_us"] for r in subset if r["sample"] and r["arm"] == arm]) for arm in ("A1", "B", "A2")}
                a1, b, a2 = (medians[name] for name in ("A1", "B", "A2"))
                saved, drift = (1 - b / min(a1, a2)) * 100, (a2 / a1 - 1) * 100
                summary = supplied[0]
                for name, value in (("a1_us", a1), ("b_us", b), ("a2_us", a2),
                                    ("conservative_time_saved_percent", saved), ("control_drift_percent", drift)):
                    require(abs(summary[name] - value) < 1e-6, "independent summary recomputation: " + name)
                require(summary["ordered_top512_validated"] is True and summary["fullmodel_speedup_proven"] is False, "summary correctness scope")
                drift_pass, saving_pass = frozen_performance_gate(saved, drift, self.protocol["performance"])
                group = self.gate("performance", f"layer-{layer}-control-{control}", drift_pass and saving_pass,
                                  layer=layer, control=control, a1_us=a1, b_us=b, a2_us=a2,
                                  conservative_time_saved_percent=saved, control_drift_percent=drift,
                                  drift_gate_passed=drift_pass, strict_savings_gate_passed=saving_pass)
                groups.append(group)
                all_pass &= group["passed"]
        timing_complete = len(rows) == 180 and len(groups) == 6
        phase2_inputs = all((layer, name, "phase2_post_timing") in self.input_seen for layer in LAYERS for name in INPUTS)
        all_inputs_pass = all(r["passed"] for r in self.input_reports)
        if self.manifest["status"] == "passed_component_and_topk_only":
            require(timing_complete and phase2_inputs and all_inputs_pass and self.window["exit_code"] == 0,
                    "completed cohort has complete phase2 inputs and child exit zero")
        elif self.manifest["status"] == "failed":
            require(self.window["exit_code"] != 0, "failed child return code")
        else:
            raise EvidenceError("known real replay completion status")
        self.performance_passed = bool(all_pass and timing_complete and phase2_inputs and all_inputs_pass)
        self.report.update(performance_groups=groups, timing_records=len(rows), formal_timing_records=sum(r["sample"] > 0 for _, r in rows),
                           warmup_timing_records=sum(r["sample"] == 0 for _, r in rows), timing_cohort_complete=timing_complete,
                           phase2_input_witnesses_complete=phase2_inputs, all_input_witnesses_passed=all_inputs_pass,
                           all_six_frozen_performance_groups_passed=self.performance_passed)

    def run(self):
        self.protocol = read_json(self.args.protocol)
        self.capture = read_json(self.args.capture_verified)
        self.build = read_json(self.args.build_manifest)
        self.metal = read_json(self.args.metal_manifest)
        self.check_protocol()
        self.check_build()
        self.check_capture()
        if not self.check_window():
            return self.report
        self.load_outputs()
        self.check_numerical()
        self.check_timing()
        return self.finish()

    def finish(self):
        require(self.consumed == set(self.raw), "every manifest raw file independently consumed; no silent unexamined payload")
        eligible = self.all_numerical_passed and self.performance_passed and self.window_gate["passed"]
        if not self.numerical_passed:
            outcome = "frozen_component_numerical_gate_failed"
        elif not self.all_numerical_passed:
            outcome = "frozen_component_formal_numerical_gate_failed"
        elif self.manifest["status"] == "failed" and not self.report["timing_cohort_complete"]:
            outcome = "component_execution_failed_after_correctness_gate"
        elif not self.performance_passed:
            outcome = "frozen_component_performance_gate_failed_correctness_passed"
        elif not self.window_gate["passed"]:
            outcome = "recognized_process_exclusivity_gate_failed"
        else:
            outcome = "component_and_ordered_top512_gates_passed_fullmodel_unrun"
        self.report.update(review_completed=True, outcome=outcome, fullmodel_eligible=eligible,
                           experiment_manifest_status=self.manifest["status"], child_exit_code=self.window["exit_code"],
                           failures=self.issues, no_threshold_relaxation=True,
                           actual_mask_filter_for_these_layers=False,
                           per_timing_outputs_checked_by_sha_against_retained_full_initial_tables=True,
                           topk_score_input_unchanged_is_runtime_witness=True)
        return self.report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("capture-verified", "protocol", "build-manifest", "metal-manifest", "window-dir", "input-root", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--capture-root", type=Path)
    p.add_argument("--result-dir", type=Path)
    p.add_argument("--jsonl", type=Path)
    p.add_argument("--baseline-identity", type=Path)
    p.add_argument("--window-wrapper", type=Path)
    p.add_argument("--path-map", action="append", default=[])
    args = p.parse_args()
    require(not args.out.exists(), "new independent report path required")
    review = None
    code = 0
    try:
        review = Review(args)
        report = review.run()
    except (EvidenceError, KeyError, OSError, ValueError, TypeError, IndexError, StopIteration) as error:
        code = 2
        report = review.report if review else dict(schema_version=1, reader="independent_real_indexer_component_reader")
        report.update(review_completed=False, outcome="evidence_invalid", evidence_error_type=type(error).__name__,
                      evidence_error=str(error), fullmodel_eligible=False, fullmodel_speedup_proven=False, production_adoption=False)
        if review:
            report["failures_before_evidence_error"] = review.issues
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report.get(key) for key in ("review_completed", "outcome", "all_layers_numerical_gate_passed",
          "all_six_frozen_performance_groups_passed", "fullmodel_eligible", "evidence_error")}, indent=2))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
