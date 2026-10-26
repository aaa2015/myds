#!/usr/bin/env python3
"""Independent CPU qualification of same-binary G_lib/off/on full-model runs.

G_lib is a deployed-library anchor using the private reference C runtime. It
does not establish deployed-executable/source or HTTP wire equivalence. Only
final full-vocabulary logits and the schema2 final-state export are covered.
No model, subprocess, GPU, benchmark or builder is executed by this reader.
Revision2 accepts only the known macOS /tmp and /private/tmp loaded-library
alias while retaining the exact frozen path resolution and complete SHA gate.
"""
import argparse
import array
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys


class EvidenceError(ValueError):
    pass


def require(value, message):
    if not value:
        raise EvidenceError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def read_json(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=unique_object)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fnv1a(data):
    value = 14695981039346656037
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def regular_child(root, name):
    rel = Path(name)
    require(isinstance(name, str) and name and len(rel.parts) == 1 and name not in (".", "..") and
            not rel.is_absolute(), "single relative raw filename")
    path = Path(root) / rel
    require(not path.is_symlink() and path.is_file() and path.resolve().is_relative_to(Path(root).resolve()),
            "regular retained raw file: " + str(path))
    return path


def integer(value):
    return type(value) is int and value >= 0


SCALARS = ("ctx_pos", "rng_state", "max_seq_len", "max_new_tokens", "seed_u32", "kv_capacity", "pcache_len",
           "anchor_n", "continue_from_cache", "v41_valid", "next_position", "engram_position",
           "snapshot_allocated_bytes", "prefill_capacity", "low_kv_enabled", "anchor_slots", "anchor_cap")
COUNTERS = ("requests", "encoded", "fallbacks", "failures", "f32_dispatches", "bf16_dispatches")
SYMBOLS = ("kernel_myds_indexer_scores_k_resident_f32", "kernel_myds_indexer_scores_k_resident_bf16")
ORIGINAL_Q0_READER_SHA = "c6e9d4d2a2a35eb642e7b9fc062b7974a2a761a853bec147ddccf1b2965a5d70"


def macos_tmp_alias_or_identical(actual, expected):
    if actual == expected:
        return True
    # Only this documented macOS lexical alias is eligible. A caller's path
    # maps cannot make a different directory or basename satisfy this gate.
    if not isinstance(actual, str) or not isinstance(expected, str):
        return False
    for value in (actual, expected):
        if not value.startswith("/") or any(part in ("", ".", "..") for part in value.split("/")[1:]):
            return False
    return ((actual.startswith("/tmp/") and expected == "/private" + actual) or
            (expected.startswith("/tmp/") and actual == "/private" + expected))


def loaded_library_identity(actual, expected, library_path, library_sha256, mapped):
    require(macos_tmp_alias_or_identical(actual, expected) and
            mapped(actual).resolve() == mapped(library_path).resolve() and
            file_sha(mapped(actual)) == library_sha256, "actual loaded Metal library path and full SHA")


LIMITATIONS = ["Final live state and published snapshot arrays only; historical snapshot Engram tails are not exported.",
               "No persistent disk-cache payload, draft-model state, allocator padding, inactive capacity, scratch or intermediate logits."]
STATE_METADATA = {"byte_order", "fresh_context", "ctx_pos", "next_position", "window_rows", "layers", "window_width",
                  "window_order", "compressed_row_bytes", "index_row_bytes", "rng_state", "engram_position", "engram_tail",
                  "anchor_layout", "anchor_slots", "anchor_cap", "anchor_n", "scalar_order"}


def state_layout(scalars, lengths, positions, output_bytes, output_tokens, snapshot_slots=44):
    require(len(scalars) == len(SCALARS) and len(positions) == snapshot_slots * 2, "complete scalar/snapshot arrays")
    s = dict(zip(SCALARS, scalars))
    require(all(integer(value) for value in scalars) and s["next_position"] <= s["max_seq_len"] and
            0 <= s["anchor_slots"] <= 16 and len(lengths) == s["anchor_slots"] and
            all(integer(value) and value <= s["anchor_cap"] for value in lengths) and
            sum(value != 0 for value in lengths) == s["anchor_n"] and
            (not s["anchor_slots"] or s["anchor_cap"] > 0) and
            all(s[name] in (0, 1) for name in ("continue_from_cache", "v41_valid", "low_kv_enabled")),
            "bounded final state scalars and active multi-anchor layout")
    require(all(integer(value) and value <= s["next_position"] for value in positions[::2]) and
            all(integer(value) for value in positions[1::2]), "published snapshot bounds and complete staged position words")
    sizes = {"context-scalars.u64le": 8 * len(SCALARS), "engram-tail.u32le": 12,
             "snapshot-positions.u32le": 8 * snapshot_slots, "prefix-tokens.i32le": s["pcache_len"] * 4,
             "anchor-lengths.u32le": s["anchor_slots"] * 4, "tokens.i32le": output_tokens * 4, "output.bin": output_bytes}
    for slot, length in enumerate(lengths):
        sizes[f"anchor-{slot:02d}-tokens.i32le"] = length * 4
    for layer in range(40):
        sizes[f"layer-{layer:02d}-window-active.f32"] = min(s["next_position"], 128) * 512 * 4
    for owner in range(4):
        rows = s["next_position"] // (1 if owner == 3 else 2)
        if owner < 3:
            sizes[f"owner-{owner}-previous-kv.f32"] = 512 * 4
            sizes[f"owner-{owner}-previous-score.f32"] = 512 * 4
        sizes[f"owner-{owner}-compressed-active.bf16"] = rows * 1024
        sizes[f"owner-{owner}-index-active.bf16"] = rows * 256
    for slot, pos in enumerate(positions[::2]):
        if not pos:
            continue
        for layer in range(40):
            sizes[f"snapshot-{slot:02d}-layer-{layer:02d}-window-active.f32"] = min(pos, 128) * 512 * 4
        for owner in range(3):
            sizes[f"snapshot-{slot:02d}-owner-{owner}-previous-kv.f32"] = 512 * 4
            sizes[f"snapshot-{slot:02d}-owner-{owner}-previous-score.f32"] = 512 * 4
    return sizes


def provider_witness(row, expected_mode):
    require(row["api_available"] is True and type(row["mode"]) is int and row["mode"] == expected_mode and row["adoption_valid"] is True and
            row["first_callback_scope"] == "prefill_plus_first_output_callback" and
            row["counters_count_encoding_not_gpu_completion"] is True, "real scoped provider API and declared witness scope")
    for name in COUNTERS:
        total, first, decode = row[name], row["through_first_callback_" + name], row["decode_" + name]
        require(all(integer(value) for value in (total, first, decode)) and total >= first and decode == total - first,
                "complete monotone six-field provider counters: " + name)
    for prefix in ("", "through_first_callback_"):
        require(row[prefix + "requests"] >= row[prefix + "encoded"] and
                row[prefix + "encoded"] == row[prefix + "f32_dispatches"] + row[prefix + "bf16_dispatches"] and
                row[prefix + "fallbacks"] == row[prefix + "failures"] == 0, "candidate encoding accounting without fallback/failure")
    if expected_mode == 1:
        require((row["symbol_f32"], row["symbol_bf16"]) == SYMBOLS and row["through_first_callback_encoded"] > 0 and
                all(row["decode_" + name] == 0 for name in ("encoded", "f32_dispatches", "bf16_dispatches")),
                "actual candidate adoption before first callback and no decode candidate encoding")
    else:
        require(row["symbol_f32"] is None and row["symbol_bf16"] is None and
                all(row[name] == 0 for name in ("encoded", "f32_dispatches", "bf16_dispatches")), "reference mode never adopts candidate")
    return {name: row[name] for name in row}


def compare_file(left, right):
    """Compare all bytes, not just SHA; retain mismatch count and first offset."""
    left, right = Path(left), Path(right)
    sizes = (left.stat().st_size, right.stat().st_size)
    first, different, offset = None, abs(sizes[0] - sizes[1]), 0
    with left.open("rb") as a, right.open("rb") as b:
        while True:
            x, y = a.read(4 * 1024 * 1024), b.read(4 * 1024 * 1024)
            if not x and not y:
                break
            if x != y:
                if first is None:
                    first = offset + next((i for i, pair in enumerate(zip(x, y)) if pair[0] != pair[1]), min(len(x), len(y)))
                different += sum(l != r for l, r in zip(x, y))
            offset += max(len(x), len(y))
    return dict(equal=different == 0, reference_bytes=sizes[0], actual_bytes=sizes[1],
                mismatched_bytes=different, first_mismatch_byte=first)


def compare_exports(left, right):
    left_names, right_names = set(left["files"]), set(right["files"])
    same_files = left_names == right_names
    comparisons = {name: compare_file(left["files"][name], right["files"][name]) for name in sorted(left_names & right_names)}
    same_metadata = left["metadata"] == right["metadata"]
    finite = left.get("finite_logits") is True and right.get("finite_logits") is True
    return dict(passed=finite and same_files and same_metadata and all(x["equal"] for x in comparisons.values()),
                finite_logits_both=finite, metadata_equal=same_metadata, complete_file_set_equal=same_files,
                missing_state_files=sorted(left_names - right_names), additional_state_files=sorted(right_names - left_names), files=comparisons,
                differences=[name for name, value in comparisons.items() if not value["equal"]])


class PayloadReader:
    def __init__(self):
        self.fnv_cache = {}

    def payload(self, root, descriptor):
        path = regular_child(root, descriptor["path"])
        data = path.read_bytes()
        require(integer(descriptor["bytes"]) and len(data) == descriptor["bytes"], "complete state payload size")
        h = digest(data)
        key = len(data), h
        if key not in self.fnv_cache:
            self.fnv_cache[key] = fnv1a(data)
        require(self.fnv_cache[key] == descriptor["fnv1a64_bytes"], "full raw state payload FNV cross-check")
        return path, data, dict(bytes=len(data), sha256=h, fnv1a64_bytes=self.fnv_cache[key])

    def read_export(self, directory, logits_path, sample, vocab_size, snapshot_slots=44):
        root = Path(directory)
        m = read_json(root / "manifest.json")
        require(set(m) == {"schema_version", "subject", "run", "scope", "metadata", "limitations", "files", "file_count", "complete"} and
                m["schema_version"] == 2 and m["subject"] == "prefill_indexer_k_resident" and m["run"] == sample["run"] and
                m["scope"] == "fresh_context_full_prefill_final_state" and m["complete"] is True and m["limitations"] == LIMITATIONS,
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


def config_settings(raw):
    """Independent implementation of the frozen loader for this ordinary file."""
    settings, count = {}, 0
    require(b"\x00" not in raw, "config contains no NUL")
    for original in raw.decode().splitlines(keepends=True):
        require(len(original.encode()) < 4095, "no truncated native config fgets line")
        line = original.rstrip("\r\n \t")
        if not line.lstrip(" \t") or line.lstrip(" \t").startswith("#"):
            continue
        require("=" in line, "no skipped malformed config line")
        key, value = line.split("=", 1)
        key, value = key.strip(" \t"), value.lstrip(" \t")
        for i, char in enumerate(value):
            if char == "#" and i and value[i - 1] in " \t":
                value = value[:i]
                break
        value = value.rstrip(" \t")
        require(key.startswith("MYDS_") and key not in settings, "unique MYDS config setting without skipped entries")
        settings[key] = value
        count += 1
    return settings, count


def effective_configuration(indexer, metadata, raw, mode, logits_dir, state_dir, config_path, expected_count=62):
    base, count = config_settings(raw)
    require(count == expected_count and "MYDS_INDEXER_K_RESIDENT" not in base and
            "MYDS_BENCH_LOGITS_DIR" not in base and "MYDS_BENCH_STATE_DIR" not in base,
            "complete original config without overriding mode or evidence sinks")
    expected = dict(base, MYDS_INDEXER_K_RESIDENT=str(mode), MYDS_BENCH_LOGITS_DIR=str(logits_dir), MYDS_BENCH_STATE_DIR=str(state_dir))
    require(indexer["kind"] == "indexer_metadata" and indexer["schema_version"] == 1 and
            indexer["subject"] == "prefill_indexer_k_resident" and indexer["execute_requested"] is True and
            indexer["provider_adapter_available"] is True and indexer["fresh_context_per_sample"] is True and
            indexer["cache_configuration_preserved"] is True and indexer["config_load_override_env"] is True and
            indexer["config_file"] == str(config_path) and indexer["config_loaded_entries"] == count and
            indexer["config_file_bytes"] == len(raw) and indexer["config_file_fnv1a64"] == fnv1a(raw) and
            indexer["config_raw"].encode() == raw and indexer["effective_configuration_count"] == len(expected) and
            indexer["effective_configuration"] == expected, "all effective MYDS settings exactly original config plus mode and raw sinks")
    require({key: value for key, value in metadata["environment"].items() if value is not None} == expected,
            "base metadata independently records the same complete effective MYDS environment")
    require(expected["MYDS_PREFIX_CACHE"] == expected["MYDS_PLD"] == "1" and expected["MYDS_METAL_CONCURRENT"] == "1",
            "production cache/PLD and ordinary asynchronous Metal configuration")
    forbidden = {key for key in expected if key.startswith(("MYDS_PROFILE", "MYDS_TRACE", "MYDS_SKIP_"))}
    forbidden |= set(expected) & {"MYDS_KPROF", "MYDS_PHASE_PROFILE", "MYDS_PHASE_PROFILE_FILE", "MYDS_BATCH_TRACE_PREFIX",
        "MYDS_PROF", "MYDS_DISPATCH_HIST", "MYDS_DEBUG_DISPATCH", "MYDS_MOE_STEP_PROF", "MYDS_DISABLE_V41_PREFILL",
        "MYDS_DSPARK_CAPTURE_BYPASS", "MYDS_ENGINE_CAPTURE_BYPASS", "MYDS_BENCH_STORAGE_COUNTERS"}
    require(not forbidden and expected.get("MYDS_BATCH2_TRACE_GPU", "0") == "0", "no profiler/trace/capture/skip/storage observer selector")
    return dict(config_entries=count, effective_entries=len(expected), effective=expected,
                original_config_sha256=digest(raw), no_unlisted_MYDS=True, normal_async_selectors=True)


def sample_contract(sample, expected_run, prompt_tokens, output_tokens):
    require(sample["kind"] == "sample" and sample["run"] == expected_run and sample["warmup"] is (expected_run == 0) and
            sample["fresh_context"] is True, "one original warmup then one fresh-context verification sample")
    require(sample["prompt_tokens"] == prompt_tokens and sample["cached_tokens"] == 0 and sample["prefill_tokens"] == prompt_tokens - 1 and
            sample["output_tokens"] == output_tokens and sample["decode_intervals"] == output_tokens - 1 and
            len(sample["token_ids"]) == output_tokens, "complete full-prefill/fixed-output/N-1 decode workload")
    for name in ("prefill_s", "wall_s", "ttft_s", "first_callback_s", "last_callback_s", "decode_s", "decode_ms_per_token"):
        require(type(sample[name]) in (float, int) and math.isfinite(sample[name]) and sample[name] > 0, "finite positive native timing: " + name)
    require(0 < sample["first_callback_s"] < sample["last_callback_s"] <= sample["wall_s"] + 2e-9 and
            abs(sample["ttft_s"] - sample["first_callback_s"]) <= 2e-9 and
            abs(sample["decode_s"] - (sample["last_callback_s"] - sample["first_callback_s"])) <= 2e-9 and
            abs(sample["decode_ms_per_token"] - sample["decode_s"] * 1000 / (output_tokens - 1)) <= 2e-6 and
            len(sample["callback_intervals_ms"]) == output_tokens - 1 and
            all(type(v) in (float, int) and math.isfinite(v) and v > 0 for v in sample["callback_intervals_ms"]) and
            abs(sum(sample["callback_intervals_ms"]) - sample["decode_s"] * 1000) <= 2e-6,
            "original callback timing arithmetic without replacement")
    require(sample["generation_ok"] is True and sample["stop_reason"] == "length" and sample["stop_reason_code"] == 1,
            "successful fixed-count generation (early EOS is not an applicable fixed-output workload)")
    return dict(run=expected_run, warmup=expected_run == 0, fresh_context=True, full_prefill=True,
                prompt_tokens=prompt_tokens, prefill_tokens=prompt_tokens - 1, output_tokens=output_tokens,
                decode_intervals=output_tokens - 1, performance_acceptance=False)


def qualification_comparisons(exports):
    require(set(exports) == {"G_lib", "off", "on"} and all(set(runs) == {0, 1} for runs in exports.values()),
            "exactly three same-binary providers with separate warmup/verification evidence")
    repeated = {name: compare_exports(runs[0], runs[1]) for name, runs in exports.items()}
    bridge = {str(run): compare_exports(exports["G_lib"][run], exports["off"][run]) for run in (0, 1)}
    candidate = {str(run): compare_exports(exports["off"][run], exports["on"][run]) for run in (0, 1)}
    bridge_passed = all(row["passed"] for row in bridge.values())
    candidate_passed = all(row["passed"] for row in candidate.values())
    repeat_passed = all(row["passed"] for row in repeated.values())
    passed = bridge_passed and candidate_passed and repeat_passed
    return dict(qualification_passed=passed, qualified_to_prepare_separate_long_protocol=passed,
                formal_performance_cohort_eligible=False, long_workload_qualification_completed=False,
                stage_next_requires_separate_freeze=True,
                gold_library_bridge_passed=bridge_passed, candidate_reference_passed=candidate_passed,
                within_provider_repeatability_passed=repeat_passed, within_provider=repeated,
                gold_library_bridge=bridge, candidate_vs_same_library_reference=candidate,
                G_lib_identity_scope="same private executable/reference C with deployed library; not deployed executable/source",
                deployed_executable_wire_bridge_completed=False, production_acceptance=False,
                fullmodel_speedup_proven=False, performance_acceptance=False)


def inventory_digest(inventory):
    return digest(json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode())


def verify_inventory(root, inventory):
    actual = {str(path.relative_to(root)) for path in Path(root).rglob("*") if path.is_file() and "__pycache__" not in path.parts}
    require(actual == set(inventory), "complete compiler-visible snapshot inventory without omitted extra files")
    for name, item in inventory.items():
        rel = Path(name)
        require(name and not rel.is_absolute() and ".." not in rel.parts, "relative compiler-visible source")
        path = Path(root) / rel
        require(not path.is_symlink() and path.is_file() and path.resolve().is_relative_to(Path(root).resolve()) and
                path.stat().st_size == item["bytes"] and file_sha(path) == item["sha256"], "complete frozen source input: " + name)


REFERENCE_SIGNATURES = {"static int indexscores_chunk(", "int myds_v41_gpu_indexscores_chunk(", "int myds_v41_gpu_indexscores_chunk_bf16("}


def complete_c_function(data, signature):
    anchor = signature.encode()
    require(data.count(anchor) == 1, "unique native selector definition")
    start = data.index(anchor)
    # Remove literals/comments while retaining offsets before matching braces.
    cleaned = re.sub(rb'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|/\*[\s\S]*?\*/|//[^\n]*',
                     lambda match: b" " * len(match[0]), data)
    opening = cleaned.find(b"{", start)
    require(opening > start and b";" not in cleaned[start:opening], "native definition rather than prototype")
    level = 0
    for index in range(opening, len(cleaned)):
        if cleaned[index] == ord("{"):
            level += 1
        elif cleaned[index] == ord("}"):
            level -= 1
            if level == 0:
                return data[start:index + 1]
    raise EvidenceError("complete native selector closing brace")


def verify_reference_proofs(old, new, proofs):
    require(set(proofs) == REFERENCE_SIGNATURES, "all three complete native reference selector proofs")
    for signature, item in proofs.items():
        before, after = complete_c_function(old, signature), complete_c_function(new, signature)
        require(item["byte_exact"] is True and len(before) == item["bytes"] and before == after and digest(before) == item["sha256"],
                "complete native reference selector retained byte-for-byte")


def verify_core_proofs(proofs, seed):
    expected = {name for name in seed if name.startswith(("src/", "include/", "third_party/")) and Path(name).suffix in (".c", ".m", ".h", ".inc")}
    require(set(proofs) == expected and all(seed[name] == item for name, item in proofs.items()), "complete matched base compiler core proof set")


def verify_private_build(path, mapped):
    b = read_json(path)
    root = Path(path).parent
    seed = read_json(root / "seed-manifest.json")
    # source_root/seed_source_root describe provenance, not a license to read
    # the live checkout. Only the retained sibling snapshots are authoritative.
    source, original = root / "source", root / "seed"
    require(b["schema_version"] == 1 and b["kind"] == "private_prefill_indexer_k_resident_runtime_build" and
            b["compile_only_pass"] is True and b["gpu_work_submitted"] is False and
            b["production_worktree_mutated"] is False and b["production_shader_registration_changed"] is False and
            b["normal_preprocessor_observer_absent"] is True and b["all_shader_source_and_builder_bytes_equal_seed"] is True and
            b["counter_fields"] == list(COUNTERS) and b["symbols"] == dict(f32=SYMBOLS[0], bf16=SYMBOLS[1]), "private default-off ordinary-prefill runtime construction")
    require(file_sha(root / "seed-manifest.json") == b["seed_manifest_sha256"] and
            inventory_digest(b["source_inventory"]) == b["complete_private_source_inventory_sha256"] and
            inventory_digest(seed["files"]) == b["complete_seed_inventory_sha256"] == seed["complete_seed_inventory_sha256"],
            "complete source/seed manifest identities")
    verify_inventory(source, b["source_inventory"]); verify_inventory(original, seed["files"])
    changed = sorted(name for name in set(b["source_inventory"]) | set(seed["files"])
                     if b["source_inventory"].get(name) != seed["files"].get(name))
    require(changed == sorted(b["changed_paths"]) == ["include/myds_v41_gpu.h", "src/myds_v41.c", "src/myds_v41_gpu.c",
                "tests/test_indexer_k_publisher_host.c", "tests/test_indexer_k_runtime_host.c", "tools/indexer_k_runtime_adapter.inc"],
            "only listed private runtime/API/adapter/host-fixture edits")
    verify_core_proofs(b["matched_source_core_files_identical"], seed["files"])
    matched_report = mapped(b["matched_component_review_path"])
    require(file_sha(matched_report) == b["matched_component_review_sha256"] and
            read_json(matched_report)["fullmodel_eligible"] is True, "independent matched component eligibility report identity")
    for name in [n for n in seed["files"] if n.startswith("metal/")] + ["tools/build_metallib.py"]:
        require(seed["files"][name] == b["source_inventory"][name], "shader and original builder source unchanged")
    old_gpu, new_gpu = (original / "src/myds_v41_gpu.c").read_bytes(), (source / "src/myds_v41_gpu.c").read_bytes()
    verify_reference_proofs(old_gpu, new_gpu, b["reference_function_byte_proofs"])
    old_caller, new_caller = (original / "src/myds_v41.c").read_bytes(), (source / "src/myds_v41.c").read_bytes()
    old = b"        if (!myds_v41_gpu_indexscores_chunk_bf16(scores_buf,q_view,w_view,g->index_cache[owner],\n                nc,V41_INDEX_HEADS,glen,g_start,ratio)) return -1;"
    new = b"        if (!myds_v41_gpu_indexscores_prefill_chunk_scoped(g->gpu,scores_buf,q_view,w_view,g->index_cache[owner],\n                nc,V41_INDEX_HEADS,glen,g_start,ratio,true,!myds_trace_active())) return -1;"
    require(old_caller.count(old) == new_caller.count(new) == 1 and new_caller.replace(new, old, 1) == old_caller,
            "unique ordinary-prefill publishing call is the only caller edit")
    for name, item in b["patches"].items():
        p = regular_child(root, name)
        require(p.stat().st_size == item["bytes"] and file_sha(p) == item["sha256"], "retained exact private patch identity")
    for name, item in b["checks"].items():
        p = regular_child(root, name)
        require(p.stat().st_size == item["bytes"] and file_sha(p) == item["sha256"], "retained CPU construction evidence")
    for kind in ("binary", "library"):
        item = b[kind]; p = mapped(item["path"])
        require(p.stat().st_size == item["bytes"] and file_sha(p) == item["sha256"], "private CPU build " + kind + " identity")
    require(file_sha(mapped(b["library"]["source_identity_path"])) == b["library"]["source_identity_sha256"], "approved complete-TU Metal construction binding")
    return dict(manifest_sha256=file_sha(path), source_inventory_sha256=b["complete_private_source_inventory_sha256"],
                seed_inventory_sha256=b["complete_seed_inventory_sha256"], source_files_verified=len(b["source_inventory"]),
                seed_files_verified=len(seed["files"]), matched_core_files_verified=len(b["matched_source_core_files_identical"]),
                changed_paths=changed, private_build=b)


def read_run_stdout(path, expected_prompt_tokens, expected_output_tokens=64):
    lines = Path(path).read_text().splitlines()
    require(lines and all(line.strip() and line.startswith("{") for line in lines), "fullmodel stdout contains strict original JSONL only")
    records = [json.loads(line, object_pairs_hook=unique_object) for line in lines]
    require([row["kind"] for row in records] == ["indexer_metadata", "metadata", "load", "sample", "sample"],
            "exactly metadata/load/warmup0/verification1 without hidden replacement")
    indexer, metadata, loaded, *samples = records
    require(metadata["subject"] == "prefill_indexer_k_resident" and metadata["protocol_version"] == 3 and
            metadata["repeats"] == metadata["warmup_runs"] == 1 and metadata["max_new_tokens"] == expected_output_tokens and
            metadata["temperature"] == 0 and metadata["top_p"] == 1 and metadata["top_k"] == 0 and
            metadata["min_p"] == 0 and metadata["repetition_penalty"] == 1 and metadata["seed"] == 42 and
            metadata["qos"] == "user_interactive" and metadata["qos_result"] == 0 and
            metadata["prompt_renderer"] == "parse_chat_prompt" and metadata["clock"] == "CLOCK_MONOTONIC" and
            metadata["transport"] == "direct_byte_callback" and metadata["decode_basis"] == "first_to_last_callback/(tokens-1)" and
            metadata["dspark_model"] is None, "unchanged direct greedy seed42 driver and no DSpark model")
    require(loaded["native_v41"] is True and loaded["layers"] == 40 and loaded["dspark_loaded"] is False and
            loaded["fresh_context_per_sample"] is True and loaded["cache_configuration_preserved"] is True and
            integer(loaded["vocab_size"]) and loaded["vocab_size"] > 0, "native V4.1 loaded model and fresh context contract")
    for run, sample in enumerate(samples):
        sample_contract(sample, run, expected_prompt_tokens, expected_output_tokens)
        require(type(sample["sample_valid"]) is bool and type(sample["output_equal_to_warmup"]) is bool and
                sample["gpu_indexer_provider"]["run"] == run, "sample native declarations and provider run identity")
    return indexer, metadata, loaded, samples


def read_run_exports(directory, stdout_path, mode, prompt_tokens, output_tokens, payload_reader=None):
    """Read every payload even when a cross-arm numerical comparison fails."""
    directory = Path(directory)
    indexer, metadata, loaded, samples = read_run_stdout(stdout_path, prompt_tokens, output_tokens)
    logits, state = directory / "logits", directory / "state"
    require(not logits.is_symlink() and not state.is_symlink() and logits.is_dir() and state.is_dir() and
            {p.name for p in logits.iterdir()} == {"run-0.f32", "run-1.f32"} and
            {p.name for p in state.iterdir()} == {"run-0", "run-1"}, "exact complete warmup/verification raw evidence roots")
    payload_reader = payload_reader or PayloadReader()
    exports, sample_reports = {}, []
    for sample in samples:
        run = sample["run"]
        witness = provider_witness(sample["gpu_indexer_provider"], mode)
        exported = payload_reader.read_export(state / f"run-{run}", logits / f"run-{run}.f32", sample, loaded["vocab_size"])
        require(exported["metadata"]["next_position"] <= metadata["max_seq_len"] and
                exported["decoded_scalars"]["max_seq_len"] == metadata["max_seq_len"] and
                exported["decoded_scalars"]["max_new_tokens"] == metadata["max_new_tokens"] and
                exported["decoded_scalars"]["seed_u32"] == metadata["seed"], "decoded state context/output/seed match actual driver request")
        exports[run] = exported
        sample_reports.append(dict(run=run, warmup=sample["warmup"], native_sample_valid=sample["sample_valid"],
            native_output_equal_to_warmup=sample["output_equal_to_warmup"], provider=witness,
            manifest_sha256=exported["manifest_sha256"], raw_files={name: evidence for name, evidence in exported["evidence"].items()},
            raw_bytes=exported["payload_bytes"], final_logits_finite=exported["finite_logits"],
            native_timing_descriptive_only={name: sample[name] for name in ("prefill_s", "ttft_s", "decode_s", "decode_ms_per_token", "wall_s")}))
    repeat = compare_exports(exports[0], exports[1])
    native_valid = all(s["sample_valid"] is True and s["output_equal_to_warmup"] is True for s in samples)
    return dict(indexer_metadata=indexer, metadata=metadata, loaded=loaded, samples=sample_reports,
                exports=exports, within_provider_comparison=repeat, native_declared_valid=native_valid,
                sample_contracts_passed=True, provider_adoption_passed=True, stdout_sha256=file_sha(stdout_path))


class Review:
    def __init__(self, args):
        self.args, self.maps = args, []
        for value in args.path_map:
            require("=" in value, "path map OLD=NEW")
            old, new = value.split("=", 1)
            require(Path(old).is_absolute() and Path(new).is_absolute(), "absolute path map")
            self.maps.append((Path(old), Path(new)))
        self.maps.sort(key=lambda pair: len(str(pair[0])), reverse=True)
        self.report = dict(schema_version=1, reader="independent_indexer_fullmodel_qualification_reader_v2", reader_revision="2_macos_tmp_alias_only", reader_sha256=file_sha(__file__),
            review_completed=False, qualification_passed=False, formal_performance_cohort_eligible=False,
            qualified_to_prepare_separate_long_protocol=False, long_workload_qualification_completed=False,
            stage_next_requires_separate_freeze=True,
            production_acceptance=False, production_adoption=False, performance_acceptance=False, fullmodel_speedup_proven=False,
            deployed_executable_wire_bridge_completed=False, scope="same_private_binary_G_lib_off_on_final_logits_and_schema2_state",
            G_lib_identity_scope="deployed library anchor with private reference C; not deployed executable/source",
            limitations=LIMITATIONS + ["No per-output-token/intermediate logits or HTTP deployed-executable wire bridge.",
                "Qualification has no performance acceptance, cold-storage claim or production acceptance."],
            path_maps=[dict(original=str(a), retained_copy=str(b)) for a, b in self.maps])
        self.payload_reader = PayloadReader()

    def mapped(self, value):
        path = Path(value)
        for old, new in self.maps:
            if path == old or path.is_relative_to(old):
                return new / path.relative_to(old)
        return path

    def descriptor(self, item):
        path = self.mapped(item["path"])
        require(not path.is_symlink() and path.is_file() and path.stat().st_size == item["bytes"] and file_sha(path) == item["sha256"],
                "complete frozen artifact bytes/SHA: " + str(path))
        return path

    def initialize(self):
        p = self.protocol = read_json(self.args.protocol)
        require(p["schema_version"] == 1 and p["kind"] == "indexer_fullmodel_q0_qualification_protocol" and p["stage"] == "Q0" and
                [(row["name"], row["mode"]) for row in p["providers"]] == [("G_lib", 0), ("off", 0), ("on", 1)],
                "independently frozen one Q0 G_lib/off/on qualification stage")
        c = p["request_contract"]
        require(all(c[name] == value for name, value in dict(prompt_tokens=7540, output_tokens=64, repeats=1, warmup_runs=1,
                verification_runs=1, max_seq_len=327680, temperature=0, top_p=1, top_k=0, min_p=0, repetition_penalty=1,
                seed=42, token_format="little_endian_int32_raw", request_wire_changed=False,
                renderer="parse_chat_prompt", tokenizer="target_gguf_metadata", qualification_generation_count=6).items()),
                "one fixed business24 qualification request, warmup1/verify1/64 outputs")
        require(p["performance_acceptance"] is False and p["fullmodel_speedup_proven"] is False and p["production_acceptance"] is False and
                p["no_retry_or_replacement"] is True and p["no_automatic_long_workload_or_performance_or_HTTP_stage"] is True and
                p["classification"]["G_lib_is_deployed_executable_or_source"] is False and
                p["classification"]["deployed_executable_wire_bridge_completed"] is False and
                p["correctness"]["whole_model_payload_sha256_verified"] is False and
                p["correctness"]["selected_symbols"] == list(SYMBOLS), "qualification scope never grants production or performance acceptance")
        self.artifacts = {name: self.descriptor(item) for name, item in p["artifacts"].items()}
        self.execution_reader_sha = p["artifacts"]["reader"]["sha256"]
        require(self.execution_reader_sha in (file_sha(__file__), ORIGINAL_Q0_READER_SHA), "frozen execution reader or explicit Q0 revision1 source identity")
        self.report.update(execution_reader_sha256=self.execution_reader_sha,
            reviewer_revision2_source_sha256=file_sha(__file__),
            revision1_source_retained_unchanged=self.execution_reader_sha == ORIGINAL_Q0_READER_SHA,
            reader_repair_scope="Only known macOS tmp loaded-library alias; all numerical/state/provider/environment/protocol gates unchanged")
        require(self.artifacts["private_build_manifest"].resolve() == self.args.private_build_manifest.resolve() and
                self.artifacts["target_build_manifest"].resolve() == self.args.target_build_manifest.resolve() and
                self.mapped(p["window_root"]).resolve() == self.args.window_root.resolve(), "actual reader inputs match frozen stage artifacts")
        proof = verify_private_build(self.args.private_build_manifest, self.mapped)
        private = proof.pop("private_build")
        self.private = private
        self.check_target_build()
        require(all(row["binary"]["sha256"] == private["binary"]["sha256"] for row in p["providers"]), "all Q0 providers use the identical frozen private executable")
        for row in p["providers"]:
            self.descriptor(row["binary"]); self.descriptor(row["library"])
        require(p["providers"][1]["library"]["sha256"] == p["providers"][2]["library"]["sha256"] == private["library"]["sha256"] and
                p["providers"][0]["library"]["sha256"] != private["library"]["sha256"], "deployed-library bridge followed by same-library off/on comparison")
        request, prompt, tokens = (self.artifacts[key].read_bytes() for key in ("request", "rendered_prompt", "rendered_tokens"))
        require(b"\0" not in request and b"\0" not in prompt and len(tokens) == c["prompt_tokens"] * 4,
                "complete request/rendered-prompt/all-rendered-token evidence")
        self.request, self.prompt, self.rendered_tokens = request, prompt, tokens
        baseline = read_json(self.artifacts["baseline_identity"])
        require(all(name in baseline and all(item[key] == baseline[name][key] for key in ("bytes", "sha256"))
                    for name, item in p["canonical_files"].items()) and p["model"] == baseline["model"], "current Q0 canonical/model expectations retain frozen deployment identity")
        deployed_libraries = [item for name, item in p["canonical_files"].items() if Path(name).name == "myds.metallib"]
        require(len(deployed_libraries) == 1 and deployed_libraries[0]["sha256"] == p["providers"][0]["library"]["sha256"],
                "G_lib is the preserved deployed Metal library anchor")
        tokenizer = read_json(self.artifacts["tokenizer"])
        require(tokenizer["prompt_renderer"] == c["renderer"] and tokenizer["tokenizer"] == c["tokenizer"] and tokenizer["add_bos"] is True and
                tokenizer["prompt_tokens"] == c["prompt_tokens"] and tokenizer["prompt_bytes"] == len(prompt) and tokenizer["request_bytes"] == len(request) and
                tokenizer["token_encoding"] == "little_endian_int32" and tokenizer["model_loader_called"] is False and tokenizer["metal_initialized"] is False and
                struct.unpack("<i", tokens[:4])[0] == tokenizer["bos_id"], "complete frozen target-metadata tokenizer/rendered-input contract")
        self.report.update(protocol_sha256=file_sha(self.args.protocol), stage="Q0", private_build=proof,
            private_build_manifest_sha256=file_sha(self.args.private_build_manifest), target_build_manifest_sha256=file_sha(self.args.target_build_manifest),
            artifacts_verified={name: dict(path=str(path), bytes=path.stat().st_size, sha256=file_sha(path)) for name, path in self.artifacts.items()},
            request_contract=c, binary_sha256=private["binary"]["sha256"], provider_library_sha256={row["name"]: row["library"]["sha256"] for row in p["providers"]})

    def check_target_build(self):
        b = read_json(self.args.target_build_manifest)
        require(b["schema_version"] == 1 and b["kind"] == "M2_private_indexer_k_runtime_target_build" and
                b["compile_only_pass"] is True and b["gpu_work_submitted"] is False and b["production_deployed"] is False and
                b["source_inventory_sha256"] == self.private["complete_private_source_inventory_sha256"] and
                b["source_build_manifest"]["sha256"] == file_sha(self.args.private_build_manifest), "target CPU build from exact private source inventory")
        for key in ("source_build_manifest", "source_check", "runtime_core_patch", "binary", "gold_binary"):
            self.descriptor(b[key])
        for item in list(b["libraries"].values()) + list(b["build_logs"].values()):
            self.descriptor(item)
        require(b["binary"]["sha256"] == b["gold_binary"]["sha256"] == self.private["binary"]["sha256"] and
                b["runtime_core_patch"]["sha256"] == self.private["patches"]["runtime-core.patch"]["sha256"] and
                b["libraries"]["private"]["sha256"] == self.private["library"]["sha256"], "same executable/private patch/approved complete-TU library")
        source = read_json(self.mapped(b["source_check"]["path"]))
        require(source["kind"] == "M2_frozen_private_source_prebuild" and source["source_files_verified"] == len(self.private["source_inventory"]) and
                source["source_inventory_sha256"] == self.private["complete_private_source_inventory_sha256"] and source["gpu_work_submitted"] is False,
                "target prebuild complete source witness")
        require(len(b["commands"]) == 1 and b["commands"][0][:4] == ["make", "-f", "tools/indexer_fullmodel.mk", "-j4"] and
                "INDEXER_ADAPTER_HEADER=tools/indexer_k_runtime_adapter.inc" in b["commands"][0], "frozen ordinary fullmodel target build and real adapter")
        self.report["target_build_verified"] = dict(manifest_sha256=file_sha(self.args.target_build_manifest), source_check=source, commands=b["commands"])

    def check_snapshot(self, actual, expected, model=False):
        require(actual["path"] == expected["path"] and actual["bytes"] == expected["bytes"] and integer(actual["device"]) and
                integer(actual["inode"]) and integer(actual["mtime_ns"]) and Path(actual["canonical_path"]).is_absolute(), "actual target artifact stat/path")
        if model:
            require(all(actual[name] == expected[name] for name in ("inode", "mtime_ns", "prefix_bytes", "prefix_sha256")) and
                    actual["whole_model_payload_sha256_verified"] is False, "actual target model stat and complete identity prefix; no whole-model SHA claim")
        else:
            require(actual["sha256"] == expected["sha256"], "actual canonical artifact SHA")

    def check_window_common(self, window, allowed_abort=None):
        p = self.protocol
        require(window["protocol_sha256"] == file_sha(self.args.protocol) and window["wrapper_sha256"] == p["wrapper_sha256"] and
                window["stage"] == "Q0" and window["abort_reason"] in (None, allowed_abort) and window["canonical_unchanged"] is True and
                window["canonical_before"] == window["canonical_after"] and window["model_unchanged"] is True and
                window["model_before"] == window["model_after"] and not window["recognized_before"] and not window["recognized_after"],
                "unchanged exclusive Q0 canonical/model window with no abort")
        require(set(window["canonical_before"]) == set(p["canonical_files"]), "all frozen canonical artifacts observed")
        for name, item in p["canonical_files"].items():
            self.check_snapshot(window["canonical_before"][name], item)
        self.check_snapshot(window["model_before"], p["model"], model=True)

    def check_inventory(self, directory, window):
        p = Path(directory) / "raw-inventory.json"
        inventory = read_json(p)
        require(file_sha(p) == window["raw_inventory_sha256"] and inventory["schema_version"] == 1 and
                inventory["whole_payload_files_rehashed"] is True and inventory["tree_sha256"] == window["raw_tree_sha256"] ==
                inventory_digest(inventory["files"]) and inventory["file_count"] == len(inventory["files"]) and
                inventory["logical_bytes"] == sum(item["bytes"] for item in inventory["files"].values()), "post-run retained raw SHA inventory binding")
        actual = {"stdout.jsonl", "stderr.txt"}
        for name in ("logits", "state"):
            for path in (Path(directory) / name).rglob("*"):
                require(not path.is_symlink(), "raw evidence never follows symlinks")
                if path.is_file():
                    actual.add(str(path.relative_to(directory)))
        require(actual == set(inventory["files"]), "all and only raw stdout/stderr/logits/state files in retained inventory")
        for name, item in inventory["files"].items():
            rel = Path(name)
            require(not rel.is_absolute() and ".." not in rel.parts, "local raw inventory path")
            path = Path(directory) / rel
            require(path.is_file() and not path.is_symlink() and path.stat().st_size == item["bytes"] and file_sha(path) == item["sha256"],
                    "complete raw inventory bytes/SHA: " + name)
        return dict(files=len(actual), logical_bytes=inventory["logical_bytes"], inventory_sha256=file_sha(p), tree_sha256=inventory["tree_sha256"])

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
        require(plan["stage"] == "Q0" and plan["provider"] == window["provider"] == provider["name"] and
                plan["mode"] == window["mode"] == provider["mode"] and plan["argv"] == expected_argv and
                plan["cwd"] == str(expected_root / "cwd") and plan["MYDS_inherited"] == expected_env and
                plan["protocol_sha256"] == file_sha(self.args.protocol) and plan["wrapper_sha256"] == p["wrapper_sha256"] and
                plan["reader_sha256"] == window["reader_sha256"] == self.execution_reader_sha and
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
        result = dict(provider=provider["name"], mode=provider["mode"], child_pid=child, exit_code=window["exit_code"],
            started_epoch=window["started_epoch"], finished_epoch=window["finished_epoch"], window_gates_passed=True,
            raw_inventory=inventory, execution_complete=window["exit_code"] == 0,
            evidence_sha256={name: file_sha(directory / name) for name in ("execution-plan.json", "window.json", "monitor.json")},
            actual_argv=plan["argv"], window=window)
        try:
            parsed = read_run_exports(directory, directory / "stdout.jsonl", provider["mode"], 7540, 64, self.payload_reader)
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
        p, root = self.protocol, self.args.window_root
        require(file_sha(self.mapped(p["wrapper_path"])) == p["wrapper_sha256"], "frozen collection-only process wrapper")
        plan, window = (read_json(root / name) for name in ("execution-plan.json", "window.json"))
        expected_names = [row["name"] for row in p["providers"]]
        require(plan["stage"] == "Q0" and plan["run_order"] == window["run_order"] == expected_names and
                plan["protocol_sha256"] == file_sha(self.args.protocol) and plan["wrapper_sha256"] == p["wrapper_sha256"] and
                plan["reader_sha256"] == window["reader_sha256"] == self.execution_reader_sha and
                plan["shared_lock_path"] == p["lock_path"] and plan["timeout_s"] == p["timeout_s"] and plan["performance_acceptance"] is False,
                "prelaunch ordered Q0 protocol/wrapper/reader/lock binding")
        actual_names = [row["provider"] for row in window["providers"]]
        require(actual_names and actual_names == expected_names[:len(actual_names)] and len(actual_names) <= 3,
                "one finite ordered provider cohort without retry or replacement")
        require({path.name for path in root.iterdir()} == {"execution-plan.json", "window.json", "model-prefix.bin"} | set(actual_names),
                "one complete outer evidence root without unregistered provider retries")
        allowed_abort = "provider collection stopped: " + actual_names[-1]
        self.check_window_common(window, allowed_abort=allowed_abort)
        require(window["lock_held_through_final_checks"] is True and window["service_or_production_mutation_performed"] is False and
                window["external_reader_required"] is True and window["qualification_passed"] is None and
                window["fullmodel_speedup_proven"] is False and window["production_acceptance"] is False and window["performance_acceptance"] is False,
                "collection completion never grants qualification, performance or production acceptance")
        prefix = self.descriptor(window["model_prefix"])
        require(window["model_prefix"]["path"] == p["model_prefix_path"] and prefix.stat().st_size == p["model"]["prefix_bytes"] == 1048576 and
                file_sha(prefix) == p["model"]["prefix_sha256"], "independently rehashed complete raw target model identity prefix")
        results, exports = [], {}
        self.report["providers"] = results
        for provider in p["providers"][:len(actual_names)]:
            result, payloads = self.review_provider(provider)
            require(result["window"] == window["providers"][len(results)], "outer and per-provider window results identical")
            if results:
                require(result["child_pid"] not in {row["child_pid"] for row in results} and result["started_epoch"] >= results[-1]["finished_epoch"],
                        "three distinct serial fresh processes")
            results.append(result)
            if payloads is not None:
                exports[provider["name"]] = payloads
        execution_complete = len(results) == 3 and all(row["execution_complete"] for row in results) and window["abort_reason"] is None
        require(window["execution_complete"] is execution_complete and plan["started_epoch"] <= results[0]["started_epoch"] and
                results[-1]["finished_epoch"] <= window["finished_epoch"], "outer completion/count and ordered chronology")
        self.report.update(execution_complete=execution_complete, outer_window_gates_passed=True, model_identity_scope=p["model"],
            model_prefix_sha256=file_sha(prefix), whole_model_payload_sha256_verified=False,
            window_evidence_sha256={name: file_sha(root / name) for name in ("execution-plan.json", "window.json")},
            complete_provider_count=len(exports), native_generation_count=sum(len(row.get("samples", [])) for row in results),
            raw_inventory_files=sum(row["raw_inventory"]["files"] for row in results),
            raw_logical_bytes=sum(row["raw_inventory"]["logical_bytes"] for row in results))
        if len(exports) == 3:
            comparison = qualification_comparisons(exports)
            valid = comparison["qualification_passed"] and execution_complete and all(row["native_declared_valid"] for row in results)
            comparison.update(qualification_passed=valid, qualified_to_prepare_separate_long_protocol=valid)
            self.report.update(**comparison)
            outcome = "Q0_qualification_passed_no_performance_or_production_acceptance" if valid else "Q0_numerical_or_native_validity_gate_failed"
        else:
            outcome = "Q0_native_execution_or_applicability_failed_incomplete_qualification"
        self.report.update(review_completed=True, outcome=outcome, next_stage_automatic_authorization=False,
            old_R2_joint_gate_remains_failed=True, schema3_468_primary_rule_remains_failed=True, literal720_remains_descriptive=True,
            finished_epoch=window["finished_epoch"])
        return self.report


def evidence_failure(report, error):
    report = report if report is not None else dict(schema_version=1, reader_sha256=file_sha(__file__))
    report.update(review_completed=False, outcome="Q0_evidence_invalid", evidence_error=str(error), qualification_passed=False,
        formal_performance_cohort_eligible=False, qualified_to_prepare_separate_long_protocol=False,
        long_workload_qualification_completed=False, stage_next_requires_separate_freeze=True,
        production_acceptance=False, production_adoption=False, performance_acceptance=False,
        fullmodel_speedup_proven=False, deployed_executable_wire_bridge_completed=False)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "window-root", "private-build-manifest", "target-build-manifest", "out"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--path-map", action="append", default=[])
    args = p.parse_args()
    require(not args.out.exists(), "new independent qualification report")
    run, code = None, 0
    try:
        run = Review(args); report = run.run()
    except (EvidenceError, KeyError, OSError, ValueError, TypeError, IndexError, StopIteration) as error:
        report = evidence_failure(run.report if run else None, error)
        code = 2
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report.get(key) for key in ("review_completed", "outcome", "qualification_passed", "gold_library_bridge_passed",
        "candidate_reference_passed", "formal_performance_cohort_eligible", "production_acceptance", "evidence_error")}, indent=2))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
