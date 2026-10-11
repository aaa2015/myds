#!/usr/bin/env python3
"""Strict offline byte replay of continuous three-row greedy target evidence.

REPORT RAW_DIR --output REVIEW.json. This runs on the CPU, does not load a
model, and never establishes performance or drafter/Engine integration.
Normal mode checks all first/second checkpoints, full/partial1/partial2 and
resumed states. Fault mode checks the legal restored persistent prefix and
then all 17 roles after explicit scalar retries; unpublished activation and
capture scratch are deliberately excluded only at the failed base prefix.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys


def require(value, message):
    if not value:
        raise ValueError(message)


def fnv(data):
    value = 14695981039346656037
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & 0xffffffffffffffff
    return f"{value:016x}"


def json_value(value):
    def pairs(items):
        result = {}
        for key, item in items:
            require(key not in result, "Duplicate JSON key: " + key)
            result[key] = item
        return result
    def invalid(value):
        raise ValueError("Invalid JSON number: " + value)
    def floating(value):
        result = float(value)
        require(math.isfinite(result), "Nonfinite JSON number")
        return result
    return json.loads(value, object_pairs_hook=pairs, parse_constant=invalid, parse_float=floating)


def exact(row, expected, message):
    require(all(row.get(key) == value and type(row.get(key)) is type(value)
                for key, value in expected.items()), message)


def integer(value, low, high, message):
    require(type(value) is int and low <= value <= high, message)
    return value


def tokens(value, count, vocab, message):
    require(isinstance(value, list) and len(value) == count and
            all(type(token) is int and 0 <= token < vocab for token in value), message)


ROLES = {"logits", "residual", "pre", "previous_kv", "previous_score", "slots",
         "window", "pcache", "snapshot"} | {
    name + str(owner) for name in ("compressed", "index") for owner in range(4)}
INVALID = {"logits", "residual", "pre", "slots"}
CP = {"logits", "residual", "pre"}
STATE_KEYS = {"schema_version", "kind", "repeat", "label", "pos", "pcache_len", "next_position",
              "rng_state", "engram_position", "engram_history", "files"}
COUNT_KEYS = {"depth2_calls", "depth2_hits1", "depth2_hits2", "depth2_misses", "depth2_failures",
              "depth2_fallbacks", "batch2_calls", "batch2_hits", "batch2_misses", "batch2_failures",
              "batch2_fallbacks", "target_forward_calls", "target_rows"}
EXEC_KEYS = {"schema_version", "kind", "repeat", "arm", "keep_count", "position", "emitted0", "seed",
             "truth", "inputs", "n_inputs", "expected_ret", "ret", "actual", "commit_call_count",
             "first_state_valid", "second_state_valid", "caller_auto_barrier_restored", "pass"} | COUNT_KEYS
CMP_KEYS = {"schema_version", "kind", "repeat", "label", "role", "dtype", "bytes", "elements",
            "mismatches", "nonfinite", "first_index", "first_reference_bits", "first_actual_bits",
            "max_abs", "max_ulp", "pass"}
SCALAR = {1: "scalar_first", 2: "scalar_second", 3: "scalar_third"}
CASES = ((0, 1), (1, 1), (1, 2), (2, 1), (2, 2))


def geometry(meta, pos):
    result = {name: ("f32", count * 4) for name, count in {
        "logits": meta["vocab"], "residual": 20480, "pre": 4, "previous_kv": 2048,
        "previous_score": 2048, "slots": 15360, "window": 40 * min(pos, 128) * 512}.items()}
    for owner in range(4):
        rows = pos // (1 if owner == 3 else 2)
        result[f"compressed{owner}"] = ("bf16", rows * 512 * 2)
        result[f"index{owner}"] = ("bf16", rows * 128 * 2)
    result.update(pcache=("i32", pos * 4), snapshot=("bin", meta["snapshot_bytes"]))
    require(set(result) == ROLES, "State geometry differs")
    return result


def protocol(faults):
    offsets = {name: offset for offset, name in SCALAR.items()}
    checkpoints, native = {}, {}
    if faults:
        offsets["scalar_base"] = 0
        for stage in (2, 3, 4):
            offsets[f"fault_stage{stage}_rolled_base"] = 0
            for count in (1, 2, 3):
                offsets[f"fault_stage{stage}_retry_{count}"] = count
            native["fault_execution", stage, 0] = -1
        native["eos_execution", 3, 2] = 2
        offsets.update(fault_eos_full=2, fault_eos_partial=2, fault_eos_resume_3=3)
        for count, suffix in ((1, "first"), (2, "second")):
            checkpoints[f"fault_eos_checkpoint_{suffix}"] = (3, 2, count)
    else:
        for arm, keep in CASES:
            prefix = f"depth2_arm{arm}_keep{keep}_"
            native["execution", arm, keep] = arm + 1
            offsets[prefix + "full"] = arm + 1
            offsets[prefix + "partial"] = keep
            for count in range(keep + 1, 4):
                offsets[prefix + f"resume_{count}"] = count
            checkpoints[prefix + "checkpoint_first"] = (arm, keep, 1)
            if arm:
                checkpoints[prefix + "checkpoint_second"] = (arm, keep, 2)
    return offsets, checkpoints, native


def metadata(meta):
    faults = meta.get("fault_mode")
    require(type(faults) is bool, "Explicit fault mode required")
    expected = dict(schema_version=1, kind="metadata", prompt_tokens=10, public_prefill_tokens=9,
                    repeats=3, trace_active=False, forced_pld_candidate=True, real_target=True,
                    support_model_loaded=False, capture_layers=[36, 37, 38], performance_measurement=False,
                    dim=5120, vocab=129280, depth2_mode=True, cases_per_repeat=4 if faults else 5,
                    unused_third_perturbed_each_repeat=not faults, fault_mode=faults,
                    capture_mode=1 if faults else 0, head_commit_indices=[4, 5], final_publication_commit_index=6, temperature=0,
                    rollback_invalid_roles=["logits", "residual", "pre", "slots"] if faults else [])
    keys = set(expected) | {"model", "token_source", "position", "prefix_ids", "prefix_ids_fnv1a64",
                            "snapshot_bytes", "environment"}
    require(set(meta) == keys, "Depth2 metadata schema differs")
    exact(meta, expected, "Fixed continuous depth2 protocol differs")
    integer(meta["position"], 11, 2047, "Invalid fixed base position")
    integer(meta["snapshot_bytes"], 1327104, 1 << 24, "Invalid complete native snapshot size")
    require(all(isinstance(meta[key], str) and meta[key] and "\0" not in meta[key]
                for key in ("model", "token_source")), "Missing model/input identity")
    tokens(meta["prefix_ids"], 10, meta["vocab"], "Exact prefix IDs missing")
    require(fnv(struct.pack("<10i", *meta["prefix_ids"])) == meta["prefix_ids_fnv1a64"], "Input ID hash differs")
    env = meta["environment"]
    require(isinstance(env, dict) and all(isinstance(k, str) and k.startswith("MYDS_") and
            isinstance(v, str) for k, v in env.items()), "Invalid captured environment")
    exact(env, {"MYDS_TREE_SPEC": "0", "MYDS_BATCH2_SPEC": "1", "MYDS_BATCH2_LAZY_HEAD": "1",
                "MYDS_BATCH2_TRACE_GPU": "0", "MYDS_DSPARK_CAPTURE_BYPASS": "1" if faults else "0",
                "MYDS_ENGINE_CAPTURE_BYPASS": "0", "MYDS_PLD": "0"}, "Native/capture/PLD environment differs")
    require("MYDS_TRACE_DIR" not in env, "Trace cannot establish native parity")
    require(env.get("MYDS_DEPTH2_SHARED3", "0") in ("0", "1"), "Invalid shared3 selector")
    if faults:
        exact(env, {"MYDS_FIXED_STATE_FAULTS": "1"}, "Explicit fault selector required")
    if "MYDS_FIXED_STATE_POSITION" in env:
        require(env["MYDS_FIXED_STATE_POSITION"] == str(meta["position"]), "Position selector differs")
    return faults


class Payload:
    def __init__(self, raw, meta, deps):
        self.raw, self.meta, self.deps = raw, meta, deps
        self.files, self.checked = {}, {}
    def read(self, row, sizes):
        descriptors = row.get("files")
        require(isinstance(descriptors, list) and len(descriptors) == len(sizes), "Exact raw descriptor set required")
        result = {}
        for item in descriptors:
            require(isinstance(item, dict) and set(item) == {"path", "role", "dtype", "bytes", "fnv1a64"},
                    "Raw descriptor schema differs")
            role = item["role"]
            require(isinstance(role, str) and role in sizes and role not in result, "Unknown/duplicate raw role")
            dtype, size = sizes[role]
            name = f"r{row['repeat']}_{row['label']}_{role}.{dtype}"
            exact(item, dict(path=name, dtype=dtype, bytes=size), "Original raw identity/full size differs")
            require(isinstance(item["fnv1a64"], str) and re.fullmatch(r"[0-9a-f]{16}", item["fnv1a64"]), "Invalid FNV")
            path = self.raw / name
            require(name not in self.files and path.is_file() and not path.is_symlink(), "Missing/linked/repeated raw")
            data = path.read_bytes()
            require(len(data) == size, "Truncated/extended original raw")
            digest = hashlib.sha256(data).hexdigest()
            identity = dtype, digest
            if identity in self.checked:
                old, value = self.checked[identity]
                require(old == data, "Distinct bytes with same SHA256")
            else:
                value = fnv(data)
                if dtype == "f32":
                    require(all(math.isfinite(v[0]) for v in struct.iter_unpack("<f", data)), "Nonfinite F32")
                elif dtype == "bf16":
                    require(all((v[0] & 0x7f80) != 0x7f80 for v in struct.iter_unpack("<H", data)), "Nonfinite BF16")
                elif dtype == "i32":
                    require(all(0 <= v[0] < self.meta["vocab"] for v in struct.iter_unpack("<i", data)), "Invalid committed ID")
                self.checked[identity] = data, value
            require(value == item["fnv1a64"], "Original FNV differs")
            self.files[name] = dict(bytes=size, dtype=dtype, role=role, sha256=digest, fnv1a64=value)
            self.deps[str(path.resolve())] = digest
            result[role] = data
        require(set(result) == set(sizes), "Missing original raw roles")
        return result


def counters(ret, failure=False):
    return dict(depth2_calls=1, depth2_hits1=int(not failure and ret == 2),
                depth2_hits2=int(not failure and ret == 3), depth2_misses=int(not failure and ret == 1),
                depth2_failures=int(failure), depth2_fallbacks=0, batch2_calls=1,
                batch2_hits=int(not failure and ret >= 2), batch2_misses=int(not failure and ret == 1),
                batch2_failures=int(failure), batch2_fallbacks=0, target_forward_calls=1, target_rows=3)


def inspect(report, raw):
    report, raw = Path(report), Path(raw)
    require(report.is_file() and not report.is_symlink() and raw.is_dir() and not raw.is_symlink(), "Missing/linked evidence")
    original = report.read_bytes()
    require(original.endswith(b"\n"), "Truncated report")
    rows = [json_value(line) for line in original.splitlines()]
    require(rows and all(isinstance(row, dict) and type(row.get("schema_version")) is int and
                         row["schema_version"] == 1 for row in rows), "Invalid report schema")
    require(rows[0].get("kind") == "metadata", "Missing initial metadata")
    meta = rows[0]
    faults = metadata(meta)
    offsets, cp_protocol, native = protocol(faults)
    record_counts = Counter(metadata=1, state=57 if faults else 63, checkpoint_file=18 if faults else 81,
                    comparison=875 if faults else 1101)
    record_counts.update(fault_execution=9, eos_execution=3, depth2_fault_complete=1) if faults else record_counts.update(execution=15, depth2_complete=1)
    require(Counter(row.get("kind") for row in rows) == record_counts, "Missing/extra complete native protocol records")
    terminal = "depth2_fault_complete" if faults else "depth2_complete"
    require(rows[-1]["kind"] == terminal, "Missing final completion")
    deps = {str(report.resolve()): hashlib.sha256(original).hexdigest(),
            str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    payload = Payload(raw, meta, deps)
    states, checkpoint, executions, comparisons = {}, {}, {}, {}
    for row in rows[1:-1]:
        kind = row["kind"]
        repeat = integer(row.get("repeat"), 0, 2, "Invalid repeat")
        if kind == "state":
            require(set(row) == STATE_KEYS and row.get("label") in offsets, "State schema/identity differs")
            key = repeat, row["label"]
            require(key not in states, "Duplicate state")
            pos = meta["position"] + offsets[row["label"]]
            exact(row, {k: pos for k in ("pos", "pcache_len", "next_position", "engram_position")}, "Position/visible prefix differs")
            require(isinstance(row["rng_state"], str) and re.fullmatch(r"0|[1-9][0-9]*", row["rng_state"]) and
                    int(row["rng_state"]) < 1 << 64, "Invalid RNG state")
            tokens(row["engram_history"], 3, meta["vocab"], "Invalid Engram history")
            states[key] = row, payload.read(row, geometry(meta, pos))
        elif kind == "checkpoint_file":
            require(set(row) == {"schema_version", "kind", "repeat", "arm", "keep_count", "checkpoint_count", "label", "files"} and
                    row.get("label") in cp_protocol, "Checkpoint schema/identity differs")
            arm, keep, cp_count = cp_protocol[row["label"]]
            exact(row, dict(arm=arm, keep_count=keep, checkpoint_count=cp_count), "Checkpoint binding differs")
            desc = row["files"]
            require(isinstance(desc, list) and len(desc) == 1 and isinstance(desc[0], dict) and desc[0].get("role") in CP,
                    "One exact checkpoint role required")
            role = desc[0]["role"]
            key = repeat, row["label"], role
            require(key not in checkpoint, "Duplicate checkpoint")
            size = geometry(meta, meta["position"] + cp_count)[role]
            checkpoint[key] = payload.read(row, {role: size})[role]
        elif kind in ("execution", "fault_execution", "eos_execution"):
            extra = {"commit_at", "retry_ready", "capture_invalid", "capture_ready"} if kind == "fault_execution" else {"checkpoint_buffers_disjoint"}
            if kind == "eos_execution":
                extra |= {"temporary_eos", "original_eos", "temperature"}
            shared3_mode = meta["environment"].get("MYDS_DEPTH2_SHARED3")
            if shared3_mode is not None:
                extra = extra | {"shared3"}
                shared = row.get("shared3")
                require(isinstance(shared, dict) and set(shared) == {"mode", "dense", "gu", "routed"}, "Missing shared3 execution")
                exact(shared, {"mode": int(shared3_mode)}, "Observed shared3 mode differs")
                for provider, count in (("dense", 200), ("gu", 40), ("routed", 40)):
                    values = shared[provider]
                    fields = {"requests", "encoded", "failures"} | ({"q8_encoded", "q4_encoded"} if provider == "dense" else set())
                    require(isinstance(values, dict) and set(values) == fields and
                            all(type(v) is int and v >= 0 for v in values.values()), "Invalid shared3 provider counts")
                    exact(values, {"requests": count if shared3_mode == "1" else 0,
                                   "encoded": count if shared3_mode == "1" else 0, "failures": 0}, "Shared3 native execution absent/failed")
                    if provider == "dense":
                        require(values["q8_encoded"] + values["q4_encoded"] == values["encoded"] and
                                (values["q4_encoded"] > 0 if shared3_mode == "1" else values["q4_encoded"] == 0), "Actual shared Q4 absent")
            require(set(row) == EXEC_KEYS | extra, "Native execution schema differs")
            identity = kind, row["arm"], row["keep_count"]
            require(type(row["arm"]) is int and type(row["keep_count"]) is int and identity in native, "Unknown native branch")
            key = repeat, *identity
            require(key not in executions, "Duplicate native call")
            ret = native[identity]
            expected = dict(position=meta["position"], n_inputs=3, expected_ret=ret, ret=ret,
                            caller_auto_barrier_restored=True, first_state_valid=ret > 0,
                            second_state_valid=ret >= 2, pass_=True)
            expected["pass"] = expected.pop("pass_")
            expected.update(counters(ret, ret == -1))
            if ret > 0:
                expected.update(commit_call_count=ret + 3, checkpoint_buffers_disjoint=True)
            else:
                expected.update(commit_at=row["arm"] + 2, retry_ready=True, capture_invalid=True, capture_ready=False)
                integer(row["commit_call_count"], row["arm"] + 2, 32, "Fault did not reach actual head commit")
            exact(row, expected, "Native outcome/counters/commit/no-fallback differs")
            for field in ("seed", "emitted0"):
                integer(row[field], 0, meta["vocab"] - 1, "Invalid seed/input")
            tokens(row["truth"], 3, meta["vocab"], "Complete dynamic scalar truth required")
            tokens(row["inputs"], 3, meta["vocab"], "Complete native three-row inputs required")
            actual = row["truth"][:ret] + [-1] * (3 - ret) if ret > 0 else [-1] * 3
            require(row["actual"] == actual and isinstance(row["actual"], list) and all(type(v) is int for v in row["actual"]),
                    "Authoritative output/sentinel binding differs")
            require(row["inputs"][0] == row["seed"], "Root consumed seed differs")
            a, b = row["inputs"][1:]
            t1, t2 = row["truth"][:2]
            if kind == "execution":
                require((a != t1 and b != t2) if row["arm"] == 0 else (a == t1 and b != t2) if row["arm"] == 1 else
                        (a == t1 and b == t2), "Continuous miss1/miss2/hit3 candidate binding differs")
            else:
                require(a == t1 and b == t2, "Fault/EOS must reach authoritative second/third head")
            if kind == "eos_execution":
                exact(row, dict(temporary_eos=t2, temperature=0), "EOS did not stop on greedy second authoritative head")
                integer(row["original_eos"], 0, meta["vocab"] - 1, "Invalid original EOS")
                require(t1 != t2 and row["original_eos"] not in (t1, t2), "EOS head staging ambiguous")
            executions[key] = row
        elif kind == "comparison":
            require(set(row) == CMP_KEYS and isinstance(row["label"], str) and row["role"] in ROLES, "Comparison schema differs")
            key = repeat, row["label"], row["role"]
            require(key not in comparisons, "Duplicate native comparison")
            exact(row, dict(mismatches=0, nonfinite=0, first_index=0, first_reference_bits="00000000",
                            first_actual_bits="00000000", max_ulp=0) | {"pass": True}, "Native byte/finite comparison failed")
            require(type(row["max_abs"]) in (int, float) and row["max_abs"] == 0, "Native error differs")
            comparisons[key] = row
        else:
            raise ValueError("Extra/unknown report record")
    require(set(states) == {(r, label) for r in range(3) for label in offsets}, "Missing/extra state set")
    require(set(checkpoint) == {(r, label, role) for r in range(3) for label in cp_protocol for role in CP}, "Missing complete first/second checkpoint")
    require(set(executions) == {(r, *key) for r in range(3) for key in native}, "Missing native call set")
    baseline = next(iter(executions.values()))
    invariant = ("position", "seed", "emitted0", "truth")
    require(all(all(run[k] == baseline[k] for k in invariant) for run in executions.values()), "Same-base truth/input drift")
    if not faults:
        first_misses, second_misses = [], []
        for repeat in range(3):
            zero = executions[repeat, "execution", 0, 1]
            first_misses.append(zero["inputs"][1])
            second_misses.append(zero["inputs"][2])
            for arm, keep in CASES:
                run = executions[repeat, "execution", arm, keep]
                require(arm == 2 or run["inputs"][2] == zero["inputs"][2], "Unused third-row perturbation binding differs")
        require(len(set(first_misses)) == len(set(second_misses)) == 3, "Tentative rows were not perturbed between repeats")
    expected_comparisons = {}
    def equal(repeat, label, role, reference, actual, dtype):
        require(reference == actual, "Original bytes differ: " + label + "/" + role)
        key = repeat, label, role
        require(key not in expected_comparisons, "Reviewer generated duplicate comparison")
        expected_comparisons[key] = dtype, len(reference)
    for repeat in range(3):
        for label, offset in offsets.items():
            row, data = states[repeat, label]
            ref_label = "scalar_base" if offset == 0 else SCALAR[offset]
            original_row, ref = states[0, ref_label]
            require({k: v for k, v in row.items() if k not in ("repeat", "label", "files")} ==
                    {k: v for k, v in original_row.items() if k not in ("repeat", "label", "files")}, "State RNG/Engram metadata drift")
            roles = ROLES - INVALID if offset == 0 else ROLES
            require(all(data[role] == ref[role] for role in roles), "Scalar/native checkpoint or prefix drift: " + label)
            if label not in SCALAR.values() and label != "scalar_base":
                sizes = geometry(meta, meta["position"] + offset)
                for role in roles:
                    equal(repeat, label, role, states[repeat, ref_label][1][role], data[role], sizes[role][0])
        first = struct.unpack(f"<{meta['position'] + 1}i", states[repeat, "scalar_first"][1]["pcache"])
        second = struct.unpack(f"<{meta['position'] + 2}i", states[repeat, "scalar_second"][1]["pcache"])
        third = struct.unpack(f"<{meta['position'] + 3}i", states[repeat, "scalar_third"][1]["pcache"])
        require(first[:10] == tuple(meta["prefix_ids"]) and first[-2:] == (baseline["emitted0"], baseline["seed"]) and
                second[:-1] == first and second[-1] == baseline["truth"][0] and third[:-1] == second and
                third[-1] == baseline["truth"][1], "Consumed/bonus input sequence differs")
        if faults:
            base_ids = struct.unpack(f"<{meta['position']}i", states[repeat, "scalar_base"][1]["pcache"])
            require(first[:-1] == base_ids, "Restored original prefix differs")
        for offset, label in SCALAR.items():
            values = [v[0] for v in struct.iter_unpack("<f", states[repeat, label][1]["logits"])]
            require(values[baseline["truth"][offset-1]] == max(values), "Scalar truth is not greedy maximal logit")
        for label, (_, _, count) in cp_protocol.items():
            for role in CP:
                equal(repeat, label, role, states[repeat, SCALAR[count]][1][role], checkpoint[repeat, label, role], "f32")
        if repeat:
            for offset, label in ((0, "scalar_base"), *SCALAR.items()) if faults else SCALAR.items():
                suffix = "base" if not offset else ("first", "second", "third")[offset-1]
                sizes = geometry(meta, meta["position"] + offset)
                for role in (ROLES - INVALID if offset == 0 else ROLES):
                    equal(repeat, "scalar_repeat_" + suffix, role, states[0, label][1][role], states[repeat, label][1][role], sizes[role][0])
    require(set(expected_comparisons) == set(comparisons) and len(comparisons) == record_counts["comparison"], "Missing/extra native comparison coverage")
    for key, (dtype, size) in expected_comparisons.items():
        width = 4 if dtype == "f32" else 2 if dtype == "bf16" else 1
        exact(comparisons[key], dict(dtype=dtype, bytes=size, elements=size // width), "Comparison full geometry differs")
    raw_count = 987 if faults else 1152
    complete = dict(schema_version=1, kind=terminal, repeats_completed=3, cases_completed=12 if faults else 15,
                    raw_files=raw_count, same_base_snapshot=True, depth2_mode=True, trace_active=False,
                    performance_measurement=False, bitwise_logits_and_visible_state=True) | {"pass": True}
    require(set(rows[-1]) == set(complete), "Completion schema differs")
    exact(rows[-1], complete, "Incomplete native gate")
    require(len(payload.files) == raw_count and {path.name for path in raw.iterdir()} == set(payload.files), "Missing/extra unreviewed raw evidence")
    return dict(schema_version=1, kind="depth2_fixed_state_review", fault_mode=faults,
                position=meta["position"], repeats=3, native_executions=12 if faults else 15,
                commit_fault_executions=9 if faults else 0, eos_executions=3 if faults else 0,
                states=record_counts["state"], checkpoint_files=record_counts["checkpoint_file"], comparisons=record_counts["comparison"],
                raw_files=raw_count, commit_indices=[4, 5, 6], dynamic_truth=baseline["truth"], seed=baseline["seed"],
                prefix_ids=meta["prefix_ids"], partial1_partial2_resumption_bitwise=True,
                first_second_checkpoints_bitwise=True, actual_native_target_rows=3,
                no_native_fallback=True, rollback_invalid_roles=sorted(INVALID) if faults else [],
                performance_measurement=False, support_model_loaded=False, inputs_sha256=deps,
                raw_manifest=payload.files) | {"pass": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("raw", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = inspect(args.report, args.raw)
    text = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.output:
        destination = args.output.resolve()
        require(str(destination) not in result["inputs_sha256"] and args.raw.resolve() not in
                (destination, *destination.parents), "Review output cannot overwrite/extend evidence")
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, TypeError, struct.error) as error:
        raise SystemExit("FAIL: " + str(error))
