#!/usr/bin/env python3
"""Replay original scalar/Batch2 or three-arm tree fixed-state evidence.

This is a CPU byte comparison, never a GPU run or performance measurement.
The immutable Batch2 reviewer remains authoritative for its original mode.
Tree mode requires the entire native miss/A/B protocol and all raw payloads;
neither matching argmax nor native pass booleans substitute for raw bytes.
"""

import argparse
from collections import Counter
import hashlib
import importlib.util
import math
from pathlib import Path
import re
import struct
import sys


FROZEN_DIR = (Path(__file__).resolve().parents[1] / "docs" / "benchmarks" /
              "2026-10-11-dspark-handover" / "scripts")
FROZEN_REVIEWER = FROZEN_DIR / "review_batch2_fixed_state.py"
FROZEN_SUPPORT = FROZEN_DIR / "review_state.py"
_write_bytecode = sys.dont_write_bytecode
sys.dont_write_bytecode = True
sys.path.insert(0, str(FROZEN_DIR))
try:
    from review_state import fnv, json_value, require, sha
    _spec = importlib.util.spec_from_file_location("_frozen_batch2_review", FROZEN_REVIEWER)
    require(_spec is not None and _spec.loader is not None, "Missing frozen Batch2 reviewer")
    _batch2 = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_batch2)
finally:
    sys.path.pop(0)
    sys.dont_write_bytecode = _write_bytecode


ROLES = {"logits", "residual", "pre", "previous_kv", "previous_score",
         "slots", "window", "pcache", "snapshot"} | {
    name + str(owner) for name in ("compressed", "index") for owner in range(4)}
FIRST = {"scalar_first", "tree_arm0_miss_first", "tree_arm1_partial_first",
         "tree_arm2_partial_first"}
SECOND = {"scalar_second", "tree_arm0_continued_after_miss"} | {
    f"tree_arm{arm}_{suffix}" for arm in (1, 2)
    for suffix in ("second", "continued_second")}
LABELS = FIRST | SECOND
CHECKPOINT_ROLES = ("logits", "residual", "pre")
STATE_KEYS = {"schema_version", "kind", "repeat", "label", "pos", "pcache_len",
              "next_position", "rng_state", "engram_position", "engram_history", "files"}
EXECUTION_KEYS = {"schema_version", "kind", "repeat", "position", "emitted0", "seed",
                  "truth", "next", "tree_mode", "arm", "inputs", "n_inputs",
                  "expected_ret", "ret", "actual", "batch2_calls", "batch2_hits",
                  "batch2_misses", "batch2_failures", "batch2_fallbacks",
                  "target_forward_calls", "target_rows", "caller_auto_barrier_restored", "pass"}
COMPARISON_KEYS = {"schema_version", "kind", "repeat", "label", "role", "dtype", "bytes",
                   "elements", "mismatches", "nonfinite", "first_index",
                   "first_reference_bits", "first_actual_bits", "max_abs", "max_ulp", "pass"}


def exact_fields(row, expected, message):
    require(all(row.get(key) == value and type(row.get(key)) is type(value)
                for key, value in expected.items()), message)


def integer(value, low, high, message):
    require(type(value) is int and low <= value <= high, message)
    return value


def rng_value(value):
    require(isinstance(value, str) and re.fullmatch(r"0|[1-9][0-9]*", value) and
            int(value) <= (1 << 64) - 1, "Invalid RNG state")
    return int(value)


def input_ids(value, count, vocab, message):
    require(isinstance(value, list) and len(value) == count and
            all(type(token) is int and 0 <= token < vocab for token in value), message)


def state_geometry(meta, pos):
    sizes = {role: ("f32", count * 4) for role, count in {
        "logits": meta["vocab"], "residual": 20480, "pre": 4,
        "previous_kv": 2048, "previous_score": 2048, "slots": 15360,
        "window": 40 * min(pos, 128) * 512}.items()}
    for owner in range(4):
        count = pos // (1 if owner == 3 else 2)
        sizes[f"compressed{owner}"] = ("bf16", count * 512 * 2)
        sizes[f"index{owner}"] = ("bf16", count * 128 * 2)
    sizes.update(pcache=("i32", pos * 4), snapshot=("bin", meta["snapshot_bytes"]))
    require(set(sizes) == ROLES, "Reviewer state geometry differs")
    return sizes


def metadata(meta, tree=True, faults=False):
    expected = dict(prompt_tokens=10, public_prefill_tokens=9, repeats=3,
                    trace_active=False, forced_pld_candidate=True, real_target=True,
                    support_model_loaded=False, capture_layers=[36, 37, 38],
                    performance_measurement=False, dim=5120, vocab=129280,
                    tree_mode=tree, cases_per_repeat=(3 if tree else 2) if faults else (3 if tree else 1),
                    unused_branch_perturbed_each_repeat=tree)
    if faults or "fault_mode" in meta:
        expected["fault_mode"] = faults
    keys = set(expected) | {"schema_version", "kind", "model", "token_source", "prefix_ids",
                           "prefix_ids_fnv1a64", "snapshot_bytes", "environment", "position"}
    require(set(meta) == keys, "Tree metadata schema differs")
    exact_fields(meta, expected, "Fixed tree protocol differs")
    integer(meta["position"], 11, 2047, "Invalid fixed base position")
    integer(meta["snapshot_bytes"], 1, (1 << 63) - 1, "Invalid snapshot size")
    require(all(isinstance(meta[key], str) and meta[key] and "\0" not in meta[key]
                for key in ("model", "token_source")), "Missing model/input identity")
    input_ids(meta["prefix_ids"], 10, meta["vocab"], "Exact input IDs missing")
    require(fnv(struct.pack("<10i", *meta["prefix_ids"])) == meta["prefix_ids_fnv1a64"],
            "Prefix IDs hash differs")
    env = meta["environment"]
    require(isinstance(env, dict) and all(isinstance(key, str) and key.startswith("MYDS_") and
            isinstance(value, str) for key, value in env.items()), "Invalid captured environment")
    exact_fields(env, {"MYDS_TREE_SPEC": "1" if tree else "0", "MYDS_BATCH2_SPEC": "1",
                      "MYDS_DSPARK_CAPTURE_BYPASS": "0", "MYDS_ENGINE_CAPTURE_BYPASS": "0"},
                 "Native tree/capture environment differs")
    if tree:
        exact_fields(env, {"MYDS_FIXED_STATE_TREE": "1"}, "Native three-row mode missing")
    if faults:
        exact_fields(env, {"MYDS_FIXED_STATE_FAULTS": "1", "MYDS_BATCH2_LAZY_HEAD": "1",
                          "MYDS_BATCH2_TRACE_GPU": "0"}, "Fault protocol environment differs")
    require("MYDS_TRACE_DIR" not in env, "Trace cannot establish native tree parity")


def payload_reader(raw, meta, deps):
    vocab = meta["vocab"]
    files, checked = {}, {}
    def read(row, label, sizes):
        descriptors = row.get("files")
        require(isinstance(descriptors, list) and len(descriptors) == len(sizes),
                "Complete exact raw descriptor set required")
        result = {}
        for item in descriptors:
            require(isinstance(item, dict) and set(item) == {"path", "role", "dtype", "bytes", "fnv1a64"},
                    "Raw descriptor schema differs")
            role = item["role"]
            require(isinstance(role, str) and role in sizes and role not in result,
                    "Unknown/duplicate raw role")
            dtype, size = sizes[role]
            name = f"r{row['repeat']}_{label}_{role}.{dtype}"
            exact_fields(item, {"path": name, "dtype": dtype, "bytes": size},
                         "Original raw name/type/full size differs")
            require(isinstance(item["fnv1a64"], str) and
                    re.fullmatch(r"[0-9a-f]{16}", item["fnv1a64"]), "Invalid raw hash")
            path = raw / name
            require(name not in files and path.is_file() and not path.is_symlink(),
                    "Missing/linked/repeated raw payload")
            data = path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            require(len(data) == size, "Original raw full size differs")
            identity = dtype, digest
            if identity in checked:
                previous, payload_fnv = checked[identity]
                require(data == previous, "Distinct original bytes have the same SHA256")
            else:
                payload_fnv = fnv(data)
                if dtype == "f32":
                    require(all(math.isfinite(value[0]) for value in struct.iter_unpack("<f", data)),
                            "Nonfinite F32 output")
                elif dtype == "bf16":
                    require(all((value[0] & 0x7f80) != 0x7f80 for value in struct.iter_unpack("<H", data)),
                            "Nonfinite BF16 visible cache")
                elif dtype == "i32":
                    require(all(0 <= value[0] < vocab for value in struct.iter_unpack("<i", data)),
                            "Invalid committed input ID")
                checked[identity] = data, payload_fnv
            require(payload_fnv == item["fnv1a64"], "Original raw FNV hash differs")
            files[name] = digest
            deps[str(path.resolve())] = files[name]
            result[role] = data
        require(set(result) == set(sizes), "Missing raw roles")
        return result

    return files, read


def inspect_tree(report, raw, rows, report_bytes):
    counts = Counter(metadata=1, state=30, execution=9, checkpoint_file=27,
                     comparison=503, tree_complete=1)
    require(Counter(row.get("kind") for row in rows) == counts,
            "All three repeated miss/A/B executions, checkpoints and states required")
    require(rows[0]["kind"] == "metadata" and rows[-1]["kind"] == "tree_complete",
            "Missing metadata/final tree completion")
    meta = rows[0]
    metadata(meta)
    vocab = meta["vocab"]
    deps = {str(report.resolve()): hashlib.sha256(report_bytes).hexdigest(),
            str(Path(__file__).resolve()): sha(__file__),
            str(FROZEN_REVIEWER): sha(FROZEN_REVIEWER), str(FROZEN_SUPPORT): sha(FROZEN_SUPPORT)}
    states, checkpoints, executions, comparisons = {}, {}, {}, {}

    files, read = payload_reader(raw, meta, deps)

    for row in rows:
        kind = row["kind"]
        if kind in ("metadata", "tree_complete"):
            continue
        repeat = integer(row.get("repeat"), 0, 2, "Invalid repetition")
        if kind == "state":
            require(set(row) == STATE_KEYS and row.get("label") in LABELS, "Tree state schema/label differs")
            label = row["label"]
            require((repeat, label) not in states, "Repeated state")
            pos = meta["position"] + (1 if label in FIRST else 2)
            exact_fields(row, {key: pos for key in ("pos", "pcache_len", "next_position", "engram_position")},
                         "State/cache/Engram position differs")
            require(isinstance(row["rng_state"], str) and re.fullmatch(r"0|[1-9][0-9]*", row["rng_state"])
                    and int(row["rng_state"]) <= (1 << 64) - 1, "Invalid RNG state")
            input_ids(row["engram_history"], 3, vocab, "Invalid Engram history")
            states[repeat, label] = (row, read(row, label, state_geometry(meta, pos)))
        elif kind == "checkpoint_file":
            require(set(row) == {"schema_version", "kind", "repeat", "tree_mode", "arm", "label", "files"},
                    "Checkpoint schema differs")
            arm = integer(row.get("arm"), 0, 2, "Invalid checkpoint arm")
            label = f"tree_arm{arm}_checkpoint_first"
            exact_fields(row, {"tree_mode": True, "label": label}, "Checkpoint branch identity differs")
            desc = row.get("files")
            require(isinstance(desc, list) and len(desc) == 1 and isinstance(desc[0], dict) and
                    desc[0].get("role") in CHECKPOINT_ROLES, "Complete first checkpoint required")
            role = desc[0]["role"]
            require((repeat, arm, role) not in checkpoints, "Repeated first checkpoint")
            size = {"logits": vocab * 4, "residual": 20480 * 4, "pre": 4 * 4}[role]
            checkpoints[repeat, arm, role] = read(row, label, {role: ("f32", size)})[role]
        elif kind == "execution":
            require(set(row) == EXECUTION_KEYS, "Native execution schema differs")
            arm = integer(row.get("arm"), 0, 2, "Invalid execution arm")
            require((repeat, arm) not in executions, "Repeated execution")
            ret = 1 if arm == 0 else 2
            exact_fields(row, dict(position=meta["position"], tree_mode=True, n_inputs=3, expected_ret=ret, ret=ret,
                                  batch2_calls=1, batch2_hits=int(ret == 2), batch2_misses=int(ret == 1),
                                  batch2_failures=0, batch2_fallbacks=0, target_forward_calls=1,
                                  target_rows=3, caller_auto_barrier_restored=True)
                         | {"pass": True}, "Native three-row execution/no-fallback differs")
            for key in ("emitted0", "seed", "truth", "next"):
                integer(row[key], 0, vocab - 1, "Invalid scalar truth/seed")
            input_ids(row["inputs"], 3, vocab, "Complete three native inputs required")
            actual = [row["truth"], row["next"] if ret == 2 else -1]
            require(row["actual"] == actual and isinstance(row["actual"], list) and
                    all(type(token) is int for token in row["actual"]), "Native output binding differs")
            require(row["inputs"][0] == row["seed"], "Root consumed input differs")
            a, b = row["inputs"][1:]
            require(a != b and ((a != row["truth"] and b != row["truth"]) if arm == 0 else
                    (a == row["truth"] and b != row["truth"]) if arm == 1 else
                    (b == row["truth"] and a != row["truth"])), "Miss/A/B candidate binding differs")
            executions[repeat, arm] = row
        elif kind == "comparison":
            require(set(row) == COMPARISON_KEYS, "Native comparison schema differs")
            require(isinstance(row["label"], str) and isinstance(row["role"], str), "Invalid comparison identity")
            key = repeat, row["label"], row["role"]
            require(key not in comparisons, "Repeated native comparison")
            exact_fields(row, {"mismatches": 0, "nonfinite": 0, "max_ulp": 0, "first_index": 0,
                               "first_reference_bits": "00000000", "first_actual_bits": "00000000", "pass": True},
                         "Native bitwise/finite comparison failed")
            require(type(row["max_abs"]) in (int, float) and row["max_abs"] == 0, "Native absolute error differs")
            comparisons[key] = row

    require(set(states) == {(repeat, label) for repeat in range(3) for label in LABELS},
            "Missing/extra scalar or branch states")
    require(set(executions) == {(repeat, arm) for repeat in range(3) for arm in range(3)},
            "Missing miss/A/B execution")
    require(set(checkpoints) == {(repeat, arm, role) for repeat in range(3) for arm in range(3)
                                 for role in CHECKPOINT_ROLES}, "Missing complete first checkpoint")
    invariant = ("position", "emitted0", "seed", "truth", "next")
    base_run = executions[0, 0]
    require(all(tuple(run[key] for key in invariant) == tuple(base_run[key] for key in invariant)
                for run in executions.values()), "Same-base scalar truth/seed drift")
    miss_a, miss_b = [], []
    for repeat in range(3):
        zero, a, b = (executions[repeat, arm] for arm in range(3))
        require(zero["inputs"][1] == b["inputs"][1] and zero["inputs"][2] == a["inputs"][2],
                "Unchosen branch perturbation identity differs across arms")
        miss_a.append(zero["inputs"][1]); miss_b.append(zero["inputs"][2])
    require(len(set(miss_a)) == len(set(miss_b)) == 3, "Unchosen branches were not perturbed each repetition")

    expected = {}
    def equal(repeat, label, role, reference, actual, dtype):
        require(reference == actual, "Original scalar/tree bytes differ: " + label + "/" + role)
        key = repeat, label, role
        require(key not in expected, "Reviewer generated duplicate comparison")
        expected[key] = dtype, len(reference)

    for repeat in range(3):
        for label in LABELS:
            row, data = states[repeat, label]
            reference = "scalar_first" if label in FIRST else "scalar_second"
            original, byte_ref = states[0, reference]
            require({key: value for key, value in row.items() if key not in ("repeat", "label", "files")} ==
                    {key: value for key, value in original.items() if key not in ("repeat", "label", "files")} and
                    data == byte_ref, "Complete repeated/branch state bytes differ: " + label)
        first = struct.unpack(f"<{meta['position'] + 1}i", states[repeat, "scalar_first"][1]["pcache"])
        second = struct.unpack(f"<{meta['position'] + 2}i", states[repeat, "scalar_second"][1]["pcache"])
        require(first[:10] == tuple(meta["prefix_ids"]) and first[-2:] == (base_run["emitted0"], base_run["seed"])
                and second[:-1] == first and second[-1] == base_run["truth"], "Committed input sequence differs")
        for arm in range(3):
            for role in CHECKPOINT_ROLES:
                equal(repeat, f"tree_arm{arm}_checkpoint_first", role,
                      states[repeat, "scalar_first"][1][role], checkpoints[repeat, arm, role], "f32")
            suffixes = ("miss_first", "continued_after_miss") if arm == 0 else (
                "second", "partial_first", "continued_second")
            for suffix in suffixes:
                label = f"tree_arm{arm}_{suffix}"
                ref_label = "scalar_first" if label in FIRST else "scalar_second"
                ref = states[repeat, ref_label][1]
                geometry = state_geometry(meta, meta["position"] + (1 if label in FIRST else 2))
                for role, data in ref.items():
                    equal(repeat, label, role, data, states[repeat, label][1][role], geometry[role][0])
        if repeat:
            for label, ref_label in (("scalar_repeat_first", "scalar_first"), ("scalar_repeat_second", "scalar_second")):
                geometry = state_geometry(meta, meta["position"] + (1 if ref_label == "scalar_first" else 2))
                for role, data in states[0, ref_label][1].items():
                    equal(repeat, label, role, data, states[repeat, ref_label][1][role], geometry[role][0])
    require(len(expected) == 503 and set(expected) == set(comparisons), "Missing/extra native comparison roles")
    for key, (dtype, size) in expected.items():
        width = 4 if dtype == "f32" else 2 if dtype == "bf16" else 1
        exact_fields(comparisons[key], {"dtype": dtype, "bytes": size, "elements": size // width},
                     "Native comparison type/full geometry differs")
    complete = rows[-1]
    expected_complete = dict(schema_version=1, kind="tree_complete", repeats_completed=3, cases_completed=9,
                             raw_files=537, same_base_snapshot=True, tree_mode=True, trace_active=False,
                             performance_measurement=False, bitwise_logits_and_visible_state=True, pass_=True)
    expected_complete["pass"] = expected_complete.pop("pass_")
    require(set(complete) == set(expected_complete), "Tree completion schema differs")
    exact_fields(complete, expected_complete, "Incomplete native tree gate")
    require(len(files) == 537 and {path.name for path in raw.iterdir()} == set(files),
            "Extra/missing/unreviewed raw evidence")
    return dict(schema_version=1, kind="fixed_tree_target_state_review", pass_=True,
                repeats=3, arms=3, executions=9, raw_files=537, comparisons=503,
                arm_returns=[1, 2, 2], target_rows_per_execution=3,
                no_native_failures_or_fallbacks=True, unused_branches_perturbed=True,
                bitwise_scalar_tree_first_second_partial_resume=True,
                visible_compressed_and_index_cache=True, dynamic_truth=base_run["truth"],
                seed=base_run["seed"], prefix_ids=meta["prefix_ids"],
                position=meta["position"],
                performance_measurement=False, inputs_sha256=deps)


def inspect_faults(report, raw, rows, report_bytes, tree):
    arms = (1, 2) if tree else (1,)
    comparison_count, raw_count = (452, 537) if tree else (290, 375)
    counts = Counter(metadata=1, state=30 if tree else 21,
                     fault_execution=6 if tree else 3, eos_execution=3,
                     checkpoint_file=27 if tree else 18,
                     comparison=comparison_count, fault_complete=1)
    require(Counter(row.get("kind") for row in rows) == counts,
            "Complete repeated commit-fault and EOS protocol required")
    require(rows[0]["kind"] == "metadata" and rows[-1]["kind"] == "fault_complete",
            "Missing metadata/final fault completion")
    meta = rows[0]
    metadata(meta, tree=tree, faults=True)
    vocab, n_inputs = meta["vocab"], 3 if tree else 2
    first_labels = {"scalar_first", "fault_eos_scalar_first", "fault_eos_first"} | {
        f"fault_arm{arm}_partial_first" for arm in arms}
    second_labels = {"scalar_second"} | {
        f"fault_arm{arm}_{suffix}" for arm in arms for suffix in ("second", "continued_second")}
    labels = first_labels | second_labels
    checkpoint_labels = {f"fault_arm{arm}_checkpoint_first": arm for arm in arms}
    checkpoint_labels["fault_eos_checkpoint_first"] = 1
    deps = {str(report.resolve()): hashlib.sha256(report_bytes).hexdigest(),
            str(Path(__file__).resolve()): sha(__file__),
            str(FROZEN_REVIEWER): sha(FROZEN_REVIEWER), str(FROZEN_SUPPORT): sha(FROZEN_SUPPORT)}
    files, read = payload_reader(raw, meta, deps)
    states, checkpoints, executions, eos_executions, comparisons = {}, {}, {}, {}, {}
    fault_keys = EXECUTION_KEYS | {"commit_at", "commit_call_count"}
    eos_keys = (EXECUTION_KEYS - {"next"}) | {
        "original_eos", "temporary_eos", "temperature", "top_k", "base_rng",
        "expected_rng", "observed_rng", "first_rng"}

    for row in rows:
        kind = row["kind"]
        if kind in ("metadata", "fault_complete"):
            continue
        repeat = integer(row.get("repeat"), 0, 2, "Invalid fault repetition")
        if kind == "state":
            require(set(row) == STATE_KEYS and row.get("label") in labels, "Fault state schema/label differs")
            label = row["label"]
            require((repeat, label) not in states, "Repeated fault state")
            pos = meta["position"] + (1 if label in first_labels else 2)
            exact_fields(row, {key: pos for key in ("pos", "pcache_len", "next_position", "engram_position")},
                         "Fault state/cache/Engram position differs")
            rng_value(row["rng_state"])
            input_ids(row["engram_history"], 3, vocab, "Invalid fault Engram history")
            states[repeat, label] = (row, read(row, label, state_geometry(meta, pos)))
        elif kind == "checkpoint_file":
            require(set(row) == {"schema_version", "kind", "repeat", "tree_mode", "arm", "label", "files"},
                    "Fault checkpoint schema differs")
            label = row.get("label")
            require(isinstance(label, str) and label in checkpoint_labels, "Unknown fault checkpoint")
            exact_fields(row, {"tree_mode": tree, "arm": checkpoint_labels[label]}, "Fault checkpoint branch differs")
            desc = row.get("files")
            require(isinstance(desc, list) and len(desc) == 1 and isinstance(desc[0], dict) and
                    desc[0].get("role") in CHECKPOINT_ROLES, "Complete fault first checkpoint required")
            role = desc[0]["role"]
            require((repeat, label, role) not in checkpoints, "Repeated fault checkpoint")
            size = {"logits": vocab * 4, "residual": 20480 * 4, "pre": 16}[role]
            checkpoints[repeat, label, role] = read(row, label, {role: ("f32", size)})[role]
        elif kind in ("fault_execution", "eos_execution"):
            is_eos = kind == "eos_execution"
            require(set(row) == (eos_keys if is_eos else fault_keys), "Fault/EOS execution schema differs")
            arm = integer(row.get("arm"), 1, 2 if tree else 1, "Invalid fault arm")
            require((repeat not in eos_executions) if is_eos else ((repeat, arm) not in executions),
                    "Repeated fault/EOS execution")
            exact_fields(row, dict(position=meta["position"], tree_mode=tree, n_inputs=n_inputs,
                                  expected_ret=1 if is_eos else 2, ret=1 if is_eos else 2,
                                  batch2_calls=1, batch2_hits=0, batch2_misses=1 if is_eos else 0,
                                  batch2_failures=0 if is_eos else 1, batch2_fallbacks=0 if is_eos else 1,
                                  target_forward_calls=1 if is_eos else 3,
                                  target_rows=n_inputs if is_eos else n_inputs + 2,
                                  caller_auto_barrier_restored=True) | {"pass": True},
                         "Commit rollback/scalar fallback or native EOS execution differs")
            for key in (("emitted0", "seed", "truth") if is_eos else ("emitted0", "seed", "truth", "next")):
                integer(row[key], 0, vocab - 1, "Invalid fault scalar truth/seed")
            actual = [row["truth"], -1 if is_eos else row["next"]]
            require(isinstance(row["actual"], list) and row["actual"] == actual and
                    all(type(token) is int for token in row["actual"]), "Fault/EOS actual token binding differs")
            inputs = row["inputs"]
            require(isinstance(inputs, list) and len(inputs) == 3 and
                    all(type(token) is int for token in inputs), "Exact fault inputs required")
            input_ids(inputs[:n_inputs], n_inputs, vocab, "Invalid native fault inputs")
            require(inputs[0] == row["seed"], "Fault root input differs")
            if is_eos:
                input_ids(inputs, 3, vocab, "Invalid EOS diagnostic candidate")
                exact_fields(row, {"arm": 1, "temporary_eos": row["truth"], "temperature": 0.7, "top_k": 1},
                             "EOS sampling protocol differs")
                integer(row["original_eos"], 0, vocab - 1, "Invalid original EOS identity")
                require(row["original_eos"] != row["truth"] and inputs[1] == row["truth"] and
                        inputs[2] not in (row["seed"], row["truth"], row["original_eos"]), "EOS candidate binding differs")
                base_rng = rng_value(row["base_rng"])
                expected_rng = (base_rng * 6364136223846793005 + 1) & ((1 << 64) - 1)
                require(all(rng_value(row[key]) == expected_rng for key in
                            ("expected_rng", "observed_rng", "first_rng")), "EOS must sample exactly one RNG step")
                eos_executions[repeat] = row
            else:
                exact_fields(row, {"commit_at": 4}, "Deferred selected-head commit fault missing")
                integer(row["commit_call_count"], 4, (1 << 32) - 1, "Fault did not reach fourth commit")
                if tree:
                    require(inputs[arm] == row["truth"] and inputs[3 - arm] not in
                            (row["seed"], row["truth"]), "Chosen fault branch binding differs")
                else:
                    require(inputs[1:] == [row["truth"], -1], "Batch2 fault candidate binding differs")
                executions[repeat, arm] = row
        elif kind == "comparison":
            require(set(row) == COMPARISON_KEYS, "Fault comparison schema differs")
            require(isinstance(row["label"], str) and isinstance(row["role"], str), "Invalid fault comparison identity")
            key = repeat, row["label"], row["role"]
            require(key not in comparisons, "Repeated fault comparison")
            exact_fields(row, {"mismatches": 0, "nonfinite": 0, "max_ulp": 0, "first_index": 0,
                               "first_reference_bits": "00000000", "first_actual_bits": "00000000", "pass": True},
                         "Native fault bitwise/finite comparison failed")
            require(type(row["max_abs"]) in (int, float) and row["max_abs"] == 0, "Native fault absolute error differs")
            comparisons[key] = row

    require(set(states) == {(repeat, label) for repeat in range(3) for label in labels}, "Missing/extra fault states")
    require(set(executions) == {(repeat, arm) for repeat in range(3) for arm in arms} and
            set(eos_executions) == set(range(3)), "Missing commit fault/EOS executions")
    require(set(checkpoints) == {(repeat, label, role) for repeat in range(3)
                                for label in checkpoint_labels for role in CHECKPOINT_ROLES}, "Missing fault checkpoint roles")
    base_run = executions[0, 1]
    invariant = ("position", "emitted0", "seed", "truth", "next")
    require(all(tuple(run[key] for key in invariant) == tuple(base_run[key] for key in invariant)
                for run in executions.values()), "Fault scalar truth/seed drift")
    require(all(all(run[key] == base_run[key] for key in invariant[:-1]) for run in eos_executions.values()),
            "EOS scalar truth/seed differs")
    require(len({run["original_eos"] for run in eos_executions.values()}) == 1 and
            len({run["base_rng"] for run in eos_executions.values()}) == 1, "Same-base EOS identity/RNG drift")
    if tree:
        wrong = []
        for repeat in range(3):
            a, b, eos = executions[repeat, 1], executions[repeat, 2], eos_executions[repeat]
            require(a["inputs"][2] == b["inputs"][1] == eos["inputs"][2], "Unused fault branch identity differs")
            wrong.append(a["inputs"][2])
        require(len(set(wrong)) == 3, "Unused fault branches were not perturbed")

    expected = {}
    def equal(repeat, label, role, reference, actual, dtype):
        require(reference == actual, "Original scalar/fault bytes differ: " + label + "/" + role)
        key = repeat, label, role
        require(key not in expected, "Reviewer generated duplicate fault comparison")
        expected[key] = dtype, len(reference)

    for repeat in range(3):
        for label in labels:
            row, data = states[repeat, label]
            reference = "fault_eos_scalar_first" if label.startswith("fault_eos_") else (
                "scalar_first" if label in first_labels else "scalar_second")
            original, byte_ref = states[0, reference]
            require({key: value for key, value in row.items() if key not in ("repeat", "label", "files")} ==
                    {key: value for key, value in original.items() if key not in ("repeat", "label", "files")} and
                    data == byte_ref, "Complete repeated fault/EOS state differs: " + label)
        eos = eos_executions[repeat]
        require(states[repeat, "scalar_first"][0]["rng_state"] == states[repeat, "scalar_second"][0]["rng_state"] ==
                eos["base_rng"] and states[repeat, "fault_eos_scalar_first"][0]["rng_state"] ==
                states[repeat, "fault_eos_first"][0]["rng_state"] == eos["expected_rng"], "Exported fault/EOS RNG differs")
        ordinary, sampled = states[repeat, "scalar_first"], states[repeat, "fault_eos_scalar_first"]
        require({key: value for key, value in ordinary[0].items() if key not in ("repeat", "label", "files", "rng_state")} ==
                {key: value for key, value in sampled[0].items() if key not in ("repeat", "label", "files", "rng_state")} and
                all(ordinary[1][role] == sampled[1][role] for role in ROLES - {"snapshot"}),
                "EOS scalar top-k1 target state differs from ordinary scalar")
        first = struct.unpack(f"<{meta['position'] + 1}i", ordinary[1]["pcache"])
        second = struct.unpack(f"<{meta['position'] + 2}i", states[repeat, "scalar_second"][1]["pcache"])
        require(first[:10] == tuple(meta["prefix_ids"]) and first[-2:] == (base_run["emitted0"], base_run["seed"])
                and second[:-1] == first and second[-1] == base_run["truth"], "Fault committed input sequence differs")
        for label in checkpoint_labels:
            ref_label = "fault_eos_scalar_first" if label.startswith("fault_eos_") else "scalar_first"
            for role in CHECKPOINT_ROLES:
                equal(repeat, label, role, states[repeat, ref_label][1][role], checkpoints[repeat, label, role], "f32")
        branch_labels = {f"fault_arm{arm}_{suffix}" for arm in arms
                         for suffix in ("second", "partial_first", "continued_second")} | {"fault_eos_first"}
        for label in branch_labels:
            ref_label = "fault_eos_scalar_first" if label.startswith("fault_eos_") else (
                "scalar_first" if label in first_labels else "scalar_second")
            geometry = state_geometry(meta, meta["position"] + (1 if label in first_labels else 2))
            for role, data in states[repeat, ref_label][1].items():
                equal(repeat, label, role, data, states[repeat, label][1][role], geometry[role][0])
        if repeat:
            for label, ref_label in (("scalar_repeat_first", "scalar_first"), ("scalar_repeat_second", "scalar_second")):
                geometry = state_geometry(meta, meta["position"] + (1 if ref_label == "scalar_first" else 2))
                for role, data in states[0, ref_label][1].items():
                    equal(repeat, label, role, data, states[repeat, ref_label][1][role], geometry[role][0])
    require(len(expected) == comparison_count and set(expected) == set(comparisons), "Missing/extra fault comparisons")
    for key, (dtype, size) in expected.items():
        width = 4 if dtype == "f32" else 2 if dtype == "bf16" else 1
        exact_fields(comparisons[key], {"dtype": dtype, "bytes": size, "elements": size // width},
                     "Fault comparison type/full geometry differs")
    complete = dict(schema_version=1, kind="fault_complete", repeats_completed=3,
                    cases_completed=9 if tree else 6, raw_files=raw_count, same_base_snapshot=True,
                    tree_mode=tree, trace_active=False, performance_measurement=False,
                    bitwise_logits_and_visible_state=True) | {"pass": True}
    require(set(rows[-1]) == set(complete), "Fault completion schema differs")
    exact_fields(rows[-1], complete, "Incomplete fault/EOS gate")
    require(len(files) == raw_count and {path.name for path in raw.iterdir()} == set(files),
            "Extra/missing/unreviewed fault raw evidence")
    return dict(schema_version=1, kind="fixed_speculative_fault_state_review", pass_=True,
                tree_mode=tree, position=meta["position"], repeats=3,
                commit_fault_executions=len(executions), eos_executions=3,
                raw_files=raw_count, comparisons=comparison_count, commit_at=4,
                commit_failure_scalar_fallback_bitwise=True, eos_exactly_one_rng_step=True,
                eos_native_no_fallback=True, bitwise_first_second_partial_resume=True,
                visible_compressed_and_index_cache=True, dynamic_truth=base_run["truth"],
                seed=base_run["seed"], prefix_ids=meta["prefix_ids"],
                performance_measurement=False, inputs_sha256=deps)


def inspect(report, raw):
    report, raw = Path(report), Path(raw)
    require(report.is_file() and not report.is_symlink(), "Missing/linked original report")
    data = report.read_bytes()
    require(data.endswith(b"\n"), "Truncated original report")
    require(raw.is_dir() and not raw.is_symlink(), "Missing/linked raw root")
    rows = [json_value(line) for line in data.splitlines()]
    require(rows and all(isinstance(row, dict) and type(row.get("schema_version")) is int and
                         row["schema_version"] == 1 for row in rows), "Invalid report schema")
    tree = rows[0].get("tree_mode", False)
    require(type(tree) is bool, "Tree mode must be Boolean")
    faults = rows[0].get("fault_mode", False)
    require(type(faults) is bool, "Fault mode must be Boolean")
    if faults:
        return inspect_faults(report, raw, rows, data, tree)
    if tree:
        return inspect_tree(report, raw, rows, data)
    result = _batch2.inspect(report, raw)
    result["inputs_sha256"][str(Path(__file__).resolve())] = sha(__file__)
    result["inputs_sha256"][str(FROZEN_SUPPORT)] = sha(FROZEN_SUPPORT)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("raw", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = inspect(args.report, args.raw)
    import json
    output = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.output:
        output_path = args.output.resolve()
        require(str(output_path) not in result["inputs_sha256"] and
                args.raw.resolve() not in (output_path, *output_path.parents),
                "Review output must not overwrite or extend original evidence")
        args.output.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, TypeError, struct.error) as error:
        raise SystemExit("FAIL: " + str(error))
