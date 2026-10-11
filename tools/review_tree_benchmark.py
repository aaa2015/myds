#!/usr/bin/env python3
"""CPU post-audit of an eager -> lazy -> tree -> eager benchmark window.

Replays the native benchmark auditor, then requires observed native Batch2
and tree execution. This reports measurements, not an adoption decision.
No GPU, subprocess, network or model-weight reads are performed.
"""

import argparse
import datetime
import hashlib
import json
import math
import posixpath
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import bench_reproduce as bench


ARMS = {"A1": ("0", "0"), "B": ("1", "0"), "C": ("1", "1"), "A2": ("0", "0")}
SELECTORS = {"MYDS_BATCH2_LAZY_HEAD", "MYDS_TREE_SPEC"}
FIXED_ENV = {"MYDS_PLD": "0", "MYDS_BATCH2_SPEC": "1", "MYDS_SPEC_SERIAL_PROBE": "0",
             "MYDS_SPEC_COST_SCHEDULER": "0", "MYDS_DSPARK_MARKOV_ONLY": "0",
             "MYDS_DSPARK_NO_MARKOV": "1", "MYDS_DSPARK_DEBUG": "0",
             "MYDS_DSPARK_CAPTURE_BYPASS": "0", "MYDS_ENGINE_CAPTURE_BYPASS": "0",
             "MYDS_DECODE_METRICS": "1", "MYDS_PREFIX_CACHE": "0", "MYDS_KV_DSK": "0",
             "MYDS_ATTN_OUT_A_SHARED2": "0"}
OUTPUT_SIGNATURE = ("run", "warmup", "token_ids", "output_bytes_fnv1a64", "output_bytes",
                    "prompt_tokens", "prefill_tokens", "output_tokens", "decode_intervals",
                    "stop_reason", "final_logits")
PATH_COUNTERS = ("target_forward_calls", "target_rows", "batch2_calls", "batch2_hits",
                 "batch2_misses", "batch2_failures", "batch2_fallbacks", "draft_cycles",
                 "draft_proposed", "draft_accepted", "draft_rejects", "confidence_rejects",
                 "margin_rejects", "backoff_steps", "pld_queries", "pld_cycles",
                 "pld_proposed", "pld_accepted")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def strict_json(data):
    def constant(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    return json.loads(data, object_pairs_hook=strict_object, parse_constant=constant)


def read_original(path, deps):
    require(path.is_file() and not path.is_symlink(), "Missing/linked original evidence: " + str(path))
    data = path.read_bytes()
    require(data.endswith(b"\n"), "Truncated original evidence: " + str(path))
    deps[str(path.resolve())] = hashlib.sha256(data).hexdigest()
    return data


def rate(numerator, denominator):
    return numerator / denominator if denominator else None


def canonical_macos_tmp(path):
    """Normalize only macOS's fixed /tmp -> /private/tmp path spelling.

    Archived paths belong to the measured Mac, so resolving them on the
    reviewer host would accept unrelated local symlinks or miss this alias.
    Other directory names and symlink aliases retain their exact identity.
    """
    path = posixpath.normpath(path)
    return "/private" + path if path == "/tmp" or path.startswith("/tmp/") else path


def native_rows(samples, tree):
    """Prove the third-row graph from the production wrapper's row accounting.

    Scalar calls add one forward and one row. A native Batch2 adds one
    forward and two rows; a native tree adds its third row. Therefore
    rows - forwards - Batch2 calls is the number of actual tree calls.
    This remains valid for the ordinary arm's scalar B-branch rescue.
    """
    result = []
    require(samples, "Missing native benchmark samples")
    for sample in samples:
        telemetry = sample.get("decode_telemetry")
        require(isinstance(telemetry, dict), "Missing native decode telemetry")
        for key in PATH_COUNTERS:
            require(type(telemetry.get(key)) is int and telemetry[key] >= 0, "Invalid native counter: " + key)
        calls, hits, misses = (telemetry[key] for key in ("batch2_calls", "batch2_hits", "batch2_misses"))
        require(calls > 0 and hits > 0, "Every warm/formal sample must execute native Batch2 with a hit")
        require(telemetry["batch2_failures"] == telemetry["batch2_fallbacks"] == 0,
                "Scalar recovery cannot validate a failed native graph")
        require(hits + misses == calls, "Native Batch2 outcomes do not conserve calls")
        require(telemetry["draft_cycles"] > 0 and telemetry["draft_proposed"] > 0,
                "Loaded neural drafter did not propose in this request")
        require(all(telemetry[key] == 0 for key in ("pld_queries", "pld_cycles", "pld_proposed", "pld_accepted")),
                "PLD work invalidates the neural-only cohort")
        actual_tree = telemetry["target_rows"] - telemetry["target_forward_calls"] - calls
        require(0 < actual_tree <= calls if tree else actual_tree == 0,
                "Configured tree selector lacks matching actual three-row execution")
        result.append(dict(run=sample["run"], warmup=sample["warmup"], output_tokens=sample["output_tokens"],
                           target_forward_calls=telemetry["target_forward_calls"], target_rows=telemetry["target_rows"],
                           batch2_calls=calls, batch2_hits=hits, batch2_misses=misses,
                           actual_tree_calls=actual_tree,
                           batch2_native_hit_rate=rate(hits, calls),
                           reported_neural_draft_acceptance_rate=rate(telemetry["draft_accepted"], telemetry["draft_proposed"]),
                           draft_proposed=telemetry["draft_proposed"], draft_accepted=telemetry["draft_accepted"],
                           draft_rejects=telemetry["draft_rejects"], backoff_steps=telemetry["backoff_steps"]))
    return result


def percentiles(values):
    distribution = bench.distribution(values)
    require(distribution is not None and all(math.isfinite(value) for value in values), "Missing/nonfinite measured distribution")
    return {"samples": values, "count": distribution["count"], "p50": distribution["median"],
            "p95": distribution["p95"], "p99": distribution["p99"],
            "min": distribution["min"], "max": distribution["max"], "tail_basis": "nearest-rank"}


def inspect(root):
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), "Missing/linked benchmark window")
    audit = bench.load_audit()
    deps = {str(Path(__file__).resolve()): bench.sha(Path(__file__).resolve()),
            str(Path(bench.__file__).resolve()): bench.sha(Path(bench.__file__).resolve()),
            str(Path(audit.__file__).resolve()): bench.sha(Path(audit.__file__).resolve())}
    states, groups, samples, evidence = {}, {}, {}, {}
    for arm, (lazy, tree) in ARMS.items():
        directory = root / ("bench-" + arm)
        require(directory.is_dir() and not directory.is_symlink(), "Missing/distinct arm directory: " + arm)
        state = strict_json(read_original(directory / "run-state.json", deps))
        require(isinstance(state, dict) and state.get("status") == "validated" and
                type(state.get("returncode")) is int and state["returncode"] == 0, "Incomplete/failed arm: " + arm)
        metrics = read_original(directory / "run-metrics.jsonl", deps)
        records = [strict_json(line) for line in metrics.splitlines()]
        require(records and all(isinstance(row, dict) for row in records), "Invalid native metrics: " + arm)
        group, original_samples = bench.validate_run(directory, state, audit)
        meta = group["metadata"]
        loaded = next(row for row in records if row.get("kind") == "load")
        require(meta.get("protocol_version") == 2 and meta.get("repeats") == 3 and meta.get("warmup_runs") == 1 and
                meta.get("max_new_tokens") == 256 and meta.get("max_seq_len") == 4096,
                "Expected protocol2, warm + 3, 256-token cap and context4096: " + arm)
        require(loaded.get("native_v41") is True and loaded.get("dspark_loaded") is True and
                loaded.get("layers") == 40 and loaded.get("dim") == 5120 and loaded.get("vocab_size") == 129280,
                "Native target and actual DSpark support load required: " + arm)
        env = state.get("environment")
        require(isinstance(env, dict) and all(isinstance(key, str) and key.startswith("MYDS_") and
                isinstance(value, str) for key, value in env.items()), "Invalid recorded environment: " + arm)
        required_env = FIXED_ENV | {"MYDS_BATCH2_LAZY_HEAD": lazy, "MYDS_TREE_SPEC": tree}
        require(all(env.get(key) == value and meta["environment"].get(key) == value
                    for key, value in required_env.items()), "Requested/effective path selectors differ: " + arm)
        require(state.get("arm") == "custom" and state.get("dspark_model") is not None,
                "Expected the frozen custom support-model cohort: " + arm)
        source = state.get("source")
        require(isinstance(source, dict) and source.get("dirty") is False and
                isinstance(source.get("head"), str) and re.fullmatch(r"[0-9a-f]{40}", source["head"]),
                "Benchmark source must be committed and clean: " + arm)
        for key in ("binary_sha256", "metallib_sha256", "request_sha256"):
            require(isinstance(state.get(key), str) and re.fullmatch(r"[0-9a-f]{64}", state[key]),
                    "Missing immutable build/request identity: " + arm + "/" + key)
        log = read_original(directory / "run.log", deps).decode("utf-8")
        libraries = re.findall(r"loaded precompiled library from (.+?) \(all required V4\.1 kernels verified\)", log)
        require(len(libraries) == 1, "Exactly one observed precompiled Metal library required: " + arm)
        library = libraries[0] if posixpath.isabs(libraries[0]) else posixpath.join(state["cwd"], libraries[0])
        expected_library = posixpath.join(posixpath.dirname(state["argv"][0]), "myds.metallib")
        require(canonical_macos_tmp(library) == canonical_macos_tmp(expected_library),
                "Runtime library differs from the adjacent library SHA binding: " + arm)
        observed = native_rows(original_samples, tree == "1")
        formal = [sample for sample in original_samples if not sample["warmup"]]
        timing = {name: percentiles([sample[name] for sample in formal]) for name in
                  ("decode_ms_per_token", "prefill_ms_per_token", "ttft_s", "wall_s", "decode_s")}
        timing["callback_intervals_ms"] = percentiles([value for sample in formal for value in sample["callback_intervals_ms"]])
        official = [row for row in observed if not row["warmup"]]
        evidence[arm] = dict(lazy_head=lazy == "1", tree_requested=tree == "1",
                             timing=timing, observed_per_run=observed,
                             formal_native_batch2_calls=sum(row["batch2_calls"] for row in official),
                             formal_actual_tree_calls=sum(row["actual_tree_calls"] for row in official),
                             formal_batch2_native_hit_rate=rate(sum(row["batch2_hits"] for row in official),
                                                               sum(row["batch2_calls"] for row in official)),
                             formal_reported_neural_draft_acceptance_rate=rate(sum(row["draft_accepted"] for row in official),
                                                                               sum(row["draft_proposed"] for row in official)),
                             actual_output_tokens=group["output_tokens"], stop_reason=group["stop_reason"],
                             observed_precompiled_library=posixpath.normpath(library),
                             reported_precompiled_library_path=libraries[0],
                             expected_adjacent_library_path=expected_library,
                             canonical_precompiled_library_path=canonical_macos_tmp(library),
                             library_path_alias_rule="Only macOS /tmp <-> /private/tmp; no arbitrary alias resolution")
        states[arm], groups[arm], samples[arm] = state, group, original_samples
        require(bench.sha(directory / "run-state.json") == deps[str((directory / "run-state.json").resolve())] and
                bench.sha(directory / "run-metrics.jsonl") == deps[str((directory / "run-metrics.jsonl").resolve())],
                "Original evidence changed during CPU review: " + arm)

    base = states["A1"]
    identities = ("argv", "cwd", "binary_sha256", "metallib_sha256", "source", "request_sha256",
                  "request_bytes", "request_fnv1a64", "model", "dspark_model", "arm")
    for arm in ARMS:
        require(all(states[arm].get(key) == base.get(key) for key in identities), "Same build/model/request identity differs: " + arm)
        require(all(states[arm]["host_before"].get(key) == base["host_before"].get(key)
                    for key in ("cpu", "memory_bytes", "os")), "Host identity differs: " + arm)
        require(all(groups[arm]["metadata"].get(key) == groups["A1"]["metadata"].get(key)
                    for key in audit.MATCH_META), "Generation configuration differs: " + arm)
        require({key: value for key, value in states[arm]["environment"].items() if key not in SELECTORS} ==
                {key: value for key, value in base["environment"].items() if key not in SELECTORS},
                "Unrelated environment changes invalidate the comparison: " + arm)
        require({key: value for key, value in groups[arm]["metadata"]["environment"].items() if key not in SELECTORS} ==
                {key: value for key, value in groups["A1"]["metadata"]["environment"].items() if key not in SELECTORS},
                "Unrelated effective environment changes invalidate the comparison: " + arm)
        require(all(all(before.get(key) == after.get(key) for key in OUTPUT_SIGNATURE)
                    for before, after in zip(samples["A1"], samples[arm])), "Token/byte/final-logits output differs: " + arm)
    require(states["A2"]["environment"] == base["environment"], "Final eager control environment differs")
    # Eager and lazy change only deferred LM-head scheduling. Their neural
    # proposal/acceptance and target-row schedules must remain comparable.
    for arm in ("B", "A2"):
        require(all(all(before["decode_telemetry"][key] == after["decode_telemetry"][key] for key in PATH_COUNTERS)
                    for before, after in zip(samples["A1"], samples[arm])), "Eager/lazy/control target or proposal schedule differs: " + arm)
    timestamps = [datetime.datetime.fromisoformat(states[arm]["utc"].replace("Z", "+00:00")) for arm in ARMS]
    require(all(before < after for before, after in zip(timestamps, timestamps[1:])), "Eager/lazy/tree/eager start order differs")
    drift = {name: 100 * (evidence["A2"]["timing"][name]["p50"] / evidence["A1"]["timing"][name]["p50"] - 1)
             for name in ("decode_ms_per_token", "prefill_ms_per_token", "ttft_s", "wall_s")}
    relative = {arm: {name: 100 * (evidence[arm]["timing"][name]["p50"] / evidence["A1"]["timing"][name]["p50"] - 1)
                     for name in ("decode_ms_per_token", "prefill_ms_per_token", "ttft_s", "wall_s")}
                for arm in ("B", "C")}
    return dict(schema_version=1, kind="tree_benchmark_cpu_postreview", audit_pass=True,
                scope="Same-build eager/lazy/tree/eager DSpark measurement; configured 256-token cap, actual output count reported; no adoption or pure-target speed claim",
                performance_adoption_decision=False, cpu_replay_only=True, source=base["source"],
                build=dict(binary_sha256=base["binary_sha256"], metallib_sha256=base["metallib_sha256"]),
                model_identity=base["model"], support_model_identity=base["dspark_model"],
                all_warm_and_formal_token_bytes_final_logits_match=True,
                all_native_failures_and_fallbacks_zero=True, eager_lazy_control_schedules_match=True,
                actual_tree_counter_formula="target_rows - target_forward_calls - batch2_calls",
                percentile_scope="Three formal request samples; nearest-rank p95/p99 equal their maximum. Callback tails pool formal token gaps, including paired zero gaps.",
                final_control_drift_percent=drift, candidate_median_difference_percent_relative_eager_a1=relative,
                arms=evidence, inputs_sha256=deps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Window root containing bench-A1, bench-B, bench-C, bench-A2")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inspect(args.root)
    output = args.output.resolve()
    require(not output.exists() and str(output) not in result["inputs_sha256"], "Review output already exists or overwrites evidence")
    require(all((args.root / ("bench-" + arm)).resolve() not in (output, *output.parents) for arm in ARMS),
            "Review output must not extend an original arm directory")
    args.output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"audit_pass": True, "output": str(args.output), "actual_tree_calls":
                      {arm: result["arms"][arm]["formal_actual_tree_calls"] for arm in ARMS}}, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, TypeError, StopIteration, UnicodeError) as error:
        raise SystemExit("FAIL: " + str(error))
