#!/usr/bin/env python3
"""CPU post-audit of an ordinary -> depth1 -> depth2 -> ordinary benchmark window.

Replays the native benchmark auditor, then requires observed native Batch2
and continuous three-row execution. This reports measurements, not an adoption decision.
--shared3-window additionally binds A1/D1/R/B/A2 to observed dense, shared
Gate/Up and routed three-input providers, with R retaining the old depth2 path.
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


ARMS = {"A1": ("0", "1"), "D1": ("1", "1"), "B": ("1", "2"), "A2": ("0", "1")}
SHARED3_ARMS = {"A1": ("0", "1"), "D1": ("1", "1"), "R": ("1", "2"),
                "B": ("1", "2"), "A2": ("0", "1")}
SHARED3_MASTER = {"A1": "1", "D1": "1", "R": "0", "B": "1", "A2": "1"}
SELECTORS = {"MYDS_BATCH2_SPEC", "MYDS_DSPARK_DEPTH"}
SHARED3_COUNTERS = {
    "gpu_depth2_shared3_dense": ("requests", "encoded", "failures", "q8_encoded", "q4_encoded"),
    "gpu_depth2_shared3_gu": ("requests", "encoded", "failures"),
    "gpu_depth2_shared3_routed": ("requests", "encoded", "failures"),
}
FIXED_ENV = {"MYDS_PLD": "0", "MYDS_BATCH2_LAZY_HEAD": "1", "MYDS_TREE_SPEC": "0", "MYDS_DSPARK_EVAL_PREFIX": "1", "MYDS_SPEC_SERIAL_PROBE": "0",
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


def native_rows(samples, spec, depth):
    result=[]
    keys=(*PATH_COUNTERS, 'depth2_calls', 'depth2_hits1', 'depth2_hits2', 'depth2_misses', 'depth2_failures', 'depth2_fallbacks')
    for sample in samples:
        t=sample['decode_telemetry']
        require(all(type(t.get(k)) is int and t[k]>=0 for k in keys), 'Invalid path counters')
        require(t['depth2_failures']==t['depth2_fallbacks']==t['batch2_failures']==t['batch2_fallbacks']==0, 'Native failure/fallback')
        require(t['batch2_calls']==t['batch2_hits']+t['batch2_misses'], 'Compatibility outcomes differ')
        require(t['depth2_calls']==t['depth2_hits1']+t['depth2_hits2']+t['depth2_misses'], 'Depth2 outcomes differ')
        require(t['target_rows']-t['target_forward_calls']-t['batch2_calls']==t['depth2_calls'], 'Actual three-row accounting differs')
        require(all(t[k]==0 for k in ('pld_queries','pld_cycles','pld_proposed','pld_accepted')), 'PLD invalidates neural cohort')
        if spec=='0':
            require(t['draft_cycles']==t['draft_proposed']==t['batch2_calls']==t['depth2_calls']==0 and
                t['target_forward_calls']==t['target_rows']==sample['output_tokens'], 'Ordinary arm executed drafter')
        elif depth=='2':
            require(t['depth2_calls']>0 and t['depth2_hits2']>0 and t['draft_proposed']>0, 'Actual continuous depth2/all-hit absent')
        else:require(t['batch2_calls']>0 and t['batch2_hits']>0 and t['depth2_calls']==0, 'Actual depth1 absent')
        result.append(dict(run=sample['run'],warmup=sample['warmup'],output_tokens=sample['output_tokens'],
            **{k:t[k] for k in keys}, native_draft_acceptance_rate=rate(t['draft_accepted'],t['draft_proposed'])))
    return result


def shared3_rows(samples, arm):
    """Bind a true shared provider to each observed three-row forward.

    Total counters reset before the request. Decode counters subtract the
    first callback snapshot; equality requires zero shared3 execution before
    that first, ordinary token. Encoding counts establish selected providers,
    not actual device bandwidth or a performance benefit.
    """
    result = []
    for sample in samples:
        observed = {"run": sample["run"], "warmup": sample["warmup"]}
        calls = sample["decode_telemetry"]["depth2_calls"]
        for name, fields in SHARED3_COUNTERS.items():
            counters = sample.get(name)
            expected_fields = {*fields, *("decode_" + field for field in fields)}
            require(isinstance(counters, dict) and set(counters) == expected_fields and
                    all(type(value) is int and value >= 0 for value in counters.values()),
                    "Missing/invalid shared3 counter protocol: " + arm + "/" + name)
            require(all(counters[field] == counters["decode_" + field] for field in fields),
                    "Shared3 executed before the ordinary first callback: " + arm + "/" + name)
            if arm == "B":
                expected = calls * 40 * (5 if name == "gpu_depth2_shared3_dense" else 1)
                require(expected > 0 and counters["requests"] == counters["encoded"] == expected and
                        counters["failures"] == 0,
                        "Actual shared3 execution count differs: " + arm + "/" + name)
                if name == "gpu_depth2_shared3_dense":
                    require(counters["q4_encoded"] > 0 and
                            counters["q4_encoded"] + counters["q8_encoded"] == counters["encoded"],
                            "Shared3 Q4/Q8 type accounting differs: " + arm)
            else:
                require(all(value == 0 for value in counters.values()),
                        "Inactive arm executed a shared3 provider: " + arm + "/" + name)
            observed[name] = dict(counters)
        cycles = sample["decode_telemetry"].get("cycle_output_tokens")
        require(isinstance(cycles, list) and cycles and cycles[0] == 1,
                "First callback must belong to a one-token ordinary cycle: " + arm)
        result.append(observed)
    return result


def percentiles(values):
    distribution = bench.distribution(values)
    require(distribution is not None and all(math.isfinite(value) for value in values), "Missing/nonfinite measured distribution")
    return {"samples": values, "count": distribution["count"], "p50": distribution["median"],
            "p95": distribution["p95"], "p99": distribution["p99"],
            "min": distribution["min"], "max": distribution["max"], "tail_basis": "nearest-rank"}


def inspect(root, shared3_window=False):
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), "Missing/linked benchmark window")
    arms = SHARED3_ARMS if shared3_window else ARMS
    selectors = SELECTORS | ({"MYDS_DEPTH2_SHARED3"} if shared3_window else set())
    audit = bench.load_audit()
    deps = {str(Path(__file__).resolve()): bench.sha(Path(__file__).resolve()),
            str(Path(bench.__file__).resolve()): bench.sha(Path(bench.__file__).resolve()),
            str(Path(audit.__file__).resolve()): bench.sha(Path(audit.__file__).resolve())}
    states, groups, samples, evidence = {}, {}, {}, {}
    for arm, (spec, depth) in arms.items():
        directory = root / ("bench-" + arm)
        require(directory.is_dir() and not directory.is_symlink(), "Missing/distinct arm directory: " + arm)
        state = strict_json(read_original(directory / "run-state.json", deps))
        require(isinstance(state, dict) and state.get("status") == "validated" and
                type(state.get("returncode")) is int and state["returncode"] == 0, "Incomplete/failed arm: " + arm)
        metrics = read_original(directory / "run-metrics.jsonl", deps)
        records = [strict_json(line) for line in metrics.splitlines()]
        require(records and all(isinstance(row, dict) for row in records), "Invalid native metrics: " + arm)
        group, original_samples = bench.validate_run(directory, state, audit)
        # This frozen cohort measures 256 emitted tokens. A legal EOS adds an
        # un-emitted target call and a trailing wall-time interval; that needs
        # its own timing/row protocol instead of this length-only equality.
        require(all(sample.get('stop_reason') == 'length' and sample.get('output_tokens') == 256
                    for sample in original_samples), 'Expected length-stop 256-token cohort: ' + arm)
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
        required_env = FIXED_ENV | {"MYDS_BATCH2_SPEC": spec, "MYDS_DSPARK_DEPTH": depth}
        if shared3_window:
            require(all(recorded_env.get("MYDS_DEPTH2_SHARED3_SHADOW") in (None, "")
                        for recorded_env in (env, meta["environment"])),
                    "Shared3 shadow adds diagnostic GPU joins and invalidates timing: " + arm)
            required_env["MYDS_DEPTH2_SHARED3"] = SHARED3_MASTER[arm]
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
        observed = native_rows(original_samples, spec, depth)
        formal = [sample for sample in original_samples if not sample["warmup"]]
        timing = {name: percentiles([sample[name] for sample in formal]) for name in
                  ("decode_ms_per_token", "prefill_ms_per_token", "ttft_s", "wall_s", "decode_s")}
        timing["callback_intervals_ms"] = percentiles([value for sample in formal for value in sample["callback_intervals_ms"]])
        official = [row for row in observed if not row["warmup"]]
        evidence[arm] = dict(spec_requested=spec=='1',depth=int(depth),timing=timing,observed_per_run=observed,
            formal_depth2_calls=sum(r['depth2_calls'] for r in official),
            formal_depth2_hits1=sum(r['depth2_hits1'] for r in official),
            formal_depth2_hits2=sum(r['depth2_hits2'] for r in official),
            formal_reported_neural_acceptance_rate=rate(sum(r['draft_accepted'] for r in official),sum(r['draft_proposed'] for r in official)),
            actual_output_tokens=group['output_tokens'],stop_reason=group['stop_reason'],observed_precompiled_library=library)
        if shared3_window:
            evidence[arm]["shared3_requested"] = SHARED3_MASTER[arm] == "1"
            evidence[arm]["shared3_observed_per_run"] = shared3_rows(original_samples, arm)
        states[arm], groups[arm], samples[arm] = state, group, original_samples
        require(bench.sha(directory / "run-state.json") == deps[str((directory / "run-state.json").resolve())] and
                bench.sha(directory / "run-metrics.jsonl") == deps[str((directory / "run-metrics.jsonl").resolve())],
                "Original evidence changed during CPU review: " + arm)

    base = states["A1"]
    identities = ("argv", "cwd", "binary_sha256", "metallib_sha256", "source", "request_sha256",
                  "request_bytes", "request_fnv1a64", "model", "dspark_model", "arm")
    for arm in arms:
        require(all(states[arm].get(key) == base.get(key) for key in identities), "Same build/model/request identity differs: " + arm)
        require(all(states[arm]["host_before"].get(key) == base["host_before"].get(key)
                    for key in ("cpu", "memory_bytes", "os")), "Host identity differs: " + arm)
        require(all(groups[arm]["metadata"].get(key) == groups["A1"]["metadata"].get(key)
                    for key in audit.MATCH_META), "Generation configuration differs: " + arm)
        require({key: value for key, value in states[arm]["environment"].items() if key not in selectors} ==
                {key: value for key, value in base["environment"].items() if key not in selectors},
                "Unrelated environment changes invalidate the comparison: " + arm)
        require({key: value for key, value in groups[arm]["metadata"]["environment"].items() if key not in selectors} ==
                {key: value for key, value in groups["A1"]["metadata"]["environment"].items() if key not in selectors},
                "Unrelated effective environment changes invalidate the comparison: " + arm)
        require(all(all(before.get(key) == after.get(key) for key in OUTPUT_SIGNATURE)
                    for before, after in zip(samples["A1"], samples[arm])), "Token/byte/final-logits output differs: " + arm)
    require(states["A2"]["environment"] == base["environment"], "Final ordinary control environment differs")
    # Ordinary A1/A2 must execute the same target schedule for drift control.
    for arm in ("A2",):
        require(all(all(before["decode_telemetry"][key] == after["decode_telemetry"][key] for key in PATH_COUNTERS)
                    for before, after in zip(samples["A1"], samples[arm])), "Ordinary control target schedule differs: " + arm)
    if shared3_window:
        schedule = (*PATH_COUNTERS, "depth2_calls", "depth2_hits1", "depth2_hits2", "depth2_misses",
                    "depth2_failures", "depth2_fallbacks")
        require(all(all(before["decode_telemetry"][key] == after["decode_telemetry"][key] for key in schedule)
                    for before, after in zip(samples["R"], samples["B"])),
                "Old/new depth2 target or draft schedule differs")
    timestamps = [datetime.datetime.fromisoformat(states[arm]["utc"].replace("Z", "+00:00")) for arm in arms]
    require(all(before < after for before, after in zip(timestamps, timestamps[1:])),
            "Benchmark arm start order differs: " + "/".join(arms))
    drift = {name: 100 * (evidence["A2"]["timing"][name]["p50"] / evidence["A1"]["timing"][name]["p50"] - 1)
             for name in ("decode_ms_per_token", "prefill_ms_per_token", "ttft_s", "wall_s")}
    relative = {arm: {name: 100 * (evidence[arm]["timing"][name]["p50"] / evidence["A1"]["timing"][name]["p50"] - 1)
                     for name in ("decode_ms_per_token", "prefill_ms_per_token", "ttft_s", "wall_s")}
                for arm in (("D1", "R", "B") if shared3_window else ("D1", "B"))}
    result = dict(schema_version=2 if shared3_window else 1, kind="depth2_benchmark_cpu_postreview", audit_pass=True,
                scope=("Same-build ordinary/depth1/old-depth2/new-shared3-depth2/ordinary" if shared3_window else
                       "Same-build ordinary/depth1/depth2/ordinary") + "; warm plus three formal requests, exact tokens/bytes/final logits; measurements do not themselves enable production adoption",
                performance_adoption_decision=False, cpu_replay_only=True, source=base["source"],
                build=dict(binary_sha256=base["binary_sha256"], metallib_sha256=base["metallib_sha256"]),
                model_identity=base["model"], support_model_identity=base["dspark_model"],
                all_warm_and_formal_token_bytes_final_logits_match=True,
                all_native_failures_and_fallbacks_zero=True, ordinary_control_schedules_match=True,
                actual_depth2_counter_formula="target_rows - target_forward_calls - batch2_calls == depth2_calls",
                percentile_scope="Three formal request samples; nearest-rank p95/p99 equal their maximum. Callback tails pool formal token gaps, including paired zero gaps.",
                final_control_drift_percent=drift, candidate_median_difference_percent_relative_ordinary_a1=relative,
                arms=evidence, inputs_sha256=deps)
    if shared3_window:
        result.update(shared3_window=True, arm_order=list(arms), old_new_depth2_schedules_match=True,
                      all_first_callback_shared3_counters_zero=True,
                      actual_shared3_counter_formula={"dense": "depth2_calls * 40 * 5",
                                                      "gu": "depth2_calls * 40", "routed": "depth2_calls * 40"},
                      new_depth2_median_difference_percent_relative_old_depth2_r={
                          name: 100 * (evidence["B"]["timing"][name]["p50"] / evidence["R"]["timing"][name]["p50"] - 1)
                          for name in ("decode_ms_per_token", "prefill_ms_per_token", "ttft_s", "wall_s")})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path,
                        help="Window root containing bench-A1/D1/B/A2; --shared3-window also requires bench-R")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shared3-window", action="store_true",
                        help="Require A1/D1/R(old depth2)/B(shared3 depth2)/A2 and actual shared3 provider counters")
    args = parser.parse_args()
    result = inspect(args.root, args.shared3_window)
    output = args.output.resolve()
    require(not output.exists() and str(output) not in result["inputs_sha256"], "Review output already exists or overwrites evidence")
    require(all((args.root / ("bench-" + arm)).resolve() not in (output, *output.parents) for arm in result["arms"]),
            "Review output must not extend an original arm directory")
    args.output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"audit_pass": True, "output": str(args.output), "formal_depth2_calls":
                      {arm: result["arms"][arm]["formal_depth2_calls"] for arm in result["arms"]}}, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, TypeError, StopIteration, UnicodeError) as error:
        raise SystemExit("FAIL: " + str(error))
