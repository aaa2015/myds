#!/usr/bin/env python3
"""Independently recheck retained register-only P stdout; CPU, no model or GPU.

Uses exact decimal fractions and raw-inventory bindings, not any project reader
or a reported summary. The tensor payloads remain remote: matching inventory
hashes is explicitly not an independent reread of those full payloads.
"""
import argparse
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
from pathlib import Path


def require(ok, message):
    if not ok:
        raise ValueError(message)


def pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def loads(text):
    return json.loads(text, parse_float=Decimal, object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def descriptor(path):
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha(path)}


def number(value):
    require(type(value) in (int, Decimal), "timing must be a JSON number, not bool/string")
    result = Fraction(value)
    require(result > 0, "positive timing required")
    return result


def describe(value):
    return {"numerator": value.numerator, "denominator": value.denominator,
            "decimal": str(Decimal(value.numerator) / Decimal(value.denominator))}


def review(root, repo):
    protocol_path = root / "fullmodel-P-v1-protocol.json"
    protocol = loads(protocol_path.read_text())
    gate = protocol["performance_gate"]
    require(gate["formal_runs"] == [1, 2, 3] and gate["warmup_run"] == 0, "frozen run set")
    require(gate["primary_min_saved_percent_strict"] == 2 and
            gate["max_abs_control_drift_percent_inclusive"] == 1 and
            gate["max_nonregression_percent_inclusive"] == Decimal("0.5"), "frozen thresholds")
    providers = {p["name"]: p for p in protocol["providers"]}
    require(set(providers) == {"A1", "B", "A2"}, "provider set")
    report = {"kind": "independent_prior_P_recalculation_for_integration", "schema_version": 1,
              "protocol": descriptor(protocol_path), "GPU_work_submitted": False,
              "full_tensor_payloads_reread": False, "production_bridge_validated": False,
              "limitations": ["Tensor SHA agreements below compare retained inventories, not remote payload bytes.",
                              "Prior private native benchmark does not prove current deployed executable or HTTP equivalence."],
              "workloads": {}, "sources": []}
    for workload in protocol["workloads"]:
        name = workload["name"]
        output = {"prompt_tokens": workload["prompt_tokens"], "arms": {}, "metrics": {}}
        invariant_tokens = None
        invariant_raw = None
        for arm in ("A1", "B", "A2"):
            folder = root / "fullmodel-P-v1-native-metadata" / name / arm
            stdout = folder / "stdout.jsonl"
            inventory_path = folder / "raw-inventory.json"
            inventory = loads(inventory_path.read_text())
            window = loads((folder / "window.json").read_text())
            plan = loads((folder / "execution-plan.json").read_text())
            require(window["exit_code"] == 0 and window["abort_reason"] is None, "window completed")
            require(window["raw_inventory_sha256"] == sha(inventory_path), "inventory binding")
            require(window["protocol_sha256"] == sha(protocol_path), "protocol binding")
            require(inventory["files"]["stdout.jsonl"] == {
                "bytes": stdout.stat().st_size, "sha256": sha(stdout)}, "stdout inventory identity")
            provider = providers[arm]
            require(plan["binary_sha256"] == provider["binary"]["sha256"] and
                    plan["library_sha256"] == provider["library"]["sha256"] and
                    plan["mode"] == provider["mode"], "planned provider identity")
            records = [loads(line) for line in stdout.read_text().splitlines() if line.strip()]
            samples = [r for r in records if r.get("kind") == "sample"]
            metadata = [r for r in records if r.get("kind") == "metadata"]
            config = [r for r in records if r.get("kind") == "register_only_metadata"]
            loaded = [r for r in records if r.get("kind") == "load"]
            require(len(samples) == 4 and [r["run"] for r in samples] == [0, 1, 2, 3], "all four runs exactly once")
            require(len(metadata) == len(config) == len(loaded) == 1, "metadata cardinality")
            require(metadata[0]["max_seq_len"] == 1048576 and metadata[0]["temperature"] == 0 and
                    metadata[0]["decode_basis"] == "first_to_last_callback/(tokens-1)", "native metric contract")
            require(config[0]["runtime_mode"] == provider["mode"] and
                    config[0]["effective_configuration_count"] == 68, "actual configuration")
            actual_path = loaded[0]["actual_metallib_path"]
            expected_path = provider["expected_loaded_library_path"]
            require(actual_path == expected_path or
                    (actual_path.startswith("/tmp/") and "/private" + actual_path == expected_path),
                    "actual loaded library path")
            for sample in samples:
                run = sample["run"]
                require(sample["warmup"] is (run == 0), "warmup contract")
                require(sample["generation_ok"] is True and sample["sample_valid"] is True and
                        sample["fresh_context"] is True and sample["stop_reason"] == "length", "valid complete sample")
                require(sample["prompt_tokens"] == workload["prompt_tokens"] and
                        sample["cached_tokens"] == 0 and sample["output_tokens"] == 256 and
                        sample["decode_intervals"] == 255 and len(sample["token_ids"]) == 256, "token/cache counts")
                counter = sample["gpu_indexer_register_only_provider"]
                expected_encoded = {"business96": 342, "code38404": 417}[name] if arm == "B" else 0
                require(counter["encoded"] == expected_encoded and
                        counter["bf16_dispatches"] == expected_encoded and
                        counter["through_first_callback_encoded"] == expected_encoded and
                        counter["adoption_valid"] is True and
                        all(counter[k] == 0 for k in ("decode_encoded", "fallbacks", "failures", "f32_dispatches")),
                        "actual candidate scope and adoption")
                require(sample["final_logits"]["finite"] is True, "finite final logits")
                tokens = sample["token_ids"]
                if invariant_tokens is None:
                    invariant_tokens = tokens
                require(tokens == invariant_tokens, "all token IDs equal across all arms/runs")
                raw = {k.replace(f"run-{run}", "run-N"): v for k, v in inventory["files"].items()
                       if (k == f"logits/run-{run}.f32" or k.startswith(f"state/run-{run}/"))
                       and not k.endswith("manifest.json")}
                require(len(raw) > 100, "complete retained numerical inventory entries")
                if invariant_raw is None:
                    invariant_raw = raw
                require(raw == invariant_raw, "logits/state/output inventory hashes agree for every arm/run")
            formal = samples[1:]
            output["arms"][arm] = {"stdout": descriptor(stdout), "raw_inventory": descriptor(inventory_path),
                "median": {metric: describe(sorted(number(s[metric]) for s in formal)[1])
                           for metric in ("prefill_s", "ttft_s", "decode_ms_per_token")},
                "formal_samples": [{"run": s["run"], **{metric: str(s[metric]) for metric in
                                      ("prefill_s", "ttft_s", "decode_ms_per_token")}} for s in formal]}
        for metric in ("prefill_s", "ttft_s", "decode_ms_per_token"):
            values = []
            for arm in ("A1", "B", "A2"):
                d = output["arms"][arm]["median"][metric]
                values.append(Fraction(d["numerator"], d["denominator"]))
            a1, b, a2 = values
            saved = 100 * (1 - b / min(a1, a2))
            drift = 100 * (a2 / a1 - 1)
            primary = metric == "prefill_s"
            passed = abs(drift) <= 1 and (saved > 2 if primary else saved >= Fraction(-1, 2))
            output["metrics"][metric] = {"saved_percent": describe(saved), "control_drift_percent": describe(drift),
                                           "pass": passed, "primary": primary}
        output["all_gates_pass"] = all(m["pass"] for m in output["metrics"].values())
        output["tokens_and_retained_raw_inventory_hashes_equal"] = True
        report["workloads"][name] = output
    transform = loads((root / "MMA-source-transform.json").read_text())
    component = loads((root / "head-control-v2-build-manifest.json").read_text())
    target = loads((root / "runtime-CPU-preparation-v2/M2-target-build-manifest.json").read_text())
    candidate = repo / "tools/indexer_register_only_head_control_v2.metal.inc"
    builder = repo / "tools/build_indexer_register_only_probe_v2.py"
    patch = root / "runtime-CPU-preparation-v2/private-runtime-core.patch"
    require(sha(candidate) == transform["v2_candidate"]["sha256"], "candidate source identity")
    require(sha(builder) == transform["v2_builder"]["sha256"], "builder source identity")
    require(sha(patch) == target["runtime_core_patch"]["sha256"], "runtime patch identity")
    require(all(p["library"]["sha256"] == component["artifacts"]["probe.metallib"]["sha256"]
                == target["libraries"]["private"]["sha256"] for p in providers.values()), "correct v2 library identity")
    report["frozen_candidate_sources"] = [descriptor(p) for p in (candidate, builder, patch)]
    report["prior_native_gates_pass"] = all(w["all_gates_pass"] for w in report["workloads"].values())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = review(args.evidence, args.repo)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({"prior_native_gates_pass": result["prior_native_gates_pass"],
                      "output": str(args.output), "full_tensor_payloads_reread": False}))


if __name__ == "__main__":
    main()
