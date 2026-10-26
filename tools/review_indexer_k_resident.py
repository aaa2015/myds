#!/usr/bin/env python3
"""Recheck saved K-residency outputs, guards, identities and timing medians.

This reader does not import the builder or the Objective-C benchmark. It
decodes the saved 72-byte ABI and full score bytes directly; it recomputes
every summary from raw GPU timestamps. It proves the recorded component
scope only, never a full-model or production adoption result.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct


def require(test, message):
    if not test:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def template(source, name):
    at = source.index("kernel void " + name + "(")
    start = source.rfind("template<", 0, at)
    depth, end = 1, source.index("{", at) + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def verify_full_native_source(build, identity, generated, combined):
    """Independently replay the frozen production writer without executing it."""
    source = (build / "native.metal").read_bytes()
    require(hashlib.sha256(source).hexdigest() == identity["native_source_sha256"] and
            len(source) == identity["native_source_bytes"], "complete native source identity")
    script = build / "native-source/tools/build_metallib.py"
    require(sha(script) == identity["build_metallib_script_sha256"], "frozen production builder identity")
    wanted = {"PREAMBLE", "SHADER_FILES", "COPY_KERNELS"}
    constants = {}
    for node in ast.parse(script.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    constants[target.id] = ast.literal_eval(node.value)
    require(set(constants) == wanted, "frozen native assembly constants")
    require(constants["SHADER_FILES"] == identity["original_shader_order"], "original shader order")
    require(hashlib.sha256(constants["PREAMBLE"].encode()).hexdigest() == identity["preamble_sha256"], "native preamble")
    require(hashlib.sha256(constants["COPY_KERNELS"].encode()).hexdigest() == identity["copy_kernels_sha256"], "native copy kernels")
    rebuilt = constants["PREAMBLE"].encode() + b"\n"
    records = identity["original_shaders"]
    require(len(records) == len(constants["SHADER_FILES"]), "full shader coverage")
    for index, name in enumerate(constants["SHADER_FILES"]):
        path = build / "native-source/metal" / name
        data = path.read_bytes()
        record = records[index]
        require(record["index"] == index and record["path"] == "metal/" + name and
                record["bytes"] == len(data) and record["sha256"] == hashlib.sha256(data).hexdigest(), "original shader snapshot identity")
        rebuilt += f'\n#line 1 "{name}"\n'.encode()
        require(len(rebuilt) == record["source_begin"], "shader source offset")
        rebuilt += data
        require(len(rebuilt) == record["source_end"] and
                source[record["source_begin"]:record["source_end"]] == data, "original shader embedded bytes")
    rebuilt += constants["COPY_KERNELS"].encode() + b"\n"
    require(rebuilt == source, "independent complete native source reconstruction")
    marker = identity["candidate_append_marker"].encode()
    require(combined.encode() == source + marker + generated.encode() and
            identity["candidate_source_begin"] == len(source) + len(marker), "native complete prefix is unmodified")
    require("struct ds4_metal_args_dsv4_indexer_scores_fused {" not in generated and
            "kernel void kernel_dsv4_indexer_scores_tiled_storage(" not in generated and
            'host_name("kernel_dsv4_indexer_scores_tiled_' not in generated, "candidate append does not redeclare native code")
    return source.decode()


def review_premature_failure(a, build, result, manifest, identity, records, paths):
    """A failing native-vs-native gate is not a candidate numerical failure."""
    require(manifest["status"] == "failed", "known incomplete outcome")
    parity = [r for r in records if r["kind"] == "parity"]
    require(parity, "failed run must preserve at least its parity record")
    examined = []
    first_failure = None
    for row in parity:
        ci, provider = row["case"], row["provider"]
        metadata = next(c for c in manifest["cases"] if c["case"] == ci)
        args = (result / f"case-{ci:03d}-args72.bin").read_bytes()
        (keys, tokens, heads, dim, pos, ratio, qs, hs, ws, ks, ss, scale) = struct.unpack("<6I5Qf4x", args)
        require(heads == 32 and dim == 128 and keys == metadata["keys"] and tokens == metadata["tokens"], "failed run ABI")
        gold = (result / f"case-{ci:03d}-provider-0.scores-strided.f32").read_bytes()
        actual = (result / f"case-{ci:03d}-provider-{provider}.scores-strided.f32").read_bytes()
        guards = (result / f"case-{ci:03d}-provider-{provider}.guards").read_bytes()
        require(len(gold) == len(actual) == tokens * ss, "complete failed output extent")
        bad, first, causal, padding, unwritten, nan_count, invisible = 0, None, True, True, 0, 0, 0
        for t in range(tokens):
            visible = min((pos + t + 1) // ratio, keys)
            for c in range(keys):
                x = struct.unpack_from("<I", gold, t * ss + c * 4)[0]
                y = struct.unpack_from("<I", actual, t * ss + c * 4)[0]
                if x != y:
                    if first is None:
                        first = t * keys + c
                    bad += 1
                unwritten += y == 0xA5A5A5A5
                nan_count += (y & 0x7F800000) == 0x7F800000 and (y & 0x7FFFFF) != 0
                if c >= visible:
                    invisible += 1
                    causal &= y == 0xFF800000
                elif not metadata["special_nan_fixture"]:
                    causal &= (y & 0x7F800000) != 0x7F800000
            padding &= actual[t * ss + keys * 4:(t + 1) * ss] == b"\xa5" * (ss - keys * 4)
        intact = guards == b"\xa5" * 128
        require(bad == row["mismatches"] and first == row["first_mismatch"] and
                causal == row["causal_tail_valid"] and padding == row["padding_unchanged"] and
                intact == row["guards_intact"] and unwritten == row["unwritten"] and
                nan_count == row["nan_count"] and invisible == row["invisible_minus_inf_count"], "failed parity versus raw")
        ok = not bad and causal and padding and intact and not unwritten
        item = dict(case=ci, provider=provider,
                    role=("frozen_native" if provider == 0 else "same_build_native" if provider == 1 else "candidate"),
                    mismatches=bad, first_mismatch=first, guards_intact=intact, causal_tail_valid=causal,
                    padding_unchanged=padding, unwritten=unwritten, bit_exact=not bad)
        examined.append(item)
        if not ok and first_failure is None:
            first_failure = item
    require(first_failure is not None, "premature failure record has no independently reproduced failure")
    failure_role = first_failure["role"]
    candidate_rows = [r for r in examined if r["role"] == "candidate"]
    report = dict(schema_version=1, raw_review_passed=True, run_completed=False, run_status="failed",
                  device=manifest["device"], fixture_kind=manifest["fixture_kind"], raw_files_verified=len(paths),
                  first_failure=first_failure, failure_classification=("native_identity_gate_failed" if failure_role != "candidate" else "candidate_component_gate_failed"),
                  candidate_failed=any(not r["bit_exact"] or not r["guards_intact"] or
                                       not r["causal_tail_valid"] or not r["padding_unchanged"] or r["unwritten"]
                                       for r in candidate_rows),
                  candidate_cases_examined=[r["case"] for r in candidate_rows],
                  candidate_reached_in_first_failing_case=any(r["case"] == first_failure["case"] for r in candidate_rows),
                  examined=examined, fullmodel_speedup_proven=False, production_adoption=False,
                  source_identity_sha256=sha(build / "source-identity.json"),
                  manifest_sha256=sha(result / "artifact-manifest.json"), jsonl_sha256=sha(a.jsonl),
                  reader_sha256=sha(Path(__file__)))
    require(not a.out.exists(), "new report path required")
    a.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build-dir", type=Path, required=True)
    p.add_argument("--result-dir", type=Path, required=True)
    p.add_argument("--jsonl", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    build, result = a.build_dir.resolve(strict=True), a.result_dir.resolve(strict=True)
    m = json.loads((result / "artifact-manifest.json").read_text())
    identity = json.loads((build / "source-identity.json").read_text())
    require(m["status"] in ("complete", "failed"), "known component outcome")
    require(m["probe"] == "indexer_k_resident" and m["fixture_kind"] == "deterministic_synthetic", "scope")
    require(m["fullmodel"] is False and m["production_registered"] is False and m["cold_dram_proven"] is False, "scope inflation")
    require(identity["production_registered"] is False and identity["gpu_work_submitted"] is False, "build scope")
    require(identity["proof"]["reverse_edits_restore_native_byte_for_byte"] is True, "source transformation proof")
    require(sha(build / "probe.metallib") == identity["metallib_sha256"] == m["probe_sha256"], "probe library identity")
    require(sha(build / "bench_indexer_k_resident") == identity["binary_sha256"] == m["binary_sha256"], "binary identity")
    require(sha(build / "probe.metal") == identity["probe_sha256"], "generated source identity")
    combined = (build / "all.metal").read_text()
    generated = (build / "probe.metal").read_text()
    require(combined.endswith(generated), "combined source contains exact generated probe")
    if "combined_source_sha256" in identity:
        require(sha(build / "all.metal") == identity["combined_source_sha256"], "combined source identity")
    if identity.get("build_mode") == "full-library":
        native_source = verify_full_native_source(build, identity, generated, combined)
        require(sha(build / "native.metallib") == identity["native_metallib_sha256"], "complete native library identity")
    else:
        native_source = generated
    if "combined_source_sha256" in identity and identity.get("build_mode") != "full-library":
        require(hashlib.sha256(combined[:-len(generated) - 1].encode()).hexdigest() == identity["preamble_sha256"], "preamble identity")
    native = template(native_source, "kernel_dsv4_indexer_scores_tiled_storage")
    restored = template(generated, "probe_indexer_scores_k_resident")
    for edit in reversed(identity["proof"]["exact_edits"]):
        require(restored.count(edit["after"]) == 1, "unique source reverse anchor")
        restored = restored.replace(edit["after"], edit["before"], 1)
    require(restored == native and hashlib.sha256(native.encode()).hexdigest() ==
            identity["proof"]["native_template_sha256"], "full reverse edit restoration")
    for rel, expected in identity["harness_input_sha256"].items():
        require(sha(build / Path(rel).name) == expected, "snapshot identity: " + rel)
    paths = set()
    for f in m["files"]:
        file = result / f["path"]
        require(not Path(f["path"]).is_absolute() and file.resolve().is_relative_to(result), "raw path containment")
        require(f["path"] not in paths, "duplicate raw path")
        paths.add(f["path"])
        require(file.stat().st_size == f["bytes"] and sha(file) == f["sha256"], "raw bytes/SHA: " + f["path"])
    records = [json.loads(x) for x in a.jsonl.read_text().splitlines() if x]
    if m["status"] == "failed":
        review_premature_failure(a, build, result, m, identity, records, paths)
        return
    require(records[-1]["kind"] == "complete" and records[-1]["cases"] == 16 and
            records[-1]["full_scores_bit_exact"] is True, "complete record")
    parity = [r for r in records if r["kind"] == "parity"]
    require(len(parity) == 48 and len(m["cases"]) == 16, "parity/case coverage")
    checked_cells, checked_output_bytes, observed_nan, invisible_cells = 0, 0, 0, 0
    for case in m["cases"]:
        ci = case["case"]
        args_path = result / f"case-{ci:03d}-args72.bin"
        args = args_path.read_bytes()
        require(len(args) == 72, "args72 size")
        (keys, tokens, heads, dim, pos, ratio, qs, hs, ws, ks, ss, scale) = struct.unpack("<6I5Qf4x", args)
        require(keys == case["keys"] and tokens == case["tokens"] and pos == case["pos0"] and
                ratio == case["ratio"] and heads == 32 and dim == 128 and scale == 1 / 64, "args values")
        require(qs >= 16384 and hs == 512 and ws >= 128 and ks >= 128 * (2 if case["bf16"] else 4) and ss >= keys * 4, "stride ranges")
        require(sha(args_path) == case["args_sha256"], "args identity")
        for filename, key, size in ((f"case-{ci:03d}-q-strided.f32", "q_sha256", tokens * qs),
                                    (f"case-{ci:03d}-w-strided.f32", "weights_sha256", tokens * ws),
                                    (f"case-{ci:03d}-cache-strided.bin", "cache_sha256", keys * ks)):
            path = result / filename
            require(path.stat().st_size == size and sha(path) == case[key], "full input identity")
        gold = (result / f"case-{ci:03d}-provider-0.scores-strided.f32").read_bytes()
        require(len(gold) == tokens * ss, "full score extent")
        nan_count = 0
        invisible = 0
        for t in range(tokens):
            visible = min((pos + t + 1) // ratio, keys)
            bits = struct.unpack_from("<" + str(keys) + "I", gold, t * ss)
            for c, b in enumerate(bits):
                require(b != 0xA5A5A5A5, "unwritten score")
                is_nan = (b & 0x7F800000) == 0x7F800000 and (b & 0x7FFFFF) != 0
                nan_count += is_nan
                if c >= visible:
                    invisible += 1
                    require(b == 0xFF800000, "causal tail is not canonical -Infinity")
                elif not case["special_nan_fixture"]:
                    require((b & 0x7F800000) != 0x7F800000, "nonfinite ordinary visible score")
            require(gold[t * ss + keys * 4:(t + 1) * ss] == b"\xa5" * (ss - keys * 4), "score row padding")
        for provider in range(3):
            actual = (result / f"case-{ci:03d}-provider-{provider}.scores-strided.f32").read_bytes()
            guards = (result / f"case-{ci:03d}-provider-{provider}.guards").read_bytes()
            require(actual == gold, "complete native/candidate bit comparison")
            require(guards == b"\xa5" * 128, "output guards")
            row = next(r for r in parity if r["case"] == ci and r["provider"] == provider)
            require(row["mismatches"] == 0 and row["nan_count"] == nan_count and
                    row["invisible_minus_inf_count"] == invisible and row["unwritten"] == 0 and
                    row["bit_exact"] is True and row["causal_tail_valid"] is True and
                    row["padding_unchanged"] is True and row["guards_intact"] is True, "parity record versus raw")
            checked_cells += tokens * keys
            checked_output_bytes += len(actual)
        observed_nan += nan_count
        invisible_cells += invisible
    timings = [r for r in records if r["kind"] == "timing"]
    summaries = [r for r in records if r["kind"] == "summary"]
    require(len(timings) == 144 and len(summaries) == 8, "timing coverage")
    recomputed = []
    for summary in summaries:
        rows = [r for r in timings if (r["case"], r["control"]) == (summary["case"], summary["control"])]
        require(len(rows) == 18, "warmup/formal arms")
        require([(r["sample"], r["arm"]) for r in rows] ==
                [(s, arm) for s in range(6) for arm in ("A1", "B", "A2")], "ordered A1/B/A2")
        values = {arm: [] for arm in ("A1", "B", "A2")}
        for row in rows:
            us = (row["gpu_end_s"] - row["gpu_start_s"]) * 1e6
            require(math.isfinite(us) and us > 0 and abs(us - row["gpu_us"]) < 1e-6, "GPU timestamp arithmetic")
            require(row["provider"] == (2 if row["arm"] == "B" else row["control"]), "actual provider per arm")
            require(bool(row["warmup"]) == (row["sample"] == 0) and row["full_scores_bit_exact"] is True, "warmup and per-sample parity")
            if row["sample"]:
                values[row["arm"]].append(us)
        a1, b, a2 = (statistics.median(values[x]) for x in ("A1", "B", "A2"))
        saved, drift = (1 - b / min(a1, a2)) * 100, (a2 / a1 - 1) * 100
        for key, computed in (("a1_us", a1), ("b_us", b), ("a2_us", a2),
                              ("conservative_time_saved_percent", saved), ("control_drift_percent", drift)):
            require(abs(summary[key] - computed) < 1e-6, "summary arithmetic: " + key)
        recomputed.append(dict(case=summary["case"], control=summary["control"], bf16=summary["bf16"],
                               a1_us=a1, b_us=b, a2_us=a2, conservative_time_saved_percent=saved, control_drift_percent=drift))
    report = dict(schema_version=1, passed=True, device=m["device"], fixture_kind=m["fixture_kind"],
                  raw_files_verified=len(paths), cases=16, parity_records=48, timing_records=144,
                  checked_score_cells=checked_cells, checked_output_bytes=checked_output_bytes,
                  visible_nan_values_observed=observed_nan, invisible_cells_checked=invisible_cells,
                  summaries=recomputed, fullmodel_speedup_proven=False, production_adoption=False,
                  source_identity_sha256=sha(build / "source-identity.json"),
                  manifest_sha256=sha(result / "artifact-manifest.json"), jsonl_sha256=sha(a.jsonl),
                  reader_sha256=sha(Path(__file__)))
    require(not a.out.exists(), "new report path required")
    a.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
