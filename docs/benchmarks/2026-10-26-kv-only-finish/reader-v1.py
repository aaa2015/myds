#!/usr/bin/env python3
"""Independent stdlib-only KV-only correctness reader. Never imports the
builder/harness, executes a process, creates a device or evaluates timing.
"""
import argparse
import array
import ast
import hashlib
import json
from pathlib import Path
import struct
import sys

PROTOCOL_SHA = "76a066e601e9788b24550d0014d21acdc3e4a285d6672c9696a0e19cfb046b00"
BUILD_SHA = "5d8d279bd6d6287402d71dabec84c89c5fb1b6740547ab7ced87fe24e690b2c4"
HOST_SHA = "cf560a89093af7eb89daf3eef17aa93d4ed3f3d96abb81eac3ac4ec19b41783a"
HOST_STDOUT_SHA = "0fb4004fba534aaabee96a220e71323b3105b318c7c33614050a93770fed4670"
ROLES = ("dummy_q", "kv", "window")
SIZES = dict(dummy_q=131072, kv=2048, window=262144)
NATIVE, CANDIDATE = "kernel_dsv41_qkv_finish", "kernel_myds_probe_kv_only_finish"
LAYERS = [x for x in range(40) if x not in {2, 8, 14, 20, 24, 28, 32, 36}]
CASES = [(l, p) for l in LAYERS for p in (0, 1, 327679)] + [(l, p) for l in (0, 3, 39) for p in (127, 128, 16383, 16384, 65535, 65536)]
ABI = dict(args_bytes=136, frequencies_offset=8, buffer_slots=dict(args=0, dummy_q=1, kv=2, window=3),
           original_groups=[80, 1, 1], candidate_groups=[80, 1, 1], threads=[32, 1, 1],
           dynamic_tg_bytes=0, dummy_q_bytes=131072, kv_bytes=2048, window_bytes=262144)


class EvidenceError(ValueError): pass


def need(value, reason):
    if not value: raise EvidenceError(reason)


def uint(value): return type(value) is int and value >= 0
def same(actual, expected):
    """JSON equality retaining bool/int distinctions in witness declarations."""
    if type(actual) is not type(expected): return False
    if isinstance(actual, dict):
        return actual.keys() == expected.keys() and all(same(actual[k], expected[k]) for k in actual)
    if isinstance(actual, list): return len(actual) == len(expected) and all(same(a, b) for a, b in zip(actual, expected))
    return actual == expected
def digest(data): return hashlib.sha256(data).hexdigest()
def sha(path): return digest(Path(path).read_bytes())


def unique(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def loads(text):
    def bad(value): raise EvidenceError("nonfinite JSON constant: " + value)
    return json.loads(text, object_pairs_hook=unique, parse_constant=bad)


def read_json(path): return loads(Path(path).read_text())


def local_file(root, name):
    need(isinstance(name, str) and name and Path(name).parts == (name,) and name not in (".", ".."), "one local raw filename")
    path = Path(root) / name
    need(path.is_file() and not path.is_symlink(), "retained regular raw file: " + str(path))
    return path


def before_sizes():
    result = {"build-identity.json": None}
    for i in range(114):
        result[f"case{i:03d}.args136.raw"] = 136
        result.update({f"case{i:03d}.{r}.before.raw": SIZES[r] for r in ROLES})
    return result


def after_sizes():
    result = {}
    for i in range(114):
        for p in range(3):
            for r in ROLES:
                stem = f"phase1.case{i:03d}.provider{p}.{r}"
                result[stem + ".raw"] = SIZES[r]
                result[stem + ".prefix-guard.raw"] = result[stem + ".suffix-guard.raw"] = 64
    return result


def indexed_files(manifest, expected):
    rows = manifest["files"]
    need(isinstance(rows, list) and all(set(row) == {"path", "bytes", "sha256"} for row in rows), "complete SHA raw descriptors")
    indexed = {row["path"]: row for row in rows}
    need(len(indexed) == len(rows) and set(indexed) == set(expected), "all and only expected complete raw files")
    for name, size in expected.items():
        need(uint(indexed[name]["bytes"]) and (size is None or indexed[name]["bytes"] == size), "raw descriptor extent: " + name)
    return indexed


def raw(root, descriptor):
    path = local_file(root, descriptor["path"])
    data = path.read_bytes()
    need(len(data) == descriptor["bytes"] and digest(data) == descriptor["sha256"], "full retained bytes/SHA: " + descriptor["path"])
    return data


def words(data):
    need(len(data) % 4 == 0, "whole F32 words")
    result = array.array("I"); result.frombytes(data)
    need(result.itemsize == 4, "32-bit word reader")
    if sys.byteorder != "little": result.byteswap()
    return result


def difference(expected, actual):
    need(len(expected) == len(actual), "full equality extent")
    if expected == actual: return 0, None
    a, b = words(expected), words(actual)
    bad, first = 0, None
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            bad += 1
            if first is None: first = i
    return bad, first


def finite_bf16(data):
    return all((x & 0xffff) == 0 and (x & 0x7f800000) != 0x7f800000 for x in words(data))


def role_result(case, provider, role, body, guards, before, gold, actual_kv):
    i, layer, pos = case
    expected = before if provider == 2 and role == "dummy_q" else gold
    bad, first = difference(expected, body)
    intact = all(g == b"\xa5" * 64 for g in guards)
    fmt = role == "dummy_q" or finite_bf16(body)
    untouched, slot = 0, True
    if role == "window":
        start = (pos % 128) * 2048
        untouched = difference(before[:start] + before[start+2048:], body[:start] + body[start+2048:])[0]
        slot = body[start:start+2048] == actual_kv
    record = dict(kind="parity", stage="phase1", case=i, layer=layer, position=pos, compressed=layer >= 2,
        provider=provider, role=role, bytes=SIZES[role], mismatches=bad, first_mismatch=first,
        guards=intact, finite_bf16_required=role != "dummy_q", finite_bf16=fmt,
        untouched_window_mismatches=untouched, current_window_slot_equals_complete_kv=slot,
        expectation="complete-before-bytes-dead-dummy-remains-unmodified" if provider == 2 and role == "dummy_q" else "actual-golden-output-complete-bytes",
        sha256=digest(body))
    return record, not bad and intact and fmt and untouched == 0 and slot


def check_cases(records, host):
    need(len(records) >= 115, "all114 cases and host contract")
    for i, (layer, pos) in enumerate(CASES):
        row = records[i]
        need(set(row) == {"kind", "case", "layer", "position", "compressed", "args_sha256", "args_bytes"} and
             row["kind"] == "case" and uint(row["case"]) and uint(row["layer"]) and uint(row["position"]) and
             row["case"] == i and row["layer"] == layer and row["position"] == pos and
             row["compressed"] is (layer >= 2) and row["args_bytes"] == 136, "complete original case order/shape")
    need(same(records[114], dict(kind="host_contract", case_count=114, eligible_layers=LAYERS, native_groups=80,
         candidate_groups=80, args_bytes=136, no_gpu_device_created=host)), "original host geometry contract")


def check_stream(records, host=False):
    check_cases(records, host)
    if host:
        need(len(records) == 115, "host preparation must have zero GPU/PSO/completion records")
        return [], []
    need(len(records) == 1488, "exact114case+host+2library+3pipeline+342completion+1026parity records; no timing")
    need([r["kind"] for r in records[115:120]] == ["loaded_library"] * 2 + ["pipeline"] * 3, "explicit library/PSO creation order")
    completions, parity = [], []
    for ordinal in range(1, 343):
        case, provider = (ordinal - 1) // 3, (ordinal - 1) % 3
        group = records[120 + (ordinal - 1) * 4:124 + (ordinal - 1) * 4]
        event = group[0]
        need(set(event) == {"kind", "stage", "case", "provider", "ordinal", "completed", "status", "error"} and
             event["kind"] == "gpu_completion" and event["stage"] == "phase1" and uint(event["case"]) and event["case"] == case and
             uint(event["provider"]) and event["provider"] == provider and uint(event["ordinal"]) and event["ordinal"] == ordinal and uint(event["status"]) and event["status"] in (4, 5) and
             type(event["completed"]) is bool and event["completed"] is (event["status"] == 4) and
             (event["error"] is None or isinstance(event["error"], str)) and
             (not event["completed"] or event["error"] is None), "every original ordinal/CB status/error, no replacement")
        need([row.get("kind") for row in group[1:]] == ["parity"] * 3 and
             [(r["case"], r["provider"], r["role"]) for r in group[1:]] == [(case, provider, role) for role in ROLES],
             "three complete role records immediately after every completion")
        completions.append(event); parity.extend(group[1:])
    return completions, parity


def closing(data, opening):
    depth, i, quote = 1, opening + 1, None
    while i < len(data):
        c = data[i:i+1]
        if quote:
            if c == b"\\": i += 2; continue
            if c == quote: quote = None
        elif data[i:i+2] == b"//":
            end = data.find(b"\n", i); i = len(data) if end < 0 else end; continue
        elif data[i:i+2] == b"/*":
            end = data.find(b"*/", i + 2); need(end >= 0, "terminated C comment"); i = end + 2; continue
        elif c in (b'"', b"'"): quote = c
        elif c == b"{": depth += 1
        elif c == b"}":
            depth -= 1
            if depth == 0: return i + 1
        i += 1
    raise EvidenceError("complete C/Metal function body")


def kernel(data, name):
    anchor = ("kernel void " + name + "(").encode()
    need(data.count(anchor) == 1, "one explicit native/candidate definition")
    start = data.index(anchor); op = data.index(b"{", start)
    return data[start:closing(data, op)]


def literal_constants(data):
    values = {}
    for node in ast.parse(data.decode()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("PREAMBLE", "SHADER_FILES", "COPY_KERNELS"):
                    need(target.id not in values, "unique production assembly constant")
                    values[target.id] = ast.literal_eval(node.value)
    need(set(values) == {"PREAMBLE", "SHADER_FILES", "COPY_KERNELS"}, "literal production assembly constants; no builder execution")
    return values


class Review:
    def __init__(self, args):
        self.a = args; self.maps = []
        for pair in args.path_map:
            old, new = pair.split("=", 1); self.maps.append((Path(old), Path(new)))
        self.maps.sort(key=lambda x: len(str(x[0])), reverse=True)
        self.report = dict(schema_version=1, reader_sha256=sha(__file__), review_completed=False, correctness_passed=False,
            GPU_correctness_evaluated=False, timing_evaluated=False, performance_acceptance=False, production_adoption=False,
            fullmodel_evidence=False, scope="one synthetic device complete-output correctness cohort; no speed or runtime adoption")

    def mapped(self, value):
        p = Path(value)
        for old, new in self.maps:
            if p == old or p.is_relative_to(old): return new / p.relative_to(old)
        return p

    def bound_path(self, actual, expected, expected_sha):
        def norm(s): return "/tmp/" + s[len("/private/tmp/"):] if s.startswith("/private/tmp/") else s
        need(isinstance(actual, str) and Path(actual).is_absolute() and norm(actual) == norm(str(expected)), "prebound path; only known macOS tmp alias allowed")
        a, b = self.mapped(actual), self.mapped(expected)
        need(a.resolve() == b.resolve() and a.is_file() and sha(a) == expected_sha, "loaded retained artifact path and complete SHA")

    def build(self):
        a = self.a; need(sha(a.protocol) == PROTOCOL_SHA, "frozen correctness-only protocol")
        self.p = read_json(a.protocol); p = self.p
        need(p["kind"] == "kv_only_finish_frozen_component_correctness_protocol" and p["cases"] == 114 and
             p["expected_phase1_command_buffers"] == 342 and p["timing_requested_by_this_protocol"] is False, "finite original correctness-only cohort")
        bpath = a.build_dir / "source-identity.json"; need(sha(bpath) == p["build_manifest_sha256"] == BUILD_SHA, "complete frozen build manifest")
        self.b = b = read_json(bpath)
        need(b["compile_only_pass"] is True and b["runtime_modified"] is False and b["production_registered"] is False and
             b["gpu_work_submitted"] is False and b["abi"] == ABI and b["eligible_decode_non_index_source_layers"] == LAYERS,
             "private build-only original ABI and32 layers")
        for name, item in b["sources"].items():
            rel = Path(name); need(not rel.is_absolute() and ".." not in rel.parts, "local frozen source")
            path = a.build_dir / "source" / rel
            need(path.is_file() and not path.is_symlink() and path.stat().st_size == item["bytes"] and sha(path) == item["sha256"], "complete frozen source: " + name)
        constants = literal_constants((a.build_dir / "source/tools/build_metallib.py").read_bytes())
        order = constants["SHADER_FILES"]
        need(order == b["original_shader_order"] and len(order) == len(set(order)) == 29, "all29 original shader order")
        native = constants["PREAMBLE"].encode() + b"\n"
        for name in order:
            data = (a.build_dir / "source/metal" / name).read_bytes(); native += f'\n#line 1 "{name}"\n'.encode()
            item = b["sources"]["metal/" + name]; need(item["source_begin"] == len(native), "source embedding begin")
            native += data; need(item["source_end"] == len(native), "source embedding end")
        native += constants["COPY_KERNELS"].encode() + b"\n"
        need(native == (a.build_dir / "native.metal").read_bytes() and len(native) == b["native_source_bytes"] and
             digest(native) == b["native_source_sha256"] == p["native_source_prefix_sha256"], "complete original production TU prefix independently assembled")
        original = kernel((a.build_dir / "source/metal/dsv41.metal").read_bytes(), NATIVE)
        start = original.index(b"    if(group<64u) {"); op = original.index(b"{", start); end = closing(original, op)
        need(original[end:end+1] == b"\n", "full dummy Q branch line")
        branch = original[start:end+1]
        candidate = original.replace(("kernel void " + NATIVE + "(").encode(), ("kernel void " + CANDIDATE + "(").encode(), 1).replace(branch, b"    if(group<64u) return;\n", 1) + b"\n"
        need(candidate == (a.build_dir / "probe.metal").read_bytes() and digest(candidate) == b["candidate_fragment_sha256"], "only kernel rename and dead-Q earlyreturn")
        marker = b'\n#line 1 "private_kv_only_finish_probe.metal"\n'
        combined = native + marker + candidate
        need(combined == (a.build_dir / "all.metal").read_bytes() and digest(combined) == b["combined_source_sha256"] and
             b["candidate_append_begin"] == len(native) + len(marker), "complete TU native prefix plus candidate-only append")
        proof = b["proof"]; kv = original[original.index(b"    const uint column="):original.rfind(b"}")]
        need(digest(original) == proof["original_kernel_sha256"] and digest(kv) == proof["kv_branch_sha256"] == proof["candidate_kv_branch_sha256"] and
             proof["deleted_dummy_q_branch"].encode() == branch and proof["exact_edits"] == [dict(before="kernel void " + NATIVE + "(", after="kernel void " + CANDIDATE + "("), dict(before=branch.decode(), after="    if(group<64u) return;\n")], "complete independent inverse/KV source proof")
        gpu = (a.build_dir / "source/src/myds_v41_gpu.c").read_bytes(); begin = gpu.index(b"static float rope_frequencies[2][32];"); end = gpu.index(b"#pragma float_control(pop)", begin) + len(b"#pragma float_control(pop)")
        host = gpu[begin:end] + b"\n"
        need(host == (a.build_dir / "frozen_rope_init.inc").read_bytes() and digest(host) == b["host_rope_init_sha256"], "original precise host frequency initializer")
        for name, expected in [("bench_kv_only_finish_probe", p["binary_sha256"]), ("probe.metallib", p["probe_metallib_sha256"]), ("native.metallib", b["native_metallib_sha256"])]:
            need(sha(a.build_dir / name) == expected, "frozen CPU binary/library: " + name)
        need(sha(self.mapped(p["golden_metallib_path"])) == p["golden_metallib_sha256"], "actual deployed gold library identity")
        self.report.update(protocol_sha256=sha(a.protocol), build_manifest_sha256=sha(bpath), complete_TU_source_verified=True,
            original_KV_branch_bytes_verified=True, source_files_verified=len(b["sources"]), original_shaders_verified=29)

    def host_reference(self):
        a = self.a; need(sha(a.host_manifest) == HOST_SHA and sha(a.host_stdout) == HOST_STDOUT_SHA, "frozen before-input provenance manifest/stdout")
        host = read_json(a.host_manifest); need(host["host_only"] is True and host["gpu_command_buffers_submitted"] == 0 and host["parity_records"] == host["timing_records"] == host["input_records"] == [], "host preparation is never GPU evidence")
        self.reference = indexed_files(host, before_sizes())
        records = [loads(line) for line in a.host_stdout.read_text().splitlines()]; check_stream(records, True)
        self.host_records = records
        for name, item in self.reference.items(): raw(a.host_manifest.parent, item)
        need(raw(a.host_manifest.parent, self.reference["build-identity.json"]) == (a.build_dir / "source-identity.json").read_bytes(), "host before-build identity bytes")
        self.report.update(host_before_manifest_sha256=HOST_SHA, host_before_stdout_sha256=HOST_STDOUT_SHA, preparation_input_files_verified=457)

    def run(self):
        self.build(); self.host_reference()
        a, p = self.a, self.p
        if a.check_host_preparation:
            self.report.update(review_completed=True, preparation_valid=True, outcome="host_preparation_only_no_GPU_correctness_evaluated")
            return self.report
        manifest = read_json(a.raw_dir / "artifact-manifest.json")
        need(manifest["kind"] == "kv_only_finish_synthetic_component" and manifest["schema_version"] == 1 and manifest["synthetic"] is True and
             manifest["host_only"] is False and manifest["timing_requested"] is False and manifest["timing_records"] == manifest["input_records"] == [] and
             manifest["runtime_modified"] is False and manifest["production_registered"] is False and manifest["full_model_evidence"] is False and
             manifest["case_count"] == 114 and manifest["eligible_layers"] == LAYERS and manifest["abi"] == ABI and
             manifest["symbols"] == [NATIVE, CANDIDATE] and manifest["providers"] == [
                 "actual-golden-original-qkv-finish-dummy-Q", "full-TU-native-original-qkv-finish-dummy-Q", "full-TU-kv-only-finish-unused-Q-removed"] and
             manifest["gpu_command_buffers_submitted"] == 342, "complete synthetic342CB no-timing manifest")
        need(isinstance(manifest["device_name"], str) and manifest["device_name"] and uint(manifest["device_registry_id"]), "one actual device identity")
        if a.expected_device: need(manifest["device_name"] == a.expected_device, "separate expected-device judgment")
        self.bound_path(manifest["binary_path"], Path(p["build_manifest_path"]).parent / "bench_kv_only_finish_probe", p["binary_sha256"])
        self.bound_path(manifest["probe_metallib_path"], Path(p["build_manifest_path"]).parent / "probe.metallib", p["probe_metallib_sha256"])
        self.bound_path(manifest["golden_metallib_path"], p["golden_metallib_path"], p["golden_metallib_sha256"])
        self.bound_path(manifest["build_manifest_path"], p["build_manifest_path"], BUILD_SHA)
        need(manifest["binary_sha256"] == p["binary_sha256"] and manifest["probe_metallib_sha256"] == p["probe_metallib_sha256"] and
             manifest["golden_metallib_sha256"] == p["golden_metallib_sha256"] and manifest["build_manifest_sha256"] == BUILD_SHA, "runtime identities match frozen protocol")
        expected = dict(before_sizes(), **after_sizes()); indexed = indexed_files(manifest, expected)
        actual_names = {path.name for path in a.raw_dir.iterdir()}
        allowed = set(indexed) | {"artifact-manifest.json"}
        for path in (a.stdout, a.stderr):
            if path is not None and path.parent.resolve() == a.raw_dir.resolve(): allowed.add(path.name)
        need(actual_names == allowed, "one complete raw output directory without extra/replaced cases")
        records = [loads(line) for line in a.stdout.read_text().splitlines()]
        completions, parity = check_stream(records)
        need(same(parity, manifest["parity_records"]), "complete identical manifest/stdout1026 parity records")
        for i, (layer, pos) in enumerate(CASES):
            need(records[i] == self.host_records[i], "same114 before-case provenance")
            for name in [f"case{i:03d}.args136.raw"] + [f"case{i:03d}.{r}.before.raw" for r in ROLES]:
                need(indexed[name] == self.reference[name], "before raw descriptor matches frozen host input")
            args = raw(a.raw_dir, indexed[f"case{i:03d}.args136.raw"])
            position, pad = struct.unpack_from("<II", args)
            need(position == pos and pad == 0 and digest(args) == records[i]["args_sha256"], "args136 position/pad/frequency bytes")
        need(raw(a.raw_dir, indexed["build-identity.json"]) == (a.build_dir / "source-identity.json").read_bytes(), "retained running-build provenance copy")
        for lib, expected_path, expected_sha in [(records[115], p["golden_metallib_path"], p["golden_metallib_sha256"]), (records[116], str(Path(p["build_manifest_path"]).parent / "probe.metallib"), p["probe_metallib_sha256"])]:
            need(set(lib) == {"kind", "provider_library", "path", "sha256", "direct_url_load"} and uint(lib["provider_library"]) and
                 lib["provider_library"] == (0 if lib is records[115] else 1) and lib["direct_url_load"] is True and lib["sha256"] == expected_sha, "actual direct library URL loading, no fallback")
            self.bound_path(lib["path"], expected_path, expected_sha)
        for row, symbol in zip(records[117:120], [NATIVE, NATIVE, CANDIDATE]):
            need(set(row) == {"kind", "symbol", "groups", "threads", "static_tg_bytes", "dynamic_tg_bytes", "fallback"} and
                 row["symbol"] == symbol and same(row["groups"], [80, 1, 1]) and same(row["threads"], [32, 1, 1]) and
                 type(row["dynamic_tg_bytes"]) is int and row["dynamic_tg_bytes"] == 0 and uint(row["static_tg_bytes"]) and row["fallback"] is False, "all three explicit original-grid PSOs")
        numerical, failures, body_bytes, guard_bytes = True, [], 0, 0
        for i, (layer, pos) in enumerate(CASES):
            before = {r: raw(a.raw_dir, indexed[f"case{i:03d}.{r}.before.raw"]) for r in ROLES}
            gold = None
            for provider in range(3):
                bodies, gs = {}, {}
                for role in ROLES:
                    stem = f"phase1.case{i:03d}.provider{provider}.{role}"
                    bodies[role] = raw(a.raw_dir, indexed[stem + ".raw"])
                    gs[role] = [raw(a.raw_dir, indexed[stem + "." + side + "-guard.raw"]) for side in ("prefix", "suffix")]
                    body_bytes += len(bodies[role]); guard_bytes += 128
                if provider == 0: gold = bodies
                for r, role in enumerate(ROLES):
                    computed, passed = role_result((i, layer, pos), provider, role, bodies[role], gs[role], before[role], gold[role], bodies["kv"])
                    need(same(computed, parity[(i*3+provider)*3+r]), "independently recomputed full parity declaration")
                    numerical &= passed
                    if not passed: failures.append(computed)
        completed = all(row["completed"] for row in completions)
        joint = bool(numerical and completed)
        need(manifest["all_complete_outputs_and_guards_pass"] is joint and manifest["status"] == ("synthetic_correctness_complete_not_adoption" if joint else "failed"), "manifest completion never overrides raw/CB gate")
        self.report.update(review_completed=True, GPU_correctness_evaluated=True, raw_integrity_passed=True, complete_run=True,
            correctness_passed=joint, GPU_completion_passed=completed, numerical_passed=bool(numerical), numerical_failures=failures,
            complete_command_buffers=342, complete_cases=114, complete_role_outputs=1026, complete_guard_files=2052,
            complete_before_files=457, total_raw_files=3535, full_body_bytes_compared=body_bytes, full_guard_bytes_checked=guard_bytes,
            device_name=manifest["device_name"], device_registry_id=manifest["device_registry_id"], stdout_sha256=sha(a.stdout),
            artifact_manifest_sha256=sha(a.raw_dir / "artifact-manifest.json"),
            manifest_status=manifest["status"], harness_failure=manifest.get("failure"),
            failed_completions=[row for row in completions if not row["completed"]],
            outcome="synthetic_component_correctness_passed_no_adoption" if joint else "synthetic_component_GPU_or_numerical_gate_failed_no_adoption")
        if a.stderr is not None:
            self.report.update(stderr_sha256=sha(a.stderr), stderr_bytes=a.stderr.stat().st_size)
        return self.report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "build-dir", "host-manifest", "host-stdout", "out"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--raw-dir", type=Path); p.add_argument("--stdout", type=Path); p.add_argument("--stderr", type=Path)
    p.add_argument("--expected-device"); p.add_argument("--path-map", action="append", default=[])
    p.add_argument("--check-host-preparation", action="store_true")
    args = p.parse_args(); need(not args.out.exists(), "new independent report")
    need(args.check_host_preparation or (args.raw_dir is not None and args.stdout is not None), "actual raw directory and full stdout required")
    review = Review(args)
    try:
        report = review.run(); code = 0
    except (EvidenceError, OSError, KeyError, ValueError, TypeError, IndexError, SyntaxError) as error:
        report = review.report
        report.update(review_completed=False, raw_integrity_passed=False, correctness_passed=False,
                      outcome="evidence_invalid_no_correctness_or_adoption", evidence_error=str(error)); code = 2
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report.get(key) for key in ("review_completed", "outcome", "correctness_passed", "evidence_error")}, indent=2))
    raise SystemExit(code)


if __name__ == "__main__": main()
