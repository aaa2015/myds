#!/usr/bin/env python3
"""Freeze and compile a private full-TU dummy-Q-free KV finish probe; no GPU work.

The production assembler and every original shader remain a byte-exact prefix.
Only a KV-only clone of qkv_finish is appended. Its complete KV branch bytes
are unchanged; the unused Q branch becomes an early return; all 80 original groups remain.
This source contract is not device parity.
The harness is synthetic/component-only and never initializes the engine.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import types

sys.dont_write_bytecode = True
SYMBOL = "kernel_myds_probe_kv_only_finish"
NATIVE = "kernel_dsv41_qkv_finish"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def body(source, start):
    at = source.index(start)
    opening = source.index("{", at)
    depth, end = 1, opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[at:end]


def candidate_fragment(source):
    native = body(source, "kernel void kernel_dsv41_qkv_finish(")
    start = native.index("    if(group<64u) {")
    end = native.index("    const uint column=(group-64u)*32u+lane;", start)
    deleted = native[start:end]
    require(deleted.endswith("        return;\n    }\n"), "Exact native Q branch required")
    kv_body = native[end:native.rfind("}")]
    require(kv_body.count("const float re=kv[i],im=kv[i+1u];") == 1 and
            kv_body.count("value=dsv41_bf16((column&1u)?re*s+im*c:re*c-im*s);") == 1,
            "Reviewed KV arithmetic source anchor required")
    edits = [
        ("kernel void kernel_dsv41_qkv_finish(", "kernel void " + SYMBOL + "("),
        (deleted, "    if(group<64u) return;\n"),
    ]
    candidate = native
    for old, new in edits:
        require(candidate.count(old) == 1, "Unique native edit anchor: " + old[:80])
        candidate = candidate.replace(old,new,1)
    restored = candidate
    for old,new in reversed(edits):
        require(restored.count(new) == 1,"Unique reverse edit anchor")
        restored = restored.replace(new,old,1)
    require(restored == native,"Unlisted native edit")
    require(candidate[candidate.index("    const uint column="):candidate.rfind("}")] == kv_body,
            "KV branch must retain exact complete original expression bytes")
    return candidate + "\n", dict(original_kernel_sha256=sha(native.encode()),
        kv_branch_sha256=sha(kv_body.encode()), candidate_kv_branch_sha256=sha(kv_body.encode()),
        complete_kv_branch_bytes_unchanged=True, reverse_edits_restore_original_body_bytes=True,
        deleted_dummy_q_branch=deleted, exact_edits=[dict(before=x,after=y) for x,y in edits],
        original_buffer_order_retained=["args","dummy_q","kv","window"],
        original_kv_group_mapping="all 80 original groups and lane 0..31 unchanged; Q groups0..63 return before reads or trig",
        no_dispatch_or_barrier_removed=True, numerical_equivalence_gpu_unverified=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--harness", type=Path, default=Path(__file__).with_name("bench_kv_only_finish_probe.m"))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--metal-minver", default=None)
    p.add_argument("--metal-std", default=None)
    p.add_argument("--cpu-minver", default="15.0")
    p.add_argument("--emit-only", action="store_true")
    a = p.parse_args()
    root, out = a.source_root.resolve(strict=True), a.out_dir.resolve()
    require(not out.exists() and not out.is_relative_to(root), "New private output outside checkout required")
    build_bytes = (root / "tools/build_metallib.py").read_bytes()
    build = types.ModuleType("frozen_kv_only_finish_production_builder")
    build.__file__ = str(root / "tools/build_metallib.py")
    exec(compile(build_bytes, build.__file__, "exec"), build.__dict__)
    require(isinstance(build.SHADER_FILES, list) and build.SHADER_FILES and
            len(set(build.SHADER_FILES)) == len(build.SHADER_FILES), "Unique production shader order required")
    native = bytearray(build.PREAMBLE.encode() + b"\n")
    sources, shader_data = {}, {}
    for name in build.SHADER_FILES:
        rel = Path(name)
        require(not rel.is_absolute() and ".." not in rel.parts, "Unsafe shader name")
        data = (root / "metal" / rel).read_bytes()
        require(b"\r" not in data, "Raw byte embedding requires LF input: " + name)
        shader_data[name] = data
        native.extend(f'\n#line 1 "{name}"\n'.encode())
        start = len(native)
        native.extend(data)
        sources["metal/" + name] = dict(bytes=len(data), sha256=sha(data), source_begin=start, source_end=len(native))
    native.extend(build.COPY_KERNELS.encode() + b"\n")
    native = bytes(native)
    generated, proof = candidate_fragment(shader_data["dsv41.metal"].decode())
    marker = b'\n#line 1 "private_kv_only_finish_probe.metal"\n'
    combined = native + marker + generated.encode()
    require(combined[:len(native)] == native, "Complete native prefix changed")
    require(SYMBOL.encode() not in native and SYMBOL.encode() in generated.encode(), "Probe must be private and new")
    gpu_bytes = (root / "src/myds_v41_gpu.c").read_bytes()
    gpu = gpu_bytes.decode()
    start, end_marker = "static float rope_frequencies[2][32];", "#pragma float_control(pop)"
    begin = gpu.index(start)
    end = gpu.index(end_marker, begin) + len(end_marker)
    host_rope = gpu[begin:end].encode() + b"\n"
    require(host_rope.count(b"static void init_rope(void)") == 1 and
            b"rope_frequencies[kind][i]=f;" in host_rope, "Original host frequency initializer required")
    minver = a.metal_minver if a.metal_minver is not None else os.environ.get("MYDS_METAL_MINVER", "15.0")
    std = a.metal_std if a.metal_std is not None else os.environ.get("MYDS_METAL_STD", "")
    require(re.fullmatch(r"[0-9]+(?:\.[0-9]+){0,2}", minver) and
            re.fullmatch(r"[0-9]+(?:\.[0-9]+){0,2}", a.cpu_minver), "Numeric deployment targets required")
    require(not std or re.fullmatch(r"[A-Za-z0-9_.-]+", std), "Invalid Metal standard")
    extra = os.environ.get("MYDS_METALLIB_DEFINES", "").split()
    require(all(re.fullmatch(r"-D[A-Za-z_][A-Za-z0-9_]*(?:=[A-Za-z0-9_.+-]+)?", x) for x in extra),
            "Only simple -D definitions accepted")
    options = ["-mmacosx-version-min=" + minver] + (["-std=" + std] if std else [])
    options += ["-O3", "-ffast-math", "-DDS4_METAL_HC_STABLE=1", "-DDS4_METAL_NORM_RSQRT_DISABLE=1"] + extra
    out.mkdir(parents=True)
    (out / "native.metal").write_bytes(native)
    (out / "all.metal").write_bytes(combined)
    (out / "probe.metal").write_text(generated)
    (out / "frozen_rope_init.inc").write_bytes(host_rope)
    frozen = dict(sources)
    for rel, data in {"tools/build_metallib.py": build_bytes, "src/myds_v41_gpu.c": gpu_bytes,
                      "tools/bench_kv_only_finish_probe.m": a.harness.resolve(strict=True).read_bytes(),
                      "tools/build_kv_only_finish_probe.py": Path(__file__).read_bytes(),
                      "src/myds_decode_candidates.inc": (root / "src/myds_decode_candidates.inc").read_bytes(),
                      "src/myds_v41_moe.c": (root / "src/myds_v41_moe.c").read_bytes(),
                      "src/myds_v41.c": (root / "src/myds_v41.c").read_bytes()}.items():
        frozen[rel] = dict(bytes=len(data), sha256=sha(data))
        path = out / "source" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    for name, data in shader_data.items():
        path = out / "source/metal" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    eligible = [i for i in range(40) if i not in {2, 8, 14, 20, 24, 28, 32, 36}]
    require(len(eligible) == 32, "Exactly 32 non-index-source layers")
    commands = []
    for stem in ("native", "probe"):
        source = "native.metal" if stem == "native" else "all.metal"
        commands += [["xcrun", "-sdk", "macosx", "metal", *options, "-c", str(out / source), "-o", str(out / (stem + ".air"))],
                     ["xcrun", "-sdk", "macosx", "metallib", str(out / (stem + ".air")), "-o", str(out / (stem + ".metallib"))]]
    binary = out / "bench_kv_only_finish_probe"
    commands.append(["xcrun", "clang", "-O3", "-Wall", "-Wextra", "-Werror", "-fobjc-arc",
                     "-Wno-deprecated-declarations", "-mmacosx-version-min=" + a.cpu_minver,
                     "-I" + str(out), str(out / "source/tools/bench_kv_only_finish_probe.m"),
                     "-framework", "Foundation", "-framework", "Metal", "-lm", "-lpthread", "-o", str(binary)])
    identity = dict(schema_version=1, kind="kv_only_finish_private_full_tu_build", source_root=str(root),
                    sources=frozen, original_shader_order=build.SHADER_FILES, proof=proof,
                    complete_native_prefix_byte_exact=True, native_oracle="complete production TU, no extracted native template",
                    native_source_bytes=len(native), native_source_sha256=sha(native),
                    combined_source_sha256=sha(combined), candidate_append_begin=len(native) + len(marker),
                    candidate_fragment_sha256=sha(generated.encode()), host_rope_init_sha256=sha(host_rope),
                    native_symbol=NATIVE,  candidate_symbol=SYMBOL,
                    eligible_decode_non_index_source_layers=eligible,
                    abi=dict(args_bytes=136, frequencies_offset=8, buffer_slots={"args":0,"dummy_q":1,"kv":2,"window":3},
                             original_groups=[80,1,1], candidate_groups=[80,1,1], threads=[32,1,1],
                             dynamic_tg_bytes=0, dummy_q_bytes=131072, kv_bytes=2048, window_bytes=262144),
                    metal_compile_options=options, cpu_minver=a.cpu_minver, commands=commands,
                    build_parameters=dict(metal_minver=minver, metal_std=std, extra_defines=extra),
                    production_registered=False, runtime_modified=False, gpu_work_submitted=False,
                    compile_only_pass=False, compilation_skipped=a.emit_only,
                    limitations=["Synthetic complete-output component gate only; no real model or speed proof.",
                                 "Deleting dummy Q work does not remove a dispatch or prove critical-path savings.",
                                 "Fast-math/source equivalence is not 0-ULP evidence; golden complete output is mandatory.",
                                 "All original 32-layer runtime joins/prestage/actual-Q RoPE remain intact in any future implementation."])
    try:
        identity["git_head_optional"] = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        identity["git_head_optional"] = None
    env = dict(os.environ)
    if "DEVELOPER_DIR" not in env:
        for path in ("/Applications/Xcode.app/Contents/Developer", "/Applications/Xcode-beta.app/Contents/Developer"):
            if Path(path).exists():
                env["DEVELOPER_DIR"] = path
                break
    identity["compile_environment"] = {k: env.get(k) for k in ("DEVELOPER_DIR", "SDKROOT", "MACOSX_DEPLOYMENT_TARGET", "TOOLCHAINS")}
    manifest = out / "source-identity.json"
    manifest.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
    if not a.emit_only:
        with (out / "compile.log").open("w") as log:
            try:
                for command in commands:
                    log.write(json.dumps(command) + "\n"); log.flush()
                    subprocess.run(command, env=env, stdout=log, stderr=log, check=True)
            except subprocess.CalledProcessError as error:
                identity["compile_failure"] = dict(command=error.cmd, returncode=error.returncode)
                manifest.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
                raise
        identity.update(compile_only_pass=True, binary_sha256=sha(binary.read_bytes()),
                        metallib_sha256=sha((out / "probe.metallib").read_bytes()),
                        native_metallib_sha256=sha((out / "native.metallib").read_bytes()))
        manifest.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
    print(json.dumps(dict(manifest=str(manifest), compile_only_pass=identity["compile_only_pass"],
                         production_registered=False, gpu_work_submitted=False)))


if __name__ == "__main__":
    main()
