#!/usr/bin/env python3
"""Build an isolated tiled-indexer K-residency probe, without runtime changes.

The production tiled-storage template is extracted verbatim as the same-build
oracle. The candidate loads its sixteen 8x8 F32 K matrices once before the
32-head loop, retaining the per-head/per-depth MMA and head-sum order. The
experiment exchanges threadgroup loads for about 32 additional F32 register
values per lane. Spills/occupancy and performance require device measurement.
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


def sha(data):
    return hashlib.sha256(data).hexdigest()


def extract(source, start, brace=True):
    at = source.index(start)
    opening = source.index("{", at)
    depth, end = 1, opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[at:end + (not brace)]


def generate(source):
    args = extract(source, "struct ds4_metal_args_dsv4_indexer_scores_fused", False)
    native = extract(source, "template<bool BF16, bool WARP_ONLY = false>\nkernel void kernel_dsv4_indexer_scores_tiled_storage")
    barrier = '''    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint head = 0; head < args.n_head; head++) {'''
    hoist = '''    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 resident_k[D/TS];
    FOR_UNROLL (uint db = 0; db < D/TS; db++) {
        simdgroup_load(resident_k[db], ktg + ((uint)sg * TS) * D + db*TS, D, 0, true);
    }

    for (uint head = 0; head < args.n_head; head++) {'''
    before = '''        for (uint db = 0; db < D/TS; db++) {
            simdgroup_float8x8 mq;
            simdgroup_float8x8 mk;
            simdgroup_load(mq, qtg + db*TS, D, 0, false);
            simdgroup_load(mk, ktg + ((uint)sg * TS) * D + db*TS, D, 0, true);
            simdgroup_multiply_accumulate(mdot, mq, mk, mdot);
        }'''
    after = '''        FOR_UNROLL (uint db = 0; db < D/TS; db++) {
            simdgroup_float8x8 mq;
            simdgroup_load(mq, qtg + db*TS, D, 0, false);
            simdgroup_multiply_accumulate(mdot, mq, resident_k[db], mdot);
        }'''
    edits = [
        ("kernel void kernel_dsv4_indexer_scores_tiled_storage(",
         "kernel void probe_indexer_scores_k_resident("),
        (barrier, hoist), (before, after)]
    candidate = native
    for old, new in edits:
        if candidate.count(old) != 1:
            raise ValueError("Unique native anchor required: " + repr(old[:90]))
        candidate = candidate.replace(old, new, 1)
    restored = candidate
    for old, new in reversed(edits):
        if restored.count(new) != 1:
            raise ValueError("Unique reverse anchor required")
        restored = restored.replace(new, old, 1)
    if restored != native:
        raise ValueError("Unlisted native code modification")
    wrappers = '''
typedef decltype(kernel_dsv4_indexer_scores_tiled_storage<false>) original_indexer_t;
template [[host_name("kernel_dsv4_indexer_scores_tiled_f32")]]
kernel original_indexer_t kernel_dsv4_indexer_scores_tiled_storage<false>;
template [[host_name("kernel_dsv4_indexer_scores_tiled_bf16")]]
kernel original_indexer_t kernel_dsv4_indexer_scores_tiled_storage<true>;
typedef decltype(probe_indexer_scores_k_resident<false>) resident_indexer_t;
template [[host_name("kernel_myds_indexer_scores_k_resident_f32")]]
kernel resident_indexer_t probe_indexer_scores_k_resident<false>;
template [[host_name("kernel_myds_indexer_scores_k_resident_bf16")]]
kernel resident_indexer_t probe_indexer_scores_k_resident<true>;
'''
    combined = args + "\n\n" + native + "\n\n" + candidate + wrappers
    return combined, dict(native_template_sha256=sha(native.encode()),
                          reverse_edits_restore_native_byte_for_byte=True,
                          exact_edits=[dict(before=a, after=b) for a, b in edits])


def candidate_fragment(source):
    """Append only a new template and new symbols to a complete native module."""
    small, proof = generate(source)
    candidate = extract(small, "template<bool BF16, bool WARP_ONLY = false>\nkernel void probe_indexer_scores_k_resident")
    wrappers = '''
typedef decltype(probe_indexer_scores_k_resident<false>) resident_indexer_t;
template [[host_name("kernel_myds_indexer_scores_k_resident_f32")]]
kernel resident_indexer_t probe_indexer_scores_k_resident<false>;
template [[host_name("kernel_myds_indexer_scores_k_resident_bf16")]]
kernel resident_indexer_t probe_indexer_scores_k_resident<true>;
'''
    fragment = candidate + wrappers
    if "struct ds4_metal_args_dsv4_indexer_scores_fused {" in fragment or \
       "kernel void kernel_dsv4_indexer_scores_tiled_storage(" in fragment or \
       'host_name("kernel_dsv4_indexer_scores_tiled_' in fragment:
        raise ValueError("Full-library append must not redeclare native definitions or symbols")
    return fragment, proof


def load_frozen_build(root):
    """Execute exactly the bytes whose SHA is recorded, without calling main."""
    path = root / "tools/build_metallib.py"
    source = path.read_bytes()
    module = types.ModuleType("frozen_myds_metallib_build")
    module.__file__ = str(path)
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module, source


def assemble_full_native(root, build):
    """Mirror build_metallib.py byte-for-byte, preserving its shader order."""
    if not isinstance(build.SHADER_FILES, list) or not build.SHADER_FILES or \
       len(set(build.SHADER_FILES)) != len(build.SHADER_FILES):
        raise ValueError("A nonempty, unique SHADER_FILES list is required")
    native = bytearray(build.PREAMBLE.encode() + b"\n")
    records, payloads = [], {}
    for index, name in enumerate(build.SHADER_FILES):
        rel = Path(name)
        if not isinstance(name, str) or rel.is_absolute() or ".." in rel.parts:
            raise ValueError("Unsafe shader-list member")
        payload = (root / "metal" / rel).read_bytes()
        # The production assembler uses universal-newline text input. Refuse
        # a normalization rather than claiming altered raw bytes are embedded.
        text = payload.decode("utf-8")
        if "\r" in text:
            raise ValueError("Strict raw embedding requires LF shader source: " + name)
        payloads[name] = payload
        native.extend(f'\n#line 1 "{name}"\n'.encode())
        begin = len(native)
        native.extend(payload)
        records.append(dict(index=index, path="metal/" + name, bytes=len(payload),
                            sha256=sha(payload), source_begin=begin, source_end=len(native)))
    native.extend(build.COPY_KERNELS.encode() + b"\n")
    native = bytes(native)
    for record in records:
        payload = payloads[record["path"][len("metal/"):]]
        if native[record["source_begin"]:record["source_end"]] != payload:
            raise ValueError("Original shader was altered during native assembly")
    return native, records, payloads


def metal_options(args, full_library):
    """Full mode follows the production builder's three environment inputs."""
    minver = args.metal_minver
    if minver is None:
        minver = os.environ.get("MYDS_METAL_MINVER", "15.0") if full_library else "15.0"
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+){0,2}", minver):
        raise ValueError("Numeric Metal deployment target required")
    std = args.metal_std
    if std is None:
        std = os.environ.get("MYDS_METAL_STD", "") if full_library else ""
    if std and not re.fullmatch(r"[A-Za-z0-9_.-]+", std):
        raise ValueError("Invalid Metal language standard")
    extra = os.environ.get("MYDS_METALLIB_DEFINES", "").split() if full_library else []
    options = ["-mmacosx-version-min=" + minver]
    if std:
        options.append("-std=" + std)
    options += ["-O3", "-ffast-math", "-DDS4_METAL_HC_STABLE=1", "-DDS4_METAL_NORM_RSQRT_DISABLE=1"]
    options += extra
    return options, dict(metal_minver=minver, metal_std=std, extra_flags=extra,
                         environment={name: os.environ.get(name) for name in
                                      ("MYDS_METAL_MINVER", "MYDS_METAL_STD", "MYDS_METALLIB_DEFINES")})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--metal-minver", default=None,
                   help="Metal deployment target; default matches build_metallib.py. CPU harness remains 26.0.")
    p.add_argument("--metal-std", default=None,
                   help="Explicit Metal language version; full-library mode otherwise uses MYDS_METAL_STD.")
    p.add_argument("--full-library", action="store_true",
                   help="Assemble the complete native library unchanged and append only the candidate.")
    p.add_argument("--emit-only", action="store_true",
                   help="Freeze sources, commands and CPU proofs; invoke no compiler or GPU.")
    a = p.parse_args()
    root, out = a.source_root.resolve(strict=True), a.out_dir.resolve()
    if out.exists() or out.is_relative_to(root):
        raise ValueError("New private output directory outside checkout required")
    build, build_source = load_frozen_build(root)
    native, shader_records, shader_payloads = None, [], {}
    if a.full_library:
        native, shader_records, shader_payloads = assemble_full_native(root, build)
        data = shader_payloads["dsv4_misc.metal"]
        generated, proof = candidate_fragment(data.decode())
        append_marker = b'\n#line 1 "myds_indexer_k_resident_probe.metal"\n'
        combined = native + append_marker + generated.encode()
        if combined[:len(native)] != native:
            raise ValueError("Native complete source changed while appending candidate")
    else:
        data = (root / "metal/dsv4_misc.metal").read_bytes()
        generated, proof = generate(data.decode())
        combined = build.PREAMBLE.encode() + b"\n" + generated.encode()
    compile_options, parameter_identity = metal_options(a, a.full_library)
    out.mkdir(parents=True)
    (out / "all.metal").write_bytes(combined)
    (out / "probe.metal").write_text(generated)
    if a.full_library:
        (out / "native.metal").write_bytes(native)
        for name, payload in shader_payloads.items():
            path = out / "native-source/metal" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        path = out / "native-source/tools/build_metallib.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(build_source)
    # Snapshot the small harness dependencies so concurrent checkout work cannot
    # silently change a build after its source identity has been recorded.
    inputs = {}
    for rel in ("tools/bench_indexer_k_resident.m", "tools/bench_attention_multihead.m",
                "src/myds_gguf.c", "include/myds_gguf.h"):
        payload = (root / rel).read_bytes()
        inputs[rel] = sha(payload)
        (out / Path(rel).name).write_bytes(payload)
    identity = dict(schema_version=1, source_root=str(root), production_registered=False,
                    gpu_work_submitted=False, dsv4_misc_sha256=sha(data),
                    builder_sha256=sha(Path(__file__).read_bytes()), probe_sha256=sha(generated.encode()),
                    proof=proof, harness_input_sha256=inputs,
                    build_metallib_script_sha256=sha(build_source),
                    preamble_sha256=sha(build.PREAMBLE.encode()),
                    combined_source_sha256=sha((out / "all.metal").read_bytes()),
                    metal_minver=parameter_identity["metal_minver"], cpu_harness_minver="26.0",
                    build_mode="full-library" if a.full_library else "small",
                    build_parameters=parameter_identity, compile_options=compile_options)
    if a.full_library:
        identity.update(original_shader_order=list(build.SHADER_FILES), original_shaders=shader_records,
                        native_source_sha256=sha(native), native_source_bytes=len(native),
                        copy_kernels_sha256=sha(build.COPY_KERNELS.encode()),
                        candidate_source_begin=len(native)+len(append_marker),
                        candidate_append_marker=append_marker.decode(),
                        complete_native_prefix_byte_exact=True,
                        original_shader_bytes_embedded_exact=True,
                        candidate_append_has_no_native_redefinitions=True,
                        native_oracle="complete native translation unit, never extracted template")
    commands = []
    if a.full_library:
        commands += [
            ["xcrun", "-sdk", "macosx", "metal"] + compile_options +
            ["-c", str(out / "native.metal"), "-o", str(out / "native.air")],
            ["xcrun", "-sdk", "macosx", "metallib", str(out / "native.air"), "-o", str(out / "native.metallib")]]
    commands += [
        ["xcrun", "-sdk", "macosx", "metal"] + compile_options +
        ["-c", str(out / "all.metal"), "-o", str(out / "probe.air")],
        ["xcrun", "-sdk", "macosx", "metallib", str(out / "probe.air"), "-o", str(out / "probe.metallib")],
        ["xcrun", "clang", "-O3", "-fobjc-arc", "-Wno-deprecated-declarations", "-mmacosx-version-min=26.0",
         "-I" + str(out), str(out / "bench_indexer_k_resident.m"), str(out / "myds_gguf.c"),
         "-framework", "Foundation", "-framework", "Metal", "-o", str(out / "bench_indexer_k_resident")]]
    env = dict(os.environ)
    if a.full_library and "DEVELOPER_DIR" not in env:
        for path in ("/Applications/Xcode.app/Contents/Developer", "/Applications/Xcode-beta.app/Contents/Developer"):
            if Path(path).exists():
                env["DEVELOPER_DIR"] = path
                break
    elif not a.full_library:
        env.setdefault("DEVELOPER_DIR", "/Applications/Xcode.app/Contents/Developer")
    identity.update(commands=commands, compilation_skipped=a.emit_only,
                    compile_environment={key: env.get(key) for key in
                                         ("DEVELOPER_DIR", "SDKROOT", "MACOSX_DEPLOYMENT_TARGET", "TOOLCHAINS")})
    (out / "source-identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    if a.emit_only:
        print(json.dumps(identity, indent=2))
        return
    with (out / "compile.log").open("w") as log:
        for command in commands:
            log.write(json.dumps(command) + "\n")
            log.flush()
            subprocess.run(command, env=env, stdout=log, stderr=log, check=True)
    identity.update(compile_only_pass=True, commands=commands,
                    metallib_sha256=sha((out / "probe.metallib").read_bytes()),
                    binary_sha256=sha((out / "bench_indexer_k_resident").read_bytes()))
    if a.full_library:
        identity["native_metallib_sha256"] = sha((out / "native.metallib").read_bytes())
    (out / "source-identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    print(json.dumps(identity, indent=2))


if __name__ == "__main__":
    main()
