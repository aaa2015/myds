#!/usr/bin/env python3
"""Build a complete-library BF16 register-only indexer probe, never GPU.

Original production PREAMBLE/SHADER_FILES/COPY_KERNELS are embedded byte-exact
in order. Only the new fragment is appended; removing the recorded suffix
restores the complete original TU. Production registration/runtime is untouched.
The source/order checks here are CPU contracts, not proof of GPU bit identity.
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
SYMBOL = "kernel_myds_indexer_scores_register_only_bf16"
NATIVE_SYMBOL = "kernel_dsv4_indexer_scores_tiled_bf16"
ARG_TYPE = "ds4_metal_args_dsv4_indexer_scores_fused"
MARKER = b'\n#line 1 "myds_indexer_register_only.metal.inc"\n'
ABI_FIELDS = [
    ("n_comp", "uint32_t", 0), ("n_tokens", "uint32_t", 4),
    ("n_head", "uint32_t", 8), ("head_dim", "uint32_t", 12),
    ("pos0", "uint32_t", 16), ("ratio", "uint32_t", 20),
    ("q_token_stride", "uint64_t", 24), ("q_head_stride", "uint64_t", 32),
    ("weights_token_stride", "uint64_t", 40), ("index_row_stride", "uint64_t", 48),
    ("score_token_stride", "uint64_t", 56), ("scale", "float", 64)]
BUILD_ENV = ("MYDS_METAL_MINVER", "MYDS_METAL_STD", "MYDS_METALLIB_DEFINES")
TOOLCHAIN_ENV = ("DEVELOPER_DIR", "SDKROOT", "MACOSX_DEPLOYMENT_TARGET", "TOOLCHAINS")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def descriptor(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path.read_bytes()))


def extract_block(source, anchor):
    require(source.count(anchor) == 1, "Unique source anchor required: " + anchor)
    start = source.index(anchor); opening = source.index("{", start)
    depth, end = 1, opening + 1
    while depth and end < len(source):
        depth += (source[end] == "{") - (source[end] == "}"); end += 1
    require(depth == 0, "Complete source block")
    return source[start:end]


def load_production_builder(root):
    path = root / "tools/build_metallib.py"; raw = path.read_bytes()
    module = types.ModuleType("frozen_production_metallib_build")
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module, raw


def assemble_native(root, build):
    require(isinstance(build.SHADER_FILES, list) and build.SHADER_FILES and
            len(set(build.SHADER_FILES)) == len(build.SHADER_FILES), "Nonempty unique production shader order")
    native = bytearray(build.PREAMBLE.encode() + b"\n"); records, payloads = [], {}
    for ordinal, name in enumerate(build.SHADER_FILES):
        require(isinstance(name, str), "String shader-list member")
        rel = Path(name)
        require(not rel.is_absolute() and ".." not in rel.parts, "Relative production shader path")
        path = root / "metal" / rel
        require(path.is_file() and not path.is_symlink(), "Regular production shader")
        raw = path.read_bytes(); raw.decode("utf-8")
        require(b"\r" not in raw, "Refuse newline normalization of raw shader bytes")
        native.extend(f'\n#line 1 "{name}"\n'.encode()); start = len(native); native.extend(raw)
        records.append(dict(index=ordinal, path="metal/" + name, bytes=len(raw), sha256=sha(raw), source_begin=start, source_end=len(native)))
        payloads["metal/" + name] = raw
    native.extend(build.COPY_KERNELS.encode() + b"\n"); native = bytes(native)
    require(all(native[r["source_begin"]:r["source_end"]] == payloads[r["path"]] for r in records), "Every original shader byte embedded unchanged")
    return native, records, payloads


def coord(lane):
    require(type(lane) is int and 0 <= lane < 32, "SIMD lane 0..31")
    qid = lane // 4
    return (qid & 2) * 2 + (lane % 2) * 2, (qid & 4) + ((lane // 2) % 4)


def validate_problem(a, buffer_bytes=None):
    """Pure CPU preflight, no dispatch. Base address alignment, owned native
    buffer objects and no-alias additionally belong to the independent harness.
    Strided spans include the final logical row and reject uint64 overflow.
    """
    uint_fields = [name for name, kind, _ in ABI_FIELDS if kind != "float"]
    require(all(type(a[name]) is int and 0 <= a[name] < (1 << (64 if "stride" in name else 32)) for name in uint_fields), "Typed bounded unsigned ABI integers")
    require(a["n_head"] == 32 and a["head_dim"] == 128 and a["ratio"] == 2 and
            512 < a["n_comp"] <= 16384 and 8 <= a["n_tokens"] <= 2048 and
            a["pos0"] + a["n_tokens"] <= 32768 and type(a["scale"]) in (int, float) and a["scale"] == 1/64,
            "BF16-only H32 D128 ratio2 narrow prefill scope")
    qs, qh, ws, ks, ss = (a[k] for k in ("q_token_stride", "q_head_stride", "weights_token_stride", "index_row_stride", "score_token_stride"))
    require(qh >= 512 and qh % 4 == 0 and qs >= 31 * qh + 512 and qs % 4 == 0 and
            ws >= 128 and ws % 4 == 0 and ks >= 256 and ks % 2 == 0 and ss >= a["n_comp"]*4 and ss % 4 == 0,
            "Nonoverlapping aligned logical Q/head/W/BF16-K/score row strides")
    n, k = a["n_tokens"], a["n_comp"]
    spans = dict(q=(n-1)*qs+31*qh+512, weights=(n-1)*ws+128,
                 cache=(k-1)*ks+256, scores=(n-1)*ss+k*4)
    require(all(0 < size < 1 << 64 for size in spans.values()), "No span arithmetic overflow")
    if buffer_bytes is not None:
        require(set(buffer_bytes) == set(spans) and all(type(buffer_bytes[key]) is int and buffer_bytes[key] >= size for key, size in spans.items()), "Full required input/output buffer spans")
    return dict(required_buffer_bytes=spans, grid=[(k+31)//32, (n+7)//8, 1], threads=[32, 4, 1], dynamic_threadgroup_bytes=0)


def validate_sources(native_misc, candidate):
    args = extract_block(native_misc, "struct " + ARG_TYPE)
    fields = re.findall(r"\b(uint32_t|uint64_t|float)\s+(\w+)\s*;", args)
    require(fields == [(kind, name) for name, kind, _ in ABI_FIELDS], "Current DSV4 ABI fields/order; old GLM is not interchangeable")
    template = extract_block(native_misc, "template<bool BF16, bool WARP_ONLY = false>\nkernel void kernel_dsv4_indexer_scores_tiled_storage")
    expr = "max(s, 0.0f) * (w[head] * args.scale)"
    require(template.count(expr) == 2 and
            candidate.count("#pragma clang fp contract(off) reassociate(off)") == 2 and
            candidate.count("const float weighted = max(s, 0.0f) * w[head];") == 2 and
            candidate.count("const float scaled = weighted * args.scale;") == 2 and
            all(candidate.count(f"acc{part} = scaled + acc{part};") == 1 for part in (0,1)),
            "V2 exact local head mul/mul/add with contraction and reassociation disabled")
    require(re.findall(r"kernel\s+void\s+(\w+)\s*\(", candidate) == [SYMBOL] and candidate.count("struct " + ARG_TYPE + " {") == 0,
            "Append only one new kernel; reuse native argument definition")
    require(SYMBOL not in native_misc and "threadgroup_barrier(" not in candidate and "simdgroup_barrier(" not in candidate and
            "[[threadgroup(0)]]" not in candidate and "simdgroup_load(" not in candidate and "simdgroup_store(" not in candidate,
            "Register-only fragment has no TG storage, barriers, loads or stores")
    for anchor in ("myds_indexer_register_only_coord", "resident_k[16]", "for (uint db = 0; db < 16u; db++)",
                   "for (uint head = 0; head < args.n_head; head++)", "args.index_row_stride", "uint(row[d]) << 16u",
                   "} else if (live) {", "dst[comp0] = -INFINITY", "dst[comp1] = -INFINITY"):
        require(anchor in candidate, "Required direct-fragment/stride/invisible-tail source contract: " + anchor)
    cells = [(coord(lane)[1], coord(lane)[0]+part) for lane in range(32) for part in (0, 1)]
    require(len(set(cells)) == 64 and set(cells) == {(r, c) for r in range(8) for c in range(8)}, "Exact 32-lane/two-element 8x8 coordinate coverage")
    return dict(abi=dict(type=ARG_TYPE, bytes=72, alignment=8, trailing_padding_bytes=4,
                        fields=[dict(name=n, type=t, offset=o) for n,t,o in ABI_FIELDS], source_sha256=sha(args.encode())),
                native_template_sha256=sha(template.encode()), local_accumulation_expression="relu(dot)*weight -> *scale -> +acc; local contract(off)/reassociate(off)", native_source_expression=expr,
                coordinate_cells=cells, coordinate_64_cells_unique=True, candidate_is_register_only_by_source=True,
                arithmetic_order_source_contract_only=True, GPU_numeric_identity_proven=False)


def compose(native, candidate):
    fragment = candidate.encode()
    require(MARKER not in native and SYMBOL.encode() not in native, "Fresh append marker/symbol with no native collision")
    combined = native + MARKER + fragment; begin = len(native) + len(MARKER)
    require(combined[:len(native)] == native and combined[len(native):] == MARKER+fragment,
            "Native full TU exact prefix and suffix-removal inverse")
    return combined, dict(kind="complete_TU_append_inverse",complete_native_prefix_byte_exact=True,
        inverse_remove_only_recorded_suffix_restores_native_byte_for_byte=True,native_bytes=len(native),native_sha256=sha(native),
        append_marker=MARKER.decode(),candidate_source_begin=begin,candidate_source_bytes=len(fragment),candidate_source_sha256=sha(fragment),
        combined_bytes=len(combined),combined_sha256=sha(combined))


def compile_options(args, environment):
    minver = args.metal_minver if args.metal_minver is not None else environment.get("MYDS_METAL_MINVER", "15.0")
    std = args.metal_std if args.metal_std is not None else environment.get("MYDS_METAL_STD", "")
    require(re.fullmatch(r"[0-9]+(?:\.[0-9]+){0,2}", minver) is not None, "Numeric Metal minimum version")
    require(not std or re.fullmatch(r"[A-Za-z0-9_.-]+", std) is not None, "Metal language standard")
    extra = environment.get("MYDS_METALLIB_DEFINES", "").split()
    options = ["-mmacosx-version-min=" + minver] + (["-std="+std] if std else [])
    options += ["-O3", "-ffast-math", "-DDS4_METAL_HC_STABLE=1", "-DDS4_METAL_NORM_RSQRT_DISABLE=1"] + extra
    return options, dict(metal_minver=minver,metal_std=std,extra_flags=extra,environment={k:environment.get(k) for k in BUILD_ENV})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--candidate-source", type=Path, default=None,
                   help="External fragment; default is next to this builder, not inside the frozen production snapshot")
    p.add_argument("--harness-source", type=Path, default=None, help="Optional standalone .m to freeze and CPU-compile, never execute")
    p.add_argument("--metal-minver", default=None)
    p.add_argument("--metal-std", default=None)
    p.add_argument("--emit-only", action="store_true", help="Freeze sources/commands/CPU proofs without invoking any compiler")
    a = p.parse_args(); root = a.source_root.resolve(strict=True); out = a.out_dir.resolve()
    require(not out.exists() and not out.is_relative_to(root), "New private output directory outside the source tree")
    original_candidate = a.candidate_source or Path(__file__).with_name("indexer_register_only.metal.inc")
    require(not original_candidate.is_symlink(), "Regular candidate fragment, not a symlink")
    candidate_path = original_candidate.resolve(strict=True)
    require(candidate_path.is_file(), "Regular candidate fragment")
    candidate_raw = candidate_path.read_bytes(); candidate = candidate_raw.decode()
    require(b"\r" not in candidate_raw, "Candidate LF bytes preserved")
    harness_raw = None
    if a.harness_source is not None:
        require(not a.harness_source.is_symlink() and a.harness_source.is_file(), "Regular external standalone harness source")
        harness_raw = a.harness_source.read_bytes()
    build, build_raw = load_production_builder(root); native, shaders, payloads = assemble_native(root, build)
    contract = validate_sources(payloads["metal/dsv4_misc.metal"].decode(), candidate)
    combined, proof = compose(native, candidate); options, parameters = compile_options(a, os.environ)
    env = dict(os.environ)
    if "DEVELOPER_DIR" not in env:
        for location in ("/Applications/Xcode.app/Contents/Developer", "/Applications/Xcode-beta.app/Contents/Developer"):
            if Path(location).is_dir(): env["DEVELOPER_DIR"] = location; break
    out.mkdir(parents=True)
    source = out/"source"
    frozen = dict(payloads, **{"tools/build_metallib.py":build_raw,"tools/indexer_register_only.metal.inc":candidate_raw,
                            "tools/build_indexer_register_only_probe.py":Path(__file__).read_bytes()})
    if harness_raw is not None:
        frozen["tools/bench_indexer_register_only.m"] = harness_raw
    for name, raw in frozen.items():
        target=source/name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(raw)
    (out/"native.metal").write_bytes(native); (out/"all.metal").write_bytes(combined); (out/"candidate.metal.inc").write_bytes(candidate_raw)
    (out/"source-transform.json").write_text(json.dumps(proof,indent=2)+"\n")
    commands=[]
    for basename in ("native","probe"):
        metal_file=out/("native.metal" if basename=="native" else "all.metal")
        commands.extend([["xcrun","-sdk","macosx","metal"]+options+["-c",str(metal_file),"-o",str(out/(basename+".air"))],
                         ["xcrun","-sdk","macosx","metallib",str(out/(basename+".air")),"-o",str(out/(basename+".metallib"))]])
    if harness_raw is not None:
        commands.append(["xcrun","clang","-O2","-fobjc-arc","-Wno-deprecated-declarations","-mmacosx-version-min=26.0",
                         "-I",str(root/"include"),"-I",str(root/"src"),"-framework","Foundation","-framework","Metal",str(source/"tools/bench_indexer_register_only.m"),str(root/"src/myds_v41_gpu.c"),str(root/"src/myds_metal.m"),str(root/"src/myds_gguf.c"),str(root/"src/myds_log.c"),"-lm","-lpthread",
                         "-o",str(out/"bench_indexer_register_only")])
    inventory={name:dict(bytes=len(raw),sha256=sha(raw)) for name,raw in frozen.items()}
    identity=dict(schema_version=1,kind="indexer_register_only_full_library_build",source_root=str(root),
        candidate_source_original=str(candidate_path),harness_source_original=str(a.harness_source.resolve()) if harness_raw is not None else None,
        selected_symbol=SYMBOL,native_oracle_symbol=NATIVE_SYMBOL,native_oracle="complete original production translation unit; external deployed library remains a separate GPU gate",
        production_registered=False,runtime_connected=False,default_enabled=False,gpu_work_submitted=False,production_adoption=False,
        numerical_passed=False,performance_evaluated=False,build_mode="full-library-only",
        compilation_skipped=a.emit_only,compile_only_pass=False,commands=commands,compile_options=options,build_parameters=parameters,
        compile_environment={key:env.get(key) for key in TOOLCHAIN_ENV},
        original_shader_order=list(build.SHADER_FILES),original_shaders=shaders,
        preamble_sha256=sha(build.PREAMBLE.encode()),copy_kernels_sha256=sha(build.COPY_KERNELS.encode()),
        native_source_bytes=len(native),native_source_sha256=sha(native),combined_source_sha256=sha(combined),
        source_inventory=inventory,source_inventory_sha256=sha(json.dumps(inventory,sort_keys=True,separators=(",",":")).encode()),
        source_transform=proof,contract=contract,
        scope=dict(cache_format="BF16",n_head=32,head_dim=128,ratio=2,n_comp_exclusive_min=512,n_comp_inclusive_max=65536,
                   n_tokens_inclusive_min=8,n_tokens_inclusive_max=2048,end_inclusive_max=131072,scale=1/64,key_window=0,
                   ordinary_nontrace_prefill_only=True,prior_K_resident_must_be_off=True),
        geometry=dict(threads=[32,4,1],tile_tokens=8,tile_keys=32,simdgroup_tokens=8,simdgroup_keys=8,
                      grid=["ceil(n_comp/32)","ceil(n_tokens/8)",1],candidate_dynamic_threadgroup_bytes=0,native_dynamic_threadgroup_bytes=21504),
        buffers={"0":"args72","1":"q_f32","2":"weights_f32","3":"cache_bf16","4":"scores_f32"},
        frozen_artifacts={name:descriptor(out/name) for name in ("native.metal","all.metal","candidate.metal.inc","source-transform.json")})
    identity_path=out/"source-identity.json"; identity_path.write_text(json.dumps(identity,indent=2)+"\n")
    manifest=dict(schema_version=1,kind=identity["kind"],selected_symbol=SYMBOL,native_oracle_symbol=NATIVE_SYMBOL,
                  source_identity=descriptor(identity_path),native_source_bytes=len(native),native_source_sha256=sha(native),
                  combined_source_sha256=sha(combined),compile_only_pass=False,compilation_skipped=a.emit_only,
                  gpu_work_submitted=False,production_adoption=False,runtime_connected=False,commands=commands)
    if not a.emit_only:
        with (out/"compile.log").open("w") as log:
            try:
                for command in commands:
                    log.write(json.dumps(command)+"\n");log.flush()
                    subprocess.run(command,env=env,stdout=log,stderr=log,check=True)
            except subprocess.CalledProcessError as error:
                manifest.update(failure=dict(kind="compiler_failure",returncode=error.returncode,command=error.cmd),compile_log=descriptor(out/"compile.log"))
                (out/"build-manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
                raise
        names=["native.air","probe.air","native.metallib","probe.metallib","compile.log"]
        if harness_raw is not None: names.append("bench_indexer_register_only")
        identity.update(compile_only_pass=True,artifacts={name:descriptor(out/name) for name in names})
        identity_path.write_text(json.dumps(identity,indent=2)+"\n")
        manifest.update(compile_only_pass=True,source_identity=descriptor(identity_path),artifacts=identity["artifacts"],
                        metallibs={key:descriptor(out/(key+".metallib")) for key in ("native","probe")})
    (out/"build-manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({key:manifest[key] for key in ("kind","selected_symbol","compile_only_pass","compilation_skipped","gpu_work_submitted","source_identity")},indent=2))


if __name__ == "__main__":
    main()
