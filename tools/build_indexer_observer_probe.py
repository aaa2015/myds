#!/usr/bin/env python3
"""Freeze a literal replay observer with an optional TILED B sham; CPU build only.

The original complete replay file is retained byte for byte in the private
snapshot. Six unique, reversible edits add the CLI/diagnostic metadata and
change only phase2 TILED-control B's PSO when --sham-tiled-b is supplied.
This builder never executes the resulting binary or submits Metal work.
"""
import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import subprocess


ORIGINAL = "tools/bench_indexer_k_capture.m"
GENERATED = "tools/bench_indexer_observer_probe.m"
EDITS = (
    (
        "sham flag declaration",
        b"BOOL host_only=NO;unsigned samples=9;",
        b"BOOL host_only=NO,sham_tiled_b=NO;unsigned samples=9;",
    ),
    (
        "boolean CLI before valued options",
        b'        if(!strcmp(argv[i],"--host-only")){host_only=YES;continue;}\n',
        b'        if(!strcmp(argv[i],"--host-only")){host_only=YES;continue;}\n'
        b'        if(!strcmp(argv[i],"--sham-tiled-b")){sham_tiled_b=YES;continue;}\n',
    ),
    (
        "diagnostic manifest only",
        b'    manifest=[@{@"schema_version":@1,@"probe":@"indexer_k_resident_real_capture",@"fixture_kind":@"actual_fullmodel_prefill",\n',
        b'    manifest=[@{@"schema_version":@1,@"probe":@"indexer_literal_observer_probe",@"fixture_kind":@"actual_fullmodel_prefill",\n'
        b'        @"diagnostic_only":@YES,@"adoption_cohort":@NO,@"production_adoption":@NO,@"fullmodel_eligible":@NO,\n'
        b'        @"tiled_b_sham_enabled":@(sham_tiled_b),@"selected_mode":sham_tiled_b?@"sham_tiled_b":@"real_candidate",\n'
        b'        @"provider_index_semantics":@"0=captured actual; 1=complete-TU TILED; 2=complete-TU candidate; logical output slot remains provider",\n',
    ),
    (
        "only phase2 TILED-control middle PSO",
        b"                unsigned provider=providers[arm];id<MTLCommandBuffer> cb=real_index_run(queue,f,provider,psos[ci][provider]);\n",
        b"                unsigned provider=providers[arm];\n"
        b"                BOOL tiled_b_sham=sham_tiled_b&&control==1u&&arm==1u;\n"
        b"                unsigned actual_provider_index=tiled_b_sham?1u:provider;\n"
        b"                id<MTLCommandBuffer> cb=real_index_run(queue,f,provider,psos[ci][actual_provider_index]);\n",
    ),
    (
        "timing binding witnesses without observer changes",
        b'                    @"sample":@(sample),@"warmup":@(sample==0),@"gpu_start_s":@(cb.GPUStartTime),@"gpu_end_s":@(cb.GPUEndTime),\n',
        b'                    @"sample":@(sample),@"warmup":@(sample==0),@"actual_provider_index":@(actual_provider_index),@"tiled_b_sham":@(tiled_b_sham),\n'
        b'                    @"gpu_start_s":@(cb.GPUStartTime),@"gpu_end_s":@(cb.GPUEndTime),\n',
    ),
    (
        "diagnostic final status",
        b'    myds_metal_cleanup();finish(@"passed_component_and_topk_only");return 0;\n',
        b'    myds_metal_cleanup();finish(@"diagnostic_complete_not_adoption");return 0;\n',
    ),
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def inverse(generated, exact_edits):
    restored = generated
    for edit in reversed(exact_edits):
        before, after = edit["before"].encode(), edit["after"].encode()
        if restored.count(after) != 1:
            raise ValueError("unique inverse edit required: " + edit["name"])
        restored = restored.replace(after, before, 1)
    return restored


def generate(original):
    generated, exact_edits = original, []
    for name, before, after in EDITS:
        if generated.count(before) != 1:
            raise ValueError("unique frozen replay anchor required: " + name)
        generated = generated.replace(before, after, 1)
        exact_edits.append(dict(name=name, before=before.decode(), after=after.decode()))
    restored = inverse(generated, exact_edits)
    if restored != original:
        raise ValueError("inverse failed to restore complete original replay bytes")
    return generated, dict(
        exact_edits=exact_edits,
        edit_count=len(exact_edits),
        original_replay_bytes=len(original),
        original_replay_sha256=sha(original),
        generated_source_bytes=len(generated),
        generated_source_sha256=sha(generated),
        inverse_restored_sha256=sha(restored),
        inverse_restores_original_byte_for_byte=True,
    )


def verify(original, generated, proof):
    expected, expected_proof = generate(original)
    if generated != expected or proof != expected_proof or inverse(generated, proof["exact_edits"]) != original:
        raise ValueError("source or transformation proof differs from the fixed six edits")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--emit-only", action="store_true")
    args = parser.parse_args()
    root, out = args.source_root.resolve(strict=True), args.out_dir.resolve()
    if out.exists() or out.is_relative_to(root):
        raise ValueError("new private output directory outside source checkout required")
    original = (root / ORIGINAL).read_bytes()
    generated, proof = generate(original)
    out.mkdir(parents=True)
    snapshot = out / "source"
    paths = [f for directory in ("src", "include", "third_party")
             for f in (root / directory).rglob("*")
             if f.is_file() and f.suffix in (".h", ".inc", ".m", ".c")]
    paths += [root / name for name in (
        ORIGINAL, "tools/bench_attention_multihead.m", "tools/build_indexer_capture_replay.py",
        "tools/build_indexer_observer_probe.py", "tools/review_indexer_capture.py")]
    inputs = {}
    for path in sorted(set(paths)):
        data = path.read_bytes()
        rel = path.relative_to(root)
        target = snapshot / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        inputs[str(rel)] = dict(bytes=len(data), sha256=sha(data))
    if (snapshot / ORIGINAL).read_bytes() != original or (root / ORIGINAL).read_bytes() != original:
        raise ValueError("original replay changed during snapshot; refusing mixed seed")
    (snapshot / GENERATED).write_bytes(generated)
    verify((snapshot / ORIGINAL).read_bytes(), (snapshot / GENERATED).read_bytes(), proof)
    source_diff = "".join(difflib.unified_diff(
        original.decode().splitlines(keepends=True), generated.decode().splitlines(keepends=True),
        fromfile=ORIGINAL, tofile=GENERATED)).encode()
    (out / "observer-source.diff").write_bytes(source_diff)
    proof_bytes = (json.dumps(proof, indent=2) + "\n").encode()
    (out / "source-transform.json").write_bytes(proof_bytes)
    common = ["xcrun", "clang", "-O3", "-Wall", "-Wextra", "-mmacosx-version-min=26.0",
              "-I" + str(snapshot / "include"), "-I" + str(snapshot / "third_party")]
    commands, objects = [], []
    for rel in ("src/myds_v41_gpu.c", "src/myds_gguf.c", "src/myds_log.c", "src/myds_configfile.c",
                "src/myds_metal.m", GENERATED):
        target = out / (Path(rel).stem + ".o")
        flags = ["-fobjc-arc", "-Wno-deprecated-declarations"] if rel.endswith(".m") else ["-std=c11"]
        commands.append(common + flags + ["-c", str(snapshot / rel), "-o", str(target)])
        objects.append(str(target))
    binary = out / "bench_indexer_observer_probe"
    commands.append(["xcrun", "clang", "-mmacosx-version-min=26.0", *objects,
                     "-framework", "Foundation", "-framework", "Metal", "-lm", "-lpthread", "-o", str(binary)])
    env = dict(os.environ)
    env.setdefault("DEVELOPER_DIR", "/Applications/Xcode.app/Contents/Developer")
    command_bytes = (json.dumps(commands, indent=2) + "\n").encode()
    (out / "build-commands.json").write_bytes(command_bytes)
    identity = dict(
        schema_version=1, kind="indexer_literal_observer_diagnostic_build",
        inputs=inputs, original_seed=str(snapshot / ORIGINAL), generated_source=str(snapshot / GENERATED),
        original_replay_sha256=sha(original), generated_source_sha256=sha(generated),
        source_diff_sha256=sha(source_diff), transformation_proof_sha256=sha(proof_bytes),
        inverse_restored_sha256=proof["inverse_restored_sha256"],
        inverse_restores_original_byte_for_byte=True,
        commands=commands, commands_sha256=sha(command_bytes),
        compile_environment={key: env.get(key) for key in (
            "DEVELOPER_DIR", "SDKROOT", "MACOSX_DEPLOYMENT_TARGET", "CPATH", "C_INCLUDE_PATH",
            "CPLUS_INCLUDE_PATH", "OBJC_INCLUDE_PATH", "LIBRARY_PATH")},
        build_method="same six CPU compilation units/flags and link flags as frozen replay builder; generated replay unit replaces original unit",
        diagnostic_only=True, adoption_cohort=False, production_adoption=False, fullmodel_eligible=False,
        runtime_registered=False, gpu_work_submitted=False, binary_executed=False,
        modes={"real_candidate": [], "sham_tiled_b": ["--sham-tiled-b"]},
        sham_scope="only phase2 control==1 and arm==1: actual PSO index1; logical provider/output index2, ABI72/grid/smem unchanged",
        original_shared_output_observer_preserved=True,
        additional_cpu_timestamps_or_output_snapshots=False,
        numeric_gate_and_summaries_are_descriptive_only=True,
        original_replay_file_unchanged=True, compile_only_pass=False,
        compilation_skipped=args.emit_only,
    )
    manifest = out / "build-manifest.json"

    def save_manifest():
        manifest.write_text(json.dumps(identity, indent=2) + "\n")

    save_manifest()
    try:
        if not args.emit_only:
            with (out / "compile.log").open("w") as log:
                for command in commands:
                    log.write(json.dumps(command) + "\n")
                    log.flush()
                    subprocess.run(command, env=env, stdout=log, stderr=log, check=True)
            identity.update(compile_only_pass=True, binary=str(binary), binary_sha256=sha(binary.read_bytes()))
        if (root / ORIGINAL).read_bytes() != original:
            identity["original_replay_file_unchanged"] = False
            raise ValueError("original replay changed while CPU build was in progress")
    except Exception as error:
        identity.update(compile_only_pass=False, build_error=repr(error))
        raise
    finally:
        save_manifest()
    print(json.dumps(dict(manifest=str(manifest), binary=identity.get("binary"),
                         binary_sha256=identity.get("binary_sha256"), compile_only_pass=identity["compile_only_pass"],
                         compilation_skipped=args.emit_only, gpu_work_submitted=False, binary_executed=False), indent=2))


if __name__ == "__main__":
    main()
