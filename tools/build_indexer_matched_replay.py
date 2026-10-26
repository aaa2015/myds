#!/usr/bin/env python3
"""CPU-build a separately frozen matched-conditioner diagnostic/component replay.

Each sample executes B's identical PSO/output2 as conditioner, then A1/B/A2.
No binary is run by this builder. --sham-tiled-b in the resulting binary is
the independently gated S calibration; absence of that flag selects R.
Neither compilation nor successful completion grants performance adoption.
"""
import argparse
import difflib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess


ORIGINAL = "tools/bench_indexer_k_capture.m"
GENERATED = "tools/bench_indexer_matched_replay.m"
OBSERVER_BUILDER = "tools/build_indexer_observer_probe.py"
MATCHED_EDITS = (
    (
        "matched replay manifest",
        b'@"probe":@"indexer_literal_observer_probe",@"fixture_kind":@"actual_fullmodel_prefill",\n',
        b'@"probe":@"indexer_matched_conditioner_replay",@"fixture_kind":@"actual_fullmodel_prefill",\n'
        b'        @"matched_conditioner_per_sample":@YES,@"conditioner_output_slot":@2,\n'
        b'        @"timing_record_count_expected":@240,@"conditioner_record_count_expected":@60,\n'
        b'        @"warmup_arm_record_count_expected":@18,@"formal_arm_record_count_expected":@162,\n',
    ),
    (
        "four steps with conditioner bound to the B arm",
        b"            for(unsigned sample=0;sample<=samples;sample++)for(unsigned arm=0;arm<3;arm++) {\n",
        b"            for(unsigned sample=0;sample<=samples;sample++)for(unsigned step=0;step<4;step++) {\n"
        b"                const BOOL is_conditioner=step==0u;\n"
        b"                const unsigned arm=is_conditioner?1u:step-1u;\n",
    ),
    (
        "conditioner excluded from all arm timing arrays",
        b"double us=(cb.GPUEndTime-cb.GPUStartTime)*1e6;if(sample)times[arm][sample-1]=us;\n",
        b"double us=(cb.GPUEndTime-cb.GPUStartTime)*1e6;if(sample&&!is_conditioner)times[arm][sample-1]=us;\n",
    ),
    (
        "explicit conditioner arm label",
        b'@"arm":@[@"A1",@"B",@"A2"][arm],\n',
        b'@"arm":is_conditioner?@"conditioner":@[@"A1",@"B",@"A2"][arm],\n',
    ),
    (
        "step and performance-sample witnesses",
        b'@"actual_provider_index":@(actual_provider_index),@"tiled_b_sham":@(tiled_b_sham),\n',
        b'@"actual_provider_index":@(actual_provider_index),@"tiled_b_sham":@(tiled_b_sham),\n'
        b'                    @"step":@(step),@"conditioner":@(is_conditioner),@"performance_sample":@(sample>0u&&!is_conditioner),\n',
    ),
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_observer(root):
    spec = importlib.util.spec_from_file_location("frozen_indexer_observer_builder", root / OBSERVER_BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate(original, observer):
    base, base_proof = observer.generate(original)
    generated = base
    exact_edits = list(base_proof["exact_edits"])
    for name, before, after in MATCHED_EDITS:
        if generated.count(before) != 1:
            raise ValueError("unique matched replay anchor required: " + name)
        generated = generated.replace(before, after, 1)
        exact_edits.append(dict(name=name, before=before.decode(), after=after.decode()))
    restored = observer.inverse(generated, exact_edits)
    if restored != original:
        raise ValueError("matched replay inverse did not restore complete old replay")
    return generated, dict(
        exact_edits=exact_edits, edit_count=len(exact_edits),
        observer_base_edit_count=base_proof["edit_count"], matched_edit_count=len(MATCHED_EDITS),
        observer_base_generated_source_sha256=sha(base),
        original_replay_bytes=len(original), original_replay_sha256=sha(original),
        generated_source_bytes=len(generated), generated_source_sha256=sha(generated),
        inverse_restored_sha256=sha(restored), inverse_restores_original_byte_for_byte=True,
    )


def verify(original, generated, proof, observer):
    expected, expected_proof = generate(original, observer)
    if generated != expected or proof != expected_proof or observer.inverse(generated, proof["exact_edits"]) != original:
        raise ValueError("matched source or proof differs from the frozen reversible edits")


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
    out.mkdir(parents=True)
    snapshot = out / "source"
    paths = [f for directory in ("src", "include", "third_party")
             for f in (root / directory).rglob("*")
             if f.is_file() and f.suffix in (".h", ".inc", ".m", ".c")]
    paths += [root / name for name in (
        ORIGINAL, OBSERVER_BUILDER, "tools/bench_attention_multihead.m",
        "tools/build_indexer_capture_replay.py", "tools/build_indexer_matched_replay.py",
        "tools/review_indexer_capture.py", "tests/test_indexer_matched_builder.py")]
    inputs = {}
    for path in sorted(set(paths)):
        data = path.read_bytes()
        rel = path.relative_to(root)
        target = snapshot / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        inputs[str(rel)] = dict(bytes=len(data), sha256=sha(data))
    if (snapshot / ORIGINAL).read_bytes() != original or (root / ORIGINAL).read_bytes() != original:
        raise ValueError("old replay changed during snapshot; refusing mixed seed")
    # Import only the frozen private helper, so a concurrent checkout edit of
    # the observer builder cannot silently change the transformation contract.
    observer = load_observer(snapshot)
    generated, proof = generate(original, observer)
    (snapshot / GENERATED).write_bytes(generated)
    verify(original, (snapshot / GENERATED).read_bytes(), proof, observer)
    source_diff = "".join(difflib.unified_diff(
        original.decode().splitlines(keepends=True), generated.decode().splitlines(keepends=True),
        fromfile=ORIGINAL, tofile=GENERATED)).encode()
    (out / "matched-source.diff").write_bytes(source_diff)
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
    binary = out / "bench_indexer_matched_replay"
    commands.append(["xcrun", "clang", "-mmacosx-version-min=26.0", *objects,
                     "-framework", "Foundation", "-framework", "Metal", "-lm", "-lpthread", "-o", str(binary)])
    env = dict(os.environ)
    env.setdefault("DEVELOPER_DIR", "/Applications/Xcode.app/Contents/Developer")
    command_bytes = (json.dumps(commands, indent=2) + "\n").encode()
    (out / "build-commands.json").write_bytes(command_bytes)
    identity = dict(
        schema_version=1, kind="indexer_matched_conditioner_replay_build", inputs=inputs,
        original_seed=str(snapshot / ORIGINAL), generated_source=str(snapshot / GENERATED),
        original_replay_sha256=sha(original), generated_source_sha256=sha(generated),
        observer_builder_sha256=inputs[OBSERVER_BUILDER]["sha256"],
        source_diff_sha256=sha(source_diff), transformation_proof_sha256=sha(proof_bytes),
        inverse_restored_sha256=proof["inverse_restored_sha256"],
        inverse_restores_original_byte_for_byte=True, commands=commands, commands_sha256=sha(command_bytes),
        compile_environment={key: env.get(key) for key in (
            "DEVELOPER_DIR", "SDKROOT", "MACOSX_DEPLOYMENT_TARGET", "CPATH", "C_INCLUDE_PATH",
            "CPLUS_INCLUDE_PATH", "OBJC_INCLUDE_PATH", "LIBRARY_PATH")},
        build_method="same six CPU compilation units/flags and link flags as old replay; generated matched unit replaces original unit",
        old_r2_468_diagnostic_and_720_literal_results_reclassified=False,
        binary_modes={"S_calibration": ["--sham-tiled-b"], "R_component": []},
        stage_gates_authorized_by_builder=False,
        S_calibration_contract="All numerical/integrity gates and each of three TILED groups same-PSO A1/B/A2 median max/min span <=0.5%; exact-control groups are not null. Independent frozen reader required.",
        R_component_prerequisite="Only after independent S calibration passes and a separate R protocol/reader freeze; no parameter selection using either result.",
        R_component_contract="All six layer/control groups abs drift<=2% and conservative savings strictly>2%; all original complete scores/actual TopK/inputs/guards gates.",
        conditioner_contract="Each sample's first step uses the middle B actual PSO with logical provider/output2, same args/grid/smem and original full direct-shared-output observer; fully recorded but never a performance sample.",
        timing_records_expected=240, conditioner_records_expected=60,
        warmup_arm_records_expected=18, formal_arm_records_expected=162,
        original_shared_output_observer_preserved=True,
        additional_cpu_timestamps_or_output_snapshots=False,
        production_adoption=False, fullmodel_eligible_from_binary_completion=False,
        runtime_registered=False, gpu_work_submitted=False, binary_executed=False,
        original_replay_file_unchanged=True, compile_only_pass=False, compilation_skipped=args.emit_only,
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
            raise ValueError("old replay changed while CPU build was in progress")
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
