#!/usr/bin/env python3
"""Prepare isolated integration CPU build or frozen Q/P protocols; never starts GPU/model.

The build subcommand compiles only the native benchmark and executes its two
--preflight modes. Run the unchanged collector separately after independent
reader --initialize-only. All output roots must be new.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PACKAGE = HERE / "frozen-driver"
if not PACKAGE.is_dir():
    PACKAGE = HERE.parent / "docs/benchmarks/2026-10-26-register-only-integration/frozen-driver"
OLD = HERE.parent / "docs/benchmarks/2026-10-26-register-only-and-narrow-audit"
SUBJECT = "prefill_indexer_register_only"
CANDIDATE = "fe55793346e15119fce4e0f40d4bf6d3023fc3694ffc231e66b8a89e3ae0f74d"
GOLD = "3837102396703da59e0eebbc0de1fdc58b68b3a3a5ec5d07504b351c67a29ba9"
CONFIG = "735e0dc73fbbf7483dba0bffe4ad379a4dab910bb734d3a6708df65c8edc16d7"
DRIVER = ("bench_indexer_register_only_fullmodel.c", "indexer_register_only_fullmodel.mk",
          "indexer_register_only_runtime_adapter.inc", "indexer_register_only_fullmodel_state.inc",
          "prepare_indexer_register_only_fullmodel.py", "bench_output_a_fullmodel.c")


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def desc(path):
    path = Path(path).absolute()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def inventory(root):
    result = {}
    for path in sorted(Path(root).rglob("*")):
        need(not path.is_symlink(), "source symlink rejected: " + str(path))
        if path.is_file() and "__pycache__" not in path.parts:
            result[str(path.relative_to(root))] = dict(bytes=path.stat().st_size, sha256=sha(path))
    return result


def invsha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def run(argv, cwd, env, prefix):
    result = subprocess.run(argv, cwd=cwd, env=env, capture_output=True)
    Path(str(prefix) + ".stdout").write_bytes(result.stdout)
    Path(str(prefix) + ".stderr").write_bytes(result.stderr)
    need(result.returncode == 0, "CPU command failed, retained logs: " + str(prefix))
    return result


def build(args):
    out = args.out.absolute()
    need(not out.exists(), "new build root required")
    need(sha(args.candidate_library) == CANDIDATE and sha(args.gold_library) == GOLD and sha(args.config) == CONFIG,
         "qualified candidate/deployed library and unchanged63 config")
    out.mkdir(parents=True)
    source, baseline = out / "source", out / "baseline"
    shutil.copytree(args.source, source)
    baseline.mkdir()
    # Baseline may include many non-build tools; compare the exact curated input
    # closure supplied by root, not an artificial equality of repository trees.
    for relative in inventory(source):
        old = args.baseline_source / relative
        if old.is_file():
            target = baseline / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old, target)
    (source / "tools").mkdir(exist_ok=True)
    frozen = load(PACKAGE / "frozen-files.json")
    for name in DRIVER:
        target = source / "tools" / name
        need(not target.exists() or sha(target) == frozen[name]["sha256"], "existing driver differs from frozen: " + name)
        shutil.copy2(PACKAGE / name, target)
    validation = out / "validation"
    validation.mkdir()
    shutil.copytree(PACKAGE, validation / "frozen-driver")
    adapter = HERE / "review_register_only_integration_fullmodel.py"
    shutil.copy2(adapter, validation / adapter.name)
    shutil.copy2(__file__, validation / Path(__file__).name)
    shutil.copy2(args.config, out / "native.conf")
    before, original = inventory(source), inventory(baseline)
    changed = sorted(n for n in original if original[n] != before[n])
    added = sorted(set(before) - set(original))
    allowed_changes = ["Makefile", "include/myds_v41_gpu.h", "src/myds_v41_gpu.c", "src/myds_v41.c",
                       "src/myds_metal.m", "tools/build_metallib.py"]
    allowed_adds = [n for n in added if n == "metal/myds_indexer_register_only.metal" or
                    n in ["tools/" + x for x in DRIVER] or n.startswith("tests/test_indexer_register_only_")]
    need(set(changed) <= set(allowed_changes) and set(added) == set(allowed_adds), "unreviewed source difference")
    spec = importlib.util.spec_from_file_location("integration_core", PACKAGE / "review_indexer_fullmodel_v2.py")
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    refs = {}
    signatures = [("src/myds_v41_gpu.c", s) for s in ("static int indexscores_chunk(",
        "int myds_v41_gpu_indexscores_chunk(", "int myds_v41_gpu_indexscores_chunk_bf16(")]
    signatures += [("src/myds_v41.c", s) for s in ("int myds_v41_forward_batch_spec(", "int myds_v41_forward_batch_spec_ex(")]
    for filename, signature in signatures:
        old = core.complete_c_function((baseline / filename).read_bytes(), signature)
        new = core.complete_c_function((source / filename).read_bytes(), signature)
        need(old == new, "unchanged reference function: " + signature)
        refs[signature] = dict(file=filename, sha256=hashlib.sha256(new).hexdigest())
    objects, bindir, gold = out / "objects", out / "bin", out / "gold"
    binary = bindir / "bench_indexer_register_only_fullmodel"
    command = ["make", "-f", "tools/indexer_register_only_fullmodel.mk", "-j4", "BUILD_DIR=" + str(objects),
               "BIN_DIR=" + str(bindir), "REGISTER_ONLY_ADAPTER_HEADER=tools/indexer_register_only_runtime_adapter.inc",
               "DEPLOY_TARGET=26.0", str(binary)]
    env = {k: v for k, v in os.environ.items() if not k.startswith("MYDS_")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    run(command, source, env, out / "build")
    need(inventory(source) == before, "source modified during build")
    gold.mkdir()
    shutil.copy2(binary, gold / binary.name)
    shutil.copy2(args.candidate_library, bindir / "myds.metallib")
    shutil.copy2(args.gold_library, gold / "myds.metallib")
    pf = dict(subject=SUBJECT, passed=False, gpu_work_submitted=False, config=desc(out / "native.conf"), runs=[])
    for mode in (0, 1):
        directory = out / ("preflight-" + str(mode))
        for leaf in ("cwd", "logits", "state"):
            (directory / leaf).mkdir(parents=True, exist_ok=True)
        pe = dict(env, MYDS_INDEXER_REGISTER_ONLY=str(mode), MYDS_INDEXER_K_RESIDENT="0", MYDS_DECODE_KV_ONLY_FINISH="0",
                  MYDS_BENCH_LOGITS_DIR=str(directory / "logits"), MYDS_BENCH_STATE_DIR=str(directory / "state"))
        argv = [str(binary), "--config", str(out / "native.conf"), "--preflight"]
        result = run(argv, directory / "cwd", pe, directory / "preflight")
        metadata = json.loads(result.stdout)
        need(metadata["execute_requested"] is False and metadata["runtime_mode"] == mode and
             metadata["effective_configuration_count"] == 68, "actual CPU-only preflight")
        pf["runs"].append(dict(mode=mode, argv=argv, cwd=str(directory / "cwd"), returncode=0, metadata=metadata,
            stdout=desc(directory / "preflight.stdout"), stderr=desc(directory / "preflight.stderr"),
            model_context_or_device_created=False, gpu_work_submitted=False))
    pf["passed"] = True
    write(out / "preflight.json", pf)
    write(out / "source-check.json", dict(complete_source_inventory=before, before_after_source_unchanged=True,
        source_inventory_sha256=invsha(before), gpu_work_submitted=False))
    private = dict(kind="register_only_integration_CPU_build", schema_version=1, CPU_build_only_passed=True,
        GPU_work_submitted=False, production_deployed=False, driver_subject=SUBJECT, source_root=str(source), baseline_root=str(baseline),
        source_inventory=before, baseline_inventory=original, complete_private_source_inventory_sha256=invsha(before),
        changed_paths=changed, added_paths=added, removed_paths=[], allowed_changed_paths=allowed_changes, allowed_added_paths=allowed_adds,
        validation_dependencies=frozen, driver_files={n: frozen[n] for n in DRIVER}, reference_functions=refs,
        generated_base=desc(objects / "bench_indexer_register_only_fullmodel_base.inc"))
    write(out / "private-build-manifest.json", private)
    target = dict(kind="M2_register_only_integration_target_build", schema_version=1, subject=SUBJECT,
        compile_only_pass=True, gpu_work_submitted=False, production_deployed=False, source_inventory_sha256=invsha(before),
        source_build_manifest=desc(out / "private-build-manifest.json"), source_check=desc(out / "source-check.json"),
        binary=desc(binary), gold_binary=desc(gold / binary.name), libraries=dict(private=desc(bindir / "myds.metallib"), G_lib=desc(gold / "myds.metallib")),
        preflight=desc(out / "preflight.json"), helper=desc(validation / Path(__file__).name), adapter=desc(validation / adapter.name),
        build_logs=dict(stdout=desc(out / "build.stdout"), stderr=desc(out / "build.stderr")), commands=[command])
    write(out / "target-build-manifest.json", target)
    print(json.dumps(dict(target_build_manifest=desc(out / "target-build-manifest.json"), GPU_work_submitted=False)))


def protocol(args):
    out = args.out.absolute()
    need(not out.exists(), "new protocol root required")
    out.mkdir(parents=True)
    target = load(args.target_build_manifest)
    stage_root = args.target_build_manifest.parent
    validation = stage_root / "validation"
    reader = validation / "review_register_only_integration_fullmodel.py"
    wrapper = validation / "frozen-driver/register-only-fullmodel-window-wrapper-v2.py"
    if args.stage == "Q":
        p = load(args.template_protocol)
        need(p["stage"] == "Q", "original Q template")
        for workload in p["workloads"]:
            for item in workload["artifacts"].values():
                need(desc(item["path"]) == item, "exact original request/tokenizer/rendered bytes retained")
        canonical = {path: desc(path) for path in p["canonical_files"]}
        model_path = Path(p["model"]["path"])
        st = model_path.stat()
        with model_path.open("rb") as stream:
            prefix = stream.read(1048576)
        need(st.st_size == p["model"]["bytes"] and hashlib.sha256(prefix).hexdigest() == p["model"]["prefix_sha256"], "model stat/prefix")
        p["model"].update(inode=st.st_ino, mtime_ns=st.st_mtime_ns)
        p["canonical_files"] = canonical
        write(out / "baseline-identity.json", dict(canonical, model=p["model"], whole_model_payload_sha256_verified=False))
        gates = load(p["artifacts"]["predeclared_p_gates"]["path"])
        write(out / "predeclared-p-gates.json", gates)
        p["artifacts"]["baseline_identity"] = desc(out / "baseline-identity.json")
        p["artifacts"]["predeclared_p_gates"] = desc(out / "predeclared-p-gates.json")
        p["request_contract"].update(repeats=1, verification_runs=1, formal_runs=0, total_native_generations=12)
        names = [("G_lib", 0), ("off", 0), ("on", 1)]
    else:
        p = load(args.qualification_protocol)
        qr = load(args.qualification_review)
        need(qr["qualification_passed"] is True and qr["two_workload_qualification_completed"] is True and
             qr["native_generation_count"] == 12 and qr["binary_sha256"] == target["binary"]["sha256"] and
             qr["protocol_sha256"] == sha(args.qualification_protocol), "current integration Q12 before P24")
        p["artifacts"]["qualification_protocol"] = desc(args.qualification_protocol)
        p["prerequisite"] = dict(report_path=str(args.qualification_review), report=desc(args.qualification_review),
            future_report_exact_descriptor_bound_in_prelaunch_plan=True, protocol_sha256=qr["protocol_sha256"], reader_sha256=qr["reader_sha256"])
        p["request_contract"].update(repeats=3, formal_runs=3, total_native_generations=24)
        names = [("A1", 0), ("B", 1), ("A2", 0)]
    p["artifacts"].update(private_build_manifest=target["source_build_manifest"], target_build_manifest=desc(args.target_build_manifest),
        config=desc(stage_root / "native.conf"), core_reader=desc(validation / "frozen-driver/review_indexer_fullmodel_v2.py"), reader=desc(reader))
    p.update(stage=args.stage, kind="register_only_fullmodel_qualification_protocol" if args.stage == "Q" else "register_only_fullmodel_performance_protocol",
        wrapper_path=str(wrapper), wrapper_sha256=sha(wrapper), window_root=str(out / (args.stage + "-window")),
        model_prefix_path=str(out / (args.stage + "-window") / "model-prefix.bin"))
    p["providers"] = []
    for name, mode in names:
        lib = target["libraries"]["G_lib" if name == "G_lib" else "private"]
        p["providers"].append(dict(name=name, mode=mode, binary=target["gold_binary" if name == "G_lib" else "binary"],
            library=lib, expected_loaded_library_path=lib["path"]))
    p["output_layout"]["provider_order"] = [n for n, _ in names]
    p["output_layout"]["stdout_kinds"] = ["register_only_metadata", "metadata", "load"] + ["sample"] * (p["request_contract"]["repeats"] + 1)
    write(out / "protocol.json", p)
    print(json.dumps(dict(protocol=desc(out / "protocol.json"), GPU_work_submitted=False)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    b = sub.add_parser("build")
    for name in ("source", "baseline-source", "candidate-library", "gold-library", "config", "out"):
        b.add_argument("--" + name, type=Path, required=True)
    for stage in ("Q", "P"):
        p = sub.add_parser(stage)
        for name in ("target-build-manifest", "out"):
            p.add_argument("--" + name, type=Path, required=True)
        if stage == "Q":
            p.add_argument("--template-protocol", type=Path, required=True)
        else:
            p.add_argument("--qualification-protocol", type=Path, required=True)
            p.add_argument("--qualification-review", type=Path, required=True)
    args = parser.parse_args()
    build(args) if args.stage == "build" else protocol(args)


if __name__ == "__main__":
    main()
