#!/usr/bin/env python3
"""Use frozen fullraw Q/P reader, replacing only the old private-source identity gate.

No numerical, provider, timing, state, raw-byte, Q-before-P or library gate is
relaxed. The integration manifest proves complete current source snapshots and
the unchanged driver. G_lib still means deployed library, not executable.
"""
import hashlib
import json
from pathlib import Path
import sys
import types

sys.dont_write_bytecode = True
PACKAGE = Path(__file__).resolve().parent / "frozen-driver"
if not PACKAGE.is_dir():
    PACKAGE = Path(__file__).resolve().parents[1] / "docs/benchmarks/2026-10-26-register-only-integration/frozen-driver"
MANIFEST = json.loads((PACKAGE / "frozen-files.json").read_text())
for name, item in MANIFEST.items():
    payload = (PACKAGE / name).read_bytes()
    if len(payload) != item["bytes"] or hashlib.sha256(payload).hexdigest() != item["sha256"]:
        raise ValueError("frozen validation dependency changed: " + name)
sys.path.insert(0, str(PACKAGE))
frozen = types.ModuleType("register_only_frozen_reader")
# The executing reader is this identity adapter plus exact frozen source. The
# adapter SHA binds the module and manifest check, and build binds every file.
frozen.__file__ = __file__
exec(compile((PACKAGE / "review_indexer_register_only_fullmodel_v2.py").read_bytes(),
             str(PACKAGE / "review_indexer_register_only_fullmodel_v2.py"), "exec"), frozen.__dict__)


def verify_runtime_build(path):
    b = frozen.read_json(path)
    require = frozen.require
    require(b["kind"] == "register_only_integration_CPU_build" and b["schema_version"] == 1 and
            b["CPU_build_only_passed"] is True and b["GPU_work_submitted"] is False and
            b["production_deployed"] is False and b["driver_subject"] == frozen.SUBJECT,
            "new integration CPU build; no old557 identity substitution")
    source, baseline = Path(b["source_root"]), Path(b["baseline_root"])
    frozen.core.verify_inventory(source, b["source_inventory"])
    frozen.core.verify_inventory(baseline, b["baseline_inventory"])
    require(frozen.core.inventory_digest(b["source_inventory"]) == b["complete_private_source_inventory_sha256"], "complete current source identity")
    for name, item in b["validation_dependencies"].items():
        require(name in MANIFEST and item == MANIFEST[name] and frozen.file_sha(PACKAGE / name) == item["sha256"], "frozen reader/driver dependency")
    for name, item in b["driver_files"].items():
        require(item == MANIFEST[name] and frozen.file_sha(source / "tools" / name) == item["sha256"], "unchanged six-hook driver files")
    before, after = b["baseline_inventory"], b["source_inventory"]
    changed = sorted(n for n in before if n in after and before[n] != after[n])
    removed = sorted(set(before) - set(after))
    added = sorted(set(after) - set(before))
    require(changed == b["changed_paths"] and removed == b["removed_paths"] and added == b["added_paths"], "complete source differences")
    require(not removed and all(n in b["allowed_changed_paths"] for n in changed), "no unlisted existing source edits")
    require(all(n in b["allowed_added_paths"] for n in added), "no unlisted added sources")
    for signature, item in b["reference_functions"].items():
        old = frozen.core.complete_c_function((baseline / item["file"]).read_bytes(), signature)
        new = frozen.core.complete_c_function((source / item["file"]).read_bytes(), signature)
        require(old == new and frozen.digest(new) == item["sha256"], "public/scalar/spec reference function remains exact")
    require(frozen.file_sha(source / "metal/myds_indexer_register_only.metal") ==
            "a62922c6f5937236bd0e354dd06943b39bfa5f788419c16a8700da793769d616", "exact verified v2 shader source")
    generated = Path(b["generated_base"]["path"])
    # Frozen generator performs exact six unique replacements and their inverse.
    from prepare_indexer_register_only_fullmodel import prepare
    expected, proof = prepare((source / "tools/bench_output_a_fullmodel.c").read_text())
    require(generated.read_text() == expected and frozen.file_sha(generated) == b["generated_base"]["sha256"] and
            proof["reverse_restores_base_byte_for_byte"] is True, "original timing/export driver with exact six hooks")
    return dict(build=b, manifest_sha256=frozen.file_sha(path), source_files_verified=len(after),
                baseline_files_verified=len(before), driver_inverse_exact=True,
                old_private_557_source_identity_claimed=False)


def check_target_build(self):
    b = frozen.read_json(self.args.target_build_manifest)
    require = frozen.require
    require(b["kind"] == "M2_register_only_integration_target_build" and b["subject"] == frozen.SUBJECT and
            b["compile_only_pass"] is True and b["gpu_work_submitted"] is False and b["production_deployed"] is False,
            "integration target build scope")
    require(b["source_inventory_sha256"] == self.private["complete_private_source_inventory_sha256"] and
            b["source_build_manifest"]["sha256"] == self.protocol["artifacts"]["private_build_manifest"]["sha256"], "target source/build binding")
    for name in ("source_build_manifest", "source_check", "binary", "gold_binary", "preflight", "helper", "adapter"):
        self.descriptor(b[name])
    for item in list(b["libraries"].values()) + list(b["build_logs"].values()):
        self.descriptor(item)
    require(b["binary"]["sha256"] == b["gold_binary"]["sha256"] and
            b["libraries"]["G_lib"]["sha256"] == frozen.GOLD_LIBRARY_SHA and
            b["libraries"]["private"]["sha256"] == "fe55793346e15119fce4e0f40d4bf6d3023fc3694ffc231e66b8a89e3ae0f74d",
            "same newC binary, real deployed library and exact previously qualified candidate library")
    source = frozen.read_json(self.descriptor(b["source_check"]))
    require(source["complete_source_inventory"] == self.private["source_inventory"] and source["before_after_source_unchanged"] is True,
            "complete build-source before/after inventory")
    pf = frozen.read_json(self.descriptor(b["preflight"]))
    require(pf["passed"] is True and pf["gpu_work_submitted"] is False and
            pf["config"] == self.protocol["artifacts"]["config"] and [r["mode"] for r in pf["runs"]] == [0, 1], "both mode CPU preflights")
    for r in pf["runs"]:
        stdout = self.descriptor(r["stdout"])
        self.descriptor(r["stderr"])
        require(r["returncode"] == 0 and r["metadata"] == frozen.read_json(stdout) and
                r["argv"] == [b["binary"]["path"], "--config", pf["config"]["path"], "--preflight"] and
                r["model_context_or_device_created"] is False, "actual CPU-only preflight")
        directory = Path(r["cwd"]).parent
        frozen.effective_configuration(r["metadata"], None, self.artifacts["config"].read_bytes(), r["mode"],
            directory / "logits", directory / "state", pf["config"]["path"], False, self.protocol["config_contract"]["file_entries"])
    require(b["adapter"]["sha256"] == frozen.file_sha(__file__), "executing integration identity adapter")
    self.target = b
    self.report["target_build_verified"] = dict(manifest_sha256=frozen.file_sha(self.args.target_build_manifest),
        source_check=source, commands=b["commands"], preflight_sha256=b["preflight"]["sha256"],
        identity_gate_revision="current_integration_snapshot_plus_unchanged_frozen_fullraw_reader")


frozen.verify_runtime_build = verify_runtime_build
frozen.Review.check_target_build = check_target_build
if __name__ == "__main__":
    frozen.main()
