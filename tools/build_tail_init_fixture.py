#!/usr/bin/env python3
"""Build a real missing-tail metallib and production-runtime test; never run GPU.

The runtime is compiled unchanged into the test translation unit so the test
can verify internal cleanup state. Only one optional shader host_name is
renamed. All generated/copied files stay in --output (default build/tail-init).
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=repo)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    output = (args.output or source / "build/tail-init").resolve()
    # Copy destinations must never replace source files or an ancestor tree.
    if output == source or output in source.parents:
        parser.error("--output must not contain the source tree")
    if source in output.parents and source / "build" not in output.parents:
        parser.error("an in-repository --output must be below build/")
    fixture = repo / "tests/test_tail_init.m"
    output.mkdir(parents=True, exist_ok=True)
    (output / "tools").mkdir(exist_ok=True)
    metal = output / "metal"
    if metal.exists():
        shutil.rmtree(metal)
    shutil.copytree(source / "metal", metal)
    shutil.copy2(source / "tools/build_metallib.py", output / "tools/build_metallib.py")
    path = metal / "moe.metal"
    old = path.read_text()
    target = 'host_name("kernel_mul_mm_id_iq2_xxs_pair_swiglu_f16_tail_cull")'
    omitted = 'host_name("kernel_myds_fixture_omitted_tail_export")'
    if old.count(target) != 1:
        parser.error("expected exactly one optional tail host_name")
    path.write_text(old.replace(target, omitted))
    runtime = source / "src/myds_metal.m"
    names = re.findall(r'^static id<[^>]+>\s+(g_\w+)\s*=\s*nil;', runtime.read_text(), re.M)
    if not names:
        parser.error("could not identify runtime object globals for cold-state checks")
    (output / "cold_id_checks.inc").write_text(''.join(f'    COLD({name});\n' for name in names))
    subprocess.run([sys.executable, str(output / "tools/build_metallib.py"),
                    "--bin-dir", str(output / "missing-tail"), "--build-dir", str(output / "build")],
                   cwd=output, check=True)
    for name in ("myds_gguf", "myds_log"):
        subprocess.run(["xcrun", "clang", "-O3", "-std=c11", "-I" + str(source / "include"),
                        "-c", str(source / f"src/{name}.c"), "-o", str(output / f"build/{name}.o")], check=True)
    subprocess.run(["xcrun", "clang", "-O3", "-fobjc-arc", "-Wall", "-Wextra",
                    "-I" + str(source / "include"), "-I" + str(source / "src"), "-I" + str(output),
                    str(fixture), str(output / "build/myds_gguf.o"), str(output / "build/myds_log.o"),
                    "-framework", "Foundation", "-framework", "Metal", "-lpthread", "-lm",
                    "-o", str(output / "test_tail_init")], check=True)
    files = {"production_runtime": runtime, "shader_moe": path,
             "missing_tail_library": output / "missing-tail/myds.metallib", "fixture": fixture}
    manifest = {name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for name, path in files.items()}
    manifest["shader_only_difference"] = {"old": target, "new": omitted}
    manifest["cold_object_count"] = len(names)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("BUILD ONLY COMPLETE; GPU fixture has not been executed.")
    print(f"Root-run command: {output / 'test_tail_init'} {output / 'missing-tail'} {source / 'bin'}")


if __name__ == "__main__":
    main()
