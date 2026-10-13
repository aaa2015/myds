#!/usr/bin/env python3
"""Build the independent Q4 sumy library without replacing production assets."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    dest = args.build_dir.expanduser().resolve()
    if dest == root or dest == root / "bin" or dest == root / "build":
        parser.error("use a dedicated probe directory, not production build/bin")
    dest.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("myds_metallib_builder", root / "tools/build_metallib.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    paths = [root / "metal" / name for name in builder.SHADER_FILES]
    paths.append(root / "metal/probe_q4_sumy.metal")
    combined = dest / "q4_sumy.metal"
    combined.write_text(builder.PREAMBLE + "\n" + "\n".join(
        f'\n#line 1 "{p.name}"\n' + p.read_text() for p in paths
    ) + builder.COPY_KERNELS)
    air = dest / "q4_sumy.air"
    lib = dest / "q4_sumy.metallib"
    subprocess.run([
        "xcrun", "-sdk", "macosx", "metal", "-mmacosx-version-min=15.0",
        "-O3", "-ffast-math", "-DDS4_METAL_HC_STABLE=1",
        "-DDS4_METAL_NORM_RSQRT_DISABLE=1", "-c", str(combined), "-o", str(air)
    ], check=True)
    subprocess.run(["xcrun", "-sdk", "macosx", "metallib", str(air), "-o", str(lib)], check=True)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = {
        "production_assets_changed": False,
        "builder_sha256": sha(Path(__file__).resolve()),
        "production_builder_sha256": sha(root / "tools/build_metallib.py"),
        "compile_options": ["-mmacosx-version-min=15.0", "-O3", "-ffast-math", "-DDS4_METAL_HC_STABLE=1", "-DDS4_METAL_NORM_RSQRT_DISABLE=1"],
        "inputs": {str(p.relative_to(root)): sha(p) for p in paths},
        "combined_sha256": sha(combined),
        "metallib_sha256": sha(lib),
        "library": str(lib),
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"library": str(lib), "metallib_sha256": sha(lib)}))


if __name__ == "__main__":
    main()
