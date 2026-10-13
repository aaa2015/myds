#!/usr/bin/env python3
"""Compile an isolated production-clone + Q4 V4.1 candidate library; no GPU."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build-dir",type=Path,required=True)
    args=ap.parse_args()
    root=Path(__file__).resolve().parents[1]
    dest=args.build_dir.expanduser().resolve()
    if dest in (root,root/"bin",root/"build"):
        ap.error("use an isolated build directory")
    dest.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location("production_shader_builder",root/"tools/build_metallib.py")
    builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
    paths=[root/"metal"/n for n in builder.SHADER_FILES]
    candidate=root/"metal/myds_q4_v41.metal"
    if candidate not in paths:paths.append(candidate)
    combined=dest/"q4_v41.metal"
    combined.write_text(builder.PREAMBLE+"\n"+"\n".join(f'\n#line 1 "{p.name}"\n'+p.read_text() for p in paths)+builder.COPY_KERNELS)
    air=dest/"q4_v41.air";lib=dest/"q4_v41.metallib"
    options=["-mmacosx-version-min=15.0","-O3","-ffast-math","-DDS4_METAL_HC_STABLE=1","-DDS4_METAL_NORM_RSQRT_DISABLE=1"]
    commands=[["xcrun","-sdk","macosx","metal",*options,"-c",str(combined),"-o",str(air)],
              ["xcrun","-sdk","macosx","metallib",str(air),"-o",str(lib)]]
    for cmd in commands:subprocess.run(cmd,check=True)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    manifest={"schema_version":1,"gpu_executed":False,"production_assets_replaced":False,
              "commands":commands,"compiler_path":subprocess.check_output(["xcrun","--find","metal"],text=True).strip(),
              "sdk_path":subprocess.check_output(["xcrun","-sdk","macosx","--show-sdk-path"],text=True).strip(),
              "builder_sha256":sha(Path(__file__).resolve()),"production_builder_sha256":sha(root/"tools/build_metallib.py"),
              "inputs":{str(p.relative_to(root)):sha(p) for p in paths},
              "combined_sha256":sha(combined),"library_sha256":sha(lib),"library":str(lib),
              "golden_requirement":"Run harness with actual production --golden-metallib; clone library is not authoritative."}
    (dest/"build-manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"library":str(lib),"sha256":sha(lib),"gpu_executed":False}))
if __name__=="__main__":main()

