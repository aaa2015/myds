#!/usr/bin/env python3
"""Build-only fixed ordinary Output_A wrapper, without production registration.

The current production Q4_K helper and wrapper source is SHA pinned. All frozen
shader sources are concatenated unchanged; a new wrapper is appended. It calls
the unchanged by-value helper with fixed shape/strides and keeps its BF16 store,
NR2, NSG4 and dispatch geometry. Identical source arithmetic is not GPU parity.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode=True
MOE_SHA256='78cb927e7884254832f3c37e97b1c4952c1a1863555616693f3b35fb787f5776'
SYMBOL='kernel_myds_outa_q4_v41_fixed_bf16'
NATIVE='kernel_dsv4_attn_out_low_q4_K_bf16'


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--emit-only',action='store_true')
    args=parser.parse_args()
    root=args.source_root.resolve(strict=True);out=args.out_dir.resolve()
    require(not out.exists() and not out.is_relative_to(root),'New private output outside source checkout required')
    require(sha(root/'metal/moe.metal')==MOE_SHA256,'Production Q4_K helper source differs from reviewed version')
    probe=root/'metal/probe_output_a_fixed.metal'
    text=probe.read_text()
    require(text.count('kernel void '+SYMBOL+'(')==1,'Expected single explicit probe symbol')
    require('kernel_mul_mv_q4_K_f32_impl<2,ds4_metal_args_mul_mv,true>' in text,
        'Expected unchanged original helper with original by-value ABI and BF16 store')
    spec=importlib.util.spec_from_file_location('frozen_build',root/'tools/build_metallib.py')
    build=importlib.util.module_from_spec(spec);spec.loader.exec_module(build)
    require('probe_output_a_fixed.metal' not in build.SHADER_FILES,'Probe must not be registered for production')
    sources=[root/'tools/build_metallib.py']+[root/'metal'/name for name in build.SHADER_FILES]+[probe]
    require(all(p.is_file() for p in sources),'Frozen shader source missing')
    out.mkdir(parents=True)
    combined=build.PREAMBLE
    for name in build.SHADER_FILES:
        combined+='\n#line 1 '+json.dumps(name)+'\n'+(root/'metal'/name).read_text()
    combined+='\n'+build.COPY_KERNELS+'\n#line 1 "probe_output_a_fixed.metal"\n'+text
    (out/'all.metal').write_text(combined)
    identity=dict(schema_version=1,gpu_work_submitted=False,production_registered=False,
        source_root=str(root),native_symbol=NATIVE,probe_symbol=SYMBOL,
        sources=[dict(path=str(p.relative_to(root)),sha256=sha(p),bytes=p.stat().st_size) for p in sources],
        builder_sha256=sha(Path(__file__)),all_metal_sha256=sha(out/'all.metal'),
        original_q4_helper_source_unchanged=True,
        abi=dict(args_bytes=136,buffer_slots=['original_id_args','q4k_weights','heads_f32','low_f32'],
            groups=[128,1,8],threads=[32,4,1],fc600_short=4,nr0=2,dynamic_threadgroup_bytes=256,
            weight_bytes=18874368,input_bytes=131072,output_bytes=32768),
        limitations=['No actual GPU work is submitted by this builder.',
            'Fast-math constant propagation can alter device rounding; full F32-bit gate is required.',
            'Repeated real layer fixtures can reuse caches and do not prove full-model speedup.'])
    if not args.emit_only:
        commands=[['xcrun','-sdk','macosx','metal','-mmacosx-version-min=26.0','-O3','-ffast-math',
            '-DDS4_METAL_HC_STABLE=1','-DDS4_METAL_NORM_RSQRT_DISABLE=1',
            '-c',str(out/'all.metal'),'-o',str(out/'probe.air')],
            ['xcrun','-sdk','macosx','metallib',str(out/'probe.air'),'-o',str(out/'probe.metallib')]]
        env=dict(os.environ);env.setdefault('DEVELOPER_DIR','/Applications/Xcode.app/Contents/Developer')
        with (out/'compile.log').open('w') as log:
            for command in commands:
                log.write(json.dumps(command)+'\n');log.flush()
                subprocess.run(command,env=env,stdout=log,stderr=log,check=True)
        identity.update(compile_only_pass=True,commands=commands,metallib_sha256=sha(out/'probe.metallib'))
    (out/'source-identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    print(json.dumps({k:identity[k] for k in ('native_symbol','probe_symbol','production_registered','gpu_work_submitted')},indent=2))


if __name__=='__main__':main()
