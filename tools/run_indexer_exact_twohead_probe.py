#!/usr/bin/env python3
"""Build/run an isolated two-head staged version of the exact indexer scorer.

This does not modify the production library or runtime.  The candidate loads two consecutive heads into separate threadgroup Q
buffers, then executes their original arithmetic in order with two barriers
per pair rather than per head.  GPU execution requires --run.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def one(source, old, new):
    if source.count(old) != 1:
        raise ValueError('source anchor is not unique: ' + repr(old[:70]))
    return source.replace(old, new, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--run', action='store_true')
    p.add_argument('--correctness-only', action='store_true', help='skip timing entirely')
    p.add_argument('--deployed-metallib', type=Path)
    args = p.parse_args()
    out = args.out_dir.resolve()
    if out.exists() or out.is_relative_to(ROOT):
        p.error('output must be a new directory outside the repository')
    if args.correctness_only and not args.run:
        p.error('--correctness-only requires --run')
    if args.run and not args.deployed_metallib:
        p.error('--run requires --deployed-metallib')

    spec = importlib.util.spec_from_file_location('frozen_builder', ROOT / 'tools/build_indexer_k_resident_probe.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    build, build_source = helper.load_frozen_build(ROOT)
    native, shaders, _ = helper.assemble_full_native(ROOT, build)
    source = (ROOT / 'metal/dsv4_misc.metal').read_text()
    original = helper.extract(source, 'kernel void kernel_myds_indexer_score_chunk_exact(')
    staging = """    threadgroup float4 *qtg = (threadgroup float4 *)shared;       /* 256 float4 = 1024 floats */
    threadgroup float  *wtg = (threadgroup float *)(qtg + 256u);  /* 256 floats */"""
    paired_staging = """    threadgroup float4 *qtg = (threadgroup float4 *)shared;       /* two Q buffers */
    threadgroup float  *wtg = (threadgroup float *)(qtg + 512u);  /* 256 floats */"""
    loop = """    for (uint head = 0; head < 32u; head++) {
        for (uint i = tid; i < q_vec_count; i += 128u) {
            const uint t = i >> 5;
            const uint l = i & 31u;
            const uint tok = tok_base + t;
            device const float4 *q4_ptr = (device const float4 *)(q +
                ((uint64_t)tok * 32u + head) * 128u * sizeof(float));
            qtg[i] = q4_ptr[l];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
"""
    paired_loop = """    for (uint head_base = 0; head_base < 32u; head_base += 2u) {
        for (uint i = tid; i < 2u * q_vec_count; i += 128u) {
            const uint head_idx = i / q_vec_count;
            const uint local_i = i - head_idx*q_vec_count;
            const uint t = local_i >> 5;
            const uint l = local_i & 31u;
            const uint tok = tok_base + t;
            device const float4 *q4_ptr = (device const float4 *)(q +
                ((uint64_t)tok * 32u + head_base + head_idx) * 128u * sizeof(float));
            qtg[i] = q4_ptr[l];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint head_offset = 0; head_offset < 2u; head_offset++) {
            const uint head = head_base + head_offset;
"""
    load = '            const float4 q4 = qtg[t * 32u + (uint)lane];'
    paired_load = '            const float4 q4 = qtg[head_offset*q_vec_count + t * 32u + (uint)lane];'
    tail = """        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint out_t ="""
    paired_tail = """        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint out_t ="""
    edits = [('kernel void kernel_myds_indexer_score_chunk_exact(', 'kernel void probe_indexer_exact_twohead('),
             (staging,paired_staging),(loop,paired_loop),(load,paired_load),(tail,paired_tail)]
    candidate=original
    for before,after in edits:candidate=one(candidate,before,after)
    restored=candidate
    for before,after in reversed(edits):restored=one(restored,after,before)
    if restored!=original:raise ValueError('candidate changes outside declared two-head transform')

    native += build.REGISTER_ONLY_MARKER.encode() + (ROOT / 'metal' / build.REGISTER_ONLY_SHADER).read_bytes()
    combined = native + b'\n#line 1 "indexer_exact_twohead_probe.metal"\n' + candidate.encode() + b'\n'
    out.mkdir(parents=True)
    (out / 'all.metal').write_bytes(combined)
    (out / 'candidate.metal').write_text(candidate)
    inputs = [ROOT / 'tools/bench_indexer_exact_twohead.m', ROOT / 'tools/indexer_long_topk_probe.inc',
              ROOT / 'src/myds_v41_gpu.c', ROOT / 'src/myds_gguf.c', ROOT / 'src/myds_log.c',
              ROOT / 'src/myds_configfile.c', ROOT / 'src/myds_metal.m'] + sorted((ROOT/'include').glob('*.h'))
    identity = {'baseline_exact_shader_sha256': sha(original.encode()),
                'candidate_sha256': sha(candidate.encode()), 'native_prefix_sha256': sha(native),
                'combined_sha256': sha(combined), 'reverse_exact': True,
                'production_runtime_connected': False, 'shader_files': shaders,
                'inputs': {str(f.relative_to(ROOT)): sha(f.read_bytes()) for f in inputs},
                'build_metallib_sha256': sha(build_source), 'commands': []}
    metal_flags = ['-mmacosx-version-min=15.0','-O3','-ffast-math',
                   '-DDS4_METAL_HC_STABLE=1','-DDS4_METAL_NORM_RSQRT_DISABLE=1']
    commands = [['xcrun','-sdk','macosx','metal',*metal_flags,'-c',str(out/'all.metal'),'-o',str(out/'probe.air')],
                ['xcrun','-sdk','macosx','metallib',str(out/'probe.air'),'-o',str(out/'probe.metallib')]]
    common = ['xcrun','clang','-O3','-Wall','-Wextra','-mmacosx-version-min=26.0',
              '-I'+str(ROOT/'include'),'-I'+str(ROOT/'third_party'),'-I'+str(ROOT/'tools')]
    objects = []
    for f in inputs[:7]:
        if f.suffix == '.inc':
            continue
        obj = out/(f.stem+'.o')
        flags = ['-fobjc-arc','-Wno-deprecated-declarations'] if f.suffix == '.m' else ['-std=c11']
        commands.append(common+flags+['-c',str(f),'-o',str(obj)])
        objects.append(str(obj))
    commands.append(['xcrun','clang','-mmacosx-version-min=26.0',*objects,
                     '-framework','Foundation','-framework','Metal','-lm','-lpthread','-o',str(out/'probe')])
    identity['commands'] = commands
    (out/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    with (out/'compile.log').open('x') as log:
        for command in commands:
            print(json.dumps(command),file=log,flush=True)
            subprocess.run(command,stdout=log,stderr=log,check=True)
    for f in inputs:
        if sha(f.read_bytes()) != identity['inputs'][str(f.relative_to(ROOT))]:
            raise RuntimeError('source changed during build: '+str(f))
    identity['metallib_sha256'] = sha((out/'probe.metallib').read_bytes())
    identity['binary_sha256'] = sha((out/'probe').read_bytes())
    (out/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    if not args.run:
        print('build complete; no GPU submitted')
        return
    procs = subprocess.check_output(['ps','-axo','pid=,comm='],text=True)
    for line in procs.splitlines():
        fields = line.strip().split(None,1)
        if len(fields)==2 and (Path(fields[1]).name in {'myds','myglm','ds4','llama-server','ollama'} or Path(fields[1]).name.startswith('bench_')):
            raise RuntimeError('inference/benchmark process conflicts with GPU test: '+line)
    lib=args.deployed_metallib.resolve(strict=True)
    command=[str(out/'probe'),str(out/'probe.metallib'),str(lib)]
    if args.correctness_only:
        command.append('--correctness-only')
    with (out/'results.jsonl').open('x') as result, (out/'gpu.stderr').open('x') as stderr:
        subprocess.run(command,stdout=result,stderr=stderr,check=True)
    print('GPU probe complete: '+str(out/'results.jsonl'))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print('exact-twohead-probe:',exc,file=sys.stderr)
        sys.exit(1)
