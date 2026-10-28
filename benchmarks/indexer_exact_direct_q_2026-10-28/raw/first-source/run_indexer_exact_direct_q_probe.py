#!/usr/bin/env python3
"""Build/run an isolated direct-Q version of the exact indexer scorer.

This does not modify the production library or runtime.  The sole candidate
change replaces threadgroup Q/weight staging with identical direct loads and
removes the two staging barriers.  GPU execution requires --run.
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
    p.add_argument('--deployed-metallib', type=Path)
    args = p.parse_args()
    out = args.out_dir.resolve()
    if out.exists() or out.is_relative_to(ROOT):
        p.error('output must be a new directory outside the repository')
    if args.run and not args.deployed_metallib:
        p.error('--run requires --deployed-metallib')

    spec = importlib.util.spec_from_file_location('frozen_builder', ROOT / 'tools/build_indexer_k_resident_probe.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    build, build_source = helper.load_frozen_build(ROOT)
    native, shaders, _ = helper.assemble_full_native(ROOT, build)
    source = (ROOT / 'metal/dsv4_misc.metal').read_text()
    original = helper.extract(source, 'kernel void kernel_myds_indexer_score_chunk_exact(')
    staging = '''    threadgroup float4 *qtg = (threadgroup float4 *)shared;       /* 256 float4 = 1024 floats */
    threadgroup float  *wtg = (threadgroup float *)(qtg + 256u);  /* 256 floats */

    for (uint i = tid; i < tile_tokens * 32u; i += 128u) {
        const uint t = i >> 5;
        const uint h = i & 31u;
        const uint tok = tok_base + t;
        wtg[i] = weights[(uint64_t)tok * 32u + h];
    }

    float acc = 0.0f;
    const uint q_vec_count = tile_tokens * 32u;

    for (uint head = 0; head < 32u; head++) {
        for (uint i = tid; i < q_vec_count; i += 128u) {
            const uint t = i >> 5;
            const uint l = i & 31u;
            const uint tok = tok_base + t;
            device const float4 *q4_ptr = (device const float4 *)(q +
                ((uint64_t)tok * 32u + head) * 128u * sizeof(float));
            qtg[i] = q4_ptr[l];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
'''
    direct = '''    float acc = 0.0f;
    for (uint head = 0; head < 32u; head++) {
'''
    local_load = '''            const float4 q4 = qtg[t * 32u + (uint)lane];
            const float w = wtg[t * 32u + head];'''
    direct_load = '''            const uint tok = tok_base + t;
            device const float4 *q4_ptr = (device const float4 *)(q +
                ((uint64_t)tok * 32u + head) * 128u * sizeof(float));
            const float4 q4 = q4_ptr[(uint)lane];
            const float w = weights[(uint64_t)tok * 32u + head];'''
    tail = '''        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint out_t ='''
    no_barrier = '''    }

    const uint out_t ='''
    edits = [
        ('kernel void kernel_myds_indexer_score_chunk_exact(', 'kernel void probe_indexer_exact_direct_q('),
        ('        threadgroup float *shared [[threadgroup(0)]],\n', ''),
        (staging, direct), (local_load, direct_load), (tail, no_barrier),
    ]
    candidate = original
    for old, new in edits:
        candidate = one(candidate, old, new)
    # Reverse using anchors; the removed argument and barrier need unique insert points.
    reversed_candidate = candidate
    for old, new in reversed(edits):
        if not new:  # Removed sole threadgroup argument.
            anchor = '        device float *scores,\n'
            reversed_candidate = one(reversed_candidate, anchor, anchor + old)
        else:
            reversed_candidate = one(reversed_candidate, new, old)
    if reversed_candidate != original:
        raise ValueError('candidate contains edits beyond the stated transformations')

    native += build.REGISTER_ONLY_MARKER.encode() + (ROOT / 'metal' / build.REGISTER_ONLY_SHADER).read_bytes()
    combined = native + b'\n#line 1 "indexer_exact_direct_q_probe.metal"\n' + candidate.encode() + b'\n'
    out.mkdir(parents=True)
    (out / 'all.metal').write_bytes(combined)
    (out / 'candidate.metal').write_text(candidate)
    inputs = [ROOT / 'tools/bench_indexer_exact_direct_q.m', ROOT / 'tools/indexer_long_topk_probe.inc',
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
    with (out/'results.jsonl').open('x') as result, (out/'gpu.stderr').open('x') as stderr:
        subprocess.run([str(out/'probe'),str(out/'probe.metallib'),str(lib)],stdout=result,stderr=stderr,check=True)
    print('GPU correctness and timing complete: '+str(out/'results.jsonl'))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print('exact-direct-q-probe:',exc,file=sys.stderr)
        sys.exit(1)
