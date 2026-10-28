#!/usr/bin/env python3
"""Build an isolated exact-indexer bit-mask skip component probe.

The candidate is derived from the current exact shader by reversible edits:
only a lane-uniform mask predicate skips rejected simdgroup/query arithmetic.
No runtime selector or deployed artifact is modified. GPU execution requires
--run and an explicitly nominated deployed metallib.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import shutil
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
    p.add_argument('--mask-samples', type=Path, help='optional extracted historical layer 24 mask samples')
    p.add_argument('--case-id', type=int, choices=range(13), help='run exactly one fixed fixture case')
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
    # Each simdgroup owns four keys, aligned within one eight-key mask bit.
    # Its predicate is lane-uniform; only the existing simd_sum is skipped.
    # All threadgroup barriers and kept-score arithmetic remain unchanged.
    edits = [
        ('kernel void kernel_myds_indexer_score_chunk_exact(', 'kernel void probe_indexer_mask_skip('),
        ('constant ds4_metal_args_glm_indexer_score_chunk & args,', 'constant probe_indexer_mask_skip_args & args,'),
        ('        device float *scores,\n', '        device float *scores,\n        device const uint *mask_words,\n'),
        ('    float acc = 0.0f;\n    const uint q_vec_count = tile_tokens * 32u;',
         """    uint allowed_queries = 0u;
    if (sg_row0 < args.n_rows) {
        const uint block = sg_row0 >> 3;
        for (uint t = 0u; t < tile_tokens; ++t) {
            const uint word = mask_words[(uint64_t)(tok_base + t) * args.mask_stride + (block >> 5)];
            if (word & (1u << (block & 31u))) allowed_queries |= 1u << t;
        }
    }
    float acc = 0.0f;
    const uint q_vec_count = tile_tokens * 32u;"""),
        ('            if (sg_row0 >= vis_t) continue;\n',
         '            if (sg_row0 >= vis_t || !(allowed_queries & (1u << t))) continue;\n'),
    ]
    candidate = original
    for old, new in edits:
        candidate = one(candidate, old, new)
    restored = candidate
    for old, new in reversed(edits):
        restored = one(restored, new, old)
    if restored != original:
        raise ValueError('candidate contains edits beyond declared mask skip')

    candidate = '''struct probe_indexer_mask_skip_args {\n    uint n_rows, n_head, head_dim, cache_f16;\n    float scale;\n    uint n_tokens, causal_start, causal_ratio, window, mask_stride;\n};\n''' + candidate

    deployed_candidate = helper.extract(source, 'kernel void kernel_myds_indexer_score_chunk_mask_skip(')
    probe_candidate = helper.extract(candidate, 'kernel void probe_indexer_mask_skip(')
    expected_candidate = probe_candidate.replace('probe_indexer_mask_skip(',
        'kernel_myds_indexer_score_chunk_mask_skip(').replace('probe_indexer_mask_skip_args',
        'myds_indexer_mask_skip_args')
    if deployed_candidate != expected_candidate:
        raise ValueError('runtime and verified probe shader bodies differ')

    native += build.REGISTER_ONLY_MARKER.encode() + (ROOT / 'metal' / build.REGISTER_ONLY_SHADER).read_bytes()
    combined = native + b'\n#line 1 "indexer_mask_skip_probe.metal"\n' + candidate.encode() + b'\n'
    out.mkdir(parents=True)
    (out / 'all.metal').write_bytes(combined)
    (out / 'candidate.metal').write_text(candidate)
    inputs = [ROOT / 'tools/bench_indexer_mask_skip.m', ROOT / 'tools/indexer_mask_skip_topk.inc',
              ROOT / 'src/myds_v41_gpu.c', ROOT / 'src/myds_gguf.c', ROOT / 'src/myds_log.c',
              ROOT / 'src/myds_configfile.c', ROOT / 'src/myds_metal.m'] + sorted((ROOT/'include').glob('*.h'))
    identity = {'baseline_exact_shader_sha256': sha(original.encode()),
                'candidate_sha256': sha(candidate.encode()), 'runtime_candidate_sha256': sha(deployed_candidate.encode()),
                'runtime_probe_shader_exact': True, 'native_prefix_sha256': sha(native),
                'combined_sha256': sha(combined), 'reverse_exact': True,
                'production_runtime_connected': True, 'runtime_scope': 'isolated production GPU API and filter; no full model', 'shader_files': shaders,
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
        if f.name == 'bench_indexer_mask_skip.m':
            flags += ['-Werror']
        if f.name == 'myds_v41_gpu.c':
            flags += ['-DMYDS_INDEXER_MASK_SKIP_TESTING']
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
    shutil.copyfile(out/'probe.metallib', out/'myds.metallib')
    if not args.run:
        print('build complete; no GPU submitted')
        return
    procs = subprocess.check_output(['ps','-axo','pid=,comm='],text=True)
    for line in procs.splitlines():
        fields = line.strip().split(None,1)
        if len(fields)==2 and (Path(fields[1]).name in {'myds','myglm','ds4','llama-server','ollama'} or Path(fields[1]).name.startswith('bench_')):
            raise RuntimeError('inference/benchmark process conflicts with GPU test: '+line)
    lib=args.deployed_metallib.resolve(strict=True)
    command=[str(out/'probe'),str(out/'myds.metallib'),str(lib)]
    if args.correctness_only:
        command.append('--correctness-only')
    env = os.environ.copy()
    if args.case_id is not None:
        env['MYDS_MASK_PROBE_CASE'] = str(args.case_id)
    if args.mask_samples:
        sample_dir = args.mask_samples.resolve(strict=True)
        env['MYDS_MASK_PROBE_SAMPLES'] = str(sample_dir)
    with (out/'results.jsonl').open('x') as result, (out/'gpu.stderr').open('x') as stderr:
        subprocess.run(command,stdout=result,stderr=stderr,check=True,env=env)
    print('GPU probe complete: '+str(out/'results.jsonl'))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print('indexer-mask-skip-probe:',exc,file=sys.stderr)
        sys.exit(1)
