#!/usr/bin/env python3
"""CPU-only independent A1/B/A2 raw and timing review for native overlap probe."""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path
import statistics
import struct

FIELDS = {'mid': 13824, 'moe': 5120, 'shared': 5120, 'sum_bf16': 5120}

def require(ok, message):
    if not ok:
        raise ValueError(message)

def finite(data):
    return all(math.isfinite(v[0]) for v in struct.iter_unpack('<f', data))

def decode(text):
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate JSON key: '+key)
            result[key] = value
        return result
    def invalid_constant(value):
        raise ValueError('nonfinite JSON constant: '+value)
    return json.loads(text, object_pairs_hook=unique_pairs, parse_constant=invalid_constant)


def integer(value, low, high, why):
    require(type(value) is int and low <= value <= high, why)
    return value


def real(value, why):
    require(type(value) in (int, float) and math.isfinite(value), why)
    return value


def digest(value, why):
    require(type(value) is str and re.fullmatch(r'[0-9a-f]{64}', value), why)
    return value


def flag(value, expected):
    # Objective-C @(C comparison) serializes as integer0/1; @YES/BOOL is
    # JSON boolean. Accept exactly these two emitted representations.
    return type(value) in (bool, int) and value in (0, 1) and bool(value) is expected


def true_fields(row, keys, why):
    require(all(row.get(key) is True for key in keys), why)


def read_arm(directory, stdout, expected_mode):
    manifest = decode((directory/'artifact-manifest.json').read_text())
    require(type(manifest.get('schema_version')) is int and manifest['schema_version'] == 1, 'unsupported raw schema')
    identity = manifest['identity']
    require(type(identity.get('mode')) is int and type(identity.get('actual_mode')) is int and
            identity['mode'] == identity['actual_mode'] == expected_mode, 'mode not adopted')
    for key, expected in [('q2_v41_actual_mode', 1), ('iq2_v41_actual_mode', 0), ('dense_q4_v41_actual_mode', 0)]:
        require(type(identity.get(key)) is int and identity[key] == expected, 'wrong baseline providers')
    require(identity['encoder_dispatch_type'] == 'concurrent' and identity['kprof'] is False, 'wrong encoder/profiler')
    require(type(identity['tensor_bytes_total']) is int and identity['tensor_bytes_total'] == 3828695040 and
            identity['fullmodel'] is False, 'wrong component scope')
    samples = integer(identity['samples'], 3, 31, 'invalid formal sample count')
    cases = integer(identity['cases'], 1, 32, 'invalid case count')
    repetitions = integer(identity['repetitions'], 1, 32, 'invalid repetition count')
    integer(identity['seed'], 0, (1 << 64)-1, 'invalid seed')
    integer(identity['model_bytes'], 1, 1 << 50, 'invalid model length')
    for key in ['model_mtime_ns', 'model_inode', 'model_device']:
        integer(identity[key], 0, 1 << 100, 'invalid model metadata')
    # External libraries/binary/GGUF are not archived raw payloads here. These
    # are declared identities only; the window controller binds their bytes.
    for key in ['actual_library_sha256', 'binary_sha256']:
        digest(identity[key], 'invalid declared digest: '+key)
    for key in ['actual_library_path', 'expected_library_path']:
        require(type(identity[key]) is str and Path(identity[key]).is_absolute(), 'invalid library path')
    require(Path(identity['actual_library_path']).resolve() == Path(identity['expected_library_path']).resolve(),
            'actual library path differs from selected library')
    names = ['blk.0.ffn_gate_exps.weight', 'blk.0.ffn_up_exps.weight', 'blk.0.ffn_down_exps.weight', 'blk.0.ffn_down_shexp.weight']
    sizes = [1167851520, 1167851520, 1486356480, 6635520]
    types = [16, 16, 10, 12]
    require(type(identity['tensors']) is list and len(identity['tensors']) == 4, 'missing tensor identities')
    require([t.get('name') for t in identity['tensors']] == names, 'tensor identity order/name differs')
    for index, tensor in enumerate(identity['tensors']):
        require(type(tensor.get('bytes')) is int and tensor['bytes'] == sizes[index] and
                type(tensor.get('type')) is int and tensor['type'] == types[index], 'tensor shape/type differs')
        offset = integer(tensor['absolute_offset'], 0, identity['model_bytes'], 'invalid tensor offset')
        require(offset + sizes[index] <= identity['model_bytes'], 'tensor exceeds model')
        digest(tensor['sha256'], 'invalid declared tensor SHA')
    expected_env = {'MYDS_MOE_DOWN_OVERLAP': str(expected_mode), 'MYDS_MOE_IQ2_V41': '0',
                    'MYDS_DENSE_Q4_V41': '0', 'MYDS_MOE_Q2_V41': '1', 'MYDS_ROUTE_SIMD_SHUFFLE': '1',
                    'MYDS_ATTN_PACKED_REGISTER': '1', 'MYDS_MOE_IQ2_POPCNT_SIGN': '1',
                    'MYDS_MOE_IQ2_CONSTANT_LUT': '0', 'MYDS_MOE_NSG': '2', 'MYDS_MOE_DOWN_NSG': '2',
                    'MYDS_DENSE_Q4_NR0': '2', 'MYDS_DENSE_Q4_NSG': '4', 'MYDS_MOE_Q2_SUM_CACHE': '0',
                    'MYDS_MOE_Q2_SUM_CACHE_FUSED': '0', 'MYDS_MOE_MM_MIN_TOKENS': '0',
                    'MYDS_METAL_CONCURRENT': '1', 'MYDS_KPROF': '0', 'MYDS_MOE_STEP_PROF': '0',
                    'MYDS_NO_RESIDENCY_HB': '1'}
    require(type(identity['environment']) is dict and
            all(identity['environment'].get(key) == value for key, value in expected_env.items()),
            'component environment contract differs')
    files = {}
    for row in manifest['files']:
        name = row['path']
        require(type(name) is str and name not in ('.', '..') and Path(name).name == name and
                name not in files, 'invalid manifest path')
        integer(row['bytes'], 0, 1 << 40, 'invalid file size')
        digest(row['sha256'], 'invalid file digest')
        data = (directory/name).read_bytes()
        require(len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256'], 'SHA/size mismatch: '+name)
        files[name] = data
    expected_files = set()

    def payload(name, length, kind=None):
        expected_files.add(name)
        data = files[name]
        require(len(data) == length, 'incomplete payload: '+name)
        if kind == 'f32':
            require(finite(data), 'nonfinite payload: '+name)
        elif kind == 'guard':
            require(data == b'\xa5'*length, 'poison/guard corruption: '+name)
        return data

    for case in range(cases):
        ids = struct.unpack('<6i', payload(f'input-case-{case:02d}-ids.i32', 24))
        require(ids[:2] == (0,383) and len(set(ids)) == 6 and all(0 <= i < 384 for i in ids), 'invalid routed IDs')
        for field, size in [('x', 20480), ('shared', 9216), ('weights', 24)]:
            payload(f'input-case-{case:02d}-{field}.f32', size, 'f32')
        require(all(v[0] & 0xffff == 0 for v in struct.iter_unpack('<I',files[f'input-case-{case:02d}-shared.f32'])), 'shared MID not BF16 grid')
        for sample in range(samples+1):
            for field, count in FIELDS.items():
                name = f'sample-{sample:02d}-case-{case:02d}-{field}'
                data = payload(name+'.f32', 4*count, 'f32')
                payload(name+'.guards', 128, 'guard')
                require(data == files[f'sample-00-case-{case:02d}-{field}.f32'], 'output changed across samples')
                if field in ['shared','sum_bf16']:
                    require(all(v[0] & 0xffff == 0 for v in struct.iter_unpack('<I', data)), 'BF16 epilogue differs')
    if expected_mode:
        for case in range(2):
            for field in ['mid','moe','shared']:
                payload(f'negative-{case}-{field}.raw', 4*FIELDS[field]+128, 'guard')
    records = [decode(line) for line in stdout.read_text().splitlines() if line.startswith('{')]
    require(all(type(row) is dict and row.get('kind') in {'identity', 'negative', 'sample', 'summary', 'complete'}
                for row in records), 'unknown/nonobject record')
    require([r for r in records if r.get('kind') == 'identity'] == [identity], 'stdout identity differs')
    complete = [r for r in records if r.get('kind') == 'complete']
    require(len(complete) == 1 and complete[0].get('cleanup_modes_reset') is True and
            complete[0].get('fullmodel') is False and type(complete[0].get('raw_files')) is int and
            complete[0]['raw_files'] == len(files), 'completion/cleanup missing')
    negatives = [r for r in records if r.get('kind') == 'negative']
    expected_negative = ['invalid_cpu_supplied_id'] + (['short_shared_output', 'shared_output_alias_moe_output'] if expected_mode else [])
    require([r.get('case') for r in negatives] == expected_negative, 'missing/duplicate negative controls')
    require(negatives[0].get('native_api_calls') == negatives[0].get('dispatch_delta') == 0 and
            type(negatives[0].get('native_api_calls')) is int and type(negatives[0].get('dispatch_delta')) is int,
            'CPU ID precheck misrepresented')
    if expected_mode:
        for row in negatives[1:]:
            for key, value in [('dispatch_delta', 0), ('encoded_delta', 0), ('requests_delta', 1), ('failures_delta', 1)]:
                require(type(row.get(key)) is int and row[key] == value, 'native negative encoded/wrote')
            require(row.get('buffers_all_poison') is True, 'native negative wrote buffers')
    sample_rows = [r for r in records if r.get('kind') == 'sample']
    require([r.get('sample') for r in sample_rows] == list(range(samples+1)) and
            all(type(r.get('sample')) is int for r in sample_rows), 'missing/duplicate sample')
    pairs = cases*repetitions
    previous = {'overlap': {'requests': 0, 'encoded': 0, 'failures': 0},
                'q2': {'requests': 0, 'encoded': 0, 'failures': 0}}
    for row in sample_rows:
        require(flag(row.get('warmup'), row['sample']==0) and type(row.get('actual_mode')) is int and
                row['actual_mode'] == expected_mode and type(row.get('pairs')) is int and row['pairs'] == pairs,
                'wrong sample protocol')
        require(type(row.get('dispatch_delta')) is int and row['dispatch_delta'] == 4*pairs and
                type(row.get('commit_delta')) is int and row['commit_delta'] == 1, 'actual dispatch/CB gate')
        true_fields(row, ['completed_cb', 'all_outputs_finite', 'all_guards', 'release_scratch_poison',
                          'full_outputs_match_warmup'], 'sample correctness/completion gate')
        start = real(row['gpu_start_s'], 'nonfinite GPU start')
        end = real(row['gpu_end_s'], 'nonfinite GPU end')
        us = real(row['gpu_us_per_pair'], 'nonfinite GPU duration')
        gpu_us = (end-start)*1e6/pairs
        require(start >= 0 and end > start and us > 0 and math.isfinite(gpu_us) and
                math.isclose(gpu_us, us, rel_tol=1e-10), 'GPUspan wrong')
        require(real(row['wall_us_per_pair'], 'wall time invalid') > 0, 'wall time invalid')
        for prefix, expected in [('overlap', pairs if expected_mode else 0), ('q2', pairs)]:
            before, after = row[prefix+'_before'], row[prefix+'_after']
            require(type(before) is dict and type(after) is dict, 'missing native counter snapshot')
            for key in ['requests', 'encoded', 'failures']:
                integer(before.get(key), 0, 1 << 64, 'invalid before counter')
                integer(after.get(key), 0, 1 << 64, 'invalid after counter')
            require(before == previous[prefix], 'counter snapshots not continuous')
            require(after['requests']-before['requests'] == expected and after['encoded']-before['encoded'] == expected and
                    after['failures'] == before['failures'], 'native adoption counter gate')
            previous[prefix] = after
    gpu = statistics.median(r['gpu_us_per_pair'] for r in sample_rows if not r['warmup'])
    wall = statistics.median(r['wall_us_per_pair'] for r in sample_rows if not r['warmup'])
    summaries = [r for r in records if r.get('kind') == 'summary']
    require(len(summaries) == 1, 'missing/duplicate summary')
    summary = summaries[0]
    require(math.isclose(real(summary['gpu_median_us_per_pair'], 'nonfinite GPU median'), gpu, rel_tol=1e-10) and
            math.isclose(real(summary['wall_median_us_per_pair'], 'nonfinite wall median'), wall, rel_tol=1e-10), 'median wrong')
    require(type(summary.get('mode')) is int and type(summary.get('actual_mode')) is int and
            summary['mode'] == summary['actual_mode'] == expected_mode and
            type(summary.get('formal_samples')) is int and summary['formal_samples'] == samples and
            summary.get('gpu_overlap_proven') is False and summary.get('fullmodel_speedup_proven') is False,
            'summary scope differs')
    # src/myds_metal.m hist_add is opt-in independently of KPROF: the
    # presence of MYDS_DISPATCH_HIST enables it (even a value of "0"). The
    # aggregate dispatch counter above is always active. A disabled dump
    # contains exactly this empty header; it is not per-kernel evidence.
    hist_name = 'dispatch-hist.txt'
    expected_files.add(hist_name)
    hist = files[hist_name]
    histogram_available = hist != b'--- DISPATCH BY KERNEL (0 kinds, 0 total) ---\n'
    if histogram_available:
        # Enabled histograms must be the complete native dump, not a subset
        # accepted by a multiline search. Equal per-kernel counts imply 25%.
        lines = hist.decode('ascii').splitlines(keepends=True)
        count = (samples+1)*pairs
        require(len(lines) == 5 and lines[0] == f'--- DISPATCH BY KERNEL (4 kinds, {4*count} total) ---\n',
                'native dispatch histogram header differs')
        expected_names = {'fused_moe_swiglu_popcnt', 'fused_moe_sum6_q2_k_v41', 'v41_add_bf16',
                          'shared_down_overlap_q4_bf16' if expected_mode else 'gemv_q4_k_bf16'}
        expected_lines = {f'{name:<46} {count:10d}  {25.0:5.2f}%\n' for name in expected_names}
        require(set(lines[1:]) == expected_lines, 'native dispatch histogram differs')
    require(set(files) == expected_files, 'unexpected/unreviewed manifest payloads')
    return {'identity':identity,'files':files,'gpu_us':gpu,'wall_us':wall,'samples':sample_rows,
            'perkernel_histogram_available':histogram_available}

def review(a1, b, a2):
    ids = [arm['identity'] for arm in [a1,b,a2]]
    require(a1['perkernel_histogram_available'] == b['perkernel_histogram_available'] ==
            a2['perkernel_histogram_available'], 'histogram instrumentation differs across arms')
    for key in ['samples','cases','repetitions','seed','model_path','model_bytes','model_mtime_ns','model_inode','model_device','tensors','tensor_bytes_total','actual_library_sha256','binary_sha256']:
        require(ids[0][key] == ids[1][key] == ids[2][key], 'arm identity drift: '+key)
    for env in set(ids[0]['environment']) | set(ids[1]['environment']) | set(ids[2]['environment']):
        if env != 'MYDS_MOE_DOWN_OVERLAP':
            require(ids[0]['environment'].get(env) == ids[1]['environment'].get(env) == ids[2]['environment'].get(env), 'environment drift: '+env)
    for case in range(ids[0]['cases']):
        for field in ['x','shared','ids','weights']:
            name=f'input-case-{case:02d}-{field}.'+('i32' if field=='ids' else 'f32')
            require(a1['files'][name] == b['files'][name] == a2['files'][name], 'arm inputs differ')
        for sample in range(ids[0]['samples']+1):
            for field in FIELDS:
                name=f'sample-{sample:02d}-case-{case:02d}-{field}.f32'
                require(a1['files'][name] == b['files'][name] == a2['files'][name], 'full arm bytes differ: '+name)
    result={'all_full_outputs_bytes_equal':True,'arm_build_and_model_identity_equal':True,
            'component_only':True,'gpu_overlap_proven':False,'fullmodel_speedup_proven':False,
            'perkernel_histogram_available':a1['perkernel_histogram_available'],
            'payload_files':sum(len(x['files']) for x in [a1,b,a2]),
            'payload_bytes':sum(sum(map(len,x['files'].values())) for x in [a1,b,a2])}
    for metric in ['gpu_us','wall_us']:
        av,bv,cv=a1[metric],b[metric],a2[metric]
        result[metric]={'a1':av,'b':bv,'a2':cv,'control_drift_percent':(cv/av-1)*100,
                       'time_saved_percent':(1-bv/((av+cv)/2))*100,
                       'conservative_time_saved_percent':(1-bv/min(av,cv))*100}
        require(all(math.isfinite(value) for value in result[metric].values()), 'nonfinite derived comparison')
    return result

def main():
    p=argparse.ArgumentParser()
    for name in ['a1','b','a2']:
        p.add_argument('--'+name,type=Path,required=True)
        p.add_argument('--'+name+'-stdout',type=Path,required=True)
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    result=review(read_arm(a.a1,a.a1_stdout,0),read_arm(a.b,a.b_stdout,1),read_arm(a.a2,a.a2_stdout,0))
    text=json.dumps(result,indent=2,sort_keys=True)+'\n'
    if a.output:a.output.write_text(text)
    print(text,end='')

if __name__=='__main__':main()
