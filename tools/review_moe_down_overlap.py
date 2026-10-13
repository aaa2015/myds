#!/usr/bin/env python3
"""CPU-only independent A1/B/A2 raw and timing review for native overlap probe."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct

FIELDS = {'mid': 13824, 'moe': 5120, 'shared': 5120, 'sum_bf16': 5120}

def require(ok, message):
    if not ok:
        raise ValueError(message)

def finite(data):
    return all(math.isfinite(v[0]) for v in struct.iter_unpack('<f', data))

def read_arm(directory, stdout, expected_mode):
    manifest = json.loads((directory/'artifact-manifest.json').read_text())
    identity = manifest['identity']
    require(identity['mode'] == identity['actual_mode'] == expected_mode, 'mode not adopted')
    require(identity['q2_v41_actual_mode'] == 1 and identity['iq2_v41_actual_mode'] == identity['dense_q4_v41_actual_mode'] == 0, 'wrong baseline providers')
    require(identity['encoder_dispatch_type'] == 'concurrent' and identity['kprof'] is False, 'wrong encoder/profiler')
    require(identity['tensor_bytes_total'] == 3828695040 and identity['fullmodel'] is False, 'wrong component scope')
    files = {}
    for row in manifest['files']:
        name = row['path']
        require(Path(name).name == name and name not in files, 'invalid manifest path')
        payload = (directory/name).read_bytes()
        require(len(payload) == row['bytes'] and hashlib.sha256(payload).hexdigest() == row['sha256'], 'SHA/size mismatch: '+name)
        files[name] = payload
    for case in range(identity['cases']):
        ids = struct.unpack('<6i', files[f'input-case-{case:02d}-ids.i32'])
        require(ids[:2] == (0,383) and len(set(ids)) == 6 and all(0 <= i < 384 for i in ids), 'invalid routed IDs')
        for field in ['x', 'shared', 'weights']:
            require(finite(files[f'input-case-{case:02d}-{field}.f32']), 'input not finite')
        require(all(v[0] & 0xffff == 0 for v in struct.iter_unpack('<I',files[f'input-case-{case:02d}-shared.f32'])), 'shared MID not BF16 grid')
        for sample in range(identity['samples']+1):
            for field, count in FIELDS.items():
                name = f'sample-{sample:02d}-case-{case:02d}-{field}'
                payload = files[name+'.f32']
                require(len(payload) == 4*count and finite(payload), 'incomplete/nonfinite output')
                require(files[name+'.guards'] == b'\xa5'*128, 'guard corruption')
                require(payload == files[f'sample-00-case-{case:02d}-{field}.f32'], 'output changed across samples')
                if field in ['shared','sum_bf16']:
                    require(all(v[0] & 0xffff == 0 for v in struct.iter_unpack('<I',payload)), 'BF16 epilogue differs')
    if expected_mode:
        for case in range(2):
            for field in ['mid','moe','shared']:
                require(set(files[f'negative-{case}-{field}.raw']) == {0xa5}, 'native rejected request wrote buffer')
    records = [json.loads(line) for line in stdout.read_text().splitlines() if line.startswith('{')]
    require([r for r in records if r.get('kind') == 'identity'] == [identity], 'stdout identity differs')
    complete = [r for r in records if r.get('kind') == 'complete']
    require(len(complete) == 1 and complete[0]['cleanup_modes_reset'] is True, 'completion/cleanup missing')
    negatives = [r for r in records if r.get('kind') == 'negative']
    require(len(negatives) == (3 if expected_mode else 1), 'negative controls missing')
    require(negatives[0]['case'] == 'invalid_cpu_supplied_id' and negatives[0]['native_api_calls'] == negatives[0]['dispatch_delta'] == 0, 'CPU ID precheck misrepresented')
    if expected_mode:
        require({r['case'] for r in negatives[1:]} == {'short_shared_output','shared_output_alias_moe_output'}, 'wrong native negatives')
        for row in negatives[1:]:
            require(row['dispatch_delta'] == row['encoded_delta'] == 0 and row['requests_delta'] == row['failures_delta'] == 1 and row['buffers_all_poison'], 'native negative encoded/wrote')
    samples = [r for r in records if r.get('kind') == 'sample']
    require([r['sample'] for r in samples] == list(range(identity['samples']+1)), 'missing sample')
    pairs = identity['cases']*identity['repetitions']
    for row in samples:
        require(row['warmup'] == (row['sample']==0) and row['actual_mode'] == expected_mode and row['pairs'] == pairs, 'wrong sample protocol')
        require(row['dispatch_delta'] == 4*pairs and row['commit_delta'] == 1 and row['completed_cb'], 'actual dispatch/CB gate')
        require(row['all_outputs_finite'] and row['all_guards'] and row['release_scratch_poison'] and row['full_outputs_match_warmup'], 'sample correctness gate')
        gpu_us = (row['gpu_end_s']-row['gpu_start_s'])*1e6/pairs
        require(gpu_us > 0 and math.isclose(gpu_us,row['gpu_us_per_pair'],rel_tol=1e-10), 'GPUspan wrong')
        require(math.isfinite(row['wall_us_per_pair']) and row['wall_us_per_pair'] > 0, 'wall time invalid')
        for prefix, expected in [('overlap', pairs if expected_mode else 0), ('q2',pairs)]:
            before, after = row[prefix+'_before'], row[prefix+'_after']
            require(after['requests']-before['requests'] == expected and after['encoded']-before['encoded'] == expected and after['failures'] == before['failures'], 'native adoption counter gate')
    gpu = statistics.median(r['gpu_us_per_pair'] for r in samples if not r['warmup'])
    wall = statistics.median(r['wall_us_per_pair'] for r in samples if not r['warmup'])
    summaries = [r for r in records if r.get('kind') == 'summary']
    require(len(summaries)==1 and math.isclose(summaries[0]['gpu_median_us_per_pair'],gpu,rel_tol=1e-10) and math.isclose(summaries[0]['wall_median_us_per_pair'],wall,rel_tol=1e-10), 'median wrong')
    return {'identity':identity,'files':files,'gpu_us':gpu,'wall_us':wall,'samples':samples}

def review(a1, b, a2):
    ids = [arm['identity'] for arm in [a1,b,a2]]
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
            'payload_files':sum(len(x['files']) for x in [a1,b,a2]),
            'payload_bytes':sum(sum(map(len,x['files'].values())) for x in [a1,b,a2])}
    for metric in ['gpu_us','wall_us']:
        av,bv,cv=a1[metric],b[metric],a2[metric]
        result[metric]={'a1':av,'b':bv,'a2':cv,'control_drift_percent':(cv/av-1)*100,
                       'time_saved_percent':(1-bv/((av+cv)/2))*100,
                       'conservative_time_saved_percent':(1-bv/min(av,cv))*100}
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
