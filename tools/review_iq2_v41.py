#!/usr/bin/env python3
"""CPU-only source/raw reviewer for bench_iq2_v41; never loads Metal."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import statistics

FROZEN_SHA = '8344379eda8be1389fb0722ff8c4e7e1eb7a0667afeeda4208bed5c7bea27a75'
ACTUAL = 'kernel_mul_mv_id_iq2_xxs_pair_swiglu_popcnt_f32'
FROZEN = 'probe_iq2_v41_frozen_f32'
ALLOWED = {'kernel_myds_iq2_v41_shape64_f32', 'kernel_myds_iq2_v41_addr32_f32', 'kernel_myds_iq2_v41_shape32_f32'}
RENAMES = {
 'ds4_metal_args_mul_mv_id': 'myds_iq2_v41_args',
 'ds4_metal_dsv4_moe_swiglu_weight_args': 'myds_iq2_v41_act',
 'block_iq2_xxs': 'myds_iq2_v41_block', 'ds4_tp_owns_expert': 'myds_iq2_v41_owns_expert',
 'ds4_metal_kmask_iq2xs': 'myds_iq2_v41_kmask', 'ds4_metal_ksigns_iq2xs': 'myds_iq2_v41_signs',
 'ds4_metal_iq2xxs_grid': 'myds_iq2_v41_grid', 'FC_mul_mv_nsg': 'MYDS_IQ2_V41_NSG',
 'N_R0_IQ2_XXS': 'MYDS_IQ2_V41_NR', 'QK_K': 'MYDS_IQ2_V41_QK',
 'ds4_iq2_pair_swiglu_popcnt_impl': 'myds_iq2_v41_frozen_impl',
}

def require(ok, why):
    if not ok:
        raise ValueError(why)

def block(s, marker):
    start = s.index(marker)
    pos = s.index('{', start)
    depth = 1
    end = pos + 1
    while depth:
        depth += (s[end] == '{') - (s[end] == '}')
        end += 1
    return s[start:end]

def source_review(repo):
    original = (repo / 'metal/moe.metal').read_text()
    candidate = (repo / 'metal/myds_iq2_v41.metal').read_text()
    ref = block(original, 'static inline void ds4_iq2_pair_swiglu_popcnt_impl(')
    require(hashlib.sha256(ref.encode()).hexdigest() == FROZEN_SHA, 'production helper changed: refreeze required')
    frozen = block(candidate, 'static inline void myds_iq2_v41_frozen_impl(')
    for old, new in RENAMES.items():
        frozen = frozen.replace(new, old)
    require(frozen == ref, 'frozen clone differs beyond identifier-only substitution')
    for old in ['ds4_metal_args_mul_mv_id', 'ds4_metal_dsv4_moe_swiglu_weight_args', 'block_iq2_xxs']:
        new = RENAMES[old]
        copied = block(candidate, 'struct '+new+' {')
        for origin, renamed in RENAMES.items():
            copied = copied.replace(renamed, origin)
        require(copied == block(original, 'struct '+old+' {'), 'ABI declaration mismatch')
    require(block(candidate, 'static inline bool myds_iq2_v41_owns_expert(').replace('myds_iq2_v41_owns_expert','ds4_tp_owns_expert') == block(original,'static inline bool ds4_tp_owns_expert('), 'frozen ownership helper differs')
    for old in ['ds4_metal_kmask_iq2xs', 'ds4_metal_ksigns_iq2xs', 'ds4_metal_iq2xxs_grid']:
        new = RENAMES[old]
        for name, text in [(old, original), (new, candidate)]:
            val = block(text, 'static constant ' + ('ulong ' if 'grid' in name else 'uchar ') + name)
            if name == old:
                table = val
            else:
                require(val.replace(new, old) == table, 'lookup table mismatch')
    # Compare the entire FP hot loop, allowing only the targeted row-stride
    # substitutions. No FMA, reordering, explicit unroll, or new table loads.
    loop_start = '    float yl[32];'
    loop_end = '    device float *dst_gate_f32 ='
    ref_renamed = ref
    for old, new in RENAMES.items():
        ref_renamed = ref_renamed.replace(old, new)
    ref_loop = ref_renamed[ref_renamed.index(loop_start):ref_renamed.index(loop_end)]
    ref_fp = re.findall(r'^.*(?:simd_sum|reduced_gate\[|reduced_up\[|const float gate =|const float up =|float g =|float u =|const float c =|route_weight =|g = min|u = clamp|const float silu|dst_mid_f32\[).*$' , ref_renamed, re.M)
    for mode in ['shape64', 'addr32', 'shape32']:
        body = block(candidate, f'static inline void myds_iq2_v41_{mode}_impl(')
        work = body[body.index(loop_start):body.index(loop_end)]
        work = work.replace('(uint)args.nb01', 'args.nb01').replace('1320u / 2', 'args.nb01 / 2')
        require(work == ref_loop, mode + ' FP hot loop changed')
        fp = re.findall(r'^.*(?:simd_sum|reduced_gate\[|reduced_up\[|const float gate =|const float up =|float g =|float u =|const float c =|route_weight =|g = min|u = clamp|const float silu|dst_mid_f32\[).*$' , body, re.M)
        require(fp == ref_fp, mode + ' reduction/activation FP expressions changed')
        require('if ((uint)i02 >= 384u) return;' in body, mode + ' invalid ID protection missing')
        require('if (args.tp_world > 1 || args.tp_expert_base != 0) return;' in body, mode + ' TP protection missing')
    for nsg in [2, 4]:
        for tg in range(2304 // (4 * nsg)):
            for sg in range(nsg):
                row = (tg * nsg + sg) * 4
                require(row + 3 < 2304, 'partial row')
                for lane in range(32):
                    require(len(range(lane, 160, 32)) == 5, 'K iteration count')
    require(384 * 3041280 == 1167851520 < 2**32, 'weight uint32 range')
    require(383 * 3041280 + 2303 * 1320 + 1319 == 1167851519, 'maximum weight offset')
    return {'source_frozen_exact': True, 'all_fp_loops_and_expressions_exact': True,
            'lookup_tables_exact': True, 'all_rows_full': True,
            'tensor_bytes': 1167851520, 'maximum_offset': 1167851519, 'gpu_executed': False}

def raw_review(raw, stdout):
    manifest = json.loads((raw / 'artifact-manifest.json').read_text())
    identity = manifest['identity']
    require(identity['actual_production_kernel'] == ACTUAL, 'wrong oracle')
    require(identity['frozen_symbol'] == FROZEN and identity['frozen_helper_sha256'] == FROZEN_SHA, 'wrong freeze')
    require(identity['nsg'] in [2, 4] and identity['fc600'] == identity['nsg'], 'wrong NSG')
    require(identity['dim'] == 5120 and identity['width'] == 2304 and identity['experts'] == 384 and identity['active'] == 6 and identity['layers'] == 40, 'wrong shape')
    candidates = identity['candidates']
    require(candidates and set(candidates) <= ALLOWED and len(candidates) == len(set(candidates)), 'invalid candidates')
    files = {}
    for row in manifest['files']:
        name = row['path']
        require(Path(name).name == name and name not in files, 'invalid or duplicate file')
        data = (raw / name).read_bytes()
        require(len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256'], 'payload SHA/size mismatch: ' + name)
        files[name] = data
    for trial in range(identity['trials']):
        prefix = f'parity-{trial:02d}'
        ids = struct.unpack('<240i', files[prefix + '-ids.i32'])
        require(len(set(ids)) == 240 and all(0 <= i < 384 for i in ids), 'IDs not valid/disjoint')
        for field in ['x', 'weights']:
            data = files[f'{prefix}-{field}.f32']
            require(all(math.isfinite(x[0]) for x in struct.iter_unpack('<f', data)), 'input not finite')
        for field in ['mid'] + (['out'] if identity['pair'] else []):
            oracle = files[f'{prefix}-{ACTUAL}-{field}.f32']
            require(len(oracle) == (40*13824*4 if field == 'mid' else 40*5120*4), 'incomplete output')
            require(all(math.isfinite(x[0]) for x in struct.iter_unpack('<f', oracle)), 'oracle not finite')
            for symbol in [FROZEN] + candidates:
                require(files[f'{prefix}-{symbol}-{field}.f32'] == oracle, 'output bytes differ: ' + symbol)
                require(files[f'{prefix}-{symbol}-{field}.guards'] == b'\xa5'*128, 'output guards differ: ' + symbol)
            require(files[f'{prefix}-{ACTUAL}-{field}.guards'] == b'\xa5'*128, 'oracle guards differ')
    for candidate in candidates:
        require(set(files[f'negative-ids-{candidate}-mid.raw']) == {0xa5}, 'invalid ID writes')
        require(set(files[f'negative-tp-{candidate}-mid.raw']) == {0xa5}, 'split TP writes')
    require(struct.unpack('<240i', files['negative-ids.i32']) == tuple([-1,384,-2147483648,2147483647,-2,385]*40), 'invalid ID fixture')
    records = [json.loads(line) for line in stdout.read_text().splitlines() if line.startswith('{')]
    require(sum(r.get('kind') == 'complete' for r in records) == 1, 'not complete')
    require([r for r in records if r.get('kind') == 'identity'] == [identity], 'identity mismatch')
    parity = [r for r in records if r.get('kind') == 'parity']
    require(len(parity) == identity['trials']*(len(candidates)+2), 'missing parity records')
    for row in parity:
        require(row['completed_cb'] and row['full_f32_bit_exact'] and row['pair_full_f32_bit_exact'] and row['guards'] and row['finite'] and row['release_sg_su_poison'], 'parity gate failure')
    summaries = []
    for candidate in candidates:
        medians = []
        for arm in ['A1', 'B', 'A2']:
            timing = [r for r in records if r.get('kind') == 'timing' and r['candidate'] == candidate and r['arm'] == arm]
            require(len(timing) == identity['samples'] + 1 and [r['sample'] for r in timing] == list(range(identity['samples']+1)), 'missing timing sample')
            symbol = candidate if arm == 'B' else ACTUAL
            for row in timing:
                require(row['kernel'] == symbol and row['warmup'] == (row['sample']==0) and row['completed_cb'] and row['full_f32_bit_exact'] and row['production_math_oracle_validated'], 'timing gate')
                recomputed = (row['gpu_end_s']-row['gpu_start_s'])*1e6/40
                require(recomputed > 0 and math.isclose(recomputed, row['us_per_layer'], rel_tol=1e-10), 'invalid CB timing')
            medians.append(statistics.median(r['us_per_layer'] for r in timing if not r['warmup']))
            for field in ['mid'] + (['out'] if identity['pair'] else []):
                require(files[f'timing-{candidate}-{arm}-{field}.f32'] == files[f'timing-{candidate}-A1-{field}.f32'], 'timing output mismatch')
                require(files[f'timing-{candidate}-{arm}-{field}.guards'] == b'\xa5'*128, 'timing guard mismatch')
        a1, b, a2 = medians
        summaries.append({'candidate': candidate, 'a1_us': a1, 'b_us': b, 'a2_us': a2,
                          'control_drift_percent': (a2/a1-1)*100,
                          'time_saved_percent': (1-b/((a1+a2)/2))*100,
                          'conservative_time_saved_percent': (1-b/min(a1,a2))*100,
                          'fullmodel_speedup_proven': False})
    return {'payload_files': len(files), 'payload_bytes': sum(map(len,files.values())),
            'full_f32_bytes_match': True, 'pair': identity['pair'], 'summaries': summaries}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--raw-root', type=Path)
    p.add_argument('--stdout', type=Path)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    result = {'source': source_review(a.repo)}
    if a.raw_root:
        require(a.stdout is not None, '--stdout required for raw review')
        result['raw'] = raw_review(a.raw_root, a.stdout)
    text = json.dumps(result, indent=2, sort_keys=True) + '\n'
    if a.output:
        a.output.write_text(text)
    print(text, end='')

if __name__ == '__main__':
    main()
