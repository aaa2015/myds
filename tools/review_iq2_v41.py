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


def raw_review(raw, stdout):
    manifest = decode((raw / 'artifact-manifest.json').read_text())
    require(type(manifest.get('schema_version')) is int and manifest['schema_version'] == 1, 'unsupported raw schema')
    identity = manifest['identity']
    require(identity['actual_production_kernel'] == ACTUAL, 'wrong oracle')
    require(identity['frozen_symbol'] == FROZEN and identity['frozen_helper_sha256'] == FROZEN_SHA, 'wrong freeze')
    nsg = integer(identity['nsg'], 2, 4, 'invalid NSG')
    require(nsg in (2, 4) and type(identity['fc600']) is int and identity['fc600'] == nsg, 'wrong NSG')
    for key, expected in [('dim', 5120), ('width', 2304), ('experts', 384), ('active', 6), ('layers', 40), ('nr0', 4), ('tensor_bytes', 1167851520)]:
        require(type(identity.get(key)) is int and identity[key] == expected, 'wrong shape: ' + key)
    trials = integer(identity['trials'], 1, 64, 'invalid parity trial count')
    samples = integer(identity['samples'], 3, 64, 'invalid formal sample count')
    require(type(identity['pair']) is bool, 'invalid pair mode')
    pair = identity['pair']
    # Digests here are metadata syntax, not a rehash of external libraries or
    # tensors. The window controller must bind them to frozen artifact bytes.
    for key in ['probe_library_sha256', 'golden_library_sha256', 'gate_sha256', 'up_sha256']:
        digest(identity[key], 'invalid declared digest: ' + key)
    if pair:
        require(identity['down_actual_symbol'] == 'kernel_mul_mv_id_q2_K_sum6_v41_f32' and
                type(identity['down_nsg']) is int and identity['down_nsg'] == 2, 'wrong pair consumer')
        digest(identity['down_weights_sha256'], 'invalid declared down digest')
    candidates = identity['candidates']
    require(type(candidates) is list and candidates and all(type(c) is str for c in candidates) and
            set(candidates) <= ALLOWED and len(candidates) == len(set(candidates)), 'invalid candidates')
    symbols = [ACTUAL, FROZEN] + candidates
    require(type(identity['psos']) is list and len(identity['psos']) == len(symbols), 'missing PSO identities')
    require({row.get('kernel') for row in identity['psos']} == set(symbols), 'duplicate/wrong PSO identities')
    for row in identity['psos']:
        require(type(row.get('nsg')) is int and row['nsg'] == nsg and
                type(row.get('nr0')) is int and row['nr0'] == 4 and
                type(row.get('groups')) is int and row['groups'] == 2304 // (4*nsg) and
                flag(row.get('actual_production_pso'), row['kernel'] == ACTUAL), 'PSO geometry/origin differs')
        integer(row['pso_max_threads'], 32*nsg, 1024, 'invalid PSO max threads')
        integer(row['static_tg_bytes'], 0, 1 << 20, 'invalid PSO TG bytes')
    require(identity.get('fullmodel') is False and identity.get('seeded_fixture_not_gguf') is True,
            'component scope differs')
    files = {}
    for row in manifest['files']:
        name = row['path']
        require(type(name) is str and name not in ('.', '..') and Path(name).name == name and
                name not in files, 'invalid or duplicate file')
        integer(row['bytes'], 0, 1 << 40, 'invalid file size')
        digest(row['sha256'], 'invalid file digest')
        data = (raw / name).read_bytes()
        require(len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256'], 'payload SHA/size mismatch: ' + name)
        files[name] = data
    expected_files = set()

    def payload(name, length, kind=None):
        expected_files.add(name)
        data = files[name]
        require(len(data) == length, 'incomplete payload: ' + name)
        if kind == 'f32':
            require(all(math.isfinite(x[0]) for x in struct.iter_unpack('<f', data)), 'nonfinite payload: ' + name)
        elif kind == 'guard':
            require(data == b'\xa5'*length, 'poison/guard corruption: ' + name)
        return data

    def inputs(prefix):
        payload(prefix + '-x.f32', 40*5120*4, 'f32')
        payload(prefix + '-weights.f32', 40*6*4, 'f32')
        ids = struct.unpack('<240i', payload(prefix + '-ids.i32', 40*6*4))
        require(len(set(ids)) == 240 and all(0 <= i < 384 for i in ids), 'IDs not valid/disjoint')

    fields = {'mid': 40*13824*4}
    if pair:
        fields['out'] = 40*5120*4
    for trial in range(trials):
        prefix = f'parity-{trial:02d}'
        inputs(prefix)
        for field, length in fields.items():
            oracle = payload(f'{prefix}-{ACTUAL}-{field}.f32', length, 'f32')
            for symbol in symbols:
                require(payload(f'{prefix}-{symbol}-{field}.f32', length, 'f32') == oracle,
                        'output bytes differ: ' + symbol)
                payload(f'{prefix}-{symbol}-{field}.guards', 128, 'guard')
    for candidate in candidates:
        for case in ['ids', 'tp']:
            payload(f'negative-{case}-{candidate}-mid.raw', fields['mid']+128, 'guard')
    invalid = struct.unpack('<240i', payload('negative-ids.i32', 40*6*4))
    require(invalid == tuple([-1,384,-2147483648,2147483647,-2,385]*40), 'invalid ID fixture')
    inputs('timing')
    records = [decode(line) for line in stdout.read_text().splitlines() if line.startswith('{')]
    require(all(type(row) is dict and row.get('kind') in {'identity', 'parity', 'negative', 'timing', 'summary', 'complete'}
                for row in records), 'unknown/nonobject record')
    require([r for r in records if r.get('kind') == 'identity'] == [identity], 'identity mismatch')
    complete = [r for r in records if r.get('kind') == 'complete']
    require(len(complete) == 1 and complete[0].get('raw_files') == len(files) and
            type(complete[0].get('raw_files')) is int and complete[0].get('fullmodel') is False,
            'not complete')
    true_fields(complete[0], ['full_f32_bit_exact', 'production_math_oracle_validated'], 'completion gate failure')
    parity = [r for r in records if r.get('kind') == 'parity']
    expected_parity = {(trial, symbol) for trial in range(trials) for symbol in symbols}
    keys = [(row.get('trial'), row.get('kernel')) for row in parity]
    require(len(keys) == len(expected_parity) and len(keys) == len(set(keys)) and set(keys) == expected_parity,
            'missing/duplicate parity trial-symbol records')
    for row in parity:
        integer(row['trial'], 0, trials-1, 'invalid parity trial')
        true_fields(row, ['completed_cb', 'full_f32_bit_exact', 'pair_full_f32_bit_exact', 'guards', 'finite'], 'parity gate failure')
        require(flag(row.get('release_sg_su_poison'), True), 'parity release SG/SU poison gate failure')
        require(row.get('comparison_reference_symbol') == ACTUAL and
                flag(row.get('actual_production_pso'), row['kernel'] == ACTUAL), 'parity oracle/origin differs')
        for key, expected in [('layers', 40), ('f32_per_layer', 13824), ('f32_total', 40*13824), ('mismatches', 0), ('pair_mismatches', 0), ('tp_world', row['trial'] % 2)]:
            require(type(row.get(key)) is int and row[key] == expected, 'invalid parity field: ' + key)
        require(row.get('first_mismatch') is None and
                real(row['clamp'], 'invalid clamp') == (0 if row['trial'] % 3 == 0 else 10), 'parity case contract differs')
        name = f'parity-{row["trial"]:02d}-{row["kernel"]}-mid.f32'
        require(digest(row['sha256'], 'invalid parity SHA') == hashlib.sha256(files[name]).hexdigest(),
                'parity record SHA does not match raw')
    negatives = [r for r in records if r.get('kind') == 'negative']
    expected_negative = {(case, candidate) for case in ['invalid_ids', 'split_tp'] for candidate in candidates}
    keys = [(row.get('case'), row.get('kernel')) for row in negatives]
    require(len(keys) == len(expected_negative) and len(keys) == len(set(keys)) and set(keys) == expected_negative,
            'missing/duplicate negative case-symbol records')
    for row in negatives:
        true_fields(row, ['all_bytes_poison', 'completed_cb'], 'negative completion gate')
    timing_rows = [r for r in records if r.get('kind') == 'timing']
    expected_timing = {(c, arm, sample) for c in candidates for arm in ['A1', 'B', 'A2'] for sample in range(samples+1)}
    keys = [(r.get('candidate'), r.get('arm'), r.get('sample')) for r in timing_rows]
    require(len(keys) == len(expected_timing) and len(keys) == len(set(keys)) and set(keys) == expected_timing,
            'missing/duplicate timing candidate-arm-sample records')
    summaries = []
    timing_oracles = {}
    for candidate in candidates:
        medians = []
        for arm in ['A1', 'B', 'A2']:
            timing = [r for r in timing_rows if r['candidate'] == candidate and r['arm'] == arm]
            require([r['sample'] for r in timing] == list(range(samples+1)), 'timing samples out of order')
            symbol = candidate if arm == 'B' else ACTUAL
            for row in timing:
                integer(row['sample'], 0, samples, 'invalid timing sample')
                require(row['kernel'] == symbol and flag(row.get('warmup'), row['sample']==0) and
                        flag(row.get('actual_production_pso'), arm != 'B') and row.get('pair') is pair,
                        'timing mode/origin gate')
                true_fields(row, ['completed_cb', 'full_f32_bit_exact', 'production_math_oracle_validated'], 'timing gate')
                require(type(row.get('layers')) is int and row['layers'] == 40 and
                        type(row.get('dispatches')) is int and row['dispatches'] == 40*(2 if pair else 1),
                        'wrong timing work denominator')
                start = real(row['gpu_start_s'], 'nonfinite GPU start')
                end = real(row['gpu_end_s'], 'nonfinite GPU end')
                us = real(row['us_per_layer'], 'nonfinite GPU duration')
                recomputed = (end-start)*1e6/40
                require(start >= 0 and end > start and us > 0 and math.isfinite(recomputed) and
                        math.isclose(recomputed, us, rel_tol=1e-10), 'invalid CB timing')
            medians.append(statistics.median(r['us_per_layer'] for r in timing if not r['warmup']))
            for field, length in fields.items():
                name = f'timing-{candidate}-{arm}-{field}'
                data = payload(name + '.f32', length, 'f32')
                payload(name + '.guards', 128, 'guard')
                if field not in timing_oracles:
                    timing_oracles[field] = data
                require(data == timing_oracles[field], 'timing output mismatch')
        a1, b, a2 = medians
        summary = {'candidate': candidate, 'a1_us': a1, 'b_us': b, 'a2_us': a2,
                   'control_drift_percent': (a2/a1-1)*100,
                   'time_saved_percent': (1-b/((a1+a2)/2))*100,
                   'conservative_time_saved_percent': (1-b/min(a1,a2))*100,
                   'fullmodel_speedup_proven': False}
        declared = [r for r in records if r.get('kind') == 'summary' and r.get('candidate') == candidate]
        require(len(declared) == 1, 'missing/duplicate candidate summary')
        for key in ['a1_us', 'b_us', 'a2_us', 'control_drift_percent', 'time_saved_percent', 'conservative_time_saved_percent']:
            require(math.isclose(real(declared[0][key], 'nonfinite summary: '+key), summary[key],
                                 rel_tol=1e-10, abs_tol=1e-10), 'summary differs: '+key)
        require(declared[0].get('pair') is pair and declared[0].get('control_kernel') == ACTUAL and
                declared[0].get('fullmodel_speedup_proven') is False and
                declared[0].get('production_math_oracle_validated') is True, 'summary scope differs')
        expected_ms = (((a1+a2)/2)-b)*40/1000
        require(math.isclose(real(declared[0]['estimated_ms_saved_per_40_layers'], 'nonfinite estimate'),
                             expected_ms, rel_tol=1e-10, abs_tol=1e-10), 'summary estimate differs')
        summaries.append(summary)
    require(sum(r.get('kind') == 'summary' for r in records) == len(candidates), 'extra summary records')
    require(set(files) == expected_files, 'unexpected/unreviewed manifest payloads')
    return {'payload_files': len(files), 'payload_bytes': sum(map(len,files.values())),
            'full_f32_bytes_match': True, 'pair': pair, 'summaries': summaries}

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
