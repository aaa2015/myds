#!/usr/bin/env python3
"""CPU integrity audit of the saved runtime-inverse case100 rejection.

This never accepts a candidate, changes an oracle, or runs Metal. Complete
0..100 raw/diagnostics and0..99 parity records are required; partial downloads
must fail closed. The original component acceptance reader stays unchanged.
"""
import argparse
import ctypes
import json
import math
from pathlib import Path
import struct
import review_attention_rope_fuse_round3 as core

CORE_SHA = 'fbfe187a651753b0eef4aa8f83f340b930178b6314c770091066bbe3a9f47b1e'


def difference(left, right):
    core.need(len(left) == len(right) and len(left) % 4 == 0, 'Full diff extent differs')
    indexes = [i for i, (a, b) in enumerate(zip(struct.iter_unpack('<I', left), struct.iter_unpack('<I', right))) if a != b]
    first = indexes[0] if indexes else -1
    return dict(same=not indexes, different_words=len(indexes), first_index=first,
        reference_bits=f'{struct.unpack_from("<I", left, first * 4)[0]:08x}' if indexes else 'none',
        candidate_bits=f'{struct.unpack_from("<I", right, first * 4)[0]:08x}' if indexes else 'none')


def declared_diff(record, actual):
    core.need(type(record) is dict and core.native_bit(record['same']) == actual['same'], 'Native diff Boolean differs')
    for key in ('different_words', 'first_index'): core.integer(record[key], actual[key], 'native diff ' + key)
    core.need(record['reference_bits'] == actual['reference_bits'] and record['candidate_bits'] == actual['candidate_bits'], 'Native diff words differ')


def bf16_bits(value):
    return (value if value & 0x7f800000 == 0x7f800000 else value + 0x7fff + ((value >> 16) & 1)) & 0xffff0000


def arithmetic_probe(unrotated, frequencies, position, index):
    """CPU libc model and nearby trig words, expressly not M2 ISA proof."""
    head, column = divmod(index, 512); pair = (column - 448) // 2
    core.need(448 <= column < 512, 'First mismatch outside RoPE tail')
    base = head * 512 + 448 + pair * 2
    raw = struct.unpack_from('<II', unrotated, base * 4); first = tuple(bf16_bits(v) for v in raw)
    fp = lambda u: struct.unpack('<f', struct.pack('<I', u))[0]
    bits = lambda x: struct.unpack('<I', struct.pack('<f', x))[0]
    f32 = lambda x: fp(bits(x))
    library = ctypes.CDLL(None)
    for name, nargs in [('sinf', 1), ('cosf', 1), ('fmaf', 3)]:
        fn = getattr(library, name); fn.argtypes = [ctypes.c_float] * nargs; fn.restype = ctypes.c_float
    real, imag = map(fp, first); frequency_bits = struct.unpack_from('<I', frequencies, pair * 4)[0]
    theta = f32(position * fp(frequency_bits)); cosine, sine = library.cosf(theta), -library.sinf(theta)
    def models(c, s):
        values = dict(nonfused=f32(f32(real * s) + f32(imag * c)),
            fma_im_c_rounded_re_s=library.fmaf(imag, c, f32(real * s)),
            fma_re_s_rounded_im_c=library.fmaf(real, s, f32(imag * c)))
        return {name: dict(f32_bits=f'{bits(value):08x}', bf16_bits=f'{bf16_bits(bits(value)):08x}') for name, value in values.items()}
    neighborhood = []
    for c_delta in range(-8, 9):
        for s_delta in range(-8, 9):
            values = models(fp(bits(cosine) + c_delta), fp(bits(sine) + s_delta))
            if values['fma_im_c_rounded_re_s']['bf16_bits'] == 'b9b70000' and values['fma_re_s_rounded_im_c']['bf16_bits'] == 'b9b80000':
                neighborhood.append(dict(cosine_word_delta=c_delta, inverse_sine_word_delta=s_delta, models=values))
    return dict(head=head, column=column, pair=pair, component='imaginary' if column & 1 else 'real',
        unrotated_input_bits=[f'{v:08x}' for v in raw], first_bf16_input_bits=[f'{v:08x}' for v in first],
        frequency_bits=f'{frequency_bits:08x}', position=position, theta_bits=f'{bits(theta):08x}',
        cpu_libc_cosine_bits=f'{bits(cosine):08x}', cpu_libc_inverse_sine_bits=f'{bits(sine):08x}',
        cpu_libc_models=models(cosine, sine), nearby_trig_words_matching_observed_direction=neighborhood,
        scope='CPU libc sinf/cosf/fmaf and a finite nearby-F32-word model. GPU precise trig values and contraction direction are not captured; this is a hypothesis probe, not M2 ISA or causal proof.')


def complete_inputs(data, prefix, case, seeds):
    mode, selected, source = case['mask_mode'], case['selected'], case['source_rows']
    for suffix, count, seed in [('q.f32', 32768, 19 + mode), ('window.f32', 65536, 37 + mode)]:
        if (count, seed) not in seeds: seeds[count, seed] = core.seeded_values(count, seed)
        core.need(data.get(prefix + '-' + suffix, count * 4) == seeds[count, seed], 'Entire seeded F32 input differs')
    maximum = struct.unpack('<f', bytes.fromhex('ffff7f7f'))[0]
    sinks = struct.pack('<64f', *([-maximum / 2] * 64 if mode == 4 else [(h - 31) / 8 for h in range(64)]))
    masks = struct.pack('<640H', *[0xfbff if mode == 2 or mode == 1 and j % 3 == 1 or mode == 3 and j < 32 else 0xbc00 if mode == 1 else 0 for j in range(640)])
    core.need(data.get(prefix + '-sinks.f32', 256) == sinks and data.get(prefix + '-mask.f16', 1280) == masks and
        data.get(prefix + '-frequencies.f32', 128) == core.frequencies(case['compressed']), 'Complete sink/mask/frequency input differs')
    if selected:
        ids = struct.unpack('<' + str(selected) + 'i', data.get(prefix + '-ids.i32', selected * 4))
        core.need(list(ids) == [(j * 53 + 19) % source for j in range(selected)] and len(set(ids)) == selected, 'Complete ordered IDs/range/uniqueness differs')
        key = source * 512, 71 + mode
        if key not in seeds: seeds[key] = core.seeded_values(*key)
        raw = seeds[key]; comp = b''.join(raw[i + 2:i + 4] for i in range(0, len(raw), 4))
        gathered = b''.join(raw[id * 2048:(id + 1) * 2048] for id in ids)
        core.need(data.get(prefix + '-comp.bf16', source * 1024) == comp and data.get(prefix + '-gathered.f32', selected * 2048) == gathered, 'Complete BF16/gathered inputs differ')


def review(args):
    core.need(core.file_sha(Path(core.__file__)) == CORE_SHA, 'Immutable acceptance-reader core changed')
    manifest = core.load(args.raw / 'artifact-manifest.json'); source = core.load(args.source_identity)
    deployment, process = core.load(args.deployment_freeze), core.load(args.deployment_identity)
    core.need(manifest['schema_version'] == 2 and manifest['status'] == 'failed_parity' and manifest['fullmodel_speedup_proven'] is False and
        manifest['gpu_physical_overlap_proven'] is False and manifest['diagnostic_timings_accepted'] is False, 'Saved failure scope/status differs')
    core.need(core.native_bit(manifest['quick']) is False and len(manifest['diagnostics']) == 101 and len(manifest['cases']) == 100 and
        len(manifest['rejections']) == 11 and manifest['failure'] == manifest['diagnostics'][-1], 'Only exact0..100 failure prefix accepted')
    # Inventory is checked first: a case100-only download can never be accepted.
    data = core.Payloads(args.raw, manifest['files'])
    core.need(core.file_sha(args.binary) == manifest['binary_sha256'] == source['output_sha256']['test_attention_rope_fuse'] and
        core.file_sha(args.source_identity) == manifest['source_identity_sha256'] and core.file_sha(args.private_metallib) == source['output_sha256']['myds.metallib'], 'Original producer/source/private build identity differs')
    source_proof = core.source_contract(args.source_root, source, deployment)
    core.need(source_proof['imag_arithmetic_variant'] == 'runtime_inverse_expression', 'Failed1 must retain original runtime-inverse expression variant')
    records = [core.parse(line) for line in core.read(args.stdout).splitlines() if line.startswith(b'{')]
    core.need(all(type(r) is dict and r.get('kind') in ('host_contract', 'fixture_input', 'library_input', 'library_identity', 'rejection', 'diagnostic_case', 'parity', 'timing') for r in records), 'Unexpected or complete record in failed producer stdout')
    libraries = core.library_contract(manifest, records, args, deployment, process)
    for kind, key in [('rejection', 'rejections'), ('diagnostic_case', 'diagnostics'), ('parity', 'cases'), ('timing', 'timings')]:
        core.need([r for r in records if r.get('kind') == kind] == manifest[key], 'Complete original stdout/manifest differs: ' + kind)
    core.need([r for r in records if r.get('kind') == 'fixture_input'] == [manifest['fixture_input']] and
        manifest['fixture_input']['before_metal_init'] is True and manifest['fixture_input']['binary_sha256'] == manifest['binary_sha256'] and
        manifest['fixture_input']['source_identity_sha256'] == manifest['source_identity_sha256'], 'Preinit original producer identity missing')
    core.need(core.canon(manifest['fixture_input']['binary_path']) == core.canon(manifest['binary_path']) and
        core.canon(manifest['fixture_input']['source_identity_path']) == core.canon(manifest['source_identity_path']) and manifest['output_providers'] == core.PROVIDERS, 'Original producer paths/provider map differ')
    for item in (manifest, manifest['fixture_input']):
        core.integer(item['args_bytes'], 336, 'common ABI'); core.integer(item['inverse'], 1, 'common inverse')
    core.need(records[0]['kind'] == 'host_contract' and records[0]['cpu_only'] is True and records[0]['strict_flag'] is True, 'Host contract missing')
    core.integer(records[0]['args_bytes'], 336, 'host ABI'); core.integer(records[0]['invalid_null_dispatch_delta'], 0, 'host null dispatch')
    for i, record in enumerate(manifest['rejections']):
        core.need(record['case'] == core.REJECTIONS[i] and record['all_observed_buffers_full_bytes_unchanged'] is True, 'Negative identity/unchanged declaration differs')
        for key, value in [('dispatch_delta', 0), ('encoded_delta', 0), ('requests_delta', 1), ('failures_delta', 1)]: core.integer(record[key], value, key)
        for j, length in enumerate(core.NEGATIVE_LENGTHS):
            core.need(data.get(f'reject{i}-buffer{j}-before.raw', length) == data.get(f'reject{i}-buffer{j}-after.raw', length), 'Entire negative buffer changed')
    samples, repetitions = manifest['samples'], manifest['repetitions']
    core.need(type(samples) is int and 3 <= samples <= 31 and type(repetitions) is int and 1 <= repetitions <= 64, 'Original timing protocol differs')
    enabled = core.native_bit(manifest['timing_enabled']); seeds = {}; cases = []; timing_count = 0
    expected_kinds = ['host_contract', 'fixture_input'] + ['library_input'] * 3 + ['library_identity'] * 3 + ['rejection'] * 11
    expected = [(p, c, d, mask) for p in range(6) for c in (False, True) for d in (False, True) for mask in range(5)]
    probe = None
    for i, diagnostic in enumerate(manifest['diagnostics']):
        p, compressed, direct, mask = expected[i]; keys = core.KEY_COUNTS[p]; nraw = min(keys, 128); selected = keys - nraw; nsource = 2 * selected + 17 if selected else 0
        geometry = dict(case=i, pos=core.POSITIONS[p], keys=keys, mask_mode=mask, args_bytes=336, inverse=1, requests_delta=1, encoded_delta=1, failures_delta=0)
        for key, value in geometry.items(): core.integer(diagnostic[key], value, key)
        core.need(core.native_bit(diagnostic['compressed']) == compressed and core.native_bit(diagnostic['direct']) == direct and
            diagnostic['output_providers'] == core.PROVIDERS and diagnostic['actual_production_fused_ran_before_debug'] is True and diagnostic['candidate_accepted'] is False and
            diagnostic['diagnostic_timings_accepted'] is False and diagnostic['guards_finite_poison_bf16_valid'] is True, 'Diagnostic complete geometry/run-order/scope differs')
        if mask == 4: core.need(diagnostic['mask_description'] == 'unmasked_no_effective_sink', 'Mode4 mislabeled')
        expected_kinds.append('diagnostic_case')
        prefix = f'case{i}'; output = [data.guarded(prefix + f'-out{j}.guarded', 131072, j != 1) for j in range(4)]
        unrotated = data.guarded(prefix + '-production-unrotated.guarded', 131072)
        partial = [data.guarded(prefix + f'-partial{j}.guarded', 4194304) for j in range(2)]
        stats = [data.guarded(prefix + f'-stats{j}.guarded', 24576) for j in range(2)]
        comparisons = dict(reference_debug_unrotated=difference(unrotated, output[1]), production_fused=difference(output[0], output[3]),
            debug_fused=difference(output[0], output[2]), partial=difference(*partial), stats=difference(*stats))
        for key, actual in comparisons.items(): declared_diff(diagnostic[key], actual)
        exact = all(x['same'] for x in comparisons.values())
        core.need(core.native_bit(diagnostic['case_exact_gate_passed']) == exact and exact == (i < 100), 'Failure prefix/exact raw gate differs')
        core.need(comparisons['reference_debug_unrotated']['same'] and comparisons['partial']['same'] and comparisons['stats']['same'], 'Unexpected earlier attention reduction mismatch')
        core.need(output[2] == output[3], 'Private/debug and deployed candidate differ from each other')
        if mask != 2 and not (mask == 3 and keys <= 32): core.need(any(u[0] & 0x7fffffff for u in struct.iter_unpack('<I', output[0])), 'All-zero active seeded output')
        metadata = dict(mask_mode=mask, selected=selected, source_rows=nsource, compressed=compressed)
        complete_inputs(data, prefix, metadata, seeds)
        if i < 100:
            parity = manifest['cases'][i]
            for key, value in {**geometry, 'n_raw': nraw, 'raw_start': 127, 'selected': selected, 'source_rows': nsource}.items(): core.integer(parity[key], value, key)
            core.need(core.native_bit(parity['compressed']) == compressed and core.native_bit(parity['direct']) == direct and parity['output_providers'] == core.PROVIDERS and
                all(parity[k] is True for k in ('full_output_same', 'partial_stats_full_bytes_same', 'finite', 'guards', 'production_golden')), 'Passed prefix parity declaration differs')
            expected_kinds.append('parity')
        else:
            target = dict(same=False, different_words=1, first_index=30205, reference_bits='b9b70000', candidate_bits='b9b80000')
            core.need(comparisons['production_fused'] == comparisons['debug_fused'] == target, 'Saved case100 rejection bytes differ')
            probe = arithmetic_probe(unrotated, data.get(prefix + '-frequencies.f32', 128), geometry['pos'], 30205)
        times = [t for t in manifest['timings'] if t['case'] == i]
        wanted = {(arm, sample) for arm in ('A1', 'B', 'A2') for sample in range(samples + 1)} if enabled and i < 100 else set(); seen = set()
        for item in times:
            core.integer(item['case'], i, 'timing case'); arm, sample = item['arm'], item['sample']; key = (arm, sample)
            core.need(type(sample) is int and key in wanted and key not in seen, 'Unexpected/duplicate saved timing'); seen.add(key)
            core.need(core.native_bit(item['warmup']) == (sample == 0) and all(item[k] is True for k in ('full_output_same', 'guards', 'finite')), 'Saved timing output declaration differs')
            core.integer(item['repetitions'], repetitions, 'timing repetitions'); core.integer(item['fused_encoded'], repetitions if arm == 'B' else 0, 'timing encoded')
            base = (2 if direct else 2 + (nraw > 1)) + (selected > 0) + (keys % 32 != 0)
            core.integer(item['dispatches'], repetitions * (base + (arm != 'B')), 'complete pair dispatch')
            start, end = item['gpu_start_s'], item['gpu_end_s']
            core.need(type(start) in (int, float) and type(end) in (int, float) and math.isfinite(start) and math.isfinite(end) and end > start, 'Saved timing timestamps invalid')
            core.need(math.isclose((end - start) * 1e6 / repetitions, item['gpu_us'], rel_tol=1e-12, abs_tol=1e-9), 'Saved timing normalization differs')
            core.need(data.guarded(f'{prefix}-timing-{arm}-s{sample}.guarded', 131072, True) == output[0], 'Entire timed output differs')
            expected_kinds.append('timing'); timing_count += 1
        core.need(seen == wanted, 'Incomplete saved-prefix timings')
        cases.append(dict(case=i, exact_gate_passed=exact, complete_raw_differences=comparisons))
    core.need([r['kind'] for r in records] == expected_kinds and len(manifest['timings']) == timing_count and data.checked == set(data.files), 'Missing/extra/out-of-order producer record or payload inventory')
    return dict(status='verified_failed_parity', cpu_only=True, integrity_pass=True, candidate_accepted=False, exact_math_gate_pass=False,
        fullmodel_speedup_proven=False, component_performance_accepted=False, gpu_physical_overlap_proven=False,
        saved_diagnostic_cases=101, successful_prefix_parity_cases=100, failed_case=100, predispatch_negatives=11,
        raw_files=len(data.checked), raw_bytes=data.bytes, complete_payload_sha_size_guards_finite_recomputed=True,
        source_contract=source_proof, libraries=libraries, cases=cases, case100_arithmetic_probe=probe,
        saved_timing_records_checked_but_not_accepted=timing_count,
        identity_sha256={name: core.file_sha(getattr(args, name)) for name in ('stdout', 'source_identity', 'binary', 'golden_metallib', 'private_metallib', 'deployed_metallib', 'deployment_identity', 'deployment_freeze')},
        manifest_sha256=core.file_sha(args.raw / 'artifact-manifest.json'), reader_sha256=core.file_sha(Path(__file__)), acceptance_core_sha256=CORE_SHA,
        scope='Complete saved0..100 component evidence and exact rejection against immutable actualproduction99. Candidate remains rejected. No120-case acceptance, full-model logits/tokens/state or decode speed claim. CPU FMA/trig models are suggestions, not measured GPU ISA.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('raw', 'stdout', 'source-identity', 'source-root', 'binary', 'golden-metallib', 'private-metallib', 'deployed-metallib', 'deployment-identity', 'deployment-freeze', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(); core.need(not args.out.exists(), 'Never overwrite prior failure evidence')
    result = review(args); args.out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps(dict(status=result['status'], candidate_accepted=False, report=str(args.out))))


if __name__ == '__main__': main()
