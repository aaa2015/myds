#!/usr/bin/env python3
"""Recompute the private attention arithmetic diagnostic from raw bytes on CPU.

Requires the original failed component as well as the diagnostic window. It
hashes complete source inputs/outputs before binding the one-query slice, never
uses hashes as a replacement for F32 comparison, and never imports Metal or
submits inference. Exit zero validates diagnostic evidence, not a candidate.
"""
import argparse
import array
import ctypes as ct
import hashlib
import json
from pathlib import Path
import struct
import sys

QUERY_BYTES, KV_BYTES, TRACE_BYTES = 131072, 655360, 5886976
HEAD_FLOATS, BLOCK_FLOATS = 22996, 1124
CONTROL = 'kernel_myds_flash_attn_vec_head_first_f16_dk512_dv512'
H2 = 'kernel_myds_prefill_attention_head2_f16_dk512_dv512'


class FlashArgs(ct.LittleEndianStructure):
    _fields_ = [(name, kind) for names, kind in (
        (('ne01', 'ne02', 'ne03'), ct.c_int32), (('nb01', 'nb02', 'nb03'), ct.c_uint64),
        (('ne11', 'ne_12_2', 'ne_12_3', 'ns10'), ct.c_int32), (('nb11', 'nb12', 'nb13'), ct.c_uint64),
        (('ns20',), ct.c_int32), (('nb21', 'nb22', 'nb23'), ct.c_uint64),
        (('ne31', 'ne32', 'ne33'), ct.c_int32), (('nb31', 'nb32', 'nb33'), ct.c_uint64),
        (('ne1', 'ne2', 'ne3'), ct.c_int32), (('scale', 'max_bias', 'm0', 'm1'), ct.c_float),
        (('n_head_log2',), ct.c_int32), (('logit_softcap',), ct.c_float), (('nb_kvq',), ct.c_uint64),
    ) for name in names]


def need(value, message):
    if not value:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1048576), b''):
            h.update(part)
    return h.hexdigest()


def inside(base, name):
    need(isinstance(name, str) and name and not Path(name).is_absolute(), 'Relative logical artifact path required')
    base = Path(base).resolve()
    path = (base / name).resolve(strict=True)
    need(path.is_relative_to(base), 'Artifact outside window')
    return path


def words(data):
    need(len(data) % 4 == 0, 'F32 alignment')
    result = array.array('I')
    need(result.itemsize == 4, 'Host uint32 size')
    result.frombytes(data)
    if sys.byteorder != 'little':
        result.byteswap()
    return result


def compare(left, right):
    need(len(left) == len(right), 'Compared F32 lengths differ')
    a, b = words(left), words(right)
    count, first = 0, None
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            count += 1
            if first is None:
                first = dict(index=i, reference_u32=x, candidate_u32=y)
    return dict(bytes=len(left), checked_f32=len(a), mismatches=count,
                full_f32_bit_exact=count == 0, first_mismatch=first,
                all_words_compared=True, tolerance_used=False)


def finite_written(data):
    a = words(data)
    poison = sum(v == 0xa5a5a5a5 for v in a)
    nonfinite = sum(v & 0x7f800000 == 0x7f800000 for v in a)
    need(poison == 0 and nonfinite == 0, 'Nonfinite or unwritten output')
    return dict(checked_f32=len(a), poison_words=poison, nonfinite=nonfinite)


def verify_manifest(base, expected_count=None):
    manifest = load(base / 'artifact-manifest.json')
    records, seen, checked = {}, set(), []
    need(isinstance(manifest.get('files'), list), 'Logical artifact manifest files')
    for row in manifest['files']:
        name = row['path']
        need(name not in seen, 'Duplicate logical artifact path')
        seen.add(name)
        path = inside(base, name)
        need(path.stat().st_size == row['bytes'] and digest(path) == row['sha256'], 'Artifact bytes/SHA mismatch: ' + name)
        records[name] = row
        checked.append(dict(path=name, bytes=row['bytes'], sha256=row['sha256']))
    if expected_count is not None:
        need(len(checked) == expected_count, 'Unexpected logical artifact count')
    aliases = []
    for row in manifest['files']:
        if 'storage_alias_of' in row:
            name, target = row['path'], row['storage_alias_of']
            need(target in records and records[target]['sha256'] == row['sha256'] and records[target]['bytes'] == row['bytes'], 'Invalid logical storage alias')
            need(inside(base, name).read_bytes() == inside(base, target).read_bytes(), 'Storage alias raw bytes differ')
            aliases.append(dict(path=name, storage_alias_of=target,
                                hardlink_preserved=inside(base, name).stat().st_ino == inside(base, target).stat().st_ino))
    return manifest, records, checked, aliases


def prefix(base, records, name, count):
    need(name in records, 'Original source descriptor missing: ' + name)
    path = inside(base, name)
    row = records[name]
    need(path.stat().st_size == row['bytes'] and digest(path) == row['sha256'], 'Original full source bytes/SHA mismatch: ' + name)
    with path.open('rb') as stream:
        data = stream.read(count)
    need(len(data) == count, 'Original source prefix incomplete')
    return data


def stage_differences(left, right):
    a, b = words(left), words(right)
    need(len(left) == len(right) == TRACE_BYTES, 'Stage dump size')
    need(0xa5a5a5a5 not in a and 0xa5a5a5a5 not in b, 'Unwritten stage value')
    result = []
    for name, offset, size in [('mqk', 0, 32), ('score', 32, 32), ('softmax', 64, 32),
                               ('stats_M_S_ms_sum', 96, 4), ('PV_lo', 100, 512), ('acc', 612, 512),
                               ('sink_acc', 22480, 512), ('sink_stats', 22992, 4)]:
        blocks = 1 if name.startswith('sink') else 20
        count, first = 0, None
        for head in range(64):
            for block in range(blocks):
                start = head * HEAD_FLOATS + offset + (0 if blocks == 1 else block * BLOCK_FLOATS)
                for index in range(size):
                    if a[start + index] != b[start + index]:
                        count += 1
                        if first is None:
                            first = dict(head=head, block=block, index=index,
                                         reference_u32=a[start + index], candidate_u32=b[start + index])
        result.append(dict(stage=name, checked_f32=64 * blocks * size, mismatches=count, first_mismatch=first))
    need(sum(row['checked_f32'] for row in result) == TRACE_BYTES // 4, 'Every stage word covered exactly once')
    return dict(complete_bytes=compare(left, right), stages=result, all_words_written=True,
                nonfinite_stage_values_not_rejected='Masked logits may legitimately be negative infinity')


def review(root, source, check_executed=False):
    device = root / 'device'
    protocol = load(root / 'protocol.json')
    need(protocol['scope'] == 'arithmetic_diagnostic_only' and protocol['performance_measured'] is False and
         protocol['fullmodel_run'] is False and protocol['production_adoption'] is False, 'Diagnostic scope only')
    manifest, records, checked, aliases = verify_manifest(device, 42)
    need(manifest['status'] == 'complete' and manifest['diagnostic_only'] is True and manifest['performance_evidence'] is False, 'Completed diagnostic collection')
    copied = device / 'source-component-manifest.json'
    original = source / 'artifact-manifest.json'
    need(copied.read_bytes() == original.read_bytes() and digest(original) == manifest['source_manifest_sha256'], 'Original failed component manifest binding')
    original_manifest = load(original)
    original_records = {row['path']: row for row in original_manifest['files']}
    need(len(original_records) == len(original_manifest['files']), 'Unique original logical records')
    case = manifest['source_case']
    need(case == 0 and manifest['query_index'] == 0, 'This reviewer validates original failing query0/case0')
    source_case = original_manifest['cases'][case]
    need(source_case['case'] == case and source_case['layer'] == 2 and source_case['base_pos'] == 0 and
         source_case['c_start'] == source_case['rows'] == 2048 and source_case['keys'] == 640,
         'Original real capture layer/chunk binding')
    lead = 'capture-case-%03d-' % case
    raw_args = prefix(source, original_records, lead + 'args200', 200)
    need(ct.sizeof(FlashArgs) == 200 and FlashArgs.nb_kvq.offset == 192, 'CPU args ABI')
    args = FlashArgs.from_buffer_copy(raw_args)
    rows = args.ne01
    need(rows == 2048 and args.ne02 == args.ne1 == 64 and args.ne11 == 640 and args.nb31 >= 1280, 'Real 2048-query original shape')
    original_shape = dict(rows=rows, keys=args.ne11, heads=args.ne02, mask_stride=args.nb31)
    args.ne01 = args.ne31 = args.ne2 = 1
    args.nb32 = args.nb33 = args.nb31
    need(bytes(args) == (device / 'slice-args200').read_bytes(), 'Exact args slice transformation')
    q = prefix(source, original_records, lead + 'q.f32', QUERY_BYTES)
    unique = prefix(source, original_records, lead + 'kv-unique.f16', original_records[lead + 'kv-unique.f16']['bytes'])
    ids = prefix(source, original_records, lead + 'kv-rows.u32', 640 * 4)
    kv = b''.join(unique[i * 1024:(i + 1) * 1024] for i in words(ids))
    need(len(kv) == KV_BYTES and all(i < len(unique) // 1024 for i in words(ids)), 'Dictionary reconstruction/range')
    mask = prefix(source, original_records, lead + 'mask-compact.f16', 1280)
    halves = struct.unpack('<640H', mask)
    need(all(v in (0, 0xfc00) for v in halves), 'Real mask encoding')
    need(all(v == 0 for v in halves), 'The diagnostic query must retain all 128 window and 512 selected keys')
    padded_mask = mask + struct.pack('<H', 0xfc00) * ((args.nb31 - 1280) // 2)
    sinks = prefix(source, original_records, lead + 'sinks.f32', 256)
    for name, expected in [('slice-q.f32', q), ('slice-kv.f16', kv), ('slice-mask-padded.f16', padded_mask), ('slice-sinks.f32', sinks)]:
        need((device / name).read_bytes() == expected, 'Exact real source slice: ' + name)
    original_outputs = {}
    for provider, label in [(0, 'native'), (2, 'h2')]:
        name = 'parity-case-%03d-provider-%d-output.f32' % (case, provider)
        expected = prefix(source, original_records, name, QUERY_BYTES)
        need((device / ('original-' + label + '-query0.f32')).read_bytes() == expected, 'Full prior output prefix binding')
        original_outputs[label] = expected
    outputs, output_integrity = [], []
    for provider in range(5):
        stem = 'provider-%d-output' % provider
        data = (device / (stem + '.f32')).read_bytes()
        need(len(data) == QUERY_BYTES, 'Complete one-query output')
        finite = finite_written(data)
        need((device / (stem + '.guards')).read_bytes() == b'\xa5' * 128, 'Output guards')
        output_integrity.append(dict(provider=provider, guards_intact=True, **finite))
        outputs.append(data)
    provider_rows = manifest['providers']
    need([p['provider'] for p in provider_rows] == list(range(5)), 'Five explicit providers')
    symbols = [CONTROL, CONTROL, H2, 'probe_mh_ref_debug', 'probe_mh_h2_debug']
    library_fields = ['production_metallib_sha256', 'probe_metallib_sha256', 'probe_metallib_sha256',
                      'debug_metallib_sha256', 'debug_metallib_sha256']
    for i, p in enumerate(provider_rows):
        batch = 2 if i in (2, 4) else 1
        need(p['symbol'] == symbols[i] and p['threads'] == [32, batch, 1] and p['groups'] == [64 // batch, 1, 1]
             and p['threadgroup_memory'] == (23040 if batch == 2 else 3328) and p['max_threads'] >= 32 * batch and
             p['library_sha256'] == manifest[library_fields[i]], 'Provider exact symbol/library/ABI/dispatch identity')
    pairs = [('sameprobe_vs_production', outputs[0], outputs[1]),
             ('slice_native_vs_full_working_set_query0', original_outputs['native'], outputs[0]),
             ('slice_h2_vs_full_working_set_query0', original_outputs['h2'], outputs[2]),
             ('debug_native_vs_native', outputs[0], outputs[3]), ('debug_h2_vs_release_h2', outputs[2], outputs[4]),
             ('release_h2_vs_native', outputs[0], outputs[2])]
    comparisons = {name: compare(a, b) for name, a, b in pairs}
    for reported in manifest['comparisons']:
        actual = comparisons[reported['kind']]
        need(actual['mismatches'] == reported['mismatches'] and actual['full_f32_bit_exact'] == bool(reported['full_f32_bit_exact']), 'Reported comparison disagrees with raw words')
    sliced = all(comparisons[name]['full_f32_bit_exact'] for name in comparisons if name.startswith('slice_') or name.startswith('sameprobe_'))
    unperturbed = sliced and comparisons['debug_native_vs_native']['full_f32_bit_exact'] and comparisons['debug_h2_vs_release_h2']['full_f32_bit_exact']
    need(manifest['slice_matches_full_working_set'] == sliced and manifest['observer_unperturbed'] == unperturbed, 'Observer/slice gates recomputed')
    traces = []
    for provider in (3, 4):
        stem = 'provider-%d-stages' % provider
        traces.append((device / (stem + '.f32')).read_bytes())
        need((device / (stem + '.guards')).read_bytes() == b'\xa5' * 128, 'Stage guards')
    trace_result = stage_differences(*traces)
    scans, names = [], set()
    expected_symbols = {H2.replace('_f16_', '_policy%d_f16_' % n) for n in range(1, 8)}
    expected_symbols |= {H2.replace('_f16_', '_direct_%s_f16_' % part) for part in ('k', 'v', 'kv')}
    need(len(manifest['scan']) == 10, 'All ten uninstrumented scan variants')
    need({row['index'] for row in manifest['scan']} == set(range(10)), 'Unique complete scan output indices')
    for row in manifest['scan']:
        index, symbol = row['index'], row['symbol']
        need(symbol not in names and symbol in expected_symbols, 'Unique expected policy/ablation symbol')
        names.add(symbol)
        stem = 'scan-%02d-output' % index
        data = (device / (stem + '.f32')).read_bytes()
        need(len(data) == QUERY_BYTES, 'Complete scan output')
        integrity = finite_written(data)
        need((device / (stem + '.guards')).read_bytes() == b'\xa5' * 128, 'Scan guards')
        actual = compare(outputs[0], data)
        need(actual['mismatches'] == row['mismatches'] and actual['full_f32_bit_exact'] == bool(row['full_f32_bit_exact']), 'Scan raw words disagree with report')
        scans.append(dict(index=index, symbol=symbol, guards_intact=True, output_integrity=integrity, **actual))
    need(names == expected_symbols, 'Complete policy/ablation symbol coverage')
    executed = []
    argv = manifest['argv']
    flags = [('binary_sha256', None), ('production_metallib_sha256', '--production-metallib'),
             ('probe_metallib_sha256', '--probe-metallib'), ('debug_metallib_sha256', '--debug-metallib'),
             ('scan_metallib_sha256', '--scan-metallib')]
    frozen = load(root / 'diagnostic-inputs.json')
    for field, flag in flags:
        path = Path(argv[0] if flag is None else argv[argv.index(flag) + 1])
        recorded = frozen.get(str(path)) or frozen.get(str(path.resolve()))
        need(recorded == manifest[field], 'Executed artifact identity absent from frozen preflight: ' + field)
        if check_executed:
            need(digest(path) == manifest[field], 'Executed artifact current bytes mismatch')
        executed.append(dict(field=field, path=str(path), sha256=manifest[field],
                             frozen_preflight_bound=True, current_file_rehashed=check_executed))
    return dict(schema_version=1, review_status='passed', review_scope='CPU independent diagnostic evidence only',
                reviewer_sha256=digest(Path(__file__)), root=str(root), original_component=str(source),
                gpu_work_submitted=False, inference_run=False, performance_evidence=False, production_adoption=False,
                component_acceptance=False, root_cause_proven=False, source_manifest_sha256=digest(original),
                diagnostic_manifest_sha256=digest(device / 'artifact-manifest.json'),
                diagnostic_input_identity_sha256=digest(root / 'diagnostic-inputs.json'),
                source_shape=original_shape, source_case=source_case, exact_real_query0_source_slice_verified=True,
                full_original_inputs_and_outputs_hashed_before_prefix_comparison=True,
                logical_raw_files_checked=len(checked), logical_raw_files=checked, storage_aliases=aliases,
                provider_output_integrity=output_integrity, comparisons=comparisons,
                slice_matches_full_working_set=sliced, observer_unperturbed=unperturbed,
                stage_attribution_allowed=unperturbed, stage_differences=trace_result,
                scan=scans, executed_artifact_bindings=executed,
                interpretation_limits=[
                    'All data are one actual query from case0/layer2 of the original 2048-query capture.',
                    'A perturbed native debug observer prevents attributing the release mismatch from stage traces.',
                    'direct_v/direct_kv alter PV memory access, loop structure and compiler optimization scope together.',
                    'A scan match is not full-chunk/multilayer numerical acceptance or a performance result.',
                    'Restoration and full executed source archives have separate window reviews.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--check-executed-files', action='store_true')
    args = parser.parse_args()
    need(not args.out.exists(), 'New independent review output required')
    result = review(args.root.resolve(strict=True), args.input_dir.resolve(strict=True), args.check_executed_files)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('review_status', 'logical_raw_files_checked', 'slice_matches_full_working_set',
                    'observer_unperturbed', 'stage_attribution_allowed', 'performance_evidence')}, sort_keys=True))


if __name__ == '__main__':
    main()
