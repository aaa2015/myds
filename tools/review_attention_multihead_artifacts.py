#!/usr/bin/env python3
"""Independent CPU review of the real full-chunk multihead attention experiment.

Exit 0 confirms internally consistent evidence, not performance acceptance.
No Metal, inference process, network request, or GPU operation is performed.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import mmap
from pathlib import Path
import statistics
import struct
import tarfile

ROWS, KEYS, QUERY_BYTES, KV_ROW_BYTES = 2048, 640, 131072, 1024
LAYERS = (2, 8, 19)
CONTROL = 'kernel_myds_flash_attn_vec_head_first_f16_dk512_dv512'
SYMBOLS = [CONTROL, CONTROL, 'kernel_myds_prefill_attention_head2_f16_dk512_dv512',
           'kernel_myds_prefill_attention_head4_f16_dk512_dv512']
FC = [dict(index=i, is_bool=j < 6, value=v) for j, (i, v) in enumerate(zip(
    (400, 401, 402, 403, 404, 405, 420, 421, 422, 423), (1, 1, 0, 0, 0, 0, 512, 512, 1, 1)))]


def require(value, message):
    if not value:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text())


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def close(a, b):
    return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-7)


def path_inside(directory, relative):
    require(isinstance(relative, str) and relative and not Path(relative).is_absolute(), 'Relative artifact path required')
    path = directory / relative
    require(not path.is_symlink() and path.resolve().is_relative_to(directory.resolve()) and path.is_file(),
            'Missing, symlinked or escaped artifact: ' + relative)
    return path


def equal_files(first, second):
    require(first.stat().st_size == second.stat().st_size, 'Full output lengths differ')
    with first.open('rb') as a, second.open('rb') as b:
        while True:
            x, y = a.read(4 * 1024 * 1024), b.read(4 * 1024 * 1024)
            require(x == y, 'Full F32 output bytes differ')
            if not x:
                break


def finite_written(path, expected_bytes, written=True):
    require(path.stat().st_size == expected_bytes, 'Full F32 output byte count')
    try:
        import numpy as np
    except ImportError:
        np = None
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            require(len(block) % 4 == 0, 'F32 alignment')
            if np is not None:
                words = np.frombuffer(block, dtype='<u4')
                require(not np.any((words & 0x7f800000) == 0x7f800000) and (not written or not np.any(words == 0xa5a5a5a5)),
                        'Nonfinite or unwritten output')
            else:
                require(all((v[0] & 0x7f800000) != 0x7f800000 and (not written or v[0] != 0xa5a5a5a5)
                            for v in struct.iter_unpack('<I', block)), 'Nonfinite or unwritten output')


def expanded_kv_sha(dictionary, index, rows=ROWS):
    require(dictionary.stat().st_size > 0 and dictionary.stat().st_size % 1024 == 0, 'KV dictionary rows')
    require(index.stat().st_size == rows * KEYS * 4, 'Complete KV row-index length')
    result = hashlib.sha256()
    with dictionary.open('rb') as source, mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ) as values:
        unique = len(values) // KV_ROW_BYTES
        # Validate the actual half bytes once; no BF16/F16 reinterpretation.
        require(all((v[0] & 0x7c00) != 0x7c00 for v in struct.iter_unpack('<H', values)), 'Nonfinite dictionary F16')
        with index.open('rb') as stream:
            for (number,) in struct.iter_unpack('<I', stream.read()):
                require(number < unique, 'KV row ID outside dictionary')
                offset = number * KV_ROW_BYTES
                result.update(values[offset:offset + KV_ROW_BYTES])
    return result.hexdigest()


def mask_coverage(path, rows=ROWS):
    data = path.read_bytes()
    require(len(data) == rows * KEYS * 2, 'Complete consumed mask length')
    picks, windows = [], []
    for row in range(rows):
        values = struct.unpack_from('<640H', data, row * 1280)
        require(all(v in (0, 0xfc00) for v in values), 'Mask is neither zero nor real gather negative infinity')
        windows.append(sum(v == 0 for v in values[:128]))
        picks.append(sum(v == 0 for v in values[128:]))
    return dict(minimum_valid_picks=min(picks), maximum_valid_picks=max(picks),
                full_512_pick_queries=sum(v == 512 for v in picks),
                minimum_valid_window=min(windows), maximum_valid_window=max(windows))


def validate_args(data):
    require(len(data) == 200, 'Attention args ABI length')
    integers = {0: ROWS, 4: 64, 8: 1, 40: 640, 44: 1, 48: 1, 52: 512,
                80: 512, 112: ROWS, 116: 1, 120: 1, 152: 64, 156: ROWS, 160: 1, 180: 0}
    for offset, expected in integers.items():
        require(struct.unpack_from('<i', data, offset)[0] == expected, 'Actual args integer mismatch at ' + str(offset))
    mask_stride = struct.unpack_from('<Q', data, 128)[0]
    require(mask_stride >= 1280 and mask_stride % 2 == 0, 'Original mask row stride')
    strides = {16: 131072, 24: 2048, 32: 131072, 56: 1024, 64: 655360, 72: 655360,
               88: 1024, 96: 655360, 104: 655360, 136: mask_stride * ROWS,
               144: mask_stride * ROWS, 192: 655360}
    for offset, expected in strides.items():
        require(struct.unpack_from('<Q', data, offset)[0] == expected, 'Actual args stride mismatch at ' + str(offset))
    expected_scale = struct.unpack('<f', struct.pack('<f', 1 / math.sqrt(512)))[0]
    require(struct.unpack_from('<f', data, 164)[0] == expected_scale, 'Actual attention scale')
    require(all(struct.unpack_from('<f', data, offset)[0] == 0 for offset in (168, 172, 176, 184)), 'Unexpected bias/softcap')
    return mask_stride


def archive_member(archive, relative):
    names = [m for m in archive.getmembers() if m.name == relative or m.name.endswith('/' + relative)]
    require(len(names) == 1, 'Missing/ambiguous archive member: ' + relative)
    stream = archive.extractfile(names[0])
    require(stream is not None, 'Archive member is not readable: ' + relative)
    value = hashlib.sha256()
    for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
        value.update(block)
    return value.hexdigest()


def review(root, source_archive, artifacts_archive):
    directory = root / 'component'
    manifest = load(directory / 'artifact-manifest.json')
    require(manifest['status'] == 'complete' and manifest['total_cases'] == 3 and manifest['samples'] == 9 and
            manifest['repetitions_per_cb'] == 1 and manifest['mode'] == 'concurrent' and not manifest['fullmodel'],
            'Full 2048-query component did not complete with the frozen sample protocol')
    protocol = load(root / 'protocol.json')
    expected_protocol = dict(scope='prefill_gathered_head_first_vec', mode='concurrent', layers=list(LAYERS),
                             base_pos=0, c_start=2048, rows=ROWS, keys=640, ratio=2, n_comp=2048, top_k=512,
                             samples=9, reps=1, cases=3, providers=4, variants=[2, 4],
                             max_absolute_control_drift_percent=2.0, minimum_conservative_time_saved_percent_strict=2.0)
    require(all(protocol.get(k) == v for k, v in expected_protocol.items()), 'Frozen protocol differs')
    before, restored = load(root / 'before.json'), load(root / 'restored.json')
    identity_keys = ('argv_raw_hex', 'myds_env_raw_hex', 'cwd', 'log', 'binary', 'binary_sha256',
                     'metallib', 'metallib_sha256', 'config', 'config_sha256', 'effective_myds_env', 'nice')
    require(all(k in before and before[k] == restored[k] for k in identity_keys), 'Restoration identity including nice differs')
    require(before['health'] == restored['health'] == {'status': 'ok'}, 'Recorded restored health')
    require(manifest['production_metallib_sha256'] == before['metallib_sha256'], 'Frozen production library identity')
    require(digest(root / 'original-myds.conf') == before['config_sha256'] and
            load(root / 'effective-env.json') == before['effective_myds_env'], 'Original config/effective environment archive')
    for kind in ('binary', 'metallib'):
        require(digest(root / 'production-artifacts' / Path(before[kind]['path']).name) == before[kind + '_sha256'],
                'Production artifact backup identity: ' + kind)
    env = before['effective_myds_env']
    require(env.get('MYDS_METAL_CONCURRENT', '1') == '1' and env.get('MYDS_ATTN_VEC_HEAD_FIRST', '1') == '1' and
            env.get('MYDS_ATTN_GATHER', '1') == '1' and env.get('MYDS_PREFILL_ATTN', 'vec') == 'vec', 'Production attention provider/config mismatch')
    steps = load(root / 'steps.json')
    require(all(s['returncode'] == 0 for s in steps) and {'tokenize', 'capture', 'host-validation', 'component'} <=
            {s['name'] for s in steps}, 'Missing/failed execution step')
    events = [json.loads(line) for line in (root / 'events.jsonl').read_text().splitlines() if line.strip()]
    require(not any(row['kind'] in ('monitor_failure', 'restoration_failure', 'window_failure') for row in events), 'Recorded window failure')
    files, inode_hashes, seen = manifest['files'], {}, set()
    for entry in files:
        relative = entry['path'];require(relative not in seen, 'Duplicate logical artifact');seen.add(relative)
        path = path_inside(directory, relative);st = path.stat();key = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
        if key not in inode_hashes:
            inode_hashes[key] = digest(path)
        require(st.st_size == entry['bytes'] and inode_hashes[key] == entry['sha256'], 'Logical raw bytes/SHA: ' + relative)
        if entry.get('storage_alias_of'):
            alias = path_inside(directory, entry['storage_alias_of'])
            require((alias.stat().st_dev, alias.stat().st_ino) == (st.st_dev, st.st_ino), 'Declared hardlink alias differs')
    capture = load(directory / 'capture.json')
    require(manifest['capture_json_sha256'] == digest(directory / 'capture.json'), 'Captured JSON identity')
    require(capture['capture_kind'] == 'myds_prefill_attention_multihead' and len(capture['cases']) == 3 and
            [c['layer'] for c in capture['cases']] == list(LAYERS), 'Three real prefill layer captures')
    original = load(root / 'packed-capture' / 'original-inputs.json')
    require(digest(root / 'packed-capture' / 'original-inputs.json') == capture['source']['original_inputs_sha256'], 'Original capture descriptor binding')
    provider = original['actual_provider']
    require(provider.get('ordinary_production_prefill') is True and provider['head_first_attempts'] == provider['head_first_encoded'] and
            provider['head_first_encoded'] >= 3 and provider['gather_calls'] >= provider['head_first_encoded'] and
            all(provider[k] == 0 for k in ('head_first_failures', 'head_first_cached_fallbacks', 'matrix_encoded', 'matrix_fallbacks')),
            'Actual capture did not use the original gathered head-first provider')
    require(capture['source']['teacher_forced_tokens_used'] == 4096 and capture['source']['performance_from_capture'] is False and
            capture['source']['expanded_working_set_required'] is True, 'Capture source/working-set provenance')
    require(capture['source']['effective_environment_sha256'] == digest(root / 'effective-env.json') and
            capture['source']['request_sha256'] == digest(root / 'request.json') and
            capture['source']['token_file_sha256'] == digest(root / 'capture-tokens.txt'), 'Rendered request/token/env source bindings')
    require(len((root / 'capture-tokens.txt').read_text().split()) == 4096, '4096 real input token IDs')
    case_reports = []
    finite_inodes = set()
    payload_kinds = {'args200': 'args200', 'q_f32': 'q.f32', 'kv_unique_f16': 'kv-unique.f16',
                     'kv_row_ids_u32': 'kv-rows.u32', 'mask_f16': 'mask-compact.f16', 'sinks_f32': 'sinks.f32'}
    for index, case in enumerate(capture['cases']):
        require(all(case[k] == v for k, v in dict(base_pos=0, c_start=2048, rows=ROWS, keys=640, ratio=2, n_comp=2048, top_k=512).items()),
                'Real complete second-chunk coverage')
        logical = {key: directory / f'capture-case-{index:03d}-{suffix}' for key, suffix in payload_kinds.items()}
        for key, path in logical.items():
            require(path.name in seen and path.stat().st_size == case[key]['bytes'] and digest(path) == case[key]['sha256'], 'Capture logical payload binding: ' + key)
        stride = validate_args(logical['args200'].read_bytes())
        finite_written(logical['q_f32'], ROWS * QUERY_BYTES, written=False)
        finite_written(logical['sinks_f32'], 256, written=False)
        coverage = mask_coverage(logical['mask_f16'])
        require(coverage == manifest['cases'][index]['mask_coverage'] and coverage['minimum_valid_picks'] == 512 and
                coverage['minimum_valid_window'] == 128 and coverage['full_512_pick_queries'] == ROWS,
                'Not all captured queries have real 128+512 active keys')
        reconstructed = expanded_kv_sha(logical['kv_unique_f16'], logical['kv_row_ids_u32'])
        require(reconstructed == case['kv_expanded_sha256'] == manifest['cases'][index]['kv_expanded_sha256'], 'Expanded full working-set SHA')
        source = next(x for x in original['payloads'] if x['layer'] == case['layer'])
        require(source['kv']['sha256'] == reconstructed and source['kv']['bytes'] == ROWS * 655360 and
                source['args']['sha256'] == case['args200']['sha256'] and source['q']['sha256'] == case['q_f32']['sha256'] and
                source['mask']['sha256'] == case['mask_f16']['sha256'] and source['sinks']['sha256'] == case['sinks_f32']['sha256'],
                'Packed/replayed bytes differ from original observer descriptors')
        oracle = directory / f'parity-case-{index:03d}-provider-0-output.f32'
        for p in range(4):
            path = directory / f'parity-case-{index:03d}-provider-{p}-output.f32'
            require(path.name in seen and path.with_suffix('.guards').read_bytes() == b'\xa5' * 128, 'Output manifest/guards')
            key = (path.stat().st_dev, path.stat().st_ino)
            if key not in finite_inodes:
                finite_written(path, ROWS * QUERY_BYTES);finite_inodes.add(key)
            # Byte comparison still runs for every logical provider, even when
            # storage aliases share an inode. SHA equality is not parity.
            equal_files(oracle, path)
        case_reports.append(dict(case=index, layer=case['layer'], rows=ROWS, keys=640, full_output_bytes=ROWS * QUERY_BYTES,
                                 expanded_kv_bytes=ROWS * 655360, expanded_kv_sha256=reconstructed,
                                 original_mask_stride=stride, mask_coverage=coverage, full_output_f32_bit_exact=True))
    require(len(manifest['bindings']) == 4, 'Four selected providers')
    for p, binding in enumerate(manifest['bindings']):
        batch = (1, 1, 2, 4)[p]
        require(binding['provider'] == p and binding['symbol'] == SYMBOLS[p] and binding['head_batch'] == batch and
                binding['library'] == ('production' if p == 0 else 'probe') and binding['groups_x'] == 64 // batch and
                binding['threads'] == [32, batch, 1] and binding['args_bytes'] == 200 and
                binding['ordered_constants'] == FC and binding['logical_nsg'] == binding['logical_nwg'] == 1 and
                binding['dynamic_threadgroup_bytes'] == (3328, 3328, 23040, 29696)[p] and
                binding['max_threads'] >= 32 * batch and binding['thread_execution_width'] == 32,
                'Actual selected PSO/FC/geometry/scratch identity')
    rows = [json.loads(line) for line in (root / 'component.stdout').read_text().splitlines() if line.strip()]
    complete = [r for r in rows if r.get('kind') == 'complete']
    require(len(complete) == 1 and rows[-1] == complete[0] and complete[0]['cases'] == 3 and complete[0]['providers'] == 4 and
            complete[0]['attention_output_checked'] is True and complete[0]['full_f32_bit_exact'] is True, 'Final component completion')
    parity = [r for r in rows if r.get('kind') == 'parity']
    require(len(parity) == 12 and {(r['case'], r['provider']) for r in parity} == {(i, p) for i in range(3) for p in range(4)} and
            all(r['full_f32_bit_exact'] and r['finite_guards_written'] and r['completed_cb'] and r['mismatches'] == 0 and
                r['rows'] == ROWS and r['keys'] == 640 and r['output_bytes'] == ROWS * QUERY_BYTES for r in parity), 'Complete parity records')
    timing = [r for r in rows if r.get('kind') == 'timing'];require(len(timing) == 360, 'Exactly 360 warm/formal timing CBs')
    expected_order = [(case, variant, control, sample, arm) for case in range(3) for variant in (2, 4)
                      for control in ('production', 'probe') for sample in range(10) for arm in ('A1', 'B', 'A2')]
    require([(r['case'], r['variant'], r['control_kind'], r['sample'], r['arm']) for r in timing] == expected_order,
            'Every warm/formal sample must execute the frozen interleaved A1/B/A2 order')
    groups = defaultdict(list);last_end = 0
    for row in timing:
        variant, control, arm = row['variant'], row['control_kind'], row['arm']
        expected_provider = (2 if variant == 2 else 3) if arm == 'B' else (0 if control == 'production' else 1)
        require(variant in (2, 4) and control in ('production', 'probe') and arm in ('A1', 'B', 'A2') and
                row['provider'] == expected_provider and row['symbol'] == SYMBOLS[expected_provider] and row['mode'] == 'concurrent' and
                row['case'] in range(3) and row['layer'] == LAYERS[row['case']] and
                row['rows'] == ROWS and row['keys'] == 640 and row['commits'] == row['dispatches'] == 1 and
                row['completed_cb'] and row['full_f32_bit_exact'], 'Timing provider/working-set/CB mismatch')
        start, end = row['gpu_start_s'], row['gpu_end_s']
        require(math.isfinite(start) and math.isfinite(end) and start >= last_end and end > start > 0,
                'Invalid/nonmonotone full CB GPU timestamps')
        require(close(row['gpu_us_per_chunk'], (end - start) * 1e6), 'GPU duration must be recomputed from raw timestamps')
        last_end = end;groups[row['case'], control, variant, arm].append(row)
    summaries = [r for r in rows if r.get('kind') == 'summary'];require(len(summaries) == 12, 'Twelve case/control/candidate summaries')
    summary_reports = []
    for case in range(3):
        for control in ('production', 'probe'):
            for variant in (2, 4):
                med = {}
                for arm in ('A1', 'B', 'A2'):
                    samples = groups[case, control, variant, arm]
                    require(len(samples) == 10 and {r['sample'] for r in samples} == set(range(10)) and
                            all(r['warmup'] == (r['sample'] == 0) for r in samples), 'One warmup plus nine formal samples')
                    med[arm] = statistics.median(r['gpu_us_per_chunk'] for r in samples if not r['warmup'])
                a1, b, a2 = (med[k] for k in ('A1', 'B', 'A2'));drift = (a2 / a1 - 1) * 100
                saved = (1 - b / min(a1, a2)) * 100;average = (1 - b / ((a1 + a2) / 2)) * 100
                matches = [r for r in summaries if (r['case'], r['control_kind'], r['variant']) == (case, control, variant)]
                require(len(matches) == 1, 'Unique summary grouping');published = matches[0]
                require(published['layer'] == LAYERS[case] and published['mode'] == 'concurrent' and
                        published['control_symbol'] == CONTROL and published['candidate_symbol'] == SYMBOLS[2 if variant == 2 else 3],
                        'Summary layer/selected provider differs from verified raw timing')
                require(all(close(published[k], v) for k, v in dict(a1_us=a1, b_us=b, a2_us=a2,
                            control_drift_percent=drift, conservative_time_saved_percent=saved, time_saved_percent=average).items()),
                        'Published median/performance differs from raw CB times')
                summary_reports.append(dict(case=case, layer=LAYERS[case], control_kind=control, variant=variant,
                    a1_us=a1, b_us=b, a2_us=a2, control_drift_percent=drift, conservative_time_saved_percent=saved,
                    time_saved_percent=average, performance_pass=abs(drift) <= 2 and saved > 2))
    stage = load(root / 'stage-inputs.json');stage_roots = [Path(p).parent for p in stage if Path(p).name == 'Makefile']
    require(len(stage_roots) == 1, 'Unique frozen source checkout');stage_root = stage_roots[0]
    bins, source_count = {}, 0
    with tarfile.open(source_archive, 'r:*') as archive:
        for path, expected in stage.items():
            p = Path(path)
            if not p.is_relative_to(stage_root):
                require(p.name == 'protocol.json' and digest(root / 'protocol.json') == expected, 'Unexpected source outside frozen checkout')
                continue
            relative = str(p.relative_to(stage_root))
            if relative.startswith('bin/'):
                bins[p.name] = expected;continue
            require(archive_member(archive, relative) == expected, 'Frozen source archive mismatch: ' + relative);source_count += 1
    required = {'bench_attention_multihead', 'capture_attention_multihead', 'count_chat_tokens', 'myds.metallib'}
    require(set(bins) == required, 'Four executed artifacts must be frozen')
    with tarfile.open(artifacts_archive, 'r:*') as archive:
        for name, expected in bins.items():
            require(archive_member(archive, 'bin/' + name) == expected, 'Executed binary archive mismatch: ' + name)
    require(manifest['binary_sha256'] == bins['bench_attention_multihead'] and manifest['probe_metallib_sha256'] == bins['myds.metallib'] and
            capture['source']['capture_binary_sha256'] == bins['capture_attention_multihead'] and
            capture['source']['capture_metallib_sha256'] == bins['myds.metallib'], 'Execution artifact/source provenance binding')
    winners = [variant for variant in (2, 4) if all(r['performance_pass'] for r in summary_reports
               if r['control_kind'] == 'production' and r['variant'] == variant)]
    return dict(schema_version=1, reviewer='independent_cpu_raw_and_timeline_review', gpu_work_submitted=False,
                evidence_valid=True, correctness_pass=True, performance_gate_pass=bool(winners), winning_variants=winners,
                decision='eligible_for_fullmodel_gate' if winners else 'reject_component_performance',
                fullmodel_run=False, production_adoption=False, cold_dram_proven=False,
                raw_logical_files_verified=len(files), raw_unique_inodes_hashed=len(inode_hashes), cases=case_reports,
                timing_command_buffers_verified=360, timing_interleaved_a1_b_a2_verified=True,
                recomputed_performance=summary_reports, restoration_exact_including_nice=True,
                original_pid=before['pid'], restored_pid=restored['pid'],
                source_archive_sha256=digest(source_archive), source_members_verified=source_count,
                artifacts_archive_sha256=digest(artifacts_archive), executed_artifact_sha256=bins,
                manifest_sha256=digest(directory / 'artifact-manifest.json'), component_stdout_sha256=digest(root / 'component.stdout'),
                expanded_kv_storage='Losslessly reconstructed 1.34GB per case, SHA bound to original observer; expanded files need not be duplicated',
                output_storage='Every logical provider byte-compared; identical bytes may share storage hardlinks',
                reviewer_sha256=digest(Path(__file__)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source-archive', type=Path, required=True)
    parser.add_argument('--artifacts-archive', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), 'Refuse review output overwrite')
    report = review(args.root.resolve(), args.source_archive, args.artifacts_archive)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('recomputed_performance', 'cases')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
