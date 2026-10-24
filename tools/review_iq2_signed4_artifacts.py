#!/usr/bin/env python3
"""Independently review an archived signed4 component window using CPU reads only.

Exit 0 means the evidence is internally valid; performance rejection is reported
separately. --live also fingerprints original artifacts and selected GGUF experts.
This program never loads Metal or starts, stops, or messages an inference process.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import struct
import sys
import tarfile
import urllib.request


NATIVE = 'kernel_mul_mv_id_iq2_xxs_pair_swiglu_popcnt_f32'
SYMBOLS = [NATIVE, 'kernel_myds_iq2_v41_shape64_f32',
           'kernel_myds_iq2_v41_signed4_f32', 'probe_iq2_v41_shape64_capture_f32',
           'probe_iq2_v41_signed4_capture_f32']
MID = 6 * 2304 * 4


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path, offset=0, length=None):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        stream.seek(offset)
        while length is None or length:
            block = stream.read(1048576 if length is None else min(length, 1048576))
            if not block:
                break
            h.update(block)
            if length is not None:
                length -= len(block)
    require(length in (None, 0), 'Short hashed file: ' + str(path))
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def finite_written(raw):
    return (len(raw) == MID and
            all(math.isfinite(v[0]) for v in struct.iter_unpack('<f', raw)) and
            all(v[0] != 0xa5a5a5a5 for v in struct.iter_unpack('<I', raw)))


def review(root, live=False, source_archive=None, artifacts_archive=None):
    before, restored, legacy = [load(root / name) for name in
                               ('before.json', 'restored.json', 'summary.json')]
    exact_keys = ('argv_raw_hex', 'myds_env_raw_hex', 'cwd', 'log', 'binary',
                  'binary_sha256', 'metallib', 'metallib_sha256', 'config',
                  'config_sha256', 'effective_myds_env')
    require(all(before[k] == restored[k] for k in exact_keys), 'Restoration identity mismatch')
    require(before['health'] == restored['health'] == {'status': 'ok'}, 'Recorded health mismatch')
    env = before['effective_myds_env']
    require(len(env) == 51, 'Expected 51 effective MYDS keys')
    require(all(env.get(k) == v for k, v in {
        'MYDS_MOE_IQ2_POPCNT_SIGN': '1', 'MYDS_MOE_IQ2_V41': '0',
        'MYDS_MOE_NSG': '2'}.items()), 'Native provider configuration mismatch')
    require(all(env.get(k, '0') == '0' for k in
                ('MYDS_MOE_IQ2_CONSTANT_LUT', 'MYDS_MOE_Q2_SUM_CACHE',
                 'MYDS_MOE_Q2_SUM_CACHE_FUSED')), 'Native provider bypass enabled')
    mm_min = int(env.get('MYDS_MOE_MM_MIN_TOKENS', '32'))
    require(mm_min <= 0 or mm_min > 1 or int(env.get('MYDS_DISABLE_MOE_MM', '0')) != 0,
            'One-token MM provider enabled')
    require(not any(x in before['argv'] for x in ('--pld', '--no-pld')),
            'Unaccounted CLI environment override')
    require(digest(root / 'original-myds.conf') == before['config_sha256'], 'Config backup SHA')
    require(load(root / 'effective-env.json') == env, 'Effective environment archive mismatch')
    for k in ('binary', 'metallib'):
        backup = root / 'production-artifacts' / Path(before[k]['path']).name
        require(digest(backup) == before[k + '_sha256'], 'Production backup SHA: ' + k)
    steps = load(root / 'steps.json')
    require(all(row['returncode'] == 0 for row in steps), 'A recorded command failed')
    require({'trace', 'component-nsg2', 'component-nsg4', 'capture-host-validation'} <=
            {row['name'] for row in steps}, 'Missing required successful commands')
    events = [json.loads(line) for line in (root / 'events.jsonl').read_text().splitlines()]
    require(not any(row['kind'] in ('monitor_failure', 'restoration_failure', 'window_failure')
                    for row in events), 'Recorded window failure')
    capture = load(root / 'capture.json')
    require(len(capture['cases']) == 12, 'Expected 12 real capture cases')
    require({(c['layer'], c['position']) for c in capture['cases']} ==
            {(l, p) for l in (0, 19, 39) for p in (0, 1, 7, 15)}, 'Capture coverage mismatch')
    result = dict(schema_version=1, reviewer='independent_cpu_raw_review', gpu_work_submitted=False,
                  evidence_valid=True, correctness_pass=True, fullmodel_run=False,
                  production_adoption=False, cold_dram_proven=False, scope='repeated real captured component',
                  effective_myds_key_count=len(env), original_pid=before['pid'], restored_pid=restored['pid'],
                  restoration_identity_keys=list(exact_keys), restoration_exact=True,
                  historical_nice_recorded='nice' in before and 'nice' in restored,
                  protocol=dict(primary_nsg=2, layers=[0, 19, 39], max_absolute_drift_percent=2.0,
                                conservative_time_saved_percent_must_exceed=2.0), components=[])
    selected_hashes = {}
    for nsg in (2, 4):
        directory = root / f'component-nsg{nsg}'
        manifest = load(directory / 'artifact-manifest.json')
        require(manifest['status'] == 'complete' and manifest['nsg'] == nsg and
                manifest['samples'] == 9 and manifest['repetitions_per_cb'] == 8 and
                manifest['total_cases'] == 12, 'Incomplete component manifest')
        require(manifest['production_metallib_sha256'] == before['metallib_sha256'], 'Golden library SHA')
        require(manifest['capture_json_sha256'] == digest(root / 'capture.json'), 'Capture JSON SHA')
        names = [entry['path'] for entry in manifest['files']]
        require(len(names) == len(set(names)), 'Duplicate artifact paths')
        for entry in manifest['files']:
            path = directory / entry['path']
            require(path.resolve().is_relative_to(directory.resolve()), 'Artifact path escapes directory')
            require(path.stat().st_size == entry['bytes'] and digest(path) == entry['sha256'],
                    'Raw artifact SHA/size: ' + entry['path'])
        for provider, binding in enumerate(manifest['bindings']):
            require(binding['provider'] == provider and binding['symbol'] == SYMBOLS[provider] and
                    binding['fc600'] == nsg and binding['nr0'] == 4 and
                    binding['groups'] == [2304 // (4 * nsg), 1, 6] and
                    binding['threads'] == [32, nsg, 1] and binding['thread_execution_width'] == 32 and
                    binding['dynamic_threadgroup_bytes'] == (5248 if provider in (2, 4) else 2176) and
                    binding['max_threads'] >= 32 * nsg, 'Provider/ABI/geometry mismatch')
        for case in range(12):
            def raw(provider, kind):
                path = directory / f'parity-case-{case:03d}-provider-{provider}-{kind}.f32'
                data = path.read_bytes()
                require(finite_written(data), 'Nonfinite/poison/short raw F32 output')
                require(path.with_suffix('.guards').read_bytes() == b'\xa5' * 128, 'Output guard corruption')
                return data
            oracle = raw(0, 'mid')
            require(all(raw(p, 'mid') == oracle for p in range(1, 5)), 'Full MID bits differ')
            require(raw(3, 'gate') == raw(4, 'gate') and raw(3, 'up') == raw(4, 'up'),
                    'Full raw gate/up bits differ')
        rows = [json.loads(line) for line in (root / f'component-nsg{nsg}.stdout').read_text().splitlines()]
        parity = [row for row in rows if row.get('kind') == 'parity']
        require(len(parity) == 60 and all(row['full_f32_bit_exact'] and row['completed_cb']
                                       for row in parity), 'Parity command records incomplete')
        require(rows[-1]['kind'] == 'complete' and rows[-1]['gate_up_mid_checked'], 'Missing final completion')
        timing = [row for row in rows if row.get('kind') == 'timing']
        require(len(timing) == 180, 'Expected 180 warm/formal timing command buffers')
        last_end = 0
        for row in timing:
            require(row['completed_cb'] and row['full_f32_bit_exact'] and row['commits'] == 1 and
                    row['dispatches'] == 32 and row['gpu_start_s'] >= last_end and
                    row['gpu_end_s'] > row['gpu_start_s'], 'Invalid timing command record')
            recomputed = (row['gpu_end_s'] - row['gpu_start_s']) * 1e6 / 32
            require(math.isclose(row['gpu_us_per_dispatch'], recomputed, rel_tol=1e-12), 'GPU timing arithmetic')
            last_end = row['gpu_end_s']
        summaries = []
        for layer in (0, 19, 39):
            med = {}
            for arm in ('A1', 'B', 'A2'):
                group = [row for row in timing if row['layer'] == layer and
                         row['control_provider'] == 0 and row['arm'] == arm and not row['warmup']]
                require(len(group) == 9 and {r['sample'] for r in group} == set(range(1, 10)),
                        'Missing nine formal A1/B/A2 samples')
                med[arm] = statistics.median(row['gpu_us_per_dispatch'] for row in group)
            a1, b, a2 = (med[k] for k in ('A1', 'B', 'A2'))
            drift, saved = (a2 / a1 - 1) * 100, (1 - b / min(a1, a2)) * 100
            saved_average = (1 - b / ((a1 + a2) / 2)) * 100
            published = [r for r in rows if r.get('kind') == 'summary' and
                         r['layer'] == layer and r['control_symbol'] == NATIVE]
            require(len(published) == 1 and math.isclose(published[0]['conservative_time_saved_percent'], saved,
                                                       rel_tol=1e-12), 'Published summary mismatch')
            summaries.append(dict(layer=layer, a1_us=a1, b_us=b, a2_us=a2,
                                  control_drift_percent=drift, conservative_time_saved_percent=saved,
                                  time_increase_percent=-saved_average,
                                  performance_pass=abs(drift) <= 2 and saved > 2))
        component = dict(nsg=nsg, raw_files_verified=len(names), parity_cases=12,
                         parity_providers=5, full_output_f32_bit_exact=True,
                         timing_command_buffers=len(timing), native_control=summaries)
        result['components'].append(component)
        if live:
            for key in ('binary', 'probe_metallib', 'production_metallib'):
                require(digest(manifest[key + '_path']) == manifest[key + '_sha256'], 'Live component artifact SHA')
            model = manifest['model']
            require(digest(model['path'], length=model['header_bytes']) == model['header_sha256'], 'Live model prefix SHA')
            for tensor in manifest['tensors']:
                for expert in tensor['selected_experts']:
                    offset = tensor['file_offset'] + expert['id'] * expert['bytes']
                    key = (model['path'], offset, expert['bytes'])
                    if key not in selected_hashes:
                        selected_hashes[key] = digest(key[0], offset, key[2])
                    require(selected_hashes[key] == expert['sha256'], 'Live selected expert SHA')
    result['performance_gate_pass'] = all(row['performance_pass'] for row in result['components'][0]['native_control'])
    result['decision'] = 'eligible_for_fullmodel_gate' if result['performance_gate_pass'] else 'reject_component_performance'
    result['legacy_runner_status'] = legacy['status']
    result['legacy_status_means_performance_pass'] = False
    result['model_identity_scope'] = 'model size + 1 MiB prefix SHA + selected real expert SHA; no full-model-file SHA'
    if source_archive:
        stage = load(root / 'stage-inputs.json')
        stage_roots = [Path(path).parent for path in stage if Path(path).name == 'Makefile']
        require(len(stage_roots) == 1, 'Ambiguous executed source root')
        checked, binaries = 0, []
        with tarfile.open(source_archive, 'r:*') as archive:
            for path, expected in stage.items():
                relative = str(Path(path).relative_to(stage_roots[0]))
                if relative.startswith('bin/'):
                    binaries.append(dict(path=relative, sha256=expected))
                    continue
                stream = archive.extractfile(relative)
                require(stream is not None and hashlib.sha256(stream.read()).hexdigest() == expected,
                        'Archived executed source SHA: ' + relative)
                checked += 1
        result['executed_source_archive_sha256'] = digest(source_archive)
        result['executed_source_members_verified'] = checked
        result['executed_binary_identities'] = binaries
        result['executed_binaries_archived_by_source_tar'] = False
    if artifacts_archive:
        stage = load(root / 'stage-inputs.json')
        expected_bins = {Path(path).name: expected for path, expected in stage.items()
                         if Path(path).parent.name == 'bin'}
        require(set(expected_bins) == {'bench_iq2_signed4', 'trace_model', 'count_chat_tokens', 'myds.metallib'},
                'Expected four executed artifacts')
        with tarfile.open(artifacts_archive, 'r:*') as archive:
            for name, expected in expected_bins.items():
                stream = archive.extractfile('bin/' + name)
                require(stream is not None and hashlib.sha256(stream.read()).hexdigest() == expected,
                        'Archived executed artifact SHA: ' + name)
        for nsg in (2, 4):
            manifest = load(root / f'component-nsg{nsg}/artifact-manifest.json')
            require(manifest['binary_sha256'] == expected_bins['bench_iq2_signed4'] and
                    manifest['probe_metallib_sha256'] == expected_bins['myds.metallib'],
                    'Component artifact identities differ from execution archive')
        require(capture['source']['trace_binary_sha256'] == expected_bins['trace_model'] and
                capture['source']['trace_metallib_sha256'] == expected_bins['myds.metallib'],
                'Trace artifact identities differ from execution archive')
        result['executed_artifacts_archive_sha256'] = digest(artifacts_archive)
        result['executed_artifact_members_verified'] = len(expected_bins)
    if live:
        for key in ('binary', 'metallib', 'config'):
            require(digest(before[key]['path']) == before[key + '_sha256'], 'Live restored artifact SHA')
        result['live_selected_expert_hashes_verified'] = len(selected_hashes)
        result['live_restored_nice'] = os.getpriority(os.PRIO_PROCESS, restored['pid'])
        with urllib.request.urlopen(f"http://127.0.0.1:{before['port']}/health", timeout=3) as response:
            require(json.load(response) == {'status': 'ok'}, 'Live restored health')
        precedence = []
        for relative in ('metal/myds.metallib', 'bin/myds.metallib', 'build/myds.metallib'):
            path = Path(before['cwd']) / relative
            require(not path.exists() or path.resolve() == Path(before['metallib']['path']).resolve(),
                    'An earlier live metallib shadows the assumed golden')
            precedence.append(dict(path=str(path), exists=path.exists()))
        result['live_library_precedence'] = precedence
        stage = load(root / 'stage-inputs.json')
        require(all(digest(path) == expected for path, expected in stage.items()), 'Live stage source/artifact SHA changed')
        result['live_stage_hashes_verified'] = len(stage)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--source-archive', type=Path)
    parser.add_argument('--artifacts-archive', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    try:
        result = review(args.root.resolve(), args.live, args.source_archive, args.artifacts_archive)
        rc = 0
    except (ValueError, KeyError, OSError, TypeError, AssertionError) as error:
        result = dict(schema_version=1, evidence_valid=False, gpu_work_submitted=False, error=str(error))
        rc = 2
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if args.out:
        with args.out.open('x') as stream:
            stream.write(text)
    print(text, end='')
    return rc


if __name__ == '__main__':
    sys.exit(main())
