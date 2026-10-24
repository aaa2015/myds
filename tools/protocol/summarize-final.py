#!/usr/bin/env python3
"""Strict CPU-only summarizer for a completed six-measurement final A/B window.

No GPU, subprocess, network, or model access. No partially written result is
accepted. Output count and decode denominator come from each measured group.
"""
import argparse
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

WORKLOADS = ('short', 'normal', 'long')
VARIANTS = ('base', 'final')
NAMES = tuple(f'{v}-{w}' for w in WORKLOADS for v in VARIANTS)
MATCH_META = ('model', 'request_fnv1a64', 'prompt_fnv1a64', 'request_bytes',
              'prompt_bytes', 'max_new_tokens', 'max_seq_len', 'temperature',
              'top_p', 'top_k', 'min_p', 'repetition_penalty', 'seed', 'qos',
              'prompt_renderer', 'clock', 'transport', 'decode_basis')
TASK_ROOT = Path(__file__).resolve().parents[1]


class AuditError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def read_records(path):
    require(path.is_file(), f'Missing file: {path}')
    text = path.read_text()
    require(text.endswith('\n'), f'Incomplete final line: {path}')
    try:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    except json.JSONDecodeError as exc:
        raise AuditError(f'Invalid JSON in {path}: {exc}') from exc
    require(records and all(isinstance(r, dict) for r in records), f'Empty/non-object records: {path}')
    return records


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def token_hash(ids):
    # The harness hashes each int32 token in explicit little-endian order.
    value = 14695981039346656037
    for token in ids:
        for byte in token.to_bytes(4, byteorder='little', signed=False):
            value = ((value ^ byte) * 1099511628211) & 0xffffffffffffffff
    return f'{value:016x}'


def numeric(value, label, positive=True):
    require(isinstance(value, (int, float)) and not isinstance(value, bool) and
            math.isfinite(value), f'Invalid number: {label}')
    require(value > 0 if positive else value >= 0, f'Invalid sign: {label}')
    return value


def close(a, b, label, abs_tol=0.000002):
    require(math.isclose(a, b, rel_tol=1e-8, abs_tol=abs_tol),
            f'Inconsistent derived metric {label}: {a} vs {b}')


def environment_off(env, label):
    require(isinstance(env, dict), f'Missing environment: {label}')
    for key in ('MYDS_PREFIX_CACHE', 'MYDS_KV_DSK', 'MYDS_PHASE_PROFILE'):
        require(env.get(key) == '0', f'{label}: {key} must be explicitly 0')
    require(env.get('MYDS_PHASE_PROFILE_FILE') in (None, ''), f'{label}: phase profile file supplied')
    # Legacy prefill profiling checks getenv presence, so even string0 is on.
    for key, value in env.items():
        if key.startswith('MYDS_PROFILE_') or key == 'MYDS_MOE_STEP_PROF':
            require(value is None, f'{label}: legacy profiling flag present: {key}={value}')


def validate_window(directory):
    path = directory / 'window.jsonl'
    events = read_records(path)
    for e in events:
        kind = e.get('kind', '')
        require(isinstance(kind, str), 'Window kind is not a string')
        require(not any(word in kind.lower() for word in ('error', 'timeout', 'interrupt')),
                f'Window failure event: {kind}')
        require(not e.get('error'), f'Window event contains error: {kind}')
    kinds = [e['kind'] for e in events]
    for kind in ('window_created', 'preflight_passed', 'production_stop',
                 'restoration_start', 'restoration_health', 'window_complete'):
        require(kinds.count(kind) == 1, f'Window must contain one {kind}')
    require(kinds[-1] == 'window_complete', 'Window has not reached final completion')
    starts = [e for e in events if e['kind'] == 'measurement_start']
    ends = [e for e in events if e['kind'] == 'measurement_end']
    exact = [e for e in events if e['kind'] == 'exact_output_pass']
    require(len(starts) == len(ends) == 6, 'Expected exactly six measurements')
    require([e.get('name') for e in starts] == list(NAMES), 'Unexpected measurement start names/order')
    require([e.get('name') for e in ends] == list(NAMES), 'Unexpected measurement end names/order')
    require(len(exact) == 3 and {e.get('request') for e in exact} == set(WORKLOADS),
            'Expected exactly three unique pair output-pass events')
    for name in NAMES:
        s = next(e for e in starts if e['name'] == name)
        e = next(e for e in ends if e['name'] == name)
        require(e.get('returncode') == 0, f'{name}: nonzero measurement returncode')
        numeric(e['elapsed_s'], name + '.elapsed_s')
        environment_off(s.get('environment'), name + '.event')
        require(events.index(s) < events.index(e), f'{name}: end precedes start')
    measurement_order = [(e['kind'], e.get('name')) for e in events if e['kind'].startswith('measurement_')]
    expected_order = [pair for name in NAMES for pair in [('measurement_start', name), ('measurement_end', name)]]
    require(measurement_order == expected_order, 'Measurements overlap or event order changed')
    health = next(e for e in events if e['kind'] == 'restoration_health')
    require(health.get('ready') is True, 'Production restoration health is not ready')
    require(health.get('restore_returncode') in (None, 0), 'Restoration process failed')
    require(kinds.index('production_stop') < kinds.index('measurement_start'), 'Production stop order invalid')
    require(kinds.index('window_created') < kinds.index('preflight_passed') < kinds.index('production_stop'),
            'Window/preflight/production-stop order invalid')
    require(kinds.index('restoration_start') > max(i for i, k in enumerate(kinds) if k == 'measurement_end'),
            'Restoration started before measurements ended')
    require(kinds.index('restoration_start') < kinds.index('restoration_health') < kinds.index('window_complete'),
            'Restoration completion order invalid')
    times = [datetime.datetime.fromisoformat(e['utc']) for e in events]
    require(all(a <= b for a, b in zip(times, times[1:])), 'Nonmonotonic window event timestamps')
    return events, starts, exact, health


def stats(values):
    return {'median': statistics.median(values), 'min': min(values), 'max': max(values), 'values': values}


def validate_group(directory, name, start_event):
    path = directory / (name + '-metrics.jsonl')
    rs = read_records(path)
    kinds = [r.get('kind') for r in rs]
    require(kinds.count('metadata') == kinds.count('load') == 1, f'{name}: metadata/load count')
    require(set(kinds) == {'metadata', 'load', 'sample'}, f'{name}: unknown/error metrics kind')
    meta = next(r for r in rs if r['kind'] == 'metadata')
    require(meta['repeats'] == 3 and meta['warmup_runs'] == 1, f'{name}: repeat protocol')
    require(meta['decode_basis'] == 'first_to_last_callback/(tokens-1)', f'{name}: decode denominator changed')
    require(meta['transport'] == 'direct_byte_callback', f'{name}: transport changed')
    environment_off(meta.get('environment'), name + '.metrics')
    for key, value in start_event['environment'].items():
        if key in meta['environment']:
            require(meta['environment'][key] == value, f'{name}: event/metadata env mismatch for {key}')
    all_samples = [r for r in rs if r['kind'] == 'sample']
    require(len(all_samples) == 4, f'{name}: expected four sample records')
    require(all(isinstance(r.get('warmup'), bool) for r in all_samples), f'{name}: invalid warmup marker')
    require([r['run'] for r in all_samples] == [0, 1, 2, 3], f'{name}: duplicate/missing/order of runs')
    warm = [r for r in all_samples if r['warmup']]
    formal = [r for r in all_samples if not r['warmup']]
    require(len(warm) == 1 and warm[0]['run'] == 0 and len(formal) == 3, f'{name}: warm/formal partition')
    ref = warm[0]
    for r in all_samples:
        label = f'{name}.run{r["run"]}'
        require(r.get('generation_ok') is True and r.get('sample_valid') is True, label + ': invalid generation/sample')
        require(r.get('output_equal_to_warmup') is True, label + ': warm output marker differs')
        require(r['cached_tokens'] == 0, label + ': cached prefill present')
        pf = numeric(r['prefill_tokens'], label + '.prefill_tokens')
        require(isinstance(pf, int), label + ': noninteger prefill count')
        require(r['prompt_tokens'] == pf + 1, label + ': unexpected prefill/prompt relationship')
        out = r['output_tokens']
        require(isinstance(out, int) and not isinstance(out, bool) and out >= 2, label + ': insufficient output for decode measurement')
        require(out <= meta['max_new_tokens'], label + ': output exceeds configured cap')
        require(isinstance(r['token_ids'], list) and len(r['token_ids']) == out and
                all(isinstance(t, int) and not isinstance(t, bool) and 0 <= t <= 0x7fffffff for t in r['token_ids']),
                label + ': invalid raw token array')
        require(r['token_ids_fnv1a64'] == token_hash(r['token_ids']), label + ': raw token/hash mismatch')
        require(r['decode_intervals'] == out - 1, label + ': decode interval count')
        require(isinstance(r['callback_intervals_ms'], list) and len(r['callback_intervals_ms']) == out - 1,
                label + ': callback interval array count')
        for value in r['callback_intervals_ms']:
            # Schema 2 can emit both verified batch tokens on the same clock
            # tick. Retain those zero gaps; total decode time remains positive.
            numeric(value, label + '.callback_interval', positive=meta.get('protocol_version', 1) != 2)
        require(r.get('stop_reason') in ('length', 'eos'), label + ': unexpected stop reason')
        require(r['stop_reason'] != 'length' or out == meta['max_new_tokens'],
                label + ': length stop before configured output cap')
        require(isinstance(r['output_bytes'], int) and not isinstance(r['output_bytes'], bool) and
                r['output_bytes'] >= 0, label + ': invalid output byte count')
        for field in ('token_ids', 'token_ids_fnv1a64', 'output_bytes_fnv1a64', 'output_bytes',
                      'prefill_tokens', 'prompt_tokens', 'output_tokens', 'decode_intervals', 'stop_reason'):
            require(r[field] == ref[field], label + ': output/count changed in ' + field)
        for field in ('token_ids_fnv1a64', 'output_bytes_fnv1a64'):
            require(isinstance(r[field], str) and len(r[field]) == 16 and
                    all(c in '0123456789abcdef' for c in r[field]), label + ': invalid hash field ' + field)
        for field in ('prefill_s', 'decode_s', 'wall_s', 'ttft_s', 'first_callback_s',
                      'last_callback_s', 'prefill_ms_per_token', 'decode_ms_per_token'):
            numeric(r[field], label + '.' + field)
        close(r['prefill_ms_per_token'], r['prefill_s'] * 1000 / pf, label + '.prefill_ms_per_token')
        close(r['decode_ms_per_token'], r['decode_s'] * 1000 / (out - 1), label + '.decode_ms_per_token')
        close(sum(r['callback_intervals_ms']) / 1000, r['decode_s'], label + '.callback_sum')
        close(r['last_callback_s'] - r['first_callback_s'], r['decode_s'], label + '.callback_endpoints')
        close(r['first_callback_s'], r['ttft_s'], label + '.ttft')
        require(r['last_callback_s'] <= r['wall_s'] + 0.000002, label + ': callback after measured wall')
    metric_stats = {field: stats([r[field] for r in formal]) for field in
                    ('prefill_ms_per_token', 'decode_ms_per_token', 'prefill_s', 'decode_s', 'ttft_s', 'wall_s')}
    return {'name': name, 'path': str(path), 'sha256': sha(path), 'metadata': meta,
            'observed_window_environment': start_event['environment'],
            'formal_runs': 3, 'warmup_runs': 1, 'prompt_tokens': ref['prompt_tokens'],
            'prefill_tokens': ref['prefill_tokens'], 'output_tokens': ref['output_tokens'],
            'decode_intervals': ref['decode_intervals'], 'stop_reason': ref['stop_reason'],
            'token_ids_fnv1a64': ref['token_ids_fnv1a64'], 'output_bytes_fnv1a64': ref['output_bytes_fnv1a64'],
            'output_bytes': ref['output_bytes'], 'warm_formal_raw_tokens_and_byte_hash_equal': True,
            'stats': metric_stats}, all_samples


def summarize(directory):
    events, starts, exact, health = validate_window(directory)
    groups, raw = {}, {}
    for name in NAMES:
        groups[name], raw[name] = validate_group(directory, name, next(e for e in starts if e['name'] == name))
    comparisons, csv_rows = [], []
    for workload in WORKLOADS:
        b, f = groups['base-' + workload], groups['final-' + workload]
        for key in MATCH_META:
            require(b['metadata'][key] == f['metadata'][key], f'{workload}: input/config mismatch {key}')
        require(b['observed_window_environment'] == f['observed_window_environment'],
                f'{workload}: differing explicit MYDS environment')
        for x, y in zip(raw['base-' + workload], raw['final-' + workload]):
            require(x['run'] == y['run'] and x['warmup'] == y['warmup'], f'{workload}: pair run mismatch')
            for field in ('token_ids', 'token_ids_fnv1a64', 'output_bytes_fnv1a64', 'output_bytes',
                          'prefill_tokens', 'prompt_tokens', 'output_tokens', 'decode_intervals', 'stop_reason'):
                require(x[field] == y[field], f'{workload}: pair output mismatch run{x["run"]} {field}')
        event = next(e for e in exact if e['request'] == workload)
        require(event['samples'] == 4 and event['token_ids_fnv1a64'] == b['token_ids_fnv1a64'] and
                event['output_tokens'] == b['output_tokens'], f'{workload}: exact event inconsistent')
        pair = {'workload': workload, 'all4_raw_token_arrays_and_byte_hashes_equal': True,
                'prompt_tokens': b['prompt_tokens'], 'prefill_tokens': b['prefill_tokens'],
                'output_tokens': b['output_tokens'], 'decode_intervals': b['decode_intervals'],
                'token_ids_fnv1a64': b['token_ids_fnv1a64'], 'phases': {}}
        for phase, field in (('prefill', 'prefill_ms_per_token'), ('decode', 'decode_ms_per_token')):
            bs, fs = b['stats'][field], f['stats'][field]
            ratios = [y[field] / x[field] for x, y in zip(raw['base-' + workload][1:], raw['final-' + workload][1:])]
            change = 100 * (fs['median'] / bs['median'] - 1)
            row = {'workload': workload, 'phase': phase, 'prefill_tokens': b['prefill_tokens'],
                   'output_tokens': b['output_tokens'], 'decode_intervals': b['decode_intervals'],
                   'base_median_ms_per_token': bs['median'], 'base_min_ms_per_token': bs['min'],
                   'base_max_ms_per_token': bs['max'], 'final_median_ms_per_token': fs['median'],
                   'final_min_ms_per_token': fs['min'], 'final_max_ms_per_token': fs['max'],
                   'time_saved_pct_from_medians': -change,
                   'paired_by_run_time_saved_pct_median': 100 * (1 - statistics.median(ratios)),
                   'formal_runs_per_variant': 3, 'warmup_runs_per_variant': 1}
            pair['phases'][phase] = dict(row, time_change_pct_from_medians=change,
                                          paired_by_run_ratios=ratios)
            csv_rows.append(row)
        comparisons.append(pair)
    summary = {'audit_pass': True, 'result_directory': str(directory),
               'measurement_scope': 'Direct callback end-to-end model requests, one warmup plus three formal runs per variant/workload. Decode interval is first-to-last callback divided by actual output_tokens-1.',
               'performance_comparison': 'Primary change is ratio of three-run medians. Run-index paired ratios are also reported; the window runs each variant as a separate process and does not interleave baseline/candidate rounds.',
               'groups': groups, 'comparisons': comparisons,
               'window_audit': {'path': str(directory/'window.jsonl'), 'sha256': sha(directory/'window.jsonl'),
                                'measurements': 6, 'measurement_returncodes': [0]*6,
                                'exact_output_pairs': 3, 'restoration_healthy': True,
                                'error_or_timeout_events': 0, 'restoration_health': health},
               'default_selection_note': 'Only recorded explicit environment values are included. Absent environment variables do not establish implementation defaults; inspect the separately attached build/source evidence.'}
    return summary, csv_rows


def external_evidence(path):
    require(path.is_file(), f'Missing external evidence: {path}')
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise AuditError(f'Invalid external evidence {path}: {exc}') from exc
    return {'path': str(path), 'sha256': sha(path), 'attached_as_external_evidence': True}, value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result_directory', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--build-manifest', type=Path, default=TASK_ROOT/'final-build-manifest.json')
    parser.add_argument('--source-verification', type=Path, default=TASK_ROOT/'final-source-verification.json')
    args = parser.parse_args()
    try:
        summary, rows = summarize(args.result_directory.resolve())
        manifest_ref, manifest = external_evidence(args.build_manifest.resolve())
        source_ref, source = external_evidence(args.source_verification.resolve())
        require(manifest.get('forced_rebuild') is True, 'External build manifest is not a forced rebuild')
        require(source.get('all_production_engine_sources_match') is True and source.get('mismatches') == [],
                'External source verification is not clean')
        summary['external_evidence'] = {'build_manifest': manifest_ref, 'source_verification': source_ref,
                                         'implementation_defaults_inferred_from_missing_env': False}
        # Do not create summary files until all checks pass.
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out/'final-summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
        with (args.out/'final-per-token.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(json.dumps({'audit_pass': True, 'out': str(args.out.resolve()),
                          'pairs': [{k:v for k,v in pair.items() if k != 'phases'} for pair in summary['comparisons']]}))
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print('Final result audit FAILED: ' + str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
