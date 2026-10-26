#!/usr/bin/env python3
"""Independent CPU-only review after the integration timing window is finished.

No project reader is imported. Original JSON decimal timings become Fractions;
every retained raw payload is reread, hashed and compared byte for byte with the
workload's first reference sample. Never run this I/O during Q/P/HTTP timing.
"""
import argparse
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import struct
import time

NATIVE = '2c7b2e7d9da8b9834bc9a5d527e4ae255ec55f0a434459137756e72cc15e18a2'
READER = '0366e87c5ee1b9f550182b8944edaa95d8df2d388c2108977a03bfcd89b399b1'
CONFIG = '735e0dc73fbbf7483dba0bffe4ad379a4dab910bb734d3a6708df65c8edc16d7'
GOLD = '3837102396703da59e0eebbc0de1fdc58b68b3a3a5ec5d07504b351c67a29ba9'
CANDIDATE = 'fe55793346e15119fce4e0f40d4bf6d3023fc3694ffc231e66b8a89e3ae0f74d'
COUNTERS = ('requests', 'encoded', 'fallbacks', 'failures', 'f32_dispatches', 'bf16_dispatches')
METRICS = ('prefill_s', 'ttft_s', 'decode_ms_per_token')
SCALARS = ('ctx_pos', 'rng_state', 'max_seq_len', 'max_new_tokens', 'seed_u32', 'kv_capacity', 'pcache_len',
           'anchor_n', 'continue_from_cache', 'v41_valid', 'next_position', 'engram_position',
           'snapshot_allocated_bytes', 'prefill_capacity', 'low_kv_enabled', 'anchor_slots', 'anchor_cap')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def pairs(items):
    result = {}
    for key, value in items:
        need(key not in result, 'duplicate JSON field: ' + key)
        result[key] = value
    return result


def loads(raw):
    def bad(value):
        raise ValueError('non-finite JSON: ' + value)
    return json.loads(raw, parse_float=Decimal, object_pairs_hook=pairs, parse_constant=bad)


def read(path):
    return loads(Path(path).read_bytes())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1048576), b''):
            h.update(block)
    return h.hexdigest()


def descriptor(path):
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), 'regular descriptor: ' + str(path))
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def check_descriptor(item):
    need(descriptor(item['path']) == item, 'artifact identity: ' + item['path'])


def contained(root, name):
    p = Path(name)
    need(not p.is_absolute() and '..' not in p.parts and str(p) == name, 'safe relative raw path')
    path = root / p
    need(not path.is_symlink() and path.is_file() and root.resolve() in path.resolve().parents, 'contained raw file')
    return path


def fnv(raw):
    value = 14695981039346656037
    for byte in raw:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return f'{value:016x}'


def fraction(value):
    need(type(value) in (int, Decimal) and value > 0, 'positive original JSON decimal')
    return Fraction(value)


def rational(value):
    return dict(numerator=value.numerator, denominator=value.denominator,
                decimal=str(Decimal(value.numerator) / Decimal(value.denominator)))


def performance(arms):
    need(list(arms) == ['A1', 'B', 'A2'], 'fixed performance arm order')
    output = {}
    for metric in METRICS:
        medians = {arm: sorted(fraction(s[metric]) for s in samples[1:])[1] for arm, samples in arms.items()}
        a1, b, a2 = (medians[a] for a in ('A1', 'B', 'A2'))
        saved, drift = 100 * (1 - b / min(a1, a2)), 100 * (a2 / a1 - 1)
        passed = abs(drift) <= 1 and (saved > 2 if metric == 'prefill_s' else saved >= Fraction(-1, 2))
        output[metric] = dict(passed=passed, medians={a: rational(v) for a, v in medians.items()},
            saved_percent=rational(saved), control_drift_percent=rational(drift),
            original_samples_including_warmup={a: [str(s[metric]) for s in samples] for a, samples in arms.items()})
    return dict(passed=all(v['passed'] for v in output.values()), metrics=output)


def verify_raw(path, expected, reference=None):
    """Hash actual bytes and compare them in the same pass; no SHA-only equality."""
    need(path.is_file() and not path.is_symlink() and path.stat().st_size == expected['bytes'], 'raw extent: ' + str(path))
    h = hashlib.sha256()
    other = reference.open('rb') if reference is not None else None
    try:
        with path.open('rb') as f:
            for block in iter(lambda: f.read(4 * 1048576), b''):
                h.update(block)
                if other is not None:
                    need(block == other.read(len(block)), 'raw byte mismatch: ' + str(path))
        if other is not None:
            need(other.read(1) == b'', 'reference has trailing bytes')
    finally:
        if other is not None:
            other.close()
    need(h.hexdigest() == expected['sha256'], 'raw inventory SHA mismatch: ' + str(path))


def state_layout(folder, sample):
    need(folder.is_dir() and not folder.is_symlink(), 'regular state directory')
    manifest = read(folder / 'manifest.json')
    need(manifest['schema_version'] == 2 and manifest['complete'] is True and
         manifest['subject'] == 'prefill_indexer_register_only' and manifest['run'] == sample['run'] and
         manifest['scope'] == 'fresh_context_final_state_after_prefill_and_decode', 'state schema/scope')
    def unpack(name, code):
        raw = (folder / name).read_bytes()
        need(len(raw) % struct.calcsize(code) == 0, 'whole scalar array')
        return list(struct.unpack('<' + code * (len(raw) // struct.calcsize(code)), raw))
    values = unpack('context-scalars.u64le', 'Q'); need(len(values) == len(SCALARS), 'complete scalars')
    s = dict(zip(SCALARS, values)); lengths = unpack('anchor-lengths.u32le', 'I'); positions = unpack('snapshot-positions.u32le', 'I')
    need(s['max_seq_len'] == 1048576 and s['max_new_tokens'] == 256 and s['seed_u32'] == 42 and
         s['next_position'] <= s['max_seq_len'] and s['anchor_slots'] == len(lengths) <= 16 and
         all(n <= s['anchor_cap'] for n in lengths) and sum(n != 0 for n in lengths) == s['anchor_n'] and
         len(positions) == 88 and all(p <= s['next_position'] for p in positions[::2]), 'current final-state scalar/anchor/snapshot bounds')
    sizes = {'context-scalars.u64le': len(SCALARS) * 8, 'engram-tail.u32le': 12,
        'snapshot-positions.u32le': 352, 'prefix-tokens.i32le': s['pcache_len'] * 4,
        'anchor-lengths.u32le': s['anchor_slots'] * 4, 'tokens.i32le': 1024, 'output.bin': sample['output_bytes']}
    sizes.update({f'anchor-{i:02d}-tokens.i32le': n * 4 for i, n in enumerate(lengths)})
    sizes.update({f'layer-{i:02d}-window-active.f32': min(s['next_position'], 128) * 512 * 4 for i in range(40)})
    for owner in range(4):
        rows = s['next_position'] // (1 if owner == 3 else 2)
        for name, width in (('compressed', 1024), ('index', 256)):
            sizes[f'owner-{owner}-{name}-active.bf16'] = rows * width
        if owner < 3:
            for name in ('kv', 'score'):
                sizes[f'owner-{owner}-previous-{name}.f32'] = 2048
    for slot, pos in enumerate(positions[::2]):
        if pos:
            sizes.update({f'snapshot-{slot:02d}-layer-{layer:02d}-window-active.f32': min(pos, 128) * 2048 for layer in range(40)})
            for owner in range(3):
                for name in ('kv', 'score'):
                    sizes[f'snapshot-{slot:02d}-owner-{owner}-previous-{name}.f32'] = 2048
    declared = {d['path']: d['bytes'] for d in manifest['files']}
    need(len(declared) == len(manifest['files']) == manifest['file_count'] and declared == sizes and
         {p.name for p in folder.iterdir()} == set(sizes) | {'manifest.json'}, 'complete independently derived state extents')
    md = manifest['metadata']
    need(md['scalar_order'] == list(SCALARS) and md['byte_order'] == 'little' and md['fresh_context'] is True and
         all(md[k] == s[k] for k in ('ctx_pos', 'next_position', 'engram_position', 'anchor_slots', 'anchor_cap', 'anchor_n')) and
         md['rng_state'] == f"{s['rng_state']:016x}", 'raw scalar/manifest consistency')
    raw_tokens = (folder / 'tokens.i32le').read_bytes(); raw_output = (folder / 'output.bin').read_bytes()
    need(unpack('tokens.i32le', 'i') == sample['token_ids'] and fnv(raw_tokens) == sample['token_ids_fnv1a64'] and
         fnv(raw_output) == sample['output_bytes_fnv1a64'], 'actual raw tokens/output versus native callbacks')
    return md, sizes, s


def counters(sample, mode, workload):
    row = sample['gpu_indexer_register_only_provider']
    need(row['api_available'] is True and row['mode'] == mode and row['run'] == sample['run'] and
         row['observed_callbacks'] == 256 and row['adoption_valid'] is True and row['symbol_f32'] is None and
         row['symbol_bf16'] == ('kernel_myds_indexer_scores_register_only_bf16' if mode else None) and
         row['first_callback_scope'] == 'chunk_prefill_through_first_output_callback' and
         row['decode_scope'] == 'N_minus_1_output_intervals_no_candidate_encodings' and
         row['no_spec_source_predicate_and_telemetry_verified'] is True, 'actual provider identity and scope')
    for name in COUNTERS:
        total, first, decode = (row[p + name] for p in ('', 'through_first_callback_', 'decode_'))
        need(all(type(v) is int and v >= 0 for v in (total, first, decode)) and total == first + decode, 'actual counter split')
    expected = {'business96': 342, 'code38404': 417}[workload] if mode else 0
    need(row['encoded'] == row['through_first_callback_encoded'] == row['bf16_dispatches'] == expected and
         row['requests'] >= expected and all(row[p + k] == 0 for p in ('', 'through_first_callback_', 'decode_')
         for k in ('fallbacks', 'failures', 'f32_dispatches')) and row['decode_encoded'] == row['decode_bf16_dispatches'] == 0,
         'actual BF16 prefill adoption, no decode candidate/fallback/failure')
    telemetry = sample['decode_telemetry']
    for name, value in telemetry.items():
        expected = 256 if name in ('target_forward_calls', 'target_rows', 'output_tokens', 'cycle_count') else False if name == 'timing_enabled' else [] if name in ('cycle_ms', 'cycle_output_tokens') else None if name in ('draft_ms', 'verify_ms', 'rollback_ms', 'snapshot_ms', 'pld_ms') else 0
        need(value == expected, 'ordinary scalar telemetry: ' + name)
    return row


def review(root, stage):
    directory = root / (stage + '-freeze-v1'); protocol_path = directory / 'protocol.json'; p = read(protocol_path)
    prior = read(directory / 'review.json'); target = read(root / 'native-build-v2/target-build-manifest.json')
    need(p['stage'] == stage and p['request_contract']['output_tokens'] == 256 and
         p['request_contract']['total_native_generations'] == (12 if stage == 'Q' else 24), 'original Q12/P24')
    gate = p['performance_gate']
    need(gate['primary_min_saved_percent_strict'] == 2 and gate['max_abs_control_drift_percent_inclusive'] == 1 and
         gate['max_nonregression_percent_inclusive'] == Decimal('.5') and gate['formal_runs'] == [1, 2, 3] and
         gate['warmup_run'] == 0, 'unchanged prospectively frozen performance gates')
    need(prior['protocol_sha256'] == sha(protocol_path) and prior['binary_sha256'] == NATIVE and prior['reader_sha256'] == READER and
         prior['review_completed'] is True and prior['execution_complete'] is True and prior['numerical_gates_passed'] is True,
         'completed current-build independent reader, exact protocol and numerical gate')
    for item in p['artifacts'].values():
        check_descriptor(item)
    need(p['artifacts']['config']['sha256'] == CONFIG and p['artifacts']['reader']['sha256'] == READER and
         target['binary']['sha256'] == target['gold_binary']['sha256'] == NATIVE, 'frozen build/config/reader')
    source = read(p['artifacts']['private_build_manifest']['path'])
    for root_key, inventory_key in (('source_root', 'source_inventory'), ('baseline_root', 'baseline_inventory')):
        source_root = Path(source[root_key]); inv = source[inventory_key]
        actual = {str(f.relative_to(source_root)) for f in source_root.rglob('*') if f.is_file() and '__pycache__' not in f.parts}
        need(actual == set(inv), 'complete source/baseline inventory')
        for name, item in inv.items():
            path = contained(source_root, name)
            need(path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], 'source byte identity: ' + name)
    window_root = directory / (stage + '-window'); window = read(window_root / 'window.json')
    need(window['execution_complete'] is True and window['abort_reason'] is None and window['protocol_sha256'] == sha(protocol_path), 'complete fixed collection window')
    repeats = 1 if stage == 'Q' else 3
    names = ['G_lib', 'off', 'on'] if stage == 'Q' else ['A1', 'B', 'A2']
    need([v['name'] for v in p['providers']] == names, 'original arm order')
    report = dict(stage=stage, protocol=descriptor(protocol_path), reader_report=descriptor(directory / 'review.json'),
        source_files=len(source['source_inventory']), baseline_files=len(source['baseline_inventory']),
        full_raw_payloads_reread=True, all_payload_bytes_compared=True, workloads={}, raw_files_rehashed=0, raw_bytes_rehashed=0)
    for workload in p['workloads']:
        name = workload['name']; reference = None; reference_md = None; arms = {}; arm_reports = {}
        for provider in p['providers']:
            arm = provider['name']; folder = window_root / name / arm; inv_path = folder / 'raw-inventory.json'; inv = read(inv_path)
            plan = read(folder / 'execution-plan.json'); child = read(folder / 'window.json')
            check_descriptor(provider['binary']); check_descriptor(provider['library'])
            need(provider['binary']['sha256'] == NATIVE and provider['library']['sha256'] == (GOLD if arm == 'G_lib' else CANDIDATE), 'actual provider artifact')
            need(child['exit_code'] == 0 and child['abort_reason'] is None and child['raw_inventory_sha256'] == sha(inv_path) and
                 child['protocol_sha256'] == sha(protocol_path) and plan['binary_sha256'] == NATIVE and
                 plan['library_sha256'] == provider['library']['sha256'], 'child/plan/raw inventory binding')
            records = [loads(line) for line in (folder / 'stdout.jsonl').read_bytes().splitlines()]
            need([r['kind'] for r in records] == ['register_only_metadata', 'metadata', 'load'] + ['sample'] * (repeats + 1), 'complete stdout order')
            outer, metadata, loaded, *samples = records
            need(metadata['decode_basis'] == 'first_to_last_callback/(tokens-1)' and metadata['max_seq_len'] == 1048576 and
                 outer['runtime_mode'] == provider['mode'] and outer['effective_configuration_count'] == 68 and
                 outer['config_loaded_entries'] == 63 and loaded['vocab_size'] == 129280, 'actual mode/config/native metric basis')
            need(Path(loaded['actual_metallib_path']).resolve() == Path(provider['expected_loaded_library_path']).resolve(), 'actual loaded metallib')
            actual_files = {str(f.relative_to(folder)) for f in folder.rglob('*') if f.is_file()}
            need(actual_files == set(inv['files']) | {'raw-inventory.json', 'monitor.json', 'window.json', 'execution-plan.json'}, 'complete child file inventory')
            canonical = json.dumps(inv['files'], sort_keys=True, separators=(',', ':')).encode()
            need(hashlib.sha256(canonical).hexdigest() == inv['tree_sha256'] and len(inv['files']) == inv['file_count'], 'complete inventory digest/count')
            # Validate every retained file, including stdout and state manifests.
            payload_names = set(); expected_raw = {'stdout.jsonl', 'stderr.txt'}
            for sample in samples:
                run = sample['run']; need(type(run) is int and run == len(payload_names), 'original sample sequence')
                payload_names.add(run)
                need(sample['warmup'] is (run == 0) and sample['generation_ok'] is True and sample['sample_valid'] is True and
                     sample['fresh_context'] is True and sample['stop_reason'] == 'length' and sample['cached_tokens'] == 0 and
                     sample['prompt_tokens'] == workload['prompt_tokens'] and sample['output_tokens'] == 256 and sample['decode_intervals'] == 255,
                     'fresh complete original 256-token sample')
                counters(sample, provider['mode'], name)
                state = folder / 'state' / f'run-{run}'; md, sizes, scalars = state_layout(state, sample)
                logits = folder / 'logits' / f'run-{run}.f32'; raw = logits.read_bytes()
                need(len(raw) == 129280 * 4 and all(math.isfinite(v[0]) for v in struct.iter_unpack('<f', raw)) and
                     sample['final_logits']['finite'] is True and sample['final_logits']['elements'] == 129280 and
                     fnv(raw) == sample['final_logits']['fnv1a64_bytes'], 'all final logits finite/complete/native FNV')
                current = {n: state / n for n in sizes}; current['final-logits.f32'] = logits
                expected_raw.update(str(path.relative_to(folder)) for path in current.values())
                expected_raw.add(str((state / 'manifest.json').relative_to(folder)))
                if reference is None:
                    reference, reference_md = current, md
                need(set(current) == set(reference) and md == reference_md, 'all raw sample file sets and final metadata equal')
                for key, path in current.items():
                    relative = str(path.relative_to(folder)); verify_raw(path, inv['files'][relative], None if path == reference[key] else reference[key])
                    report['raw_files_rehashed'] += 1; report['raw_bytes_rehashed'] += path.stat().st_size
            need(set(inv['files']) == expected_raw, 'inventory contains exactly all samples and no unreviewed payloads')
            raw_names = {n for n in inv['files'] if n.startswith(('state/', 'logits/')) and not n.endswith('/manifest.json')}
            for relative in set(inv['files']) - raw_names:
                path = contained(folder, relative); verify_raw(path, inv['files'][relative]); report['raw_files_rehashed'] += 1; report['raw_bytes_rehashed'] += path.stat().st_size
            arms[arm] = samples
            arm_reports[arm] = dict(stdout=descriptor(folder / 'stdout.jsonl'), raw_inventory=descriptor(inv_path),
                actual_library=loaded['actual_metallib_path'], actual_counters=[s['gpu_indexer_register_only_provider'] for s in samples],
                raw_final_state=scalars, samples=len(samples))
        result = dict(prompt_tokens=workload['prompt_tokens'], arms=arm_reports, all_tokens_outputs_logits_final_state_bytes_equal=True)
        if stage == 'P':
            result['performance'] = performance(arms)
        report['workloads'][name] = result
    report['numerical_recheck_passed'] = True
    if stage == 'P':
        report['native_prefill_gate_passed'] = all(v['performance']['passed'] for v in report['workloads'].values())
        need(report['native_prefill_gate_passed'] is prior['native_prefill_gate_passed'], 'independent native gate agrees with frozen reader')
    else:
        need(prior['qualification_passed'] is True and prior['native_generation_count'] == 12, 'completed Q12 qualification')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', choices=('Q', 'P'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); need(not args.out.exists(), 'new independent result path required')
    result = dict(kind='independent_current_register_only_full_raw_review', script_sha256=sha(__file__),
                  started_epoch=time.time(), GPU_work_submitted=False, production_acceptance=False,
                  limitations=['No HTTP/deployed-executable bridge is inferred from native results.',
                    'State scope is final live state and published snapshots; intermediate logits, disk cache and draft state are excluded.',
                    'All state bytes and inventory SHA are independently checked; per-file state FNV declarations are checked by the frozen reader.'])
    code = 0
    try:
        result.update(review(args.root, args.stage), passed=True)
    except Exception as exc:
        result.update(passed=False, error=type(exc).__name__ + ': ' + str(exc)); code = 2
    result['finished_epoch'] = time.time()
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2, default=str); stream.write('\n')
    print(json.dumps({k: result.get(k) for k in ('passed', 'stage', 'numerical_recheck_passed', 'native_prefill_gate_passed', 'error', 'raw_files_rehashed', 'raw_bytes_rehashed')}))
    raise SystemExit(code)


if __name__ == '__main__':
    main()
