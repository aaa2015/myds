#!/usr/bin/env python3
"""Explicit real-input prefill multihead component window, never implicit stop.

Exactly one of --prepare/--run is required. Build a separate /tmp source tree
first. Prepare is CPU preflight only. Run captures layers 2/8/19, the second
2048-query chunk of a real 4096-token prefill, packs KV losslessly on the CPU,
replays the full working set, and restores production in finally.

Original args200 and consumed mask values are preserved. KV row dedup is storage
only: the harness reconstructs and SHA-checks all 2048*640*1024 original bytes.
Capture observer timing is diagnostic; only the standalone harness is timed.
All original expanded files remain until a separate archive/replay verifies them.
The script inherits capture/stop/restore helpers, not the signed4 execution path.
"""
import argparse
import array
import fcntl
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import shutil
import signal
import struct
import sys
import threading

sys.dont_write_bytecode = True
import run_iq2_signed4_window as safety

need, sha, metadata, write_json = safety.need, safety.sha, safety.metadata, safety.write_json
LAYERS, ROWS, KEYS, KV_ROW_BYTES = (2, 8, 19), 2048, 640, 1024
CONTROL = 'kernel_myds_flash_attn_vec_head_first_f16_dk512_dv512'
CANDIDATES = {2: 'kernel_myds_prefill_attention_head2_f16_dk512_dv512',
              4: 'kernel_myds_prefill_attention_head4_f16_dk512_dv512'}


def protocol():
    return dict(schema_version=1, scope='prefill_gathered_head_first_vec', mode='concurrent',
                layers=list(LAYERS), base_pos=0, c_start=2048, rows=ROWS, keys=KEYS,
                ratio=2, n_comp=2048, top_k=512, minimum_live_picks_per_query=512,
                teacher_forced_tokens=4096, context=327680, cases=3, providers=4,
                variants=[2, 4], control_symbol=CONTROL, candidate_symbols=CANDIDATES,
                primary_control_kind='production',
                supplemental_control_kind='probe', samples=9, reps=1,
                max_absolute_control_drift_percent=2.0,
                minimum_conservative_time_saved_percent_strict=2.0,
                correctness='Every F32 output bit, finite, guards, written; full 2048-query working set',
                kv_storage='Lossless exact 1024-byte row dictionary and u32 IDs; re-expand before GPU',
                mask_storage='Only consumed first640 halves per row; original args stride preserved',
                fullmodel_run=False, production_adoption=False)


def pack_kv(source, directory, rows=ROWS, keys=KEYS):
    """Lossless CPU storage transform with an independent reconstructed digest."""
    source, directory = Path(source), Path(directory)
    expected = rows * keys * KV_ROW_BYTES
    need(source.stat().st_size == expected, 'Expanded KV payload size mismatch')
    lookup, unique, ids = {}, [], array.array('I')
    need(ids.itemsize == 4, 'Expected uint32 row IDs')
    original = hashlib.sha256()
    with source.open('rb') as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
        for offset in range(0, expected, KV_ROW_BYTES):
            row = mapped[offset:offset + KV_ROW_BYTES]
            original.update(row)
            number = lookup.get(row)
            if number is None:
                number = len(unique)
                lookup[row] = number
                unique.append(row)
            ids.append(number)
    directory.mkdir(parents=True, exist_ok=False)
    dictionary, index = directory / 'kv-unique.f16', directory / 'kv-row-ids.u32le'
    with dictionary.open('xb') as stream:
        for row in unique:
            stream.write(row)
    reconstructed = hashlib.sha256()
    for number in ids:
        reconstructed.update(unique[number])
    need(reconstructed.digest() == original.digest(), 'KV row IDs do not reconstruct original bytes')
    if sys.byteorder != 'little':
        ids.byteswap()
    with index.open('xb') as stream:
        ids.tofile(stream)
    return dict(kv_unique_f16=dict(path=str(dictionary), bytes=dictionary.stat().st_size, sha256=sha(dictionary)),
                kv_row_ids_u32=dict(path=str(index), bytes=index.stat().st_size, sha256=sha(index)),
                kv_expanded_sha256=original.hexdigest(), expanded_bytes=expected,
                unique_rows=len(unique), lossless_reconstruction_verified=True)


def descriptor(path, expected=None):
    path = Path(path)
    size = path.stat().st_size
    need(expected is None or size == expected, 'Payload size mismatch: ' + str(path))
    return dict(path=str(path), bytes=size, sha256=sha(path))


def build_capture(raw_directory, destination, model_info, source_info):
    """Bind observer records, preserve args/mask, and pack three real full chunks."""
    raw_directory, destination = Path(raw_directory), Path(destination)
    records_path = raw_directory / 'capture-records.jsonl'
    records = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
    need(len(records) == 3 and {row.get('layer') for row in records} == set(LAYERS),
         'Observer must capture exactly layers 2,8,19 once')
    provider_path = raw_directory / 'capture-provider.json'
    provider = json.loads(provider_path.read_text())
    attempts, encoded = provider.get('head_first_attempts'), provider.get('head_first_encoded')
    need(type(attempts) is int and type(encoded) is int and attempts == encoded and encoded >= 3 and
         provider.get('gather_calls', 0) >= encoded and provider.get('ordinary_production_prefill') is True,
         'Capture must encode the actual ordinary head-first provider')
    for key in ('head_first_failures', 'head_first_cached_fallbacks', 'matrix_encoded', 'matrix_fallbacks'):
        need(provider.get(key) == 0, 'Actual capture provider fallback/failure: ' + key)
    destination.mkdir(parents=True, exist_ok=False)
    cases, raw_bindings = [], []
    for row in sorted(records, key=lambda item: item['layer']):
        layer = row['layer']
        for key, value in dict(base_pos=0, c_start=2048, rows=ROWS, keys=KEYS,
                               n_comp=2048, top_k=512, ratio=2).items():
            need(row.get(key) == value, 'Unexpected actual captured geometry: ' + key)
        need(row.get('gathered') is True and row.get('head_first') is True and
             row.get('matrix') is False and row.get('observer_commit_wait') is True and
             row.get('performance_from_capture') is False, 'Required actual provider/observer evidence')
        def path(kind):
            return raw_directory / f'layer-{layer}-{kind}.bin'
        args = path('args').read_bytes()
        need(len(args) == 200, 'Expected exact 200-byte attention args')
        need(struct.unpack_from('<ii', args, 0) == (ROWS, 64) and
             struct.unpack_from('<i', args, 40)[0] == KEYS and
             struct.unpack_from('<Q', args, 192)[0] == 655360,
             'Captured args are not the full gathered head-first geometry')
        stride = struct.unpack_from('<Q', args, 128)[0]
        need(stride == row.get('mask_source_stride') and stride >= 1280 and stride % 2 == 0,
             'Original mask stride must match observer record')
        mask = path('mask-f16').read_bytes()
        need(len(mask) == ROWS * KEYS * 2, 'Observer compact consumed-mask size mismatch')
        # Target chunk lies past the initial causal prefix. Check the real mask;
        # do not infer 512 active picks merely from args.ne11==640.
        live_picks, live_windows = [], []
        for index in range(ROWS):
            halves = struct.unpack_from('<640H', mask, index * 1280)
            need(all(value in (0, 0xfc00) for value in halves), 'Unexpected captured mask value')
            live_windows.append(sum(value == 0 for value in halves[:128]))
            live_picks.append(sum(value == 0 for value in halves[128:]))
        need(min(live_picks) == 512 and min(live_windows) == 128, 'Target full640-active chunk required')
        packed = pack_kv(path('kv-f16'), destination / f'layer-{layer}-kv', rows=ROWS, keys=KEYS)
        q = descriptor(path('q-f32'), ROWS * 131072)
        raw_row = dict(layer=layer, args=descriptor(path('args'), 200), q=q,
                       kv=descriptor(path('kv-f16'), ROWS * 655360),
                       mask=descriptor(path('mask-f16'), ROWS * 1280),
                       sinks=descriptor(path('sinks-f32'), 256))
        need(raw_row['kv']['sha256'] == packed['kv_expanded_sha256'],
             'Expanded raw KV SHA differs from reconstructed packed KV')
        raw_bindings.append(raw_row)
        cases.append(dict(layer=layer, base_pos=0, c_start=2048, c_len=ROWS, rows=ROWS, keys=KEYS,
                          ratio=2, n_comp=2048, top_k=512, args200=raw_row['args'], q_f32=q,
                          kv_unique_f16=packed['kv_unique_f16'], kv_row_ids_u32=packed['kv_row_ids_u32'],
                          kv_expanded_sha256=packed['kv_expanded_sha256'], mask_f16=raw_row['mask'],
                          sinks_f32=raw_row['sinks'], live_picks_min=min(live_picks),
                          live_windows_min=min(live_windows), unique_kv_rows=packed['unique_rows'],
                          original_mask_stride=stride))
    write_json(destination / 'original-inputs.json', dict(records=descriptor(records_path),
                                                        provider=descriptor(provider_path), actual_provider=provider,
                                                        payloads=raw_bindings,
                                                        expanded_inputs_retained=True, args_bytes_unchanged=True))
    capture = destination / 'capture.json'
    write_json(capture, dict(schema_version=1, capture_kind='myds_prefill_attention_multihead',
                            model_bytes=model_info['model']['bytes'], model_header_bytes=1048576,
                            model_header_sha256=model_info['model_header_sha256'],
                            source=dict(source_info, original_inputs_sha256=sha(destination / 'original-inputs.json'),
                                        model_metadata=model_info['model'], original_args_bytes_preserved=True,
                                        observer_commit_wait=True, performance_from_capture=False,
                                        kv_lossless_storage_only=True, expanded_working_set_required=True), cases=cases))
    return capture


def analyse_component(directory, frozen_protocol):
    """CPU-only interpretation, checking complete raw outputs before performance."""
    directory = Path(directory)
    manifest = json.loads((directory / 'artifact-manifest.json').read_text())
    need(manifest.get('status') == 'complete', 'Harness manifest must finish complete')
    entries = manifest.get('files')
    need(isinstance(entries, list) and entries, 'Harness raw manifest is empty')
    seen, verified_inodes = set(), {}
    for entry in entries:
        relative = entry.get('path')
        need(isinstance(relative, str) and relative and relative not in seen, 'Unique payload path required')
        seen.add(relative)
        path = (directory / relative).resolve(strict=True)
        need(directory.resolve() in path.parents and path.is_file(), 'Raw payload escapes output directory')
        st = path.stat()
        key = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
        digest = verified_inodes.setdefault(key, None)
        if digest is None:
            digest = verified_inodes[key] = sha(path)
        need(st.st_size == entry.get('bytes') and digest == entry.get('sha256'), 'Raw payload size/SHA mismatch')
        if entry.get('storage_alias_of'):
            alias = directory / entry['storage_alias_of']
            ast = alias.stat()
            need((ast.st_dev, ast.st_ino) == (st.st_dev, st.st_ino), 'Declared storage hardlink is not identical')
    stdout = directory.parent / 'component.stdout'
    rows = [json.loads(line) for line in stdout.read_text().splitlines() if line.strip()]
    complete = [row for row in rows if row.get('kind') == 'complete']
    need(len(complete) == 1 and complete[0].get('full_f32_bit_exact') is True and
         complete[0].get('attention_output_checked') is True and complete[0].get('cases') == 3 and
         complete[0].get('providers') == 4, 'Complete three-case/four-provider attention result required')
    parity = [row for row in rows if row.get('kind') == 'parity']
    need(len(parity) == 12 and {(row.get('case'), row.get('provider')) for row in parity} ==
         {(case, provider) for case in range(3) for provider in range(4)}, 'All attention provider/case parity required')
    for row in parity:
        need(row.get('mode') == 'concurrent' and row.get('rows') == ROWS and row.get('keys') == KEYS and
             row.get('output_bytes') == ROWS * 131072 and row.get('full_f32_bit_exact') is True and
             row.get('finite_guards_written') is True and row.get('mismatches') == 0,
             'Recorded attention output/shape correctness failure')
    summaries = [row for row in rows if row.get('kind') == 'summary']
    groups = {(case, kind, variant) for case in range(3) for kind in ('production', 'probe') for variant in (2, 4)}
    need(len(summaries) == len(groups) and
         {(row.get('case'), row.get('control_kind'), row.get('variant')) for row in summaries} == groups,
         'All attention case/control/variant timing summaries required')
    symbols = {}
    for row in summaries:
        need(row.get('mode') == 'concurrent' and row.get('control_symbol') == CONTROL,
             'Unexpected attention baseline symbol/mode')
        symbol = row.get('candidate_symbol')
        need(symbol == CANDIDATES[row['variant']], 'Unexpected multihead candidate symbol')
        need(symbols.setdefault(row['variant'], symbol) == symbol, 'Candidate symbol changed between samples')
        for key in ('a1_us', 'b_us', 'a2_us'):
            need(type(row.get(key)) in (int, float) and math.isfinite(row[key]) and row[key] > 0,
                 'Positive finite timing required: ' + key)
        a1, b, a2 = row['a1_us'], row['b_us'], row['a2_us']
        for key, expected in (('control_drift_percent', (a2 / a1 - 1) * 100),
                              ('conservative_time_saved_percent', (1 - b / min(a1, a2)) * 100)):
            need(type(row.get(key)) in (int, float) and math.isfinite(row[key]) and
                 math.isclose(row[key], expected, abs_tol=1e-6, rel_tol=1e-6), 'Timing arithmetic mismatch')
        row['performance_gate_pass'] = (
            abs(a2 - a1) <= a1 * frozen_protocol['max_absolute_control_drift_percent'] / 100 and
            b < min(a1, a2) * (1 - frozen_protocol['minimum_conservative_time_saved_percent_strict'] / 100))
    primary = [row for row in summaries if row['control_kind'] == 'production']
    winners = [variant for variant in (2, 4) if
               all(row['performance_gate_pass'] for row in primary if row['variant'] == variant)]
    return dict(correctness_pass=True, performance_gate_pass=bool(winners), winning_variants=winners,
                primary_performance=primary, supplemental_performance=[row for row in summaries if row not in primary],
                raw_logical_files_verified=len(entries), raw_unique_inodes_hashed=len(verified_inodes),
                artifact_manifest_sha256=sha(directory / 'artifact-manifest.json'), component_stdout_sha256=sha(stdout))


class AttentionWindow(safety.Window):
    def scan(self):
        # Keep the safety helper unchanged while recognizing this new native
        # capture tool even when it belongs to an unrelated process tree.
        rows = []
        for line in safety.command(['/bin/ps', '-axww', '-o', 'pid=,ppid=,comm=']).splitlines():
            parts = line.strip().split(None, 2)
            need(len(parts) == 3, 'Cannot parse process inventory')
            rows.append((int(parts[0]), int(parts[1]), parts[2]))
        allowed = {os.getpid(), *self.allowed}
        while True:
            more = {pid for pid, parent, _ in rows if parent in allowed}
            if more <= allowed:
                break
            allowed |= more
        names = {'myds', 'myglm', 'ds4', 'llama-server', 'llama-cli', 'llama-bench',
                 'ollama', 'mlx_lm', 'vllm', 'lms', 'trace_model', 'capture_attention_multihead'}
        relevant = []
        for pid, _, executable in rows:
            name = Path(executable).name.lower()
            if name in names or name.startswith(('bench_', 'bench-', 'moe_prefill_replay')):
                relevant.append(dict(pid=pid, executable=executable, permitted=pid in allowed))
        need(all(row['permitted'] for row in relevant), 'Competing model/benchmark/capture: ' + repr(relevant))
        return relevant

    def capture_cases(self):
        unit = ('请分析下面的 Python 稳定排序代码，说明重复元素如何保持先后次序，并检查边界情况。\n'
                'def stable_sort(values):\n    return sorted(enumerate(values), key=lambda item: (item[1], item[0]))\n'
                '测试输入包括空列表、单个元素、负数和重复整数。\n')
        request_path = self.a.out / 'request.json'
        write_json(request_path, dict(model='local-myds-deepseek-v4.1-flash-1017-q2-q4', stream=True,
                                     messages=[dict(role='user', content=unit * 192)],
                                     max_tokens=1, temperature=0, top_p=1, thinking=False))
        prefix = self.a.out / 'tokenized'
        self.run_command('tokenize', [self.a.stage / 'bin/count_chat_tokens', self.a.model,
                                     request_path, prefix], env=self.environment, timeout=120)
        raw = prefix.with_suffix('.tokens.i32le.bin').read_bytes()
        need(len(raw) % 4 == 0 and len(raw) >= 4096 * 4, 'Need 4096 actual rendered tokenizer IDs')
        ids = struct.unpack('<4096i', raw[:4096 * 4])
        need(all(value >= 0 for value in ids), 'Negative tokenizer ID')
        tokens = self.a.out / 'capture-tokens.txt'
        tokens.write_text(' '.join(map(str, ids)) + '\n')
        raw_directory = self.a.out / 'raw-capture'
        self.run_command('capture', [self.a.stage / 'bin/capture_attention_multihead', self.a.model,
                                     '--tokens-file', tokens, '--out-dir', raw_directory],
                         env=self.environment, timeout=1200)
        source = dict(kind='actual_4096token_prefill_second_full_chunk',
                      capture_binary_sha256=sha(self.a.stage / 'bin/capture_attention_multihead'),
                      capture_metallib_sha256=sha(self.a.stage / 'bin/myds.metallib'),
                      request_sha256=sha(request_path), token_file_sha256=sha(tokens),
                      rendered_tokens_available=len(raw) // 4, teacher_forced_tokens_used=4096,
                      effective_environment_sha256=sha(self.a.out / 'effective-env.json'))
        self.event('cpu_lossless_pack_begin', raw_directory=str(raw_directory))
        capture = build_capture(raw_directory, self.a.out / 'packed-capture', self.summary, source)
        self.event('cpu_lossless_pack_complete', capture_json_sha256=sha(capture))
        return capture

    def preflight(self):
        self.before, self.config_raw = safety.capture(self.a.port)
        self.allowed.add(self.before['pid'])
        need(os.getpriority(os.PRIO_PROCESS, os.getpid()) <= self.before['nice'], 'Cannot guarantee nice restoration')
        need(not any(flag in self.before['argv'] for flag in ('--pld', '--no-pld')), 'CLI PLD override requires audit')
        need(Path(safety.option(self.before['argv'], '-m')).resolve(strict=True) == self.a.model,
             'Requested model is not the original service model')
        need(str(self.a.stage) not in self.before['binary']['path'], 'Stage overlaps production')
        effective = self.before['effective_myds_env']
        expected = {'MYDS_METAL_CONCURRENT': '1', 'MYDS_PREFILL_CHUNK_SIZE': '2048',
                    'MYDS_ATTN_GATHER': '1', 'MYDS_ATTN_GATHER_MIN_KEYS': '0',
                    'MYDS_PREFILL_ATTN': 'vec', 'MYDS_ATTN_VEC_HEAD_FIRST': '1'}
        defaults = {'MYDS_ATTN_GATHER': '1', 'MYDS_ATTN_GATHER_MIN_KEYS': '0',
                    'MYDS_PREFILL_ATTN': 'vec', 'MYDS_ATTN_VEC_HEAD_FIRST': '1'}
        for key, value in expected.items():
            need(effective.get(key, defaults.get(key)) == value, 'Unexpected production prefill provider: ' + key)
        for key, value in effective.items():
            need(not key.startswith(('MYDS_TRACE_', 'MYDS_PROFILE_', 'MYDS_SKIP_')),
                 'Production diagnostic/path-skipping environment is outside capture scope')
            if 'MULTIHEAD' in key:
                need(value == '0', 'Production must retain its original attention provider: ' + key)
        self.summary.update(model=metadata(self.a.model), model_header_sha256=sha(self.a.model, 1048576),
                            component='prefill_attention_multihead',
                            runner_nice=os.getpriority(os.PRIO_PROCESS, os.getpid()),
                            explicit_mode='run' if self.a.run else 'prepare')
        write_json(self.a.out / 'before.json', self.before)
        (self.a.out / 'original-myds.conf').write_bytes(self.config_raw)
        write_json(self.a.out / 'effective-env.json', effective)
        self.frozen_protocol = protocol()
        write_json(self.a.out / 'protocol.json', self.frozen_protocol)
        self.environment = {key: value for key, value in os.environ.items() if not key.startswith('MYDS_')}
        self.environment.update(effective)
        self.environment.update(DEVELOPER_DIR='/Applications/Xcode.app/Contents/Developer',
                                PATH='/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin')
        self.event('preflight_processes', relevant=self.scan())
        binaries = [self.a.stage / 'bin' / name for name in
                    ('count_chat_tokens', 'capture_attention_multihead', 'bench_attention_multihead', 'myds.metallib')]
        need(all(path.is_file() for path in binaries), 'Build all capture/harness binaries outside this runner first')
        need(shutil.disk_usage(self.a.out).free >= 24 * 1024 ** 3, 'Need 24GiB free for full captured input/output evidence')
        inputs = [self.a.stage / 'Makefile', *binaries, self.a.out / 'protocol.json']
        for directory in ('src', 'include', 'metal', 'tools'):
            inputs.extend(path for path in (self.a.stage / directory).rglob('*')
                          if path.is_file() and path.suffix in ('.c', '.h', '.m', '.metal', '.py', '.inc'))
        for relative in ('tools/attention_multihead_capture.inc', 'tools/reference/trace_input.h'):
            need(self.a.stage / relative in inputs, 'Capture source dependency is not frozen: ' + relative)
        self.input_hashes = {str(path): sha(path) for path in sorted(set(inputs))}
        need(str(Path(safety.__file__).resolve()) in self.input_hashes and
             str(Path(__file__).resolve()) in self.input_hashes, 'Runner and safety helper must execute from frozen stage')
        write_json(self.a.out / 'stage-inputs.json', self.input_hashes)
        backup = self.a.out / 'production-artifacts'
        backup.mkdir()
        for key in ('binary', 'metallib'):
            destination = backup / Path(self.before[key]['path']).name
            shutil.copy2(self.before[key]['path'], destination)
            need(sha(destination) == self.before[key + '_sha256'], 'Production artifact copy hash mismatch')
        self.unchanged()

    def run(self):
        status, lock = 'failed', None
        def interrupted(number, frame):
            if self.recovering:
                self.event('deferred_signal_during_restoration', signal=number)
            else:
                raise KeyboardInterrupt('Window interrupted by signal ' + str(number))
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, interrupted)
        try:
            lock = open('/tmp/myds-codex-decode-window.lock', 'a+')
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.preflight()
            if self.a.prepare:
                status = 'prepared'
                self.summary.update(service_operations=False, gpu_work_submitted=False, tokenization_run=False,
                                    capture_run=False, lossless_pack_run=False)
            else:
                need(self.a.run, 'Explicit --run is required before service operations')
                self.thread = threading.Thread(target=self.monitor, daemon=True)
                self.thread.start()
                self.stop()
                capture = self.capture_cases()
                self.summary['capture_json_sha256'] = sha(capture)
                argv = [self.a.stage / 'bin/bench_attention_multihead', '--model', self.a.model,
                        '--capture-json', capture, '--probe-metallib', self.a.stage / 'bin/myds.metallib',
                        '--production-metallib', self.a.out / 'production-artifacts/myds.metallib']
                self.run_command('host-validation', [*argv, '--out-dir', self.a.out / 'host-validation',
                                                     '--host-only'], env=self.environment)
                try:
                    self.run_command('component', [*argv, '--out-dir', self.a.out / 'component',
                                                   '--samples', '9', '--reps', '1'], env=self.environment)
                    result = analyse_component(self.a.out / 'component', self.frozen_protocol)
                except BaseException:
                    self.summary.update(correctness_pass=False, component_gate_failed=True)
                    raise
                need(not self.fault.is_set(), 'Competition observed during component window')
                for path, digest in self.input_hashes.items():
                    need(sha(path) == digest, 'Frozen stage/protocol input changed: ' + path)
                self.summary.update(**result, component_gate_failed=False)
                status = 'component_passed' if result['performance_gate_pass'] else 'component_rejected_performance'
        except BaseException as exc:
            self.summary['error'] = repr(exc)
            self.event('window_failure', error=repr(exc))
        finally:
            self.recovering = True
            try:
                self.drain()
                if self.stopped:
                    self.summary['restoration'] = self.restore()
                else:
                    if self.before and 'model' in self.summary:
                        self.unchanged()
                    self.summary['restoration'] = dict(restored=True, service_not_stopped=True)
            except BaseException as exc:
                status = 'failed'
                self.summary['restoration'] = dict(restored=False, error=repr(exc))
                self.event('restoration_failure', error=repr(exc))
            self.done.set()
            if self.thread:
                self.thread.join(timeout=15)
            if self.fault.is_set() or (self.thread and self.thread.is_alive()):
                status = 'failed'
                self.summary['monitor_failed'] = True
            reason = 'Component-only full-working-set experiment; separate full-model gate required'
            if status == 'component_rejected_performance':
                reason = 'Neither variant passed every production/concurrent full-chunk performance gate; no full-model run'
            elif self.summary.get('component_gate_failed'):
                reason = 'Component correctness/process gate failed; no full-model run'
            self.summary.update(status=status, fullmodel_not_run=True, fullmodel_not_run_reason=reason,
                                expanded_capture_inputs_retained=True,
                                monitor_scope='Sampled model/native benchmark names and advisory lock; unknown renamed GPU programs may evade inspection')
            write_json(self.a.out / 'summary.json', self.summary)
            self.event('window_complete', status=status)
            if lock:
                lock.close()
        return 0 if status in ('prepared', 'component_passed', 'component_rejected_performance') else 1


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8187)
    parser.add_argument('--out', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true', help='CPU preflight only; no build/stop/tokenize/capture/GPU')
    mode.add_argument('--run', action='store_true', help='Explicitly run real prefill capture and full-chunk component tests')
    return parser.parse_args(argv)


def main():
    args = arguments()
    need(sys.platform == 'darwin', 'This runner requires macOS KERN_PROCARGS2 and Metal')
    for key in ('stage', 'model', 'out'):
        value = getattr(args, key)
        need(value.is_absolute(), '--' + key + ' must be absolute')
        setattr(args, key, value.resolve())
    need(Path('/tmp').resolve() in args.stage.parents, 'Stage must be independent below /tmp')
    need((args.stage / 'Makefile').is_file() and args.model.is_file(), 'Invalid stage/model')
    need(1 <= args.port <= 65535, 'Invalid service port')
    args.out.mkdir(parents=True, exist_ok=False)
    os.chmod(args.out, 0o700)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    try:
        os.setsid()
    except PermissionError:
        pass
    with (args.out / 'runner.log').open('ab', buffering=0) as log, open(os.devnull, 'rb') as null:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        os.dup2(null.fileno(), 0)
    return AttentionWindow(args).run()


if __name__ == '__main__':
    sys.exit(main())
