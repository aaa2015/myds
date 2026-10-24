#!/usr/bin/env python3
"""Explicit isolated Qa/KV component window; --prepare never stops production.

Exactly one of --prepare or --run is mandatory. Build the separate /tmp source
tree before invoking this runner; neither mode builds or touches canonical code.
--prepare performs CPU preflight and freezes identities only. --run additionally
stops the exact captured service PID, collects real attn_norm inputs, executes
host/GPU component gates, and restores the original service in finally.

The original argv, inherited MYDS bytes, config bytes, binary/library identities,
cwd, common log and nice are retained through the shared signed4 safety helpers.
The experiment environment comes from config plus inherited overrides; restored
production receives only its original inherited MYDS values and original argv.
Runner descriptors are redirected and SIGHUP ignored so SSH disconnects cannot
abandon restoration. No full-model test or production adoption is performed.
"""
import argparse
import fcntl
import hashlib
import json
import math
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


def protocol():
    return dict(schema_version=1, primary_mode='concurrent', supplemental_mode='serial',
                primary_control_kind='production', supplemental_control_kind='probe',
                control_symbol='kernel_mul_mv_q4_K_dense_f32_bf16',
                candidate_symbol='kernel_myds_qkv_dual_q4_k_bf16_nr2',
                required_layers=[0, 19, 39], required_positions=[0, 1, 7, 15],
                cases=12, providers=3, modes=['serial', 'concurrent'],
                max_absolute_control_drift_percent=2.0,
                minimum_conservative_time_saved_percent_strict=2.0,
                correctness='Full F32 Qa/KV bits, BF16 contract, finite, guards, writes',
                fullmodel_run=False, production_adoption=False)


def analyse_component(directory, frozen_protocol):
    """CPU-only audit of a complete harness output; raw files remain untouched."""
    directory = Path(directory)
    manifest = json.loads((directory / 'artifact-manifest.json').read_text())
    need(manifest.get('status') == 'complete', 'Harness manifest must finish complete')
    entries = manifest.get('files')
    need(isinstance(entries, list) and entries, 'Harness raw manifest is empty')
    observed = set()
    for entry in entries:
        relative = entry.get('path')
        need(isinstance(relative, str) and relative and relative not in observed,
             'Unique raw manifest paths required')
        observed.add(relative)
        path = (directory / relative).resolve(strict=True)
        need(directory.resolve() in path.parents and path.is_file(), 'Raw payload escapes component directory')
        need(path.stat().st_size == entry.get('bytes') and sha(path) == entry.get('sha256'),
             'Raw component payload bytes/SHA mismatch: ' + relative)
    rows = [json.loads(line) for line in (directory.parent / 'component.stdout').read_text().splitlines()
            if line.strip()]
    complete = [row for row in rows if row.get('kind') == 'complete']
    need(len(complete) == 1 and complete[0].get('full_f32_bit_exact') is True and
         complete[0].get('qa_kv_checked') is True and complete[0].get('cases') == 12 and
         set(complete[0].get('modes', [])) == {'serial', 'concurrent'},
         'Complete Qa/KV correctness evidence for both modes is required')
    parity = [row for row in rows if row.get('kind') == 'parity']
    wanted = {(mode, case, provider) for mode in ('serial', 'concurrent')
              for case in range(12) for provider in range(3)}
    need(len(parity) == len(wanted) and
         {(row.get('mode'), row.get('case'), row.get('provider')) for row in parity} == wanted,
         'All twelve cases, three providers, and both modes require parity evidence')
    for row in parity:
        need(row.get('full_f32_bit_exact') is True and row.get('finite_guards_written') is True
             and row.get('qa_mismatches') == row.get('kv_mismatches') == 0,
             'Recorded Qa/KV correctness failure')
    summaries = [row for row in rows if row.get('kind') == 'summary']
    groups = {(mode, kind, layer) for mode in ('serial', 'concurrent')
              for kind in ('production', 'probe') for layer in (0, 19, 39)}
    need(len(summaries) == len(groups) and
         {(row.get('mode'), row.get('control_kind'), row.get('layer')) for row in summaries} == groups,
         'Three layers and all mode/control summaries are required')
    for row in summaries:
        need(row.get('control_symbol') == frozen_protocol['control_symbol'] and
             row.get('candidate_symbol') == frozen_protocol['candidate_symbol'],
             'Unexpected native/candidate performance symbols')
        for key in ('a1_us', 'b_us', 'a2_us'):
            need(isinstance(row.get(key), (int, float)) and math.isfinite(row[key]) and row[key] > 0,
                 'Positive finite timing required: ' + key)
        a1, b, a2 = row['a1_us'], row['b_us'], row['a2_us']
        drift, saved = (a2 / a1 - 1) * 100, (1 - b / min(a1, a2)) * 100
        for key, computed in (('control_drift_percent', drift),
                              ('conservative_time_saved_percent', saved)):
            need(isinstance(row.get(key), (int, float)) and math.isfinite(row[key]) and
                 math.isclose(row[key], computed, abs_tol=1e-6, rel_tol=1e-6),
                 'Summary arithmetic mismatch: ' + key)
        row['performance_gate_pass'] = (
            abs(a2 - a1) <= a1 * frozen_protocol['max_absolute_control_drift_percent'] / 100 and
            b < min(a1, a2) * (1 - frozen_protocol['minimum_conservative_time_saved_percent_strict'] / 100))
    primary = [row for row in summaries if row['mode'] == 'concurrent' and
               row['control_kind'] == 'production']
    return dict(correctness_pass=True, performance_gate_pass=all(row['performance_gate_pass'] for row in primary),
                primary_performance=primary,
                supplemental_performance=[row for row in summaries if row not in primary],
                raw_payload_files_verified=len(entries), artifact_manifest_sha256=sha(directory / 'artifact-manifest.json'),
                component_stdout_sha256=sha(directory.parent / 'component.stdout'))


class QKVWindow(safety.Window):
    def capture_cases(self):
        request_path = self.a.out / 'request.json'
        write_json(request_path, dict(
            model='local-myds-deepseek-v4.1-flash-1017-q2-q4', stream=True,
            messages=[dict(role='user', content='请用 Python 写一个稳定排序函数，输入是整数列表，并解释重复元素如何保持顺序。')],
            max_tokens=32, temperature=0, top_p=1, thinking=False))
        prefix = self.a.out / 'tokenized'
        self.run_command('tokenize', [self.a.stage / 'bin/count_chat_tokens', self.a.model,
                                     request_path, prefix], env=self.environment, timeout=120)
        raw = prefix.with_suffix('.tokens.i32le.bin').read_bytes()
        need(len(raw) % 4 == 0 and len(raw) >= 64, 'Need at least 16 actual rendered prompt tokens')
        ids = struct.unpack('<' + 'i' * (len(raw) // 4), raw)
        need(all(value >= 0 for value in ids), 'Negative tokenizer token')
        token_path = self.a.out / 'trace-tokens.txt'
        token_path.write_text(' '.join(map(str, ids[:16])) + '\n')
        trace = self.a.out / 'trace'
        trace.mkdir()
        trace_env = dict(self.environment, MYDS_TRACE_PREFIX=str(trace / 'myds'),
                         MYDS_TRACE_LAYER='0,19,39', MYDS_TRACE_POS='0,1,7,15')
        self.run_command('trace', [self.a.stage / 'bin/trace_model', self.a.model,
                                  '--tokens-file', token_path], env=trace_env)
        cases = []
        for layer in self.frozen_protocol['required_layers']:
            for position in self.frozen_protocol['required_positions']:
                path = trace / f'myds_attn_norm-{layer}_pos{position}.bin'
                raw = path.read_bytes()
                need(len(raw) == 20480 and all(math.isfinite(v) for v in struct.unpack('<5120f', raw)),
                     'attn_norm must be exactly 5120 finite F32 values')
                cases.append(dict(layer=layer, position=position,
                                  input_f32=dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest())))
        capture_path = self.a.out / 'capture.json'
        write_json(capture_path, dict(
            schema_version=1, capture_kind='myds_decode_qkv',
            model_bytes=self.summary['model']['bytes'], model_header_bytes=1048576,
            model_header_sha256=self.summary['model_header_sha256'],
            source=dict(kind='trace_model_actual_model_teacher_forced',
                        diagnostic_only=True, performance_from_trace=False,
                        trace_binary_sha256=sha(self.a.stage / 'bin/trace_model'),
                        trace_metallib_sha256=sha(self.a.stage / 'bin/myds.metallib'),
                        request_sha256=sha(request_path), token_file_sha256=sha(token_path),
                        effective_environment_sha256=sha(self.a.out / 'effective-env.json'),
                        model_metadata=self.summary['model']), cases=cases))
        return capture_path

    def preflight(self):
        self.before, self.config_raw = safety.capture(self.a.port)
        self.allowed.add(self.before['pid'])
        need(os.getpriority(os.PRIO_PROCESS, os.getpid()) <= self.before['nice'],
             'Runner priority cannot guarantee unprivileged nice restoration')
        need(not any(flag in self.before['argv'] for flag in ('--pld', '--no-pld')),
             'CLI PLD overrides require an explicit environment audit')
        need(Path(safety.option(self.before['argv'], '-m')).resolve(strict=True) == self.a.model,
             'Requested model is not the original service artifact')
        need(str(self.a.stage) not in self.before['binary']['path'], 'Stage overlaps production')
        effective = self.before['effective_myds_env']
        need(effective.get('MYDS_METAL_CONCURRENT') == '1',
             'Primary concurrent-mode gate requires actual production concurrent configuration')
        need(effective.get('MYDS_QKV_MERGE', '0') == '0', 'Existing merged QKV provider changes native baseline')
        need(effective.get('MYDS_QKV_FUSE_BF16', '1') == '1' and
             effective.get('MYDS_F16_GEMV_BF16_FUSE', '1') == '1',
             'Production BF16 projection provider must be enabled')
        need(effective.get('MYDS_DENSE_Q4_NR0', '2') == '2' and
             effective.get('MYDS_DENSE_Q4_NSG', '4') == '4',
             'Production native Q4 provider must match NR2/NSG4 component geometry')
        for key, value in effective.items():
            if key.startswith('MYDS_QKV_DUAL'):
                need(value == '0', 'Production must retain its native QKV provider: ' + key)
        self.summary.update(model=metadata(self.a.model),
                            model_header_sha256=sha(self.a.model, 1048576),
                            component='dual_Qa_KV', runner_nice=os.getpriority(os.PRIO_PROCESS, os.getpid()),
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
        # Both modes only inspect an existing build; compilation belongs outside the window.
        binaries = [self.a.stage / 'bin' / name for name in
                    ('count_chat_tokens', 'trace_model', 'bench_qkv_dual', 'myds.metallib')]
        need(all(path.is_file() for path in binaries), 'Build all required stage binaries before this runner')
        inputs = [self.a.stage / 'Makefile', *binaries, self.a.out / 'protocol.json']
        for directory in ('src', 'include', 'metal', 'tools'):
            inputs.extend(path for path in (self.a.stage / directory).iterdir()
                          if path.is_file() and path.suffix in ('.c', '.h', '.m', '.metal', '.py'))
        self.input_hashes = {str(path): sha(path) for path in sorted(set(inputs))}
        need(str(Path(safety.__file__).resolve()) in self.input_hashes,
             'Safety helper must be imported from this frozen stage')
        need(str(Path(__file__).resolve()) in self.input_hashes,
             'Runner must execute from this frozen stage')
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
                self.summary.update(service_operations=False, gpu_work_submitted=False,
                                    tokenization_run=False, trace_run=False)
            else:
                need(self.a.run, 'Explicit --run is required before any service operation')
                self.thread = threading.Thread(target=self.monitor, daemon=True)
                self.thread.start()
                self.stop()
                capture_path = self.capture_cases()
                self.summary['capture_json_sha256'] = sha(capture_path)
                argv = [self.a.stage / 'bin/bench_qkv_dual', '--model', self.a.model,
                        '--capture-json', capture_path, '--probe-metallib', self.a.stage / 'bin/myds.metallib',
                        '--production-metallib', self.a.out / 'production-artifacts/myds.metallib']
                self.run_command('host-validation', [*argv, '--out-dir', self.a.out / 'host-validation',
                                                     '--host-only'], env=self.environment)
                try:
                    self.run_command('component', [*argv, '--out-dir', self.a.out / 'component',
                                                   '--samples', '9', '--reps', '8'], env=self.environment)
                    result = analyse_component(self.a.out / 'component', self.frozen_protocol)
                except BaseException:
                    self.summary.update(correctness_pass=False, component_gate_failed=True)
                    raise
                need(not self.fault.is_set(), 'Competition observed during window')
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
            reason = 'Component-only window; separate full-model gate required'
            if status == 'component_rejected_performance':
                reason = 'Primary production/concurrent component gate rejected candidate; no full-model run'
            elif self.summary.get('component_gate_failed'):
                reason = 'Component correctness/process gate failed; no full-model run'
            self.summary.update(status=status, fullmodel_not_run=True, fullmodel_not_run_reason=reason,
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
    mode.add_argument('--prepare', action='store_true', help='CPU preflight only; no build/stop/tokenize/trace/GPU')
    mode.add_argument('--run', action='store_true', help='Explicitly run the isolated component service window')
    return parser.parse_args(argv)


def main():
    args = arguments()
    need(sys.platform == 'darwin', 'This runner requires macOS KERN_PROCARGS2 and Metal')
    for key in ('stage', 'model', 'out'):
        value = getattr(args, key)
        need(value.is_absolute(), '--' + key + ' must be absolute')
        setattr(args, key, value.resolve())
    need(Path('/tmp').resolve() in args.stage.parents, 'Stage must be an independent source tree below /tmp')
    need((args.stage / 'Makefile').is_file() and args.model.is_file(), 'Invalid source stage/model')
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
    return QKVWindow(args).run()


if __name__ == '__main__':
    sys.exit(main())
