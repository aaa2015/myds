#!/usr/bin/env python3
"""Verify saved evidence or run the greedy full-model harness on an idle Mac.

This tool does not stop, deploy, or restart services. It clears inherited MYDS_
settings, captures stdout/stderr separately, and audits the completed samples.
The historical fleet window scripts remain unchanged in docs/benchmarks.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'docs/benchmarks/2026-10-05-optimization'
DEFAULT_ENV = {
    'MYDS_PREFIX_CACHE': '0', 'MYDS_KV_DSK': '0', 'MYDS_PHASE_PROFILE': '0',
    'MYDS_PREFILL_CHUNK_SIZE': '512', 'MYDS_MOE_CHUNKS': '9',
    'MYDS_SUFFIX_SHRINK_SHIFT': '512', 'MYDS_RESIDENCY_KEEP_ALIVE_S': '86400',
}
GPU_PROCESSES = {'myds', 'llama-server', 'bench_phase', 'bench_qkv',
                 'bench_attention_group', 'bench_attention_vec',
                 'bench_decode_iq2', 'moe_prefill_replay'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fnv_bytes(data):
    value = 14695981039346656037
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & 0xffffffffffffffff
    return f'{value:016x}'


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def verify(args):
    errors = []
    checked = 0
    for relative in ('2026-10-05-phase', '2026-10-05-optimization'):
        directory = ROOT / 'docs/benchmarks' / relative
        if relative.endswith('optimization'):
            entries = json.loads((directory / 'files.sha256.json').read_text())
        else:
            entries = json.loads((directory / 'manifest.json').read_text())['sha256']
        for name, expected in entries.items():
            path = directory / name
            digest = expected['sha256'] if isinstance(expected, dict) else expected
            if not path.is_file() or sha(path) != digest:
                errors.append(str(path))
            elif isinstance(expected, dict) and path.stat().st_size != expected['bytes']:
                errors.append(str(path) + ': size')
            checked += 1
    if args.check_source:
        manifest = json.loads((EVIDENCE / 'manifest.json').read_text())
        for name, digest in manifest['production_source_sha256'].items():
            path = ROOT / name
            if not path.is_file() or sha(path) != digest:
                errors.append('source: ' + name)
            checked += 1
    print(json.dumps({'verified': not errors, 'files_checked': checked, 'errors': errors}))
    return int(bool(errors))


def command_output(argv):
    result = subprocess.run(argv, text=True, capture_output=True, timeout=15)
    if result.returncode:
        raise RuntimeError(f'{argv[0]} failed: {result.stderr.strip()}')
    return result.stdout.strip()


def host_state():
    return {'platform': platform.platform(),
            'cpu': command_output(['sysctl', '-n', 'machdep.cpu.brand_string']),
            'memory_bytes': int(command_output(['sysctl', '-n', 'hw.memsize'])),
            'os': command_output(['sw_vers']),
            'swap': command_output(['sysctl', 'vm.swapusage']),
            'vm_stat': command_output(['vm_stat'])}


def check_idle(port):
    active = []
    for line in command_output(['ps', '-axo', 'pid=,comm=']).splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) == 2 and int(fields[0]) != os.getpid():
            if Path(fields[1]).name in GPU_PROCESSES:
                active.append(line.strip())
    if active:
        raise RuntimeError('Other inference/benchmark processes are running: ' + '; '.join(active))
    result = subprocess.run(['lsof', '-nP', '-t', '-a', f'-iTCP:{port}', '-sTCP:LISTEN'],
                            text=True, capture_output=True, timeout=15)
    if result.stdout.strip():
        raise RuntimeError(f'Port {port} has a listener; arrange an exclusive window before running')
    if result.returncode not in (0, 1) or result.stderr.strip():
        raise RuntimeError('Could not check the server listener: ' + result.stderr.strip())


def load_audit():
    path = EVIDENCE / 'protocol/summarize-final.py'
    spec = importlib.util.spec_from_file_location('myds_saved_phase_audit', path)
    module = importlib.util.module_from_spec(spec)
    # Evidence directories contain versioned raw bytes, not generated Python caches.
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def validate_run(directory, state, audit):
    group, samples = audit.validate_group(directory, 'run', {'environment': state['environment']})
    metadata = group['metadata']
    fixed = {'temperature': 0, 'top_p': 1, 'top_k': 0, 'min_p': 0,
             'repetition_penalty': 1, 'seed': 42, 'repeats': 3, 'warmup_runs': 1,
             'clock': 'CLOCK_MONOTONIC', 'prompt_renderer': 'parse_chat_prompt',
             'transport': 'direct_byte_callback', 'qos': 'user_interactive', 'qos_result': 0}
    for key, expected in fixed.items():
        audit.require(metadata.get(key) == expected, 'Fixed protocol differs: ' + key)
    argv = state['argv']
    audit.require(isinstance(argv, list) and len(argv) == 6, 'Invalid recorded harness argv')
    expected = {'model': argv[1], 'request_file': argv[2], 'repeats': int(argv[3]),
                'max_new_tokens': int(argv[4]), 'max_seq_len': int(argv[5]),
                'request_bytes': state['request_bytes'], 'request_fnv1a64': state['request_fnv1a64']}
    for key, value in expected.items():
        audit.require(metadata.get(key) == value, 'Metrics/input binding differs: ' + key)
    audit.require(state['model']['path'] == metadata['model'], 'Recorded model identity/path differs')
    audit.require(metadata['max_new_tokens'] in (32, 64) and 0 < metadata['max_seq_len'] <= 0xffffffff,
                  'Recorded output/context range differs')
    return group, samples


def compare(args):
    audit = load_audit()
    groups, raw, states = [], [], []
    for directory in (args.base.resolve(), args.candidate.resolve()):
        state = json.loads((directory / 'run-state.json').read_text())
        audit.require(state.get('status') == 'validated' and state.get('returncode') == 0,
                      f'Incomplete/failed run: {directory}')
        group, samples = validate_run(directory, state, audit)
        groups.append(group)
        raw.append(samples)
        states.append(state)
    base, candidate = groups
    for key in audit.MATCH_META:
        audit.require(base['metadata'][key] == candidate['metadata'][key], 'Input/config differs: ' + key)
    for key in ('request_sha256', 'model'):
        audit.require(states[0][key] == states[1][key], 'File identity differs: ' + key)
    for key in ('cpu', 'memory_bytes', 'os'):
        audit.require(states[0]['host_before'][key] == states[1]['host_before'][key], 'Host differs: ' + key)
    for before, after in zip(*raw):
        for key in ('token_ids', 'output_bytes_fnv1a64', 'output_bytes', 'prompt_tokens',
                    'prefill_tokens', 'output_tokens', 'decode_intervals', 'stop_reason'):
            audit.require(before[key] == after[key], f'Output/count differs in run {before["run"]}: {key}')
    reductions = {}
    for phase in ('prefill', 'decode'):
        before = base['stats'][phase + '_ms_per_token']
        after = candidate['stats'][phase + '_ms_per_token']
        reductions[phase] = {'base': before, 'candidate': after,
                             'time_saved_percent': 100 * (1 - after['median'] / before['median'])}
    summary = {'audit_pass': True, 'all_warm_and_formal_outputs_match': True,
               'scope': 'Separate processes, one warmup plus three formal samples; direct callback timing',
               'base': base, 'candidate': candidate, 'comparison': reductions,
               'builds': [{'binary_sha256': s['binary_sha256'], 'metallib_sha256': s['metallib_sha256'],
                           'environment': s['environment']} for s in states]}
    output = args.out.resolve()
    if output.exists():
        raise ValueError('Comparison output already exists: ' + str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, summary)
    print(json.dumps({'audit_pass': True, 'output': str(output), 'comparison': reductions}))
    return 0


def stop_child(child):
    if child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=5)


def run(args):
    if platform.system() != 'Darwin':
        raise ValueError('The model runner requires macOS/Metal; verify works without a GPU')
    binary, model, request = (p.expanduser().resolve() for p in (args.binary, args.model, args.request))
    for path in (binary, model, request, binary.parent / 'myds.metallib'):
        if not path.is_file():
            raise ValueError('Missing input: ' + str(path))
    if not os.access(binary, os.X_OK):
        raise ValueError('Harness is not executable: ' + str(binary))
    explicit_env = dict(DEFAULT_ENV)
    for setting in args.env:
        name, separator, value = setting.partition('=')
        if not separator or not name.startswith('MYDS_'):
            raise ValueError('--env requires MYDS_NAME=value')
        explicit_env[name] = value
    for name in ('MYDS_PREFIX_CACHE', 'MYDS_KV_DSK', 'MYDS_PHASE_PROFILE'):
        if explicit_env[name] != '0':
            raise ValueError(name + '=0 is required for this full-prefill protocol')
    forbidden = ('MYDS_PROFILE_', 'MYDS_SKIP_', 'MYDS_TRACE_', 'MYDS_BATCH_TRACE_')
    for name in explicit_env:
        if name.startswith(forbidden) or name in ('MYDS_MOE_STEP_PROF', 'MYDS_PHASE_PROFILE_FILE',
                                                 'MYDS_DISABLE_V41_PREFILL'):
            raise ValueError('Profiling/path-skipping setting is incompatible with this run: ' + name)
    check_idle(args.server_port)
    output = args.out.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    environment = {k: v for k, v in os.environ.items() if not k.startswith('MYDS_')}
    environment.update(explicit_env)
    argv = [str(binary), str(model), str(request), '3', str(args.max_new_tokens), str(args.context)]
    request_data = request.read_bytes()
    state = {'argv': argv, 'cwd': str(binary.parent.parent), 'environment': explicit_env,
             'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
             'binary_sha256': sha(binary), 'metallib_sha256': sha(binary.parent / 'myds.metallib'),
             'request_sha256': hashlib.sha256(request_data).hexdigest(),
             'request_bytes': len(request_data), 'request_fnv1a64': fnv_bytes(request_data),
             'model': {'path': str(model),
             'bytes': model.stat().st_size, 'mtime_ns': model.stat().st_mtime_ns,
             'identity_note': 'File metadata only; the complete GGUF was not SHA-hashed'},
             'host_before': host_state(), 'status': 'started'}
    write_json(output / 'run-state.json', state)
    child = None
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    def interrupted(signum, frame):
        raise RuntimeError('Measurement interrupted by signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    try:
        with (output / 'run-metrics.jsonl').open('w') as stdout, (output / 'run.log').open('w') as stderr:
            child = subprocess.Popen(argv, cwd=binary.parent.parent, env=environment,
                                     stdout=stdout, stderr=stderr, start_new_session=True)
            result = child.wait(timeout=args.timeout)
        state['returncode'] = result
        if result:
            raise RuntimeError(f'Harness exited {result}; inspect {output / "run.log"}')
        group, _ = validate_run(output, state, load_audit())
        write_json(output / 'run-summary.json', group)
        state['status'] = 'validated'
    except BaseException as exc:
        state.update(status='failed', error=str(exc))
        raise
    finally:
        try:
            if child:
                try:
                    stop_child(child)
                except (OSError, subprocess.SubprocessError) as exc:
                    state.update(status='failed', cleanup_error=str(exc))
            try:
                state['host_after'] = host_state()
            except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
                state.update(status='failed', host_after_error=str(exc))
            write_json(output / 'run-state.json', state)
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)
    if state['status'] != 'validated':
        raise RuntimeError('Final state capture/cleanup failed; inspect ' + str(output / 'run-state.json'))
    print(json.dumps({'audit_pass': True, 'output': str(output),
                      'prefill_ms_per_token': group['stats']['prefill_ms_per_token'],
                      'decode_ms_per_token': group['stats']['decode_ms_per_token']}))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    verification = commands.add_parser('verify', help='Check committed evidence SHA manifests; no GPU/model access')
    verification.add_argument('--check-source', action='store_true', help='Also check the packaged source snapshot')
    comparison = commands.add_parser('compare', help='Audit raw base/candidate samples and compare medians; no GPU')
    comparison.add_argument('--base', type=Path, required=True)
    comparison.add_argument('--candidate', type=Path, required=True)
    comparison.add_argument('--out', type=Path, required=True)
    measurement = commands.add_parser('run', help='Run one model configuration; requires an already idle machine')
    measurement.add_argument('--binary', type=Path, default=ROOT / 'bin/bench_phase')
    measurement.add_argument('--model', type=Path, required=True)
    measurement.add_argument('--request', type=Path, required=True)
    measurement.add_argument('--out', type=Path, required=True, help='New output directory; existing directories are refused')
    measurement.add_argument('--env', action='append', default=[], help='Explicit MYDS_NAME=value, repeatable')
    measurement.add_argument('--max-new-tokens', type=int, choices=(32, 64), default=64)
    measurement.add_argument('--context', type=int, default=262144)
    measurement.add_argument('--timeout', type=float, default=900)
    measurement.add_argument('--server-port', type=int, default=8181)
    args = parser.parse_args()
    try:
        if args.command == 'run' and (not 0 < args.context <= 0xffffffff or
                                     not math.isfinite(args.timeout) or args.timeout <= 0 or
                                     not 1 <= args.server_port <= 65535):
            raise ValueError('context must be 1..UINT32_MAX, timeout finite/positive, server-port 1..65535')
        if args.command == 'verify':
            return verify(args)
        if args.command == 'compare':
            return compare(args)
        return run(args)
    except (ValueError, RuntimeError, OSError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
