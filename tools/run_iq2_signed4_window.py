#!/usr/bin/env python3
"""Build an isolated tree, capture real IQ2 inputs, and restore the original service.

Run on the Mac hosting the service. --stage must be a separate source tree below
/tmp, with Makefile/src/metal/tools. --prepare builds and performs read-only
preflight only. A normal run stops exactly the captured port PID, runs trace and
component tests, then restores its original argv/cwd/log/MYDS bytes in finally.
No canonical source, production artifact or configuration is modified.

The runner ignores SIGHUP, redirects its own descriptors into the new output
directory, and uses independent process groups for workers and the restored
service. It continues restoring if the SSH connection disappears. Example:
  nohup python3 tools/run_iq2_signed4_window.py --stage /tmp/signed4/source \
    --model /absolute/model.gguf --port 8187 --out /tmp/signed4/window >launch.log 2>&1 &

This is a component experiment. Successful repeated fixtures do not establish
cold DRAM performance, a full-model speedup, or production adoption.
"""
import argparse
import ctypes
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import struct
import subprocess
import sys
import threading
import time
import urllib.request


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path, limit=None):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        remaining = limit
        while remaining is None or remaining:
            block = stream.read(1024 * 1024 if remaining is None else min(1024 * 1024, remaining))
            if not block:
                break
            h.update(block)
            if remaining is not None:
                remaining -= len(block)
        need(remaining in (None, 0), 'Short file while hashing: ' + str(path))
    return h.hexdigest()


def metadata(path):
    path = Path(path).resolve(strict=True)
    st = path.stat()
    return dict(path=str(path), bytes=st.st_size, mtime_ns=st.st_mtime_ns,
                inode=st.st_ino, device=st.st_dev)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.new')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def command(argv, timeout=10):
    return subprocess.check_output(argv, text=True, timeout=timeout).strip()


def listener(port, required=True):
    result = subprocess.run(['/usr/sbin/lsof', '-nP', '-t', '-a', f'-iTCP:{port}',
                             '-sTCP:LISTEN'], capture_output=True, text=True, timeout=10)
    need(result.returncode in (0, 1) and not result.stderr.strip(), 'Listener inspection failed')
    pids = set(result.stdout.split())
    if not pids and not required:
        return None
    need(len(pids) == 1, f'Exactly one listener required on {port}')
    return int(pids.pop())


def health(port):
    with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=3) as response:
        value = json.loads(response.read())
    need(value.get('status') == 'ok', 'Service health is not ok')
    return value


def kernel_arguments(pid):
    """Return complete argv bytes and only MYDS environment bytes.

    Schema deliberately matches tools/run_decode_probe_window.py: both the
    MYDS key and its value are hexadecimal in myds_env_raw_hex. The older
    replay/process_identity.py instead uses a decoded key; do not mix them.
    KERN_PROCARGS2 contains inherited env, not later --config setenv updates.
    """
    lib = ctypes.CDLL('/usr/lib/libSystem.B.dylib', use_errno=True)
    lib.sysctl.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_uint, ctypes.c_void_p,
                          ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p, ctypes.c_size_t]
    lib.sysctl.restype = ctypes.c_int
    mib = (ctypes.c_int * 3)(1, 49, pid)
    size = ctypes.c_size_t()
    if lib.sysctl(mib, 3, None, ctypes.byref(size), None, 0):
        raise OSError(ctypes.get_errno(), 'KERN_PROCARGS2 size')
    buf = ctypes.create_string_buffer(size.value)
    if lib.sysctl(mib, 3, buf, ctypes.byref(size), None, 0):
        raise OSError(ctypes.get_errno(), 'KERN_PROCARGS2 read')
    data = buf.raw[:size.value]
    argc = struct.unpack_from('i', data)[0]
    need(0 < argc < 4096, 'Invalid KERN argc')
    cursor = data.index(b'\0', 4) + 1
    while cursor < len(data) and data[cursor] == 0:
        cursor += 1
    argv = []
    for _ in range(argc):
        end = data.index(b'\0', cursor)
        argv.append(data[cursor:end])
        cursor = end + 1
    env, raw_env = {}, {}
    for raw in data[cursor:].split(b'\0'):
        if raw.startswith(b'MYDS_') and b'=' in raw:
            key, value = raw.split(b'=', 1)
            name = os.fsdecode(key)
            need(name not in env, 'Duplicate inherited MYDS key')
            env[name], raw_env[key.hex()] = os.fsdecode(value), value.hex()
    return dict(argv=[os.fsdecode(value) for value in argv],
                argv_raw_hex=[value.hex() for value in argv],
                myds_env=env, myds_env_raw_hex=raw_env)


def config_values(raw):
    """Mirror myds_configfile_load's plain KEY=VALUE/comment rules; no shell eval."""
    values = {}
    for line in raw.decode('utf-8').splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        need(len(line.encode()) < 4095 and '=' in line, 'Unsupported config line')
        key, value = line.split('=', 1)
        key, value = key.strip(' \t'), value.lstrip(' \t')
        match = re.search(r'(?<=[ \t])#', value)
        if match:
            value = value[:match.start()]
        value = value.rstrip(' \t\r\n')
        need(key.startswith('MYDS_') and '\0' not in key + value, 'Config accepts MYDS keys only')
        # The C loader without override keeps the first occurrence.
        values.setdefault(key, value)
    return values


def option(argv, flag):
    need(argv.count(flag) == 1 and argv.index(flag) + 1 < len(argv), 'Missing/duplicate option ' + flag)
    return argv[argv.index(flag) + 1]


def capture(port):
    pid = listener(port)
    value = kernel_arguments(pid)
    names = re.findall(r'^n(.+)$', command(['/usr/sbin/lsof', '-a', '-p', str(pid),
                                           '-d', 'cwd,1,2', '-Fn']), re.M)
    need(len(names) == 3 and names[1] == names[2], 'Cannot bind common stdout/stderr log')
    binary = Path(value['argv'][0]).resolve(strict=True)
    library = binary.parent / 'myds.metallib'
    # myds_metal_init("metal") searches cwd-relative libraries before execdir.
    # An earlier candidate would invalidate the assumed production golden.
    library_precedence = []
    for relative in ('metal/myds.metallib', 'bin/myds.metallib', 'build/myds.metallib'):
        candidate = Path(names[0]) / relative
        exists = candidate.exists()
        library_precedence.append(dict(path=str(candidate), exists=exists))
        need(not exists or candidate.resolve() == library.resolve(),
             'An earlier cwd-relative metallib could shadow the captured production library')
    config = Path(option(value['argv'], '--config')).resolve(strict=True)
    config_raw = config.read_bytes()
    texts = command(['/usr/sbin/lsof', '-a', '-p', str(pid), '-d', 'txt', '-Fni'])
    records, inode = [], None
    for line in texts.splitlines():
        if line.startswith('i'):
            inode = int(line[1:])
        elif line.startswith('n'):
            records.append((line[1:], inode))
    need((str(binary), binary.stat().st_ino) in records,
         'Live mapped binary inode differs from current production file')
    effective = config_values(config_raw)
    effective.update(value['myds_env'])
    value.update(pid=pid, port=port, cwd=names[0], log=names[1],
                 nice=os.getpriority(os.PRIO_PROCESS, pid),
                 binary=metadata(binary), binary_sha256=sha(binary),
                 metallib=metadata(library), metallib_sha256=sha(library),
                 config=metadata(config), config_sha256=hashlib.sha256(config_raw).hexdigest(),
                 library_precedence=library_precedence,
                 effective_myds_env=effective, health=health(port))
    need(listener(port) == pid, 'Production PID changed during snapshot')
    return value, config_raw


def gguf_clamp(path):
    """Read the scalar DeepSeek metadata without mapping tensor payloads."""
    with Path(path).open('rb') as stream:
        size = os.fstat(stream.fileno()).st_size

        def read(n):
            raw = stream.read(n)
            need(len(raw) == n, 'Truncated GGUF metadata')
            return raw

        def integer(fmt):
            return struct.unpack('<' + fmt, read(struct.calcsize('<' + fmt)))[0]

        def skip(n):
            need(0 <= n <= size - stream.tell(), 'GGUF metadata extends beyond file')
            stream.seek(n, 1)

        def string(want=False):
            n = integer('Q')
            need(n <= size - stream.tell(), 'Invalid GGUF string length')
            if want:
                need(n <= 4096, 'Unbounded GGUF metadata key')
                return read(n).decode('utf-8')
            skip(n)

        formats = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i',
                   6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}

        def value(kind, wanted=False):
            if kind in formats:
                result = integer(formats[kind])
                return result if wanted else None
            if kind == 8:
                string()
                return None
            need(kind == 9 and not wanted, 'Unsupported target metadata type')
            element, count = integer('I'), integer('Q')
            need(count <= 10000000, 'Unbounded GGUF metadata array')
            if element in formats:
                skip(count * struct.calcsize('<' + formats[element]))
            else:
                need(element == 8, 'Unsupported GGUF metadata array type')
                for _ in range(count):
                    string()

        need(read(4) == b'GGUF' and integer('I') in (2, 3), 'Expected GGUF v2/v3')
        integer('Q')  # tensor count
        count = integer('Q')
        need(count < 100000, 'Unbounded GGUF key count')
        for _ in range(count):
            key, kind = string(True), integer('I')
            if key == 'deepseek41.swiglu_limit':
                need(kind == 6, 'Expected F32 deepseek41.swiglu_limit')
                clamp = value(kind, True)
                need(math.isfinite(clamp) and clamp == 10.0, 'Model clamp must be verified as 10.0')
                return dict(key=key, type='GGUF_TYPE_FLOAT32', value=clamp)
            value(kind)
    raise RuntimeError('Required deepseek41.swiglu_limit metadata is absent')


def analyse_component_stdout(path, protocol):
    """CPU-only interpretation; never changes original harness results."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    complete = [row for row in rows if row.get('kind') == 'complete']
    need(len(complete) == 1 and complete[0].get('full_f32_bit_exact') is True and
         complete[0].get('gate_up_mid_checked') is True and complete[0].get('cases') == 12,
         'A complete 12-case full-F32 correctness result is required')
    parity = [row for row in rows if row.get('kind') == 'parity']
    need(len(parity) == 60 and {(row.get('case'), row.get('provider')) for row in parity} ==
         {(case, provider) for case in range(12) for provider in range(5)},
         'All 12 cases and five explicit providers require parity evidence')
    for row in parity:
        need(row.get('full_f32_bit_exact') is True and row.get('finite_guards_written') is True
             and row.get('mid_mismatches') == 0, 'Recorded component correctness failure')
        if row['provider'] >= 3:
            need(row.get('gate_mismatches') == row.get('up_mismatches') == 0,
                 'Recorded gate/up bit mismatch')
    primary = [row for row in rows if row.get('kind') == 'summary' and
               row.get('control_symbol') == protocol['primary_control']]
    need(len(primary) == 3 and {row['layer'] for row in primary} == set(protocol['required_layers']),
         'Exactly three primary native-control layer summaries required')
    for row in primary:
        for key in ('a1_us', 'b_us', 'a2_us'):
            need(isinstance(row.get(key), (int, float)) and math.isfinite(row[key]) and row[key] > 0,
                 'Positive finite timing required: ' + key)
        drift = (row['a2_us'] / row['a1_us'] - 1) * 100
        saved = (1 - row['b_us'] / min(row['a1_us'], row['a2_us'])) * 100
        for key, computed in (('control_drift_percent', drift),
                              ('conservative_time_saved_percent', saved)):
            need(isinstance(row.get(key), (int, float)) and math.isfinite(row[key]) and
                 math.isclose(row[key], computed, abs_tol=1e-6, rel_tol=1e-6),
                 'Summary timing arithmetic mismatch: ' + key)
        row['performance_gate_pass'] = (
            abs(row['a2_us'] - row['a1_us']) <=
            row['a1_us'] * protocol['max_absolute_control_drift_percent'] / 100 and
            row['b_us'] < min(row['a1_us'], row['a2_us']) *
            (1 - protocol['minimum_conservative_time_saved_percent_strict'] / 100))
    return dict(correctness_pass=True,
                performance_gate_pass=all(row['performance_gate_pass'] for row in primary),
                primary_performance=primary,
                shape64_performance=[row for row in rows if row.get('kind') == 'summary' and
                                     row.get('control_symbol') != protocol['primary_control']])


class Window:
    def __init__(self, args):
        self.a, self.child, self.before = args, None, None
        self.stopped, self.recovering = False, False
        self.done, self.fault = threading.Event(), threading.Event()
        self.guard, self.allowed = threading.RLock(), set()
        self.steps, self.thread = [], None
        self.summary = dict(schema_version=1, status='failed', fullmodel_run=False,
                            production_adoption=False, cold_dram_proven=False)

    def event(self, kind, **data):
        with self.guard:
            with (self.a.out / 'events.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(kind=kind, monotonic=time.monotonic(), **data),
                                        sort_keys=True, ensure_ascii=True, allow_nan=False) + '\n')

    def scan(self):
        rows = []
        for line in command(['/bin/ps', '-axww', '-o', 'pid=,ppid=,comm=']).splitlines():
            parts = line.strip().split(None, 2)
            need(len(parts) == 3, 'Cannot parse process inventory')
            rows.append((int(parts[0]), int(parts[1]), parts[2]))
        allowed = {os.getpid(), *self.allowed}
        while True:
            more = {pid for pid, parent, _ in rows if parent in allowed}
            if more <= allowed:
                break
            allowed |= more
        relevant = []
        models = {'myds', 'myglm', 'ds4', 'llama-server', 'llama-cli', 'llama-bench',
                  'ollama', 'mlx_lm', 'vllm', 'lms', 'trace_model'}
        for pid, _, exe in rows:
            name = Path(exe).name.lower()
            if name in models or name.startswith(('bench_', 'bench-', 'moe_prefill_replay')):
                relevant.append(dict(pid=pid, executable=exe, permitted=pid in allowed))
        need(all(row['permitted'] for row in relevant), 'Competing model/benchmark: ' + repr(relevant))
        return relevant

    def monitor(self):
        while not self.done.wait(0.5):
            try:
                with self.guard:
                    rows = self.scan()
                self.event('monitor', relevant=rows)
            except BaseException as exc:
                self.event('monitor_failure', error=repr(exc))
                self.fault.set()
                return

    def run_command(self, name, argv, env=None, timeout=1200):
        self.event('step_begin', name=name, argv=[str(v) for v in argv])
        started, rc = time.monotonic(), None
        try:
            with (self.a.out / (name + '.stdout')).open('wb') as stdout, \
                 (self.a.out / (name + '.stderr')).open('wb') as stderr, self.guard:
                self.child = subprocess.Popen([str(v) for v in argv], cwd=self.a.stage,
                                              env=env, stdout=stdout, stderr=stderr,
                                              stdin=subprocess.DEVNULL, start_new_session=True)
                self.allowed.add(self.child.pid)
            while self.child.poll() is None:
                need(not self.fault.is_set(), 'Competition/monitor failure invalidates window')
                need(time.monotonic() - started < timeout, name + ' timed out')
                time.sleep(0.2)
            rc = self.child.returncode
            need(rc == 0, f'{name} failed with return code {rc}')
        finally:
            self.drain()
            self.steps.append(dict(name=name, argv=[str(v) for v in argv], returncode=rc,
                                   seconds=time.monotonic() - started))
            write_json(self.a.out / 'steps.json', self.steps)

    def drain(self):
        if self.child is None:
            return
        proc = self.child
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)
            except ProcessLookupError:
                pass
        self.child = None

    def unchanged(self):
        for key in ('binary', 'metallib', 'config'):
            row = self.before[key]
            need(metadata(row['path']) == row, 'Production artifact metadata changed: ' + key)
            need(sha(row['path']) == self.before[key + '_sha256'],
                 'Production artifact bytes changed: ' + key)
        need(metadata(self.a.model) == self.summary['model'], 'Model artifact metadata changed')
        need(sha(self.a.model, 1048576) == self.summary['model_header_sha256'], 'Model header changed')

    def stop(self):
        self.unchanged()
        current, raw = capture(self.a.port)
        need(current == self.before and raw == self.config_raw, 'Production changed before stop')
        connections = subprocess.run(['/usr/sbin/lsof', '-nP', '-a', '-p', str(current['pid']),
                                      f'-iTCP:{self.a.port}', '-sTCP:ESTABLISHED'],
                                     capture_output=True, text=True, timeout=10)
        need(connections.returncode in (0, 1) and not connections.stderr.strip()
             and not connections.stdout.strip(), 'Production has established connections')
        self.event('production_stop', pid=current['pid'])
        self.stopped = True
        os.kill(current['pid'], signal.SIGTERM)
        end = time.monotonic() + 20
        while time.monotonic() < end:
            try:
                os.kill(current['pid'], 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        else:
            need(kernel_arguments(current['pid'])['argv_raw_hex'] == current['argv_raw_hex'],
                 'PID identity changed before kill escalation')
            os.kill(current['pid'], signal.SIGKILL)
            end = time.monotonic() + 10
            while time.monotonic() < end:
                try:
                    os.kill(current['pid'], 0)
                except ProcessLookupError:
                    break
                time.sleep(0.1)
            else:
                raise RuntimeError('Original process did not exit')
        need(listener(self.a.port, False) is None, 'Port occupied after original service exit')

    def restore(self):
        self.unchanged()
        pid = listener(self.a.port, False)
        if pid is not None:
            need(pid == self.before['pid'], 'Unknown listener: refusing to kill/adopt it')
        else:
            try:
                os.kill(self.before['pid'], 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError('Original PID alive without listener; refusing duplicate restore')
            # Restore only inherited MYDS keys, not the effective 51 config keys.
            # Original --config argv repopulates the runtime environment itself.
            env = {key: value for key, value in os.environ.items() if not key.startswith('MYDS_')}
            env.update(self.before['myds_env'])
            env['PATH'] = '/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin'
            with Path(self.before['log']).open('ab') as log, self.guard:
                proc = subprocess.Popen(self.before['argv'], cwd=self.before['cwd'], env=env,
                                        stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                        start_new_session=True)
                pid = proc.pid
                self.allowed.add(pid)
                os.setpriority(os.PRIO_PROCESS, pid, self.before['nice'])
            self.event('restoration_launched', pid=pid)
        end = time.monotonic() + 240
        last_error = None
        while time.monotonic() < end:
            try:
                restored, config_raw = capture(self.a.port)
                need(restored['pid'] == pid, 'Restored port owned by unexpected PID')
                for key in ('argv_raw_hex', 'myds_env_raw_hex', 'cwd', 'log', 'nice', 'binary',
                            'binary_sha256', 'metallib', 'metallib_sha256', 'config',
                            'config_sha256', 'effective_myds_env'):
                    need(restored[key] == self.before[key], 'Restoration identity mismatch: ' + key)
                need(config_raw == self.config_raw, 'Configuration bytes changed')
                write_json(self.a.out / 'restored.json', restored)
                return dict(restored=True, pid=pid, exact_argv_myds_artifacts_config=True)
            except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
                last_error = repr(exc)
                time.sleep(0.5)
        raise RuntimeError('Restored service did not pass identity and health: ' + str(last_error))

    def payload(self, path, bytes_expected):
        raw = path.read_bytes()
        need(len(raw) == bytes_expected, 'Trace payload byte length: ' + str(path))
        return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest())

    def capture_cases(self):
        request = dict(model='local-myds-deepseek-v4.1-flash-1017-q2-q4', stream=True,
                       messages=[dict(role='user', content='请用 Python 写一个稳定排序函数，输入是整数列表，并解释重复元素如何保持顺序。')],
                       max_tokens=32, temperature=0, top_p=1, thinking=False)
        request_path = self.a.out / 'request.json'
        write_json(request_path, request)
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
        trace_prefix = trace / 'myds'
        trace_env = dict(self.environment, MYDS_TRACE_PREFIX=str(trace_prefix),
                         MYDS_TRACE_LAYER='0,19,39', MYDS_TRACE_POS='0,1,7,15')
        self.run_command('trace', [self.a.stage / 'bin/trace_model', self.a.model,
                                  '--tokens-file', token_path], env=trace_env)
        cases = []
        for layer in (0, 19, 39):
            for position in (0, 1, 7, 15):
                def path(stage):
                    return trace / f'myds_{stage}-{layer}_pos{position}.bin'
                input_path, weight_path = path('ffn_norm'), path('route_weights')
                ids_raw = path('route_ids').read_bytes()
                need(len(ids_raw) == 24, 'Route ID trace must contain six F32 values')
                float_ids = struct.unpack('<6f', ids_raw)
                need(all(math.isfinite(v) and 0 <= v < 384 and v == int(v) for v in float_ids),
                     'Route IDs must be exact finite expert integers')
                need(len(set(float_ids)) == 6, 'Route IDs must be distinct')
                ids_path = trace / f'route_ids-{layer}_pos{position}.i32le'
                ids_path.write_bytes(struct.pack('<6i', *map(int, float_ids)))
                for p, count in ((input_path, 5120), (weight_path, 6)):
                    values = struct.unpack('<' + 'f' * count, p.read_bytes())
                    need(all(math.isfinite(v) for v in values), 'Nonfinite trace input')
                cases.append(dict(layer=layer, position=position,
                                  clamp=self.summary['clamp_metadata']['value'],
                                  input_f32=self.payload(input_path, 20480),
                                  ids_i32=self.payload(ids_path, 24),
                                  route_weights_f32=self.payload(weight_path, 24),
                                  ids_trace_f32=self.payload(path('route_ids'), 24)))
        capture_json = self.a.out / 'capture.json'
        capture_data = dict(schema_version=1, capture_kind='myds_decode_routed_moe',
                            model_bytes=self.summary['model']['bytes'], model_header_bytes=1048576,
                            model_header_sha256=self.summary['model_header_sha256'],
                            source=dict(kind='trace_model_actual_model_teacher_forced',
                                        diagnostic_only=True, performance_from_trace=False,
                                        trace_binary_sha256=sha(self.a.stage / 'bin/trace_model'),
                                        trace_metallib_sha256=sha(self.a.stage / 'bin/myds.metallib'),
                                        request_sha256=sha(request_path),
                                        token_file_sha256=sha(token_path),
                                        effective_environment_sha256=sha(self.a.out / 'effective-env.json'),
                                        model_metadata=self.summary['model'],
                                        clamp_metadata=self.summary['clamp_metadata']), cases=cases)
        write_json(capture_json, capture_data)
        return capture_json

    def run(self):
        status = 'failed'
        lock = open('/tmp/myds-codex-decode-window.lock', 'a+')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def interrupted(number, frame):
            if self.recovering:
                self.event('deferred_signal_during_restoration', signal=number)
            else:
                raise KeyboardInterrupt('Window interrupted by signal ' + str(number))
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, interrupted)
        try:
            self.before, self.config_raw = capture(self.a.port)
            self.allowed.add(self.before['pid'])
            need(os.getpriority(os.PRIO_PROCESS, os.getpid()) <= self.before['nice'],
                 'Runner priority is below original service; cannot guarantee unprivileged nice restoration')
            need(not any(flag in self.before['argv'] for flag in ('--pld', '--no-pld')),
                 'CLI PLD environment overrides require an explicit effective environment audit')
            need(Path(option(self.before['argv'], '-m')).resolve(strict=True) == self.a.model,
                 'Requested model is not the original service model')
            need(str(self.a.stage) not in self.before['binary']['path'], 'Stage overlaps production')
            expected = {'MYDS_MOE_IQ2_POPCNT_SIGN': '1', 'MYDS_MOE_IQ2_V41': '0', 'MYDS_MOE_NSG': '2'}
            for key, value in expected.items():
                need(self.before['effective_myds_env'].get(key) == value, 'Unexpected production ' + key)
            need(self.before['effective_myds_env'].get('MYDS_MOE_IQ2_CONSTANT_LUT', '0') == '0',
                 'Production IQ2 constant LUT must be off')
            for key in ('MYDS_MOE_Q2_SUM_CACHE', 'MYDS_MOE_Q2_SUM_CACHE_FUSED'):
                need(self.before['effective_myds_env'].get(key, '0') == '0',
                     'Production IQ2 provider may be bypassed: ' + key)
            mm_min = self.before['effective_myds_env'].get('MYDS_MOE_MM_MIN_TOKENS')
            need(mm_min is None or (re.fullmatch(r'-?[0-9]+', mm_min) and int(mm_min) != 1),
                 'One-token MM forcing changes the production IQ2 provider')
            self.summary.update(model=metadata(self.a.model),
                                model_header_sha256=sha(self.a.model, 1048576),
                                clamp_metadata=gguf_clamp(self.a.model))
            write_json(self.a.out / 'before.json', self.before)
            (self.a.out / 'original-myds.conf').write_bytes(self.config_raw)
            write_json(self.a.out / 'effective-env.json', self.before['effective_myds_env'])
            protocol = dict(schema_version=1, primary_nsg=2, supplemental_nsg=4,
                            primary_control='kernel_mul_mv_id_iq2_xxs_pair_swiglu_popcnt_f32',
                            required_layers=[0, 19, 39], max_absolute_control_drift_percent=2.0,
                            minimum_conservative_time_saved_percent_strict=2.0,
                            correctness='All native/release/capture gate-up-MID full F32 bits, finite, guards, writes',
                            shape64='Reported separately; not required for primary native control gate',
                            fullmodel_run=False, production_adoption=False)
            write_json(self.a.out / 'protocol.json', protocol)
            self.environment = {k: v for k, v in os.environ.items() if not k.startswith('MYDS_')}
            self.environment.update(self.before['effective_myds_env'])
            self.environment.update(DEVELOPER_DIR='/Applications/Xcode.app/Contents/Developer',
                                    PATH='/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin')
            self.scan()
            self.run_command('sdk-preflight', ['/usr/bin/xcrun', '--find', 'metal'], env=self.environment, timeout=30)
            if not self.a.skip_build:
                self.run_command('build', ['/usr/bin/make', '-C', self.a.stage, '-j8',
                                          'BIN_DIR=' + str(self.a.stage / 'bin'),
                                          'BUILD_DIR=' + str(self.a.stage / 'build'),
                                          *[self.a.stage / 'bin' / name for name in
                                            ('count_chat_tokens', 'trace_model', 'bench_iq2_signed4')]],
                                 env=self.environment)
            else:
                self.event('existing_build_reused', stage=str(self.a.stage))
            # Copy the real loaded production library before stopping it.
            backup = self.a.out / 'production-artifacts'
            backup.mkdir()
            for key in ('binary', 'metallib'):
                destination = backup / Path(self.before[key]['path']).name
                shutil.copy2(self.before[key]['path'], destination)
                need(sha(destination) == self.before[key + '_sha256'], 'Production copy hash mismatch')
            inputs = [self.a.stage / 'Makefile', self.a.stage / 'bin/myds.metallib',
                      *[self.a.stage / 'bin' / name for name in
                        ('count_chat_tokens', 'trace_model', 'bench_iq2_signed4')]]
            for directory in ('src', 'include', 'metal', 'tools'):
                inputs.extend(p for p in (self.a.stage / directory).iterdir()
                              if p.is_file() and p.suffix in ('.c', '.h', '.m', '.metal', '.py'))
            inputs.append(self.a.out / 'protocol.json')
            self.input_hashes = {str(p): sha(p) for p in sorted(set(inputs))}
            write_json(self.a.out / 'stage-inputs.json', self.input_hashes)
            self.unchanged()
            if self.a.prepare:
                status = 'prepared'
                self.summary.update(service_operations=False, gpu_work_submitted=False)
            else:
                self.thread = threading.Thread(target=self.monitor, daemon=True)
                self.thread.start()
                self.stop()
                capture_json = self.capture_cases()
                common = [self.a.stage / 'bin/bench_iq2_signed4', '--model', self.a.model,
                          '--capture-json', capture_json, '--probe-metallib', self.a.stage / 'bin/myds.metallib',
                          '--production-metallib', backup / 'myds.metallib']
                self.run_command('capture-host-validation', [*common, '--out-dir', self.a.out / 'host-validation',
                                                             '--host-only'], env=self.environment)
                nsg4_error = None
                for nsg in (2, 4):
                    try:
                        self.run_command('component-nsg' + str(nsg),
                                         [*common, '--out-dir', self.a.out / ('component-nsg' + str(nsg)),
                                          '--nsg', str(nsg), '--samples', '9', '--reps', '8'], env=self.environment)
                    except BaseException as exc:
                        if nsg == 2:
                            self.summary['correctness_pass'] = False
                            self.summary['parity_gate_fail'] = True
                            raise
                        if isinstance(exc, (KeyboardInterrupt, SystemExit)) or self.fault.is_set():
                            raise
                        nsg4_error = repr(exc)
                        self.event('supplemental_nsg4_failed', error=nsg4_error)
                need(not self.fault.is_set(), 'Competition observed')
                for path, digest in self.input_hashes.items():
                    need(sha(path) == digest, 'Stage input changed during window: ' + path)
                primary_result = analyse_component_stdout(self.a.out / 'component-nsg2.stdout', protocol)
                won = primary_result['performance_gate_pass']
                self.summary.update(**primary_result, parity_gate_fail=False,
                                    capture_json_sha256=sha(capture_json),
                                    nsg4_scope='supplemental alternate geometry; does not determine primary NSG2 gate',
                                    nsg4_correctness_pass=nsg4_error is None,
                                    nsg4_error=nsg4_error)
                status = 'component_passed' if won else 'component_rejected_performance'
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
            reason = ('Primary NSG2 native-control performance gate rejected candidate; no full-model run'
                      if status == 'component_rejected_performance' else
                      'Component-only window; separate full-model adoption gate required')
            if self.summary.get('parity_gate_fail'):
                reason = 'Primary component correctness/process gate failed; no full-model run'
            self.summary.update(status=status, fullmodel_not_run=True,
                                fullmodel_not_run_reason=reason,
                                monitor_scope='Sampled model/native benchmark process names; advisory lock; unknown renamed GPU programs may evade inspection')
            write_json(self.a.out / 'summary.json', self.summary)
            self.event('window_complete', status=status)
            lock.close()
        return 0 if status in ('prepared', 'component_passed', 'component_rejected_performance') else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8187)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true', help='Build/read-only preflight; no stop, trace, or Metal dispatch')
    parser.add_argument('--skip-build', action='store_true', help='Reuse prepared stage binaries, still hash all inputs')
    args = parser.parse_args()
    need(sys.platform == 'darwin', 'This window requires macOS KERN_PROCARGS2 and Metal')
    for key in ('stage', 'model', 'out'):
        value = getattr(args, key)
        need(value.is_absolute(), '--' + key + ' must be absolute')
        setattr(args, key, value.resolve())
    need(Path('/tmp').resolve() in args.stage.parents, '--stage must be a separate tree below /tmp')
    need((args.stage / 'Makefile').is_file() and (args.stage / 'metal/myds_iq2_v41.metal').is_file(),
         '--stage must contain the isolated myds source tree')
    need(1 <= args.port <= 65535 and args.model.is_file(), 'Invalid port/model')
    args.out.mkdir(parents=True, exist_ok=False)
    os.chmod(args.out, 0o700)
    # No connection-owned pipes remain; SIGHUP cannot abort the finally block.
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    try:
        os.setsid()
    except PermissionError:
        pass  # Already a session/process-group leader (e.g. nohup wrapper).
    with (args.out / 'runner.log').open('ab', buffering=0) as log, open(os.devnull, 'rb') as null:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        os.dup2(null.fileno(), 0)
    return Window(args).run()


if __name__ == '__main__':
    sys.exit(main())
