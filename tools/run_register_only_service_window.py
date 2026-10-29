#!/usr/bin/env python3
"""Exclusive experiment with exact macOS service capture and recovery.

Only the checked --pid gets the initial SIGTERM. An interrupted owned experiment
Popen child may get SIGINT to run its finally. Unknown processes are never killed.
Full original argv/env and artifact backups are private. No SIGKILL or pkill.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import time
import urllib.request


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def query(*args):
    return subprocess.check_output(args, text=True).strip()


def parse_procargs2(raw):
    need(len(raw) >= 5, 'short KERN_PROCARGS2 record')
    argc = struct.unpack_from('=i', raw)[0]
    need(0 < argc < 100000, 'invalid KERN_PROCARGS2 argc')
    cursor = 4
    def string():
        nonlocal cursor
        end = raw.find(b'\0', cursor)
        need(end >= 0, 'unterminated KERN_PROCARGS2 string')
        value = os.fsdecode(raw[cursor:end]); cursor = end + 1
        return value
    executable = string()
    # KERN_PROCARGS2 keeps the execname used at launch, including ./bin/myds.
    # process_arguments resolves this against the live cwd and proc_pidpath;
    # it must never rewrite argv[0], which may intentionally be relative.
    need(executable, 'empty kernel executable name')
    while cursor < len(raw) and raw[cursor] == 0:
        cursor += 1
    argv = [string() for _ in range(argc)]
    need(argv[0], 'empty original argv[0] is unsupported')
    environment = {}
    while cursor < len(raw) and raw[cursor] != 0:
        entry = string()
        need('=' in entry, 'malformed original environment entry')
        key, value = entry.split('=', 1)
        need(key and key not in environment, 'duplicate/empty original environment key')
        environment[key] = value
    need(environment, 'empty original environment is not accepted for restoration')
    return dict(executable=executable, argv=argv, environment=environment)


def process_arguments(pid):
    need(sys.platform == 'darwin', 'exact capture requires macOS KERN_PROCARGS2')
    libc = ctypes.CDLL(None, use_errno=True)
    mib = (ctypes.c_int * 3)(1, 49, pid)
    length = ctypes.c_size_t()
    call = libc.sysctl
    call.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_uint, ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p, ctypes.c_size_t]
    call.restype = ctypes.c_int
    if call(mib, 3, None, ctypes.byref(length), None, 0) != 0:
        raise OSError(ctypes.get_errno(), 'KERN_PROCARGS2 size query failed')
    need(4 < length.value <= 16 * 1048576, 'unexpected process argument size')
    buffer = ctypes.create_string_buffer(length.value)
    if call(mib, 3, buffer, ctypes.byref(length), None, 0) != 0:
        raise OSError(ctypes.get_errno(), 'KERN_PROCARGS2 read failed')
    captured = parse_procargs2(buffer.raw[:length.value])
    return normalize_process_capture(captured, process_executable(pid), process_cwd(pid))


def process_executable(pid):
    """Authoritative macOS executable path, independent of argv/execname text."""
    need(sys.platform == 'darwin', 'exact executable capture requires macOS libproc')
    library = ctypes.CDLL('/usr/lib/libproc.dylib', use_errno=True)
    call = library.proc_pidpath
    call.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
    call.restype = ctypes.c_int
    buffer = ctypes.create_string_buffer(4096)  # PROC_PIDPATHINFO_MAXSIZE
    count = call(pid, buffer, ctypes.sizeof(buffer))
    if count <= 0:
        raise OSError(ctypes.get_errno(), 'proc_pidpath failed')
    need(count < ctypes.sizeof(buffer), 'proc_pidpath result may be truncated')
    path = os.fsdecode(buffer.value)
    need(Path(path).is_absolute(), 'proc_pidpath did not return an absolute path')
    return path


def normalize_process_capture(captured, authoritative, cwd):
    """Compare execname with actual cwd/path; preserve original argv and env.

    A restored execve may use an absolute executable while retaining the old
    relative argv[0]. Compare the normalized executable, not raw execname text.
    """
    cwd = Path(cwd)
    actual = Path(authoritative)
    need(cwd.is_absolute() and actual.is_absolute(), 'authoritative cwd/path must be absolute')
    reported = Path(captured['executable'])
    resolved = (cwd / reported).resolve()
    need(resolved == actual.resolve(), 'kernel execname/cwd does not match proc_pidpath')
    return dict(executable=str(actual.resolve()), argv=captured['argv'],
                environment=captured['environment'])


def process_cwd(pid):
    rows = query('lsof', '-a', '-p', str(pid), '-d', 'cwd', '-Fn').splitlines()
    names = [row[1:] for row in rows if row.startswith('n')]
    need(len(names) == 1, 'cannot identify original process cwd')
    return Path(names[0]).resolve()


def process_nice(pid):
    raw = query('ps', '-p', str(pid), '-o', 'nice=')
    need(raw.lstrip('-').isdigit(), 'cannot read process niceness')
    return int(raw)


def stdio_targets(pid):
    """fd 0/1/2 -> absolute path, None when that descriptor is not a file.

    Production inherits stdin from start_fleet.sh's `while read ... done <
    $CONFIG_FILE` loop and appends stdout/stderr to its service log. A restore
    that silently swaps these for /dev/null and private files changes the
    process it claims to have preserved (ledger §108)."""
    rows = query('lsof', '-nP', '-a', '-p', str(pid), '-d', '0,1,2', '-Fn').splitlines()
    found, fd = {}, None
    for row in rows:
        if row.startswith('f'):
            fd = row[1:]
        elif row.startswith('n') and fd is not None:
            found[fd] = row[1:]
            fd = None
    return {key: found.get(key) for key in ('0', '1', '2')}


def processes():
    rows = []
    for line in query('ps', '-axo', 'pid=,comm=').splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        name = Path(parts[1]).name.lower()
        if name in {'myds', 'myglm', 'ds4', 'llama-server', 'ollama'} or name.startswith('bench_'):
            rows.append((int(parts[0]), name))
    return rows


def write(path, data):
    path = Path(path); temporary = path.with_name(path.name + '.new')
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, 'w') as stream:
            fd = -1
            json.dump(data, stream, indent=2, sort_keys=True)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if fd >= 0:
            os.close(fd)


def configuration_paths(argv, env, cwd):
    values = []
    for i, value in enumerate(argv):
        if value == '--config':
            need(i + 1 < len(argv), 'missing --config operand')
            values.append(argv[i + 1])
    if env.get('MYDS_CONFIG'):
        values.append(env['MYDS_CONFIG'])
    # Protect absent fallback paths too: creation could alter a restored launch.
    values += ['myds.conf', 'etc/myds.conf']
    return list(dict.fromkeys(cwd / Path(value) for value in values))


class Window:
    def __init__(self, args):
        self.a = args
        self.state = dict(stopped=False, stop_signal_sent=False, stop_requested=False,
                          restored=False, restore_blocked=False, captured_epoch=time.time())
        self.experiment = self.restored_child = None
        self.recovering = self.root_owned = False
        self.spawning_experiment = False

    def persist(self, critical=False):
        try:
            write(self.a.root / 'lease.json', self.state)
            public = {k: self.state[k] for k in ('original_pid', 'original_port',
                'environment_sha256', 'MYDS_environment', 'stopped', 'stop_signal_sent',
                'restored', 'restore_blocked', 'experiment_exit', 'experiment_pid',
                'restored_pid', 'error', 'restore_error', 'lease_write_errors') if k in self.state}
            write(self.a.root / 'summary.json', public)
        except OSError as exc:
            if critical:
                raise
            self.state.setdefault('lease_write_errors', []).append(type(exc).__name__)

    def preflight(self):
        a = self.a
        need(a.command and a.root.is_absolute() and a.cwd.is_absolute() and a.pid > 1,
             'absolute root/cwd, positive PID and experiment command required')
        a.root.mkdir(exist_ok=False, parents=True, mode=0o700); self.root_owned = True
        os.chmod(a.root, 0o700)
        captured = process_arguments(a.pid); cwd = process_cwd(a.pid)
        need(cwd == a.cwd.resolve(), 'supplied cwd differs from live cwd')
        argv = captured['argv']; executable = Path(captured['executable'])
        need(executable.name == 'myds', 'original executable is not myds')
        positions = [i for i, arg in enumerate(argv) if arg == '--port']
        need(len(positions) == 1 and positions[0] + 1 < len(argv) and
             argv[positions[0] + 1] == str(a.port), 'original port differs')
        nice = process_nice(a.pid)
        stdio = stdio_targets(a.pid)
        need(processes() == [(a.pid, 'myds')], 'other inference work prevents exclusive window')
        start = query('ps', '-p', str(a.pid), '-o', 'lstart=')
        # Match the precompiled library search paths used by myds_metal_init
        # (the engine passes "metal"). Preserve absence as well as contents:
        # creating a higher-priority library must not change the restored load.
        libraries = [cwd / 'metal/myds.metallib', cwd / 'bin/myds.metallib',
                     cwd / 'build/myds.metallib', executable.parent / 'myds.metallib',
                     executable.parent / '../bin/myds.metallib']
        protected = list(dict.fromkeys([executable, *libraries,
            *(cwd / path for path in a.protect),
            *configuration_paths(argv, captured['environment'], cwd)]))
        backups = a.root / 'original-artifacts'; backups.mkdir(mode=0o700)
        identities = {}
        for index, path in enumerate(protected):
            item = {'exists': path.exists()}
            if item['exists']:
                need(path.is_file(), 'protected artifact is not a regular file')
                item.update(resolved=str(path.resolve()), bytes=path.stat().st_size,
                            mode=path.stat().st_mode & 0o777, sha256=sha(path))
                backup = backups / f'{index:03d}-{path.name}'
                fd = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, 'wb') as target, path.open('rb') as source:
                    for block in iter(lambda: source.read(1048576), b''):
                        target.write(block)
                need(sha(backup) == item['sha256'], 'artifact changed during backup')
                item['backup'] = str(backup)
            identities[str(path)] = item
        need(identities[str(executable)]['exists'] and
             identities[str(executable.parent / 'myds.metallib')]['exists'],
             'original executable/library missing')
        env = captured['environment']
        self.state.update(original_pid=a.pid, original_port=a.port, argv=argv,
            executable=str(executable), cwd=str(cwd), start=start, environment=env,
            environment_sha256=hashlib.sha256(json.dumps(env, sort_keys=True,
                ensure_ascii=True, separators=(',', ':')).encode()).hexdigest(),
            MYDS_environment={k: v for k, v in env.items() if k.startswith('MYDS_')},
            identities=identities, experiment_command=a.command,
            nice=nice, stdio=stdio)
        self.persist(critical=True); self.verify()
        need(process_arguments(a.pid) == captured and
             query('ps', '-p', str(a.pid), '-o', 'lstart=') == start,
             'original process identity changed during capture')
        con = subprocess.run(['lsof', '-nP', '-a', '-p', str(a.pid), '-iTCP',
                              '-sTCP:ESTABLISHED'], capture_output=True, text=True)
        need(con.returncode == 1 and not con.stdout,
             'active client or failed inspection; leave original running')

    def verify(self):
        for value, item in self.state['identities'].items():
            path = Path(value)
            need(path.exists() == item['exists'], 'artifact existence changed: ' + value)
            if item['exists']:
                need(str(path.resolve()) == item['resolved'] and
                     path.stat().st_mode & 0o777 == item['mode'] and
                     path.stat().st_size == item['bytes'] and sha(path) == item['sha256'],
                     'protected artifact changed: ' + value)
                need(sha(item['backup']) == item['sha256'], 'original artifact backup changed')

    def original_alive(self):
        try:
            os.kill(self.a.pid, 0)
        except ProcessLookupError:
            return False
        # A reused PID is never a permissible target for another signal.
        return query('ps', '-p', str(self.a.pid), '-o', 'lstart=') == self.state['start']

    def wait_original_exit(self):
        deadline = time.monotonic() + self.a.stop_timeout
        while self.original_alive():
            need(time.monotonic() < deadline,
                 'original did not terminate; no SIGKILL and no duplicate restore')
            time.sleep(0.2)
        self.state['stopped'] = True

    def handle_signal(self, signum, frame):
        if self.recovering:
            self.state.setdefault('deferred_recovery_signals', []).append(signum)
            return
        if self.spawning_experiment:
            # Popen may already have started a child before returning its handle.
            # Keep ownership assignment uninterrupted so finally can drain it.
            self.state.setdefault('deferred_spawn_signals', []).append(signum)
            return
        raise RuntimeError(f'service window interrupted by signal {signum}')

    def drain_experiment(self):
        if self.experiment is not None and self.experiment.poll() is None:
            # Exact owned Popen, never a name-discovered PID. Python experiment
            # finally gets a chance to clean its own exact service children.
            self.experiment.send_signal(signal.SIGINT)
            try:
                self.experiment.wait(timeout=self.a.drain_timeout)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError('owned experiment remains; no SIGKILL or overlapping restore') from exc

    def _apply_priority(self):
        """Give the restored child the original niceness before execve.

        `nice -n N` is relative, so it cannot express "exactly N" when the
        window runner itself has a non-zero niceness; setpriority in the child
        is the only way to reproduce the value that was observed.
        """
        nice = self.state.get('nice', 0)
        if nice:
            os.setpriority(os.PRIO_PROCESS, 0, nice)

    def restore(self):
        self.drain_experiment(); self.wait_original_exit(); self.verify()
        remaining = processes()
        need(not remaining, f'unknown inference processes remain; no signals sent: {remaining}')
        # Re-create the original stdio instead of pointing it at private evidence
        # files: the restored process is equivalent only if it writes where the
        # stopped one wrote. The paths and their digests are kept as evidence.
        targets = self.state.get('stdio') or {}
        stdin_target = targets.get('0')
        stdin_handle = None
        if stdin_target and Path(stdin_target).is_file() and os.access(stdin_target, os.R_OK):
            stdin_handle = open(stdin_target, 'rb')
        out = err = None
        try:
            if targets.get('1') or targets.get('2'):
                need(targets.get('1') and targets.get('2'),
                     'partial stdout/stderr capture: refusing to guess the missing one')
                out, err = open(targets['1'], 'ab'), open(targets['2'], 'ab')
            else:
                out, err = ((self.a.root / 'restored.stdout').open('xb'),
                            (self.a.root / 'restored.stderr').open('xb'))
            self.restored_child = subprocess.Popen(self.state['argv'],
                executable=self.state['executable'], cwd=self.state['cwd'],
                env=self.state['environment'], stdin=stdin_handle or subprocess.DEVNULL,
                stdout=out, stderr=err, start_new_session=True,
                preexec_fn=self._apply_priority)
        finally:
            for handle in (stdin_handle, out, err):
                if handle is not None:
                    handle.close()
        child = self.restored_child; self.state['restored_pid'] = child.pid; self.persist()
        deadline = time.monotonic() + self.a.restore_timeout
        while time.monotonic() < deadline:
            need(child.poll() is None, 'restored service exited')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{self.a.port}/health', timeout=2) as response:
                    body = json.load(response)
                    if response.status == 200 and isinstance(body, dict) and body.get('status') == 'ok':
                        break
            except (OSError, ValueError):
                pass
            time.sleep(0.5)
        else:
            raise RuntimeError('restored health timed out; exact restored_pid retained in lease')
        self.verify()
        need(process_arguments(child.pid) == {k: self.state[k] for k in ('executable', 'argv', 'environment')},
             'restored argv/executable/full initial environment differs')
        need(process_cwd(child.pid) == Path(self.state['cwd']), 'restored cwd differs')
        need(int(query('ps', '-p', str(child.pid), '-o', 'nice=')) == self.state.get('nice', 0),
             'restored nice differs')
        restored_stdio = stdio_targets(child.pid)
        for fd, expected in (self.state.get('stdio') or {}).items():
            if expected and Path(expected).exists():
                need(restored_stdio.get(fd) == expected,
                     f'restored stdio[{fd}] differs: {restored_stdio.get(fd)} != {expected}')
        listener = query('lsof', '-nP', '-a', '-p', str(child.pid),
                         '-iTCP:' + str(self.a.port), '-sTCP:LISTEN', '-Fp')
        need('p' + str(child.pid) in listener.splitlines(), 'health listener is not restored child')
        need(processes() == [(child.pid, 'myds')], 'unexpected inference during restoration')
        self.state.update(restored=True, health_status=200, finished_epoch=time.time())

    def run(self):
        handlers = {}; exitcode = 1
        try:
            self.preflight()
            for signum in (signal.SIGINT, signal.SIGTERM):
                handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, self.handle_signal)
            need(process_arguments(self.a.pid) == {k: self.state[k] for k in
                 ('executable', 'argv', 'environment')} and
                 query('ps', '-p', str(self.a.pid), '-o', 'lstart=') == self.state['start'] and
                 process_cwd(self.a.pid) == Path(self.state['cwd']),
                 'original process changed immediately before stop')
            # Set requested before kill: even a signal arriving immediately
            # after kill returns cannot bypass the finally recovery boundary.
            self.state['stop_requested'] = True
            os.kill(self.a.pid, signal.SIGTERM)
            self.state['stop_signal_sent'] = True; self.persist()
            self.wait_original_exit(); self.persist()
            need(not processes(), 'unexpected inference process after stop')
            with (self.a.root / 'experiment.stdout').open('xb') as out, (self.a.root / 'experiment.stderr').open('xb') as err:
                self.spawning_experiment = True
                try:
                    self.experiment = subprocess.Popen(self.a.command, stdin=subprocess.DEVNULL,
                        stdout=out, stderr=err, start_new_session=True)
                finally:
                    self.spawning_experiment = False
                self.state['experiment_pid'] = self.experiment.pid; self.persist()
                if self.state.get('deferred_spawn_signals'):
                    raise RuntimeError('service window interrupted while spawning experiment: ' +
                                       repr(self.state['deferred_spawn_signals']))
                exitcode = self.experiment.wait()
            self.state['experiment_exit'] = exitcode
        except BaseException as exc:
            self.state['error'] = f'{type(exc).__name__}: {exc}'; exitcode = 1
        finally:
            self.recovering = True
            if self.state['stop_requested']:
                try:
                    self.restore()
                except BaseException as exc:
                    self.state.update(restore_blocked=True,
                        restore_error=f'{type(exc).__name__}: {exc}'); exitcode = 1
            if self.root_owned:
                self.persist()
            for signum, handler in handlers.items():
                signal.signal(signum, handler)
        print(json.dumps({k: self.state[k] for k in ('stopped', 'restored', 'restore_blocked',
            'experiment_exit', 'restored_pid', 'error', 'restore_error') if k in self.state}), flush=True)
        return exitcode


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--pid', type=int, required=True)
    p.add_argument('--cwd', type=Path, required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--protect', type=Path, action='append', default=[])
    p.add_argument('--stop-timeout', type=float, default=30)
    p.add_argument('--drain-timeout', type=float, default=75,
                   help='owned experiment cleanup budget, separate from original service stop')
    p.add_argument('--restore-timeout', type=float, default=120)
    p.add_argument('command', nargs=argparse.REMAINDER)
    args = p.parse_args(argv)
    args.command = args.command[1:] if args.command[:1] == ['--'] else args.command
    need(args.stop_timeout > 0 and args.drain_timeout > 0 and args.restore_timeout > 0,
         'timeouts must be positive')
    return Window(args).run()


if __name__ == '__main__':
    sys.exit(main())
