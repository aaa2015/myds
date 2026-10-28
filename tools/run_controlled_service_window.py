#!/usr/bin/env python3
"""Controlled macOS service window, dry-run by default.

Dry-run reports a snapshot digest; execution requires its exact expected digest,
explicit --execute and an explicit command. Never infer a PID or kill by name.
Equivalent regular-file stdio is reconstructed, not the original open-file
objects. The operator supplies an idle snapshot; this does not prove GPU
exclusivity or prevent new connections. Results are screening-only.
"""
import argparse
import ctypes
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import urllib.request

_spec = importlib.util.spec_from_file_location('base_window', Path(__file__).with_name('run_register_only_service_window.py'))
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)


def need(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def canonical_digest(value):
    data = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(',', ':')).encode()
    return hashlib.sha256(data).hexdigest()


def effective_env(initial_env, config):
    """Compute the driver's effective MYDS_ bytes without printing values."""
    effective = {os.fsencode(k):os.fsencode(v) for k,v in initial_env.items() if k.startswith('MYDS_')}
    data = config.read_bytes()
    need(b'\0' not in data, 'NUL in configuration unsupported')
    for line in data.splitlines(keepends=True):
        need(len(line) < 4096, 'overlong config line unsupported')
        if not line.strip() or line.lstrip(b' \t').startswith(b'#'):
            continue
        body = line.rstrip(b'\n\r \t')
        need(b'=' in body, 'malformed config line unsupported')
        key, value = body.split(b'=',1)
        key = key.strip(b' \t'); value = value.lstrip(b' \t')
        for i, char in enumerate(value):
            if char == 35 and i and value[i-1] in b' \t':
                value = value[:i]; break
        value = value.rstrip(b' \t')
        if key.startswith(b'MYDS_'):
            need(b'=' not in key and key, 'invalid config key')
            effective[key] = value
    return effective


def effective_env_digest(initial_env, config):
    effective = effective_env(initial_env, config)
    entries = sorted(key+b'='+value for key,value in effective.items())
    return hashlib.sha256(b''.join(entry+b'\0' for entry in entries)).hexdigest()


class ProcFileInfo(ctypes.Structure):
    _fields_ = [('flags', ctypes.c_uint32), ('status', ctypes.c_uint32),
                ('offset', ctypes.c_int64), ('kind', ctypes.c_int32), ('guard', ctypes.c_uint32)]


class VnodeStatHead(ctypes.Structure):
    _fields_ = [('dev',ctypes.c_uint32), ('mode',ctypes.c_uint16),
                ('nlink',ctypes.c_uint16), ('inode',ctypes.c_uint64)]


def fd_info(pid, fd):
    need(sys.platform == 'darwin', 'proc_pidfdinfo requires macOS')
    library = ctypes.CDLL('/usr/lib/libproc.dylib', use_errno=True)
    fn = library.proc_pidfdinfo
    fn.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
    fn.restype = ctypes.c_int
    data = ctypes.create_string_buffer(4096)
    n = fn(pid, fd, 2, data, len(data))
    need(n >= ctypes.sizeof(ProcFileInfo) + ctypes.sizeof(VnodeStatHead), 'proc_pidfdinfo failed')
    v = ProcFileInfo.from_buffer_copy(data)
    vnode = VnodeStatHead.from_buffer_copy(data, ctypes.sizeof(ProcFileInfo))
    rows = base.query('lsof', '-nP', '-a', '-p', str(pid), '-d', str(fd), '-Fn').splitlines()
    names = [r[1:] for r in rows if r.startswith('n')]
    need(len(names) == 1 and Path(names[0]).is_absolute(), 'cannot identify descriptor path')
    p = Path(names[0]); st = p.stat()
    need(p.is_file(), 'stdio must be regular files')
    need((st.st_dev,st.st_ino,st.st_mode) == (vnode.dev,vnode.inode,vnode.mode),
         'stdio path no longer points to process vnode')
    return dict(path=str(p), dev=st.st_dev, inode=st.st_ino, mode=st.st_mode,
                flags=v.flags, status=v.status, offset=v.offset, kind=v.kind, guard=v.guard)


def active_clients(pid):
    result = subprocess.run(['lsof', '-nP', '-a', '-p', str(pid), '-iTCP', '-sTCP:ESTABLISHED'],
                            capture_output=True, text=True)
    need(result.returncode == 1 and not result.stdout and not result.stderr,
         'active connection or lsof failed')


def model_identity(path):
    resolved = path.resolve(strict=True)
    st = resolved.stat()
    need(resolved.is_file(), 'model is not a regular file')
    with resolved.open('rb') as f:
        header = hashlib.sha256(f.read(1048576)).hexdigest()
    return dict(path=str(resolved), dev=st.st_dev, inode=st.st_ino, size=st.st_size,
                mtime_ns=st.st_mtime_ns, header_sha256=header)


def harness_identity(args):
    path = args.harness
    need(path.is_absolute() and path.name == 'bench_indexer_mask_warm',
         'only the fixed mask-warm native harness is supported')
    need(not path.is_symlink(), 'harness symlink is unsupported')
    st = path.stat()
    need(path.is_file() and os.access(path, os.X_OK), 'native harness is unavailable')
    with path.open('rb') as f: magic = f.read(4)
    need(magic in (b'\xcf\xfa\xed\xfe', b'\xca\xfe\xba\xbe', b'\xbe\xba\xfe\xca'),
         'harness is not a supported Mach-O binary')
    digest = base.sha(path)
    need(digest == args.harness_sha256, 'native harness SHA mismatch')
    cwd = path.parent
    candidates = [cwd/'metal/myds.metallib', cwd/'bin/myds.metallib', cwd/'build/myds.metallib',
                  cwd/'metal/myds.metallib', path.parent/'myds.metallib', path.parent/'../bin/myds.metallib']
    libraries = []
    for candidate in candidates:
        item = {'path':str(candidate), 'exists':candidate.exists()}
        if item['exists']:
            need(candidate.is_file(), 'experimental library search candidate is not regular')
            item.update(resolved=str(candidate.resolve()),sha256=base.sha(candidate))
        libraries.append(item)
    existing = [item for item in libraries if item['exists']]
    need(existing and Path(existing[0]['resolved']) == args.experiment_library.resolve(strict=True),
         'experiment library is not the first existing search candidate')
    arms = getattr(args, 'harness_arms', None) or [{'role':'single', 'argv':args.harness_arg}]
    need([x.get('role') for x in arms] in (['single'], ['A1','B','A2']), 'invalid arm schedule')
    completed = getattr(args, 'completed_arms', 0)
    rows = []; outputs = []
    for index, arm in enumerate(arms):
        need(isinstance(arm,dict) and set(arm) == {'role','argv'}, 'invalid arm fields')
        argv = arm.get('argv')
        need(isinstance(argv,list) and all(isinstance(x,str) for x in argv) and
             len(argv) == 11 and argv[0] == '--run', 'invalid cold/warm native harness arguments')
        need(Path(argv[1]).is_absolute() and Path(argv[1]).resolve() == args.model.resolve(),
             'harness model differs from frozen model')
        inputs = {}
        for value in (argv[2], argv[3], argv[4], argv[5], argv[8]):
            f = Path(value)
            need(f.is_absolute() and f.is_file() and not f.is_symlink(), 'harness input must be an absolute regular file')
            inputs[str(f)] = base.sha(f)
        output = Path(argv[6]); outputs.append(str(output))
        need(output.is_absolute() and output.parent.is_dir() and not output.is_symlink() and
             (output.is_dir() if index < completed else not output.exists()),
             'harness output directory has unexpected state')
        need(argv[7].isdigit() and 0 < int(argv[7]) <= 0xffffffff, 'invalid max sequence length')
        need(re.fullmatch(r'[0-9a-f]{64}', argv[9]) is not None, 'invalid effective MYDS environment SHA')
        need(argv[10] in ('on', 'off'), 'invalid mask-skip mode')
        rows.append(dict(role=arm['role'],argv=list(argv),inputs=inputs))
    need(len(set(outputs)) == len(outputs), 'arm output directories must differ')
    if len(rows) == 3:
        aa, bb, cc = (row['argv'] for row in rows)
        need([aa[10],bb[10],cc[10]] == ['off','on','off'], 'A1/B/A2 must use off/on/off')
        need(aa[1:6] == bb[1:6] == cc[1:6] and aa[7] == bb[7] == cc[7], 'arm workload differs')
        need(Path(aa[8]).read_bytes() == Path(cc[8]).read_bytes() and aa[9] == cc[9], 'A1/A2 config differs')
        baseline, baseline_value = mask_config(Path(args.config))
        need(baseline_value in (None, '0'), 'production mask selector is not baseline off')
        for argv in (aa,bb,cc):
            other, value = mask_config(Path(argv[8]))
            need(other == baseline, 'arm config differs beyond mask selector')
            need(value == ('1' if argv[10] == 'on' else '0') or
                 (argv[10] == 'off' and value is None), 'arm config selector mismatch')
    return dict(path=str(path), dev=st.st_dev, inode=st.st_ino, size=st.st_size,
                mtime_ns=st.st_mtime_ns, sha256=digest, arms=rows,
                library_search=libraries, selected_library=existing[0])


def mask_config(path):
    data = path.read_bytes(); rest = []; found = []
    for line in data.splitlines(keepends=True):
        body = line.rstrip(b'\r\n')
        if re.match(rb'^[ \t]*MYDS_INDEXER_MASK_SKIP[ \t]*=', body):
            match = re.fullmatch(rb'[ \t]*MYDS_INDEXER_MASK_SKIP[ \t]*=[ \t]*([01])[ \t]*(?:#.*)?', body)
            need(match is not None, 'unsupported mask selector syntax')
            found.append(match.group(1).decode())
        else: rest.append(line)
    need(len(found) <= 1, 'duplicate mask selector')
    return b''.join(rest), found[0] if found else None


def validate_arm_summary(identity, index):
    harness = identity['harness']; arm = harness['arms'][index]; argv = arm['argv']
    output = Path(argv[6]); summary_path = output/'summary.json'
    need(summary_path.is_file() and not summary_path.is_symlink(), 'arm summary missing or unsupported')
    summary = json.loads(summary_path.read_text())
    need(summary.get('kind') == 'ordinary_warm_prefill_screen' and summary.get('observer') is False,
         'arm is not ordinary warm-prefill screen')
    need(summary.get('executable_sha256') == harness['sha256'] and
         Path(summary.get('executable_path','')).resolve() == Path(harness['path']).resolve(),
         'arm actual executable identity mismatch')
    selected = harness['selected_library']
    need(summary.get('metallib_sha256') == selected['sha256'] and
         Path(summary.get('metallib_path','')).resolve() == Path(selected['resolved']),
         'arm actual metallib identity mismatch')
    need(summary.get('config_sha256') == arm['inputs'][argv[8]] and
         Path(summary.get('config_path','')).resolve() == Path(argv[8]).resolve(), 'arm config identity mismatch')
    need(summary.get('effective_myds_environment',{}).get('sha256') == argv[9] and
         summary.get('post_run_myds_environment',{}).get('sha256') == argv[9] and
         summary.get('mask_skip_expected') == argv[10], 'arm effective environment or mode mismatch')
    model = identity['model']
    need(summary.get('model_bytes') == model['size'] and summary.get('model_header_sha256') == model['header_sha256'] and
         Path(summary.get('model_path','')).resolve() == Path(model['path']), 'arm model identity mismatch')
    phases = summary.get('phases')
    need(isinstance(phases,list) and len(phases)==2 and
         [p.get('phase') for p in phases] == ['cold','warm_append'], 'missing cold/warm phase')
    need(phases[0].get('cached_tokens') == 0 and type(phases[1].get('cached_tokens')) is int and
         phases[1]['cached_tokens'] > 0, 'warm prefix was not reused')
    for phase, prompt, tokens in zip(phases,(argv[2],argv[4]),(argv[3],argv[5])):
        need(phase.get('prompt_sha256') == arm['inputs'][prompt] and
             phase.get('tokens_sha256') == arm['inputs'][tokens], 'arm input digest mismatch')
        for kind in ('output','output_token_ids'):
            item = phase.get(kind)
            need(isinstance(item,dict) and isinstance(item.get('path'),str), 'missing arm output descriptor')
            name = item['path']; payload = output/name
            need(Path(name).name == name and payload.is_file() and not payload.is_symlink(), 'invalid output payload')
            need(payload.stat().st_size == item.get('bytes') and base.sha(payload) == item.get('sha256'),
                 'arm output payload mismatch')
    counts = summary.get('mask_skip_warm_counters',{})
    need(counts.get('failures') == 0 and
         (type(counts.get('encoded')) is int and
          (counts['encoded'] > 0 if argv[10] == 'on' else counts['encoded'] == 0)),
         'arm provider witness mismatch')
    return summary


def compare_arm_outputs(reference, current):
    for left, right in zip(reference['phases'], current['phases']):
        for key in ('actual_tokens','cached_tokens','prefill_tokens','stop_reason'):
            need(left.get(key) == right.get(key), 'arm token or cache geometry differs')
        for kind in ('output','output_token_ids'):
            need(left[kind]['sha256'] == right[kind]['sha256'] and
                 left[kind]['bytes'] == right[kind]['bytes'], 'arm output differs')


def check_available_memory(min_gib):
    page_bytes = int(base.query('sysctl', '-n', 'hw.pagesize'))
    stats = base.query('vm_stat')
    counts = {}
    for key in ('free', 'inactive', 'speculative'):
        match = re.search(r'^Pages ' + key + r':\s*(\d+)\.', stats, re.MULTILINE)
        need(match is not None, 'vm_stat available-memory counter missing')
        counts[key] = int(match.group(1))
    need(sum(counts.values()) * page_bytes >= min_gib * 1024**3,
         'available memory below frozen threshold')


def verify_priority_can_be_restored(identity):
    # Harmless, short-lived CPU-only child, under the controller's credentials.
    source = ('import os,sys; n=int(sys.argv[1]); '
              'os.setpriority(os.PRIO_PROCESS,0,n); '
              'print(os.geteuid(),os.getegid(),os.getpriority(os.PRIO_PROCESS,0))')
    result = subprocess.run([sys.executable, '-c', source, str(identity['nice'])],
                            capture_output=True, text=True, timeout=5)
    expected = f"{identity['uid']} {identity['gid']} {identity['nice']}"
    need(result.returncode == 0 and result.stdout.strip() == expected,
         'cannot restore target uid/gid/nice under current credentials')


def snapshot(args):
    cap = base.process_arguments(args.pid)
    cwd = base.process_cwd(args.pid)
    need(cwd == args.cwd.resolve(), 'cwd mismatch')
    need(Path(cap['executable']).name == 'myds', 'not myds executable')
    positions = [i for i, x in enumerate(cap['argv']) if x == '--port']
    need(len(positions) == 1 and cap['argv'][positions[0] + 1:positions[0] + 2] == [str(args.port)],
         'port mismatch')
    ctx_positions = [i for i, x in enumerate(cap['argv']) if x == '--ctx-size']
    need(len(ctx_positions) == 1 and ctx_positions[0] + 1 < len(cap['argv']), 'ctx-size missing')
    expected_ctx = cap['argv'][ctx_positions[0] + 1]
    need(expected_ctx.isdigit(), 'ctx-size invalid')
    need(base.processes() == [(args.pid, 'myds')], 'other inference process present')
    active_clients(args.pid)
    listeners = base.query('lsof', '-nP', '-iTCP:' + str(args.port), '-sTCP:LISTEN', '-Fp')
    need([r for r in listeners.splitlines() if r.startswith('p')] == ['p' + str(args.pid)],
         'target is not the sole port listener')
    nice = int(base.query('ps', '-p', str(args.pid), '-o', 'nice='))
    need(nice == args.expected_nice, 'nice mismatch')
    uid = int(base.query('ps', '-p', str(args.pid), '-o', 'uid='))
    gid = int(base.query('ps', '-p', str(args.pid), '-o', 'gid='))
    need(uid == os.geteuid() and gid == os.getegid(), 'process uid/gid differ from controller')
    start = base.query('ps', '-p', str(args.pid), '-o', 'lstart=')
    fds = [fd_info(args.pid, i) for i in (0, 1, 2)]
    # FREAD=1, FWRITE=2 and FAPPEND=8. XNU FWASWRITTEN=0x10000 is
    # historical (set after a write), not an open mode to reproduce.
    need(fds[0]['flags'] == 0x1 and fds[0]['kind'] == 1 and not fds[0]['guard'], 'unsupported stdin')
    need(all((x['flags'] & ~0x10000) == 0xa and x['kind'] == 1 and not x['guard'] for x in fds[1:]),
         'unsupported log descriptors')
    need(all(x['status'] & 1 for x in fds[1:]), 'log descriptors lack observed shared status')
    need(all(fds[1][k] == fds[2][k] for k in ('path', 'dev', 'inode', 'mode', 'flags', 'offset')),
         'stdout/stderr differ')
    need(Path(fds[0]['path']) == args.stdin_path and Path(fds[1]['path']) == args.log_path,
         'stdio path mismatch')
    need(0 <= fds[0]['offset'] <= Path(fds[0]['path']).stat().st_size, 'stdin offset unsupported')
    lib = Path(args.library).resolve(strict=True)
    config = Path(args.config).resolve(strict=True)
    need(str(config) in cap['argv'], 'config absent from original argv')
    executable = Path(cap['executable'])
    candidates = [cwd/'metal/myds.metallib', cwd/'bin/myds.metallib', cwd/'build/myds.metallib',
                  executable.parent/'myds.metallib', executable.parent/'../bin/myds.metallib']
    existing = [candidate for candidate in candidates if candidate.is_file()]
    need(existing and existing[0].resolve() == lib, 'library does not match first search path')
    model = model_identity(args.model)
    need(any(Path(x).resolve() == Path(model['path']) for x in cap['argv'] if x.startswith('/')),
         'model absent from original argv')
    # Recheck on return, then again immediately before stop. Log offset is
    # intentionally excluded: production may append while it is still running.
    for item in fds[1:]:
        item.pop('offset')
        item['flags'] &= ~0x10000
    harness = harness_identity(args)
    need(all(row['argv'][7] == expected_ctx for row in harness['arms']), 'harness ctx-size differs from service')
    for row in harness['arms']:
        argv = row['argv']; env = effective_env(cap['environment'], Path(argv[8]))
        need(effective_env_digest(cap['environment'], Path(argv[8])) == argv[9],
             'expected effective environment digest differs before stop')
        need(env.get(b'MYDS_PREFIX_CACHE') == b'1' and env.get(b'MYDS_PLD') == b'1' and
             env.get(b'MYDS_METAL_CONCURRENT') == b'1', 'production cache/PLD/concurrency not preserved')
        selector = env.get(b'MYDS_INDEXER_MASK_SKIP')
        need(selector == b'1' if argv[10] == 'on' else selector in (None,b'0'),
             'effective mask selector differs before stop')
    return dict(pid=args.pid, start=start, cwd=str(cwd), nice=nice, uid=uid, gid=gid,
                process=cap, fds=fds, model=model, ctx_size=int(expected_ctx),
                harness=harness, min_available_gib=args.min_available_gib,
                artifacts={str(p): base.sha(p) for p in (Path(cap['executable']), lib, config)})


class ControlledWindow(base.Window):
    def __init__(self, args):
        super().__init__(args)
        self.stdio = []
        self.lock_fd = None
        self.identity = None
        self.spawning_restore = False
        self.state.update(screening_only=True, exclusivity_unproven=True)

    def persist(self, critical=False):
        super().persist(critical=critical)
        summary_path = self.a.root/'summary.json'
        try:
            if summary_path.exists():
                summary = json.loads(summary_path.read_text())
                summary.update(screening_only=True, exclusivity_unproven=True)
                base.write(summary_path, summary)
        except (OSError, ValueError) as exc:
            if critical: raise
            self.state.setdefault('lease_write_errors', []).append(type(exc).__name__)

    def preflight(self):
        a = self.a
        need(a.command and a.root.is_absolute() and a.cwd.is_absolute() and a.pid > 1, 'invalid inputs')
        need(a.lock.is_absolute(), 'absolute shared lock required')
        self.lock_fd = os.open(a.lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.identity = snapshot(a)
        need(canonical_digest(self.identity) == a.expected_identity, 'expected identity digest differs')
        verify_priority_can_be_restored(self.identity)
        a.root.mkdir(exist_ok=False, parents=True, mode=0o700); self.root_owned = True
        os.chmod(a.root, 0o700)
        captured = self.identity['process']; cwd = Path(self.identity['cwd']); executable = Path(captured['executable'])
        libraries = [cwd / 'metal/myds.metallib', cwd / 'bin/myds.metallib', cwd / 'build/myds.metallib',
                     executable.parent / 'myds.metallib', executable.parent / '../bin/myds.metallib']
        protected = list(dict.fromkeys([executable, Path(a.library), Path(a.config), *libraries,
            *(cwd / p for p in a.protect), *base.configuration_paths(captured['argv'], captured['environment'], cwd)]))
        backups = a.root / 'original-artifacts'; backups.mkdir(mode=0o700)
        identities = {}
        for index, path in enumerate(protected):
            item = {'exists': path.exists()}
            if item['exists']:
                need(path.is_file(), 'protected artifact is not regular')
                item.update(resolved=str(path.resolve()), bytes=path.stat().st_size,
                            mode=path.stat().st_mode & 0o777, sha256=base.sha(path))
                backup = backups / f'{index:03d}-{path.name}'
                fd = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, 'wb') as target, path.open('rb') as source:
                    for block in iter(lambda: source.read(1048576), b''): target.write(block)
                need(base.sha(backup) == item['sha256'], 'backup differs')
                item['backup'] = str(backup)
            identities[str(path)] = item
        self.state.update(original_pid=a.pid, original_port=a.port, argv=captured['argv'],
            executable=str(executable), cwd=str(cwd), start=self.identity['start'],
            environment=captured['environment'], environment_sha256=canonical_digest(captured['environment']),
            MYDS_environment={k:v for k,v in captured['environment'].items() if k.startswith('MYDS_')},
            identities=identities, expected_identity=a.expected_identity, experiment_command=a.command,
            nice=self.identity['nice'], stdio_capture=self.identity['fds'])
        self.persist(critical=True); self.verify()
        # No create/truncate. Ensure the opened handles and paths still refer to
        # the captured inodes; stdout and stderr will duplicate one opened fd.
        try:
            self.stdio.append(os.open(a.stdin_path, os.O_RDONLY | os.O_NOFOLLOW))
            self.stdio.append(os.open(a.log_path, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW))
            self.verify_stdio()
            os.lseek(self.stdio[0], self.identity['fds'][0]['offset'], os.SEEK_SET)
            need(snapshot(a) == self.identity, 'original identity drifted during capture')
        except BaseException:
            self.close_fds(); raise

    def verify_stdio(self):
        need(len(self.stdio) == 2, 'missing captured stdio')
        for fd, original in zip(self.stdio, (self.identity['fds'][0], self.identity['fds'][1])):
            s = os.fstat(fd); path = os.stat(original['path'], follow_symlinks=False)
            expected = (original['dev'], original['inode'], original['mode'])
            need((s.st_dev,s.st_ino,s.st_mode) == expected and
                 (path.st_dev,path.st_ino,path.st_mode) == expected, 'stdio inode or path drift')
        need(fcntl.fcntl(self.stdio[1], fcntl.F_GETFL) & os.O_APPEND, 'log O_APPEND lost')

    def verify(self):
        super().verify()
        need(model_identity(self.a.model) == self.identity['model'], 'model identity drift')

    def close_fds(self):
        for fd in self.stdio: os.close(fd)
        self.stdio = []

    def handle_signal(self, signum, frame):
        if self.spawning_restore:
            self.state.setdefault('deferred_recovery_signals', []).append(signum); return
        return super().handle_signal(signum, frame)

    def restore(self):
        self.drain_experiment(); self.wait_original_exit(); self.verify(); self.verify_stdio()
        if self.experiment is not None:
            # A remaining member of the owned session's original process group
            # prevents overlapping restoration; never signal the group.
            try:
                os.killpg(self.experiment.pid, 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError('experiment process group still exists')
        remaining = base.processes()
        need(not remaining, f'unknown inference processes remain: {remaining}')
        # The controller can verify its own experiment process, not arbitrary
        # escaped descendants. The command's documented no-daemon contract is
        # an explicit external assumption of execution.
        os.lseek(self.stdio[0], self.identity['fds'][0]['offset'], os.SEEK_SET)
        self.spawning_restore = True
        try:
            self.restored_child = subprocess.Popen(self.state['argv'], executable=self.state['executable'],
                cwd=self.state['cwd'], env=self.state['environment'], stdin=self.stdio[0],
                stdout=self.stdio[1], stderr=self.stdio[1], start_new_session=True,
                preexec_fn=lambda: os.setpriority(os.PRIO_PROCESS, 0, self.state['nice']))
        finally:
            self.spawning_restore = False
        child = self.restored_child; self.state['restored_pid'] = child.pid; self.persist()
        deadline = time.monotonic() + self.a.restore_timeout
        while time.monotonic() < deadline:
            need(child.poll() is None, 'restored service exited')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{self.a.port}/health', timeout=2) as response:
                    body = json.load(response)
                    if response.status == 200 and isinstance(body, dict) and body.get('status') == 'ok': break
            except (OSError, ValueError): pass
            time.sleep(.5)
        else: raise RuntimeError('restored health timed out; restored PID retained')
        self.verify(); self.verify_stdio()
        need(base.process_arguments(child.pid) == {k:self.state[k] for k in ('executable','argv','environment')}, 'restored process identity mismatch')
        need(base.process_cwd(child.pid) == Path(self.state['cwd']), 'restored cwd mismatch')
        need(int(base.query('ps','-p',str(child.pid),'-o','nice=')) == self.state['nice'], 'restored nice mismatch')
        need(int(base.query('ps','-p',str(child.pid),'-o','uid=')) == self.identity['uid'] and
             int(base.query('ps','-p',str(child.pid),'-o','gid=')) == self.identity['gid'],
             'restored uid/gid mismatch')
        for n, original in enumerate(self.identity['fds']):
            actual = fd_info(child.pid, n)
            if n in (1,2): actual['flags'] &= ~0x10000
            for key in ('path','dev','inode','mode','flags','kind','guard'):
                need(actual[key] == original[key], f'restored fd {n} {key} mismatch')
            if n == 0: need(actual['offset'] == original['offset'], 'restored stdin offset mismatch')
            if n in (1,2): need(actual['status'] & 1, 'restored log fd is not shared')
        listener = base.query('lsof','-nP','-iTCP:'+str(self.a.port),'-sTCP:LISTEN','-Fp')
        need([r for r in listener.splitlines() if r.startswith('p')] == ['p'+str(child.pid)],
             'restored child is not sole listener')
        need(base.processes() == [(child.pid, 'myds')], 'unexpected inference process')
        self.state.update(restored=True, health_status=200, finished_epoch=time.time())

    def run(self):
        return self.run_controlled()

    def run_controlled(self):
        handlers = {}; exitcode = 1
        try:
            self.preflight()
            for signum in (signal.SIGINT, signal.SIGTERM):
                handlers[signum] = signal.getsignal(signum); signal.signal(signum, self.handle_signal)
            need(snapshot(self.a) == self.identity, 'last-moment identity or clients changed')
            self.verify_stdio(); self.verify()
            active_clients(self.a.pid)
            need(base.process_arguments(self.a.pid) == self.identity['process'] and
                 base.query('ps','-p',str(self.a.pid),'-o','lstart=') == self.identity['start'],
                 'original process changed immediately before signal')
            self.state['stop_requested'] = True
            os.kill(self.a.pid, signal.SIGTERM)
            self.state['stop_signal_sent'] = True; self.persist()
            self.wait_original_exit(); self.persist()
            need(not base.processes(), 'unexpected inference after stop')
            check_available_memory(self.a.min_available_gib)
            need(harness_identity(self.a) == self.identity['harness'], 'harness changed before launch')
            self.state['arms'] = []
            baseline_summary = None
            for index, arm in enumerate(self.a.harness_arms):
                self.a.completed_arms = index
                check_available_memory(self.a.min_available_gib)
                need(harness_identity(self.a) == self.identity['harness'], 'harness or inputs changed before arm')
                need(not base.processes(), 'unknown inference before arm')
                with (self.a.root/f'experiment-{arm["role"]}.stdout').open('xb') as out, \
                     (self.a.root/f'experiment-{arm["role"]}.stderr').open('xb') as err:
                    self.spawning_experiment = True
                    try:
                        self.experiment = subprocess.Popen([str(self.a.harness), *arm['argv']],
                            stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True,
                            cwd=str(self.a.harness.parent), env=self.state['environment'])
                    finally:
                        self.spawning_experiment = False
                    self.state['experiment_pid'] = self.experiment.pid
                    self.state['arms'].append(dict(role=arm['role'],pid=self.experiment.pid))
                    self.persist()
                    need(not self.state.get('deferred_spawn_signals'), 'interrupted during experiment spawn')
                    exitcode = self.experiment.wait()
                self.state['arms'][-1]['exit'] = exitcode
                self.state['experiment_exit'] = exitcode; self.persist()
                if exitcode != 0: break
                self.a.completed_arms = index + 1
                need(harness_identity(self.a) == self.identity['harness'], 'harness, inputs or output state changed')
                summary = validate_arm_summary(self.identity, index)
                if baseline_summary is None:
                    baseline_summary = summary
                else:
                    compare_arm_outputs(baseline_summary, summary)
                try: os.killpg(self.experiment.pid, 0)
                except ProcessLookupError: pass
                else: raise RuntimeError('completed experiment process group still exists')
                need(not base.processes(), 'unknown inference after arm')
        except BaseException as exc:
            self.state['error'] = f'{type(exc).__name__}: {exc}'; exitcode = 1
        finally:
            self.recovering = True
            if self.state['stop_requested']:
                try: self.restore()
                except BaseException as exc:
                    self.state.update(restore_blocked=True, restore_error=f'{type(exc).__name__}: {exc}'); exitcode = 1
            if self.root_owned: self.persist()
            for signum, handler in handlers.items(): signal.signal(signum, handler)
            self.close_fds()
            if self.lock_fd is not None: os.close(self.lock_fd); self.lock_fd = None
        print(json.dumps({k:self.state[k] for k in ('stopped','restored','restore_blocked','experiment_exit','restored_pid','error','restore_error','screening_only','exclusivity_unproven') if k in self.state}), flush=True)
        return exitcode


def main():
    if len(sys.argv) == 4 and sys.argv[1] == '--effective-env-digest':
        pid = int(sys.argv[2]); config = Path(sys.argv[3])
        need(pid > 1 and config.is_absolute() and config.is_file(), 'PID and absolute config required')
        captured = base.process_arguments(pid)
        print(effective_env_digest(captured['environment'], config))
        return 0
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pid', type=int, required=True); p.add_argument('--cwd', type=Path, required=True)
    p.add_argument('--port', type=int, required=True); p.add_argument('--stdin-path', type=Path, required=True)
    p.add_argument('--log-path', type=Path, required=True); p.add_argument('--config', type=Path, required=True)
    p.add_argument('--library', type=Path, required=True); p.add_argument('--expected-nice', type=int, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--harness', type=Path, required=True)
    p.add_argument('--harness-sha256', required=True)
    p.add_argument('--experiment-library', type=Path, required=True)
    p.add_argument('--harness-arg', action='append', default=[])
    p.add_argument('--sequence-json', type=Path, help='frozen JSON with A1/B/A2 argv arrays')
    p.add_argument('--min-available-gib', type=float, required=True)
    p.add_argument('--execute', action='store_true'); p.add_argument('--expected-identity')
    p.add_argument('--root', type=Path); p.add_argument('--lock', type=Path)
    p.add_argument('--acknowledge-idle-snapshot-only', action='store_true')
    p.add_argument('--protect', type=Path, action='append', default=[])
    p.add_argument('--stop-timeout', type=float, default=30); p.add_argument('--drain-timeout', type=float, default=75)
    p.add_argument('--restore-timeout', type=float, default=120)
    a = p.parse_args()
    if a.sequence_json:
        need(not a.harness_arg and a.sequence_json.is_absolute() and
             a.sequence_json.is_file() and not a.sequence_json.is_symlink(),
             'sequence JSON must be an absolute regular file and cannot be combined with harness args')
        value = json.loads(a.sequence_json.read_text())
        need(isinstance(value, dict) and set(value) == {'arms'} and isinstance(value['arms'], list),
             'sequence JSON must contain only arms')
        a.harness_arms = value['arms']
    else:
        a.harness_arms = [dict(role='single', argv=a.harness_arg)]
    need(a.harness_arms and isinstance(a.harness_arms[0],dict) and
         isinstance(a.harness_arms[0].get('argv'),list), 'missing harness arm')
    a.completed_arms = 0
    a.command = [str(a.harness), *a.harness_arms[0]['argv']]
    need(all(x.is_absolute() for x in (a.cwd,a.stdin_path,a.log_path,a.config,a.library,a.model,
                                       a.harness,a.experiment_library)), 'absolute paths required')
    if not a.execute:
        value = snapshot(a)
        print(json.dumps({'dry_run':True, 'identity_sha256':canonical_digest(value),
            'pid':a.pid, 'cwd':value['cwd'], 'nice':value['nice'], 'artifacts':value['artifacts'],
            'stdio':value['fds'], 'model':value['model'], 'harness':value['harness'],
            'min_available_gib':value['min_available_gib'], 'screening_only':True,
            'exclusivity_unproven':True}, sort_keys=True))
        return 0
    need(a.expected_identity and re.fullmatch(r'[0-9a-f]{64}', a.expected_identity) and a.root and a.lock,
         'execution requires identity digest, root and lock')
    need(a.acknowledge_idle_snapshot_only,
         'execution requires acknowledging that the idle snapshot does not prove exclusivity')
    need(a.min_available_gib > 0, 'positive memory threshold required')
    need(all(x>0 for x in (a.stop_timeout,a.drain_timeout,a.restore_timeout)), 'timeouts must be positive')
    return ControlledWindow(a).run()


if __name__ == '__main__':
    sys.exit(main())
