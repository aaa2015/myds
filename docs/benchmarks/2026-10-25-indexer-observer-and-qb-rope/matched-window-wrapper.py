import pathlib, json, os, subprocess, fcntl, time, hashlib, sys

root = pathlib.Path('/private/tmp/myds-indexer-r3-20261025')
base = pathlib.Path('/private/tmp/myds-indexer-r2-20261025')
stage=sys.argv[1]
assert stage in ('calibration','formal')
out = root / ('matched-'+stage)
out.mkdir(exist_ok=False)
cwd = root / ('matched-cwd-'+stage)
cwd.mkdir(exist_ok=False)
lock = open('/tmp/myds-codex-decode-window.lock', 'a+')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

canonical = [pathlib.Path(p) for p in (
    '/Users/fangqingyuan/Projects/myds/bin/myds',
    '/Users/fangqingyuan/Projects/myds/bin/myds.metallib',
    '/Users/fangqingyuan/Projects/myfleet/myds.conf')]

def identities():
    return {str(p): sha(p) for p in canonical}

def processes():
    p = subprocess.run(['ps', '-axo', 'pid=,comm='], capture_output=True, text=True, check=True)
    rows = []
    for line in p.stdout.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        pid = int(parts[0])
        name = pathlib.Path(parts[1]).name.lower()
        if name in {'myds', 'myglm', 'ds4', 'llama-server', 'ollama', 'capture_indexer_k'} or name.startswith('bench_'):
            rows.append({'pid': pid, 'comm': parts[1]})
    return rows

before = identities()
expected = json.loads((base / 'baseline-identity.json').read_text())
if any(before[str(p)] != expected[str(p)]['sha256'] for p in canonical):
    (out / 'refused.json').write_text(json.dumps({'reason': 'canonical artifact identity drift', 'actual': before}, indent=2))
    sys.exit('canonical identity changed since freeze')
initial = processes()
if initial:
    (out / 'refused.json').write_text(json.dumps({'recognized': initial}, indent=2))
    sys.exit('recognized competing inference process')
cmd = [str(root / 'matched-build-v2/bench_indexer_matched_replay'),
       '--capture-json', str(base / 'capture-verified.json'),
       '--metallib', str(base / 'full-build/probe.metallib'),
       '--native-metallib', str(base / 'deployed/myds.metallib'),
       '--config', str(base / 'deployed/myds.conf'),
       '--out-dir', str(out / 'outputs'), '--samples', '9']
if stage=='calibration':cmd.append('--sham-tiled-b')
reader_sha=sha(root/'source/tools/review_indexer_matched.py')
prior=None
if stage=='formal':
    prior_path=root/'matched-calibration-review.json'
    prior=json.loads(prior_path.read_text())
    assert prior['stage']=='calibration' and prior['review_completed'] and prior['calibration_passed']
    assert prior['reader_sha256']==reader_sha and prior['protocol_sha256']==sha(root/'matched-protocol.json')
env = {k: v for k, v in os.environ.items() if not k.startswith('MYDS_')}
env['DEVELOPER_DIR'] = '/Applications/Xcode.app/Contents/Developer'
plan = {'argv': cmd, 'cwd': str(cwd),
        'protocol_sha256': sha(root / 'matched-protocol.json'),
        'wrapper_sha256': sha(__file__), 'stage':stage, 'reader_sha256':reader_sha, 'calibration_report_sha256':sha(root/'matched-calibration-review.json') if prior else None,
        'build_manifest_sha256': sha(root / 'matched-build-v2/build-manifest.json'),
        'binary_sha256': sha(cmd[0]), 'probe_sha256': sha(base / 'full-build/probe.metallib'),
        'native_sha256': sha(base / 'deployed/myds.metallib'),
        'capture_json_sha256': sha(base / 'capture-verified.json'),
        'config_sha256': sha(base / 'deployed/myds.conf'),
        'MYDS_inherited': {k: v for k, v in env.items() if k.startswith('MYDS_')},
        'exclusive_recognized_process_check': True, 'timeout_s': 300,
        'fullmodel': False, 'adoption_cohort':stage=='formal', 'started_epoch': time.time()}
(out / 'execution-plan.json').write_text(json.dumps(plan, indent=2, sort_keys=True) + '\n')
monitor = []
proc = None
reason = None
started = time.monotonic()
try:
    with open(out / 'stdout.jsonl', 'xb') as sout, open(out / 'stderr.txt', 'xb') as serr:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=sout, stderr=serr)
        while proc.poll() is None:
            rows = processes()
            others = [r for r in rows if r['pid'] != proc.pid]
            monitor.append({'elapsed_s': time.monotonic() - started, 'recognized': rows})
            if others:
                reason = 'competing recognized inference process'
                proc.terminate()
                break
            if time.monotonic() - started > 300:
                reason = 'timeout'
                proc.terminate()
                break
            time.sleep(.25)
        try:
            rc = proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            rc = proc.wait()
finally:
    after = identities()
    final = processes()
    (out / 'monitor.json').write_text(json.dumps(monitor, indent=2) + '\n')
    result = {'stage':stage, 'exit_code': proc.returncode if proc else None, 'child_pid': proc.pid if proc else None,
              'abort_reason': reason, 'elapsed_s': time.monotonic() - started,
              'canonical_before': before, 'canonical_after': after,
              'canonical_unchanged': before == after,
              'recognized_before': initial, 'recognized_after': final,
              'protocol_sha256': plan['protocol_sha256'], 'finished_epoch': time.time()}
    (out / 'window.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result), flush=True)
if reason or before != after:
    sys.exit(2)
sys.exit(rc)
