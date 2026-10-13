#!/usr/bin/env python3
"""Run a prepared experiment plan, preserving the exact running myds service."""
import argparse, fcntl, hashlib, json, os, pathlib, re, shlex, signal
import subprocess, sys, time, urllib.request

def command(argv):
    return subprocess.check_output(argv, text=True).strip()

def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()

def write(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')

def capture(pid):
    argv = shlex.split(command(['ps', '-p', str(pid), '-o', 'command=']))
    raw = command(['ps', 'eww', '-p', str(pid), '-o', 'command='])
    env = dict(re.findall(r'(?:^|\s)(MYDS_[A-Za-z0-9_]+)=([^\s]*)', raw))
    detail = command(['lsof', '-a', '-p', str(pid), '-d', 'cwd,1,2', '-Fn'])
    names = re.findall(r'^n(.+)$', detail, re.M)
    if len(names) != 3 or names[1] != names[2]:
        raise RuntimeError('Cannot bind service cwd and common stdout/stderr log')
    binary = pathlib.Path(argv[0]).resolve()
    library = binary.parent / 'myds.metallib'
    return dict(pid=pid, argv=argv, myds_env=env, cwd=names[0], log=names[1],
                binary=str(binary), binary_sha256=digest(binary),
                metallib=str(library), metallib_sha256=digest(library))

def listener():
    p = subprocess.run(['lsof', '-nP', '-t', '-a', '-iTCP:8181', '-sTCP:LISTEN'],
                       capture_output=True, text=True)
    values = set(p.stdout.split())
    if len(values) != 1:
        raise RuntimeError('Expected exactly one original 8181 listener')
    return int(values.pop())

def health():
    with urllib.request.urlopen('http://127.0.0.1:8181/health', timeout=3) as r:
        return json.load(r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('plan', type=pathlib.Path)
    ap.add_argument('output', type=pathlib.Path)
    args = ap.parse_args()
    plan = json.loads(args.plan.read_text())
    if not plan or not isinstance(plan, list):
        raise ValueError('Nonempty array of prepared steps required')
    args.output.mkdir(parents=True, exist_ok=False)
    lock = open('/tmp/myds-codex-decode-window.lock', 'w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    before = capture(listener())
    if health().get('status') != 'ok':
        raise RuntimeError('Original service unhealthy')
    active = subprocess.run(['lsof','-nP','-a','-iTCP:8181','-sTCP:ESTABLISHED'],
                            capture_output=True, text=True)
    if active.stdout.strip():
        raise RuntimeError('Original service has established connections')
    # Bind prepared native programs/libraries before interrupting the service.
    inputs = {}
    for step in plan:
        if not isinstance(step.get('argv'), list) or not step['argv']:
            raise ValueError('Step requires an argv array')
        for p in step.get('identities', []) + [step['argv'][0]]:
            resolved = str(pathlib.Path(p).resolve())
            inputs[resolved] = digest(resolved)
    write(args.output / 'before.json', before)
    write(args.output / 'plan.json', plan)
    write(args.output / 'inputs.json', inputs)
    original_env = {k:v for k,v in os.environ.items() if not k.startswith('MYDS_')}
    original_env.update(before['myds_env'])
    original_env['PATH'] = '/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin'
    original_env['DEVELOPER_DIR'] = '/Applications/Xcode.app/Contents/Developer'
    stopped = False
    outcomes = []
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        stopped = True
        os.kill(before['pid'], signal.SIGTERM)
        for _ in range(40):
            p = subprocess.run(['ps','-p',str(before['pid']),'-o','pid='], capture_output=True)
            if not p.stdout.strip(): break
            time.sleep(1)
        else: raise RuntimeError('Original service did not terminate')
        from_path = plan[0].get('source_tools')
        if from_path:
            sys.path.insert(0, from_path)
            import bench_reproduce
            bench_reproduce.check_idle(8181)
        print('Exclusive inference window acquired', flush=True)
        for step in plan:
            for path, sha in inputs.items():
                if digest(path) != sha: raise RuntimeError('Prepared input changed: ' + path)
            env = dict(original_env)
            env.update(step.get('env', {}))
            for key in step.get('unset', []): env.pop(key, None)
            start = time.time()
            print('Running ' + step['name'], flush=True)
            with open(args.output / (step['name'] + '.stdout'), 'wb') as out, \
                 open(args.output / (step['name'] + '.stderr'), 'wb') as err:
                child = subprocess.Popen(step['argv'], cwd=step.get('cwd', before['cwd']), env=env,
                                         stdout=out, stderr=err, start_new_session=True)
                try:
                    rc = child.wait(timeout=step.get('timeout',900))
                finally:
                    if child.poll() is None:
                        os.killpg(child.pid, signal.SIGTERM)
                        try: child.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait()
            outcomes.append(dict(name=step['name'], returncode=rc, seconds=time.time()-start))
            write(args.output / 'steps.json', outcomes)
            if rc: raise RuntimeError('Experiment failed: ' + step['name'])
    finally:
        if stopped:
            if digest(before['binary']) != before['binary_sha256'] or digest(before['metallib']) != before['metallib_sha256']:
                raise RuntimeError('Original production artifacts changed; cannot certify restore')
            live = subprocess.run(['ps','-p',str(before['pid']),'-o','pid='], capture_output=True)
            if not live.stdout.strip():
                with open(before['log'], 'ab') as log:
                    proc = subprocess.Popen(before['argv'], cwd=before['cwd'], env=original_env,
                                            stdout=log, stderr=log, start_new_session=True)
                restore_pid = proc.pid
            else: restore_pid = before['pid']
            restored = None
            for _ in range(90):
                try:
                    h = health()
                    if h.get('status') == 'ok':
                        restored = capture(listener()); restored['health'] = h; break
                except Exception: pass
                time.sleep(2)
            if restored is None:
                write(args.output / 'restored.json', dict(healthy=False, started_pid=restore_pid))
                raise RuntimeError('Service restore did not become healthy')
            for key in ('argv','myds_env','cwd','log','binary_sha256','metallib_sha256'):
                if restored[key] != before[key]: raise RuntimeError('Restored identity differs: ' + key)
            write(args.output / 'restored.json', restored)
            print('Original argv, MYDS environment, artifacts and health restored', flush=True)
    return 0

if __name__ == '__main__':
    sys.exit(main())
