#!/usr/bin/env python3
"""Authorized one-machine measurement window, with unconditional fleet restoration."""
import datetime, hashlib, json, os, signal, subprocess, sys, time, urllib.request
from pathlib import Path

ROOT = Path('/tmp/myds-opt-20261005-run')
FLEET = Path('~/Projects/myfleet')
MODEL = '~/.lmstudio/models/antirez/deepseek-v4.1-flash-gguf/DeepSeek-V4.1-Flash-Q2.gguf'
OUT = ROOT / 'results' / ('exclusive-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
OUT.mkdir(parents=True)
REPLAY = OUT / 'replay'
REPLAY.mkdir()
events = OUT / 'window.jsonl'
child = None
stopped = False

def emit(kind, **values):
    row = dict(kind=kind, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **values)
    with events.open('a') as f: f.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)

def run(args):
    return subprocess.run(args, text=True, capture_output=True, check=True).stdout.strip()

def alive(pid):
    try: os.kill(pid, 0); return True
    except ProcessLookupError: return False

def finish_child():
    global child
    if child and child.poll() is None:
        os.killpg(child.pid, signal.SIGINT)
        try: child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL); child.wait(timeout=5)
    child = None

def measure(name, profile, timeout, extra_env=None, binary=None, request="/tmp/128.json"):
    global child
    env = dict(os.environ)
    for key in list(env):
        if key.startswith('MYDS_'): del env[key]
    env.update(MYDS_PREFIX_CACHE='0', MYDS_KV_DSK='0', MYDS_PREFILL_CHUNK_SIZE='512',
               MYDS_MOE_CHUNKS='9', MYDS_SUFFIX_SHRINK_SHIFT='512',
               MYDS_RESIDENCY_KEEP_ALIVE_S='86400', MYDS_PHASE_PROFILE=str(int(profile)))
    if profile: env['MYDS_PHASE_PROFILE_FILE'] = str(OUT / 'profile.jsonl')
    if extra_env: env.update(extra_env)
    exe = ROOT / (binary or ('profiled' if profile else 'unprofiled')) / 'bin/bench_phase'
    if profile and name!='profile': env['MYDS_PHASE_PROFILE_FILE']=str(OUT/(name+'-profile.jsonl'))
    args = [str(exe), MODEL, request, '3', '64', '262144']
    start = time.monotonic()
    emit('measurement_start', name=name, argv=args, environment={k:v for k,v in env.items() if k.startswith('MYDS_')})
    with (OUT / (name + '-metrics.jsonl')).open('w') as stdout, (OUT / (name + '.log')).open('w') as stderr:
        child = subprocess.Popen(args, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
        try: rc = child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            finish_child(); emit('measurement_timeout', name=name, elapsed_s=time.monotonic()-start); raise
        child = None
    emit('measurement_end', name=name, returncode=rc, elapsed_s=time.monotonic()-start)
    if rc: raise RuntimeError(name + ' exited ' + str(rc))

def component(name,args,cwd,timeout=60,extra_env=None):
    global child
    env=dict(os.environ)
    for key in list(env):
        if key.startswith('MYDS_'): del env[key]
    if extra_env: env.update(extra_env)
    start=time.monotonic()
    emit('component_start',name=name,argv=args,cwd=str(cwd),environment=extra_env or {})
    with (OUT/(name+'.jsonl')).open('w') as out,(OUT/(name+'.log')).open('w') as err:
        child=subprocess.Popen(args,cwd=cwd,env=env,stdout=out,stderr=err,start_new_session=True)
        try: rc=child.wait(timeout=timeout)
        except subprocess.TimeoutExpired: finish_child(); raise
        child=None
    emit('component_end',name=name,returncode=rc,elapsed_s=time.monotonic()-start)
    if rc: raise RuntimeError(name+' failed '+str(rc))

def signal_handler(signum, frame):
    raise RuntimeError('window interrupted by signal ' + str(signum))

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)
emit('window_created', output=str(OUT))
try:
    pid = int(run(['lsof','-nP','-t','-a','-iTCP:8181','-sTCP:LISTEN']))
    command = run(['ps','-p',str(pid),'-o','command='])
    if '/myds/bin/myds ' not in command or '--ctx-size 262144' not in command: raise RuntimeError('production command drift')
    if subprocess.run(['pgrep','-x','bench_phase'],capture_output=True).returncode == 0: raise RuntimeError('another measurement is running')
    enabled=[]
    for line in (FLEET/'fleet.conf').read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'): continue
        columns=[c.strip() for c in line.split('|')]
        if columns[0] in ('1','2','3'): enabled.append(columns[:4])
    if enabled != [['3','15','local-myds-deepseek-v4.1-flash-q2','8181']]: raise RuntimeError('fleet configuration drift')
    expected={'myds':'64e67230abd8287396b810522c3eef40eb929baa4e407eb9220d67fe5b196b9e',
              'myds.metallib':'b19fccf2dedb577b0bbf2d2167015b748a29dc30408001ede867b8d2d5c08170'}
    for file,digest in expected.items():
        if hashlib.sha256((FLEET/'../myds/bin'/file).read_bytes()).hexdigest()!=digest: raise RuntimeError('production artifact drift')
    connections = subprocess.run(['lsof','-nP','-a','-p',str(pid),'-iTCP:8181','-sTCP:ESTABLISHED'],capture_output=True,text=True)
    if connections.returncode == 0: raise RuntimeError('active production connections; no requests interrupted')
    emit('preflight_passed', pid=pid, command=command, enabled=enabled, hashes=expected)
    stopped = True
    pause_start = time.monotonic()
    emit('production_stop', pid=pid)
    os.kill(pid, signal.SIGTERM)
    deadline=time.monotonic()+15
    while alive(pid) and time.monotonic()<deadline: time.sleep(.1)
    if alive(pid): os.kill(pid,signal.SIGKILL)
    time.sleep(1)
    old_replay=ROOT/'results/exclusive-20261005T014346Z/replay'
    measure('original-long',False,180,binary='/tmp/myds-phase-20261005-run/unprofiled',request=str(ROOT/'long.json'))
    measure('exact-long',False,180,{'MYDS_PREFILL_ATTN':'vec','MYDS_MOE_IQ2_TAIL_CULL':'1','MYDS_QKV_NR0':'4'},binary='combined',request=str(ROOT/'long.json'))
except BaseException as exc:
    emit('window_error', error=repr(exc))
finally:
    finish_child()
    if stopped:
        emit('restoration_start', entry=str(FLEET/'start_fleet.sh'))
        # A clean environment prevents any measurement-only flags leaking into production.
        env=dict(os.environ)
        for key in list(env):
            if key.startswith('MYDS_'): del env[key]
        with (OUT/'restore.log').open('w') as log:
            restore=subprocess.Popen(['/bin/zsh',str(FLEET/'start_fleet.sh')],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
            ready=False
            deadline=time.monotonic()+135
            while time.monotonic()<deadline:
                try:
                    with urllib.request.urlopen('http://127.0.0.1:8181/health',timeout=1) as response:
                        if json.loads(response.read()).get('status')=='ok': ready=True; break
                except Exception: pass
                time.sleep(.5)
            emit('restoration_health', ready=ready, downtime_s=time.monotonic()-pause_start,
                 restore_process_pid=restore.pid, restore_returncode=restore.poll())
    emit('window_complete', output=str(OUT))
