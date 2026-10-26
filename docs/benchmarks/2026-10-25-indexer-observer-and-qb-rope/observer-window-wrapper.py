import pathlib, json, os, subprocess, fcntl, time, hashlib, sys

root=pathlib.Path('/private/tmp/myds-indexer-r3-20261025')
base=pathlib.Path('/private/tmp/myds-indexer-r2-20261025')
out=root/'observer-window';out.mkdir(exist_ok=False)
lock=open('/tmp/myds-codex-decode-window.lock','a+')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)

def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()

canonical=[pathlib.Path(p) for p in (
    '/Users/fangqingyuan/Projects/myds/bin/myds',
    '/Users/fangqingyuan/Projects/myds/bin/myds.metallib',
    '/Users/fangqingyuan/Projects/myfleet/myds.conf')]

def identities():
    return {str(p):sha(p) for p in canonical}

def jobs():
    result=[]
    for line in subprocess.run(['ps','-axo','pid=,comm='],capture_output=True,text=True,check=True).stdout.splitlines():
        parts=line.strip().split(None,1)
        if len(parts)!=2:continue
        name=pathlib.Path(parts[1]).name.lower()
        if name in {'myds','myglm','ds4','llama-server','ollama','capture_indexer_k'} or name.startswith('bench_'):
            result.append(dict(pid=int(parts[0]),comm=parts[1]))
    return result

protocol=json.loads((root/'observer-protocol.json').read_text())
assert protocol['schedule']==['real','sham','sham','real'] and protocol['adoption_cohort'] is False
before=identities();baseline=json.loads((base/'baseline-identity.json').read_text())
assert all(value==baseline[path]['sha256'] for path,value in before.items()),'canonical identity drift'
initial=jobs();assert not initial,initial
env={k:v for k,v in os.environ.items() if not k.startswith('MYDS_')}
env['DEVELOPER_DIR']='/Applications/Xcode.app/Contents/Developer'
results=[];protocol_sha=sha(root/'observer-protocol.json')
for ordinal,mode in enumerate(protocol['schedule']):
    run=out/('run-%u-%s'%(ordinal,mode));run.mkdir(exist_ok=False)
    cwd=root/('observer-cwd-%u'%ordinal);cwd.mkdir(exist_ok=False)
    binary=root/'observer-build/bench_indexer_observer_probe'
    cmd=[str(binary),'--capture-json',str(base/'capture-verified.json'),
         '--metallib',str(base/'full-build/probe.metallib'),
         '--native-metallib',str(base/'deployed/myds.metallib'),
         '--config',str(base/'deployed/myds.conf'),'--out-dir',str(run/'outputs'),'--samples','9']
    if mode=='sham':cmd.append('--sham-tiled-b')
    run_before=identities();run_initial=jobs();assert not run_initial,run_initial
    plan=dict(argv=cmd,cwd=str(cwd),run_index=ordinal,mode=mode,protocol_sha256=protocol_sha,
              wrapper_sha256=sha(__file__),build_manifest_sha256=sha(root/'observer-build/build-manifest.json'),
              binary_sha256=sha(binary),probe_sha256=sha(base/'full-build/probe.metallib'),
              native_sha256=sha(base/'deployed/myds.metallib'),capture_json_sha256=sha(base/'capture-verified.json'),
              config_sha256=sha(base/'deployed/myds.conf'),MYDS_inherited={},
              exclusive_recognized_process_check=True,timeout_s=300,fullmodel=False,adoption_cohort=False,
              started_epoch=time.time())
    (run/'execution-plan.json').write_text(json.dumps(plan,indent=2,sort_keys=True)+'\n')
    monitoring=[];reason=None;started=time.monotonic()
    with (run/'stdout.jsonl').open('xb') as stdout,(run/'stderr.txt').open('xb') as stderr:
        process=subprocess.Popen(cmd,cwd=cwd,env=env,stdout=stdout,stderr=stderr)
        while process.poll() is None:
            current=jobs();monitoring.append(dict(elapsed_s=time.monotonic()-started,recognized=current))
            if any(p['pid']!=process.pid for p in current):reason='recognized inference competitor';process.terminate();break
            if time.monotonic()-started>300:reason='timeout';process.terminate();break
            time.sleep(.25)
        try:rc=process.wait(timeout=10)
        except subprocess.TimeoutExpired:process.kill();rc=process.wait()
    after=identities()
    result=dict(exit_code=rc,child_pid=process.pid,abort_reason=reason,elapsed_s=time.monotonic()-started,
                canonical_before=run_before,canonical_after=after,canonical_unchanged=run_before==after,
                recognized_before=run_initial,recognized_after=jobs(),protocol_sha256=protocol_sha,
                run_index=ordinal,mode=mode,finished_epoch=time.time())
    (run/'monitor.json').write_text(json.dumps(monitoring,indent=2)+'\n')
    (run/'window.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    results.append(result);print(json.dumps(result),flush=True)
    if rc or reason or after!=before:break
final=dict(protocol_sha256=protocol_sha,schedule=protocol['schedule'],runs=results,
           all_four_runs_completed=len(results)==4 and all(r['exit_code']==0 for r in results),
           canonical_before=before,canonical_after=identities(),recognized_after=jobs(),
           fullmodel_eligible=False,production_adoption=False)
(out/'window.json').write_text(json.dumps(final,indent=2,sort_keys=True)+'\n')
sys.exit(0 if final['all_four_runs_completed'] and final['canonical_before']==final['canonical_after'] else 1)
