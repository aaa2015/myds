import os,json,hashlib,pathlib,subprocess,time,fcntl
R=pathlib.Path('/private/tmp/myds-kv-only-finish-M2-screen-20261026')
P=pathlib.Path('/private/tmp/myds-kv-only-finish-M2-window-protocol-20261026.json')
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
lock=open('/tmp/myds-codex-decode-window.lock','a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
p=json.loads(P.read_text());assert not R.exists();assert sha(__file__)==p['wrapper_sha256']
prior=json.loads(pathlib.Path(p['M4_prerequisite_report']).read_text());assert prior['correctness_passed'] is True
for d in p['frozen_files']:
 b=pathlib.Path(d['path']).read_bytes();assert len(b)==d['bytes'] and hashlib.sha256(b).hexdigest()==d['sha256'],d['path']
R.mkdir();env={k:v for k,v in os.environ.items() if not k.startswith('MYDS_')}
plan={'kind':'M2_correctness_only_private_KV_finish_window','argv':p['argv'],'protocol_sha256':sha(P),'wrapper_sha256':sha(__file__),'frozen_files':p['frozen_files'],'MYDS_env':{},'timeout_s':180,'started_epoch':time.time(),'timing_requested':False,'expected_device':'Apple M2 Ultra','production_mutation':False}
write(R/'execution-plan.json',plan);start=time.monotonic();error=None;rc=None
try:
 with (R/'stdout.jsonl').open('xb') as out,(R/'stderr.txt').open('xb') as err:
  q=subprocess.run(p['argv'],env=env,stdout=out,stderr=err,timeout=180);rc=q.returncode
except Exception as e:error=repr(e)
changed=[]
for d in p['frozen_files']:
 q=pathlib.Path(d['path'])
 if not q.is_file() or q.stat().st_size!=d['bytes'] or sha(q)!=d['sha256']:changed.append(d['path'])
mf=R/'outputs/artifact-manifest.json';md=json.loads(mf.read_text()) if mf.is_file() else {}
manifest_gate=(md.get('failure') is None and md.get('status')=='synthetic_correctness_complete_not_adoption' and md.get('all_complete_outputs_and_guards_pass') is True)
if sha(P)!=plan['protocol_sha256']:changed.append(str(P))
result={'manifest_gate':manifest_gate,'manifest_failure':md.get('failure'),'manifest_status':md.get('status'),'kind':plan['kind'],'exit_code':rc,'error':error,'elapsed_s':time.monotonic()-start,'finished_epoch':time.time(),'protocol_sha256':sha(P),'wrapper_sha256':sha(__file__),'frozen_files_unchanged':not changed,'changed_files':changed,'timing_requested':False,'GPU_correctness_passed':None,'independent_reader_required':True,'performance_acceptance':False,'production_mutation':False}
write(R/'window.json',result);print(json.dumps(result));raise SystemExit(0 if rc==0 and not error and not changed and manifest_gate else 1)
