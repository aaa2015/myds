#!/usr/bin/env python3
"""Conditional exact-artifact promptclear-v4 deployment prototype; root review required.

Nothing is deployed by --prepare or --self-test. --run is a future explicit
activation command, not used while preparing this tool. Every gate/report/archive
must exist and pass before stop/rename/config mutation. No build, source merge,
git update, shell evaluation, or remote access is implemented.

The immutable release is separate from canonical dirty sources. The original
binary/config INODES are renamed to unique same-directory backups, retained on
success and renamed back on caught failure. c692 metallib is never replaced.
SIGINT/TERM trigger rollback; SIGKILL/power loss cannot run Python finally: retain
journal and private environment backup for operator recovery. Non-MYDS values
never enter public receipts, release archives, or stdout.

Manifest schema is printed by --print-schema. Paths are absolute, references are
exact {path,sha256} or {path,sha256,bytes}. The final independent full/probe reports
must bind their original input-map SHA, not a reserialized protocol copy.
"""
import argparse
import ctypes
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import struct
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.request
import uuid
sys.dont_write_bytecode = True

PLAN_SHA = 'a83127b1427b09b703883150f3ec6e8e90e7ecdead3f48b798c7755368bd9d9c'
BASE_SHA = '29058d822c95d86ffdc6e84798fe87093d7f8c094b1002f1250177b12066d72b'
BIN_SHA = '9cf2a670a22632874829398eece031111e438b4ca803860a2e6bcbcc7e94c57a'
LIB_SHA = 'c692d821aed43ecce1cd9003f11e19083cbe5ce2923bef44a8df23c1182dfc81'
BASE_COMMIT = '868d94e9e3168aeadac478e75d096d992b6bc92e'
CORE = {
 'src/myds_v41.c':'554387ef6d41612b4e7dc537cf759105bcb7fcaf103d74a6f01ebdb6bf6cc15f',
 'src/myds_metal.m':'28344f08a27c6c87b51676afab73a9daaaa0a653d8dcd0e2d7c71d094e27f0b7',
 'include/myds_internal.h':'9869b7692fa8a7fbb2f7dc02a74768423668651005238574d2e33cb65066cd65',
 'include/myds_metal.h':'3926c8a7eded05a90abccaf4dd8792c8b45cd744f7536f91eacc538b029821d5'}
FLAGS = {'MYDS_PREFILL_MOE_WORKSPACE':'1','MYDS_PREFILL_PROMPT_VIEW_CLEAR':'1'}
SAFETY_SHA = 'f093a967ea76487168863461b3f5273ebee5faa2dca6527ec58376a5927d69bb'
READER_SHA = 'e2baa03cf9ae488be73200d406dcc1447dcc0d8effeaa871bdac934caaf42c3a'
NATIVE = 'bin/bench_moe_workspace_fullmodel_promptclear'
ROLES = {'component':'component_gate_pass','recovery':'recovery_gate_pass',
         'causal':'causal_lifetime_gate_pass','probe':'growth_gate_pass','full':'performance_gate_pass'}
IDENTITY_KEYS = ('argv_raw_hex','myds_env_raw_hex','cwd','log','nice','binary','binary_sha256',
                 'metallib','metallib_sha256','config','config_sha256','effective_myds_env','library_precedence')


def need(value, message):
    if not value: raise ValueError(message)


def sha(path, count=None):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        remaining=count
        while remaining is None or remaining:
            block=f.read(1048576 if remaining is None else min(1048576,remaining))
            if not block: break
            h.update(block)
            if remaining is not None: remaining-=len(block)
    need(remaining in (None,0),'Truncated prefix: '+str(path))
    return h.hexdigest()


def load(path):
    def pairs(items):
        out={}
        for k,v in items:
            need(k not in out,'Duplicate JSON key: '+k);out[k]=v
        return out
    return json.loads(Path(path).read_text(),object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(path, value, mode=0o600):
    path=Path(path);temporary=path.with_name(path.name+'.new')
    fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL,mode)
    try:
        with os.fdopen(fd,'w')as f:
            json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
        os.replace(temporary,path);fsync_dir(path.parent)
    finally:
        if temporary.exists():temporary.unlink()


def fsync_dir(path):
    fd=os.open(path,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def reference(record):
    need(type(record)is dict and set(record)in ({'path','sha256'},{'path','sha256','bytes'}),'Exact path/SHA reference required')
    path=Path(record['path']);need(path.is_absolute() and path.is_file() and not path.is_symlink(),'Absolute regular original reference required')
    need(type(record['sha256'])is str and re.fullmatch('[0-9a-f]{64}',record['sha256']) and sha(path)==record['sha256'],'Reference SHA differs: '+str(path))
    if 'bytes'in record:need(type(record['bytes'])is int and record['bytes']>=0 and path.stat().st_size==record['bytes'],'Reference extent differs')
    return path.resolve(strict=True)


def descriptor(path):return dict(path=str(Path(path).resolve()),sha256=sha(path),bytes=Path(path).stat().st_size)


def module(path, name, digest):
    need(sha(path)==digest,'Reviewed helper bytes differ: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def parse_config(raw):
    out={}
    for line in raw.decode('utf-8').splitlines():
        if not line.strip()or line.lstrip().startswith('#'):continue
        need(len(line.encode())<4095 and '='in line,'Unsupported configuration line')
        k,v=line.split('=',1);k=k.strip(' \t');v=v.lstrip(' \t');m=re.search(r'(?<=[ \t])#',v)
        if m:v=v[:m.start()]
        v=v.rstrip(' \t\r\n');need(k.startswith('MYDS_')and '\0'not in k+v,'Only plain MYDS configuration accepted');out.setdefault(k,v)
    return out


def candidate_config(raw, inherited, expected):
    current=parse_config(raw);effective=dict(current,**inherited)
    need(len(effective)==51 and effective==expected,'Original effective51 configuration/environment required')
    need(not FLAGS.keys() & (current.keys()|inherited.keys()),'Candidate flags must be absent from original config/inherited env')
    result=raw+(b''if not raw or raw.endswith(b'\n')else b'\n')+b'\n# Exact promptclear-v4 conditional release\n'+b''.join((k+'=1\n').encode()for k in FLAGS)
    actual=dict(parse_config(result),**inherited)
    need(len(actual)==53 and actual==dict(expected,**FLAGS),'Only two explicit candidate additions allowed')
    return result


def identity_core(identity):
    need(type(identity)is dict and all(identity.get(k)==v for k,v in CORE.items())and identity.get('bin/myds.metallib')==LIB_SHA,'Exact v4 core4/c692 identity required')


def prerequisite_map(value):
    if type(value)is list:return {r['key']:dict(path=r['path'],sha256=r['sha256'])for r in value}
    need(type(value)is dict,'Exact prerequisite mapping required');return value


def recovery_coverage(value):
    """A completed two-window correctness matrix, never one fabricated28-run window."""
    windows=value.get('source_windows');need(type(windows)is list and len(windows)==2,'Explicit two recovery source windows required')
    need(value.get('native_invocations')==28 and value.get('pair_readers_replayed')==14,'Complete10+18 native/14pair independent recovery coverage')
    need(len({w.get('window')for w in windows})==2 and all(Path(w.get('window','')).is_absolute()for w in windows),'Two distinct absolute original windows')
    need([(w.get('native_invocations'),w.get('pair_readers_replayed'))for w in windows]==[(10,5),(18,9)],'Exact original10/replayed5 plus continuation18/replayed9')
    need(windows[0].get('status')=='failed'and windows[1].get('status')=='continuation_passed_review_required','Original failed status and explicit continuation status retained')
    for w in windows:
        need(re.fullmatch('[0-9a-f]{64}',w.get('input_map_sha256',''))and w.get('restoration',{}).get('restored')is True and w['restoration'].get('exact_argv_myds_artifacts_config')is True,'Each original window has an exact input-map and finally restoration')
        need(all(type(w.get(k))is dict for k in ('source_archive','artifacts_archive','archive_bindings')),'Each window archived source/artifact/mapping identity required')
    scenarios=('deterministic','async','allocation','allocation-owner','shared-pso','prompt-allocation-first','prompt-allocation-last')
    expected=[(s,t,n)for s in scenarios for t in ('0','0.7')for n in (0,1)]
    arms=value.get('arm_executions');pairs=value.get('pair_reports')
    need(type(arms)is list and type(pairs)is list and len(arms)==28 and len(pairs)==14,'Complete ordered native/pair evidence records')
    need([(r.get('scenario'),str(r.get('temperature')),r.get('setting'))for r in arms]==expected,'Exact frozen28-arm unique original order')
    need([(r.get('scenario'),str(r.get('temperature')))for r in pairs]==[(s,t)for s in scenarios for t in ('0','0.7')],'Exact14 pair original order')
    need(all(r.get('evidence_valid')is True and r.get('correctness_pass')is True for r in pairs),'All14 independent pair decisions pass')
    return windows


def probe_binding_gate(independent,controller,probe_input_sha,index,proof_sha,proof_bytes):
    need(independent.get('archive_verification',{}).get('input_map_sha256')==probe_input_sha and independent['gate']==controller['gate'],'Probe independent replay binds original input-map/gate')
    need(type(independent.get('raw_comparisons'))is list and independent['raw_comparisons']==controller.get('raw_comparisons'),'Independent probe full raw comparisons match original controller')
    need(any(v['archive']=='source.tar.gz'and v['sha256']==proof_sha and v['bytes']==proof_bytes for v in index.values()),'Complete independent probe-long bytes frozen in final source archive; NEW input copy allowed')


def gate_set(values, refs, identity, env, model, header, input_map_sha):
    """Pure fail-closed checks; test fixtures cannot bypass the production callers."""
    for role,key in ROLES.items():
        v=values[role];need(v.get('evidence_valid')is True and v.get(key)is True and v.get('implementation_version')==4 and v.get('plan_sha256')==PLAN_SHA,'Independent '+role+' gate must pass under implementation4/current plan')
        need(v.get('production_adoption')is False and 'error'not in v,'Unadopted complete independent report required')
    c=values['component'];need(c.get('candidate_core_sha256')==CORE and c.get('metallib_sha256')==LIB_SHA,'Component exact normal core/library')
    need(c.get('model_backed_full_weight_intervals_verified_by_long_reader')is True,'Component real full weight intervals must be independently verified')
    for role in ('component','causal'):
        v=values[role];need(v.get('production_environment')==env and v.get('model')==model and v.get('model_header_sha256')==header,'Exact '+role+' environment/model')
    r=values['recovery'];need(r.get('recovery_plan_sha256')==PLAN_SHA and r.get('component_review_sha256')==refs['component']['sha256']and r.get('independent_fullraw_replay_completed')is True,'Exact independently replayed28-native recovery prerequisite')
    need(r.get('independent_review',{}).get('full_bytes',{}).get('pair_readers_replayed')==14,'All14 real recovery pair readers required')
    need(values['causal'].get('mode')=='causal_diagnostic'and values['causal'].get('correctness_pass')is True and values['causal'].get('native_identity')=={'candidate':identity['candidate']},'Exact FULLv2 causal native identity')
    expected={ROLES[k]:dict(path=refs[k]['path'],sha256=refs[k]['sha256'])for k in ('component','recovery','causal')}
    for role,mode in (('probe','probe'),('full','full')):
        v=values[role];need(v.get('mode')==mode and v.get('correctness_pass')is True and v.get('recorded_restoration_exact')is True and v.get('native_identity')==identity and v.get('all91_exact_production_source_inputs')is True,'Complete independently replayed normal '+role+' proof')
        need(prerequisite_map(v.get('prerequisites'))==expected,'Normal '+role+' prerequisite SHA identity differs')
        need(v.get('gate',{}).get(ROLES[role])is True,'Independent embedded '+role+' gate must also pass')
    need(values['probe'].get('performance_measured')is False and values['full'].get('performance_measured')is True,'Probe/full measurement distinction')
    need(values['full'].get('archive_verification',{}).get('input_map_sha256')==input_map_sha,'Full independent raw review must bind exact archived full input map')


def validate_bundle(a):
    need(sha(a.manifest)==a.manifest_sha256,'Explicit manifest SHA differs')
    m=load(a.manifest)
    need(set(m)=={'schema_version','implementation_version','candidate_stage','full_window','plan','gates','probe_controller','production_reference','executed','functional_request','functional_expected'}and m['schema_version']==1 and m['implementation_version']==4,'Exact adoption manifest schema')
    stage=Path(m['candidate_stage']);window=Path(m['full_window'])
    need(stage.is_absolute()and window.is_absolute()and stage.is_dir()and window.is_dir(),'Absolute frozen FULLv2 stage/window required')
    stage=stage.resolve();window=window.resolve()
    plan_path=reference(m['plan']);need(m['plan']['sha256']==PLAN_SHA,'Frozen original plan SHA')
    plan=load(plan_path)
    need(set(m['gates'])==set(ROLES),'All five independent gates required')
    refs=m['gates'];values={k:load(reference(v))for k,v in refs.items()}
    reference(m['production_reference']);production=load(m['production_reference']['path'])
    need(production.get('all_checks_pass')is True and len(production.get('checks',{}))==41 and all(v is True for v in production['checks'].values())and production.get('production_adoption')is False,'Final independent41-check original production restoration required')
    before=load(window/'before.json');summary=load(window/'summary.json');restored=load(window/'restored.json')
    need(all(before[k]==restored[k]==production['live'][k]for k in IDENTITY_KEYS),'Last full window and final live reference must preserve exact original identity')
    need(production['live']['pid']==restored['pid']==summary['restoration']['pid'],'Final reference must observe the final full restored process')
    need(before['binary_sha256']==BASE_SHA and before['metallib_sha256']==LIB_SHA and before['nice']==0 and len(before['effective_myds_env'])==51,'Original production binary/library/nice0/51 settings')
    need(summary.get('status')=='full_passed_review_required'and summary.get('mode')=='full'and summary.get('plan_sha256')==PLAN_SHA and summary.get('correctness_pass')is True and summary.get('performance_gate_pass')is True and summary.get('normal_sequence_native_complete')is True and summary.get('correctness_generations_completed')==18 and summary.get('performance_generations_completed')==36 and summary.get('total_generations_completed')==54,'Exactly one completed original18+36 full cohort must pass')
    events=[json.loads(x)for x in (window/'events.jsonl').read_text().splitlines()if x.strip()]
    need(events[-1].get('kind')=='sequence_decision_complete'and events[-1].get('status')==summary['status']and events[-1].get('evidence_valid')is True and events[-1].get('performance_gate_pass')is True,'Final controller event/summary agreement')
    need(set(m['executed'])=={'input_map','bindings','source_archive','artifacts_archive'},'Complete executed archive references')
    ep={k:reference(v)for k,v in m['executed'].items()};need(ep['input_map']==(window/'stage-inputs.json').resolve(),'Use original final-window input map')
    inputs=load(ep['input_map']);need(inputs.get(str(plan_path))==PLAN_SHA,'Final window original plan bound by bytes')
    safety_path=stage/'tools/run_iq2_signed4_window.py';reader_path=stage/'tools/review_moe_workspace_fullmodel_promptclear.py'
    need(inputs.get(str(safety_path))==SAFETY_SHA and inputs.get(str(reader_path))==READER_SHA,'Use reviewed actual frozen safety/archive reader')
    safety=module(safety_path,'promptclear_adoption_safety',SAFETY_SHA);reader=module(reader_path,'promptclear_adoption_reader',READER_SHA)
    archive=reader.executed_archives(window,ep['source_archive'],ep['artifacts_archive'],ep['bindings']);index=archive['index']
    identity=summary['native_identity'];identity_core(identity['candidate']);need(values['probe'].get('native_identity')==identity,'Probe and full exact native identity')
    for rel,digest in reader.GOLD_SOURCE_HASHES.items():
        expected=CORE.get(rel,digest);need(inputs.get(str(stage/rel))==expected and sha(stage/rel)==expected,'Full91 runtime source inputs must equal exact868+v4: '+rel)
    need(set(identity['candidate'])==set(CORE)|{NATIVE,'bin/myds.metallib'},'Exact normal full native identity field set')
    for rel,digest in {**identity['candidate'],**CORE,'bin/myds':BIN_SHA,'bin/myds.metallib':LIB_SHA}.items():
        need(inputs.get(str(stage/rel))==digest and sha(stage/rel)==digest,'Frozen candidate live bytes/archive binding differs: '+rel)
    symbols=subprocess.check_output(['/usr/bin/nm','-gU',str(stage/'bin/myds')],text=True,timeout=10)
    need(not any(s in symbols for s in ('myds_metal_moe_workspace_test_control','myds_v41_moe_workspace_probe_','myds_v41_moe_workspace_cancel_control','myds_v41_prompt_workspace_test_','myds_v41_prompt_workspace_probe_')),'Deployed normal binary must not expose TESTING/fault/probe API')
    need(all(index.get(str(stage/k),{}).get('sha256')==v for k,v in identity['candidate'].items()),'Candidate native identity fully archived')
    gate_set(values,refs,identity,before['effective_myds_env'],summary['model'],summary['model_header_sha256'],sha(ep['input_map']))
    expected_prereqs={ROLES[k]:dict(path=refs[k]['path'],sha256=refs[k]['sha256'])for k in ('component','recovery','causal')}
    need(summary['prerequisites']==expected_prereqs,'Full controller prerequisite identity')
    for role,field in (('component','long_report'),('recovery','long_report'),('causal','full_report')):
        rec=values[role].get(field);p=reference(rec);need(inputs.get(str(Path(rec['path'])))==sha(p),'Original independent long report must be frozen in final window')
        v=load(p);need(v.get('evidence_valid')is True and v.get(ROLES[role])is True,'Referenced independent long gate must also pass')
        if role=='recovery':
            identity_core(v.get('candidate_identity'))
            need(v.get('production_environment')==before['effective_myds_env']and v.get('model')==summary['model']and v.get('model_header_sha256')==summary['model_header_sha256'],'Recovery long actual core/environment/model identity')
            source_windows=recovery_coverage(values[role]);need(recovery_coverage(v)==source_windows,'Small/long two-window recovery coverage must be identical')
            for w in source_windows:
                archive_paths={k:reference(w[k])for k in ('source_archive','artifacts_archive','archive_bindings')};mapping=load(archive_paths['archive_bindings'])
                need(mapping.get('status')=='complete'and mapping.get('stage_inputs_sha256')==w['input_map_sha256'],'Actual recovery source-window archive/input-map binding')
                for key,name in (('source_archive','source.tar.gz'),('artifacts_archive','artifacts.tar.gz')):
                    need(mapping['archives'][name]['sha256']==w[key]['sha256']and mapping['archives'][name]['bytes']==archive_paths[key].stat().st_size,'Actual recovery source-window archive physical identity')
            for field in ('original_failed_top','original_top'):
                original=load(reference(values[role][field]));need(original.get('recovery_gate_pass')is False,'Preserve both incomplete original recovery tops, never relabel as a single complete window')
    pc=load(reference(m['probe_controller']));need(inputs.get(str(Path(m['probe_controller']['path'])))==m['probe_controller']['sha256'],'Actual original probe controller prerequisite must be frozen')
    need(pc.get('evidence_valid')is True and pc.get('growth_gate_pass')is True and pc.get('mode')=='probe'and pc.get('plan_sha256')==PLAN_SHA and pc.get('native_identity')==identity and pc.get('prerequisites')==expected_prereqs and pc.get('production_environment')==before['effective_myds_env']and pc.get('model')==summary['model']and pc.get('model_header_sha256')==summary['model_header_sha256'],'Probe controller actual full prerequisite identity')
    probe_window=Path(m['probe_controller']['path']).parent
    probe_binding_gate(values['probe'],pc,sha(probe_window/'stage-inputs.json'),index,refs['probe']['sha256'],Path(refs['probe']['path']).stat().st_size)
    need(values['full']['gate']==summary['performance'],'Independent full gate exactly matches the single original cohort')
    request_path=reference(m['functional_request']);need(request_path.stat().st_size<=1048576,'Bounded functional request JSON');request=load(request_path);expected=load(reference(m['functional_expected']))
    need(request.get('stream',False)is False and request.get('n',1)==1 and request.get('temperature')==0 and request.get('seed')==42 and type(request.get('max_tokens'))is int and 1<=request['max_tokens']<=32 and request.get('model')==safety.option(before['argv'],'--alias'),'Bound deterministic small production HTTP functional request')
    need(set(expected)=={'content_sha256','completion_tokens','finish_reason'}and re.fullmatch('[0-9a-f]{64}',expected['content_sha256'])and type(expected['completion_tokens'])is int and 1<=expected['completion_tokens']<=32 and expected['finish_reason']in ('stop','length'),'Explicit functional response oracle required')
    binary=Path(before['binary']['path']);canonical=binary.parent.parent
    need(a.port==8187 and binary==Path('$HOME/Projects/myds/bin/myds')and Path(before['config']['path'])==Path('$HOME/Projects/deploy/myds.conf')and Path(before['argv'][0]).resolve()==binary,'Single known8187 canonical binary/config service only')
    need(a.release_dir.is_absolute()and not a.release_dir.exists()and canonical not in a.release_dir.resolve().parents and a.release_dir.resolve()!=canonical and stage not in a.release_dir.resolve().parents,'Independent new release directory, outside canonical/frozen stage')
    need(not any(p.resolve()in a.release_dir.resolve().parents for p in (Path('/tmp'),Path('/var/tmp'))),'Release source authority must be durable, outside temporary directories')
    need(all(p not in a.out.parents and a.out!=p for p in (canonical,stage,window)),'New deployment evidence directory must be outside canonical/frozen source/window')
    model=Path(safety.option(before['argv'],'-m')).resolve(strict=True)
    need(safety.metadata(model)==summary['model']and sha(model,1048576)==summary['model_header_sha256'],'Actual production GGUF metadata/header')
    return dict(manifest=m,stage=stage,window=window,safety=safety,reader=reader,archive=archive,inputs=inputs,identity=identity,values=values,before=before,reference=production['live'],summary=summary,canonical=canonical,plan=plan,request=request,expected=expected,model=model)


def parse_environment_blob(data):
    """KERN_PROCARGS2 env ends at its first empty record, never trailing storage."""
    argc=struct.unpack_from('i',data)[0];need(0<argc<4096,'Valid original argc')
    cursor=data.index(b'\0',4)+1
    while cursor<len(data)and data[cursor]==0:cursor+=1
    for _ in range(argc):cursor=data.index(b'\0',cursor)+1
    result={}
    for raw in data[cursor:].split(b'\0'):
        if not raw:break
        need(b'='in raw,'Malformed original env');key,value=raw.split(b'=',1);need(key and key not in result and b'\0'not in key+value,'Duplicate/malformed original env');result[key]=value
    need(result,'Complete original inherited environment required');return result


def raw_environment(pid):
    """Initial complete process env, private only; no use of controller env."""
    lib=ctypes.CDLL('/usr/lib/libSystem.B.dylib',use_errno=True);lib.sysctl.argtypes=[ctypes.POINTER(ctypes.c_int),ctypes.c_uint,ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t),ctypes.c_void_p,ctypes.c_size_t];lib.sysctl.restype=ctypes.c_int
    mib=(ctypes.c_int*3)(1,49,pid);size=ctypes.c_size_t()
    if lib.sysctl(mib,3,None,ctypes.byref(size),None,0):raise OSError(ctypes.get_errno(),'KERN environment size')
    buf=ctypes.create_string_buffer(size.value)
    if lib.sysctl(mib,3,buf,ctypes.byref(size),None,0):raise OSError(ctypes.get_errno(),'KERN environment')
    return parse_environment_blob(buf.raw[:size.value])


def private_env_record(env):return {k.hex():v.hex()for k,v in env.items()}


def canonical_snapshot(path):
    env=dict(os.environ,GIT_OPTIONAL_LOCKS='0')
    run=lambda *args:subprocess.check_output(['/usr/bin/git','-C',str(path),*args],env=env)
    untracked=run('ls-files','--others','--exclude-standard','-z').split(b'\0');files={};total=0
    for raw in untracked:
        if not raw:continue
        p=path/os.fsdecode(raw);need(p.is_file()and not p.is_symlink(),'Unsupported untracked canonical object');total+=p.stat().st_size;need(total<=64*1024*1024,'Untracked canonical snapshot exceeds64MiB; require separate reviewed content archive');files[os.fsdecode(raw)]=sha(p)
    return dict(head=run('rev-parse','HEAD').decode().strip(),status_porcelain=run('status','--porcelain=v1','--untracked-files=all').decode().splitlines(),tracked_diff_sha256=hashlib.sha256(run('diff','--binary','HEAD')).hexdigest(),untracked_sha256=files)


def process_footprint(pid):
    lib=ctypes.CDLL('/usr/lib/libproc.dylib',use_errno=True);lib.proc_pid_rusage.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_void_p];lib.proc_pid_rusage.restype=ctypes.c_int
    buffer=ctypes.create_string_buffer(1024)
    if lib.proc_pid_rusage(pid,2,buffer):raise OSError(ctypes.get_errno(),'proc_pid_rusage')
    return struct.unpack_from('Q',buffer,72)[0]


def wired_bytes():
    raw=subprocess.check_output(['/usr/bin/vm_stat'],text=True,timeout=10);page=re.search(r'page size of (\d+) bytes',raw);wired=re.search(r'Pages wired down:\s*(\d+)\.',raw)
    need(page and wired,'Actual wired-memory parse');return int(page[1])*int(wired[1])


def save_release(bundle,directory,manifest_path):
    """Preserve complete executed source/artifact archive bytes; never env secrets."""
    directory.mkdir(mode=0o700,parents=False,exist_ok=False)
    m=bundle['manifest'];frozen=directory/'executed';frozen.mkdir(mode=0o700)
    for name,rec in m['executed'].items():
        target=frozen/{'input_map':'stage-inputs.json','bindings':'archive-bindings.json','source_archive':'source.tar.gz','artifacts_archive':'artifacts.tar.gz'}[name]
        shutil.copy2(reference(rec),target);need(sha(target)==rec['sha256'],'Release archive copy differs')
    artifacts=directory/'artifacts';artifacts.mkdir(mode=0o700)
    for name,digest in (('myds',BIN_SHA),('myds.metallib',LIB_SHA)):
        p=artifacts/name;shutil.copy2(bundle['stage']/'bin'/name,p);need(sha(p)==digest,'Exact release artifact copy');os.chmod(p,0o755 if name=='myds'else 0o644)
    originals=directory/'original-artifact-copies';originals.mkdir(mode=0o700)
    for key,name in (('binary','myds'),('metallib','myds.metallib'),('config','myds.conf')):
        source=Path(bundle['reference'][key]['path']);target=originals/name;shutil.copy2(source,target);need(sha(target)==bundle['reference'][key+'_sha256'],'Archived original artifact copy differs')
    shutil.copy2(manifest_path,directory/'adoption-input-manifest.json')
    records=directory/'gate-records';records.mkdir(mode=0o700)
    for role,rec in m['gates'].items():
        shutil.copy2(reference(rec),records/(role+'.json'))
        if role in ('component','recovery','causal'):
            field='full_report'if role=='causal'else'long_report';shutil.copy2(reference(bundle['values'][role][field]),records/(role+'-long.json'))
    write(directory/'release-identity.json',dict(schema_version=1,source_authority='immutable exact executed archives; canonical dirty HEAD is not this source version',baseline_commit=BASE_COMMIT,implementation_version=4,core4=CORE,bin_sha256=BIN_SHA,metallib_sha256=LIB_SHA,normal_native_identity=bundle['identity'],manifest_sha256=sha(manifest_path),production_adoption=False,non_myds_environment_included=False))
    files=[p for p in directory.rglob('*')if p.is_file()]
    write(directory/'release-files.json',[dict(path=str(p.relative_to(directory)),sha256=sha(p),bytes=p.stat().st_size)for p in sorted(files)])
    for p in [*files,directory/'release-files.json']:
        fd=os.open(p,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
        os.chmod(p,0o555 if p==artifacts/'myds'else 0o444)
    for d in sorted((p for p in directory.rglob('*')if p.is_dir()),key=lambda p:len(p.parts),reverse=True):fsync_dir(d);os.chmod(d,0o555)
    fsync_dir(directory);os.chmod(directory,0o555);fsync_dir(directory.parent);return artifacts/'myds'


class RealOps:
    def __init__(self,a,b):
        self.a=a;self.b=b;self.s=b['safety'];self.before=None;self.env=None;self.canonical=None;self.owned=set();self.children={};self.start_logs={};self.pending_candidate=None;self.active_pid=None;self.memory_records=[];self.done=threading.Event();self.fault=None;self.guard=threading.Lock();self.thread=None
    def scan(self):
        with self.guard:return self._scan()
    def _scan(self):
        rows=self.s.command(['/bin/ps','-axww','-o','pid=,ppid=,comm=']).splitlines();relevant=[]
        names={'myds','myglm','ds4','llama-server','llama-cli','llama-bench','ollama','mlx_lm','vllm','lms','trace_model'}
        for line in rows:
            p=line.split(None,2);need(len(p)==3,'Process scan parse');pid=int(p[0]);name=Path(p[2]).name.lower()
            if name in names or name.startswith(('bench_','bench-','test_moe_workspace','test_prompt_workspace','capture_moe_workspace','moe_prefill_replay')):
                need(pid in self.owned,'Competing/unknown model process: '+str(pid));relevant.append(pid)
        return relevant
    def preflight(self):
        need(sys.platform=='darwin'and os.getpriority(os.PRIO_PROCESS,os.getpid())==0,'macOS controller at nice0 required')
        value,raw=self.s.capture(self.a.port)
        need(value==self.b['reference'],'Live production changed after independent final41 audit')
        need(value['pid']>0 and value['nice']==0,'Known production listener/nice0')
        env=raw_environment(value['pid']);need({os.fsdecode(k):os.fsdecode(v)for k,v in env.items()if k.startswith(b'MYDS_')}==value['myds_env'],'Complete env MYDS subset differs from safety capture')
        self.before=value;self.env=env;self.original_config=raw;self.canonical=canonical_snapshot(self.b['canonical']);self.owned.add(value['pid']);self.active_pid=value['pid'];self.scan()
        vision=Path(self.s.option(value['argv'],'--vision')).resolve(strict=True);self.vision=dict(metadata=self.s.metadata(vision),header_sha256=sha(vision,1048576))
        need(not any(x in value['argv']for x in ('--pld','--no-pld')),'No production PLD CLI override')
        for key in ('binary','config'):
            p=Path(value[key]['path']);st=p.stat();need(not p.is_symlink()and p.is_file()and st.st_nlink==1 and st.st_uid==os.geteuid()and not(st.st_mode&0o6000),'Single same-user regular non-setuid original inode required')
            need(key!='binary'or os.access(p,os.X_OK),'Original binary must remain executable for exact restoration')
        return candidate_config(raw,value['myds_env'],value['effective_myds_env'])
    def check(self):
        need(self.fault is None,'Monitor failed: '+str(self.fault));self.scan()
    def begin_monitor(self):
        def loop():
            while not self.done.wait(.5):
                try:
                    self.scan();wired=wired_bytes();pid=self.active_pid;footprint=None
                    if pid is not None:
                        try:footprint=process_footprint(pid)
                        except OSError:
                            try:os.kill(pid,0)
                            except ProcessLookupError:pass
                            else:raise
                    limits=self.b['plan']['safety'];need(wired<limits['stop_on_system_wired_bytes'],'System wired safety ceiling')
                    need(footprint is None or footprint<limits['stop_on_process_phys_footprint_bytes'],'Process physical safety ceiling')
                    self.memory_records.append(dict(monotonic=time.monotonic(),pid=pid,process_phys_footprint_bytes=footprint,system_wired_bytes=wired))
                except BaseException as e:self.fault=type(e).__name__+': '+str(e);return
        self.thread=threading.Thread(target=loop,daemon=True);self.thread.start()
    def close_monitor(self):
        self.done.set()
        if self.thread:self.thread.join(timeout=15);need(not self.thread.is_alive(),'Monitor did not stop')
    def finish_monitor(self):
        self.close_monitor();self.check();write(self.a.out/'activation-memory-guard.json',dict(scope='Activation safety only; not performance or memory benefit proof',samples=self.memory_records))
    def stop(self,pid,original_precheck=True):
        child=self.children.get(pid)
        if child is not None and child.poll()is not None:
            self.owned.discard(pid);return
        try:os.kill(pid,0)
        except ProcessLookupError:
            self.owned.discard(pid);return
        if self.s.listener(self.a.port,False)not in (None,pid):raise ValueError('Unknown listener, refusing to stop it')
        args=self.s.kernel_arguments(pid)['argv_raw_hex'];need(args==self.before['argv_raw_hex'],'Stop PID argv identity changed')
        if pid==self.before['pid']and original_precheck:
            c=subprocess.run(['/usr/sbin/lsof','-nP','-a','-p',str(pid),f'-iTCP:{self.a.port}','-sTCP:ESTABLISHED'],capture_output=True,text=True,timeout=10)
            need(c.returncode in (0,1)and not c.stderr.strip()and not c.stdout.strip(),'Production has established connections')
            current,raw=self.s.capture(self.a.port);need(current==self.before and raw==self.original_config and raw_environment(pid)==self.env,'Production changed before stop')
        os.kill(pid,signal.SIGTERM)
        for deadline,kind in ((time.monotonic()+20,None),(time.monotonic()+30,signal.SIGKILL)):
            if kind:
                try:need(self.s.kernel_arguments(pid)['argv_raw_hex']==args,'PID reused before escalation');os.kill(pid,kind)
                except ProcessLookupError:break
            while time.monotonic()<deadline:
                if child is not None and child.poll()is not None:break
                try:os.kill(pid,0)
                except ProcessLookupError:break
                time.sleep(.1)
            else:continue
            break
        else:raise ValueError('Owned service did not exit')
        need(self.s.listener(self.a.port,False)is None,'Port remains occupied after stop')
        if self.active_pid==pid:self.active_pid=None
        self.owned.discard(pid)
    def start(self,role):
        need(self.s.listener(self.a.port,False)is None,'Refuse duplicate service start')
        need(os.getpriority(os.PRIO_PROCESS,os.getpid())==0,'Parent must remain nice0')
        argv=[bytes.fromhex(v)for v in self.before['argv_raw_hex']]
        with self.guard,Path(self.before['log']).open('ab')as log:
            st=os.fstat(log.fileno());offset=st.st_size
            p=subprocess.Popen(argv,cwd=os.fsencode(self.before['cwd']),env=self.env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
            self.owned.add(p.pid);self.children[p.pid]=p
            self.active_pid=p.pid
            self.start_logs[p.pid]=dict(inode=st.st_ino,device=st.st_dev,offset=offset)
            if role=='candidate':self.pending_candidate=p.pid
        try:os.setpriority(os.PRIO_PROCESS,p.pid,0)
        except BaseException:
            self.stop(p.pid);raise
        return p.pid
    def startup_library(self,pid):
        record=self.start_logs.get(pid);need(record is not None,'New process startup log offset required')
        path=Path(self.before['log']);st=path.stat();need(st.st_ino==record['inode']and st.st_dev==record['device']and 0<=st.st_size-record['offset']<=16*1024*1024,'Startup log inode/extent changed')
        with path.open('rb')as f:f.seek(record['offset']);raw=f.read()
        matches=re.findall(rb'myds_metal: loaded precompiled library from (.+?) \(all required V4.1 kernels verified\)',raw)
        need(len(matches)==1,'Exactly one actual verified precompiled-library startup log required; no silent runtime compile')
        loaded=Path(os.fsdecode(matches[0]));loaded=loaded if loaded.is_absolute()else Path(self.before['cwd'])/loaded
        need(loaded.resolve(strict=True)==Path(self.before['metallib']['path'])and sha(loaded)==LIB_SHA,'Actual logged selected c692 library differs')
        return dict(path=str(loaded.resolve()),sha256=LIB_SHA,log_inode=st.st_ino,log_offset=record['offset'],observed_segment_bytes=len(raw),observed_segment_sha256=hashlib.sha256(raw).hexdigest(),required_kernels_verified=True)
    def restore_original(self):
        pid=self.s.listener(self.a.port,False)
        if pid is not None:
            need(pid==self.before['pid'],'Unknown listener blocks original restoration')
            return self.verify('original',pid)
        try:os.kill(self.before['pid'],0)
        except ProcessLookupError:pass
        else:
            # Never create a duplicate while a failed stop is still in progress.
            self.stop(self.before['pid'],original_precheck=False)
        pid=self.start('original');return self.verify('original',pid)
    def verify(self,role,pid):
        deadline=time.monotonic()+240;last='not ready'
        while time.monotonic()<deadline:
            if role=='candidate':self.check()
            try:
                value,raw=self.s.capture(self.a.port)
            except (OSError,ValueError,RuntimeError,subprocess.SubprocessError)as e:
                last=repr(e);time.sleep(.5);continue
            need(value['pid']==pid,'Unexpected listener PID')
            for k in ('argv_raw_hex','myds_env_raw_hex','cwd','log','nice','metallib','metallib_sha256','library_precedence'):
                need(value[k]==self.before[k],'Started identity differs: '+k)
            need(raw_environment(pid)==self.env,'Complete initial environment differs after start')
            model=Path(self.s.option(value['argv'],'-m')).resolve(strict=True)
            need(self.s.metadata(model)==self.b['summary']['model']and sha(model,1048576)==self.b['summary']['model_header_sha256'],'Original main GGUF stat/header identity changed')
            vision=Path(self.s.option(value['argv'],'--vision')).resolve(strict=True)
            need(dict(metadata=self.s.metadata(vision),header_sha256=sha(vision,1048576))==self.vision,'Vision artifact stat/header identity changed')
            if role=='original':
                need(all(value[k]==self.before[k]for k in IDENTITY_KEYS)and raw==self.original_config,'Original inode/config/raw identity not restored')
            else:
                need(value['binary_sha256']==BIN_SHA and value['config_sha256']==hashlib.sha256(self.candidate_raw).hexdigest()and raw==self.candidate_raw and value['effective_myds_env']==dict(self.before['effective_myds_env'],**FLAGS)and len(value['effective_myds_env'])==53,'Candidate binary/config/env53 identity')
                value['actual_loaded_library_startup_evidence']=self.startup_library(pid)
                value['candidate_provider_evidence']=dict(basis='Exact accepted binary plus effective53 strict startup configuration; server has no workspace-counter HTTP export',direct_server_workspace_counters_sampled=False)
                value['vision_artifact_identity']=self.vision
            return value
        raise ValueError('Service readiness/identity failed: '+last)
    def functional(self):
        request=urllib.request.Request(f'http://127.0.0.1:{self.a.port}/v1/chat/completions',data=json.dumps(self.b['request']).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=60)as response:
            need(response.status==200,'HTTP functional status');raw=response.read(8*1024*1024+1);need(len(raw)<=8*1024*1024,'Bounded HTTP response');value=json.loads(raw)
        choices=value.get('choices');need(type(choices)is list and len(choices)==1,'Exactly one functional choice');choice=choices[0];text=choice.get('message',{}).get('content')
        need(type(text)is str,'Functional assistant text');result=dict(content_sha256=hashlib.sha256(text.encode()).hexdigest(),completion_tokens=value.get('usage',{}).get('completion_tokens'),finish_reason=choice.get('finish_reason'))
        need(result==self.b['expected'],'Production functional content/usage oracle differs');self.check()
        return dict(http_status=200,response_sha256=hashlib.sha256(raw).hexdigest(),response_bytes=len(raw),oracle=result,performance_claim=False)
    def source_unchanged(self):need(canonical_snapshot(self.b['canonical'])==self.canonical,'Canonical dirty source changed; no source merge/update is permitted')
    def checkpoint(self,name):pass


class Transaction:
    """Filesystem/inode transaction; process effects are supplied by RealOps or CPU mocks."""
    def __init__(self,ops,out,release_binary,new_config,release_identity=None):
        self.ops=ops;self.out=out;self.release_binary=release_binary;self.new_config=new_config;self.paths={k:Path(ops.before[k]['path'])for k in ('binary','config')};nonce=uuid.uuid4().hex
        self.backups={k:p.with_name(p.name+'.promptclear-original-'+nonce)for k,p in self.paths.items()};self.staged={k:p.with_name(p.name+'.promptclear-new-'+nonce)for k,p in self.paths.items()}
        self.release_identity=release_identity
        self.renamed=[];self.installed=[];self.original_stopped=False;self.candidate_pid=None;self.success=False;self.rolling_back=False;self.in_critical=False;self.pending_signal=False;self.state=dict(schema_version=1,status='prepared',original_pid=ops.before['pid'],paths={k:str(v)for k,v in self.paths.items()},backups={k:str(v)for k,v in self.backups.items()},staged={k:str(v)for k,v in self.staged.items()},private_environment_file='original-environment.private.json',non_myds_values_included=False)
    def event(self,name):
        self.state.update(checkpoint=name,renamed=list(self.renamed),installed=list(self.installed));write(self.out/'transaction-journal.private.json',self.state);self.ops.checkpoint(name)
    def prepare_files(self):
        for k,p in self.staged.items():
            need(not p.exists()and not self.backups[k].exists(),'Unique same-filesystem staging/backup required')
            source=self.release_binary if k=='binary'else None;original_stat=self.paths[k].stat();mode=stat.S_IMODE(original_stat.st_mode)
            fd=os.open(p,os.O_CREAT|os.O_EXCL|os.O_WRONLY,mode)
            with os.fdopen(fd,'wb')as f:
                os.fchown(f.fileno(),original_stat.st_uid,original_stat.st_gid)
                if source:
                    with source.open('rb')as s:shutil.copyfileobj(s,f)
                else:f.write(self.new_config)
                f.flush();os.fsync(f.fileno())
            os.chmod(p,mode);need(p.stat().st_dev==self.paths[k].stat().st_dev,'Staging must be same filesystem');fsync_dir(p.parent)
        need(sha(self.staged['binary'])==sha(self.release_binary)and self.staged['config'].read_bytes()==self.new_config,'Staged payload validation')
        self.event('files_staged')
    def start_candidate(self):
        # Python signal handlers defer exceptions until the spawned PID is tracked.
        self.in_critical=True
        try:self.candidate_pid=self.ops.start('candidate')
        finally:self.in_critical=False
        if self.pending_signal:raise KeyboardInterrupt('Deferred deployment signal')
    def execute(self):
        try:
            self.prepare_files();self.ops.check();self.event('before_stop');self.original_stopped=True;self.state['original_stop_attempted']=True;self.event('stop_requested');self.ops.stop(self.ops.before['pid']);self.event('original_stopped')
            for key in ('binary','config'):
                os.rename(self.paths[key],self.backups[key]);self.renamed.append(key);fsync_dir(self.paths[key].parent);self.event('backup_'+key)
                os.replace(self.staged[key],self.paths[key]);self.installed.append(key);fsync_dir(self.paths[key].parent);self.event('install_'+key)
            self.start_candidate();self.event('candidate_started');actual=self.ops.verify('candidate',self.candidate_pid);self.event('candidate_verified')
            functional=self.ops.functional();self.event('functional_verified');actual=self.ops.verify('candidate',self.candidate_pid);self.ops.source_unchanged();self.ops.check();self.event('source_unchanged');self.ops.finish_monitor()
            receipt=dict(schema_version=1,status='adopted_exact_artifact',production_adoption=True,completed_utc=utc(),source_authority='Separate immutable release archive; canonical dirty HEAD is not claimed as tested',immutable_release_identity=self.release_identity,actual=actual,functional=functional,backups={k:str(v)for k,v in self.backups.items()},original_inodes={k:self.ops.before[k]['inode']for k in self.paths},source_unchanged=True,performance_proven_by_prior_frozen_full_gate_only=True)
            write(self.out/'adoption-receipt.json',receipt);self.event('receipt_written');self.state['status']='adopted';write(self.out/'transaction-journal.private.json',self.state);self.success=True;return receipt
        except BaseException as failure:
            self.rollback(failure)
            raise
        finally:
            for p in self.staged.values():
                if p.exists():p.unlink()
    def rollback(self,failure):
        self.success=False;self.rolling_back=True;self.state.update(status='rolling_back',failure_type=type(failure).__name__)
        # Reporting failures must not prevent file or process recovery attempts.
        try:write(self.out/'transaction-journal.private.json',self.state)
        except OSError:pass
        errors=[]
        candidate=self.candidate_pid or getattr(self.ops,'pending_candidate',None)
        if candidate is not None:
            try:self.ops.stop(candidate)
            except BaseException as e:errors.append(type(e).__name__)
        # Actual backup existence covers a signal between rename and list append.
        for key in ('config','binary'):
            if not self.backups[key].exists():continue
            try:
                need(self.backups[key].is_file(),'Original inode backup missing')
                if self.paths[key].exists():
                    failed=self.paths[key].with_name(self.paths[key].name+'.failed-'+uuid.uuid4().hex);os.rename(self.paths[key],failed)
                os.rename(self.backups[key],self.paths[key]);fsync_dir(self.paths[key].parent)
            except BaseException as e:errors.append(type(e).__name__)
        restored=None
        try:
            restored=self.ops.restore_original()if self.original_stopped else self.ops.before
            self.ops.source_unchanged()
        except BaseException as e:errors.append(type(e).__name__)
        restoration_verified=not errors
        # A formerly written success receipt must never survive as the final decision.
        receipt=self.out/'adoption-receipt.json'
        superseded=self.out/('superseded-adoption-receipt-'+uuid.uuid4().hex+'.private.json')
        if receipt.exists():
            try:
                os.replace(receipt,superseded);fsync_dir(self.out)
            except OSError as e:
                errors.append('receipt_rename:'+type(e).__name__)
                try:receipt.unlink();fsync_dir(self.out)
                except OSError as deletion:errors.append('receipt_unlink:'+type(deletion).__name__)
            try:write(receipt,dict(schema_version=1,status='rolled_back'if restoration_verified else'rollback_failed',production_adoption=False,decision_superseded_by='rollback-receipt.json'))
            except OSError as e:
                errors.append('receipt_false_write:'+type(e).__name__)
                # In particular ENOSPC while writing the negative replacement must
                # not leave the previous active positive record in place.
                if receipt.exists():
                    try:receipt.unlink();fsync_dir(self.out)
                    except OSError as deletion:errors.append('receipt_cleanup:'+type(deletion).__name__)
        invalidated=not receipt.exists()
        if not invalidated:
            try:invalidated=load(receipt).get('production_adoption')is False
            except (OSError,ValueError):invalidated=False
        if not invalidated:errors.append('active_success_receipt_not_invalidated')
        self.state.update(status='rollback_failed'if errors else'rolled_back',recovery_failure_types=list(errors),restored_pid=restored['pid']if restored else None,original_inode_argv_full_env_config_health_restored=restoration_verified,active_success_receipt_invalidated=invalidated,superseded_receipt=str(superseded)if superseded.exists()else None)
        try:write(self.out/'rollback-receipt.json',self.state);write(self.out/'transaction-journal.private.json',self.state)
        except OSError as e:
            errors.append('rollback_report_write:'+type(e).__name__);self.state.update(status='rollback_failed',recovery_failure_types=list(errors))
        if errors:raise RuntimeError('Deployment failed; inspect private recovery journal and original inode backups')


def schema():
    ref=dict(path='/ABS/PATH',sha256='64 lowercase hexadecimal characters',bytes='optional exact nonnegative int, not bool')
    return dict(schema_version=1,implementation_version=4,candidate_stage='/tmp/myds-moe-phys-cause-fullv2-868d-20261024-prototype',full_window='/tmp/FINAL_FULL_WINDOW',plan=ref,gates={k:ref for k in ROLES},probe_controller=ref,production_reference=ref,executed={k:ref for k in ('input_map','bindings','source_archive','artifacts_archive')},functional_request=ref,functional_expected=ref)


def self_test():
    """Only temporary files and fake processes. No capture, model, Metal or network."""
    controls=[]
    def case(name,fn):fn();controls.append(dict(case=name,passed=True))
    def reject(fn):
        try:fn()
        except (ValueError,RuntimeError):return
        raise AssertionError('Invalid input accepted')
    env={f'MYDS_FIXTURE_{i}':str(i)for i in range(48)};env.update(MYDS_METAL_CONCURRENT='1',MYDS_PLD='1',MYDS_PREFIX_CACHE='1');raw=b''.join((k+'='+v+'\n').encode()for k,v in env.items())
    case('config_original51_exact_prefix_plus_two_flags53',lambda:need(candidate_config(raw,{},env).startswith(raw)and len(parse_config(candidate_config(raw,{},env)))==53,'config'))
    case('reject_inherited_candidate_override',lambda:reject(lambda:candidate_config(raw,{next(iter(FLAGS)):'0'},env)))
    case('reject_modified_original51',lambda:reject(lambda:candidate_config(raw,{'MYDS_PLD':'0'},env)))
    last=next(iter(env));partial=b''.join((k+'='+v+'\n').encode()for k,v in env.items()if k!=last)
    case('effective51_with_fifty_config_plus_inherited_missing_key',lambda:need(len(parse_config(candidate_config(partial,{last:env[last]},env)))==52,'Valid config/inherited distribution'))
    blob_prefix=struct.pack('i',2)+b'/CPU/executable\0\0/CPU/executable\0--CPU-only\0'
    case('full_initial_env_stops_before_trailing_storage',lambda:need(parse_environment_blob(blob_prefix+b'NON_MYDS=private\0MYDS_X=1\0\0NOT_AN_ENV=trailing\0')=={b'NON_MYDS':b'private',b'MYDS_X':b'1'},'Do not inject trailing KERN storage as env'))
    case('full_initial_env_preserves_empty_values_and_raw_bytes',lambda:need(parse_environment_blob(blob_prefix+b'EMPTY=\0RAW=\xff\0\0')=={b'EMPTY':b'',b'RAW':b'\xff'},'Exact byte environment'))
    case('reject_duplicate_initial_env_key',lambda:reject(lambda:parse_environment_blob(blob_prefix+b'K=1\0K=2\0\0')))
    case('reject_malformed_initial_env_record',lambda:reject(lambda:parse_environment_blob(blob_prefix+b'NO_EQUALS\0\0')))
    identity={'candidate':dict(CORE,**{'bin/myds.metallib':LIB_SHA,NATIVE:'f'*64}),'gold':{'CPU':'fixture'}}
    refs={k:dict(path='/CPU/'+k,sha256=hashlib.sha256(k.encode()).hexdigest())for k in ROLES};model=dict(path='/CPU/model',bytes=123);header='a'*64;inp='b'*64
    values={k:dict(evidence_valid=True,implementation_version=4,plan_sha256=PLAN_SHA,production_adoption=False,**{key:True})for k,key in ROLES.items()}
    values['component'].update(candidate_core_sha256=CORE,metallib_sha256=LIB_SHA,production_environment=env,model=model,model_header_sha256=header,model_backed_full_weight_intervals_verified_by_long_reader=True)
    values['recovery'].update(recovery_plan_sha256=PLAN_SHA,component_review_sha256=refs['component']['sha256'],independent_fullraw_replay_completed=True,independent_review=dict(full_bytes=dict(pair_readers_replayed=14)))
    values['causal'].update(mode='causal_diagnostic',correctness_pass=True,native_identity={'candidate':identity['candidate']},production_environment=env,model=model,model_header_sha256=header)
    prereqs={ROLES[k]:refs[k]for k in ('component','recovery','causal')}
    for k in ('probe','full'):values[k].update(mode=k,correctness_pass=True,recorded_restoration_exact=True,native_identity=identity,all91_exact_production_source_inputs=True,prerequisites=prereqs,performance_measured=k=='full',gate={ROLES[k]:True})
    values['full']['archive_verification']=dict(input_map_sha256=inp)
    call=lambda v:gate_set(v,refs,identity,env,model,header,inp)
    case('all_five_bound_independent_gates_positive_CPU_fixture',lambda:call(values))
    import copy
    for role,key in ROLES.items():
        v=copy.deepcopy(values);v[role][key]=False;case('reject_'+role+'_gate_false',lambda v=v:reject(lambda:call(v)))
    for name,fn in (
      ('wrong_core',lambda v:v['component']['candidate_core_sha256'].update({'src/myds_v41.c':'0'*64})),
      ('wrong_library',lambda v:v['component'].update(metallib_sha256='0'*64)),
      ('wrong_environment',lambda v:v['causal'].update(production_environment={})),
      ('wrong_model',lambda v:v['component'].update(model={})),
      ('recovery_pair_13',lambda v:v['recovery']['independent_review']['full_bytes'].update(pair_readers_replayed=13)),
      ('full_archive_unbound',lambda v:v['full']['archive_verification'].update(input_map_sha256='0'*64)),
      ('probe_native_wrong',lambda v:v['probe'].update(native_identity={})),
      ('full_prerequisite_wrong',lambda v:v['full'].update(prerequisites={})),
      ('full_not_independent_raw',lambda v:v['full'].update(all91_exact_production_source_inputs=False)),
      ('recovery_component_wrong',lambda v:v['recovery'].update(component_review_sha256='0'*64))):
        v=copy.deepcopy(values);fn(v);case('reject_'+name,lambda v=v:reject(lambda:call(v)))
    independent=dict(gate=dict(growth_gate_pass=True),raw_comparisons=[dict(full_original_bytes_equal=True)],archive_verification=dict(input_map_sha256=inp));controller=dict(gate=dict(growth_gate_pass=True),raw_comparisons=[dict(full_original_bytes_equal=True)]);idx={'/CPU/NEW-inputs/copied-long.json':dict(archive='source.tar.gz',sha256='d'*64,bytes=123)}
    case('probe_controller_and_purelong_distinct_roles_NEW_input_copy_bound',lambda:probe_binding_gate(independent,controller,inp,idx,'d'*64,123))
    for name,fn in (
      ('raw_comparison_changed',lambda v:v.update(raw_comparisons=[])),
      ('embedded_gate_changed',lambda v:v.update(gate=dict(growth_gate_pass=False))),
      ('wrong_original_probe_inputmap',lambda v:v.update(archive_verification=dict(input_map_sha256='0'*64)))):
        v=copy.deepcopy(independent);fn(v);case('reject_probe_binding_'+name,lambda v=v:reject(lambda:probe_binding_gate(v,controller,inp,idx,'d'*64,123)))
    case('reject_probe_long_not_frozen',lambda:reject(lambda:probe_binding_gate(independent,controller,inp,{},'d'*64,123)))
    case('reject_probe_long_extent_changed',lambda:reject(lambda:probe_binding_gate(independent,controller,inp,idx,'d'*64,124)))
    scenarios=('deterministic','async','allocation','allocation-owner','shared-pso','prompt-allocation-first','prompt-allocation-last')
    window_fixture=[dict(window='/CPU/window'+str(i),status=s,native_invocations=n,pair_readers_replayed=p,input_map_sha256=str(i+1)*64,restoration=dict(restored=True,exact_argv_myds_artifacts_config=True),source_archive={},artifacts_archive={},archive_bindings={})for i,(s,n,p)in enumerate((('failed',10,5),('continuation_passed_review_required',18,9)))]
    recovery=dict(source_windows=window_fixture,native_invocations=28,pair_readers_replayed=14,arm_executions=[dict(scenario=s,temperature=t,setting=n)for s in scenarios for t in ('0','0.7')for n in (0,1)],pair_reports=[dict(scenario=s,temperature=t,evidence_valid=True,correctness_pass=True)for s in scenarios for t in ('0','0.7')])
    case('explicit_two_window10plus18_correctness_coverage',lambda:recovery_coverage(recovery))
    for name,fn in (
      ('pretend_single_window',lambda v:v.update(source_windows=v['source_windows'][:1])),
      ('erase_original_failed_status',lambda v:v['source_windows'][0].update(status='passed')),
      ('missing_native',lambda v:v.update(native_invocations=27)),
      ('missing_pair',lambda v:v.update(pair_reports=v['pair_reports'][:-1])),
      ('duplicate_arm',lambda v:v['arm_executions'].__setitem__(1,v['arm_executions'][0])),
      ('source_window_restore_failed',lambda v:v['source_windows'][1]['restoration'].update(restored=False))):
        v=copy.deepcopy(recovery);fn(v);case('reject_recovery_'+name,lambda v=v:reject(lambda:recovery_coverage(v)))
    with tempfile.TemporaryDirectory(prefix='promptclear-adoption-CPU-')as tmp:
        base=Path(tmp);p=base/'reference';p.write_bytes(b'CPU ref');record=descriptor(p)
        case('strict_reference_correct_optional_bytes',lambda:reference(record))
        for name,bad in [('negative',-1),('bool',True),('wrong',record['bytes']+1)]:
            d=dict(record,bytes=bad);case('reject_reference_bytes_'+name,lambda d=d:reject(lambda:reference(d)))
        d=dict(record,extra=1);case('reject_reference_extra_key',lambda:reject(lambda:reference(d)))
        points=['files_staged','before_stop','stop_requested','original_stopped','backup_binary','install_binary','backup_config','install_config','candidate_started','candidate_verified','functional_verified','source_unchanged','receipt_written']
        for point in [None,*points,'stop_original_pre_signal','start_candidate','verify_candidate','functional','finish_monitor']:
            d=base/('success'if point is None else point);d.mkdir();binary=d/'myds';binary.write_bytes(b'CPU old');config=d/'myds.conf';config.write_bytes(raw);release=d/'release';release.write_bytes(b'CPU new');out=d/'out';out.mkdir()
            inodes={'binary':binary.stat().st_ino,'config':config.stat().st_ino}
            class Fake:
                def __init__(self):self.before=dict(pid=101,binary=dict(path=str(binary),inode=inodes['binary']),config=dict(path=str(config),inode=inodes['config']));self.running=101;self.once=False
                def fail(self,name):
                    if point==name and not self.once:self.once=True;raise ValueError('CPU injected '+name)
                def checkpoint(self,name):self.fail(name)
                def check(self):pass
                def finish_monitor(self):self.fail('finish_monitor')
                def stop(self,pid):
                    if pid==101:self.fail('stop_original_pre_signal')
                    need(self.running==pid,'Owned fake PID');self.running=None
                def start(self,role):self.fail('start_'+role);need(self.running is None,'No duplicate fake process');self.running=202 if role=='candidate'else 303;return self.running
                def verify(self,role,pid):
                    self.fail('verify_'+role);need(self.running==pid,'Fake listener');need(binary.read_bytes()==(b'CPU new'if role=='candidate'else b'CPU old'),'Fake binary');need(config.read_bytes()==(new if role=='candidate'else raw),'Fake config');return dict(pid=pid,complete_environment_restored=True)
                def functional(self):self.fail('functional');return dict(cpu_fixture_only=True)
                def source_unchanged(self):pass
                def restore_original(self):
                    if self.running==101:return self.verify('original',101)
                    return self.verify('original',self.start('original'))
            fake=Fake();new=candidate_config(raw,{},env);tx=Transaction(fake,out,release,new)
            if point is None:
                tx.execute();need(fake.running==202 and binary.read_bytes()==b'CPU new'and all(tx.backups[k].stat().st_ino==v for k,v in inodes.items()),'Success keeps original inodes');controls.append(dict(case='CPU_success_retains_original_inode_backups',passed=True))
            else:
                reject(tx.execute);need(binary.read_bytes()==b'CPU old'and config.read_bytes()==raw and binary.stat().st_ino==inodes['binary']and config.stat().st_ino==inodes['config']and fake.running in (101,303),'Exact original inode/content/process restoration')
                if(out/'adoption-receipt.json').exists():need(load(out/'adoption-receipt.json')['production_adoption']is False,'No stale success receipt after rollback')
                controls.append(dict(case='CPU_rollback_'+point,passed=True))
        private={b'NON_MYDS_CPU_SECRET':b'not-for-stdout',b'MYDS_FIXTURE':b'1'};secret=base/'environment.private.json';write(secret,private_env_record(private));need(stat.S_IMODE(secret.stat().st_mode)==0o600,'Private0600');controls.append(dict(case='complete_environment_backup0600_only',passed=True))
        # Run the actual CLI/main decision tree with only fake process effects.
        import contextlib,io
        from unittest.mock import patch
        original_open=open
        for mode in ('prepare','run','run-functional-failure','run-postsummary-and-false-receipt-error'):
            d=base/('main-'+mode);d.mkdir();binary=d/'myds';binary.write_bytes(b'CPU old');config=d/'myds.conf';config.write_bytes(raw);newbin=d/'newbin';newbin.write_bytes(b'CPU new');manifest=d/'manifest.json';manifest.write_text('{}');out=d/'out';release_dir=d/'release';calls=[];old_inode=binary.stat().st_ino
            before=dict(pid=101,effective_myds_env=env,binary=dict(path=str(binary),inode=old_inode),config=dict(path=str(config),inode=config.stat().st_ino))
            class CLIRealOpsFixture:
                def __init__(self,a,b):self.a=a;self.before=before;self.env=private;self.canonical=dict(head='CPU');self.running=101
                def preflight(self):calls.append('preflight');return candidate_config(raw,{},env)
                def begin_monitor(self):calls.append('monitor')
                def close_monitor(self):pass
                def finish_monitor(self):pass
                def check(self):pass
                def checkpoint(self,name):pass
                def stop(self,pid):calls.append('stop');need(self.running==pid,'fake owned');self.running=None
                def start(self,role):calls.append('start-'+role);need(self.running is None,'No duplicate');self.running=202 if role=='candidate'else 303;return self.running
                def verify(self,role,pid):calls.append('verify-'+role);return dict(pid=pid,effective_myds_env=dict(env,**FLAGS)if role=='candidate'else env)
                def functional(self):
                    calls.append('functional')
                    if mode=='run-functional-failure':raise ValueError('CPU fake HTTP failure')
                    return dict(cpu_fixture_only=True)
                def source_unchanged(self):pass
                def restore_original(self):return self.verify('original',self.start('original'))
            def fake_release(*args):
                calls.append('release');release_dir.mkdir();write(release_dir/'release-identity.json',dict(cpu_fixture=True));return newbin
            def only_temp_open(path,*args,**kwargs):
                return original_open(d/'cpu-only-lock'if str(path)=='/tmp/myds-codex-decode-window.lock'else path,*args,**kwargs)
            real_write=write
            def publication_fault(path,value,*args,**kwargs):
                if mode=='run-postsummary-and-false-receipt-error':
                    if Path(path).name=='summary.json'and value.get('production_adoption')is True:raise OSError(28,'CPU injected summary ENOSPC')
                    if Path(path).name=='adoption-receipt.json'and value.get('production_adoption')is False:raise OSError(28,'CPU injected negative-receipt ENOSPC')
                return real_write(path,value,*args,**kwargs)
            arguments=['--prepare'if mode=='prepare'else'--run','--manifest',str(manifest),'--manifest-sha256',sha(manifest),'--release-dir',str(release_dir),'--out',str(out)]
            with patch.object(sys,'platform','darwin'),patch(__name__+'.validate_bundle',return_value={}),patch(__name__+'.RealOps',CLIRealOpsFixture),patch(__name__+'.save_release',side_effect=fake_release),patch(__name__+'.write',side_effect=publication_fault),patch('builtins.open',side_effect=only_temp_open),patch.object(fcntl,'flock'),patch.object(subprocess,'Popen',side_effect=AssertionError('CPU controls must never spawn')),patch.object(signal,'signal'),contextlib.redirect_stdout(io.StringIO()):
                code=main(arguments)
            if mode=='prepare':need(code==0 and calls==['preflight']and binary.stat().st_ino==old_inode and not release_dir.exists(),'Prepare has no release/stop/start/network side effects')
            elif mode=='run':need(code==0 and calls.count('stop')==1 and calls.count('start-candidate')==1 and load(out/'summary.json')['production_adoption']is True,'Mock main success')
            elif mode=='run-postsummary-and-false-receipt-error':
                r=load(out/'rollback-receipt.json');need(code==1 and binary.stat().st_ino==old_inode and binary.read_bytes()==b'CPU old'and r['original_inode_argv_full_env_config_health_restored']is True and r['status']=='rollback_failed'and r['active_success_receipt_invalidated']is True and not(out/'adoption-receipt.json').exists()and load(out/'summary.json')['production_adoption']is False and any('receipt_false_write'in x for x in r['recovery_failure_types']),'After successful install, summary and negative receipt write failures restore original and cannot retain an active positive receipt')
            else:need(code==1 and binary.stat().st_ino==old_inode and binary.read_bytes()==b'CPU old'and load(out/'summary.json')['production_adoption']is False and load(out/'rollback-receipt.json')['status']=='rolled_back','Mock main finally exact original recovery')
            for leaf in out.glob('*.json'):
                if leaf.name!='original-environment.private.json':need(b'not-for-stdout'not in leaf.read_bytes()and b'6e6f742d666f722d7374646f7574'not in leaf.read_bytes(),'No non-MYDS secret in public receipt')
            controls.append(dict(case='actual_CLI_'+mode+'_CPU_mock_no_Popen',passed=True))
        # Real bundle validation rejects incomplete manifests before lock/output/ops.
        for label,digest in (('missing_gates',sha(manifest)),('wrong_manifest_SHA','0'*64)):
            badout=base/('blocked-'+label);badrelease=base/('blocked-release-'+label)
            args=['--run','--manifest',str(manifest),'--manifest-sha256',digest,'--release-dir',str(badrelease),'--out',str(badout)]
            with patch.object(sys,'platform','darwin'),patch(__name__+'.RealOps',side_effect=AssertionError('No production facade before gates')),patch.object(subprocess,'Popen',side_effect=AssertionError('No Popen before gates')),contextlib.redirect_stdout(io.StringIO()):code=main(args)
            need(code==2 and not badout.exists()and not badrelease.exists(),'Fail before any staging/mutation');controls.append(dict(case='actual_CLI_blocks_'+label+'_before_staging',passed=True))
        from types import SimpleNamespace
        s=SimpleNamespace(listener=lambda *x:(_ for _ in ()).throw(AssertionError('Dead child must not query/kill listener')))
        op=RealOps(SimpleNamespace(),dict(safety=s));op.children[123]=SimpleNamespace(poll=lambda:1);op.owned.add(123)
        with patch.object(os,'kill',side_effect=AssertionError('No actual signal in CPU control')):op.stop(123)
        controls.append(dict(case='actual_stop_already_exited_owned_child_no_signal_or_listener',passed=True))
    return dict(schema_version=1,scope='CPU fixtures and fake process transaction only',all_controls_pass=True,controls=controls,control_count=len(controls),gpu_model_network_service_executed=False,production_adoption=False,tool_sha256=sha(Path(__file__)))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--run',action='store_true');mode.add_argument('--self-test',action='store_true');mode.add_argument('--print-schema',action='store_true')
    p.add_argument('--manifest',type=Path);p.add_argument('--manifest-sha256');p.add_argument('--release-dir',type=Path);p.add_argument('--out',type=Path);p.add_argument('--port',type=int,default=8187);a=p.parse_args(argv)
    if a.print_schema:print(json.dumps(schema(),indent=2,sort_keys=True));return 0
    if a.self_test:
        result=self_test()
        if a.out:
            need(not a.out.exists(),'New CPU report path required');write(a.out,result)
        print(json.dumps(dict(all_controls_pass=result['all_controls_pass'],control_count=result['control_count'],production_adoption=False)));return 0
    need(sys.platform=='darwin','macOS required for future prepare/activation')
    need(all(getattr(a,n)is not None for n in ('manifest','manifest_sha256','release_dir','out')),'Manifest exact SHA, NEW release-dir and NEW out required')
    for name in ('manifest','release_dir','out'):
        path=getattr(a,name);need(path.is_absolute(),'Absolute paths required');setattr(a,name,path.resolve())
    need(not a.out.exists()and a.out.parent.is_dir()and not a.release_dir.exists()and a.release_dir.parent.is_dir(),'Existing parents and new output/release paths required')
    # All independent gates/archive bytes, before output staging or production mutation.
    try:
        bundle=validate_bundle(a)
        lock=open('/tmp/myds-codex-decode-window.lock','a+')
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BaseException:lock.close();raise
    except BaseException as e:
        print(json.dumps(dict(status='blocked_before_mutation',production_adoption=False,error_type=type(e).__name__)));return 2
    ops=RealOps(a,bundle);tx=None
    try:
        new_config=ops.preflight();ops.candidate_raw=new_config
        a.out.mkdir(mode=0o700);write(a.out/'before.json',ops.before);write(a.out/'source-before.json',ops.canonical)
        env_serial=private_env_record(ops.env);write(a.out/'original-environment.private.json',env_serial)
        summary=dict(schema_version=1,status='prepared',production_adoption=False,manifest_sha256=sha(a.manifest),bin_sha256=BIN_SHA,metallib_sha256=LIB_SHA,source_baseline=BASE_COMMIT,core4=CORE,original_effective51=ops.before['effective_myds_env'],candidate_effective53=dict(ops.before['effective_myds_env'],**FLAGS),full_initial_environment_sha256=hashlib.sha256(json.dumps(env_serial,sort_keys=True).encode()).hexdigest(),full_initial_environment_count=len(ops.env),non_myds_values_publicly_recorded=False,service_operations=False)
        if a.prepare:write(a.out/'summary.json',summary);print(json.dumps(dict(status='prepared',production_adoption=False)));return 0
        release_binary=save_release(bundle,a.release_dir,a.manifest);tx=Transaction(ops,a.out,release_binary,new_config,descriptor(a.release_dir/'release-identity.json'))
        def interrupted(number,frame):
            if tx.rolling_back:return
            if tx.in_critical:tx.pending_signal=True;return
            raise KeyboardInterrupt('Deployment interrupted')
        for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,interrupted)
        signal.signal(signal.SIGHUP,signal.SIG_IGN);ops.begin_monitor()
        receipt=tx.execute();summary.update(status='adopted_exact_artifact',production_adoption=True,service_operations=True,release_dir=str(a.release_dir),actual_pid=receipt['actual']['pid'],functional=receipt['functional']);write(a.out/'summary.json',summary)
        print(json.dumps(dict(status=summary['status'],production_adoption=True,report=str(a.out/'adoption-receipt.json'))));return 0
    except BaseException as e:
        # Private journal has details; stdout contains no original environment or payload.
        if tx and tx.success:
            try:tx.rollback(e)
            except BaseException:pass
        if a.out.exists():
            failed=dict(summary)if 'summary'in locals()else dict(schema_version=1,manifest_sha256=a.manifest_sha256)
            failed.update(status='failed',production_adoption=False,error_type=type(e).__name__,rollback_status=tx.state['status']if tx else'no_production_mutation');write(a.out/'summary.json',failed)
        print(json.dumps(dict(status='failed',production_adoption=False,error_type=type(e).__name__,report=str(a.out))));return 1
    finally:
        try:ops.close_monitor()
        finally:lock.close()


if __name__=='__main__':raise SystemExit(main())
