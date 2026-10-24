#!/usr/bin/env python3
"""Independent read-only live identity audit after explicit promptclear adoption.

No completion, model invocation, Metal operation, stop, signal, restart, install,
or repair is implemented. Outputs are new audit JSON/config snapshots only.
"""
import argparse,ast,ctypes,datetime,fcntl,hashlib,importlib.util,json,os,re,stat,struct,subprocess,sys,tarfile
from pathlib import Path
sys.dont_write_bytecode=True
PLAN='a83127b1427b09b703883150f3ec6e8e90e7ecdead3f48b798c7755368bd9d9c'
BASE='29058d822c95d86ffdc6e84798fe87093d7f8c094b1002f1250177b12066d72b'
BIN='9cf2a670a22632874829398eece031111e438b4ca803860a2e6bcbcc7e94c57a'
LIB='c692d821aed43ecce1cd9003f11e19083cbe5ce2923bef44a8df23c1182dfc81'
SAFETY='f093a967ea76487168863461b3f5273ebee5faa2dca6527ec58376a5927d69bb'
READER='e2baa03cf9ae488be73200d406dcc1447dcc0d8effeaa871bdac934caaf42c3a'
ADOPTER='347f9657efc26e59318549682f70ffd705dc53b2faa4a1a64edcee303b51b72e'
CORE={'src/myds_v41.c':'554387ef6d41612b4e7dc537cf759105bcb7fcaf103d74a6f01ebdb6bf6cc15f','src/myds_metal.m':'28344f08a27c6c87b51676afab73a9daaaa0a653d8dcd0e2d7c71d094e27f0b7','include/myds_internal.h':'9869b7692fa8a7fbb2f7dc02a74768423668651005238574d2e33cb65066cd65','include/myds_metal.h':'3926c8a7eded05a90abccaf4dd8792c8b45cd744f7536f91eacc538b029821d5'}
FLAGS={'MYDS_PREFILL_MOE_WORKSPACE':'1','MYDS_PREFILL_PROMPT_VIEW_CLEAR':'1'}

def need(v,m):
    if not v:raise ValueError(m)
def load(p):
    def pairs(items):
        out={}
        for k,v in items:need(k not in out,'Duplicate JSON key');out[k]=v
        return out
    return json.loads(Path(p).read_bytes(),object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))
def sha(p,n=None):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        if n is not None:
            data=f.read(n);need(len(data)==n,'Truncated header');h.update(data)
        else:
            for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def desc(p):
    p=Path(p).resolve(strict=True);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def reference(r):
    need(type(r)is dict and set(r)in ({'path','sha256'},{'path','sha256','bytes'}),'Exact original reference')
    p=Path(r['path']);need(p.is_absolute() and p.is_file() and not p.is_symlink() and sha(p)==r['sha256'],'Original reference bytes/SHA')
    if 'bytes'in r:need(type(r['bytes'])is int and r['bytes']>=0 and p.stat().st_size==r['bytes'],'Original reference extent')
    return p.resolve(strict=True)
def write_new(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def module(p,name,digest):
    need(sha(p)==digest,'Frozen read-only helper bytes');spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def environment_digest_from_blob(data):
    argc=struct.unpack_from('i',data)[0];need(0<argc<4096,'Actual argc');cursor=data.index(b'\0',4)+1
    while cursor<len(data) and data[cursor]==0:cursor+=1
    for _ in range(argc):cursor=data.index(b'\0',cursor)+1
    private={};terminated=False
    for item in data[cursor:].split(b'\0'):
        if not item:terminated=True;break
        need(b'='in item,'Actual environment encoding');k,v=item.split(b'=',1);need(k and k.hex()not in private,'Unique initial environment');private[k.hex()]=v.hex()
    need(terminated and private,'Complete initial environment terminator')
    return dict(count=len(private),sha256=hashlib.sha256(json.dumps(private,sort_keys=True).encode()).hexdigest(),non_myds_names_or_values_recorded=False)
def complete_environment_digest(pid):
    lib=ctypes.CDLL('/usr/lib/libSystem.B.dylib',use_errno=True);lib.sysctl.argtypes=[ctypes.POINTER(ctypes.c_int),ctypes.c_uint,ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t),ctypes.c_void_p,ctypes.c_size_t];lib.sysctl.restype=ctypes.c_int
    mib=(ctypes.c_int*3)(1,49,pid);size=ctypes.c_size_t();need(lib.sysctl(mib,3,None,ctypes.byref(size),None,0)==0,'Actual process initial environment size');buf=ctypes.create_string_buffer(size.value);need(lib.sysctl(mib,3,buf,ctypes.byref(size),None,0)==0,'Actual process initial environment bytes')
    return environment_digest_from_blob(buf.raw[:size.value])
def canonical_snapshot(root):
    env=dict(os.environ,GIT_OPTIONAL_LOCKS='0')
    def git(*v):return subprocess.check_output(['/usr/bin/git','-C',str(root),*v],env=env)
    untracked={}
    for n in git('ls-files','--others','--exclude-standard','-z').split(b'\0'):
        if not n:continue
        p=root/os.fsdecode(n);need(p.is_file() and not p.is_symlink(),'Untracked regular original source');untracked[os.fsdecode(n)]=sha(p)
    return dict(head=git('rev-parse','HEAD').decode().strip(),status_porcelain=git('status','--porcelain=v1','--untracked-files=all').decode().splitlines(),tracked_diff_sha256=hashlib.sha256(git('diff','--binary','HEAD')).hexdigest(),untracked_sha256=untracked)
def archive_sources(release,manifest):
    d=release/'executed';mapping=load(d/'archive-bindings.json');inputs=load(d/'stage-inputs.json');need(mapping['status']=='complete' and sha(d/'stage-inputs.json')==mapping['stage_inputs_sha256']==manifest['executed']['input_map']['sha256'],'Release input map identity')
    records=mapping['files'];by_original={r['original_path']:r for r in records};need(len(records)==len(by_original)==len(inputs) and {k:v['sha256'] for k,v in by_original.items()}==inputs,'Complete executed source mapping')
    counts={}
    for kind,key in (('source.tar.gz','source_archive'),('artifacts.tar.gz','artifacts_archive')):
        path=d/kind;v=mapping['archives'][kind];need(sha(path)==v['sha256']==manifest['executed'][key]['sha256'] and path.stat().st_size==v['bytes'],'Release physical source/artifact archive')
        selected={r['tar_path']:r for r in records if r['archive']==kind};seen=set()
        with tarfile.open(path,'r:gz') as tar:
            for member in tar:
                need(member.isfile() and not member.islnk() and not member.issym() and not Path(member.name).is_absolute() and '..'not in Path(member.name).parts and member.name not in seen and member.name in selected,'Bounded exact archived member')
                r=selected[member.name];f=tar.extractfile(member);need(f is not None,'Readable archive member');h=hashlib.sha256();n=0
                for block in iter(lambda:f.read(1048576),b''):h.update(block);n+=len(block)
                need(n==member.size==r['bytes'] and h.hexdigest()==r['sha256'],'All actual executed source/member bytes');seen.add(member.name)
        need(seen==selected.keys() and len(seen)==v['members'],'Complete archive member set');counts[kind]=len(seen)
    return by_original,dict(original_frozen_inputs=len(inputs),members=counts,input_map_sha256=sha(d/'stage-inputs.json'),bindings_sha256=sha(d/'archive-bindings.json'),all_member_bytes_sha_verified=True)

def inspect(a):
    started=datetime.datetime.now(datetime.timezone.utc).isoformat();need(sha(a.manifest)==a.manifest_sha256,'Explicit adoption manifest SHA');m=load(a.manifest);need(m['implementation_version']==4,'Implementation4 manifest');stage=Path(m['candidate_stage']);full=Path(m['full_window']);summary=load(full/'summary.json');receipt=load(a.adoption_dir/'adoption-receipt.json');decision=load(a.adoption_dir/'summary.json');before=load(a.adoption_dir/'before.json');journal=load(a.adoption_dir/'transaction-journal.private.json');need(sha(a.adopter)==ADOPTER,'Actual reviewed347 adopter source')
    safety=module(a.tools_dir/'run_iq2_signed4_window.py','adoption_live_safety',SAFETY);reader_path=a.tools_dir/'review_moe_workspace_fullmodel_promptclear.py';need(sha(reader_path)==READER,'Original source hashes from frozen reader');tree=ast.parse(reader_path.read_text());gold=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='GOLD_SOURCE_HASHES' for t in n.targets));need(len(gold)==91,'Original91 source inputs')
    live,raw=safety.capture(a.port);write_new(a.out_dir/'live-identity.json',live)
    with (a.out_dir/'live-myds.conf').open('xb') as f:f.write(raw)
    checks={}
    def check(k,v):checks[k]=bool(v)
    check('successful_actual_adoption_receipt_summary_journal',receipt.get('production_adoption')is True and receipt.get('status')=='adopted_exact_artifact' and decision.get('production_adoption')is True and decision.get('status')=='adopted_exact_artifact' and journal['status']=='adopted')
    check('no_rollback_receipt',not(a.adoption_dir/'rollback-receipt.json').exists())
    check('manifest_and_plan_frozen',decision['manifest_sha256']==a.manifest_sha256 and m['plan']['sha256']==PLAN and reference(m['plan']).is_file())
    original41=load(reference(m['production_reference']));check('original41_is_baseline_only',original41['all_checks_pass']is True and len(original41['checks'])==41 and all(x is True for x in original41['checks'].values()) and original41['live']==before)
    check('actual_listener_is_new_adopted_pid',live['pid']==receipt['actual']['pid']==decision['actual_pid'] and live['pid']!=before['pid'])
    for k in ('argv_raw_hex','argv','myds_env_raw_hex','myds_env','cwd','log','nice','metallib','metallib_sha256','library_precedence'):
        check('actual_original_'+k,live[k]==before[k])
    check('original_inherited_myds_exact_two',len(live['myds_env'])==2 and live['myds_env'].get('MYDS_DSPARK')=='0' and 'MYDS_DSPARK_FILE'in live['myds_env'])
    check('actual_nice_zero',live['nice']==0)
    check('actual_mapped_binary_canonical_9cf',live['binary_sha256']==BIN and Path(live['binary']['path'])==Path(__file__).resolve().parent.parent / 'bin' / 'myds')
    check('actual_c692_metallib',live['metallib_sha256']==LIB and sha(live['metallib']['path'])==LIB)
    check('actual_health_ok',live['health']=={'status':'ok'})
    original_raw=Path(receipt['backups']['config']).read_bytes();expected_raw=original_raw+(b''if not original_raw or original_raw.endswith(b'\n')else b'\n')+b'\n# Exact promptclear-v4 conditional release\n'+b'MYDS_PREFILL_MOE_WORKSPACE=1\nMYDS_PREFILL_PROMPT_VIEW_CLEAR=1\n'
    check('exact_config_original_bytes_plus_two_flags',raw==expected_raw)
    expected=dict(before['effective_myds_env'],**FLAGS);check('original51_plus_two_effective53',len(before['effective_myds_env'])==51 and len(live['effective_myds_env'])==53 and live['effective_myds_env']==expected==decision['candidate_effective53'])
    check('config_sha_matches_actual_receipt',live['config_sha256']==receipt['actual']['config_sha256']==hashlib.sha256(raw).hexdigest())
    env_digest=complete_environment_digest(live['pid']);check('complete_initial_environment_private_digest_preserved',env_digest['sha256']==decision['full_initial_environment_sha256'] and env_digest['count']==decision['full_initial_environment_count'])
    model=Path(safety.option(live['argv'],'-m')).resolve(strict=True);actual_model=safety.metadata(model);check('actual_main_model_all_stat_fields',actual_model==summary['model']);check('actual_main_model_first1MiB',sha(model,1048576)==summary['model_header_sha256'])
    vision=Path(safety.option(live['argv'],'--vision')).resolve(strict=True);vision_id=dict(metadata=safety.metadata(vision),header_sha256=sha(vision,1048576));check('actual_vision_same_path_and_before_stop_receipt_identity',str(vision)==str(Path(safety.option(before['argv'],'--vision')).resolve(strict=True)) and vision_id==receipt['actual']['vision_artifact_identity'])
    release_identity_path=reference(receipt['immutable_release_identity']);release=release_identity_path.parent;release_identity=load(release_identity_path);check('durable_release_path_and_source_authority',release==Path(decision['release_dir']).resolve(strict=True) and not release.is_relative_to(Path('/tmp').resolve()) and release_identity['core4']==CORE and release_identity['bin_sha256']==BIN and release_identity['metallib_sha256']==LIB and release_identity['manifest_sha256']==a.manifest_sha256 and 'canonical dirty HEAD is not' in release_identity['source_authority'])
    check('release_original_manifest_copy',sha(release/'adoption-input-manifest.json')==a.manifest_sha256)
    gate_records={};roles={'component':'component_gate_pass','recovery':'recovery_gate_pass','causal':'causal_lifetime_gate_pass','probe':'growth_gate_pass','full':'performance_gate_pass'}
    need(set(m['gates'])==set(roles),'All five distinct independent gates')
    for role,key in roles.items():
        path=reference(m['gates'][role]);value=load(path);need(value.get('evidence_valid')is True and value.get(key)is True and value.get('implementation_version')==4 and value.get('plan_sha256')==PLAN,'All five actual independent acceptance reports pass')
        need(sha(release/'gate-records'/(role+'.json'))==m['gates'][role]['sha256'],'Exact gate-record release copy');gate_records[role]=desc(path)
        if role in ('probe','full'):need(value['native_identity']==release_identity['normal_native_identity'] and value.get('all91_exact_production_source_inputs')is True,'Accepted actual native source identity')
        if role=='full':need(value['archive_verification']['input_map_sha256']==m['executed']['input_map']['sha256'] and value['performance_measured']is True,'Independent accepted full cohort actual input map')
    files=load(release/'release-files.json');seen=set();release_records=[]
    for r in files:
        p=release/r['path'];need(not Path(r['path']).is_absolute() and '..'not in Path(r['path']).parts and r['path']not in seen and p.is_file() and not p.is_symlink(),'Bounded exact release file');seen.add(r['path']);need(p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],'Every durable release file SHA');release_records.append(dict(path=r['path'],bytes=r['bytes'],sha256=r['sha256']))
    check('release_manifest_complete_readonly_file_set',seen=={str(p.relative_to(release)) for p in release.rglob('*') if p.is_file() and p.name!='release-files.json'} and all(not(p.stat().st_mode&0o222) for p in release.rglob('*')) and not(release.stat().st_mode&0o222))
    index,archive=archive_sources(release,m);source_rows=[]
    for name,old in gold.items():
        expected_sha=CORE.get(name,old);r=index.get(str(stage/name));ok=r is not None and r['sha256']==expected_sha;source_rows.append(dict(path=name,expected_sha256=expected_sha,archived_sha256=r['sha256']if r else None,pass_check=ok))
    check('all91_actual_archived_sources_exact868_plus_core4',all(r['pass_check'] for r in source_rows) and len(source_rows)==91)
    check('archived_server_bin_and_library_match_live',index[str(stage/'bin/myds')]['sha256']==BIN==sha(release/'artifacts/myds') and index[str(stage/'bin/myds.metallib')]['sha256']==LIB==sha(release/'artifacts/myds.metallib'))
    check('archived_mapping_copy_matches_manifest',sha(release/'executed/archive-bindings.json')==m['executed']['bindings']['sha256'])
    backups={}
    for key,digest in (('binary',BASE),('config',before['config_sha256'])):
        p=Path(receipt['backups'][key]);need(p.is_file() and not p.is_symlink(),'Actual original backup file');st=p.stat();v=dict(path=str(p),bytes=st.st_size,device=st.st_dev,inode=st.st_ino,mtime_ns=st.st_mtime_ns,sha256=sha(p));backups[key]=v
        check('original_'+key+'_backup_inode_SHA_preserved',v['inode']==before[key]['inode']==receipt['original_inodes'][key] and v['device']==before[key]['device'] and v['bytes']==before[key]['bytes'] and v['mtime_ns']==before[key]['mtime_ns'] and v['sha256']==digest)
    log=Path(live['log']);e=receipt['actual']['actual_loaded_library_startup_evidence'];st=log.stat();need(st.st_ino==e['log_inode'],'Actual startup log inode');need(0<=e['observed_segment_bytes']<=16*1024*1024,'Bounded original startup segment')
    with log.open('rb') as f:f.seek(e['log_offset']);segment=f.read(e['observed_segment_bytes'])
    matches=re.findall(rb'myds_metal: loaded precompiled library from (.+?) \(all required V4.1 kernels verified\)',segment);selected=Path(os.fsdecode(matches[0]))if len(matches)==1 else None
    if selected and not selected.is_absolute():selected=Path(live['cwd'])/selected
    check('actual_original_startup_segment_c692_witness',hashlib.sha256(segment).hexdigest()==e['observed_segment_sha256'] and len(matches)==1 and selected.resolve(strict=True)==Path(live['metallib']['path']).resolve(strict=True) and sha(selected)==LIB)
    source_before=load(a.adoption_dir/'source-before.json');canonical=Path(live['binary']['path']).parent.parent;current=canonical_snapshot(canonical);check('canonical_HEAD_status_and_dirty_contents_unchanged',current==source_before)
    lock=Path('/tmp/myds-codex-decode-window.lock');free=True
    if lock.exists():
        with lock.open('rb') as f:
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:free=False
            else:fcntl.flock(f,fcntl.LOCK_UN)
    check('advisory_window_lock_unheld',free)
    relevant=[];names={'myds','myglm','ds4','llama-server','llama-cli','llama-bench','ollama','mlx_lm','vllm','lms','trace_model'}
    for line in subprocess.check_output(['/bin/ps','-axww','-o','pid=,ppid=,comm='],text=True).splitlines():
        v=line.split(None,2);need(len(v)==3,'Actual process scan');pid=int(v[0]);name=Path(v[2]).name.lower()
        if name in names or name.startswith(('bench_','bench-','test_moe_workspace','test_prompt_workspace','capture_moe_workspace','capture_output_a','capture_attention_','moe_prefill_replay')):relevant.append(dict(pid=pid,ppid=int(v[1]),comm=v[2],permitted=pid==live['pid']))
    check('no_competing_model_or_benchmark_process',bool(relevant) and all(r['permitted'] for r in relevant) and {r['pid'] for r in relevant}=={live['pid']})
    check('listener_stable_through_independent_review',safety.listener(a.port)==live['pid'])
    return dict(schema_version=1,scope='independent_read_only_post_adoption_live_identity',review_started_utc=started,review_completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),evidence_valid=all(checks.values()),all_checks_pass=all(checks.values()),checks=checks,observed_pid=live['pid'],original_pid=before['pid'],live_identity=desc(a.out_dir/'live-identity.json'),live_config=desc(a.out_dir/'live-myds.conf'),adoption_receipt=desc(a.adoption_dir/'adoption-receipt.json'),adoption_summary=desc(a.adoption_dir/'summary.json'),manifest=desc(a.manifest),adopter=desc(a.adopter),gate_records=gate_records,tool_sha256=sha(__file__),frozen_helpers=dict(safety_sha256=SAFETY,reader_sha256=READER),complete_initial_environment_digest=env_digest,actual_model=actual_model,model_header_sha256=summary['model_header_sha256'],vision_identity=vision_id,vision_baseline_scope='Actual original argv path plus adopter before-stop stat/header baseline recorded in receipt; not an independently captured pre-stop Vision snapshot',durable_release=dict(path=str(release),identity=release_identity,all_files=release_records,executed_archive=archive,source91=source_rows),original_backups=backups,canonical=dict(current=current,original_snapshot=desc(a.adoption_dir/'source-before.json'),full_dirty_content_comparison=True),lock=dict(path=str(lock),exists=lock.exists(),opened_read_only=True,unheld=free),relevant_processes=relevant,performance_measured=False,extra_completion_requested=False,model_or_GPU_executed=False,service_or_configuration_mutation=False,source_authority='Durable exact executed release archive; canonical dirty HEAD remains a separate source version',decision='post_adoption_live_identity_verified'if all(checks.values())else'withhold_incomplete_live_identity',limits=['This is a point-in-time live identity audit, not another numerical/state/performance cohort.','Safety capture performs only GET /health; no completion or GPU request is issued.','Actual provider counters are not exported by the server; flags1 plus accepted exact binary are distinguished from directly sampled workspace counters.','Main model payload is bound by full stat and first1MiB; the362GB model is not rehashed.','No backups, release, canonical source, stage, configuration, process or lock file contents are modified.'])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('adoption-dir','manifest','tools-dir','adopter','out-dir'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--manifest-sha256',required=True);p.add_argument('--port',type=int,default=8187);p.add_argument('--run',action='store_true',required=True);a=p.parse_args();need(sys.platform=='darwin','macOS only')
    for k in ('adoption_dir','manifest','tools_dir','adopter','out_dir'):
        q=getattr(a,k);need(q.is_absolute(),'Absolute paths');setattr(a,k,q.resolve())
    need(not a.out_dir.exists() and a.out_dir.parent.is_dir(),'NEW audit output directory')
    m=load(a.manifest);protected=[a.adoption_dir,Path(m['candidate_stage']).resolve(),Path(m['full_window']).resolve(),Path(__file__).resolve().parent.parent]
    decision_path=a.adoption_dir/'summary.json'
    if decision_path.is_file():
        d=load(decision_path)
        if d.get('release_dir'):protected.append(Path(d['release_dir']).resolve())
    need(all(a.out_dir!=q and not a.out_dir.is_relative_to(q) for q in protected),'Audit output must not modify original adoption, release, canonical or frozen evidence trees');a.out_dir.mkdir(mode=0o700)
    try:r=inspect(a)
    except Exception as e:r=dict(schema_version=1,evidence_valid=False,all_checks_pass=False,error=repr(e),decision='withhold_incomplete_live_identity',tool_sha256=sha(__file__),model_or_GPU_executed=False,service_or_configuration_mutation=False)
    write_new(a.out_dir/'independent-review.json',r);print(json.dumps({k:r.get(k) for k in ('all_checks_pass','observed_pid','decision','error')},sort_keys=True));return 0 if r['all_checks_pass']else 1
if __name__=='__main__':raise SystemExit(main())
