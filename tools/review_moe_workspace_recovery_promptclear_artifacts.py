#!/usr/bin/env python3
"""Independent CPU replay of frozen v4 28-native/14-pair recovery evidence.

This reads original files and lossless executed-source/artifact archives. It does
not run a model, call Metal, wait for GPU work, repair state or modify raw files.
The optional small top is emitted only after the independent full replay and
binds both the original controller top and this separate long report.
"""
import argparse,copy,hashlib,importlib.util,json,math,sys,tempfile
from pathlib import Path
sys.dont_write_bytecode=True

def module(name,file):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(file))
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
seq=module('promptclear_sequence_recovery_artifacts','review_moe_workspace_fullmodel_promptclear.py')
cancel=module('promptclear_cancel_recovery_artifacts','test_moe_workspace_cancel_promptclear.py')
recovery=module('promptclear_failure_recovery_artifacts','test_moe_workspace_recovery_promptclear.py')
need,load,sha=seq.need,seq.load,seq.sha
SCENARIOS=('deterministic','async','allocation','allocation-owner','shared-pso','prompt-allocation-first','prompt-allocation-last')
TEMPERATURES=('0','0.7')
CORE=('src/myds_v41.c','src/myds_metal.m','include/myds_internal.h','include/myds_metal.h')
NATIVES=('test_moe_workspace_cancel_promptclear','test_moe_workspace_recovery_promptclear')

def pair_name(s,t):return s+'-t'+t.replace('.','p')
def matrix():return [dict(scenario=s,temperature=t,setting=f)for s in SCENARIOS for t in TEMPERATURES for f in (0,1)]
def write_new(path,value):
    with Path(path).open('x')as stream:json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')

def frozen_plan_gate(protocol,index,small):
    seq.frozen_protocol(protocol)
    plans=[path for path,data in small.items()if index[path]['sha256']==seq.PLAN_SHA and hashlib.sha256(data).hexdigest()==seq.PLAN_SHA and seq.parse(data.decode())==protocol]
    need(len(plans)==1,'Unique archived original plan actual bytes/SHA and protocol JSON object')
    return plans[0]

def coverage_gate(arms,reports):
    expected=[(s,t)for s in SCENARIOS for t in TEMPERATURES]
    need(type(arms)is list and len(arms)==28 and [(r['scenario'],r['temperature'],r['setting'])for r in arms]==[(s,t,f)for s,t in expected for f in (0,1)],'Exact ordered28 original/candidate native arms')
    need(type(reports)is list and len(reports)==14 and [(r['scenario'],r['temperature'])for r in reports]==expected and [r['pair']for r in reports]==[pair_name(s,t)for s,t in expected],'Exact ordered14 scenario/temperature pair reports')
    need([r['label']for r in arms]==[pair_name(s,t)+('-on'if f else'-off')for s,t in expected for f in (0,1)],'Native labels identify all unique original raw directories')
    need(all(r.get('evidence_valid')is True and r.get('correctness_pass')is True for r in reports),'Original controller recorded complete pair readers')
    return expected

def original_top_gate(value,summary):
    need(value.get('schema_version')==1 and value.get('implementation_version')==4 and value.get('plan_sha256')==value.get('recovery_plan_sha256')==seq.PLAN_SHA,'Original top belongs to frozen implementation4 plan')
    need(value.get('evidence_valid')is True and value.get('recovery_gate_pass')is True and value.get('production_adoption')is False and value.get('performance_measured')is False,'Original recovery top completion scope')
    need(value.get('component_review_sha256')==summary['component_review_sha256'] and value.get('restoration')==summary['restoration'],'Original top component/restoration binding')
    return coverage_gate(value['arm_executions'],value['pair_reports'])

def recorded_identity(root):
    before,after,summary=[load(root/n)for n in ('before.json','restored.json','summary.json')]
    need(all(before[k]==after[k]for k in seq.IDENTITY_KEYS) and before['health']==after['health']=={'status':'ok'},'Recorded exact argv/raw inherited environment/nice/config/cwd/log/artifact restoration')
    need(before['binary_sha256']==seq.PRODUCTION_BINARY and before['metallib_sha256']==seq.PRODUCTION_LIBRARY and summary['restoration']['restored']is True and summary['restoration']['pid']==after['pid'] and summary['restoration']['exact_argv_myds_artifacts_config']is True and summary['production_adoption']is False,'Production restore/artifact identity')
    need(sha(root/'production-artifacts/myds')==seq.PRODUCTION_BINARY and sha(root/'production-artifacts/myds.metallib')==seq.PRODUCTION_LIBRARY and sha(root/'original-myds.conf')==before['config_sha256'] and load(root/'effective-env.json')==before['effective_myds_env'],'Original production/config/environment payload bytes')
    env=before['effective_myds_env'];need(len(env)==51 and env.get('MYDS_METAL_CONCURRENT')==env.get('MYDS_PREFIX_CACHE')==env.get('MYDS_PLD')=='1' and env.get(seq.FLAG,'0')==env.get(seq.CLEAR_FLAG,'0')=='0','Original51 effective settings and both candidate defaults off')
    need(summary['status']=='recovery_passed_review_required' and summary['recovery_gate_pass']is True and summary['recovery_executions_completed']==28 and summary['recovery_pair_reports_completed']==14 and summary['performance_measured']is False and summary['recovery_plan_sha256']==summary['plan_sha256']==seq.PLAN_SHA,'Completed28native/14pair frozen v4 recovery')
    return before,after,summary

def steps_gate(steps,events,original_root,candidate,model,requests):
    expected=[];original_root=Path(original_root);candidate=Path(candidate)
    for s in SCENARIOS:
        for t in TEMPERATURES:
            pair=pair_name(s,t);directories=[]
            for f in (0,1):
                label=pair+('-on'if f else'-off');d=original_root/'arms'/label;directories.append(d)
                native=NATIVES[0]if s in ('deterministic','async')else NATIVES[1]
                argv=[candidate/'bin'/native,'--run','--model',model,'--out-dir',d,'--scenario',s,'--temperature',t,'--recover-tokens','8']
                if s in ('deterministic','async'):argv+=['--warm-request',requests['warm'],'--cancel-request',requests['cancel'],'--recover-request',requests['recover'],'--checkpoint','1','--delay-ms','0']
                else:argv+=['--warm-request',requests['warm'],'--failure-request',requests['failure'],'--recover-request',requests['recover']]
                expected.append((label,list(map(str,argv))))
            reader='test_moe_workspace_cancel_promptclear.py'if s in ('deterministic','async')else'test_moe_workspace_recovery_promptclear.py'
            argv=['/usr/bin/python3',candidate/'tools'/reader,'--off',directories[0],'--on',directories[1],'--recover-request',requests['recover'],'--out',original_root/'reports'/(pair+'.json'),'--warm-request',requests['warm']]
            argv+=['--cancel-request',requests['cancel']]if s in ('deterministic','async')else['--failure-request',requests['failure']]
            expected.append((pair+'-reader',list(map(str,argv))))
    need([(s['name'],s['argv'])for s in steps]==expected and all(s['returncode']==0 and type(s['seconds'])in (int,float) and math.isfinite(s['seconds'])and s['seconds']>=0 for s in steps),'All42 successful exact frozen native/reader invocations')
    begins=[e for e in events if e['kind']=='step_begin'];need([(e['name'],e['argv'])for e in begins]==expected,'Actual step events match recorded42 invocations')
    need(events[-1]['kind']=='recovery_decision_complete' and events[-1]['recovery_gate_pass']is True and not any(e['kind']in ('monitor_failure','restoration_failure','window_failure')for e in events),'Complete recovery/competition/restore monitor history')
    return dict(steps_verified=42,native_invocations=28,reader_invocations=14)

def two_flag_environment(actual,original,setting):
    need(actual==dict(original,**{seq.FLAG:str(setting),seq.CLEAR_FLAG:str(setting)}),'All51 original values plus only two explicitly paired flags')
    return True

def byte_totals(results):
    files=phases=bytes_compared=rehashes=0
    for item in results:
        r=item['replay'];need(all(type(v)is int and v>=0 for v in r['raw_bytes_compared'].values()),'Actual full-byte comparison counts');phases+=len(r['raw_bytes_compared']);bytes_compared+=sum(r['raw_bytes_compared'].values())
        for arm in r['arms']:
            for state in arm.values():rehashes+=len(state['witnessed'])
        for phase in r['raw_bytes_compared']:files+=len(r['arms'][0][phase]['witnessed'])
    return dict(pair_readers_replayed=len(results),defined_state_files_rehashed=rehashes,phase_full_byte_comparisons=phases,full_original_files_compared=files,full_original_bytes_compared=bytes_compared)

def review(root,source,artifacts,bindings,model=None):
    root=Path(root);before,after,summary=recorded_identity(root);env=before['effective_myds_env'];top_path=root/'recovery-review.json';top=load(top_path);expected=original_top_gate(top,summary)
    need(top['original_pid']==before['pid'],'Original controller top binds the actual stopped production PID')
    plan=load(root/'protocol.json');seq.frozen_protocol(plan)
    need(load(root/'recovery-protocol.json')==plan and load(root/'recovery-matrix.json')==matrix(),'Exact original28-execution frozen matrix')
    archive=seq.executed_archives(root,source,artifacts,bindings);idx,small=archive['index'],archive['small']
    frozen_plan_gate(plan,idx,small)
    paths=[p for p in small if idx[p]['sha256']==summary['component_review_sha256']];need(len(paths)==1,'Unique exact archived component small top');component=seq.parse(small[paths[0]].decode())
    need(component.get('implementation_version')==4 and component.get('plan_sha256')==seq.PLAN_SHA and component.get('evidence_valid')is True and component.get('component_gate_pass')is True,'Current independently reviewed component prerequisite')
    seq.long_reference_gate(component.get('long_report'),idx,'recovery component prerequisite')
    for m in (seq,cancel,recovery):
        matches=[p for p in idx if p.endswith('/tools/'+Path(m.__file__).name)]
        need(matches and any(idx[p]['sha256']==sha(Path(m.__file__))for p in matches),'Replay uses the exact frozen original reader source: '+Path(m.__file__).name)
    sources=[Path(p).parent.parent for p in idx if p.endswith('/src/myds_v41.c')];need(len(sources)==2,'Two independent candidate/exact-production gold source sets')
    gold=[s for s in sources if idx.get(str(s/'bin/myds'),{}).get('sha256')==seq.PRODUCTION_BINARY and all(idx.get(str(s/k),{}).get('sha256')==v for k,v in seq.GOLD_SOURCE_HASHES.items())];need(len(gold)==1,'All91 exact-production gold source/build inputs')
    candidate=next(s for s in sources if s!=gold[0]);identity={'candidate':{k:idx[str(candidate/k)]['sha256']for k in (*CORE,'bin/myds.metallib',*(str(Path('bin')/n)for n in NATIVES))}}
    seq.component_identity_gate(component,identity,env,summary['model'],summary['model_header_sha256'])
    for n in NATIVES:need(str(candidate/'bin'/n)in idx,'Actual frozen recovery executable: '+n)
    arms,reports=load(root/'recovery-arms.json'),load(root/'recovery-pairs.json');need(arms==top['arm_executions'] and reports==top['pair_reports'],'Original top retains exact arm/pair coverage');coverage_gate(arms,reports)
    manifest=load(bindings);original_root=Path(manifest['original_window']);results=[];facts=[]
    with tempfile.TemporaryDirectory(prefix='promptclear-recovery-original-requests-')as tmp:
        requests={};original_requests={}
        for key,stem in (('warm','pf7k'),('cancel','pf16k'),('failure','pf16k'),('recover','pf7k')):
            wanted=plan['fullmodel']['requests'][stem]['sha256'];paths=[p for p in small if Path(p).name==stem+'.json'and idx[p]['sha256']==wanted];need(len(paths)==1,'Unique exact archived original request '+key)
            original_requests[key]=Path(paths[0]);path=Path(tmp)/(key+'.json');path.write_bytes(small[paths[0]]);requests[key]=path
        event_rows=seq.rows(root/'events.jsonl');invocation=steps_gate(load(root/'steps.json'),event_rows,original_root,candidate,summary['model']['path'],original_requests)
        need([e['pid']for e in event_rows if e['kind']=='production_stop']==[before['pid']]and [e['pid']for e in event_rows if e['kind']=='restoration_launched']==[after['pid']],'One exact production stop and restored launch identity')
        for s,t in expected:
            pair=pair_name(s,t);records=[r for r in arms if(r['scenario'],r['temperature'])==(s,t)];directories=[root/'arms'/(pair+'-'+arm)for arm in ('off','on')];values=[]
            for rec,d,f in zip(records,directories,(0,1)):
                need(rec['raw_directory']==str(original_root/'arms'/rec['label'])and sha(d/'summary.json')==rec['native_summary_sha256'],'Original native directory/summary identity')
                v=load(d/'summary.json');need(v['implementation_version']==4 and v['workspace_mode']==v['prompt_view_clear_mode']==f and v['scenario']==s and math.isclose(v['temperature'],float(t),abs_tol=1e-6),'Actual two-flag native scenario/config identity');values.append(v)
                two_flag_environment(load(root/'arm-environments'/(pair+('-on'if f else'-off')+'.json')),env,f)
            if s in ('deterministic','async'):result=cancel.review(*directories,requests['warm'],requests['cancel'],requests['recover'])
            else:result=recovery.review(*directories,{k:requests[k]for k in ('warm','failure','recover')})
            old=root/'reports'/(pair+'.json');record=next(r for r in reports if r['pair']==pair);prior=load(old)
            need(record['path']==str(original_root/'reports'/(pair+'.json'))and sha(old)==record['sha256'] and prior['evidence_valid']is True and prior['correctness_pass']is True,'Preserved original full pair report')
            need(result['evidence_valid']is True and result['correctness_pass']is True and result['raw_bytes_compared']==prior['raw_bytes_compared'],'Independent original-reader fullraw replay agrees with original report')
            facts.append(dict(pair=pair,scenario=s,temperature=t,observed_progress=[v.get('observed_progress',v.get('failed_moe_progress'))for v in values],thread_flag_progress=[v.get('thread_flag_progress')for v in values],fault_ordinal=[v.get('fault_ordinal')for v in values],injected_kind=[next((a['kind']for a in v.get('allocation_trace',[])if not a['success']),None)for v in values],injected_site=[next((a['site']for a in v.get('allocation_trace',[])if not a['success']),None)for v in values],selected_provider=[v.get('selected_provider')for v in values],prompt_fault_ordinal=[v.get('prompt_fault_ordinal')for v in values],prompt_injected_ordinal=[v.get('prompt_injected_ordinal')for v in values],prompt_injected_hits=[v.get('prompt_injected_hits')for v in values],failed_batch=[v.get('after_moe_batch')for v in values],prompt_old_owners_retained=[v.get('prompt_old_owners_retained')for v in values],prompt_old_views_retained=[v.get('prompt_old_views_retained')for v in values],source_summary_sha256=[r['native_summary_sha256']for r in records],original_pair_report_sha256=record['sha256']))
            results.append(dict(pair=pair,replay=result))
    actual_model=None
    if model is not None:
        model=Path(model);need(model.stat().st_size==summary['model']['bytes']and sha(model,1048576)==summary['model_header_sha256'],'Actual model extent/header prefix');actual_model=dict(path=str(model),bytes=model.stat().st_size,header_sha256=summary['model_header_sha256'])
    return dict(schema_version=1,evidence_valid=True,recovery_gate_pass=True,correctness_pass=True,implementation_version=4,plan_sha256=seq.PLAN_SHA,recovery_plan_sha256=seq.PLAN_SHA,component_review_sha256=summary['component_review_sha256'],performance_measured=False,production_adoption=False,decision='pass_independently_replayed28native14pair_promptclear_recovery_exported_scope',before_pid=before['pid'],restored_pid=after['pid'],recorded_restoration_exact=True,live_production_sampled=False,production_environment=env,model=summary['model'],model_header_sha256=summary['model_header_sha256'],source_and_artifact_archives={k:v for k,v in archive.items()if k not in ('small','index')},logical_frozen_inputs_verified=len(idx),all91_exact_production_gold_source_inputs=True,candidate_identity=identity['candidate'],actual_model=actual_model,invocation=invocation,full_bytes=byte_totals(results),facts=facts,pair_replays=results,original_top=dict(path=str(original_root/'recovery-review.json'),sha256=sha(top_path)),postprocessor_source_sha256=sha(Path(__file__)),limitations=['Injected NULL exercises actual allocator branches; no real process/system-memory-pressure OOM is applied.','A later prompt allocation failure preserves old prompt6/caps and compact curview identities; prior MoE B16384 growth may already be committed. Off may have replaced its CW, so fresh empty curviews are recorded rather than forced equal to warm views.','Async partial state legality is checked separately; only equal deterministic progress permits an intermediate byte oracle. Every complete warm/recovery raw payload is rehashed and compared, including RNG and published/historical snapshots.','Unpublished capacity, interrupted partially updated live cache/carry, allocator padding and persistent disk-cache contents are excluded.','Recorded restoration is verified. Current production PID/health/locks are a separate later read-only live audit. No latency, physical-memory gain or adoption is proven.'])

def small_summary(result,original_top,report_path,original_top_path):
    need(load(Path(original_top_path))==original_top and load(Path(report_path))==result,'Small summary must bind the exact original top and independently written long report')
    if result.get('evidence_valid')is True:need(result.get('recovery_gate_pass')is True and result.get('full_bytes',{}).get('pair_readers_replayed')==14 and result.get('all91_exact_production_gold_source_inputs')is True and result.get('recorded_restoration_exact')is True,'Complete independent source/state/restore replay required for small gate')
    value=copy.deepcopy(original_top)
    for key in ('evidence_valid','recovery_gate_pass','correctness_pass','implementation_version','plan_sha256','recovery_plan_sha256','component_review_sha256','performance_measured','production_adoption','decision'):
        if key in result:value[key]=result[key]
    value.update(independent_fullraw_replay_completed=result.get('evidence_valid')is True,long_report=dict(path=str(Path(report_path).resolve()),sha256=sha(Path(report_path)),bytes=Path(report_path).stat().st_size),original_top=copy.deepcopy(result.get('original_top',dict(path=str(Path(original_top_path).resolve()),sha256=sha(Path(original_top_path))))),independent_review=dict(full_bytes=result.get('full_bytes'),source_and_artifact_archives=result.get('source_and_artifact_archives'),logical_frozen_inputs_verified=result.get('logical_frozen_inputs_verified'),all91_exact_production_gold_source_inputs=result.get('all91_exact_production_gold_source_inputs'),recorded_restoration_exact=result.get('recorded_restoration_exact'),live_production_sampled=False,limitations=result.get('limitations',[])))
    if 'error'in result:value['error']=result['error']
    need(len(json.dumps(value,sort_keys=True,allow_nan=False).encode())<=100*1024,'Small independent recovery prerequisite exceeds100KiB')
    return value

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('root','source-archive','artifacts-archive','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--archive-bindings',type=Path);p.add_argument('--model',type=Path);p.add_argument('--summary-out',type=Path);a=p.parse_args()
    need(a.root.is_absolute()and a.out.is_absolute()and(not a.summary_out or a.summary_out.is_absolute()),'Absolute original root and new report paths required')
    try:value=review(a.root,a.source_archive,a.artifacts_archive,a.archive_bindings or a.source_archive.parent/'archive-bindings.json',a.model)
    except Exception as exc:value=dict(schema_version=1,evidence_valid=False,recovery_gate_pass=False,correctness_pass=False,implementation_version=4,plan_sha256=seq.PLAN_SHA,recovery_plan_sha256=seq.PLAN_SHA,performance_measured=False,production_adoption=False,decision='reject_incomplete_promptclear_recovery_artifact_review',error=repr(exc))
    write_new(a.out,value)
    if a.summary_out:write_new(a.summary_out,small_summary(value,load(a.root/'recovery-review.json'),a.out,a.root/'recovery-review.json'))
    print(json.dumps({k:v for k,v in value.items()if k!='pair_replays'},indent=2,sort_keys=True,allow_nan=False));return 0 if value.get('evidence_valid')else 1
if __name__=='__main__':raise SystemExit(main())
