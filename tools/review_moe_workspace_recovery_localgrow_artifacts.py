#!/usr/bin/env python3
"""CPU-only replay/binding review for completed localgrow recovery evidence.

Uses the frozen pair readers after independently verifying their executed source
SHA, all archived source/bin bytes, all51 inherited settings, and restoration.
No inference, GPU, service mutation, source extraction or raw changes are made.
"""
import argparse,hashlib,importlib.util,json,math,sys,tempfile
from pathlib import Path
sys.dont_write_bytecode=True

def module(name,file):
 s=importlib.util.spec_from_file_location(name,Path(__file__).with_name(file));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
seq=module('localgrow_sequence_recovery_artifacts','review_moe_workspace_fullmodel_localgrow.py')
cancel=module('localgrow_cancel_recovery_artifacts','test_moe_workspace_cancel_localgrow.py')
recovery=module('localgrow_failure_recovery_artifacts','test_moe_workspace_recovery_localgrow.py')
need,load,sha=seq.need,seq.load,seq.sha
SCENARIOS=('deterministic','async','allocation','allocation-owner','shared-pso')
TEMPERATURES=('0','0.7')
def name(s,t):return s+'-t'+t.replace('.','p')

def review(root,source,artifacts,bindings,model=None):
 root=Path(root);before,after,summary=[load(root/n)for n in ('before.json','restored.json','summary.json')]
 need(all(before[k]==after[k]for k in seq.IDENTITY_KEYS) and before['health']==after['health']=={'status':'ok'},'Recorded exact original argv/env/nice/config/cwd/log/artifacts restored')
 need(before['binary_sha256']==seq.PRODUCTION_BINARY and before['metallib_sha256']==seq.PRODUCTION_LIBRARY and summary['restoration']['restored'] is True and summary['restoration']['pid']==after['pid'] and summary['restoration']['exact_argv_myds_artifacts_config'] is True and summary['production_adoption'] is False,'Production artifact/restore identity')
 need(sha(root/'production-artifacts/myds')==seq.PRODUCTION_BINARY and sha(root/'production-artifacts/myds.metallib')==seq.PRODUCTION_LIBRARY and sha(root/'original-myds.conf')==before['config_sha256'] and load(root/'effective-env.json')==before['effective_myds_env'],'Original recovery payload/environment bytes')
 env=before['effective_myds_env'];need(len(env)==51 and env.get('MYDS_METAL_CONCURRENT')==env.get('MYDS_PREFIX_CACHE')==env.get('MYDS_PLD')=='1' and env.get(seq.FLAG,'0')=='0','Original51 production settings/defaultoff')
 need(summary['status']=='recovery_passed_review_required' and summary['recovery_gate_pass'] is True and summary['recovery_executions_completed']==20 and summary['recovery_pair_reports_completed']==10 and summary['performance_measured'] is False and summary['recovery_plan_sha256']==seq.PLAN_SHA,'Completed20native/10pair frozen v2 recovery')
 plan=load(root/'protocol.json');seq.frozen_protocol(plan);need(summary['plan_sha256']==seq.PLAN_SHA,'Current execution plan SHA')
 archive=seq.executed_archives(root,source,artifacts,bindings);idx=archive['index'];small=archive['small']
 execution_plans=[p for p,data in small.items()if idx[p]['sha256']==seq.PLAN_SHA and seq.parse(data.decode())==plan];need(len(execution_plans)==1,'Unique complete archived frozen execution plan')
 need(load(root/'recovery-protocol.json')==plan and load(root/'recovery-matrix.json')==[dict(scenario=s,temperature=t,setting=f)for s in SCENARIOS for t in TEMPERATURES for f in (0,1)],'Exact20 original frozen recovery matrix')
 proofs=[seq.parse(data.decode())for p,data in small.items()if idx[p]['sha256']==summary['component_review_sha256']];need(len(proofs)==1 and proofs[0].get('implementation_version')==3 and proofs[0].get('evidence_valid')is True and proofs[0].get('component_gate_pass')is True,'Actual archived impl3 component prerequisite')
 for m in (seq,cancel,recovery):
  original=[p for p in idx if p.endswith('/tools/'+Path(m.__file__).name)]
  need(original and any(idx[p]['sha256']==sha(Path(m.__file__))for p in original),'Replay uses original frozen reader source: '+Path(m.__file__).name)
 sources=[p for p in idx if p.endswith('/src/myds_v41.c')];need(len(sources)==2,'Independent exact production gold/candidate source sets')
 stages=[Path(p).parent.parent for p in sources];gold=[p for p in stages if idx.get(str(p/'bin/myds'),{}).get('sha256')==seq.PRODUCTION_BINARY and all(idx.get(str(p/k),{}).get('sha256')==v for k,v in seq.GOLD_SOURCE_HASHES.items())];need(len(gold)==1,'All91 exact-production gold source inputs')
 candidate=next(p for p in stages if p!=gold[0]);need(idx[str(candidate/'bin/myds.metallib')]['sha256']==seq.PRODUCTION_LIBRARY,'Same arithmetic production Metal library')
 for native in ('test_moe_workspace_cancel_localgrow','test_moe_workspace_recovery_localgrow'):
  need(str(candidate/'bin'/native)in idx,'Frozen actual recovery executable')
 reports=load(root/'recovery-pairs.json');arms=load(root/'recovery-arms.json');expected=[(s,t)for s in SCENARIOS for t in TEMPERATURES]
 need([(r['scenario'],r['temperature'])for r in reports]==expected and len(arms)==20,'Exact frozen scenario/temperature/arm matrix')
 need([(r['scenario'],r['temperature'],r['setting'])for r in arms]==[(s,t,f)for s,t in expected for f in (0,1)],'Exact ordered20original/candidate arms')
 events=seq.rows(root/'events.jsonl');need(events[-1]['kind']=='recovery_decision_complete' and not any(e['kind']in ('monitor_failure','restoration_failure','window_failure')for e in events),'Complete recovery decision/monitor event')
 results=[];facts=[]
 with tempfile.TemporaryDirectory(prefix='localgrow-recovery-original-requests-')as tmp:
  requests={}
  for key,request in (('warm','pf7k'),('cancel','pf16k'),('failure','pf16k'),('recover','pf7k')):
   expected_sha=plan['fullmodel']['requests'][request]['sha256'];paths=[p for p in small if Path(p).name==request+'.json'and idx[p]['sha256']==expected_sha];need(len(paths)==1,'Unique frozen request '+key);path=Path(tmp)/(key+'.json');path.write_bytes(small[paths[0]]);requests[key]=path
  for s,t in expected:
   pair=name(s,t);records=[r for r in arms if (r['scenario'],r['temperature'])==(s,t)];directories=[root/'arms'/(pair+'-'+arm)for arm in ('off','on')]
   values=[]
   for record,d,f in zip(records,directories,(0,1)):
    need(sha(d/'summary.json')==record['native_summary_sha256'],'Original native summary SHA');value=load(d/'summary.json');need(value['workspace_mode']==f and math.isclose(value['temperature'],float(t),abs_tol=1e-6),'Native arm identity');values.append(value)
    need(load(root/'arm-environments'/(pair+('-on'if f else'-off')+'.json'))==dict(env,**{seq.FLAG:str(f)}),'All51 settings plus only explicit candidate flag')
   if s in ('deterministic','async'):result=cancel.review(*directories,requests['warm'],requests['cancel'],requests['recover'])
   else:result=recovery.review(*directories,{k:requests[k]for k in ('warm','failure','recover')})
   previous=root/'reports'/(pair+'.json');record=next(r for r in reports if r['pair']==pair);need(sha(previous)==record['sha256'] and load(previous)['correctness_pass'] is True,'Preserved original pair report')
   need(result['evidence_valid']is True and result['correctness_pass']is True,'Independent frozen-reader fullraw replay')
   facts.append(dict(pair=pair,scenario=s,temperature=t,observed_progress=[v.get('observed_progress',v.get('failed_moe_progress'))for v in values],thread_flag_progress=[v.get('thread_flag_progress')for v in values],fault_ordinal=[v.get('fault_ordinal')for v in values],injected_kind=[next((a['kind']for a in v.get('allocation_trace',[])if not a['success']),None)for v in values],injected_site=[next((a['site']for a in v.get('allocation_trace',[])if not a['success']),None)for v in values],request_scope='same-context7k warm ->16k interruption/failure ->7k recovery8, no RNG reset',source_summary_sha256=[r['native_summary_sha256']for r in records],original_pair_report_sha256=record['sha256']))
   results.append(dict(pair=pair,replay=result))
 actual_model=None
 if model is not None:
  model=Path(model);need(model.stat().st_size==summary['model']['bytes'] and sha(model,1048576)==summary['model_header_sha256'],'Actual model stat extent/header prefix');actual_model=dict(path=str(model),bytes=model.stat().st_size,header_sha256=summary['model_header_sha256'])
 return dict(schema_version=1,evidence_valid=True,recovery_gate_pass=True,correctness_pass=True,implementation_version=3,recovery_plan_sha256=seq.PLAN_SHA,performance_measured=False,production_adoption=False,decision='pass_replayed20native10pair_localgrow_recovery_exported_scope',before_pid=before['pid'],restored_pid=after['pid'],recorded_restoration_exact=True,live_production_sampled=False,production_environment=env,source_and_artifact_archives={k:v for k,v in archive.items()if k not in ('small','index')},logical_frozen_inputs_verified=len(idx),all91_exact_production_gold_source_inputs=True,actual_model=actual_model,facts=facts,pair_replays=results,limitations=['Injected NULL exercises actual allocator branches but is not real system-memory-pressure OOM.','Async intermediate states are validated separately; differing observed progress is not forced byte-equal. Deterministic equal-progress defined state and all complete warm/recovery raw bytes are compared.','Unpublished capacity, partially updated live cache after interruption and persistent disk-cache contents are excluded. Published snapshots and historical Engram use each published position.','Recorded restoration is verified. Current live service identity/lock are a separate read-only snapshot. No performance is measured.'])

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('root','source-archive','artifacts-archive','out'):p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--archive-bindings',type=Path);p.add_argument('--model',type=Path);a=p.parse_args()
 try:v=review(a.root,a.source_archive,a.artifacts_archive,a.archive_bindings or a.source_archive.parent/'archive-bindings.json',a.model)
 except Exception as e:v=dict(evidence_valid=False,recovery_gate_pass=False,correctness_pass=False,performance_measured=False,decision='reject_incomplete_recovery_artifact_review',error=repr(e))
 with a.out.open('x')as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
 print(json.dumps({k:x for k,x in v.items()if k!='pair_replays'},indent=2,sort_keys=True));return 0 if v.get('evidence_valid')else 1
if __name__=='__main__':raise SystemExit(main())
