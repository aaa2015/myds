#!/usr/bin/env python3
"""Bounded combined recovery acceptance controls; argv fixtures never execute."""
import copy,importlib.util,json,tempfile,types
from pathlib import Path
def module(name):
    s=importlib.util.spec_from_file_location('combined_controls_'+name,Path(__file__).with_name(name+'.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
r=module('review_moe_workspace_recovery_combined_promptclear')
thin=module('run_moe_workspace_recovery_continue_promptclear')

def run():
    checks=[]
    def test(name,fn,reject=False):
        try:fn();ok=not reject
        except (ValueError,KeyError,TypeError,OSError):ok=reject
        checks.append(dict(name=name,passed=ok));r.need(ok,name)
    records=[dict(x,label=r.base.pair_name(x['scenario'],x['temperature'])+('-on'if x['setting']else'-off'))for x in r.MATRIX]
    test('exact10plus18_once',lambda:r.complete_coverage(records[:10],records[10:]))
    for key,value in [('scenario','allocation'),('temperature','0'),('setting',True),('label','allocation-t0-off')]:
        q=copy.deepcopy(records[10:]);q[-1][key]=value
        test('reject_suffix_'+key,lambda q=q:r.complete_coverage(records[:10],q),True)
    test('reject_missing_last_native',lambda:r.complete_coverage(records[:10],records[10:-1]),True)
    q=copy.deepcopy(records[10:]);q[-1]=q[0];test('reject_duplicate_replacing_stochastic_last',lambda:r.complete_coverage(records[:10],q),True)
    q=copy.deepcopy(records[10:]);q[0],q[1]=q[1],q[0];test('reject_suffix_order_changed',lambda:r.complete_coverage(records[:10],q),True)
    test('reject_boundary8plus20',lambda:r.complete_coverage(records[:8],records[8:]),True)
    test('reject_full28_as_new_suffix',lambda:r.complete_coverage(records[:10],records),True)
    w=object.__new__(thin.RecoveryContinuation);w.a=types.SimpleNamespace(stage=Path('/CPU-fixture-stage'),model=Path('/CPU-fixture-model'))
    w.requests={k:Path('/CPU-fixture-inputs')/('pf16k.json'if k in ('failure','cancel')else'pf7k.json')for k in ('warm','failure','cancel','recover')};root=Path('/CPU-fixture-window')
    expected=[]
    for i in range(10,28,2):
        x=records[i];pair=r.base.pair_name(x['scenario'],x['temperature']);ds=[]
        for f in (0,1):
            d=root/'arms'/(pair+('-on'if f else'-off'));ds.append(d);expected.append((pair+('-on'if f else'-off'),list(map(str,w.native_command(x['scenario'],x['temperature'],d)))))
        expected.append((pair+'-reader',list(map(str,w.reader_command(x['scenario'],*ds,root/'reports'/(pair+'.json'))))))
    test('exact27suffix_actual_thin_controller_argv',lambda:r.need(r.commands(w.a.stage,root,w.requests,str(w.a.model),True)[15:]==expected,'actual thin interface'))
    windows=[]
    for i,(n,p,status)in enumerate(((10,5,'failed'),(18,9,'continuation_passed_review_required'))):
        d=dict(path='/CPU-fixture-archive-'+str(i),sha256='a'*64,bytes=1)
        windows.append(dict(window='/CPU-fixture-window-'+str(i),native_invocations=n,pair_readers_replayed=p,status=status,input_map_sha256='b'*64,
            restoration=dict(restored=True,exact_argv_myds_artifacts_config=True),source_archive=d,artifacts_archive=d,archive_bindings=d))
    test('exact_separate_source_windows_metadata',lambda:r.source_windows_gate(windows))
    for field,value in [('status','complete'),('window',windows[0]['window']),('native_invocations',28),('pair_readers_replayed',14),('input_map_sha256','wrong')]:
        q=copy.deepcopy(windows);q[1][field]=value;test('reject_source_window_'+field,lambda q=q:r.source_windows_gate(q),True)
    q=copy.deepcopy(windows);q[0]['restoration']['restored']=False;test('reject_original_failed_window_not_restored',lambda:r.source_windows_gate(q),True)
    q=copy.deepcopy(windows);q[1]['source_archive']['bytes']=True;test('reject_boolean_archive_extent',lambda:r.source_windows_gate(q),True)
    q=copy.deepcopy(windows);q[1]['source_archive']['path']='relative';test('reject_relative_archive_reference',lambda:r.source_windows_gate(q),True)
    with tempfile.TemporaryDirectory(prefix='combined-recovery-reader-CPU-')as tmp:
        long=Path(tmp)/'long.json';value=dict(evidence_valid=True,recovery_gate_pass=True,correctness_pass=True,implementation_version=4,plan_sha256=r.seq.PLAN_SHA,recovery_plan_sha256=r.seq.PLAN_SHA,performance_measured=False,production_adoption=False,native_invocations=28,pair_readers_replayed=14,source_windows=windows,
            full_bytes=dict(pair_readers_replayed=14),source_and_artifact_archives=[],logical_frozen_inputs_verified=[335,350],limitations=[],arm_executions=records,
            pair_reports=[dict(pair=r.base.pair_name(s,t),scenario=s,temperature=t,evidence_valid=True,correctness_pass=True)for s in r.base.SCENARIOS for t in r.base.TEMPERATURES])
        def small(v):long.write_text(json.dumps(v));return r.small_summary(v,long)
        test('small_binds_own_exact_long_and_all28records',lambda:r.need(small(value)['long_report']['sha256']==r.sha(long)and len(small(value)['arm_executions'])==28,'small binding'))
        for key,new in [('evidence_valid',False),('recovery_gate_pass',False),('correctness_pass',False),('implementation_version',3),('plan_sha256','other'),('performance_measured',True),('production_adoption',True),('native_invocations',27),('pair_readers_replayed',13)]:
            q=copy.deepcopy(value);q[key]=new;test('reject_small_'+key,lambda q=q:small(q),True)
        q=copy.deepcopy(value);q['full_bytes']['pair_readers_replayed']=13;test('reject_incomplete_fullraw_replay_count',lambda:small(q),True)
        q=copy.deepcopy(value);q['pair_reports'][-1]['correctness_pass']=False;test('reject_small_failed_pair_decision',lambda:small(q),True)
        long.write_text(json.dumps(value));test('reject_modified_long_content',lambda:r.small_summary(dict(value,correctness_pass=False),long),True)
    return dict(schema_version=1,scope='CPU_two_window_recovery_reader_coverage_identity_controls',passed=all(x['passed']for x in checks),controls_count=len(checks),controls=checks,fake_argv_executed=False,gpu_run=False,model_run=False)
if __name__=='__main__':print(json.dumps(run(),indent=2,sort_keys=True))
