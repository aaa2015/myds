#!/usr/bin/env python3
"""Bounded CPU controls; fake paths/argv are data and are never executed."""
import copy,importlib.util,json,sys,tempfile,types
from pathlib import Path
sys.dont_write_bytecode=True

def module(name,file):
    s=importlib.util.spec_from_file_location(name,Path(__file__).with_name(file));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
r=module('promptclear_recovery_postreader_controls','review_moe_workspace_recovery_promptclear_artifacts.py')
controller=module('promptclear_recovery_controller_data_only','run_moe_workspace_recovery_window_promptclear.py')

def run():
    checks=[]
    def test(name,fn,reject=False):
        try:fn();ok=not reject
        except (ValueError,KeyError,TypeError,OSError):ok=reject
        checks.append(dict(name=name,passed=ok));r.need(ok,'CPU control '+name)
    plan_path=Path(__file__).parent.parent/'docs/benchmarks/2026-10-24-moe-phys-cause/experiment-plan.json'
    original=plan_path.read_bytes();plan=r.seq.parse(original.decode());canonical=json.dumps(plan,indent=2,sort_keys=True)+'\n'
    index={'/CPU-original-plan.json':dict(sha256=r.sha(plan_path),bytes=len(original))};small={'/CPU-original-plan.json':original}
    test('canonical_protocol_bytes_differ_but_object_and_original_plan_bind',lambda:r.need(r.hashlib.sha256(canonical.encode()).hexdigest()!=r.seq.PLAN_SHA and r.frozen_plan_gate(r.seq.parse(canonical),index,small)=='/CPU-original-plan.json','protocol serialization'))
    bad=copy.deepcopy(plan);bad['fullmodel']['max_new_tokens']=8;test('reject_protocol_numeric_scope_changed',lambda:r.frozen_plan_gate(bad,index,small),True)
    bad=dict(small);bad['/CPU-original-plan.json']=original+b'\n';test('reject_original_plan_byte_SHA_changed_despite_equal_JSON',lambda:r.frozen_plan_gate(plan,index,bad),True)
    duplicate=dict(small,**{'/CPU-second-plan.json':original});di=dict(index,**{'/CPU-second-plan.json':dict(sha256=r.seq.PLAN_SHA,bytes=len(original))});test('reject_duplicate_original_plan',lambda:r.frozen_plan_gate(plan,di,duplicate),True)
    arms=[];pairs=[]
    for s in r.SCENARIOS:
        for t in r.TEMPERATURES:
            pair=r.pair_name(s,t);pairs.append(dict(pair=pair,scenario=s,temperature=t,evidence_valid=True,correctness_pass=True))
            for f in (0,1):arms.append(dict(label=pair+('-on'if f else'-off'),scenario=s,temperature=t,setting=f))
    test('complete28native14pair_matrix',lambda:r.coverage_gate(arms,pairs))
    test('reject_missing_native',lambda:r.coverage_gate(arms[:-1],pairs),True)
    bad=copy.deepcopy(arms);bad[-1]=bad[0];test('reject_duplicate_native_replacing_last',lambda:r.coverage_gate(bad,pairs),True)
    bad=copy.deepcopy(pairs);bad[-1]['temperature']='0';test('reject_lost_stochastic_prompt_last_pair',lambda:r.coverage_gate(arms,bad),True)
    top=dict(schema_version=1,implementation_version=4,plan_sha256=r.seq.PLAN_SHA,recovery_plan_sha256=r.seq.PLAN_SHA,evidence_valid=True,recovery_gate_pass=True,production_adoption=False,performance_measured=False,component_review_sha256='synthetic-component-sha',restoration=dict(restored=True,pid=123),arm_executions=arms,pair_reports=pairs,original_pid=122,other_original_field='preserved')
    summary=dict(component_review_sha256=top['component_review_sha256'],restoration=top['restoration'])
    test('original_top_plan_counts_component_restore',lambda:r.original_top_gate(top,summary))
    for field,value in [('implementation_version',3),('recovery_plan_sha256','fde7-old-plan'),('component_review_sha256','other-component'),('performance_measured',True)]:
        q=copy.deepcopy(top);q[field]=value;test('reject_original_top_'+field,lambda q=q:r.original_top_gate(q,summary),True)
    env={'MYDS_PLD':'1','MYDS_PREFIX_CACHE':'1'}
    test('preserve_original_env_plus_two_flags',lambda:r.two_flag_environment(dict(env,MYDS_PREFILL_MOE_WORKSPACE='1',MYDS_PREFILL_PROMPT_VIEW_CLEAR='1'),env,1))
    test('reject_mismatched_clear_env',lambda:r.two_flag_environment(dict(env,MYDS_PREFILL_MOE_WORKSPACE='1',MYDS_PREFILL_PROMPT_VIEW_CLEAR='0'),env,1),True)
    test('reject_mutated_production_env',lambda:r.two_flag_environment(dict(env,MYDS_PLD='0',MYDS_PREFILL_MOE_WORKSPACE='1',MYDS_PREFILL_PROMPT_VIEW_CLEAR='1'),env,1),True)
    w=object.__new__(controller.RecoveryWindow);w.a=types.SimpleNamespace(stage=Path('/CPU-fixture-candidate'),model=Path('/CPU-fixture-model'))
    w.requests={k:Path('/CPU-fixture-inputs')/(('pf16k'if k in ('failure','cancel')else'pf7k')+'.json')for k in ('warm','failure','cancel','recover')}
    original=Path('/CPU-fixture-window');steps=[]
    for s in r.SCENARIOS:
        for t in r.TEMPERATURES:
            pair=r.pair_name(s,t);ds=[]
            for f in (0,1):
                label=pair+('-on'if f else'-off');d=original/'arms'/label;ds.append(d);steps.append(dict(name=label,argv=list(map(str,w.native_command(s,t,d))),returncode=0,seconds=0.001))
            steps.append(dict(name=pair+'-reader',argv=list(map(str,w.reader_command(s,*ds,original/'reports'/(pair+'.json')))),returncode=0,seconds=.001))
    events=[dict(kind='step_begin',name=s['name'],argv=s['argv'])for s in steps]+[dict(kind='recovery_decision_complete',recovery_gate_pass=True)]
    def inv(s=steps,e=events):return r.steps_gate(s,e,original,w.a.stage,w.a.model,w.requests)
    test('actual_controller42argv_matches_independent_protocol',inv)
    bad=copy.deepcopy(steps);bad[0]['argv'][bad[0]['argv'].index('--checkpoint')+1]='2';test('reject_changed_checkpoint',lambda:inv(bad),True)
    bad=copy.deepcopy(steps);bad[-2]['returncode']=1;test('reject_failed_native_exit',lambda:inv(bad),True)
    bad=copy.deepcopy(steps);bad[-1]['argv'][1]='/different-reader.py';test('reject_different_pair_reader',lambda:inv(bad),True)
    bad=copy.deepcopy(events);bad.append(dict(kind='monitor_failure'));test('reject_monitor_failure',lambda:inv(e=bad),True)
    test('reject_missing_reader_invocation',lambda:inv(steps[:-1]),True)
    partial=dict(arms=[{'before':dict(witnessed={'a':{},'b':{}}),'recovery':dict(witnessed={'a':{},'b':{},'c':{}})}]*2,raw_bytes_compared={'before':8,'recovery':16})
    totals=r.byte_totals([dict(replay=partial)])
    test('independent_full_byte_totals',lambda:r.need(totals==dict(pair_readers_replayed=1,defined_state_files_rehashed=10,phase_full_byte_comparisons=2,full_original_files_compared=5,full_original_bytes_compared=24),'counts'))
    bad=copy.deepcopy(partial);bad['raw_bytes_compared']['before']=True;test('reject_boolean_comparison_count',lambda:r.byte_totals([dict(replay=bad)]),True)
    with tempfile.TemporaryDirectory(prefix='promptclear-smalltop-CPU-')as t:
        q=Path(t);original_file=q/'original-top.json';report=q/'long.json';original_file.write_text(json.dumps(top));before=r.sha(original_file)
        result=dict(evidence_valid=True,recovery_gate_pass=True,correctness_pass=True,implementation_version=4,plan_sha256=r.seq.PLAN_SHA,recovery_plan_sha256=r.seq.PLAN_SHA,component_review_sha256=top['component_review_sha256'],performance_measured=False,production_adoption=False,decision='independently_replayed',full_bytes=dict(pair_readers_replayed=14),all91_exact_production_gold_source_inputs=True,recorded_restoration_exact=True)
        report.write_text(json.dumps(result));small=r.small_summary(result,top,report,original_file)
        test('smalltop_preserves_original28_and14_fields',lambda:r.need(small['arm_executions']==top['arm_executions']and small['pair_reports']==top['pair_reports']and small['other_original_field']==top['other_original_field']and small['long_report']==dict(path=str(report.resolve()),sha256=r.sha(report),bytes=report.stat().st_size)and small['original_top']['sha256']==before,'small binding'))
        test('smalltop_does_not_mutate_original_top',lambda:r.need(r.sha(original_file)==before,'original top mutation'))
        fail=dict(evidence_valid=False,recovery_gate_pass=False,correctness_pass=False,performance_measured=False,production_adoption=False,error='CPU rejection');report.write_text(json.dumps(fail));small=r.small_summary(fail,top,report,original_file)
        test('rejecting_long_cannot_inherit_controller_true_gate',lambda:r.need(small['evidence_valid']is False and small['recovery_gate_pass']is False and small['independent_fullraw_replay_completed']is False,'self report leaked'))
        report.write_text(json.dumps(result));bad=copy.deepcopy(result);bad['full_bytes']['pair_readers_replayed']=13;report.write_text(json.dumps(bad));test('reject_incomplete_independent_replay_smalltop',lambda:r.small_summary(bad,top,report,original_file),True)
        report.write_text(json.dumps(result));bad=copy.deepcopy(top);bad['other_original_field']='changed';test('reject_original_top_content_mismatch',lambda:r.small_summary(result,bad,report,original_file),True)
        test('reject_long_content_mismatch',lambda:r.small_summary(dict(result,decision='changed'),top,report,original_file),True)
    return dict(schema_version=1,scope='bounded_CPU_promptclear_recovery_postprocessor_controls',gpu_run=False,model_run=False,fake_argv_executed=False,controls=checks,passed=all(x['passed']for x in checks))
if __name__=='__main__':print(json.dumps(run(),indent=2,sort_keys=True))
