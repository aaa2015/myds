#!/usr/bin/env python3
"""Strict prompt-lifetime branch controls; no model, Metal, or service calls."""
import copy,importlib.util,json
from pathlib import Path
spec=importlib.util.spec_from_file_location('recovery_promptclear_v2',Path(__file__).with_name('test_moe_workspace_recovery_promptclear_v2.py'))
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)

def snapshot(cap,mask,first,cw,*,views=True):
    sizes=[cap*81920,cap*96,cap*2048,cap*mask,cap*24576,cap*24576]
    owners=[dict(wrapper_id=first+i,resource_id=first+100+i,offset=0,bytes=n,is_view=0)for i,n in enumerate(sizes)]
    v=[dict(wrapper_id=first+200+i,resource_id=o['resource_id'],offset=0,bytes=min(64,o['bytes']),is_view=1)for i,o in enumerate(owners)] if views else [dict(wrapper_id=0,resource_id=0,offset=0,bytes=0,is_view=0)for _ in range(6)]
    return dict(cw_id=cw,cap=cap,mask_bytes=mask,total_bytes=sum(sizes),owners=owners,views=v)

def run():
    checks=list(r.self_test()['controls'])
    def test(name,fn,reject=False):
        try:fn();ok=not reject
        except (r.common.Rejected,ValueError,KeyError,TypeError):ok=reject
        checks.append(dict(name=name,passed=ok));r.need(ok,name)
    warm=snapshot(7168,112,10,1);failed=snapshot(15360,240,1000,0,views=False);recovered=copy.deepcopy(failed);recovered['cw_id']=2
    recovered['views']=snapshot(15360,240,1000,2)['views']
    test('old_cap_predicate_reproduces_false_rejection',lambda:r.need(not(warm['cap']==recovered['cap']==7168),'old exact predicate'))
    for scenario in ('allocation','allocation-owner'):
        test('legacy_'+scenario+'_already_grown_prompt_retained',lambda scenario=scenario:r.prompt_failure_transition_gate(warm,failed,recovered,0,scenario))
        test('compact_'+scenario+'_early_failure_preserves_prompt',lambda scenario=scenario:r.prompt_failure_transition_gate(warm,warm,warm,1,scenario))
    for key,value in [('cap',7168),('mask_bytes',112),('total_bytes',0),('cw_id',1)]:
        bad=copy.deepcopy(failed);bad[key]=value
        test('reject_legacy_failed_'+key,lambda bad=bad:r.prompt_failure_transition_gate(warm,bad,recovered,0,'allocation'),True)
    bad=copy.deepcopy(recovered);bad['owners'][0]['resource_id']+=1000
    test('reject_lost_recovered_legacy_owner',lambda:r.prompt_failure_transition_gate(warm,failed,bad,0,'allocation'),True)
    bad=copy.deepcopy(failed);bad['views'][0]=dict(warm['views'][0])
    test('reject_nonempty_view_in_absent_legacy_CW',lambda:r.prompt_failure_transition_gate(warm,bad,recovered,0,'allocation'),True)
    for field in ('owners','views'):
        bad=copy.deepcopy(warm);bad[field][0]['resource_id']+=1
        test('reject_compact_early_failure_changed_'+field,lambda bad=bad:r.prompt_failure_transition_gate(warm,bad,warm,1,'allocation'),True)
    test('reject_candidate_silent_prompt_growth',lambda:r.prompt_failure_transition_gate(warm,failed,recovered,1,'allocation'),True)
    test('compact_shared_PSO_early_failure_retains_prompt',lambda:r.prompt_failure_transition_gate(warm,warm,warm,1,'shared-pso'))
    empty=dict(cw_id=0,cap=0,mask_bytes=0,total_bytes=0,owners=[dict(wrapper_id=0,resource_id=0,offset=0,bytes=0,is_view=0)for _ in range(6)],views=[dict(wrapper_id=0,resource_id=0,offset=0,bytes=0,is_view=0)for _ in range(6)])
    test('legacy_shared_PSO_runtime_failure_recreates_prompt',lambda:r.prompt_failure_transition_gate(warm,empty,warm,0,'shared-pso'))
    test('reject_legacy_shared_PSO_not_cleaned',lambda:r.prompt_failure_transition_gate(warm,warm,warm,0,'shared-pso'),True)
    for scenario in ('prompt-allocation-first','prompt-allocation-last'):
        test(scenario+'_compact_prompt_transaction_preserves',lambda scenario=scenario:r.prompt_failure_transition_gate(warm,warm,warm,1,scenario))
        legacy=copy.deepcopy(warm);legacy['cw_id']=2;legacy['views']=[dict(v,resource_id=0,bytes=0)for v in legacy['views']]
        test(scenario+'_legacy_prompt_transaction_preserves',lambda scenario=scenario:r.prompt_failure_transition_gate(warm,legacy,warm,0,scenario))
        test('reject_'+scenario+'_partial_publication',lambda scenario=scenario:r.prompt_failure_transition_gate(warm,failed,recovered,1,scenario),True)
    bad=copy.deepcopy(recovered);bad['owners'][5]['bytes']-=1;bad['total_bytes']-=1
    test('reject_plausible_cap_wrong_owner_length',lambda:r.prompt_failure_transition_gate(warm,failed,bad,0,'allocation'),True)
    bad=copy.deepcopy(recovered);bad['views'][0]['offset']=bad['owners'][0]['bytes']
    test('reject_view_past_owner_extent',lambda:r.prompt_failure_transition_gate(warm,failed,bad,0,'allocation'),True)
    return dict(schema_version=1,scope='strict_actual_source_order_prompt_capacity_v2_CPU_controls',passed=all(x['passed']for x in checks),controls_count=len(checks),controls=checks,gpu_run=False,model_run=False)

if __name__=='__main__':print(json.dumps(run(),indent=2,sort_keys=True))
