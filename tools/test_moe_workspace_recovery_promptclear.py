#!/usr/bin/env python3
"""Independent CPU gate: successful real context -> injected prefill failure -> recovery.

This imports only raw-byte/hash/state verification from the cancellation reader.
No GPU or service operations. Failed-engine workspace retention/release is recorded,
not forced to be identical between original and transactional factory contracts.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('moe_cancel_reader', Path(__file__).with_name('test_moe_workspace_cancel_promptclear.py'))
common = importlib.util.module_from_spec(spec); spec.loader.exec_module(common)
need = common.need


def metadata_gate(s):
    need(s.get('schema_version') == 1 and s.get('probe') == 'same_context_moe_workspace_failure_recovery_promptclear' and
         s.get('status') == 'complete' and s.get('fullmodel_recovery_within_arm_completed') is True, 'Complete real-context failure/recovery schema')
    need(s.get('scenario') in ('allocation','allocation-owner','shared-pso','prompt-allocation-first','prompt-allocation-last') and s.get('workspace_mode') in (0, 1), 'Actual fault scope')
    need(s.get('implementation_version')==4 and s.get('prompt_view_clear_mode')==s.get('workspace_mode'),'Actual v4 two-flag selection')
    need(s.get('same_context') is True and s.get('rng_manually_reset') is False and
         s.get('extra_gpu_waits_inserted') == 0 and s.get('performance_measured') is False and s.get('real_oom') is False, 'Simulated fault, same context, no reset/wait')
    need(s.get('seed') == 42 and s.get('recovery_tokens') in (8, 32) and
         (s.get('temperature') == 0 or abs(s.get('temperature', -100) - .7) < 1e-6), 'Frozen sampling')
    need(s.get('vocab_size', 0) > 0 and s.get('snapshot_slots') == 44, 'Real state shapes')
    need(s.get('warm_generation_ok') is True and s.get('warm_state_exported') is True and
         s.get('warm_output_tokens') == s['recovery_tokens'] and s.get('old_workspace_present_before_failure') is True and
         s.get('old_capacity', 0) > 0 and s.get('old_moe_batch', 0) >= s['old_capacity'], 'Actual successful old context/workspace')
    need(s.get('failure_api_ok') is False and s.get('failure_stop_reason') == 'error' and
         s.get('failure_output_tokens') == 0 and s.get('failure_gate_pass') is True and s.get('failure_state_exported') is True and
         s.get('failure_prompt_tokens', 0) > s.get('warm_prompt_tokens', 0) > 1, 'Actual longer prefill failure before first callback')
    need(s.get('fault_cleared_while_idle') is True and s.get('failed_moe_progress') == 0,
         'Existing failure path is idle, fault cleared, no successful failed-request MoE encode')
    if s['scenario'] in ('allocation','allocation-owner'):
        ordinal=3 if s['scenario']=='allocation-owner' else 1
        trace=s.get('allocation_trace',[])
        need(s.get('fault_ordinal')==ordinal and s.get('mm_pso_mask')==0 and s.get('injected_nulls')==1 and len(trace)==ordinal and
             [t.get('ordinal')for t in trace]==list(range(1,ordinal+1)) and all(t.get('success') is True for t in trace[:-1]) and trace[-1].get('success') is False and
             all(type(t.get('bytes'))is int and t['bytes']>0 and type(t.get('site'))is str and t['site']for t in trace),'Complete actual failed allocation trace')
        hit=trace[-1]
        expected_kind=1 if ordinal==3 or not s['workspace_mode'] else 3
        need(hit.get('kind')==expected_kind,'Host metadata versus actual Metal owner allocation scope')
        if s['workspace_mode']:
            need(hit['site']==('mm_mid' if ordinal==3 else 'localgrow.down_views.metadata'),'Actual local transaction allocation site')
    elif s['scenario'].startswith('prompt-allocation-'):
        target=1 if s['scenario']=='prompt-allocation-first'else 6
        need(s.get('fault_ordinal')==0 and s.get('mm_pso_mask')==0 and s.get('prompt_fault_ordinal')==target and s.get('prompt_allocation_attempts')==6 and s.get('prompt_injected_ordinal')==target and s.get('prompt_injected_hits')==1 and s.get('injected_nulls')==0 and s.get('selected_provider')==1 and bool(s['allocation_trace']) and all(t['success']is True for t in s['allocation_trace']),'Actual independent prompt-six allocator NULL hit')
        need(s.get('prompt_old_owners_retained')is True and (not s['workspace_mode']or s.get('prompt_old_views_retained')is True),'Prompt ownership preserved after later failure; MoE already committed')
    else:
        need(s.get('fault_ordinal')==0 and s.get('mm_pso_mask')==8 and s.get('selected_provider')==0 and s.get('injected_nulls')==0,'Shared sum dependency unavailable: NONE, no MV fallback')
    need(s.get('recovery_api_ok') is True and s.get('recovery_stop_reason') == 'length' and
         s.get('recovery_generation_ok') is True and s.get('recovery_state_exported') is True and
         s.get('recovery_output_tokens') == s['recovery_tokens'] and s.get('recovery_prompt_tokens', 0) > 1, 'Same original context successfully recovered')
    need(s['warm_prompt_tokens']==s['recovery_prompt_tokens']==6917 and s['failure_prompt_tokens']==15187,'Actual frozen 7k->16k->7k context sequence')
    for key in ('warm_workspace', 'recovery_workspace'):
        common.layout_reader.workspace(s[key],s['workspace_mode'],16384 if key=='recovery_workspace'and s['scenario'].startswith('prompt-allocation-')else 8192)
        w = s.get(key)
        need(isinstance(w, dict) and w.get('mode') == s['workspace_mode'] and w.get('requests', 0) > 0 and
             w.get('mm_encoded', 0) > 0 and w.get('failures') == 0, 'Actual normal-path adoption: ' + key)
        if s['workspace_mode']:
            need(w.get('legacy_gate_bytes') == w.get('legacy_up_bytes') == w.get('legacy_mid_bytes') == 0 and
                 w.get('arena', {}).get('mm_encoded', 0) > 0, 'Compact arena actual adoption')


def review(off, on, request_paths):
    with common.Hashes() as hashes:
        summaries = [common.layout_reader.load(root/'summary.json') for root in (off,on)]
        for s in summaries:
            metadata_gate(s)
        a, b = summaries
        need([s['workspace_mode'] for s in summaries] == [0, 1], 'Independent original then compact arms')
        for key in ('scenario', 'seed', 'temperature', 'recovery_tokens', 'warm_prompt_tokens', 'failure_prompt_tokens',
                    'recovery_prompt_tokens', 'vocab_size', 'snapshot_slots', 'request_inputs'):
            need(a.get(key) == b.get(key), 'Frozen original/compact request/config: ' + key)
        requests = {}
        for name, path in request_paths.items():
            actual = hashes.file(path); requests[name] = actual
            need(all(all(s['request_inputs'][name].get(k) == actual[k] for k in ('bytes', 'fnv1a64_bytes')) for s in summaries), 'Original request identity: ' + name)
        layout_evidence=review_failure_layouts(off,on,summaries)
        phases = ('before-failure', 'after-failure', 'recovery')
        arms = [{phase: common.read_state(root, phase, s, hashes) for phase in phases}
                for root, s in zip((off, on), summaries)]
        compared = {phase: common.compare_states(arms[0][phase], arms[1][phase], phase)
                    for phase in ('before-failure', 'recovery')}
        # Failure timing/cleanup differs by design. Individually validate defined
        # raw failed state; never demand identical partial live scratch or owner.
        return {'schema_version': 1, 'evidence_valid': True, 'correctness_pass': True,
                'decision': 'pass_same_context_failure_recovery_within_exported_state_scope',
                'scenario': a['scenario'],'implementation_version':4, 'temperature': a['temperature'], 'seed': 42,
                'recovery_tokens': a['recovery_tokens'], 'same_context_recovery_raw_bit_exact': True,
                'performance_measured': False, 'real_oom': False, 'request_inputs': requests,'localgrow_layout_evidence':layout_evidence,
                'raw_bytes_compared': compared, 'transient_failure_raw_forced_equal': False,
                'workspace_contracts': [{k: s[k] for k in ('workspace_present_after_failure', 'old_capacity',
                                                         'old_moe_batch', 'old_generation', 'after_capacity',
                                                         'after_moe_batch', 'after_generation')} for s in summaries],
                'arms': [{phase: {k: state[k] for k in ('manifest_sha256', 'witnessed', 'published')}
                          for phase, state in arm.items()} for arm in arms],
                'limitations': ['Injected NULL simulates the allocator failure path; no real process/system memory-pressure OOM is applied.',
                                'Shared PSO unavailability is a diagnostic requirement mask, not destruction of live PSOs.',
                                'Partially updated live cache/carry and unpublished capacity after failure are excluded. Published snapshots/CPU state are validated individually.',
                                'MoE local transaction failure retains old pools; a later prompt allocation failure preserves old prompt6 while the already-committed B16k MoE pool is retained. Transactional localgrow must retain all old CW/B/pool IDs and actual non-MoE bindings on failure; legacy control retention/release is reported. Successful recovery is compared byte for byte.',
                                'No extra GPU wait, RNG reset, fresh-context recovery substitute, persistent disk-cache payload or performance proof.']}


def prompt_failure_identity_gate(warm,failed,recovered,setting):
    need(warm['owners']==failed['owners']==recovered['owners'] and all(warm[k]==failed[k]==recovered[k]for k in ('cap','mask_bytes','total_bytes')),'Prompt NULL preserves six owners, extents and caps')
    if setting:need(warm['views']==failed['views'],'Compact prompt NULL preserves all six old view identities')
    else:need(failed['cw_id']!=warm['cw_id'] and all(not x['resource_id']for x in failed['views']),'Legacy newly-created CW views are empty; no fabricated warm-view preservation')
    return True

def review_failure_layouts(off,on,summaries):
    lr=common.layout_reader;hashes=lr.ByteHashes();evidence=[]
    try:
        for root,s in zip((off,on),summaries):
            v=[lr.read_layout(root/('layout-'+phase+'.json'),hashes)for phase in ('before-failure','after-failure','recovery')]
            warm,failed,recovered=[q['value']for q in v]
            need(all(not q['value']['present']or q['value']['compact']==s['workspace_mode']for q in v),'Actual compact selector throughout fault/recovery')
            prompt=s['scenario'].startswith('prompt-allocation-')
            need(warm['present'] and warm['capacity']==2048 and warm['moe_batch']==8192 and recovered['present'] and recovered['moe_batch']==(16384 if prompt else 8192),'Actual warm/recovered workspace')
            if prompt:
                lr.layout_transition(v[0],v[1],1,s['workspace_mode']);lr.stable_bindings(v[1],v[2],True)
                need(all(failed[k]==recovered[k]for k in ('moe_batch','moe_generation','down_id','arena_id','acc_ids','localgrow_attempts','localgrow_successes','localgrow_failures')),'Already-committed grown MoE pools retained into recovery')
                if s['workspace_mode']:lr.counter_transition(s['warm_workspace'],s['recovery_workspace'],v[0],v[2],1)
            elif s['workspace_mode']:
                lr.stable_bindings(v[0],v[1],True)
                need(all(warm[k]==failed[k]for k in ('moe_batch','moe_generation','down_id','down_bytes','arena_id','acc_ids','acc_bytes','localgrow_successes')) and
                     failed['localgrow_attempts']==warm['localgrow_attempts']+1 and failed['localgrow_failures']==warm['localgrow_failures']+1,'Failed transaction must preserve old pools and report failure')
                lr.stable_bindings(v[0],v[2],True)
                need(all(failed[k]==recovered[k]for k in ('moe_batch','moe_generation','down_id','arena_id','acc_ids','localgrow_attempts','localgrow_successes','localgrow_failures')),'Short recovery reuses preserved pool')
                lr.counter_transition(s['warm_workspace'],s['recovery_workspace'],v[0],v[2],1)
            ps=common.prompt_snapshot_evidence(root,('before-failure','after-failure','recovery'),s['workspace_mode'])
            w,f,r=[x['value']for x in ps]
            need(w['cap']==r['cap']==7168,'Successful warm and recovery prompt extent')
            if prompt:
                prompt_failure_identity_gate(w,f,r,s['workspace_mode'])
            elif s['scenario']=='shared-pso'and not s['workspace_mode']:
                need(not f['cap']and not f['total_bytes']and not f['cw_id'],'Original off runtime failure releases complete CW/prompt group')
            else:need(w['owners']==f['owners']==r['owners'],'Early MoE transaction failure preserves old prompt group')
            evidence.append([dict(path=q['path'],sha256=q['sha256'],value={k:x for k,x in q['value'].items()if k!='bindings'},actual_bindings_fully_compared=True,prompt_identity=ps[i])for i,q in enumerate(v)])
    finally:hashes.close()
    return evidence


def fixture_summary():
    w=common.test_workspace(0,8192)
    return dict(schema_version=1,implementation_version=4,prompt_view_clear_mode=0,probe='same_context_moe_workspace_failure_recovery_promptclear',status='complete',fullmodel_recovery_within_arm_completed=True,scenario='allocation',workspace_mode=0,seed=42,temperature=.7,recovery_tokens=8,same_context=True,rng_manually_reset=False,extra_gpu_waits_inserted=0,performance_measured=False,real_oom=False,vocab_size=129280,snapshot_slots=44,warm_generation_ok=True,warm_state_exported=True,warm_output_tokens=8,old_workspace_present_before_failure=True,old_capacity=2048,old_moe_batch=8192,old_generation=1,warm_prompt_tokens=6917,failure_prompt_tokens=15187,failure_api_ok=False,failure_stop_reason='error',failure_output_tokens=0,failure_gate_pass=True,failure_state_exported=True,fault_cleared_while_idle=True,failed_moe_progress=0,fault_ordinal=1,mm_pso_mask=0,injected_nulls=1,selected_provider=1,allocation_trace=[dict(ordinal=1,success=False,kind=1,bytes=655360,site='x_buf')],recovery_api_ok=True,recovery_stop_reason='length',recovery_generation_ok=True,recovery_state_exported=True,recovery_output_tokens=8,recovery_prompt_tokens=6917,warm_workspace=w,recovery_workspace=w)


def self_test():
    controls=[];s=fixture_summary()
    def test(name,fn,reject=False):
        try:fn();passed=not reject
        except (common.Rejected,ValueError,OSError,KeyError,TypeError):passed=reject
        controls.append(dict(name=name,passed=passed));need(passed,'CPU control '+name)
    test('actual_legacy_owner_null',lambda:metadata_gate(s))
    local=copy.deepcopy(s);local.update(workspace_mode=1,prompt_view_clear_mode=1,warm_workspace=common.test_workspace(1,8192),recovery_workspace=common.test_workspace(1,8192));local['allocation_trace']=[dict(ordinal=1,success=False,kind=3,bytes=16384,site='localgrow.down_views.metadata')]
    test('actual_local_host_metadata_null',lambda:metadata_gate(local))
    owner=copy.deepcopy(local);owner.update(scenario='allocation-owner',fault_ordinal=3,allocation_trace=[dict(ordinal=1,success=True,kind=3,bytes=16384,site='localgrow.down_views.metadata'),dict(ordinal=2,success=True,kind=3,bytes=256,site='arena.metadata'),dict(ordinal=3,success=False,kind=1,bytes=16384*6*2304*2,site='mm_mid')]);test('actual_local_metal_owner_null',lambda:metadata_gate(owner))
    bad=copy.deepcopy(local);bad['allocation_trace'][0]['kind']=1;test('reject_host_metadata_as_metal_owner',lambda:metadata_gate(bad),True)
    bad=copy.deepcopy(owner);bad['allocation_trace'][-1]['kind']=3;test('reject_owner_test_as_metadata',lambda:metadata_gate(bad),True)
    ps=copy.deepcopy(local);ps.update(scenario='shared-pso',fault_ordinal=0,mm_pso_mask=8,selected_provider=0,injected_nulls=0,allocation_trace=[]);test('shared_sum_none',lambda:metadata_gate(ps))
    bad=copy.deepcopy(ps);bad['selected_provider']=2;test('reject_shared_sum_fake_mv',lambda:metadata_gate(bad),True)
    for ordinal,name in ((1,'first'),(6,'last')):
        q=copy.deepcopy(local);q.update(scenario='prompt-allocation-'+name,fault_ordinal=0,mm_pso_mask=0,injected_nulls=0,selected_provider=1,prompt_fault_ordinal=ordinal,prompt_allocation_attempts=6,prompt_injected_ordinal=ordinal,prompt_injected_hits=1,prompt_old_owners_retained=True,prompt_old_views_retained=True,allocation_trace=[dict(ordinal=1,success=True,kind=3,bytes=16384,site='localgrow.down_views.metadata')],recovery_workspace=common.test_workspace(1,16384))
        test('actual_prompt_'+name+'_null_after_moe_commit',lambda q=q:metadata_gate(q))
        for field,value in [('selected_provider',2),('prompt_injected_ordinal',0),('prompt_injected_hits',0),('prompt_allocation_attempts',ordinal-1),('prompt_old_owners_retained',False),('prompt_old_views_retained',False)]:
            bad=copy.deepcopy(q);bad[field]=value;test('reject_prompt_'+name+'_'+field,lambda bad=bad:metadata_gate(bad),True)
    warm=dict(cw_id=1,cap=7168,mask_bytes=112,total_bytes=96,owners=[dict(wrapper_id=i+1,resource_id=i+10,bytes=16,offset=0,is_view=0)for i in range(6)],views=[dict(wrapper_id=i+100,resource_id=i+10,bytes=8,offset=0,is_view=1)for i in range(6)])
    failed=copy.deepcopy(warm);recovered=copy.deepcopy(warm)
    test('compact_prompt_identity_after_later_failure',lambda:prompt_failure_identity_gate(warm,failed,recovered,1))
    bad=copy.deepcopy(failed);bad['owners'][5]['resource_id']=999;test('reject_partial_prompt_owner_publication',lambda:prompt_failure_identity_gate(warm,bad,recovered,1),True)
    bad=copy.deepcopy(failed);bad['views'][0]['resource_id']=0;test('reject_clear_before_prompt_allocation_complete',lambda:prompt_failure_identity_gate(warm,bad,recovered,1),True)
    bad=copy.deepcopy(failed);bad['cap']=15360;test('reject_new_cap_on_failed_prompt_transaction',lambda:prompt_failure_identity_gate(warm,bad,recovered,1),True)
    legacy=copy.deepcopy(failed);legacy['cw_id']=2;legacy['views']=[dict(v,resource_id=0,bytes=0)for v in legacy['views']]
    test('legacy_whole_cw_changed_prompt_owners_preserved',lambda:prompt_failure_identity_gate(warm,legacy,recovered,0))
    for key,badvalue in [('rng_manually_reset',True),('real_oom',True),('warm_generation_ok',False),('failure_api_ok',True),('failure_output_tokens',1),('failure_prompt_tokens',6917),('fault_cleared_while_idle',False),('failed_moe_progress',1),('recovery_output_tokens',7)]:
        x=copy.deepcopy(local);x[key]=badvalue;test('reject_'+key,lambda x=x:metadata_gate(x),True)
    return dict(schema_version=1,scope='cpu_promptclear_engine_failure_reader_controls',gpu_run=False,controls=controls,passed=all(v['passed']for v in controls))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('off', 'on', 'warm-request', 'failure-request', 'recover-request', 'out'):
        p.add_argument('--'+name, type=Path)
    p.add_argument('--self-test', action='store_true'); args = p.parse_args()
    try:
        if args.self_test:
            report = self_test()
        else:
            need(all((args.off, args.on, args.warm_request, args.failure_request, args.recover_request)), 'Original/compact roots plus three original requests')
            report = review(args.off.resolve(), args.on.resolve(), {k: v.resolve() for k, v in
                            [('warm', args.warm_request), ('failure', args.failure_request), ('recover', args.recover_request)]})
    except (common.Rejected, OSError, ValueError, KeyError, TypeError) as error:
        report = dict(evidence_valid=False, correctness_pass=False, performance_measured=False,
                      decision='reject_incomplete_or_inconsistent_failure_recovery', error=str(error))
    if args.out:
        with args.out.open('x') as f:
            json.dump(report, f, indent=2); f.write('\n')
    print(json.dumps(report, indent=2)); return 0 if report.get('passed', report.get('correctness_pass', False)) else 1


if __name__ == '__main__':
    raise SystemExit(main())
