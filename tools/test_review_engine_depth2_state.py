#!/usr/bin/env python3
"""CPU unit tests of Engine reviewer scheduling and residency protocol gates.

Synthetic dictionaries exercise the strict protocol functions, including a
legitimate EOS continuation. They are not model/GPU runs or full raw replay.
"""
import copy
import json
import sys
sys.dont_write_bytecode=True
import review_engine_depth2_state as review


def fixture():
    states,requests={},{}
    rows=[{'kind':'metadata'},{'kind':'residency','label':'model_baseline','allocation_count':10,
        'allocated_size':1000,'device_allocated_size':2000,'live_mapped_allocations':8,
        'pending_unregister':0,'supported':True,'active':True,'heartbeat_running':True,'pass':True}]
    baseline=rows[1]
    for c in range(5):
        loaded=copy.deepcopy(baseline);loaded.update(label=f'case{c}_loaded',allocation_count=12,
            allocated_size=1200,live_mapped_allocations=10);rows.append(loaded)
        for suffix in ('','_continue') if c<2 else ('',):
            for arm in ('candidate','ordinary'):
                label=f'case{c}_{arm}{suffix}';spec=arm=='candidate';continuing=bool(suffix)
                cap=16 if continuing else 256 if c<2 else c-1
                outputs=cap if continuing or c>=2 else c+2
                d={key:0 for key in review.COUNTERS}
                d.update(target_forward_calls=outputs,target_rows=outputs)
                if spec and not continuing and c<2:
                    d.update(target_forward_calls=2,target_rows=4,batch2_calls=1,batch2_hits=1,
                        depth2_calls=1,depth2_hits2=1,draft_cycles=1,draft_proposed=2,draft_accepted=c)
                if spec and not continuing and c==4:
                    d.update(target_forward_calls=2,target_rows=3,batch2_calls=1,batch2_hits=1,
                        draft_cycles=1,draft_proposed=1,draft_accepted=1)
                prompt=b' Continue with the next point.' if continuing else b'Synthetic rendered chat prompt'
                hit=c+1 if not continuing and c<2 and spec else 0
                cancel=c+2 if not continuing and c<2 and not spec else 0
                requests[label]=dict(kind='request',label=label,spec=spec,cap=cap,cancel_after=cancel,
                    hit_output_cancel=hit,hit_index=hit,continuation=continuing,temperature=0,
                    prompt_tokens=40 if continuing else 12,cached_tokens=states[f'case{c}_{arm}']['pos'] if continuing else 0,
                    prompt_bytes=len(prompt),prompt_fnv1a64=review.fnv(prompt),**{'pass':True})
                states[label]=dict(kind='state',label=label,reason=4 if not continuing and c<2 else 1,
                    output_tokens=outputs,pos=12+outputs,decode=d)
                rows.extend((requests[label],states[label]))
        if c==1:rows.append({'kind':'prefix_probe'})
        freed=copy.deepcopy(baseline);freed['label']=f'case{c}_freed';rows.append(freed)
    freed=copy.deepcopy(baseline);freed.update(label='model_freed',allocation_count=0,live_mapped_allocations=0);rows.append(freed)
    clean=copy.deepcopy(freed);clean.update(label='runtime_cleanup',allocated_size=0,active=False,heartbeat_running=False);rows.append(clean)
    rows.append({'kind':'complete'})
    return states,requests,rows


def main():
    states,requests,rows=fixture()
    assert review.validate_protocol(states,requests)
    assert len(review.validate_residency(rows))==13
    eos=copy.deepcopy(states)
    for arm in ('candidate','ordinary'):
        eos[f'case0_{arm}_continue'].update(reason=0,output_tokens=2)
    assert review.validate_protocol(eos,requests)
    results=[]
    def reject(name,change,resource=False):
        s,q,r=copy.deepcopy((states,requests,rows));change(s,q,r)
        try:
            review.validate_residency(r) if resource else review.validate_protocol(s,q)
        except (ValueError,KeyError,TypeError) as error:
            results.append(dict(name=name,rejected=True,reason=str(error)))
        else:raise AssertionError('Accepted invalid protocol: '+name)
    def request(label,key,value):return lambda s,q,r:q[label].__setitem__(key,value)
    def counter(label,key,value):return lambda s,q,r:s[label]['decode'].__setitem__(key,value)
    reject('candidate_spec_off',request('case0_candidate','spec',False))
    reject('wrong_cancel_boundary',request('case1_candidate','hit_output_cancel',1))
    reject('ordinary_wrong_cancel_count',request('case0_ordinary','cancel_after',1))
    reject('short_tail_cap_changed',request('case4_candidate','cap',4))
    reject('sampling_instead_of_greedy',request('case0_candidate','temperature',0.7))
    reject('false_continuation_flag',request('case0_candidate_continue','continuation',False))
    reject('wrong_retained_prefix',request('case1_candidate_continue','cached_tokens',1))
    reject('continuation_prompt_changed',request('case0_candidate_continue','prompt_fnv1a64','0'*16))
    reject('cold_prompt_changed',request('case3_ordinary','prompt_fnv1a64','0'*16))
    reject('ordinary_reference_used_drafter',counter('case0_ordinary','draft_cycles',1))
    reject('depth1_tail_not_executed',lambda s,q,r:s['case4_candidate']['decode'].update(batch2_calls=0,batch2_hits=0,target_rows=2))
    reject('native_fallback',counter('case0_candidate','depth2_fallbacks',1))
    reject('short_tail_executes_native_depth2',counter('case3_candidate','depth2_calls',1))
    reject('no_actual_all_hit',lambda s,q,r:s['case1_candidate']['decode'].update(depth2_hits2=0,depth2_misses=1))
    reject('continuation_short_length_not_eos',lambda s,q,r:s['case0_candidate_continue'].update(output_tokens=2,reason=1))
    def residue(label,key,value):
        return lambda s,q,r:next(x for x in r if x.get('label')==label).__setitem__(key,value)
    reject('resident_context_allocation_leak',residue('case3_freed','allocation_count',11),True)
    reject('pending_unregister',residue('case4_freed','pending_unregister',1),True)
    reject('heartbeat_not_stopped',residue('runtime_cleanup','heartbeat_running',True),True)
    reject('missing_churn_case',lambda s,q,r:r.pop(next(i for i,x in enumerate(r) if x.get('label')=='case4_freed')),True)
    reject('resident_payload_size_leak',residue('case2_freed','allocated_size',1001),True)
    reject('residency_boolean_as_counter',residue('case0_loaded','allocation_count',True),True)
    reject('residency_case_order',lambda s,q,r:r.reverse(),True)
    result=dict(schema_version=1,scope=__doc__.strip(),positive_cases=['fixed_greedy_lifecycle_protocol',
        'legitimate_eos_continuation','thirteen_record_residency_churn'],negative_count=len(results),
        negative=results,**{'pass':True})
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=='__main__':main()
