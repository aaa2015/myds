#!/usr/bin/env python3
"""CPU-only synthetic contracts, never inference/GPU or causal device evidence."""
import copy
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile

sys.dont_write_bytecode=True
spec=importlib.util.spec_from_file_location('causal_reader',Path(__file__).with_name('review_moe_prompt_lifetime_diagnostic.py'))
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
SIZES=[587202560,688128,14680064,802816,176160768,176160768]


def identity(w,res,n,view=0):
    return dict(wrapper_id=w,resource_id=res,offset=0,bytes=n,is_view=view)


def actual(t,device):
    return dict(task_valid=True,device_valid=True,task_info_return=0,task_info_count_words=100,
        task_info_requested_words=100,snapshot_begin_s=t+.001,snapshot_end_s=t+.002,
        phys_footprint_bytes=1000000000,rss_bytes=2000000000,device_allocated_bytes=device,
        task_vm=dict(phys_footprint=1000000000,resident_size=2000000000,
            internal=500000000,external=300000000,device=1000000,compressed=2000000,reusable=0,page_size=16384))


def common(kind,t,device,stage=1):
    return dict(kind=kind,sequence=0,stage=stage,in_generation=True,monotonic_s=t,
        observer_CPU_only=True,GPU_work_submitted_by_callback=False,extra_gpu_waits=0,actual=actual(t,device))


def fixture(clear_mode):
    records=[];allocated=[(n+16383)//16384*16384 for n in SIZES]
    for stage in (0,1):
        cap=7168 if stage==0 else 15360;mask=112 if stage==0 else 240
        old=[identity(100+i,200+i,SIZES[i])for i in range(6)]if stage else[identity(0,0,0)for _ in range(6)]
        sizes=SIZES if stage==0 else[cap*81920,cap*96,cap*2048,cap*mask,cap*24576,cap*24576]
        new=[identity(100+i,200+i,sizes[i])for i in range(6)]if stage==0 else[identity(300+i,400+i,sizes[i])for i in range(6)]
        views=[identity(700+i,200+i,SIZES[i]//2,1)for i in range(6)]if stage else[identity(700+i,0,0,1)for i in range(6)]
        empty=[identity(700+i,0,0,1)for i in range(6)]
        for event in (1,2,3):
            idx=stage*3+event;device=10000000000-(sum(allocated)if stage and clear_mode and event==3 else 0)
            v=common('prompt_growth',float(idx),device,stage)
            v.update(event=event,implementation_version=4,compact=1,clear_mode=clear_mode,quiescent=1,
                clear_applied=int(bool(clear_mode)and event>1),old_cap=7168 if stage else 0,new_cap=cap,
                prompt_sequence=idx,cw_id=99,old_mask_bytes=112 if stage else 0,new_mask_bytes=mask,
                old_total_bytes=sum(SIZES)if stage else 0,new_total_bytes=sum(sizes),owner_order=list(r.OWNERS),
                old_owners=old,new_owners=new,views_before=views,
                views_current=empty if clear_mode and event>1 else views,
                old_allocated_bytes=allocated if stage else[0]*6,
                new_allocated_bytes=[(n+16383)//16384*16384 for n in sizes])
            records.append(v)
    serial=0;epoch=0
    def drain(adds,t):
        nonlocal serial,epoch
        epoch+=1;pending=[]
        for offset,(res,n,alloc,origin)in enumerate(adds):
            serial+=1;v=common('deferred_buffer',t+offset*.01,9000000000)
            v.update(implementation_version=4,event=1,batch_active=1,inflight_count=0,origin=origin,
                deferred_sequence=serial,commit_epoch=epoch,resource_id=res,resource_bytes=n,
                resource_allocated_size=alloc,entries=len(pending)+1);records.append(v);pending.append(v)
        for event in (2,3):
            serial+=1;v=common('deferred_buffer',t+.2+event*.01,9000000000-(sum({x[0]:x[2]for x in adds}.values())if event==3 else 0))
            v.update(implementation_version=4,event=event,batch_active=0,inflight_count=0,origin=0,
                deferred_sequence=serial,commit_epoch=epoch,resource_id=0,resource_bytes=0,
                resource_allocated_size=0,entries=len(pending)if event==2 else 0);records.append(v)
    first=[(900,16777216,16777216,3)]
    if not clear_mode:first +=[(200+i,SIZES[i],allocated[i],2)for i in range(5)]
    drain(first,6.2)
    if not clear_mode:drain([(205,SIZES[5],allocated[5],2)],8.0)
    else:drain([(200,64,16384,2)],8.0) # equal pointer, after old proven-freed interval
    records.sort(key=lambda x:x['monotonic_s'])
    for i,v in enumerate(records,1):v['record_index']=i
    return records


def arm(root,label,mode):
    records=fixture(mode);prompt=r.prompt_groups(records,mode);deferred=r.deferred_groups(records);stages=[]
    for stage in r.STAGES:
        directory=root/label/stage;directory.mkdir(parents=True)
        (directory/'final-logits.f32').write_bytes(struct.pack('<4f',1,2,3,4))
        (directory/'output.bin').write_bytes(b'complete output including utf8 \xe4\xb8\xad')
        stages.append(dict(stage=stage,sample=dict(prompt_tokens=10,cached_tokens=0,prefill_tokens=9,output_tokens=8),
            state=dict(path=str(directory),files={'final-logits.f32':{},'output.bin':{}}),
            actual_peak_with_lifetime_events={k:10000000000-(100000000 if mode else 0)for k in r.METRICS}))
    return dict(metadata=dict(max_new_tokens=8),setting=1,clear_mode=mode,prompt=prompt,deferred=deferred,
        sequences=[dict(sequence=0,stages=stages)])


def run():
    controls=[]
    def test(name,fn,reject=False):
        try:fn();ok=not reject
        except (ValueError,KeyError,TypeError,OSError):ok=reject
        controls.append(dict(name=name,passed=ok));r.need(ok,'CPU control '+name)
    for mode in (0,1):
        v=fixture(mode)
        test('actual_six_prompt_parents_clear'+str(mode),lambda v=v,mode=mode:r.prompt_groups(v,mode))
        test('complete_deferred_queue_clear'+str(mode),lambda v=v:r.deferred_groups(v))
        test('readonly_task_device_fields_clear'+str(mode),lambda v=v:[r.actual(q)for q in v])
    v=fixture(1)
    def mutate(name,key,value):
        q=copy.deepcopy(v);q[4][key]=value;test(name,lambda:r.prompt_groups(q,1),True)
    mutate('reject_unknown_view_clear_quiescence','quiescent',0)
    mutate('reject_clear_not_applied','clear_applied',0)
    mutate('reject_wrong_actual_cap','new_cap',16384)
    q=copy.deepcopy(v);q[4]['views_current'][0]['resource_id']=200;test('reject_old_parent_survives_clear',lambda:r.prompt_groups(q,1),True)
    q=copy.deepcopy(v);q[3]['views_before'][0]['resource_id']=999;test('reject_unproven_old_view_parent',lambda:r.prompt_groups(q,1),True)
    q=copy.deepcopy(v)
    for row in q[:3]:row['new_owners'][0]['resource_id']=999
    test('reject_initial_parent_chain_changed',lambda:r.prompt_groups(q,1),True)
    q=copy.deepcopy(v);q[3]['old_allocated_bytes'][0]=1;test('reject_allocated_smaller_than_requested',lambda:r.prompt_groups(q,1),True)
    q=copy.deepcopy(v);q.pop(1);test('reject_missing_known_lifetime_point',lambda:r.prompt_groups(q,1),True)
    q=copy.deepcopy(v);q[3]['prompt_sequence']=3;test('reject_equal_event_sequence',lambda:r.prompt_groups(q,1),True)
    q=copy.deepcopy(v);q[0]['GPU_work_submitted_by_callback']=True;test('reject_observer_gpu_work',lambda:r.actual(q[0]),True)
    q=copy.deepcopy(v);q[0]['actual']['snapshot_begin_s']=float('nan');test('reject_nonfinite_snapshot',lambda:r.actual(q[0]),True)
    q=copy.deepcopy(v);q[0]['actual']['task_vm']['phys_footprint']+=1;test('reject_task_ledger_duplicate_lie',lambda:r.actual(q[0]),True)
    q=fixture(0);next(a for a in q if a['kind']=='deferred_buffer'and a['event']==1)['entries']+=1
    test('reject_deferred_entry_count_lie',lambda:r.deferred_groups(q),True)
    q=fixture(0);q.pop(next(i for i,a in enumerate(q)if a['kind']=='deferred_buffer'and a['event']==3))
    test('reject_missing_drain_end',lambda:r.deferred_groups(q),True)
    q=fixture(0);next(a for a in q if a['kind']=='deferred_buffer'and a['event']==2)['inflight_count']=1
    test('reject_drain_while_gpu_inflight',lambda:r.deferred_groups(q),True)
    q=fixture(0);next(a for a in q if a['kind']=='deferred_buffer'and a['event']==1)['origin']=0
    test('reject_unknown_deferred_origin',lambda:r.deferred_groups(q),True)
    with tempfile.TemporaryDirectory(prefix='prompt-causal-CPU-')as temp:
        root=Path(temp);a,b,c=(arm(root,label,mode)for label,mode in zip(r.ORDER,(0,1,0)))
        gate=lambda:r.causal_gate(a,b,c)
        test('full_original_state_and_samebuild_lifetime_intervention',lambda:r.need(gate()['causal_lifetime_gate_pass'],'causal gate'))
        test('clear1_later_equal_pointer_not_misattributed',lambda:r.need(len(r.lifetime(b)['post_release_pointer_address_matches_not_attributed_to_old_resources'])==1,'address lifetime'))
        q=copy.deepcopy(a);q['deferred']['drains'][0]['adds'][1]['origin']=1
        test('reject_prompt_hold_wrong_source_origin',lambda:r.need(r.lifetime(q)['lifetime_prediction_pass'],'source'),True)
        q=copy.deepcopy(b);q['prompt'][1]['expected_release_observed']=False
        test('negative_actual_release_is_valid_but_fails_prediction',lambda:r.need(not r.causal_gate(a,q,c)['causal_lifetime_gate_pass'],'negative'))
        q=copy.deepcopy(b);q['deferred']['drains'][0]['adds'][0]['resource_bytes']=8388608
        test('reject_missing_16MiB_idmap_path',lambda:r.need(r.lifetime(q)['lifetime_prediction_pass'],'map'),True)
        q=copy.deepcopy(b);q['deferred']['drains'][0]['adds'][0]['resource_allocated_size']+=16384
        test('reject_idmap_actual_allocation_changed_between_arms',lambda:r.need(r.causal_gate(a,q,c)['causal_lifetime_gate_pass'],'map equality'),True)
        for stage in b['sequences'][0]['stages']:stage['actual_peak_with_lifetime_events']['phys_footprint_bytes']=11000000000
        test('mechanism_pass_does_not_claim_phys_peak_reduction',lambda:r.need(gate()['causal_lifetime_gate_pass']and not gate()['observed_phys_peak_lower_than_both_controls'],'effect separate'))
        path=Path(c['sequences'][0]['stages'][2]['state']['path'])/'final-logits.f32';path.write_bytes(struct.pack('<4I',0x3f800000,0x40000000,0x40400000,0x40800001))
        test('reject_single_ulp_in_last_stage_raw_logits',lambda:gate(),True)
        path.write_bytes(struct.pack('<4f',1,2,3,4));out=Path(c['sequences'][0]['stages'][0]['state']['path'])/'output.bin';out.write_bytes(out.read_bytes()[:-1]+b'x')
        test('reject_exact_output_last_byte',lambda:gate(),True)
        metadata=dict(schema_version=1,probe='moe_workspace_promptclear_fullsequence',phase='probe',workspace_mode=1,
            prompt_view_clear_mode=0,implementation_version=4,repeats=0,warmup_sequences=0,max_new_tokens=8,
            diagnostic_prompt_mode=False,deferred_lifetime_observer=False,prompt_lifetime_observer=True,growth_lifetime_observer=True)
        d=root/'normal-raw';d.mkdir();(d/'summary.jsonl').write_text('\n'.join(json.dumps(x)for x in (dict(kind='metadata',**metadata),dict(kind='sequence_complete'),dict(kind='complete')))+'\n')
        test('reject_normal_scope_as_causal',lambda:r.read_arm(d,{}, {},0),True)
        synthetic=dict(schema_version=1,evidence_valid=True,implementation_version=4,mode='causal_diagnostic',correctness_pass=True,
            causal_lifetime_gate_pass=True,performance_measured=False,production_adoption=False,plan_sha256=r.PLAN_SHA,
            reader_sha256=r.sha(Path(r.__file__)),native_identity={'candidate':{p:'cpu-only'for p in (*r.CORE,r.NATIVE,'bin/myds.metallib')}},
            production_environment={'MYDS_METAL_CONCURRENT':'1'},model=dict(path='cpu-fixture-only',bytes=1),model_header_sha256='cpu',
            gate=dict(measured_peak_intervention={},observed_phys_peak_lower_than_both_controls=False,actual_id_map_requested_and_allocated_extents_preserved=True))
        test('bounded_prerequisite_top_schema',lambda:r.need(r.small_review(synthetic)['native_identity']==synthetic['native_identity'],'schema'))
    return dict(schema_version=1,scope='bounded_synthetic_prompt_lifetime_reader_CPU_controls',gpu_run=False,model_run=False,
        reader_sha256=r.sha(Path(r.__file__)),controls=controls,passed=all(v['passed']for v in controls))


if __name__=='__main__':print(json.dumps(run(),indent=2,sort_keys=True))
