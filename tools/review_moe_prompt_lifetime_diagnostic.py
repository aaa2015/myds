#!/usr/bin/env python3
"""Independent CPU review of the same-build workspace1 clear0/1/0 diagnostic.

This deliberately has a separate read_arm contract. Normal performance/probe
read_arm remains strict and rejects the diagnostic deferred observer.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import tempfile


def _module():
    p=Path(__file__).with_name('review_moe_workspace_fullmodel_promptclear.py')
    spec=importlib.util.spec_from_file_location('promptclear_normal_CPU_helpers',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


base=_module();need=base.need;sha=base.sha;load=base.load;rows=base.rows
PLAN_SHA='a83127b1427b09b703883150f3ec6e8e90e7ecdead3f48b798c7755368bd9d9c'
FLAG='MYDS_PREFILL_MOE_WORKSPACE';CLEAR='MYDS_PREFILL_PROMPT_VIEW_CLEAR'
ORDER=('C0a','C1','C0b');STAGES=('setup','grow','retain')
OWNERS=('RES','PRE','SELECTED','MASK','ENGRAM0','ENGRAM1')
METRICS=('phys_footprint_bytes','rss_bytes','device_allocated_bytes')
NATIVE='bin/bench_moe_workspace_fullmodel_promptclear'
CORE=('src/myds_v41.c','src/myds_metal.m','include/myds_internal.h','include/myds_metal.h')


def actual(record):
    a=record['actual']
    need(record['observer_CPU_only'] is True and record['GPU_work_submitted_by_callback'] is False and
         record['extra_gpu_waits']==0 and a['task_valid'] is True and a['device_valid'] is True and
         a['task_info_return']==0 and a['task_info_count_words']>0 and
         a['task_info_requested_words']>=a['task_info_count_words'] and
         all(type(v) in (int,float) and math.isfinite(v) for v in
             (record['monotonic_s'],a['snapshot_begin_s'],a['snapshot_end_s'])) and
         record['monotonic_s']<=a['snapshot_begin_s']<=a['snapshot_end_s'] and
         all(type(a[k]) is int and a[k]>0 for k in METRICS),'Actual readonly task/device observer scope')
    vm=a['task_vm'];need(vm['phys_footprint']==a['phys_footprint_bytes'] and vm['resident_size']==a['rss_bytes'] and
         all(type(vm[k]) is int and vm[k]>=0 for k in ('internal','external','device','compressed','reusable','page_size')),
         'Actual task ledger extent/duplicate values')
    return a


def identities(record):
    need(record['owner_order']==list(OWNERS),'Fixed six-resource order')
    for key in ('old_owners','new_owners','views_before','views_current'):
        values=record[key];need(len(values)==6,'All six actual identities')
        for v in values:
            need(set(v)=={'wrapper_id','resource_id','offset','bytes','is_view'} and
                 all(type(v[k]) is int and v[k]>=0 for k in v) and v['is_view'] in (0,1),'Bounded raw resource identity')
    for kind in ('old','new'):
        sizes=record[kind+'_allocated_bytes'];owners=record[kind+'_owners']
        need(len(sizes)==6 and all(type(n) is int and n>=0 for n in sizes) and
             sum(v['bytes'] for v in owners)==record[kind+'_total_bytes'],'Requested/actual allocated sizes separately recorded')
        for v,n in zip(owners,sizes):
            need((v['resource_id']>0 and v['wrapper_id']>0 and v['is_view']==0 and v['offset']==0 and n>=v['bytes']>0) or
                 (not v['resource_id'] and not v['wrapper_id'] and not v['bytes'] and n==0),'Actual owned resource allocation')


def prompt_groups(records,clear_mode):
    p=[v for v in records if v['kind']=='prompt_growth']
    need([(v['stage'],v['event'])for v in p]==[(i,j)for i in (0,1)for j in (1,2,3)],
         'Complete initial and grow prompt transactions; retain may not replace')
    need(all(b['prompt_sequence']==a['prompt_sequence']+1 for a,b in zip(p,p[1:])),
         'Per-emitted-event prompt sequence must be consecutive')
    out=[]
    for stage in (0,1):
        group=[v for v in p if v['stage']==stage];first,cleared,freed=group
        for v in group:
            identities(v);need(v['implementation_version']==4 and v['compact']==1 and v['clear_mode']==clear_mode and
                v['quiescent']==1 and v['in_generation'] is True and v['old_cap']==(0 if stage==0 else 7168) and
                v['new_cap']==(7168 if stage==0 else 15360),'Actual compact prompt transaction extent/eligibility')
        cap=first['new_cap'];mask=112 if stage==0 else 240
        need([v['bytes']for v in first['new_owners']]==[cap*81920,cap*96,cap*2048,cap*mask,cap*24576,cap*24576],
             'Actual six new prompt resource extents')
        fixed=('old_cap','new_cap','cw_id','old_mask_bytes','new_mask_bytes','old_total_bytes','new_total_bytes',
               'old_owners','new_owners','old_allocated_bytes','new_allocated_bytes','views_before')
        need(all(v[k]==first[k] for v in group for k in fixed),'Transaction identity changed between checkpoints')
        need(first['views_current']==first['views_before'],'READY must precede view mutation')
        if stage==0:
            need(first['old_total_bytes']==0 and all(v['resource_id']==0 for v in first['views_before']),
                 'Fresh prompt must not claim old owners/views')
        else:
            need(first['old_mask_bytes']==112 and first['new_mask_bytes']==240 and
                 [v['bytes']for v in first['old_owners']]==[587202560,688128,14680064,802816,176160768,176160768] and
                 first['old_total_bytes']==955695104,'Actual warm cap7168 six requested extents')
            need(len({v['resource_id']for v in first['old_owners']})==6 and
                 all(v['resource_id']==o['resource_id'] and v['wrapper_id']>0 and v['is_view']==1
                     and v['offset']+v['bytes']<=o['bytes'] for v,o in zip(first['views_before'],first['old_owners'])),
                 'Six actual old view parents must be the old six owners')
            need(first['cw_id']>0 and len({v['resource_id']for v in first['new_owners']})==6 and
                 not({v['resource_id']for v in first['old_owners']}&{v['resource_id']for v in first['new_owners']}),
                 'Old and new actual groups must coexist as distinct resources at READY')
        if clear_mode:
            need(cleared['clear_applied']==freed['clear_applied']==1 and
                 all(v['wrapper_id']==old['wrapper_id'] and not v['resource_id'] and not v['offset'] and not v['bytes']
                     for v,old in zip(cleared['views_current'],first['views_before'])) and
                 freed['views_current']==cleared['views_current'],'Clear must preserve wrappers and remove every old parent')
        else:need(cleared['clear_applied']==freed['clear_applied']==0 and
                  cleared['views_current']==freed['views_current']==first['views_before'],'Clear0 may not silently clear old parents')
        drop=cleared['actual']['device_allocated_bytes']-freed['actual']['device_allocated_bytes']
        out.append(dict(stage=stage,checkpoints=group,actual_old_allocated_size_sum=sum(first['old_allocated_bytes']),
                        old_requested_size_sum=first['old_total_bytes'],device_drop_after_old_owner_release=drop,
                        expected_release_observed=(drop==sum(first['old_allocated_bytes']) if clear_mode else drop==0)))
    initial,grow=(g['checkpoints'][0]for g in out)
    need(initial['cw_id']==grow['cw_id'] and initial['new_owners']==grow['old_owners'] and
         initial['new_allocated_bytes']==grow['old_allocated_bytes'],
         'Same-context initial prompt parents must be the actual old grow parents')
    return out


def deferred_groups(records):
    deferred=[r for r in records if r['kind']=='deferred_buffer'];pending=[];drains=[];previous=None;opening=None
    for r in deferred:
        need(r['implementation_version']==4 and r['event'] in (1,2,3) and r['batch_active'] in (0,1) and
             type(r['inflight_count']) is int and r['inflight_count']>=0 and type(r['commit_epoch']) is int and r['commit_epoch']>=0,
             'Actual deferred event fields')
        if previous:need(r['deferred_sequence']==previous['deferred_sequence']+1 and r['commit_epoch']>=previous['commit_epoch'],
                         'Consecutive deferred emit sequence and monotonic commit epoch')
        previous=r
        if r['event']==1:
            need(opening is None and r['origin'] in (1,2,3) and r['resource_id']>0 and
                 r['resource_allocated_size']>=r['resource_bytes']>0 and (r['batch_active'] or r['inflight_count']),
                 'Actual deferred ADD resource and active-lifetime origin')
            pending.append(r);need(r['entries']==len(pending),'Deferred ADD entry cardinality must include duplicates')
        elif r['event']==2:
            need(opening is None and pending and r['entries']==len(pending) and r['resource_id']==r['resource_bytes']==r['resource_allocated_size']==0 and
                 r['origin']==0 and not r['batch_active'] and r['inflight_count']==0,'Actual drain begins with the complete retained list')
            opening=r
        else:
            need(opening is not None and r['entries']==0 and r['resource_id']==r['resource_bytes']==r['resource_allocated_size']==0 and r['origin']==0 and
                 not r['batch_active'] and r['inflight_count']==0 and r['commit_epoch']==opening['commit_epoch'],'Actual drain ends in same commit epoch with zero retained entries')
            unique={}
            for a in pending:
                key=a['resource_id']
                if key in unique:need(unique[key]['resource_bytes']==a['resource_bytes'] and unique[key]['resource_allocated_size']==a['resource_allocated_size'],
                                      'One resource ID may not change length while deferred')
                else:unique[key]=a
            drains.append(dict(begin=opening,end=r,adds=pending,unique_resources=list(unique.values()),
                unique_requested_bytes=sum(a['resource_bytes']for a in unique.values()),
                unique_allocated_bytes=sum(a['resource_allocated_size']for a in unique.values()),
                observed_device_drop=opening['actual']['device_allocated_bytes']-r['actual']['device_allocated_bytes']))
            pending=[];opening=None
    need(not pending and opening is None,'Diagnostic must finish with every observed deferred hold drained')
    return dict(events=deferred,drains=drains)


def read_arm(directory,requests,expected_env,clear_mode,expected_model=None,request_original_paths=None):
    directory=Path(directory);data=rows(directory/'summary.jsonl')
    need(len(data)==3 and [v['kind']for v in data]==['metadata','sequence_complete','complete'],'Exactly one diagnostic sequence')
    meta,seq,complete=data
    need(meta['schema_version']==1 and meta['probe']=='moe_workspace_promptclear_fullsequence' and meta['phase']=='probe' and
         meta['workspace_mode']==1 and meta['prompt_view_clear_mode']==clear_mode and meta['implementation_version']==4 and
         meta['repeats']==0 and meta['warmup_sequences']==0 and meta['max_new_tokens']==8 and
         meta['diagnostic_prompt_mode'] is True and meta['deferred_lifetime_observer'] is True and
         meta['prompt_lifetime_observer'] is True and meta['growth_lifetime_observer'] is True,
         'Strict diagnostic contract; normal read_arm must remain separate')
    need(meta['fresh_context_per_sequence'] is True and meta['same_context_within_sequence'] is True and meta['manual_cache_or_rng_reset'] is False and
         meta['extra_gpu_waits_inserted']==0 and meta['testing_api_used'] is False and meta['observer_cpu_cost_in_generation_wall'] is True and
         meta['sampling_interval_ms']==10 and meta['performance_measured'] is False and
         (meta['max_seq_len'],meta['temperature'],meta['seed'])==(327680,0,42) and
         meta['total_wall_basis']=='sum_of_three_generation_calls_excluding_between_stage_exports','Unperturbed normal compute API and exact probe config')
    need(meta['environment']==expected_env and expected_env[FLAG]=='1' and expected_env[CLEAR]==str(clear_mode) and
         (expected_model is None or Path(meta['model']).resolve()==Path(expected_model).resolve()),'Actual complete diagnostic environment/model')
    need(seq['sequence']==0 and seq['warmup'] is False and seq['fresh_context'] is True and seq['same_context_within_sequence'] is True and
         seq['generations']==3 and seq['generation_outputs']==8 and complete['sequences']==1 and complete['generations']==3 and
         complete['max_new_tokens']==8 and complete['full_original_raw_exported'] is True and complete['adoption_gate_pass'] is False and
         complete['independent_review_required'] is True,'All three diagnostic generations complete')
    sd=directory/'seq-000';raw=rows(sd/'summary.jsonl');samples=[v for v in raw if v['kind']=='sample']
    need(len(raw)==12 and len(samples)==3 and [v['stage']for v in samples]==list(STAGES),'All three exact native stages')
    hashes=base.ByteHashes();stages=[];layouts=[];previous=None
    try:
        need([v['stage']for v in meta['requests']]==list(STAGES),'Actual request order')
        for request,stage in zip(meta['requests'],STAGES):
            p=Path(requests[stage]);identity=hashes.file(p);original=Path(request_original_paths[stage])if request_original_paths else p
            need(Path(request['path']).resolve()==original.resolve() and identity['bytes']==request['bytes'] and
                 identity['fnv1a64_bytes']==request['fnv1a64_bytes'],'Original diagnostic request bytes and FNV')
        for run,s in enumerate(samples):
            need(s['sequence']==0 and s['warmup'] is False and s['run']==run and s['same_context'] is True and
                 s['generation_ok'] is True and s['raw_state_exported'] is True and s['stop_reason']=='length' and
                 s['prompt_tokens']==(6917,15187,6917)[run] and s['output_tokens']==8 and
                 0<=s['cached_tokens']<s['prompt_tokens'] and s['prefill_tokens']==s['prompt_tokens']-1-s['cached_tokens'] and
                 s['decode_intervals']==7 and len(s['callback_intervals_ms'])==7 and
                 base.close(sum(s['callback_intervals_ms']),s['decode_s']*1000,1e-5) and
                 base.close(s['last_callback_s']-s['first_callback_s'],s['decode_s'],3e-9) and
                 all(type(s[k]) in (int,float) and math.isfinite(s[k]) and s[k]>0 for k in ('wall_s','prefill_s','decode_s')),
                 'Complete actual cache/output/callback diagnostic scope')
            base.workspace(s['workspace'],1,8192 if run==0 else 16384)
            before=base.read_layout(sd/f'layout-{STAGES[run]}-before.json',hashes);after=base.read_layout(sd/f'layout-{STAGES[run]}-after.json',hashes)
            base.layout_transition(before,after,run,1);base.counter_transition(previous,s['workspace'],before,after,1)
            if run:need(before['value']==layouts[-1][1]['value'],'Same context retains prior layout before next request')
            previous=s['workspace'];layouts.append((before,after))
            state=base.state(sd/'state'/STAGES[run],STAGES[run],s,hashes,8)
            stages.append(dict(stage=STAGES[run],sample=s,state=state,before_layout=before,after_layout=after))
        need(base.close(seq['total_wall_s'],sum(s['wall_s']for s in samples),3e-8) and
             seq['actual_B8192_to16384_growth'] is True and seq['actual_B16384_retained'] is True and
             seq['native_localgrow_identity_gate'] is True,'Actual compact growth/retention and complete wall accounting')
        memory=base.memory(raw,sd/'memory/samples.jsonl');moe=base.observer(sd/'growth-observer.jsonl',0,1,False,layouts)
        records=rows(sd/'prompt-lifetime.jsonl');need(records,'Complete diagnostic prompt/deferred records required')
        need([v['record_index']for v in records]==list(range(1,len(records)+1)) and
             all(v['sequence']==0 and v['stage']in range(3) and v['kind']in ('prompt_growth','deferred_buffer') for v in records) and
             all(a['actual']['snapshot_end_s']<=b['monotonic_s'] for a,b in zip(records,records[1:])),
             'One serial diagnostic callback stream with exact record coverage')
        for record in records:actual(record)
        groups=prompt_groups(records,clear_mode);deferred=deferred_groups(records)
        for run,stage in enumerate(stages):
            peaks=next(v['peaks']for v in memory['phases']if v['run']==run and v['phase']==2).copy()
            for record in records+moe['events']:
                if record['stage']==run and record.get('in_generation') is True:
                    peaks={k:max(peaks[k],record['actual'][k])for k in METRICS}
            stage['actual_peak_with_lifetime_events']=peaks
            for key in ('before_layout','after_layout'):
                stage[key]['value']={k:v for k,v in stage[key]['value'].items()if k!='bindings'}
                stage[key]['actual_bindings_fully_validated']=True
    finally:hashes.close()
    return dict(directory=str(directory),phase='probe',setting=1,clear_mode=clear_mode,gold=False,metadata=meta,
        summary_sha256=sha(directory/'summary.jsonl'),complete=True,sequences=[dict(sequence=0,warmup=False,
        total_wall_s=seq['total_wall_s'],stages=stages,memory=memory,observer=moe)],prompt=groups,deferred=deferred,
        diagnostic_records_sha256=sha(sd/'prompt-lifetime.jsonl'))


def lifetime(arm):
    group=arm['prompt'][1];ready,clear,free=group['checkpoints'];old=ready['old_owners'];ids={v['resource_id']for v in old}
    by_id={v['resource_id']:(v['bytes'],n)for v,n in zip(old,ready['old_allocated_bytes'])}
    holds=[];drains=[];seen=set();post_release_address_matches=[]
    for drain in arm['deferred']['drains']:
        if arm['clear_mode']:
            selected=[a for a in drain['adds']if a['resource_id']in ids and
                      ready['monotonic_s']<=a['monotonic_s']<=free['monotonic_s']]
            post_release_address_matches.extend(a for a in drain['adds']if a['resource_id']in ids and a['monotonic_s']>free['monotonic_s'])
        else:
            selected=[a for a in drain['adds']if a['resource_id']in ids and a['resource_id']not in seen and
                      a['stage']==1 and a['monotonic_s']>free['monotonic_s'] and
                      (a['resource_bytes'],a['resource_allocated_size'])==by_id[a['resource_id']]]
        if selected:
            unique_selected={a['resource_id']:a for a in selected}
            holds.extend(selected);drains.append(dict(begin=drain['begin'],end=drain['end'],old_prompt_entries=selected,
                old_prompt_unique_requested_bytes=sum(a['resource_bytes']for a in unique_selected.values()),
                old_prompt_unique_allocated_bytes=sum(a['resource_allocated_size']for a in unique_selected.values()),
                all_unique_requested_bytes=drain['unique_requested_bytes'],all_unique_allocated_bytes=drain['unique_allocated_bytes'],
                observed_device_drop=drain['observed_device_drop']))
            seen.update(a['resource_id']for a in selected)
    map_adds=[a for d in arm['deferred']['drains']for a in d['adds']if a['stage']==1 and a['origin']==3]
    map_ok=(len(map_adds)==1 and map_adds[0]['resource_bytes']==16777216 and map_adds[0]['resource_id']not in ids)
    target_drains=[]
    for drain in arm['deferred']['drains']:
        included={a['resource_id']:a for a in holds+map_adds if a in drain['adds']}
        if included:
            target_drains.append(dict(begin=drain['begin'],end=drain['end'],
                target_resource_ids=sorted(included),target_requested_bytes=sum(a['resource_bytes']for a in included.values()),
                target_allocated_bytes=sum(a['resource_allocated_size']for a in included.values()),
                observed_device_drop=drain['observed_device_drop'],
                interpretation='Synchronous whole-list release observation; other deferred resources may contribute. No requested-to-physical-byte equality is assumed.'))
    controls_ok=(set(a['resource_id']for a in holds)==ids and all(a['origin']==2 and a['in_generation'] is True and
                 a['monotonic_s']>free['monotonic_s'] for a in holds) and group['expected_release_observed'] and
                 all(d['end']['monotonic_s']>max(a['monotonic_s']for a in d['old_prompt_entries'])for d in drains))
    cleared_ok=(not holds and group['expected_release_observed'] and
                clear['actual']['device_allocated_bytes']==ready['actual']['device_allocated_bytes'])
    return dict(old_prompt_resource_ids=sorted(ids),old_requested_bytes=group['old_requested_size_sum'],
        old_actual_allocated_size_sum=group['actual_old_allocated_size_sum'],
        prompt_owner_release_device_drop=group['device_drop_after_old_owner_release'],
        old_prompt_deferred_holds=holds,old_prompt_drains=drains,
        post_release_pointer_address_matches_not_attributed_to_old_resources=post_release_address_matches,
        pointer_identity_scope='Opaque addresses identify resources only within a proven live interval; after clear1 release, later equal addresses may be reused.',
        id_map_adds=map_adds,id_map_16MiB_path_preserved=map_ok,
        target_prompt_and_id_map_drain_observations=target_drains,
        lifetime_prediction_pass=(controls_ok if arm['clear_mode']==0 else cleared_ok) and map_ok)


def causal_gate(C0a,C1,C0b):
    arms=(C0a,C1,C0b);need([v['clear_mode']for v in arms]==[0,1,0] and all(v['setting']==1 for v in arms),'Same-build workspace1 clear0/1/0 order')
    comparisons=[base.compare_arms(C0a,a)for a in arms]
    life=[lifetime(a)for a in arms]
    values=[a['sequences'][0]['stages'][1]['actual_peak_with_lifetime_events']for a in arms]
    peaks={k:dict(C0a=values[0][k],C1=values[1][k],C0b=values[2][k],C1_minus_C0a=values[1][k]-values[0][k],
       C1_minus_C0b=values[1][k]-values[2][k],conservative_reduction=min(values[0][k],values[2][k])-values[1][k])for k in METRICS}
    map_extents=[[(v['resource_bytes'],v['resource_allocated_size'])for v in q['id_map_adds']]for q in life]
    map_preserved=map_extents[0]==map_extents[1]==map_extents[2] and len(map_extents[0])==1
    passed=all(v['lifetime_prediction_pass']for v in life) and map_preserved
    return dict(evidence_valid=True,correctness_pass=True,causal_lifetime_gate_pass=passed,
        performance_measured=False,production_adoption=False,full_original_state_comparisons=comparisons,
        lifetime_by_arm=dict(zip(ORDER,life)),measured_peak_intervention=peaks,
        actual_id_map_requested_and_allocated_extents_preserved=map_preserved,
        observed_phys_peak_lower_than_both_controls=peaks['phys_footprint_bytes']['conservative_reduction']>0,
        decision='lifetime_intervention_verified'if passed else'lifetime_prediction_rejected_or_incomplete',
        limitations=['Resource identities and events test the prompt/deferred lifetime mechanism; byte-size equality alone is not proof.',
          'Task physical-footprint intervention is measured separately. No exact mapping of logical bytes to phys_footprint is assumed.',
          'Observer CPU/stdio cost is present; no GPU waits/commits are inserted. This is not a performance or adoption gate.'])


def post_window_review(root,source_archive,artifacts_archive,bindings_path,model=None):
    root=Path(root);before=load(root/'before.json');after=load(root/'restored.json');summary=load(root/'summary.json')
    need(all(before[k]==after[k]for k in base.IDENTITY_KEYS) and before['health']==after['health']=={'status':'ok'} and
         summary['restoration']['restored'] is True and summary['restoration']['pid']==after['pid'] and
         summary['restoration']['exact_argv_myds_artifacts_config'] is True,
         'Recorded exact finally production restoration')
    need(before['binary_sha256']==base.PRODUCTION_BINARY and before['metallib_sha256']==base.PRODUCTION_LIBRARY and
         sha(root/'production-artifacts/myds')==base.PRODUCTION_BINARY and
         sha(root/'production-artifacts/myds.metallib')==base.PRODUCTION_LIBRARY and
         sha(root/'original-myds.conf')==before['config_sha256'] and
         load(root/'effective-env.json')==before['effective_myds_env'],'Actual frozen production recovery bytes')
    archive=base.executed_archives(root,source_archive,artifacts_archive,bindings_path);index=archive['index'];small=archive['small']
    plan=load(root/'protocol.json');base.frozen_protocol(plan)
    need(summary['plan_sha256']==PLAN_SHA and summary['implementation_version']==4 and summary['mode']=='causal_diagnostic' and summary['performance_measured'] is False and
         summary['production_adoption'] is False,'Frozen diagnostic scope only')
    need(any(v['sha256']==PLAN_SHA and name in small and base.parse(small[name].decode())==plan for name,v in index.items()),
         'Actual immutable pre-data plan binding')
    records=load(root/'arms.json');need([r['label']for r in records]==list(ORDER) and
         [r['workspace_mode']for r in records]==[1,1,1] and [r['clear_mode']for r in records]==[0,1,0],
         'Exactly one diagnostic cohort/order')
    identity=summary['native_identity']['candidate']
    need(set(summary['native_identity'])=={'candidate'} and set(identity)==set((*CORE,NATIVE,'bin/myds.metallib')),
         'Diagnostic must bind all four candidate core sources, native and library; it has no gold run')
    stages=[Path(name).parent.parent for name,v in index.items()if name.endswith('/'+NATIVE)and v['sha256']==identity[NATIVE]]
    need(len(stages)==1,'Unique same-build diagnostic native');stage=stages[0]
    for rel,digest in identity.items():need(index[str(stage/rel)]['sha256']==digest,'Actual frozen source/bin/lib identity')
    need(identity['bin/myds.metallib']==base.PRODUCTION_LIBRARY and index[str(stage/'tools'/Path(__file__).name)]['sha256']==sha(Path(__file__)),
         'Frozen exact diagnostic reader and unchanged numerical shader')
    prereq=summary['prerequisites']['component_gate_pass'];path=prereq['path']
    need(path in small and index[path]['sha256']==prereq['sha256'],'Frozen prerequisite exact bytes')
    component=base.parse(small[path].decode())
    need(component['evidence_valid'] is True and component['component_gate_pass'] is True and component['implementation_version']==4,
         'Actual independent v4 real-weight component prerequisite')
    env=before['effective_myds_env'];need(len(env)==51 and env['MYDS_METAL_CONCURRENT']=='1','Actual production51 complete environment')
    need(component['plan_sha256']==PLAN_SHA and component['candidate_core_sha256']=={p:identity[p]for p in CORE} and
         component['metallib_sha256']==identity['bin/myds.metallib'] and component['production_environment']==env and
         component['model']==summary['model'] and component['model_header_sha256']==summary['model_header_sha256'],
         'Independent component exact same core/library/environment/model prerequisite')
    long=component['long_report'];need(long['path'] in index and index[long['path']]['sha256']==long['sha256'],
         'Original independent component long report is frozen with its small prerequisite')
    with tempfile.TemporaryDirectory(prefix='prompt-diag-request-CPU-')as temp:
        requests={};original={}
        for label,name in zip(STAGES,('pf7k','pf16k','pf7k')):
            digest=plan['fullmodel']['requests'][name]['sha256'];paths=[p for p,v in index.items()if p in small and v['sha256']==digest and Path(p).name==name+'.json']
            need(len(paths)==1,'Unique archived exact request');original[label]=paths[0]
            requests[label]=Path(temp)/(label+'.json');requests[label].write_bytes(small[paths[0]])
        arms=[]
        for record in records:
            directory=root/'arms'/record['label'];expected=dict(env,**{FLAG:'1',CLEAR:str(record['clear_mode'])})
            need(record['native_binary_sha256']==identity[NATIVE] and record['metallib_sha256']==identity['bin/myds.metallib'] and
                 sha(directory/'summary.jsonl')==record['native_summary_sha256'],'Same actual native/lib per arm')
            arm=read_arm(directory,requests,expected,record['clear_mode'],model or summary['model']['path'],original)
            need(Path(arm['metadata']['actual_metallib_path']).resolve()==(stage/'bin/myds.metallib').resolve(),'Actual diagnostic library')
            arms.append(arm)
    gate=causal_gate(*arms);need(summary['correctness_pass'] is True and summary['causal_lifetime_gate_pass']==gate['causal_lifetime_gate_pass'],
                              'Independent causal decision matches original raw controller')
    if model is not None:
        model=Path(model);need(model.stat().st_size==summary['model']['bytes'] and
          sha(model,1048576)==summary['model_header_sha256'],'Actual unchanged model extent/header')
    events=rows(root/'events.jsonl');need(events[-1]['kind']=='diagnostic_decision_complete' and events[-1]['status']==summary['status'] and
      events[-1]['evidence_valid'] is True and events[-1]['causal_lifetime_gate_pass']==gate['causal_lifetime_gate_pass'] and
      not any(v['kind']in ('monitor_failure','restoration_failure')for v in events),
                                         'Final diagnostic lifecycle/competition/restoration event')
    return dict(schema_version=1,evidence_valid=True,implementation_version=4,mode='causal_diagnostic',
        correctness_pass=True,causal_lifetime_gate_pass=gate['causal_lifetime_gate_pass'],performance_measured=False,
        production_adoption=False,plan_sha256=PLAN_SHA,reader_sha256=sha(Path(__file__)),gate=gate,arms=arms,
        archive_verification={k:v for k,v in archive.items()if k not in ('small','index')},frozen_inputs_verified=len(index),
        native_identity={'candidate':identity},production_environment=env,model=summary['model'],
        model_header_sha256=summary['model_header_sha256'],recorded_restoration_exact=True,before_pid=before['pid'],restored_pid=after['pid'],
        limitation='Independent CPU resource-lifetime/correctness diagnostic; no fullmodel performance/adoption result.')


def small_review(report,report_path=None):
    keys=('schema_version','evidence_valid','implementation_version','mode','correctness_pass','causal_lifetime_gate_pass',
          'performance_measured','production_adoption','plan_sha256','reader_sha256','native_identity',
          'production_environment','model','model_header_sha256','recorded_restoration_exact','before_pid','restored_pid')
    value={k:report[k]for k in keys if k in report}
    if report_path is not None:value['full_report']=dict(path=str(Path(report_path)),sha256=sha(Path(report_path)))
    if report.get('evidence_valid'):
        value['measured_peak_intervention']=report['gate']['measured_peak_intervention']
        value['observed_phys_peak_lower_than_both_controls']=report['gate']['observed_phys_peak_lower_than_both_controls']
        value['actual_id_map_requested_and_allocated_extents_preserved']=report['gate']['actual_id_map_requested_and_allocated_extents_preserved']
    if 'error'in report:value['error']=report['error']
    need(len(json.dumps(value).encode())<=102400,'Prerequisite top report must stay <=100KiB')
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','source-archive','artifacts-archive','out'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--archive-bindings',type=Path);p.add_argument('--model',type=Path);p.add_argument('--summary-out',type=Path);a=p.parse_args()
    try:value=post_window_review(a.root,a.source_archive,a.artifacts_archive,a.archive_bindings or a.source_archive.parent/'archive-bindings.json',a.model)
    except Exception as exc:value=dict(evidence_valid=False,correctness_pass=False,causal_lifetime_gate_pass=False,
      performance_measured=False,production_adoption=False,error=repr(exc),reader_sha256=sha(Path(__file__)))
    with a.out.open('x')as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
    if a.summary_out:
        with a.summary_out.open('x')as f:json.dump(small_review(value,a.out),f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({k:value[k]for k in ('evidence_valid','causal_lifetime_gate_pass','error')if k in value}))
    return 0 if value['evidence_valid']else 1


if __name__=='__main__':raise SystemExit(main())
