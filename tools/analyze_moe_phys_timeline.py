#!/usr/bin/env python3
"""Read existing v3 probe samples and actual bindings; never run engine/GPU."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ARMS=('A1','B','A2');STAGES=('setup','grow','retain')
METRICS=('phys_footprint_bytes','rss_bytes','device_allocated_bytes')
PHASES={0:'transition',1:'request_prepare',2:'generation',3:'raw_export'}
PROMPT_NAMES=('cur_res_view','cur_pre_view','cur_selected_view','cur_mask_view','cur_engram_view[0]','cur_engram_view[1]')


def need(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb')as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def parse(text):
    def pairs(items):
        d={}
        for k,v in items:need(k not in d,'Duplicate JSON key');d[k]=v
        return d
    return json.loads(text,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))


def load(path):return parse(path.read_text())
def rows(path):return [parse(x)for x in path.read_text().splitlines()if x.strip()]


def extent(points,t0):
    need(points,'Missing phase observations');times=[p['monotonic_s']for p in points]
    result={}
    for key in METRICS:
        low=min(points,key=lambda p:p['value'][key]);high=max(points,key=lambda p:p['value'][key])
        result[key]=dict(first=points[0]['value'][key],minimum=low['value'][key],maximum=high['value'][key],
            last=points[-1]['value'][key],maximum_monotonic_s=high['monotonic_s'],
            maximum_after_first_generation_sample_s=high['monotonic_s']-t0,
            minimum_monotonic_s=low['monotonic_s'],last_minus_first=points[-1]['value'][key]-points[0]['value'][key])
    gaps=[b-a for a,b in zip(times,times[1:])]
    return dict(count=len(points),first_monotonic_s=times[0],last_monotonic_s=times[-1],
        observed_duration_s=times[-1]-times[0],median_observation_gap_s=statistics.median(gaps)if gaps else None,
        maximum_observation_gap_s=max(gaps)if gaps else None,metrics=result)


def analyze(raw,out):
    inputs={};arms={};all_samples=[];all_events=[];all_steps=[];prompt={}
    def frozen(path):
        inputs[str(path)]=dict(bytes=path.stat().st_size,sha256=sha(path));return path
    for arm in ARMS:
        root=raw/'arms'/('probe-'+arm)/'seq-000';points=rows(frozen(root/'memory/samples.jsonl'))
        events=rows(frozen(root/'growth-observer.jsonl'));native=rows(frozen(root/'summary.jsonl'))
        need(all(p['kind']=='memory_sample'and p['run']in range(3)and p['phase']in range(4)and
            p['value']['task_valid']is True and p['value']['device_valid']is True and
            all(type(p['value'][k])is int and p['value'][k]>0 for k in METRICS)for p in points),'Actual complete task/device sampler')
        need(all(a['monotonic_s']<=b['monotonic_s']for a,b in zip(points,points[1:])),'Sampler timestamps reversed')
        samples=[q for q in native if q['kind']=='sample']
        need([q['stage']for q in samples]==list(STAGES),'Complete three native stages')
        stages=[]
        for run,stage in enumerate(STAGES):
            generation=[p for p in points if p['run']==run and p['phase']==2];t0=generation[0]['monotonic_s']
            groups={PHASES[phase]:extent([p for p in points if p['run']==run and p['phase']==phase],t0)
                for phase in range(4)if any(p['run']==run and p['phase']==phase for p in points)}
            observed=[e for e in events if e['stage']==run and e['in_generation']is True]
            combined={k:max([p['value'][k]for p in generation]+[e['actual'][k]for e in observed])for k in METRICS}
            steps=[]
            for a,b in zip(generation,generation[1:]):
                row=dict(arm=arm,stage=stage,previous_monotonic_s=a['monotonic_s'],monotonic_s=b['monotonic_s'],
                    after_first_generation_sample_s=b['monotonic_s']-t0,observation_gap_s=b['monotonic_s']-a['monotonic_s'],
                    **{k:b['value'][k]for k in METRICS},**{k+'_delta':b['value'][k]-a['value'][k]for k in METRICS})
                steps.append(row);all_steps.append(row)
            biggest={key:dict(largest_drops=sorted(steps,key=lambda p:p[key+'_delta'])[:8],
                largest_increases=sorted(steps,key=lambda p:p[key+'_delta'],reverse=True)[:8])for key in METRICS}
            stages.append(dict(stage=stage,run=run,first_generation_sample_monotonic_s=t0,
                phase_observations=groups,generation_peak_including_actual_MoE_events=combined,
                native_generation={k:samples[run][k]for k in ('prompt_tokens','cached_tokens','prefill_tokens','output_tokens','wall_s','prefill_s','decode_s')},
                actual_MoE_events=observed,largest_consecutive_generation_sample_steps=biggest))
            for p in points:
                if p['run']==run:
                    all_samples.append(dict(arm=arm,stage=stage,phase=p['phase'],phase_label=PHASES[p['phase']],
                        monotonic_s=p['monotonic_s'],after_first_generation_sample_s=p['monotonic_s']-t0,**p['value']))
            for e in observed:all_events.append(dict(arm=arm,stage_label=stage,after_first_generation_sample_s=e['monotonic_s']-t0,**e))
        before=load(frozen(root/'layout-grow-before.json'));after=load(frozen(root/'layout-grow-after.json'))
        bb={b['name']:b for b in before['bindings']};ab={b['name']:b for b in after['bindings']}
        stable=[b for b in before['bindings']if b['scope']in (1,2)]
        stable_equal=sum(b==ab.get(b['name'])for b in stable)
        prompt[arm]=dict(cw_before=before['cw_id'],cw_after=after['cw_id'],
            stable_before_count=len(stable),stable_same_actual_descriptor_count=stable_equal,
            stable_bytes_before=sum(b['bytes']for b in stable if b['scope']==1),
            stable_bytes_after=sum(b['bytes']for b in after['bindings']if b['scope']==1),
            six_views=[dict(name=name,before=bb[name],after=ab[name],
                wrapper_preserved=bb[name]['wrapper_id']==ab[name]['wrapper_id'],
                parent_replaced=bb[name]['resource_id']!=ab[name]['resource_id'])for name in PROMPT_NAMES])
        arms[arm]=dict(sample_count=len(points),first_monotonic_s=points[0]['monotonic_s'],
            last_monotonic_s=points[-1]['monotonic_s'],stages=stages)
    grow={arm:arms[arm]['stages'][1]for arm in ARMS}
    differences={k:dict(**{arm:grow[arm]['generation_peak_including_actual_MoE_events'][k]for arm in ARMS},
        B_minus_A1=grow['B']['generation_peak_including_actual_MoE_events'][k]-grow['A1']['generation_peak_including_actual_MoE_events'][k],
        B_minus_A2=grow['B']['generation_peak_including_actual_MoE_events'][k]-grow['A2']['generation_peak_including_actual_MoE_events'][k])for k in METRICS}
    bgen=grow['B']['phase_observations']['generation']['metrics'];drop=bgen['device_allocated_bytes']['maximum']-bgen['device_allocated_bytes']['last']
    requested=dict(RES=7168*81920,PRE=7168*96,SELECTED=7168*2048,MASK=7168*112,
        ENGRAM0=7168*24576,ENGRAM1=7168*24576)
    formula=dict(warm_cap=7168,mask_bytes_per_token=112,requested_prompt_lengths=requested,
        requested_prompt_sum=sum(requested.values()),hypothesized_old_id_map_requested_bytes=16777216,
        predicted_sum=sum(requested.values())+16777216,observed_sampled_Metal_peak_to_last=drop,
        exact_numeric_match=drop==sum(requested.values())+16777216,
        status='Requested-size and source-derived hypothesis only; v3 lacks actual allocatedSize/old-owner ADD/DRAIN identity logs.')
    proof=dict(old_raw_mutated=False,model_or_GPU_run=False,remote_access=False,performance_measured=False,
        phase2_peak_does_not_overlap_its_own_phase3_export=True,
        prior_exports_and_request_preparation_can_affect_page_history=True,
        sampling_basis='Relative time starts at the first observed phase2 sample, not the unobserved exact generation entry.',
        same_CW_nonMoE_bindings_reused=(prompt['B']['cw_before']==prompt['B']['cw_after']and
            prompt['B']['stable_before_count']==prompt['B']['stable_same_actual_descriptor_count']),
        missing_from_v3_raw=['per-resource allocatedSize','actual six prompt owner-ID replacement events',
            'per-resource deferred ADD/DRAIN/commit epoch','task_vm internal/external/compressed/graphics ledgers',
            'continuous hardware residency or physical-page attribution'])
    report=dict(schema_version=1,scope='read_only_existing_v3_complete_sample_timeline',
        analyzer_sha256=sha(Path(__file__)),inputs=inputs,arms=arms,actual_binding_reuse=prompt,
        independent_grow_peak_comparison=differences,size_match_hypothesis=formula,evidence_boundaries=proof,
        hypotheses=[
          dict(id='H1',mechanism='Retained CW prompt views extend old prompt-parent lifetime into active CB deferred lists.',
            already_observed='Six old nonzero parent IDs before grow, new parents after grow; size/timing matches972472320B.',
            discriminator='Same-build clear0/1/0; actual old owner IDs match cur parents, clear1 removes six refs at quiescence; exact actual allocatedSize releases at AFTER_FREE; controls ADD originVIEW_RETARGET and later drain. Keep16MiB map origin separately.'),
          dict(id='H2',mechanism='Different touched/resident pages or task/driver accounting lag explains part of phys/Metal divergence.',
            already_observed='At known MoE release Metal drops2408710144B while same-point phys/RSS stay constant; later large phys changes can occur with constant Metal.',
            discriminator='Record actual task_vm internal/external/compressed/reusable/graphics ledgers at the same CPU lifetime events and following samples. Device/requested bytes are not physical bytes.'),
          dict(id='H3',mechanism='Other scratch growth or prior raw-export CPU/file-cache history contributes to the physical peak.',
            already_observed='Candidate stable C owners/fixed views remain unchanged; expected B-dependent pools change. Grow peak is during phase2, but setup exports occurred earlier.',
            discriminator='Actual binding inventory constrains C scratch; independent retained/grow sample trajectory and CPU ledger changes test attribution. A no-export run would be a separate protocol and cannot rewrite original gate.')],
        conclusion='Original v3 observed phys gate remains failed. Static reachability plus exact size/time correlation motivates a controlled lifetime intervention; this analyzer does not prove the514589440B phys increase belongs entirely to those buffers.')
    out.mkdir(parents=True,exist_ok=False)
    def save(name,value):
        with(out/name).open('x')as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
    save('v3-memory-timeline.json',report)
    for name,data in (('all-memory-samples.csv',all_samples),('all-generation-steps.csv',all_steps)):
        with(out/name).open('x',newline='')as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    save('all-growth-observer-events.json',all_events)
    need(all(p.stat().st_size==v['bytes']and sha(p)==v['sha256']for name,v in inputs.items()for p in [Path(name)]),'Old raw bytes changed during CPU analysis')
    small=dict(schema_version=1,evidence_valid=True,raw_inputs_unchanged=True,scope=report['scope'],
        analyzer_sha256=report['analyzer_sha256'],sample_counts={a:arms[a]['sample_count']for a in ARMS},
        independent_grow_peak_comparison=differences,size_match_hypothesis=formula,evidence_boundaries=proof,
        report=dict(path=str(out/'v3-memory-timeline.json'),sha256=sha(out/'v3-memory-timeline.json')))
    save('v3-memory-timeline-summary.json',small)
    return small


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out-dir',type=Path,required=True);a=p.parse_args()
    print(json.dumps(analyze(a.root.resolve(),a.out_dir.resolve()),indent=2,sort_keys=True))


if __name__=='__main__':main()
