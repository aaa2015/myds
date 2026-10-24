#!/usr/bin/env python3
"""Small CPU-only normal-probe summary, preserving the complete reader report."""
import argparse
import hashlib
import json
from pathlib import Path


METRICS=('phys_footprint_bytes','rss_bytes','device_allocated_bytes')
STAGES=('setup','grow','retain')
IDENTITY=('present','capacity','moe_batch','cw_id','moe_generation','localgrow_attempts',
          'localgrow_successes','localgrow_failures','stable_owner_fingerprint','stable_view_fingerprint',
          'stable_owner_bytes','stable_owners','stable_views','down_id','down_bytes','arena_id','acc_ids','acc_bytes')


def need(value,message):
    if not value:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()


def summarize(value):
    need(value['evidence_valid'] is True and value['correctness_pass'] is True and value['mode']=='probe' and
         value['implementation_version']==3 and value['performance_measured'] is False and
         value['production_adoption'] is False,'Complete independent normal probe only')
    arms=value['arms'];need(len(arms)==3 and [a['setting'] for a in arms]==[0,1,0] and
         all(a['phase']=='probe' and not a['gold'] and len(a['sequences'])==1 for a in arms),'Exactly9 generation normal ABA scope')
    rows=[];grow=[];coverage=True;comparison_count=0;comparison_bytes=0
    for label,arm in zip(('A1','B','A2'),arms):
        seq=arm['sequences'][0];need(seq['sequence']==0 and seq['warmup'] is False and len(seq['stages'])==3,'One normal sequence per arm')
        coverage=coverage and seq['observer']['coverage'] is True
        for run,stage in enumerate(seq['stages']):
            need(stage['stage']==STAGES[run] and stage['sample']['output_tokens']==8,'Complete original8-token stage')
            phase=next(p for p in seq['memory']['phases'] if p['run']==run and p['phase']==2)
            events=[e for e in seq['observer']['events'] if e['stage']==run]
            peak={k:max([phase['peaks'][k]]+[e['actual'][k] for e in events]) for k in METRICS}
            need(peak==stage['actual_peak_with_lifetime_events'],'Published combined stage peak differs from samples plus events')
            before,after=stage['before_layout'],stage['after_layout'];a,b=before['value'],after['value']
            need(before['actual_bindings_fully_validated'] is True and after['actual_bindings_fully_validated'] is True,
                 'Independent full binding validation required')
            stable=None
            if label=='B' and run in (1,2):
                stable=all(a[k]==b[k] for k in ('cw_id','capacity','stable_owner_fingerprint','stable_view_fingerprint',
                           'stable_owner_bytes','stable_owners','stable_views'))
                need(stable,'Actual candidate non-MoE identity changed')
            s=stage['sample']
            rows.append(dict(arm=label,setting=arm['setting'],stage=STAGES[run],
                prompt_tokens=s['prompt_tokens'],cached_tokens=s['cached_tokens'],prefill_tokens=s['prefill_tokens'],output_tokens=8,
                wall_s_diagnostic_only=s['wall_s'],prefill_s_diagnostic_only=s['prefill_s'],decode_s_diagnostic_only=s['decode_s'],
                observed_generation_samples=phase['observed_samples'],maximum_observed_gap_ms=phase['maximum_observed_gap_ms'],
                sampled_generation_peak=phase['peaks'],actual_peak_with_lifetime_events=peak,
                before_layout={k:a.get(k) for k in IDENTITY},after_layout={k:b.get(k) for k in IDENTITY},
                full_binding_tuples_validated=True,candidate_non_MoE_identity_stable=stable,
                lifetime_events=[{k:e[k] for k in ('event','growth_sequence','monotonic_s','cw_id','old_arena_id',
                    'new_arena_id','old_down_id','new_down_id','stable_owner_bytes','old_pool_bytes','new_pool_bytes','actual')} for e in events]))
            if run==1:grow.append(peak)
    peaks={k:dict(A1=grow[0][k],B=grow[1][k],A2=grow[2][k],B_minus_A1=grow[1][k]-grow[0][k],
                  B_minus_A2=grow[1][k]-grow[2][k],conservative_excess=grow[1][k]-min(grow[0][k],grow[2][k])) for k in METRICS}
    gate=coverage and all(peaks[k]['conservative_excess']<=134217728 for k in ('phys_footprint_bytes','device_allocated_bytes'))
    need(value['gate']['actual_growth_peaks']==peaks and value['gate']['known_lifetime_coverage']==coverage and
         value['gate']['growth_gate_pass']==gate==value['growth_gate_pass'],'Recomputed actual memory gate differs')
    for group in value['raw_comparisons']:
        need(group['full_original_bytes_equal'] is True,'Complete raw output/state oracle required')
        for c in group['comparisons']:
            need(c['full_original_bytes_equal'] is True,'Original state comparison failed')
            comparison_count+=1;comparison_bytes+=c['bytes_compared']
    need(len(value['raw_comparisons'])==3 and comparison_count==9,'Exactly three arms times three full state comparisons')
    return dict(schema_version=1,scope='Small summary of complete independently replayed normal localgrow probe',
        evidence_valid=True,correctness_pass=True,implementation_version=3,performance_measured=False,production_adoption=False,
        plan_sha256=value['plan_sha256'],decision=value['decision'],growth_gate_pass=gate,known_lifetime_coverage=coverage,
        generations=9,outputs_per_generation=8,actual_growth_peaks=peaks,arm_stage_observations=rows,
        state_comparison_groups=len(value['raw_comparisons']),full_raw_state_comparisons=comparison_count,
        full_raw_state_bytes_compared=comparison_bytes,full_original_state_bytes_equal=True,
        native_identity=value['native_identity'],before_pid=value['before_pid'],restored_pid=value['restored_pid'],
        recorded_restoration_exact=value['recorded_restoration_exact'],frozen_logical_inputs_verified=value['frozen_logical_inputs_verified'],
        boundaries=['Times are actual single probe generation observations, not formal performance estimates.',
          'Memory peaks combine10ms generation samples with known synchronous allocation lifetime checkpoints; exact continuous physical residency is unproven.',
          'Actual Metal allocation and task phys/RSS are separate observed metrics; requested pool formulas are not memory evidence.',
          'Opaque resource IDs and full binding tuples are compared only within a process; cross-arm IDs have no equality requirement.',
          'Old full-CW v2 and new localgrow v3 are different cohorts. Their absolute peaks or internal excesses cannot be combined into a causal same-cohort gain.',
          'The complete original independent reader report and all raw files remain preserved.'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();result=summarize(json.loads(a.report.read_text()))
    result['complete_reader_report']=dict(path=str(a.report),bytes=a.report.stat().st_size,sha256=sha(a.report))
    result['summary_tool_sha256']=sha(Path(__file__))
    content=json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n'
    need(len(content.encode())<=102400,'Summary must remain at most100KiB; preserve the complete report separately')
    with a.out.open('x') as stream:stream.write(content)
    print(json.dumps({k:result[k] for k in ('evidence_valid','growth_gate_pass','generations','performance_measured')}))


if __name__=='__main__':main()
