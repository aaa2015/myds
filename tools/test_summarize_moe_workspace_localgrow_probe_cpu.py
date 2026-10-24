#!/usr/bin/env python3
"""Small synthetic CPU controls for the bounded normal-probe summary."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import summarize_moe_workspace_localgrow_probe as s


def fixture():
    arms=[]
    for setting in (0,1,0):
        stages=[];phases=[]
        for run,name in enumerate(s.STAGES):
            value={k:1000-setting*100 for k in s.METRICS}
            before=dict(present=run>0,cw_id=10,capacity=2048,moe_batch=8192 if run<2 else 16384,
              stable_owner_fingerprint=20,stable_view_fingerprint=30,stable_owner_bytes=40,stable_owners=3,stable_views=4)
            after=dict(before,present=True,moe_batch=8192 if run==0 else 16384)
            stages.append(dict(stage=name,sample=dict(output_tokens=8,prompt_tokens=(6917,15187,6917)[run],
              cached_tokens=0,prefill_tokens=(6916,15186,6916)[run],wall_s=2,prefill_s=1,decode_s=0.5),
              actual_peak_with_lifetime_events=value,before_layout=dict(value=before,actual_bindings_fully_validated=True),
              after_layout=dict(value=after,actual_bindings_fully_validated=True)))
            phases.append(dict(run=run,phase=2,peaks=value,observed_samples=20,maximum_observed_gap_ms=11))
        arms.append(dict(phase='probe',setting=setting,gold=False,sequences=[dict(sequence=0,warmup=False,
          stages=stages,memory=dict(phases=phases),observer=dict(coverage=True,events=[]))]))
    peaks={k:dict(A1=1000,B=900,A2=1000,B_minus_A1=-100,B_minus_A2=-100,conservative_excess=-100) for k in s.METRICS}
    return dict(evidence_valid=True,correctness_pass=True,mode='probe',implementation_version=3,
      performance_measured=False,production_adoption=False,arms=arms,gate=dict(actual_growth_peaks=peaks,
      known_lifetime_coverage=True,growth_gate_pass=True),growth_gate_pass=True,plan_sha256='CPU',decision='CPU',
      raw_comparisons=[dict(full_original_bytes_equal=True,comparisons=[dict(full_original_bytes_equal=True,
       bytes_compared=32) for _ in range(3)]) for _ in range(3)],native_identity={},before_pid=1,restored_pid=2,
      recorded_restoration_exact=True,frozen_logical_inputs_verified=268)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);a=parser.parse_args()
    checks=[]
    def test(name,value,accepted):
        try:s.summarize(value);actual=True
        except (ValueError,KeyError,StopIteration):actual=False
        checks.append(dict(case=name,accepted=actual,expected_accept=accepted,passed=actual==accepted))
    value=fixture();test('complete9generation_actual_peak_and_ID_summary',value,True)
    bad=copy.deepcopy(value);bad['gate']['actual_growth_peaks']['device_allocated_bytes']['B']+=1
    test('reject_published_peak_not_recomputed_from_arm',bad,False)
    bad=copy.deepcopy(value);bad['growth_gate_pass']=False;test('reject_gate_mismatch',bad,False)
    bad=copy.deepcopy(value);bad['arms'][1]['sequences'][0]['stages'][1]['after_layout']['value']['cw_id']=999
    test('reject_candidate_CW_replacement',bad,False)
    bad=copy.deepcopy(value);bad['raw_comparisons'].pop();test('reject_missing_full_state_arm',bad,False)
    bad=copy.deepcopy(value);bad['performance_measured']=True;test('reject_formal_performance_relabelled_probe',bad,False)
    negative=copy.deepcopy(value)
    for arm in negative['arms']:arm['sequences'][0]['observer']['coverage']=False
    negative['gate']['known_lifetime_coverage']=False;negative['gate']['growth_gate_pass']=False;negative['growth_gate_pass']=False
    test('valid_negative_coverage_preserved_as_reject',negative,True)
    result=dict(schema_version=1,all_pass=all(c['passed'] for c in checks),CPU_only=True,GPU_or_model_executed=False,
      checks=checks,summary_source_sha256=hashlib.sha256(Path(s.__file__).read_bytes()).hexdigest(),
      positive_summary_bytes=len(json.dumps(s.summarize(value)).encode()))
    with a.out.open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True);stream.write('\n')
    print(json.dumps(result));return 0 if result['all_pass'] else 1


if __name__=='__main__':raise SystemExit(main())
