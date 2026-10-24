#!/usr/bin/env python3
"""CPU integrity/lifetime review of C0a after C1 failed before model entry.

This never supplies a causal prerequisite: it has only one control arm and no
clear1 intervention or cross-arm full-state comparison.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode=True


def review(root,reader,source,artifacts,bindings):
    s=importlib.util.spec_from_file_location('frozen_causal_CPU_reader',reader)
    r=importlib.util.module_from_spec(s);s.loader.exec_module(r)
    root=Path(root);before=r.load(root/'before.json');after=r.load(root/'restored.json');summary=r.load(root/'summary.json')
    r.need(all(before[k]==after[k]for k in r.base.IDENTITY_KEYS)and before['health']==after['health']=={'status':'ok'}and
        summary['restoration']['restored']is True and summary['restoration']['pid']==after['pid']and
        summary['restoration']['exact_argv_myds_artifacts_config']is True,'Recorded exact finally restoration')
    r.need(summary['status']=='failed'and summary['evidence_valid']is False and summary['causal_lifetime_gate_pass']is False and
        summary['plan_sha256']==r.PLAN_SHA and summary['mode']=='causal_diagnostic'and summary['performance_measured']is False,
        'Original failed diagnostic scope remains failed')
    steps=r.load(root/'steps.json');r.need([v['name']for v in steps]==['C0a','C1']and
        [v['returncode']for v in steps]==[0,2],'C0a complete; C1 nonzero before further arms')
    stderr=(root/'C1.stderr').read_bytes()
    r.need(b'Prompt clearing requires compact workspace mode'in stderr,'Preserve actual native entry guard failure')
    archive=r.base.executed_archives(root,source,artifacts,bindings);index=archive['index']
    r.need(index[str(Path(reader).resolve())]['sha256']==r.sha(Path(reader)),'Exact frozen causal reader used')
    identity=summary['native_identity']['candidate']
    candidates=[Path(p).parent.parent for p,v in index.items()if p.endswith('/'+r.NATIVE)and v['sha256']==identity[r.NATIVE]]
    r.need(len(candidates)==1,'Actual complete native identity');stage=candidates[0]
    r.need(all(index[str(stage/p)]['sha256']==digest for p,digest in identity.items()),'Frozen four-core/native/library binding')
    plan=r.load(root/'protocol.json');r.base.frozen_protocol(plan)
    original=[p for p,v in index.items()if Path(p).name=='experiment-plan.json'and v['sha256']==r.PLAN_SHA]
    r.need(len(original)==1 and r.base.parse(archive['small'][original[0]].decode())==plan,'Original pre-data plan/copy identity')
    directory=root/'arms/C0a';meta=r.rows(directory/'summary.jsonl')[0];requests={}
    for item,name in zip(meta['requests'],('pf7k','pf16k','pf7k')):
        path=Path(item['path']);r.need(r.sha(path)==plan['fullmodel']['requests'][name]['sha256'],'Actual frozen request bytes')
        requests[item['stage']]=path
    expected=dict(before['effective_myds_env'],**{r.FLAG:'1',r.CLEAR:'0'})
    arm=r.read_arm(directory,requests,expected,0,expected_model=summary['model']['path'])
    r.need(Path(arm['metadata']['actual_metallib_path']).resolve()==(stage/'bin/myds.metallib').resolve(),'Actual native library')
    record=r.load(root/'arms.json');r.need(len(record)==1 and record[0]['label']=='C0a'and
        record[0]['native_binary_sha256']==identity[r.NATIVE]and record[0]['native_summary_sha256']==arm['summary_sha256'],
        'Exactly one completed arm retained in original controller')
    life=r.lifetime(arm)
    result=dict(schema_version=1,mode='partial_causal_diagnostic_control',implementation_version=4,
        local_arm_integrity_valid=True,full_nine_generation_causal_experiment_complete=False,
        causal_lifetime_gate_pass=False,production_adoption=False,performance_measured=False,
        original_window_status='failed',original_failure=summary['error'],
        completed_control_generations=3,completed_intervention_generations=0,
        control_lifetime_prediction_pass=life['lifetime_prediction_pass'],control_lifetime=life,control=arm,
        frozen_native_identity={'candidate':identity},production_environment=before['effective_myds_env'],
        model=summary['model'],model_header_sha256=summary['model_header_sha256'],plan_sha256=r.PLAN_SHA,
        recorded_restoration=dict(before_pid=before['pid'],restored_pid=after['pid'],exact=True),
        archives={k:v for k,v in archive.items()if k not in ('small','index')},frozen_inputs_verified=len(index),
        frozen_causal_reader_sha256=r.sha(Path(reader)),partial_reader_sha256=r.sha(Path(__file__)),
        C1_stderr=dict(bytes=len(stderr),sha256=r.sha(root/'C1.stderr')),
        limitations=['C0a full raw-state self-integrity is checked, but no clear1 output or cross-arm0ULP comparison exists.',
            'One control resource lifetime chain can corroborate reachable retention; the failed intervention cannot establish a causal effect or physical-memory benefit.',
            'The original failed window, stage, native guard and reports remain unchanged. No new GPU/model execution is requested by this reader.'])
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','reader','source-archive','artifacts-archive','archive-bindings','out','summary-out'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();v=review(a.root,a.reader,a.source_archive,a.artifacts_archive,a.archive_bindings)
    with a.out.open('x')as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n')
    keys=('schema_version','mode','implementation_version','local_arm_integrity_valid','full_nine_generation_causal_experiment_complete',
        'causal_lifetime_gate_pass','production_adoption','performance_measured','original_window_status','original_failure',
        'completed_control_generations','completed_intervention_generations','control_lifetime_prediction_pass',
        'frozen_native_identity','production_environment','model','model_header_sha256','plan_sha256','recorded_restoration',
        'frozen_inputs_verified','frozen_causal_reader_sha256','partial_reader_sha256','C1_stderr','limitations')
    small={k:v[k]for k in keys};life=v['control_lifetime']
    small['control_lifetime']={k:life[k]for k in ('old_requested_bytes','old_actual_allocated_size_sum','prompt_owner_release_device_drop',
        'old_prompt_resource_ids','id_map_16MiB_path_preserved','lifetime_prediction_pass')}
    small['actual_grow_peaks']=v['control']['sequences'][0]['stages'][1]['actual_peak_with_lifetime_events']
    small['deferred_old_prompt_holds']=len(life['old_prompt_deferred_holds'])
    small['old_prompt_drains']=[dict(actual_drop=d['observed_device_drop'],
        target_prompt_allocated_bytes=d['old_prompt_unique_allocated_bytes'],commit_epoch=d['begin']['commit_epoch'])for d in life['old_prompt_drains']]
    small['long_report']=dict(path=str(a.out.resolve()),sha256=v['control']['metadata']and __import__('hashlib').sha256(a.out.read_bytes()).hexdigest())
    with a.summary_out.open('x')as f:json.dump(small,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({k:v[k]for k in ('local_arm_integrity_valid','control_lifetime_prediction_pass','causal_lifetime_gate_pass')}))


if __name__=='__main__':main()
