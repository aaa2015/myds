#!/usr/bin/env python3
"""Small prerequisite bound to a frozen, independently reviewed v4 component.

The original long reader/report and all executed inputs stay unchanged. This
CPU-only identity wrapper rechecks the executed archive path map and attaches
the actual core/library/model/environment to the existing component decision.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tarfile

sys.dont_write_bytecode=True
s=importlib.util.spec_from_file_location('normal_archive_CPU_helpers',Path(__file__).with_name('review_moe_workspace_fullmodel_promptclear.py'))
r=importlib.util.module_from_spec(s);s.loader.exec_module(r)
PLAN_SHA='a83127b1427b09b703883150f3ec6e8e90e7ecdead3f48b798c7755368bd9d9c'
CORE=('src/myds_v41.c','src/myds_metal.m','include/myds_internal.h','include/myds_metal.h')
NATIVE='bin/bench_moe_workspace_promptclear'


def summarize(root,source,artifacts,bindings,long_report,model,corrected_reader=None,correction_proof=None,original_failure=None):
    root=Path(root);long_report=Path(long_report);model=Path(model)
    value=r.load(long_report);summary=r.load(root/'summary.json');before=r.load(root/'before.json')
    r.need(value['evidence_valid']is True and value['artifact_review_passed']is True and
        value['implementation_version']==4 and type(value['component_gate_pass'])is bool and
        value['fullmodel_run']is False and value['production_adoption']is False,'Complete original independent v4 component review')
    archive=r.executed_archives(root,source,artifacts,bindings);index=archive['index']
    cm=r.load(root/'correctness/artifact-manifest.json')
    candidates=[Path(p).parent.parent for p,v in index.items()if p.endswith('/'+NATIVE)and v['sha256']==cm['binary_sha256']]
    r.need(len(candidates)==1,'Unique actual candidate component native');stage=candidates[0]
    lib=index[str(stage/'bin/myds.metallib')]['sha256']
    r.need(lib==cm['metallib_sha256']==before['metallib_sha256']==r.PRODUCTION_LIBRARY,'Actual unchanged library')
    reader=stage/'tools/review_moe_workspace_promptclear_component.py'
    original_reader_sha=index[str(reader)]['sha256'];correction=None
    if original_reader_sha!=value['reader_sha256']:
        r.need(all(p is not None for p in (corrected_reader,correction_proof,original_failure)),'Preserved correction source/proof/original failure required')
        proof=r.load(correction_proof);failure=r.load(original_failure)
        r.need(proof['old_reader_sha256']==original_reader_sha and proof['new_reader_sha256']==value['reader_sha256']==r.sha(corrected_reader) and
            proof['reverse_restore_byte_exact']is True and proof['original_reader_modified']is False and proof['raw_modified']is False and
            proof['numerical_thresholds_modified']is False and failure['artifact_review_passed']is False and
            'Exact frozen v4 plan required'in failure['error'],'Correction lineage must preserve original failure and all gates')
        text=Path(corrected_reader).read_text()
        begin=text.index('def frozen_plan_copy(root,protocol):\n');end=text.index('def review(root,model,source_archive,artifacts_archive):\n',begin)
        new="    need(frozen_plan_copy(root,plan) and plan['implementation_version']==4,'Exact frozen v4 plan required')\n"
        old="    need(sha(root/'protocol.json')=='a83127b1427b09b703883150f3ec6e8e90e7ecdead3f48b798c7755368bd9d9c' and plan['implementation_version']==4,'Exact frozen v4 plan required')\n"
        r.need(text.count(new)==1,'Only the protocol predicate is corrected')
        restored=(text[:begin]+text[end:]).replace(new,old).encode()
        with tarfile.open(source,'r:*')as tar:
            original_source=tar.extractfile(index[str(reader)]['tar_path']).read()
        r.need(restored==original_source,'Corrected reader must reverse to the complete original frozen source byte for byte')
        correction=dict(source=dict(path=str(corrected_reader),sha256=r.sha(corrected_reader)),
            reverse_proof=dict(path=str(correction_proof),sha256=r.sha(correction_proof)),
            original_failure=dict(path=str(original_failure),sha256=r.sha(original_failure)),
            reverse_restore_byte_exact=True,numerical_thresholds_modified=False)
    original_plans=[p for p,v in index.items()if Path(p).name=='experiment-plan.json'and v['sha256']==PLAN_SHA]
    r.need(len(original_plans)==1 and summary['plan_sha256']==PLAN_SHA and
        r.parse(archive['small'][original_plans[0]].decode())==r.load(root/'protocol.json') and
        r.load(root/'protocol.json')['implementation_version']==4,'Original immutable plan SHA and exact serialized-copy object')
    r.need(Path(summary['model']['path']).resolve(strict=True)==model.resolve(strict=True) and
        model.stat().st_size==summary['model']['bytes'] and r.sha(model,1048576)==summary['model_header_sha256'],
        'Actual full-interval-reviewed model retained, unchanged extent/header')
    env=before['effective_myds_env']
    r.need(len(env)==51 and env['MYDS_METAL_CONCURRENT']=='1'and
        'MYDS_PREFILL_MOE_WORKSPACE'not in env and 'MYDS_PREFILL_PROMPT_VIEW_CLEAR'not in env,
        'Complete original51 production environment without experimental flags')
    r.need(summary['restoration']['restored']is True and summary['restoration']['pid']==value['restored_pid'],
        'Original component finally restoration bound to independent long report')
    for report,path in zip(value['archives'],(source,artifacts)):
        r.need(report['sha256']==r.sha(path),'Small wrapper must refer to original long-reviewed archives')
    result=dict(schema_version=1,evidence_valid=True,component_gate_pass=value['component_gate_pass'],
        implementation_version=4,plan_sha256=PLAN_SHA,candidate_core_sha256={p:index[str(stage/p)]['sha256']for p in CORE},
        metallib_sha256=lib,model=summary['model'],model_header_sha256=summary['model_header_sha256'],
        production_environment=env,long_report=dict(path=str(long_report.resolve()),bytes=long_report.stat().st_size,sha256=r.sha(long_report)),
        executed_archives=[dict(path=str(p),sha256=r.sha(p))for p in (source,artifacts,bindings)],
        original_pid=value['original_pid'],restored_pid=value['restored_pid'],
        original_reader_sha256=original_reader_sha,independent_reader_sha256=value['reader_sha256'],
        reader_correction=correction,identity_wrapper_sha256=r.sha(Path(__file__)),
        model_backed_full_weight_intervals_verified_by_long_reader=True,weight_payload_exported=False,
        performance_scope='Existing real fixture GPU component only; separate causal/recovery/normal/fullmodel gates required.',
        production_adoption=False)
    r.need(len(json.dumps(result).encode())<=102400,'Component prerequisite must be <=100KiB');return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','source-archive','artifacts-archive','archive-bindings','long-report','model','out'):
        p.add_argument('--'+name,type=Path,required=True)
    for name in ('corrected-reader','correction-proof','original-failure'):p.add_argument('--'+name,type=Path)
    a=p.parse_args();v=summarize(a.root,a.source_archive,a.artifacts_archive,a.archive_bindings,a.long_report,a.model,
        a.corrected_reader,a.correction_proof,a.original_failure)
    with a.out.open('x')as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({k:v[k]for k in ('evidence_valid','component_gate_pass','implementation_version')}))


if __name__=='__main__':main()
