#!/usr/bin/env python3
"""Mocked causal-controller CPU controls; no subprocess, inference or service."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

sys.dont_write_bytecode=True
spec=importlib.util.spec_from_file_location('causal_controller',Path(__file__).with_name('run_moe_prompt_lifetime_diagnostic_window.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def run():
    controls=[]
    def test(name,fn,reject=False):
        try:fn();ok=not reject
        except (ValueError,RuntimeError,KeyError,TypeError,OSError):ok=reject
        controls.append(dict(name=name,passed=ok));m.need(ok,'CPU control '+name)
    with tempfile.TemporaryDirectory(prefix='causal-controller-CPU-')as temp:
        root=Path(temp);stage=root/'stage';stage.mkdir();(stage/'bin').mkdir()
        paths=(*m.CORE,m.NATIVE,'bin/myds.metallib','tools/bench_moe_workspace_fullmodel_promptclear.c',
               'tools/moe_prompt_lifetime_diagnostic.inc','tools/review_moe_prompt_lifetime_diagnostic.py',
               'tools/run_moe_prompt_lifetime_diagnostic_window.py')
        for rel in paths:
            p=stage/rel;p.parent.mkdir(exist_ok=True);p.write_text('CPU fake artifact, not executable\n')
        component=root/'component.json'
        plan=json.loads((Path(__file__).parent.parent/'docs/benchmarks/2026-10-24-moe-phys-cause/experiment-plan.json').read_text())
        env={'MYDS_CPU_FIXTURE_'+str(i):'1'for i in range(50)};env['MYDS_METAL_CONCURRENT']='1'
        long_report=root/'original-independent-long.json';long_report.write_text('{"CPU_fixture_only":true}\n')
        good_component=dict(evidence_valid=True,component_gate_pass=True,implementation_version=4,plan_sha256=m.workspace.PLAN_SHA,
            candidate_core_sha256={p:m.sha(stage/p)for p in m.CORE},metallib_sha256=m.sha(stage/'bin/myds.metallib'),
            model=dict(path='CPU-model-not-loaded',bytes=123),model_header_sha256='CPU-header-not-model-evidence',production_environment=env,
            long_report=dict(path=str(long_report),sha256=m.sha(long_report),bytes=long_report.stat().st_size))
        component.write_text(json.dumps(good_component))
        def instance(name):
            out=root/name;out.mkdir();o=object.__new__(m.DiagnosticWindow)
            o.a=SimpleNamespace(stage=stage,out=out,component_review=component,inputs=root,model=root/'CPU-model',prepare=False)
            o.before=dict(effective_myds_env=copy.deepcopy(env));o.environment=copy.deepcopy(env)
            o.plan=copy.deepcopy(plan);o.summary={k:good_component[k]for k in ('model','model_header_sha256')};o.input_hashes={str(stage/p):m.sha(stage/p)for p in paths}
            o.input_hashes[str(Path(m.__file__).resolve())]=m.sha(Path(m.__file__))
            o.input_hashes[str(Path(m.review.__file__).resolve())]=m.sha(Path(m.review.__file__))
            o.event=lambda *args,**kwargs:None;return o
        symbols='myds_v41_prompt_workspace_growth_observer_set\nmyds_metal_deferred_buffer_observer_set\n'
        def preflight(o,text=symbols):
            with patch.object(m.workspace.WorkspaceWindow,'preflight',lambda self:None),patch.object(m.workspace.safety,'command',lambda argv:text):o.preflight()
        o=instance('preflight-positive');test('strict_preflight_cpu_mock',lambda:preflight(o))
        o=instance('preflight-fault-symbol');test('reject_testing_symbol',lambda:preflight(o,symbols+'myds_v41_prompt_workspace_test_control'),True)
        o=instance('preflight-missing-hook');o.input_hashes.pop(str(stage/'tools/moe_prompt_lifetime_diagnostic.inc'))
        test('reject_missing_actual_include_inventory',lambda:preflight(o),True)
        component.write_text(json.dumps(dict(good_component,implementation_version=3)))
        o=instance('preflight-old-impl');test('reject_previous_component_implementation',lambda:preflight(o),True)
        component.write_text(json.dumps(dict(good_component,component_gate_pass=False)))
        o=instance('preflight-component-negative');test('reject_negative_component_gate',lambda:preflight(o),True)
        for field,new_value in (('candidate_core_sha256',{}),('metallib_sha256','wrong'),('model',{}),('model_header_sha256','wrong'),('production_environment',{})):
            component.write_text(json.dumps(dict(good_component,**{field:new_value})))
            o=instance('preflight-component-'+field);test('reject_component_identity_'+field,lambda:preflight(o),True)
        component.write_text(json.dumps(dict(good_component,long_report=dict(path=str(long_report),sha256='0'*64))))
        o=instance('preflight-long-sha');test('reject_original_component_long_SHA_changed',lambda:preflight(o),True)
        component.write_text(json.dumps(good_component))
        o=instance('component');preflight(o);calls=[]
        test('freeze_original_component_long_bytes',lambda:m.need(o.input_hashes[str(long_report)]==m.sha(long_report),'frozen original long'))
        def fake_command(label,argv,env,timeout):
            calls.append(dict(label=label,argv=list(map(str,argv)),env=env,timeout=timeout))
            directory=o.a.out/'arms'/label;directory.mkdir();(directory/'summary.jsonl').write_text('CPU fixture raw only\n')
        def fake_read(directory,requests,expected,clear,expected_model):
            m.need(expected==dict(env,MYDS_PREFILL_MOE_WORKSPACE='1',MYDS_PREFILL_PROMPT_VIEW_CLEAR=str(clear)),'Retain entire51 fixture keys')
            return dict(metadata=dict(actual_metallib_path=str(stage/'bin/myds.metallib')),clear_mode=clear)
        o.run_command=fake_command
        with patch.object(m.review,'read_arm',fake_read),patch.object(m.review,'causal_gate',lambda *arms:dict(evidence_valid=True,causal_lifetime_gate_pass=False)):
            test('complete_three_arm_negative_is_retained',o.component)
        test('samebuild_clear0_1_0_only',lambda:m.need([v['label']for v in calls]==['C0a','C1','C0b']and [v['env'][m.workspace.CLEAR_FLAG]for v in calls]==['0','1','0']and all(v['env'][m.workspace.FLAG]=='1'for v in calls),'order'))
        test('diagnostic_flag_and_no_formal_samples',lambda:m.need(all(v['argv'][-5:]==['--phase','probe','--repeats','0','--diagnostic-prompt-observer']for v in calls),'exactargv'))
        test('negative_lifetime_is_not_correctness_failure',lambda:m.need(o.summary['correctness_pass']is True and o.summary['causal_lifetime_gate_pass']is False and o.summary['generations_completed']==9,'validnegative'))
        def finished(name,lifetime,error=False,restored=True,arms=3):
            o=instance(name);o.summary=dict(status='component_complete_review_required',diagnostic_native_complete=True,
                causal_lifetime_gate_pass=lifetime,correctness_pass=True,restoration=dict(restored=restored))
            if error:o.summary['error']='CPU injected failure'
            o.arms=[{}]*arms;o.gate={};o.native_identity={'candidate':{}};o.prerequisites={}
            with patch.object(m.workspace.WorkspaceWindow,'run',lambda self:0):code=o.run()
            return code,json.loads((o.a.out/'summary.json').read_text()),json.loads((o.a.out/'causal-review.json').read_text())
        test('completed_negative_is_valid_no_adoption',lambda:m.need((lambda q:q[0]==0 and q[1]['evidence_valid']is True and q[1]['causal_lifetime_gate_pass']is False and q[1]['production_adoption']is False)(finished('negative',False)),'negative'))
        test('completed_positive_is_causal_only',lambda:m.need((lambda q:q[0]==0 and q[1]['causal_lifetime_gate_pass']is True and q[1]['production_adoption']is False and q[2]['performance_measured']is False)(finished('positive',True)),'diagnostic'))
        test('reject_error_even_zero_returncode',lambda:m.need(finished('error',True,error=True)[0]==1,'error'))
        test('reject_failed_finally',lambda:m.need(finished('restore',True,restored=False)[0]==1,'restore'))
        test('reject_partial_three_arm_coverage',lambda:m.need(finished('partial',True,arms=2)[0]==1,'partial'))
    return dict(schema_version=1,scope='mocked_controller_CPU_only',controller_sha256=m.sha(Path(m.__file__)),
        workspace_helper_sha256=m.sha(Path(m.workspace.__file__)),actual_three_key_long_reference_positive_control=True,
        reader_sha256=m.sha(Path(m.review.__file__)),gpu_run=False,model_run=False,subprocesses_run=False,service_operations=False,
        controls=controls,passed=all(v['passed']for v in controls))


if __name__=='__main__':print(json.dumps(run(),indent=2,sort_keys=True))
