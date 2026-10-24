#!/usr/bin/env python3
"""CPU controls for current-plan selection among retained frozen history."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import tempfile


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    tool=Path(__file__).with_name('review_moe_workspace_fullmodel_artifacts_v2.py')
    spec=importlib.util.spec_from_file_location('fullmodel_plan_v2',tool);r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
    controls=[]
    def test(name,fn,accept=True):
        error=None
        try:fn();actual=True
        except (ValueError,KeyError,OSError)as e:actual=False;error=str(e)
        controls.append(dict(name=name,expected_accept=accept,accepted=actual,passed=accept==actual,error=error))
    with tempfile.TemporaryDirectory(prefix='moe-fullmodel-plan-v2-')as temp:
        root=Path(temp);v2=dict(experiment='Provider-specific prefill MoE scratch arena',schema_version=2,cpu_fixture=True)
        v1=dict(v2,schema_version=1)
        def fixture(history=True,duplicate=False):
            entries={}
            for name,value in [('component.json',dict(evidence_valid=True,component_gate_pass=True)),('recovery.json',dict(evidence_valid=True,recovery_gate_pass=True)),('plan-v2.json',v2)]:
                path=root/name;path.write_text(json.dumps(value));entries[str(path)]=r.sha(path)
            if history:
                path=root/'plan-v1.json';path.write_text(json.dumps(v1));entries[str(path)]=r.sha(path)
            if duplicate:
                path=root/'plan-v2-copy.json';path.write_text(json.dumps(v2));entries[str(path)]=r.sha(path)
            (root/'protocol.json').write_text(json.dumps(v2));(root/'summary.json').write_text(json.dumps(dict(plan_sha256=r.sha(root/'plan-v2.json'))))
            (root/'stage-inputs.json').write_text(json.dumps(entries));return entries
        def save(entries):(root/'stage-inputs.json').write_text(json.dumps(entries))
        fixture()
        def positive():
            x=r.approval_inputs(root);r.need(x['plan_path']==str(root/'plan-v2.json')and len(x['retained_historical_plans'])==1,'Wrong chosen current/history')
        test('unique_v2_with_frozen_v1_history_accepted',positive)
        fixture(history=False);test('unique_v2_without_history_accepted',lambda:r.approval_inputs(root))
        fixture(duplicate=True);test('reject_two_content_matching_v2_paths',lambda:r.approval_inputs(root),False)
        entries=fixture();del entries[str(root/'plan-v2.json')];save(entries)
        test('reject_only_v1_history',lambda:r.approval_inputs(root),False)
        entries=fixture();entries[str(root/'plan-v2.json')]='0'*64;save(entries)
        test('reject_v2_sha_mismatch',lambda:r.approval_inputs(root),False)
        entries=fixture();entries[str(root/'plan-v1.json')]='0'*64;save(entries)
        test('reject_retained_v1_sha_mismatch',lambda:r.approval_inputs(root),False)
        fixture();(root/'summary.json').write_text(json.dumps(dict(plan_sha256='0'*64)))
        test('reject_controller_frozen_plan_sha_mismatch',lambda:r.approval_inputs(root),False)
        fixture();changed=copy.deepcopy(v2);changed['threshold_changed']=True;(root/'protocol.json').write_text(json.dumps(changed))
        test('reject_protocol_content_mismatch',lambda:r.approval_inputs(root),False)
    result=dict(schema_version=1,scope='CPU frozen-plan selection controls only',gpu_or_model_executed=False,
                reader_sha256=r.sha(tool),controls=controls,all_pass=all(c['passed']for c in controls))
    with a.out.open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(all_pass=result['all_pass'],controls=len(controls))));return 0 if result['all_pass']else 1


if __name__=='__main__':raise SystemExit(main())
