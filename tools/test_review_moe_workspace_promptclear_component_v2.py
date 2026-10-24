#!/usr/bin/env python3
"""Only the protocol copy identity correction; CPU metadata fixtures."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
sys.dont_write_bytecode=True
s=importlib.util.spec_from_file_location('v2reader',Path(__file__).with_name('review_moe_workspace_promptclear_component_v2.py'))
r=importlib.util.module_from_spec(s);s.loader.exec_module(r)


def run():
    controls=[]
    def test(name,fn,reject=False):
        try:fn();ok=not reject
        except (ValueError,KeyError,TypeError,OSError):ok=reject
        controls.append(dict(name=name,passed=ok));r.need(ok,'CPU control '+name)
    with tempfile.TemporaryDirectory(prefix='component-protocol-copy-CPU-')as temp:
        root=Path(temp);original=root/'experiment-plan.json'
        source=Path(__file__).parent.parent/'docs/benchmarks/2026-10-24-moe-phys-cause/experiment-plan.json'
        original.write_bytes(source.read_bytes());digest=r.sha(original)
        r.need(digest=='a83127b1427b09b703883150f3ec6e8e90e7ecdead3f48b798c7755368bd9d9c','Original frozen real plan SHA')
        index={str(original):digest};path=root/'stage-inputs.json';path.write_text(json.dumps(index));obj=r.load(original)
        test('same_original_object',lambda:r.frozen_plan_copy(root,obj))
        reordered=dict(reversed(list(obj.items())));(root/'protocol.json').write_text(json.dumps(reordered,sort_keys=True,indent=4)+'\n')
        test('whitespace_and_key_order_copy_positive',lambda:r.frozen_plan_copy(root,r.load(root/'protocol.json')))
        test('serialized_copy_has_different_bytes',lambda:r.need(r.sha(root/'protocol.json')!=digest,'Actual whitespace control'))
        q=copy.deepcopy(obj);q['component']['maximum_absolute_time_control_drift_percent']=3
        test('reject_changed_numeric_gate',lambda:r.frozen_plan_copy(root,q),True)
        q=copy.deepcopy(obj);q['capture']['rows']=16384
        test('reject_changed_capture_scope',lambda:r.frozen_plan_copy(root,q),True)
        path.write_text(json.dumps({str(original):'0'*64}));test('reject_wrong_frozen_plan_sha',lambda:r.frozen_plan_copy(root,obj),True)
        path.write_text(json.dumps(index));old=original.read_bytes();original.write_bytes(old+b' ')
        test('reject_original_bytes_modified_even_same_object',lambda:r.frozen_plan_copy(root,obj),True)
        original.write_bytes(old);original.unlink();test('reject_missing_original_plan',lambda:r.frozen_plan_copy(root,obj),True)
        original.write_bytes(old);duplicate=root/'other';duplicate.mkdir();other=duplicate/'experiment-plan.json';other.write_bytes(old)
        path.write_text(json.dumps(dict(index,**{str(other):digest})));test('reject_ambiguous_original_plan',lambda:r.frozen_plan_copy(root,obj),True)
        path.write_text('{}');test('reject_missing_original_plan_binding',lambda:r.frozen_plan_copy(root,obj),True)
    return dict(schema_version=1,scope='protocol_serialization_contract_CPU_controls_only',gpu_run=False,model_run=False,
        numerical_gate_changes=False,reader_sha256=r.sha(Path(r.__file__)),controls=controls,passed=all(v['passed']for v in controls))


if __name__=='__main__':print(json.dumps(run(),indent=2,sort_keys=True))
