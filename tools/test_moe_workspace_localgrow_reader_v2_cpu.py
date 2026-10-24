#!/usr/bin/env python3
"""CPU controls for the one-expression localgrow observer sequence correction."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def condition(source):
    tree=ast.parse(source)
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and
           n.func.id=='need' and len(n.args)==2 and isinstance(n.args[1],ast.Constant) and
           n.args[1].value=='Actual all-new-live/before-old-free and after-release process/device observations required']
    assert len(calls)==1
    return compile(ast.Expression(calls[0].args[0]),'<exact-reader-observer-condition>','eval')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    original=(root/'review_moe_workspace_localgrow_artifacts.py').read_bytes()
    revised=(root/'review_moe_workspace_localgrow_artifacts_v2.py').read_bytes()
    before=b"ev[0]['sequence']==ev[1]['sequence']"
    after=b"ev[1]['sequence']==ev[0]['sequence']+1"
    assert digest(original)=='97dd6361453b8093e696ba6c2e97951fe6e06d412a81b6761b8a8cc3dca1a18b'
    assert original.count(before)==1 and revised.count(after)==1
    assert original.replace(before,after)==revised and revised.replace(after,before)==original
    old,new=condition(original),condition(revised)
    records=[]
    def check(name,ev,expected,expression=new):
        result=bool(eval(expression,dict(ev=ev,r={'before_layout':{'cw_id':2}})))
        records.append(dict(case=name,accepted=result,expected_accept=expected,passed=result==expected))
    def fixture(first=1):
        return [dict(event=event,sequence=first+i,monotonic_s=100+i/1000,cpu_only=True,gpu_waits=0,
                     capacity=2048,old_batch=8192,new_batch=16384,cw_id=2,
                     actual_memory=dict(metal_current_allocated_size=15460089856,
                                        process_phys_footprint=3007354280,process_resident_size=6587711488))
                for i,event in enumerate((1,2))]
    for first in (1,3,5):
        check('actual_event_sequences_'+str(first)+'_'+str(first+1),fixture(first),True)
        check('frozen_original_reproduces_sequence_contract_failure_'+str(first),fixture(first),False,old)
    for name,mutation in (
        ('equal_sequence_rejected',lambda e:e[1].update(sequence=1)),
        ('reversed_sequence_rejected',lambda e:e[1].update(sequence=0)),
        ('sequence_gap_rejected',lambda e:e[1].update(sequence=3)),
        ('event_order_rejected',lambda e:e[0].update(event=2)),
        ('backwards_time_rejected',lambda e:e[1].update(monotonic_s=99)),
        ('wrong_capacity_rejected',lambda e:e[1].update(capacity=32)),
        ('wrong_old_batch_rejected',lambda e:e[1].update(old_batch=16384)),
        ('wrong_CW_identity_rejected',lambda e:e[1].update(cw_id=3)),
        ('non_CPU_observer_rejected',lambda e:e[1].update(cpu_only=False)),
        ('GPU_wait_rejected',lambda e:e[1].update(gpu_waits=1)),
        ('zero_actual_device_observation_rejected',lambda e:e[1]['actual_memory'].update(metal_current_allocated_size=0)),
        ('zero_actual_phys_observation_rejected',lambda e:e[1]['actual_memory'].update(process_phys_footprint=0)),
        ('zero_actual_RSS_observation_rejected',lambda e:e[1]['actual_memory'].update(process_resident_size=0))):
        ev=copy.deepcopy(fixture());mutation(ev);check(name,ev,False)
    result=dict(schema_version=1,all_pass=all(r['passed'] for r in records),controls=records,
                controls_count=len(records),CPU_only=True,GPU_or_model_executed=False,
                original_reader_sha256=digest(original),revised_reader_sha256=digest(revised),
                exact_single_expression_change=True,reverse_restore_original_bytes=True,
                previous_predicate=before.decode(),revised_predicate=after.decode(),
                scope='Observer event sequence is strictly consecutive; all other conditions and thresholds remain exact original bytes.')
    with args.out.open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True);stream.write('\n')
    print(json.dumps({k:result[k] for k in ('all_pass','controls_count','revised_reader_sha256')}))
    return 0 if result['all_pass'] else 1


if __name__=='__main__':raise SystemExit(main())
