#!/usr/bin/env python3
"""CPU-only independent reader for isolated actual-Metal prompt transaction raw."""
import argparse,hashlib,json
from pathlib import Path
class Rejected(ValueError):pass
def require(value,message):
    if not value:raise Rejected(message)
def obj(p):
    def pairs(a):
        d={}
        for k,v in a:
            if k in d:raise Rejected('Duplicate key '+k)
            d[k]=v
        return d
    return json.loads(p.read_text(),object_pairs_hook=pairs)
def identity(x,*,view):
    require(set(x)=={'wrapper_id','resource_id','offset','bytes','is_view'},'Identity keys')
    require(all(type(v) is int and v>=0 for v in x.values()),'Identity integer fields')
    require(x['wrapper_id']>0 and x['is_view']==int(view),'Wrapper/view kind')
def inspect(root):
    records_path=root/'records.jsonl';lines=records_path.read_text().splitlines()
    require(lines and all(lines),'Nonempty complete JSONL')
    # Duplicate JSON keys are rejected independently of the native summary.
    records=[]
    for line in lines:
        def pairs(a):
            d={}
            for k,v in a:
                if k in d:raise Rejected('Duplicate key '+k)
                d[k]=v
            return d
        records.append(json.loads(line,object_pairs_hook=pairs))
    complete=records.pop();require(complete['kind']=='complete' and complete['status']=='complete' and complete['gate_pass'] is True and complete['cases']==9,'Final completion')
    require(complete['snapshot_or_result_records']==len(records),'Raw record count')
    summary=obj(root/'summary.json');require(summary==dict(status='complete',implementation_version=4,prompt_allocations_gate_pass=True,cases=9),'Native summary')
    expected=[('allocation_null',i) for i in range(1,7)]+[('clear_rejected',0),('clear_success',0),('clear_disabled',0)]
    checked=[];cursor=0
    for name,ordinal in expected:
        snapshots=[]
        while cursor<len(records) and records[cursor]['kind']=='snapshot':snapshots.append(records[cursor]);cursor+=1
        require(cursor<len(records),'Missing case result');result=records[cursor];cursor+=1
        require(result['kind']=='case_result' and result['case']==name and result['ordinal']==ordinal and result['pass'] is True and result['cleanup_exact'] is True,'Case order/result')
        require(type(result['device_before']) is int and 0<=result['device_before']<2**64-1 and result['device_before']==result['device_after_free'],'Actual device cleanup')
        points=['before','failed','retry'] if ordinal or name=='clear_rejected' else ['before','success']
        require([s['point'] for s in snapshots]==points,'Snapshot point order')
        for s in snapshots:
            require(s['case']==name and s['ordinal']==ordinal and s['implementation_version']==4 and s['quiescent']==1,'Snapshot identity')
            require(len(s['owners'])==len(s['views'])==len(s['allocated_bytes'])==6,'Six bindings')
            for x in s['owners']:identity(x,view=False)
            for x in s['views']:identity(x,view=True)
            require(len({x['resource_id'] for x in s['owners']})==6 and all(x['resource_id']>0 for x in s['owners']),'Six distinct actual owners')
            sizes=[s['cap']*81920,s['cap']*96,s['cap']*2048,s['cap']*s['mask_bytes'],s['cap']*24576,s['cap']*24576]
            require([x['bytes'] for x in s['owners']]==sizes and s['total_bytes']==sum(sizes),'Requested extents')
            require(all(type(a) is int and a>=b for a,b in zip(s['allocated_bytes'],sizes)),'Actual allocated extents')
        before,after=snapshots[0],snapshots[-1]
        require(before['cap']==4 and before['mask_bytes']==4 and after['cap']==8 and after['mask_bytes']==8,'Capacity contract')
        require(all(v['resource_id']==o['resource_id'] and v['bytes']==o['bytes'] and v['offset']==0 for v,o in zip(before['views'],before['owners'])),'Actual old-parent aliases')
        if ordinal or name=='clear_rejected':
            failure=snapshots[1]
            for key in ('cap','mask_bytes','total_bytes','owners','views','allocated_bytes'):require(failure[key]==before[key],'Failed transaction mutated '+key)
            require(failure['allocation_count']==6 and failure['injected_ordinal']==ordinal and failure['injected_hits']==int(ordinal>0),'Actual scoped injection')
        require(after['allocation_count']==6 and after['injected_ordinal']==0 and after['injected_hits']==0,'Retry/success fault disabled')
        require(all(a['resource_id']!=b['resource_id'] for a,b in zip(after['owners'],before['owners'])),'New owner generation')
        require([v['wrapper_id'] for v in after['views']]==[v['wrapper_id'] for v in before['views']],'Reusable wrappers stable')
        if name=='clear_disabled':require(after['views']==before['views'],'Clear0 control lifetime')
        else:require(all(v['resource_id']==v['offset']==v['bytes']==0 for v in after['views']),'Atomic six-view clear')
        checked.append(dict(case=name,ordinal=ordinal,pass_check=True))
    require(cursor==len(records),'Unexpected extra cases')
    return dict(schema_version=1,evidence_valid=True,prompt_allocations_gate_pass=True,implementation_version=4,cases=checked,
        scope='Actual-Metal diagnostic raw transaction, exact source/bin/window binding is supplied by the enclosing window reviewer',
        raw_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (records_path,root/'summary.json')})
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():p.error('New output required')
    try:r=inspect(a.root)
    except Exception as e:r=dict(schema_version=1,evidence_valid=False,prompt_allocations_gate_pass=False,error=repr(e))
    with a.out.open('x') as f:json.dump(r,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:r[k] for k in ('evidence_valid','prompt_allocations_gate_pass')},sort_keys=True));return 0 if r['evidence_valid'] else 1
if __name__=='__main__':raise SystemExit(main())
