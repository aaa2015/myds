#!/usr/bin/env python3
import copy,importlib.util,json,tempfile
from pathlib import Path
spec=importlib.util.spec_from_file_location('reader',Path(__file__).with_name('review_prompt_workspace_allocations.py'));reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
def fixture():
 rows=[]
 def snap(name,ordinal,point,cap,new=False,disabled=False):
  sizes=[cap*81920,cap*96,cap*2048,cap*(4 if cap==4 else 8),cap*24576,cap*24576]
  owners=[dict(wrapper_id=1000+i+(100 if new else 0),resource_id=2000+i+(100 if new else 0),offset=0,bytes=b,is_view=0) for i,b in enumerate(sizes)]
  oldsizes=[4*81920,4*96,4*2048,4*4,4*24576,4*24576]
  views=[dict(wrapper_id=3000+i,resource_id=2000+i if not new or disabled else 0,offset=0,bytes=oldsizes[i] if not new or disabled else 0,is_view=1) for i in range(6)]
  hit=ordinal if point=='failed' else 0
  return dict(kind='snapshot',case=name,point=point,ordinal=ordinal,allocation_count=6,injected_ordinal=hit,injected_hits=int(hit>0),implementation_version=4,cap=cap,mask_bytes=4 if cap==4 else 8,total_bytes=sum(sizes),quiescent=1,owners=owners,views=views,allocated_bytes=[(b+16383)//16384*16384 for b in sizes])
 for name,ordinal in [('allocation_null',i) for i in range(1,7)]+[('clear_rejected',0),('clear_success',0),('clear_disabled',0)]:
  rows.append(snap(name,ordinal,'before',4))
  if ordinal or name=='clear_rejected':rows.append(snap(name,ordinal,'failed',4));point='retry'
  else:point='success'
  rows.append(snap(name,ordinal,point,8,True,name=='clear_disabled'))
  rows.append(dict(kind='case_result',case=name,ordinal=ordinal,pass_check=True,**{'pass':True},device_before=1000000,device_after_free=1000000,cleanup_exact=True))
 rows.append(dict(kind='complete',status='complete',gate_pass=True,cases=9,snapshot_or_result_records=len(rows)))
 return rows
with tempfile.TemporaryDirectory(prefix='prompt-reader-cpu-') as tmp:
 p=Path(tmp);summary=dict(status='complete',implementation_version=4,prompt_allocations_gate_pass=True,cases=9);(p/'summary.json').write_text(json.dumps(summary));tests=[]
 def run(name,rows,want):
  (p/'records.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows));ok=True
  try:reader.inspect(p)
  except Exception:ok=False
  assert ok==want,(name,ok,want);tests.append(dict(name=name,passed=True))
 run('complete_positive',fixture(),True)
 cases=[('missing_case',lambda x:x.pop(0)),('actual_injection_wrong',lambda x:x[1].__setitem__('injected_ordinal',6)),('old_owner_changed_on_failure',lambda x:x[1]['owners'][0].__setitem__('resource_id',99999)),('partialclear_before_failure',lambda x:x[1]['views'][0].__setitem__('resource_id',0)),('actualallocated_too_small',lambda x:x[0]['allocated_bytes'].__setitem__(0,1)),('retained_after_clear',lambda x:x[2]['views'][0].__setitem__('resource_id',2000)),('wrapper_identity_changed',lambda x:x[2]['views'][0].__setitem__('wrapper_id',999)),('device_cleanup_leak',lambda x:x[3].__setitem__('device_after_free',1000016)),('false_native_cleanup',lambda x:x[3].__setitem__('cleanup_exact',False)),('short_attempt_count',lambda x:x[1].__setitem__('allocation_count',1)),('native_result_false',lambda x:x[3].__setitem__('pass',False))]
 for name,mutate in cases:
  data=copy.deepcopy(fixture());mutate(data);run(name,data,False)
 report=dict(schema_version=1,scope='Synthetic raw reader CPU false-accept controls only; no Metal/GPU/model',all_pass=True,controls=tests)
 out=Path('/tmp/myds-moe-phys-cause-868d-20261024-v4-prototype/promptclear-evidence/reader-cpu-controls.json');out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(all_pass=True,controls=len(tests),out=str(out))))
