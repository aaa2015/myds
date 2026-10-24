#!/usr/bin/env python3
"""Bounded CPU metadata/raw controls; never accepted as model/GPU evidence."""
import copy,importlib.util,json,struct,sys,tempfile
from pathlib import Path
sys.dont_write_bytecode=True
s=importlib.util.spec_from_file_location('reader',Path(__file__).with_name('review_moe_workspace_fullmodel_promptclear.py'));r=importlib.util.module_from_spec(s);s.loader.exec_module(r)
def run():
 controls=[]
 def test(name,fn,reject=False):
  try:fn();ok=not reject
  except (ValueError,KeyError,TypeError,OSError):ok=reject
  controls.append(dict(name=name,passed=ok));r.need(ok,'CPU control '+name)
 with tempfile.TemporaryDirectory(prefix='localgrow-reader-controls-')as temp:
  root=Path(temp);h=r.ByteHashes()
  try:
   bindings=[dict(name='Cowner',kind=1,scope=1,index=0,wrapper_id=10,resource_id=11,offset=0,bytes=16,is_view=0),dict(name='Cview',kind=2,scope=2,index=0,wrapper_id=12,resource_id=11,offset=4,bytes=4,is_view=1),dict(name='dynamic',kind=2,scope=5,index=0,wrapper_id=13,resource_id=11,offset=8,bytes=4,is_view=1),dict(name='down',kind=1,scope=3,index=0,wrapper_id=14,resource_id=15,offset=0,bytes=32,is_view=0)]
   v=dict(schema_version=1,scope='process_local_actual_binding_identity',complete=True,present=True,implementation_version=4,capacity=2048,moe_batch=8192,compact=1,cw_id=1,moe_generation=1,localgrow_attempts=0,localgrow_successes=0,localgrow_failures=0,stable_owner_fingerprint=int(h.bytes(struct.pack('<3Q',10,11,16)),16),stable_view_fingerprint=int(h.bytes(struct.pack('<2Q',12,13)),16),stable_owner_bytes=16,stable_owners=1,stable_views=2,down_id=14,down_bytes=32,arena_id=20,acc_ids=list(range(30,37)),acc_bytes=[32]*7,binding_count=len(bindings),bindings=bindings)
   path=root/'layout.json';path.write_text(json.dumps(v));test('actual_binding_fingerprint',lambda:r.read_layout(path,h));before=r.read_layout(path,h)
   bad=copy.deepcopy(v);bad['bindings'][0]['resource_id']+=1;path.write_text(json.dumps(bad));test('reject_cached_fingerprint_lie',lambda:r.read_layout(path,h),True)
   path.write_text(json.dumps(v));grow=copy.deepcopy(before);g=grow['value'];g.update(moe_batch=16384,moe_generation=2,localgrow_attempts=1,localgrow_successes=1,down_id=114,arena_id=120,acc_ids=list(range(130,137)));g['bindings'][2].update(resource_id=115,offset=0,bytes=0);g['bindings'][3].update(wrapper_id=114,resource_id=115,bytes=64)
   test('localgrow_nonmoe_and_dynamic_wrapper',lambda:r.layout_transition(before,grow,1,1))
   bad=copy.deepcopy(grow);bad['value']['bindings'][1]['offset']+=1;test('reject_fixed_view_changed',lambda:r.layout_transition(before,bad,1,1),True)
   bad=copy.deepcopy(grow);bad['value']['bindings'][2]['wrapper_id']+=1;test('reject_dynamic_wrapper_replaced',lambda:r.layout_transition(before,bad,1,1),True)
   test('retain_all_actual_bindings',lambda:r.layout_transition(grow,copy.deepcopy(grow),2,1))
   bad=copy.deepcopy(grow);bad['value']['bindings'][3]['resource_id']+=1;test('reject_retain_pool_changed',lambda:r.layout_transition(grow,bad,2,1),True)
   old=dict(requests=40,mm_encoded=39,mv_encoded=1,arena=dict(requests=40,mm_encoded=39,mv_encoded=1));new=dict(requests=100,mm_encoded=98,mv_encoded=2,arena=dict(requests=60,mm_encoded=59,mv_encoded=1))
   test('cw_lifetime_vs_new_arena_delta',lambda:r.counter_transition(old,new,before,grow,1))
   bad=copy.deepcopy(new);bad['arena']['requests']=100;test('reject_reusing_v2_arena_equality',lambda:r.counter_transition(old,bad,before,grow,1),True)
   test('duplicate_json_rejected',lambda:r.parse('{"x":1,"x":2}'),True)
   left=root/'a';right=root/'b';left.write_bytes(bytes(range(64)));right.write_bytes(left.read_bytes());test('full_original_bytes_equal',lambda:r.need(r.file_difference(left,right)['equal'],'raw'))
   right.write_bytes(right.read_bytes()[:-1]+b'x');test('reject_last_raw_byte',lambda:r.need(r.file_difference(left,right)['equal'],'raw'),True)
  finally:h.close()
 # New normal-only prompt lifetime evidence controls use CPU metadata fixtures.
 with tempfile.TemporaryDirectory(prefix='promptclear-observer-cpu-')as temp:
  path=Path(temp)/'prompt-observer.jsonl'
  def ident(w,res,n,view=0):return dict(wrapper_id=w,resource_id=res,offset=0,bytes=n,is_view=view)
  def fixture():
   out=[]
   for stage in (0,1):
    old=[ident(100+i,200+i,16)for i in range(6)]if stage else[ident(0,0,0)for i in range(6)]
    owners=[ident(300+stage*10+i,400+stage*10+i,32)for i in range(6)]
    views=[ident(500+i,200+i if stage else 0,8 if stage else 0,1)for i in range(6)]
    empty=[ident(500+i,0,0,1)for i in range(6)]
    for event in (1,2,3):
     idx=stage*3+event
     out.append(dict(kind='prompt_growth',event=event,sequence=0,stage=stage,in_generation=True,record_index=idx,monotonic_s=float(idx),implementation_version=4,compact=1,clear_mode=1,quiescent=1,clear_applied=int(event>1),old_cap=7168 if stage else 0,new_cap=15360 if stage else 7168,prompt_sequence=idx,cw_id=1,old_mask_bytes=112 if stage else 0,new_mask_bytes=240 if stage else 112,old_total_bytes=96 if stage else 0,new_total_bytes=192,owner_order=['RES','PRE','SELECTED','MASK','ENGRAM0','ENGRAM1'],old_owners=old,new_owners=owners,views_before=views,views_current=views if event==1 else empty,old_allocated_bytes=[16 if stage else 0]*6,new_allocated_bytes=[32]*6,observer_CPU_only=True,GPU_work_submitted_by_callback=False,extra_gpu_waits=0,actual=dict(task_valid=True,device_valid=True,phys_footprint_bytes=100,rss_bytes=200,device_allocated_bytes=300,task_info_count_words=100,snapshot_begin_s=idx+.001,snapshot_end_s=idx+.002)))
   return out
  layouts=[(None,dict(value=dict(cw_id=1)))for _ in range(3)]
  def check(v):path.write_text(''.join(json.dumps(x)+'\n'for x in v));return r.prompt_observer(path,0,1,False,layouts)
  v=fixture();test('all_prompt_lifetime_points_and_clear_identity',lambda:check(v))
  bad=copy.deepcopy(v);bad[0]['new_allocated_bytes'][5]=0;test('reject_unmeasured_prompt_allocated_size',lambda:check(bad),True)
  bad=copy.deepcopy(v);bad.pop(1);test('reject_missing_prompt_clear_checkpoint',lambda:check(bad),True)
  bad=copy.deepcopy(v);bad[4]['views_current'][0]['resource_id']=200;test('reject_old_parent_retained_after_clear',lambda:check(bad),True)
  bad=copy.deepcopy(v);bad[3]['views_before'][0]['resource_id']=999;test('reject_unproven_old_parent',lambda:check(bad),True)
  bad=copy.deepcopy(v);bad[0]['quiescent']=0;test('reject_clear_at_nonquiescent_boundary',lambda:check(bad),True)
  bad=copy.deepcopy(v);bad[1]['GPU_work_submitted_by_callback']=True;test('reject_observer_gpu_work',lambda:check(bad),True)
  bad=copy.deepcopy(v);bad[2]['views_current'][0]['wrapper_id']+=1;test('reject_clear_replacing_view_wrapper',lambda:check(bad),True)
  bad=copy.deepcopy(v);bad[3]['new_cap']=16384;test('reject_fabricated_prompt_capacity',lambda:check(bad),True)
  normal=dict(implementation_version=4,prompt_view_clear_mode=1,prompt_lifetime_observer=True,deferred_lifetime_observer=False,diagnostic_prompt_mode=False)
  env={r.FLAG:'1',r.CLEAR_FLAG:'1'}
  test('two_normal_candidate_flags',lambda:r.normal_scope(normal,1,env))
  bad=copy.deepcopy(normal);bad['diagnostic_prompt_mode']=True;test('reject_diag_as_formal',lambda:r.normal_scope(bad,1,env),True)
  bad=copy.deepcopy(normal);bad['deferred_lifetime_observer']=True;test('reject_deferred_observer_as_formal',lambda:r.normal_scope(bad,1,env),True)
  test('reject_mismatched_clear_flag',lambda:r.normal_scope(normal,1,{r.FLAG:'1',r.CLEAR_FLAG:'0'}),True)
  bad=copy.deepcopy(normal);bad['prompt_lifetime_observer']=False;test('reject_missing_normal_prompt_observer',lambda:r.normal_scope(bad,1,env),True)
 # Causal diagnostics use only the candidate; independent gold binding remains mandatory.
 identity={'candidate':{'src/myds_metal.m':'candidate-metal','include/myds_internal.h':'candidate-internal','include/myds_metal.h':'candidate-metalh','src/myds_v41.c':'candidate-core','bin/bench_moe_workspace_fullmodel_promptclear':'candidate-bin','bin/myds.metallib':r.PRODUCTION_LIBRARY},'gold':{'src/myds_v41.c':'gold-core','bin/bench_moe_workspace_fullmodel_promptclear':'gold-bin','bin/myds.metallib':r.PRODUCTION_LIBRARY}}
 env={'MYDS_PLD':'1'};model={'path':'CPU-fixture-only','bytes':123};header='model-prefix'
 causal=dict(native_identity={'candidate':identity['candidate']},production_environment=env,model=model,model_header_sha256=header)
 component=dict(candidate_core_sha256={k:identity['candidate'][k]for k in ('src/myds_v41.c','src/myds_metal.m','include/myds_internal.h','include/myds_metal.h')},metallib_sha256=r.PRODUCTION_LIBRARY,production_environment=env,model=model,model_header_sha256=header)
 test('component_binds_exact_core4_library_model_env',lambda:r.component_identity_gate(component,identity,env,model,header))
 bad=copy.deepcopy(component);bad['candidate_core_sha256']['src/myds_metal.m']='changed';test('reject_component_other_v4_core',lambda:r.component_identity_gate(bad,identity,env,model,header),True)
 bad=copy.deepcopy(component);bad['model_header_sha256']='changed';test('reject_component_other_model',lambda:r.component_identity_gate(bad,identity,env,model,header),True)
 test('causal_binds_candidate_without_fabricated_gold',lambda:r.causal_identity_gate(causal,identity,env,model,header))
 bad=copy.deepcopy(causal);bad['native_identity']['candidate']['src/myds_v41.c']='changed';test('reject_causal_candidate_core_changed',lambda:r.causal_identity_gate(bad,identity,env,model,header),True)
 bad=copy.deepcopy(causal);bad['production_environment']['MYDS_PLD']='0';test('reject_causal_environment_changed',lambda:r.causal_identity_gate(bad,identity,env,model,header),True)
 gold=dict(native_binary_sha256='gold-bin',metallib_sha256=r.PRODUCTION_LIBRARY)
 test('independent_gold_native_record',lambda:r.native_record_gate(gold,identity,'gold'))
 bad=copy.deepcopy(gold);bad['native_binary_sha256']='changed';test('reject_gold_native_record_changed',lambda:r.native_record_gate(bad,identity,'gold'),True)
 index={'/CPU-only-long.json':dict(sha256='a'*64,bytes=12)};link=dict(path='/CPU-only-long.json',sha256='a'*64)
 test('long_reference_two_fields',lambda:r.long_reference_gate(link,index,'CPU'))
 test('long_reference_optional_exact_bytes',lambda:r.long_reference_gate(dict(link,bytes=12),index,'CPU'))
 for value in (True,-1,13):test('reject_long_reference_bad_bytes_'+str(value),lambda value=value:r.long_reference_gate(dict(link,bytes=value),index,'CPU'),True)
 test('reject_long_reference_unrecognized_field',lambda:r.long_reference_gate(dict(link,other=1),index,'CPU'),True)
 # Metrics controls are explicitly synthetic and cannot be accepted as raw evidence.
 # Exact executed-archive mapping controls use only synthetic source bytes.
 with tempfile.TemporaryDirectory(prefix='localgrow-archive-controls-')as temp:
  spec=importlib.util.spec_from_file_location('archiver',Path(__file__).with_name('archive_moe_workspace_stage.py'));ar=importlib.util.module_from_spec(spec);spec.loader.exec_module(ar)
  q=Path(temp);stage=q/'stage';(stage/'bin').mkdir(parents=True);(stage/'Makefile').write_bytes(b'CPU fixture only');(stage/'bin/myds').write_bytes(b'CPU fake binary');window=q/'window';window.mkdir();paths=[stage/'Makefile',stage/'bin/myds']
  (window/'stage-inputs.json').write_text(json.dumps({str(p):r.sha(p)for p in paths}));(window/'summary.json').write_text(json.dumps(dict(status='complete',restoration=dict(restored=True))));out=q/'executed';ar.create(window,out)
  test('all_executed_archive_bytes_and_path_map',lambda:r.executed_archives(window,out/'source.tar.gz',out/'artifacts.tar.gz',out/'archive-bindings.json'))
  original=(out/'archive-bindings.json').read_text();v=json.loads(original);v['files'][0]['original_path']+='changed';(out/'archive-bindings.json').write_text(json.dumps(v));test('reject_original_path_map_changed',lambda:r.executed_archives(window,out/'source.tar.gz',out/'artifacts.tar.gz',out/'archive-bindings.json'),True)
  (out/'archive-bindings.json').write_text(original);v=(out/'artifacts.tar.gz').read_bytes();(out/'artifacts.tar.gz').write_bytes(v[:-1]+bytes([v[-1]^1]));test('reject_archive_last_byte_changed',lambda:r.executed_archives(window,out/'source.tar.gz',out/'artifacts.tar.gz',out/'archive-bindings.json'),True)
 plan=r.load(Path(__file__).parent.parent/'docs/benchmarks/2026-10-24-moe-phys-cause/experiment-plan.json');test('frozen_v4_sequence_protocol',lambda:r.frozen_protocol(plan))
 bad=copy.deepcopy(plan);bad['fullmodel']['adoption']['setup_material_gain_threshold']=True;test('reject_setup_scope_reclassified',lambda:r.frozen_protocol(bad),True)
 def arm(setting):
  seq=[]
  for i in range(4):
   stages=[]
   for j in range(3):
    amount=(210 if j==0 else 300)if setting else 0
    peaks={k:(4000-amount)*1048576 for k in r.METRICS}
    stages.append(dict(stage=r.LAYOUT[j],sample=dict(prefill_s=10,decode_s=1,wall_s=12),actual_peak_with_lifetime_events=peaks))
   seq.append(dict(observer=dict(coverage=True),prompt_observer=dict(coverage=True),total_wall_s=36,stages=stages))
  return dict(phase='performance',setting=setting,sequences=seq)
 a,b,c=arm(0),arm(1),arm(0)
 test('setup_no_material_gain_primary256',lambda:r.need(r.performance_gate(a,b,c)['performance_gate_pass'],'gate'))
 bad=copy.deepcopy(b)
 for q in bad['sequences']:
  q['stages'][2]['actual_peak_with_lifetime_events']={k:3900*1048576 for k in r.METRICS}
 test('reject_primary_below256_and_no_speed',lambda:r.need(r.performance_gate(a,bad,c)['performance_gate_pass'],'gate'),True)
 bad=copy.deepcopy(c)
 for q in bad['sequences']:q['total_wall_s']=37
 test('reject_total_wall_drift',lambda:r.need(r.performance_gate(a,b,bad)['performance_gate_pass'],'gate'),True)
 bad=copy.deepcopy(b)
 for q in bad['sequences']:q['observer']['coverage']=False
 test('reject_missing_lifetime_observer',lambda:r.need(r.performance_gate(a,bad,c)['performance_gate_pass'],'gate'),True)
 bad=copy.deepcopy(b)
 for q in bad['sequences']:q['prompt_observer']['coverage']=False
 test('reject_missing_prompt_lifetime_coverage',lambda:r.need(r.performance_gate(a,bad,c)['performance_gate_pass'],'gate'),True)
 bad=copy.deepcopy(b)
 for q in bad['sequences']:q['stages'][1]['actual_peak_with_lifetime_events']['device_allocated_bytes']=4200*1048576
 test('reject_growth_simultaneous_live_regression',lambda:r.need(r.performance_gate(a,bad,c)['performance_gate_pass'],'gate'),True)
 return dict(schema_version=1,scope='bounded_cpu_promptclear_sequence_reader_controls',gpu_run=False,model_run=False,controls=controls,passed=all(v['passed']for v in controls))
if __name__=='__main__':
 report=run();print(json.dumps(report,indent=2))
