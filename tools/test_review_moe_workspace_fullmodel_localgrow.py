#!/usr/bin/env python3
"""Bounded CPU metadata/raw controls; never accepted as model/GPU evidence."""
import copy,importlib.util,json,struct,sys,tempfile
from pathlib import Path
sys.dont_write_bytecode=True
s=importlib.util.spec_from_file_location('reader',Path(__file__).with_name('review_moe_workspace_fullmodel_localgrow.py'));r=importlib.util.module_from_spec(s);s.loader.exec_module(r)
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
   v=dict(schema_version=1,scope='process_local_actual_binding_identity',complete=True,present=True,implementation_version=3,capacity=2048,moe_batch=8192,compact=1,cw_id=1,moe_generation=1,localgrow_attempts=0,localgrow_successes=0,localgrow_failures=0,stable_owner_fingerprint=int(h.bytes(struct.pack('<3Q',10,11,16)),16),stable_view_fingerprint=int(h.bytes(struct.pack('<2Q',12,13)),16),stable_owner_bytes=16,stable_owners=1,stable_views=2,down_id=14,down_bytes=32,arena_id=20,acc_ids=list(range(30,37)),acc_bytes=[32]*7,binding_count=len(bindings),bindings=bindings)
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
 # Metrics controls are explicitly synthetic and cannot be accepted as raw evidence.
 # Exact executed-archive mapping controls use only synthetic source bytes.
 with tempfile.TemporaryDirectory(prefix='localgrow-archive-controls-')as temp:
  spec=importlib.util.spec_from_file_location('archiver',Path(__file__).with_name('archive_moe_workspace_stage.py'));ar=importlib.util.module_from_spec(spec);spec.loader.exec_module(ar)
  q=Path(temp);stage=q/'stage';(stage/'bin').mkdir(parents=True);(stage/'Makefile').write_bytes(b'CPU fixture only');(stage/'bin/myds').write_bytes(b'CPU fake binary');window=q/'window';window.mkdir();paths=[stage/'Makefile',stage/'bin/myds']
  (window/'stage-inputs.json').write_text(json.dumps({str(p):r.sha(p)for p in paths}));(window/'summary.json').write_text(json.dumps(dict(status='complete',restoration=dict(restored=True))));out=q/'executed';ar.create(window,out)
  test('all_executed_archive_bytes_and_path_map',lambda:r.executed_archives(window,out/'source.tar.gz',out/'artifacts.tar.gz',out/'archive-bindings.json'))
  original=(out/'archive-bindings.json').read_text();v=json.loads(original);v['files'][0]['original_path']+='changed';(out/'archive-bindings.json').write_text(json.dumps(v));test('reject_original_path_map_changed',lambda:r.executed_archives(window,out/'source.tar.gz',out/'artifacts.tar.gz',out/'archive-bindings.json'),True)
  (out/'archive-bindings.json').write_text(original);v=(out/'artifacts.tar.gz').read_bytes();(out/'artifacts.tar.gz').write_bytes(v[:-1]+bytes([v[-1]^1]));test('reject_archive_last_byte_changed',lambda:r.executed_archives(window,out/'source.tar.gz',out/'artifacts.tar.gz',out/'archive-bindings.json'),True)
 plan=r.load(Path(__file__).parent.parent/'docs/benchmarks/2026-10-24-moe-localgrow/experiment-plan.json');test('frozen_v2_sequence_protocol',lambda:r.frozen_protocol(plan))
 bad=copy.deepcopy(plan);bad['fullmodel']['adoption']['setup_material_gain_threshold']=True;test('reject_setup_scope_reclassified',lambda:r.frozen_protocol(bad),True)
 def arm(setting):
  seq=[]
  for i in range(4):
   stages=[]
   for j in range(3):
    amount=(210 if j==0 else 300)if setting else 0
    peaks={k:(4000-amount)*1048576 for k in r.METRICS}
    stages.append(dict(stage=r.LAYOUT[j],sample=dict(prefill_s=10,decode_s=1,wall_s=12),actual_peak_with_lifetime_events=peaks))
   seq.append(dict(observer=dict(coverage=True),total_wall_s=36,stages=stages))
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
 for q in bad['sequences']:q['stages'][1]['actual_peak_with_lifetime_events']['device_allocated_bytes']=4200*1048576
 test('reject_growth_simultaneous_live_regression',lambda:r.need(r.performance_gate(a,bad,c)['performance_gate_pass'],'gate'),True)
 return dict(schema_version=1,scope='bounded_cpu_localgrow_sequence_reader_controls',gpu_run=False,model_run=False,controls=controls,passed=all(v['passed']for v in controls))
if __name__=='__main__':
 report=run();print(json.dumps(report,indent=2))
