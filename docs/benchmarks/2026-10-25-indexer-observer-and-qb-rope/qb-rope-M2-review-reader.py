from pathlib import Path
import array,hashlib,json,math,struct
BASE=Path('/tmp/myds-indexer-r3-local-20261025/qb-rope-target-screen');OUT=BASE/'outputs'
M4=Path('/tmp/myds-qb-rope-gpu-screen-20261025/outputs');BUILD=Path('/tmp/myds-qb-rope-probe-build-v1-20261025')
HOST=Path('/tmp/myds-qb-rope-host-only-v1-20261025');LOCAL_GOLD=Path('/tmp/myds-indexer-r2-real-local-20261025/deployed/myds.metallib')
PROTOCOL=BASE.parent/'qb-rope-target-protocol.json'
def sha(x):return hashlib.sha256(x).hexdigest()
def unique(pairs):
 d={}
 for k,v in pairs:
  assert k not in d,('duplicate_json_key',k);d[k]=v
 return d
def js(p):return json.loads(p.read_text(),object_pairs_hook=unique,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
def words(b):
 a=array.array('I');a.frombytes(b);assert a.itemsize==4;return a
m=js(OUT/'artifact-manifest.json');m4=js(M4/'artifact-manifest.json');b=js(BUILD/'source-identity.json');p=js(PROTOCOL);e=js(BASE/'execution-plan.json');w=js(BASE/'window.json')
r=[json.loads(x,object_pairs_hook=unique,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x))) for x in (BASE/'stdout.jsonl').read_text().splitlines()]
assert m['device_name']=='Apple M2 Ultra' and m['status']=='failed' and m['case_count']==114
assert m['gpu_command_buffers_submitted']==342 and m['all_complete_outputs_and_inputs_pass'] is False and m['timing_requested'] is False
assert w['exit_code']==1 and w['abort_reason'] is None and w['recognized_before']==w['recognized_after']==[]
assert w['canonical_before']==w['canonical_after'] and w['canonical_unchanged'] is True
assert p['fullmodel_eligible'] is False and p['production_adoption'] is False and p['M4_failure_permanently_preserved'] is True
assert p['timing_requested'] is False and '--timing' not in e['argv'] and e['fullmodel'] is False
assert sha(PROTOCOL.read_bytes())==e['protocol_sha256']==w['protocol_sha256']
assert m['binary_sha256']==m4['binary_sha256']==p['frozen_binary_sha256']==e['binary_sha256']==b['binary_sha256']==sha((BUILD/'bench_qb_rope_probe').read_bytes())
assert m['probe_metallib_sha256']==m4['probe_metallib_sha256']==p['probe_sha256']==e['probe_sha256']==b['metallib_sha256']==sha((BUILD/'probe.metallib').read_bytes())
assert m['golden_metallib_sha256']==m4['golden_metallib_sha256']==e['native_sha256']==sha(LOCAL_GOLD.read_bytes())
assert m['build_manifest_sha256']==m4['build_manifest_sha256']==e['build_manifest_sha256']==sha((BUILD/'source-identity.json').read_bytes())
assert (BASE.parent/'qb-rope-frozen/source-identity.json').read_bytes()==(BUILD/'source-identity.json').read_bytes()==(OUT/'build-identity.json').read_bytes()
assert (BUILD/'all.metal').read_bytes()[:b['native_source_bytes']]==(BUILD/'native.metal').read_bytes()
assert sha((BUILD/'native.metal').read_bytes())==b['native_source_sha256']
for rel in ['tools/build_qb_rope_probe.py','tools/bench_qb_rope_probe.m']:
 assert sha(Path('/Users/fangqingyuan/Projects/myds',rel).read_bytes())==b['sources'][rel]['sha256']
files={};guards=0
for f in m['files']:
 name=f['path'];assert name not in files and not Path(name).is_absolute() and '..' not in Path(name).parts
 data=(OUT/name).read_bytes();assert len(data)==f['bytes'] and sha(data)==f['sha256'];files[name]=data
 if name.endswith('guard.raw'):assert data==bytes([0xa5])*64;guards+=1
assert len(files)==1715 and guards==914
assert {x.name for x in OUT.iterdir()}==set(files)|{'artifact-manifest.json'}
before_records=[];before_bytes=0
for f in js(HOST/'artifact-manifest.json')['files']:
 name=f['path'];a=files[name];x=(M4/name).read_bytes();h=(HOST/name).read_bytes()
 assert a==x==h and len(a)==f['bytes'] and sha(a)==f['sha256']
 before_records.append(dict(path=name,bytes=len(a),sha256=sha(a),bytes_equal_M4=True,bytes_equal_host_only=True));before_bytes+=len(a)
assert len(before_records)==344
case=[x for x in r if x['kind']=='case'];par=[x for x in r if x['kind']=='parity'];comp=[x for x in r if x['kind']=='gpu_completion'];inp=[x for x in r if x['kind']=='inputs']
assert len(case)==114 and len(par)==len(comp)==342 and par==m['parity_records'] and inp==m['input_records'] and len(inp)==115
assert not m['timing_records'] and not any(x['kind'] in ('timing','component_summary') for x in r)
assert [x['ordinal'] for x in comp]==list(range(1,343)) and [(x['case'],x['provider']) for x in comp]==[(i,j) for i in range(114) for j in range(3)]
assert all(x['completed'] and x['status']==4 and x['stage']=='phase1' for x in comp)
eligible=[i for i in range(40) if i not in {2,8,14,20,24,28,32,36}]
expected=[(l,pos) for l in eligible for pos in [0,1,327679]]+[(l,pos) for l in [0,3,39] for pos in [127,128,16383,16384,65535,65536]]
assert [(x['layer'],x['position']) for x in case]==expected and m['eligible_layers']==eligible
assert all(x['case']==i and x['compressed']==(x['layer']>=2) for i,x in enumerate(case))
input_bytes=0
for x in inp:
 post=files[x['raw_prefix']+'.raw'];before=files['weights.before.raw'] if x['role']=='weights' else files[f"case{x['case']:03}.input.before.raw"]
 assert post==before and sha(post)==x['before_sha256']==x['after_sha256'] and x['guards'] and x['bytes_equal_before']
 assert len(post)==x['bytes'] and x['stage']=='phase1_post';input_bytes+=len(post)
all_bad=[];cross=[dict(provider=j,complete_output_pairs=114,mismatching_cases=[],mismatching_f32=0) for j in range(3)];finite=0
for i in range(114):
 native=files[f'phase1.case{i:03}.provider0.output.raw'];assert len(native)==131072 and native==files[f'phase1.case{i:03}.provider1.output.raw']
 gold=words(native)
 for provider in range(3):
  raw=files[f'phase1.case{i:03}.provider{provider}.output.raw'];v=words(raw);assert len(v)==32768
  diffs=[];tail=0
  for j,(x,y) in enumerate(zip(gold,v)):
   assert not(y&0xffff) and y&0x7f800000!=0x7f800000;finite+=1
   if x==y:continue
   assert provider==2;tail+=j%512>=448
   a=struct.unpack('<f',struct.pack('<I',x))[0];z=struct.unpack('<f',struct.pack('<I',y))[0]
   order=lambda u:0x80000000-(u&0x7fffffff) if u&0x80000000 else 0x80000000+u
   ulp=abs(order(x)-order(y));ab=abs(a-z);rel=ab/abs(a) if a else math.inf
   q=dict(case=i,layer=case[i]['layer'],position=case[i]['position'],compressed=case[i]['compressed'],row=j,head=j//512,head_dimension=j%512,rope_pair=(j%512-448)//2,component='im' if j%2 else 're',gold_u32=f'0x{x:08x}',candidate_u32=f'0x{y:08x}',gold_bf16=f'0x{x>>16:04x}',candidate_bf16=f'0x{y>>16:04x}',gold_f32=a,candidate_f32=z,f32_ulp_distance=ulp,bf16_ulp_distance=ulp//65536,absolute_error=ab,relative_error_to_gold=rel)
   diffs.append(q);all_bad.append(q)
  row=par[i*3+provider];assert row['mismatches']==len(diffs) and row['tail_mismatches']==tail and row['unrotated_mismatches']==len(diffs)-tail
  assert row['first_mismatch']==(diffs[0]['row'] if diffs else None) and row['sha256']==sha(raw) and row['finite_bf16'] and row['guards']
  m4raw=(M4/f'phase1.case{i:03}.provider{provider}.output.raw').read_bytes()
  if raw!=m4raw:
   count=sum(x!=y for x,y in zip(v,words(m4raw)));cross[provider]['mismatching_cases'].append(dict(case=i,mismatching_f32=count));cross[provider]['mismatching_f32']+=count
assert len(all_bad)==6 and len({x['case'] for x in all_bad})==6
assert all(x['head_dimension']>=448 and x['component']=='im' and x['f32_ulp_distance']==65536 and x['bf16_ulp_distance']==1 for x in all_bad)
m4cases={x['case'] for x in m4['parity_records'] if x['mismatches']};m2cases={x['case'] for x in all_bad}
loaded=[x for x in r if x['kind']=='loaded_library'];pipes=[x for x in r if x['kind']=='pipeline']
assert len(loaded)==2 and len(pipes)==5 and all(x['fallback'] is False and x['thread_execution_width']==32 for x in pipes)
assert loaded[0]['sha256']==m['golden_metallib_sha256'] and loaded[1]['sha256']==m['probe_metallib_sha256']
report=dict(schema_version=1,kind='qb_forward_rope_M2_target_independent_raw_review',review_completed=True,device=m['device_name'],numerical_gate_passed=False,performance_timing_started=False,production_adoption=False,full_model_evidence=False,source_and_tools_frozen=True,raw_root=str(BASE),evidence_sha256=dict(manifest=sha((OUT/'artifact-manifest.json').read_bytes()),stdout=sha((BASE/'stdout.jsonl').read_bytes()),protocol=sha(PROTOCOL.read_bytes()),execution_plan=sha((BASE/'execution-plan.json').read_bytes()),window=sha((BASE/'window.json').read_bytes())),identity=dict(binary_sha256=m['binary_sha256'],probe_sha256=m['probe_metallib_sha256'],actual_gold_sha256=m['golden_metallib_sha256'],build_manifest_sha256=m['build_manifest_sha256'],native_source_sha256=b['native_source_sha256'],all_identical_to_frozen_M4_v1=True,remote_paths_verified_via_frozen_wrapper_and_local_mirror=True,live_remote_state_not_accessed_by_this_review=True),inputs=dict(before_payloads_full_bytes_and_sha_equal_M4=len(before_records),before_payload_bytes_compared=before_bytes,before_payloads=before_records,post_input_records_full_bytes_equal_before=len(inp),post_input_bytes_compared=input_bytes,guard_raw_files_all0xa5=guards,guard_bytes_each=64),raw=dict(files_size_sha_verified=len(files),all_directory_entries_accounted=True,complete_output_raw_files=342,output_f32_verified=finite,all_output_finite_bf16=True),execution=dict(all_completed_ordered_cbs=342,case_count=114,eligible_layer_count=32,all_five_explicit_pipelines_no_fallback=True,correctness_only_argv=True,timing_records=0,expected_fail_closed_exit_code=1,wrapper_reports_canonical_unchanged=w['canonical_unchanged']),parity=dict(actual_gold_vs_same_probe_native_complete_equal_pairs=114,actual_gold_vs_candidate_complete_equal_pairs=108,candidate_mismatching_cases=6,candidate_mismatching_f32=6,all_unrotated_equal=True,max_f32_ulp_distance=max(x['f32_ulp_distance'] for x in all_bad),max_bf16_ulp_distance=max(x['bf16_ulp_distance'] for x in all_bad),max_absolute_error=max(x['absolute_error'] for x in all_bad),max_relative_error_to_gold=max(x['relative_error_to_gold'] for x in all_bad),differences=all_bad),cross_device=dict(shared_mismatch_cases=sorted(m2cases&m4cases),M2_only_mismatch_cases=sorted(m2cases-m4cases),M4_only_mismatch_cases=sorted(m4cases-m2cases),M2_mismatch_set_is_superset_M4=False,complete_output_raw_comparison_by_provider=cross),causal_limit='Same frozen AIR/library/input bytes does not establish identical device-machine lowering. The shared fast im affine AIR ordering difference is documented in M4 review; no deterministic repair is proved and no source/FMA retest occurred.',decision='Reject frozen v1 on target M2 exact gate as well as M4. No timing, runtime integration, performance regression conclusion, or all-fusion impossibility conclusion.')
path=Path('/tmp/myds-qb-rope-M2-independent-review-20261025.json');path.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(review=str(path),before_payloads_equal_M4=len(before_records),raw_files_verified=len(files),output_f32_verified=finite,differences=all_bad,cross_device=cross),indent=2))
