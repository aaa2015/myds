"""CPU-only adversarial contracts. Mock records are never GPU evidence."""
import copy,hashlib,importlib.util,json,os,struct,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('narrow_reader',ROOT/'tools/review_narrow_decode.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def desc(p):b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=m.digest(b))
class Fixture:
 def __init__(self,root,host=False):
  self.root=root;self.host=host;self.output=root/('host'if host else'gpu');self.output.mkdir();self.files=[];self.alias={}
  f={};k=n=32
  payloads=dict(weight=struct.pack('<H',0x3c00)*(k*n),codes=b'\x38'*(k*n),scales=struct.pack('<f',1),input=struct.pack('<f',1)*k)
  for role,b in payloads.items():p=root/(role+'.bin');p.write_bytes(b);f[role]=desc(p)
  c=dict(id='cpu-mock-case',family='engram',columns=k,rows=n,profile='CPU_mock_never_GPU',outputs=['f32','bf16'],**f)
  fixture=dict(schema_version=1,kind='narrow_decode_fixture_set',correctness_only=True,cases=[c]);p=root/'fixtures.json';p.write_text(json.dumps(fixture));self.fixture=p
  artifacts=dict(fixtures=desc(p))
  for name in('harness','reader','binary','build_identity','deployed_library','native_library','probe_library'):
   p=root/(name+'.bin');p.write_bytes(name.encode());artifacts[name]=desc(p)
  artifacts['host_fixture_manifest']=dict(path=str(self.output/'artifact-manifest.json'),bytes=1,sha256='0'*64)
  self.protocol=dict(artifacts=artifacts,expected_device='CPU mock forbidden as real target')
  self.frozen={k:Path(v['path'])for k,v in artifacts.items()};self.cases=m.fixture_cases(fixture,m.Paths());case=self.cases[0]
  self.manifest=dict(schema_version=1,probe='narrow_decode_correctness',correctness_only=True,timing_collected=False,fullmodel=False,production_registered=False,adoption_eligible=False,host_only=host,gpu_work_submitted=not host,guard_bytes=64,poison_byte=165,case_count=1,device=self.protocol['expected_device'],libraries=[])
  for key,artifact in(('fixture','fixtures'),('binary','binary'),('build_identity','build_identity'),('harness_source','harness')):self.manifest[key+'_path']=artifacts[artifact]['path'];self.manifest[key+'_sha256']=artifacts[artifact]['sha256']
  for i,name in enumerate(('deployed','native','probe')):self.manifest['libraries'].append(dict(index=i,**artifacts[name+'_library']))
  for field,name,artifact in(('fixtures_copy','fixtures.json','fixtures'),('build_identity_copy','build-identity.json','build_identity'),('harness_source_copy','harness-source.m','harness')):self.manifest[field]=self.raw(name,self.frozen[artifact].read_bytes())
  rec=dict(case=0,id=c['id'],family=c['family'],columns=k,rows=n,fixture=c,expected_f32_weights=case['f32'],inputs_before={},providers=[],expansions=[],reconstruction_case=0)
  for role in m.ROLES:rec['inputs_before'][role]=self.guarded('case000.'+role+'.before',payloads[role])
  ordinal=0
  for mode in c['outputs']:
   nr=2 if mode=='bf16'else 1;original=m.mv_bytes(k,n,2,nr);narrow=m.mv_bytes(k,n,1,nr);rec[mode+'_args']=self.raw('case000.'+mode+'.args112.raw',original);rec[mode+'_candidate_args']=self.raw('case000.'+mode+'.candidate-args112.raw',narrow)
   for provider in range(4):
    li=provider if provider<3 else 2;native='kernel_mul_mv_f16_f32_4_bf16_nr2'if mode=='bf16'else'kernel_mul_mv_f16_f32_4';candidate='kernel_myds_narrow_e4m3_bf16_nr2'if mode=='bf16'else'kernel_myds_narrow_e4m3_f32_nr1';row=dict(provider=provider,mode=mode,symbol=native if provider<3 else candidate,library_index=li,library_sha256=self.manifest['libraries'][li]['sha256'],args_bytes=112,args_kind='narrow'if provider==3 else'original',args_sha256=m.digest(narrow if provider==3 else original),nr0=nr,nsg=8,dynamic_threadgroup_bytes=128*nr,threads=[32,8,1],grid=[(n+nr-1)//nr,1,1],buffers=['args','codes','input','output','scales']if provider==3 else['args','weight','input','output'])
    if not host:
     ordinal+=1;row.update(static_threadgroup_bytes=0,thread_execution_width=32,max_threads=1024,completion_ordinal=ordinal,gpu_status=4,gpu_error=None,gpu_completed=True,inputs_after={},inputs_unchanged=True,matches_deployed=True,finite=True,bf16_output_bits_valid=True,all_output_words_written=True,guards_intact=True,numerical_passed=True)
     for role in m.ROLES:row['inputs_after'][role]=self.guarded(f'case000.{mode}.provider{provider}.{role}.after',payloads[role])
     row['output']=self.guarded(f'case000.{mode}.provider{provider}.output',struct.pack('<f',32)*n)
    rec['providers'].append(row)
  for kind in('f16','f32'):
   symbol='kernel_myds_narrow_expand_e4m3_'+kind;row=dict(symbol=symbol,args=self.raw('case000.'+symbol+'.args16.raw',struct.pack('<IIQ',k,n,k*n)),args_bytes=16,threads=[256,1,1],grid=[4,1,1],output_kind=kind.upper(),library_sha256=artifacts['probe_library']['sha256'])
   if not host:
    ordinal+=1;row.update(static_threadgroup_bytes=0,thread_execution_width=32,max_threads=1024,completion_ordinal=ordinal,gpu_status=4,gpu_error=None,gpu_completed=True,inputs_unchanged=True,guards_intact=True,matches_original_bits=True,numerical_passed=True)
    for role in('codes','scales'):row[role+'_after']=self.guarded('case000.'+symbol+'.'+role+'.after',payloads[role])
    row['output']=self.guarded('case000.'+symbol+'.output',payloads['weight']if kind=='f16'else struct.pack('<f',1)*(k*n))
   rec['expansions'].append(row)
  self.manifest.update(cases=[rec],gpu_completion_count=ordinal,all_numerical_passed=not host,status='host_only_complete_not_numerical'if host else'correctness_complete_not_adoption');self.save()
 def raw(self,name,b):
  p=self.output/name;h=m.digest(b);key=(len(b),h);r=dict(path=name,bytes=len(b),sha256=h)
  if key in self.alias:os.link(self.output/self.alias[key],p);r['storage_alias_of']=self.alias[key]
  else:p.write_bytes(b);self.alias[key]=name
  self.files.append(r);return r
 def guarded(self,name,b):return dict(body=self.raw(name+'.body.raw',b),prefix_guard=self.raw(name+'.prefix-guard.raw',m.GUARD),suffix_guard=self.raw(name+'.suffix-guard.raw',m.GUARD))
 def save(self):self.manifest['files']=self.files;(self.output/'artifact-manifest.json').write_text(json.dumps(self.manifest))
 def change_raw(self,d,newbytes):
  p=self.output/d['path'];p.unlink();p.write_bytes(newbytes);d.pop('storage_alias_of',None);d.update(bytes=len(newbytes),sha256=m.digest(newbytes));self.save()
 def review(self):
  args=type('Args',(),dict(path_map=[]))();r=m.Review(args);r.protocol=self.protocol;r.frozen=self.frozen;r.cases=self.cases;r.expected_raw_count=3+12+62*2+10*2;result=r.check_manifest(self.output,self.host);return r,result
class Tests(unittest.TestCase):
 def make(self,host=False):t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup);return Fixture(Path(t.name),host)
 def test_mock_complete_payload_positive(self):f=self.make();r,_=f.review();self.assertTrue(r.report['numerical_passed']);self.assertEqual(r.report['gpu_completion_count'],10);self.assertEqual(r.report['raw_file_count'],159)
 def test_host_never_numerical_or_gpu(self):f=self.make(True);r,d=f.review();self.assertFalse(d['gpu_work_submitted']);self.assertEqual(d['gpu_completion_count'],0)
 def test_one_ulp_consistent_numeric_failure_is_complete(self):
  f=self.make();row=f.manifest['cases'][0]['providers'][3];data=(f.output/row['output']['body']['path']).read_bytes();f.change_raw(row['output']['body'],struct.pack('<I',0x42000001)+data[4:]);row.update(matches_deployed=False,numerical_passed=False);f.manifest.update(all_numerical_passed=False,status='numerical_failed');f.save();r,_=f.review();self.assertFalse(r.report['numerical_passed'])
 def test_all_provider_untouched_poison_is_numeric_failure(self):
  f=self.make()
  for row in f.manifest['cases'][0]['providers']:
   d=row['output']['body'];f.change_raw(d,b'\xa5'*128);row.update(all_output_words_written=False,bf16_output_bits_valid=row['mode']!='bf16',numerical_passed=False)
  f.manifest.update(all_numerical_passed=False,status='numerical_failed');f.save();r,_=f.review();self.assertFalse(r.report['numerical_passed'])
 def test_false_all_written_claim_rejected(self):
  f=self.make();row=f.manifest['cases'][0]['providers'][3];f.change_raw(row['output']['body'],b'\xa5'*128);row.update(matches_deployed=False,numerical_passed=False);f.manifest.update(all_numerical_passed=False,status='numerical_failed');f.save();self.assertRaises(m.EvidenceError,f.review)
 def test_false_numeric_declaration_rejected(self):f=self.make();f.manifest['cases'][0]['providers'][3]['numerical_passed']=False;f.save();self.assertRaises(m.EvidenceError,f.review)
 def test_missing_raw_rejected(self):f=self.make();(f.output/f.files[-1]['path']).unlink();self.assertRaises(m.EvidenceError,f.review)
 def test_ignored_extra_raw_rejected(self):f=self.make();(f.output/'ignored.raw').write_bytes(b'x');self.assertRaises(m.EvidenceError,f.review)
 def test_guard_corruption_consistent_failure(self):
  f=self.make();row=f.manifest['cases'][0]['providers'][3];f.change_raw(row['output']['prefix_guard'],b'\x00'+m.GUARD[1:]);row.update(guards_intact=False,numerical_passed=False);f.manifest.update(all_numerical_passed=False,status='numerical_failed');f.save();r,_=f.review();self.assertFalse(r.report['numerical_passed'])
 def test_input_corruption_consistent_failure(self):
  f=self.make();row=f.manifest['cases'][0]['providers'][3];d=row['inputs_after']['codes']['body'];b=(f.output/d['path']).read_bytes();f.change_raw(d,b'\x00'+b[1:]);row.update(inputs_unchanged=False,numerical_passed=False);f.manifest.update(all_numerical_passed=False,status='numerical_failed');f.save();r,_=f.review();self.assertFalse(r.report['numerical_passed'])
 def test_candidate_uses_original_stride_rejected(self):f=self.make();row=f.manifest['cases'][0]['providers'][3];row['args_sha256']=f.manifest['cases'][0]['providers'][0]['args_sha256'];f.save();self.assertRaises(m.EvidenceError,f.review)
 def test_bool_provider_rejected(self):f=self.make();f.manifest['cases'][0]['providers'][1]['provider']=True;f.save();self.assertRaises(m.EvidenceError,f.review)
 def test_bool_completion_rejected(self):f=self.make();f.manifest['cases'][0]['providers'][0]['completion_ordinal']=True;f.save();self.assertRaises(m.EvidenceError,f.review)
 def test_reconstruction_bit_difference_consistent_failure(self):
  f=self.make();row=f.manifest['cases'][0]['expansions'][1];d=row['output']['body'];b=(f.output/d['path']).read_bytes();f.change_raw(d,struct.pack('<I',0x3f800001)+b[4:]);row.update(matches_original_bits=False,numerical_passed=False);f.manifest.update(all_numerical_passed=False,status='numerical_failed');f.save();r,_=f.review();self.assertFalse(r.report['numerical_passed'])
 def test_noncomplete_gpu_consistent_numeric_failure(self):
  f=self.make();row=f.manifest['cases'][0]['providers'][3];row.update(gpu_status=5,gpu_error='CPU_mock_error',gpu_completed=False,numerical_passed=False);f.manifest.update(all_numerical_passed=False,status='numerical_failed');f.save();r,_=f.review();self.assertFalse(r.report['numerical_passed'])
 def test_half_integer_widen_all_bits(self):
  for h in range(65536):
   if h&0x7c00==0x7c00:continue
   native=struct.pack('<f',struct.unpack('<e',struct.pack('<H',h))[0]);self.assertEqual(native,m.HALF_TABLE[h])
 def test_half_signedzero_subnormals(self):
  self.assertEqual(m.bits_half(0x8000),0x80000000);self.assertEqual(m.bits_half(1),0x33800000);self.assertEqual(m.bits_half(0x83ff),0xb87fc000)
 def test_actual_stage_rejects_toy_shape(self):f=self.make();self.assertRaises(m.EvidenceError,m.fixture_cases,m.load(f.fixture),m.Paths(),True)
 def test_paths_alias_boundary(self):p=m.Paths();self.assertEqual(str(p.path('/tmp/a')),'/private/tmp/a');self.assertEqual(str(p.path('/tmpX/a')),'/tmpX/a')
 def test_float_nonfinite_and_bf16_bits(self):
  t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup);p=Path(t.name)/'data';p.write_bytes(struct.pack('<I',0x3f800001));self.assertTrue(m.finite_file(p));self.assertFalse(m.finite_file(p,True));p.write_bytes(struct.pack('<I',0x7f800000));self.assertFalse(m.finite_file(p))
if __name__=='__main__':unittest.main()
