import ctypes, importlib.util, json
from pathlib import Path
import struct, tempfile, unittest
ROOT=Path(__file__).resolve().parents[1]
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
f=load('narrow_fixture_tool',ROOT/'tools/make_narrow_decode_fixtures.py')
a=load('narrow_fixture_auditor',ROOT/'tools/audit_v41_narrow_weights.py')
class Fixtures(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.tmp=tempfile.TemporaryDirectory(prefix='narrow-fixture-CPU-');cls.root=Path(cls.tmp.name);w=cls.root/'worker';w.mkdir()
  p=ROOT/'docs/benchmarks/2026-10-26-upstream-plan-review/upstream-commits'
  if not p.exists():p=ROOT.parent/'mydsdoc/benchmarks/2026-10-26-upstream-plan-review/upstream-commits'
  cls.lib,cls.proof=a.build_worker(w,'/usr/bin/clang',p)
 @classmethod
 def tearDownClass(cls):cls.tmp.cleanup()
 def run_case(self,name,family,raw=None,expected=None):
  out=self.root/name;out.mkdir();(out/'payloads').mkdir()
  return f.encode_matrix(out,name,family,128,64,lambda row:raw if raw is not None else f.synthetic_stripe(128,row,family),self.lib,a,dict(tensor_name='fixture'),expected)
 def test_all_three_families_complete_exact_streams(self):
  for family in f.FAMILIES:
   c=self.run_case('valid-'+family+'-0',family)
   self.assertEqual(c['provenance']['reconstructed_source_sha256'],c['weight']['sha256'])
   self.assertEqual(c['codes']['bytes'],128*64*(2 if family=='router'else 1))
   self.assertEqual(c['scales']['bytes'],0 if family=='router'else 2*4*4)
   for k in ('roundtrip_mismatches','shader_f32_mismatches','failed_blocks','infinity','nan'):self.assertEqual(c['provenance']['counts'][k],0)
 def test_mixed_subnormals_and_signed_zero_inputs(self):
  raw=f.input_bytes(5120,2,42);bits=struct.unpack('<5120I',raw)
  self.assertTrue(any(x==0x80000000 for x in bits));self.assertTrue(any(x&0x7fffffff and x&0x7f800000==0 for x in bits));self.assertTrue(all(x&0x7f800000!=0x7f800000 for x in bits))
 def test_BF16_representable_inputs(self):self.assertTrue(all(x&65535==0 for x, in struct.iter_unpack('<I',f.input_bytes(5120,0,77))))
 def test_source_with_nonzero_low16_rejected(self):
  with self.assertRaisesRegex(ValueError,'finite exact'):self.run_case('bad-router','router',struct.pack('<I',0x3f800001)*(32*128))
 def test_nonfinite_projection_rejected(self):
  with self.assertRaisesRegex(ValueError,'every block'):self.run_case('bad-proj','engram',struct.pack('<H',0x7c00)*(32*128))
 def test_prior_qualification_SHA_mismatch_rejected(self):
  with self.assertRaisesRegex(ValueError,'prior audited stream'):self.run_case('bad-sha','router',expected=dict(name='fixture',qualification_passed=True,source_sha256='0'*64,codes_stream_sha256='0'*64))
 def test_partial_shapes_rejected(self):
  for columns,rows in ((127,64),(128,33),(0,64)):
   with self.assertRaisesRegex(ValueError,'qualification shape'):f.encode_matrix(self.root,'bad-shape','router',columns,rows,lambda r:b'',self.lib,a)
 def test_outputs_never_overwritten(self):
  self.run_case('once','router')
  with self.assertRaises(FileExistsError):self.run_case('once','router')
if __name__=='__main__':unittest.main()
