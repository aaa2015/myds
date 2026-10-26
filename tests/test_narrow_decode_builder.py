import ctypes
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('narrow_builder',ROOT/'tools/build_narrow_decode_probe.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)

def args(family,symbol):
    f=b.FAMILIES[family];v=b.GEMV[symbol];c,r=f['cols'],f['rows'];e=2 if v['dtype']=='BF16'else 1
    return dict(ne00=c,ne01=r,ne02=1,nb00=e,nb01=c*e,nb02=c*r*e,nb03=c*r*e,ne10=c,ne11=1,ne12=1,nb10=4,nb11=c*4,nb12=c*4,nb13=c*4,ne0=r,ne1=1,nr0=v['nr0'],r2=1,r3=1)

class NarrowBuilder(unittest.TestCase):
    def test_full_code_scale_lattice(self):
        d=b.cpu_codec_contract();self.assertEqual(d['code_scale_pairs_checked'],256*47);self.assertEqual(d['qualified_F16_pairs_checked'],9318)
    def test_full_half_scale_envelope_including_minus32(self):
        # A finite F16 block selects e0 in[-32,8], then tries e0+1/e0-1.
        # Cover a wider envelope so both min-subnormal half block scales and
        # all-zero block scale0 are explicit, independently of builder summary.
        checked=0;half=0
        for power in range(-40,17):
            sb=struct.unpack('<I',struct.pack('<f',2.**power))[0]
            for c in range(256):
                got=b.e4m3_f32_bits(c,sb);mag=c&127;e=mag>>3;m=mag&7
                if mag==127:self.assertEqual(got&0x7fffffff,0x7fc00000);checked+=1;continue
                value=((1+m/8)*2.**(e-7)if e else m*2.**-9)*2.**power
                if c&128:value=-value
                self.assertEqual(got,struct.unpack('<I',struct.pack('<f',value))[0]);checked+=1
                try:h=struct.pack('<e',value)
                except OverflowError:continue
                if struct.pack('<f',struct.unpack('<e',h)[0])!=struct.pack('<f',value):continue
                self.assertEqual(b.exact_f16_bits(got),struct.unpack('<H',h)[0]);half+=1
        self.assertEqual(checked,14592);self.assertEqual(half,9346)
    def test_signed_zeros_and_min_half_subnormal(self):
        scale=struct.unpack('<I',struct.pack('<f',2.**-15))[0]
        self.assertEqual(b.e4m3_f32_bits(0,scale),0);self.assertEqual(b.e4m3_f32_bits(128,scale),0x80000000)
        self.assertEqual(b.exact_f16_bits(b.e4m3_f32_bits(1,scale)),1);self.assertEqual(b.exact_f16_bits(b.e4m3_f32_bits(129,scale)),0x8001)
    def test_invalid_scales_rejected(self):
        for v in (0,0x80000000,0xbf800000,0x7f800000,0x7fc00000,0x3f800001):
            with self.subTest(v=v),self.assertRaises(ValueError):b.e4m3_f32_bits(1,v)
    def test_exact_half_diagnostic_does_not_truncate(self):
        for x in (0x3f800001,0x33800001,0x33000000,0x47800000,0x7f800000):self.assertEqual(b.exact_f16_bits(x)&0x7fff,0x7e00)
    def test_raw_vs_fused_actual_geometry(self):
        for f,s,nr in [('router','kernel_myds_narrow_router_bf16_nr1',1),('engram','kernel_myds_narrow_e4m3_f32_nr1',1),('index_qb','kernel_myds_narrow_e4m3_bf16_nr2',2)]:
            d=b.validate_problem(f,s,args(f,s));self.assertEqual(d['dynamic_threadgroup_bytes'],128*nr);self.assertEqual(d['grid'][0],b.FAMILIES[f]['rows']//nr)
    def test_rejects_wrong_actual_nr_and_noncontiguous_or_batch(self):
        f='engram';s='kernel_myds_narrow_e4m3_f32_nr1'
        for k,v in [('nr0',2),('ne11',2),('nb01',6145),('nb00',2),('r2',0),('ne00',6152)]:
            a=args(f,s);a[k]=v
            with self.subTest(k=k),self.assertRaises(ValueError):b.validate_problem(f,s,a)
    def test_rejects_bool_fields_and_short_buffers(self):
        f='router';s='kernel_myds_narrow_router_bf16_nr1';a=args(f,s);a['ne11']=True
        with self.assertRaises(ValueError):b.validate_problem(f,s,a)
        a=args(f,s);spans=b.validate_problem(f,s,a)['required_buffer_bytes'];spans['codes']-=1
        with self.assertRaises(ValueError):b.validate_problem(f,s,a,spans)
    def test_native_source_and_abi_contract(self):
        native,payloads,records=b.assemble_native(ROOT);c=(ROOT/'tools/narrow_decode.metal.inc').read_text();d=b.source_contract(payloads,c)
        self.assertEqual(d['dense_abi']['bytes'],112);self.assertEqual(d['dense_abi']['fields'][-3]['offset'],104)
        self.assertEqual(len(records),29);self.assertEqual(b.sha(native),'bc6603c97ec24af7597f6b10c7ed2a7bfe1344379733a0a4df780250a13f3224')
    def test_source_extra_kernel_or_changed_order_rejected(self):
        _,p,_=b.assemble_native(ROOT);c=(ROOT/'tools/narrow_decode.metal.inc').read_text()
        for other in (c+'\nkernel void unexpected() {}\n',c.replace('sumf[row] += sumq;','sumf[row] = sumq;'),c.replace('constexpr short NSG = 8;','constexpr short NSG = 4;')):
            with self.assertRaises(ValueError):b.source_contract(p,other)
    def test_emit_freezes_entire_full_tu_and_no_gpu(self):
        with tempfile.TemporaryDirectory()as tmp:
            out=Path(tmp)/'emit';r=subprocess.run([sys.executable,'-B',str(ROOT/'tools/build_narrow_decode_probe.py'),'--source-root',str(ROOT),'--out-dir',str(out),'--emit-only'],capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr);d=json.loads((out/'build-manifest.json').read_text());proof=json.loads((out/'source-transform.json').read_text())
            self.assertFalse(d['gpu_work_submitted']);self.assertFalse(d['compile_only_pass']);self.assertTrue(d['compilation_skipped']);self.assertEqual(len(d['commands']),4);self.assertEqual(d['source_file_count'],32)
            self.assertEqual((out/'all.metal').read_bytes()[:proof['native_bytes']],(out/'native.metal').read_bytes());self.assertEqual(set(d['metallibs']),set())
    def test_dangling_output_symlink_refused_before_writes(self):
        with tempfile.TemporaryDirectory()as tmp:
            target=Path(tmp)/'unowned';link=Path(tmp)/'out';link.symlink_to(target)
            r=subprocess.run([sys.executable,'-B',str(ROOT/'tools/build_narrow_decode_probe.py'),'--source-root',str(ROOT),'--out-dir',str(link),'--emit-only'],capture_output=True,text=True)
            self.assertNotEqual(r.returncode,0);self.assertFalse(target.exists());self.assertTrue(link.is_symlink())

if __name__=='__main__':unittest.main()
