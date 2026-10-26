"""CPU fixtures for the independent register-only correctness reader.

Fixture score values are deliberately not a numerical model oracle. These tests
challenge complete-byte evidence/gate semantics without submitting Metal work.
"""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import tempfile
import contextlib
import io
from unittest import mock
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('register_reader', ROOT/'tools/review_indexer_register_only.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


class Fixture:
    def __init__(self, directory, ci=0):
        self.root, self.files, self.serial = Path(directory), [], 0
        self.ci = ci
        self.expected = r.CASES[ci]
        self.a = r.case_description(self.expected)
        self.case = self.make_case()

    def raw(self, b):
        name = f'payload{self.serial:04d}.raw'; self.serial += 1
        (self.root/name).write_bytes(b)
        d = dict(path=name, bytes=len(b), sha256=hashlib.sha256(b).hexdigest())
        self.files.append(d)
        return d

    def guarded(self, b):
        return dict(body=self.raw(b), prefix_guard=self.raw(b'\xa5'*64), suffix_guard=self.raw(b'\xa5'*64))

    def mutate(self, d, offset, value):
        b = bytearray((self.root/d['path']).read_bytes())
        b[offset:offset+len(value)] = value
        (self.root/d['path']).write_bytes(b)
        # This explicitly rehashes tampered fixtures: the reader must detect the
        # semantic error even with a self-consistent artifact inventory.
        d['sha256'] = hashlib.sha256(b).hexdigest()

    def payloads(self):
        return r.Payloads(self.root, self.files)

    def evaluate(self):
        return r.review_case(self.case, self.payloads(), self.expected, self.ci)

    def make_case(self):
        a, ci = self.a, self.ci
        nt, nk, pos = a['n_tokens'], a['n_comp'], a['pos0']
        before = dict(strided=[], dense=[])
        strided, dense = [], []
        for role, rows, cols, elem, stride in ((0,nt,32,512,a['q_token_stride']),
                (1,nt,1,128,a['weights_token_stride']), (2,nk,1,256,a['index_row_stride'])):
            data = bytearray(b'\xa5'*(rows*stride)); packed = []
            for row in range(rows):
                for col in range(cols):
                    index = row*cols+col
                    piece = bytearray((struct.pack('<f',1.0) if role<2 else struct.pack('<H',0x3f80))*(elem//(4 if role<2 else 2)))
                    # Literal finite profile data independent of the producer.
                    if ci%3 and role<2:
                        piece[:16] = struct.pack('<4I',0,0x80000000,0x3f800001,1 if ci%3==2 else 0x3f810001)
                    if ci%3==2 and role==2:
                        piece[:8] = struct.pack('<4H',0,0x8000,1,0x8001)
                    off=row*stride+(col*a['q_head_stride'] if role==0 else 0)
                    data[off:off+elem]=piece;packed.append(piece)
            strided.append(bytes(data));dense.append(b''.join(packed))
            before['strided'].append(self.guarded(bytes(data)))
            before['dense'].append(self.guarded(b''.join(packed)))
        args=struct.pack('<6I5Qf4x',*(a[k] for k in r.FIELDS))
        exact=struct.pack('<4If4I',nk,32,128,2,1/64,nt,pos,2,0)
        c=dict(case=ci,padding_profile=self.expected[3],data_profile=r.DATA_PROFILES[ci%3],cache_type='BF16',args=a,
               args72=self.raw(args),exact_args36=self.raw(exact),inputs_before=before,providers=[])
        logical=[]
        for p in range(6):
            stride=a['score_token_stride'] if p<4 else nk*4
            data=bytearray(b'\xa5'*(nt*stride))
            for t in range(nt):
                visible=min((pos+t+1)//2,nk)
                # Exact has intentional (descriptive) visible-cell differences.
                bits=0x3f800000 if p<4 else 0x40000000
                for k in range(nk):
                    struct.pack_into('<I',data,t*stride+k*4,bits if k<visible else 0xff800000)
            logical.append(b''.join(data[t*stride:t*stride+nk*4] for t in range(nt)))
            stats=dict(padding_unchanged=True,causal_tail_valid=True,visible_finite=True,all_scores_written=True,
                       invisible_minus_inf_count=sum(nk-min((pos+t+1)//2,nk) for t in range(nt)),guards_intact=True,
                       tiled_mismatches=0 if p<4 else sum(min((pos+t+1)//2,nk) for t in range(nt)),
                       exact_bridge_mismatches=0 if p==5 else None,numerical_passed=True)
            layout='strided' if p<4 else 'dense'
            source=strided if p<4 else dense
            c['providers'].append(dict(provider=p,completion_ordinal=ci*6+p+1,gpu_completed=True,
                symbol=r.SYMBOLS[p],library_index=r.LIBRARIES[p],library_sha256='0'*64,input_layout=layout,
                grid=[(nk+(31 if p<4 else 15))//(32 if p<4 else 16),(nt+7)//8,1],threads=[32,4,1],
                dynamic_threadgroup_bytes=r.TG_BYTES[p],score_stride=stride,scores=self.guarded(bytes(data)),
                inputs_after=[self.guarded(b) for b in source],inputs_unchanged=True,**stats))
        return c


class ReaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.f=Fixture(self.tmp.name)

    def assert_invalid(self, function):
        with self.assertRaises(r.EvidenceError): function()

    def test_complete_case_and_exact_is_not_candidate_oracle(self):
        result=self.f.evaluate()
        self.assertTrue(result['all_numerical_passed'])
        self.assertGreater(result['actual_exact_mismatches_to_tiled_descriptive_only'],0)
        self.assertEqual(result['candidate_mismatches_to_deployed_tiled'],0)

    def test_candidate_last_word_mismatch_is_not_sampled_away(self):
        p=self.f.case['providers'][3]
        # Last logical word is invisible and must be -INF, not A5 or zero.
        self.f.mutate(p['scores']['body'],p['score_stride']*8-4,struct.pack('<I',0))
        self.assert_invalid(self.f.evaluate)

    def test_honest_candidate_mismatch_classifies_numerical_failure(self):
        p=self.f.case['providers'][3]
        # Token 1/key0 is visible. A one-bit finite change fails exact identity.
        self.f.mutate(p['scores']['body'],p['score_stride'],struct.pack('<I',0x3f800001))
        p['tiled_mismatches']=1;p['numerical_passed']=False
        result=self.f.evaluate()
        self.assertFalse(result['all_numerical_passed'])
        self.assertEqual(result['candidate_mismatches_to_deployed_tiled'],1)

    def test_probe_tu_native_must_also_equal_external_tiled(self):
        p=self.f.case['providers'][2]
        self.f.mutate(p['scores']['body'],p['score_stride'],struct.pack('<I',0x3f800001))
        p['tiled_mismatches']=1;p['numerical_passed']=False
        self.assertFalse(self.f.evaluate()['all_numerical_passed'])

    def test_actual_exact_bridge_is_independent_required_gate(self):
        p=self.f.case['providers'][5]
        self.f.mutate(p['scores']['body'],p['score_stride'],struct.pack('<I',0x40000001))
        p['exact_bridge_mismatches']=1;p['numerical_passed']=False
        self.assertFalse(self.f.evaluate()['all_numerical_passed'])

    def test_input_last_byte_mutation_even_with_rehashed_payload(self):
        d=self.f.case['providers'][3]['inputs_after'][2]['body']
        self.f.mutate(d,d['bytes']-1,b'\x81')
        self.assert_invalid(self.f.evaluate)

    def test_input_guard_mutation_is_not_hidden_by_body_equality(self):
        d=self.f.case['providers'][3]['inputs_after'][0]['suffix_guard']
        self.f.mutate(d,63,b'\0')
        self.assert_invalid(self.f.evaluate)

    def test_output_guard_mutation_is_required(self):
        d=self.f.case['providers'][3]['scores']['prefix_guard']
        self.f.mutate(d,0,b'\0')
        self.assert_invalid(self.f.evaluate)

    def test_poison_unwritten_whole_invisible_block_rejected(self):
        p=self.f.case['providers'][3]
        self.f.mutate(p['scores']['body'],32*4,struct.pack('<I',0xa5a5a5a5))
        self.assert_invalid(self.f.evaluate)

    def test_identical_visible_nan_across_all_oracles_still_fails_finite(self):
        for p in self.f.case['providers']:
            self.f.mutate(p['scores']['body'],p['score_stride'],struct.pack('<I',0x7fc12345))
        self.assert_invalid(self.f.evaluate)

    def test_bool_cannot_impersonate_provider_or_completion_integer(self):
        self.f.case['providers'][1]['provider']=True
        self.assert_invalid(self.f.evaluate)
        self.f.case['providers'][1]['provider']=1
        self.f.case['providers'][0]['completion_ordinal']=True
        self.assert_invalid(self.f.evaluate)

    def test_int_cannot_impersonate_numerical_boolean(self):
        self.f.case['providers'][3]['numerical_passed']=1
        self.assert_invalid(self.f.evaluate)

    def test_missing_or_reordered_provider_rejected(self):
        self.f.case['providers'].pop()
        self.assert_invalid(self.f.evaluate)

    def test_distinct_named_post_witness_may_hardlink_identical_bytes(self):
        before=self.f.case['inputs_before']['strided'][0]['body']
        after=self.f.case['providers'][3]['inputs_after'][0]['body']
        (self.f.root/after['path']).unlink()
        os.link(self.f.root/before['path'],self.f.root/after['path'])
        after['storage_alias_of']=before['path']
        self.assertTrue(self.f.evaluate()['all_numerical_passed'])

    def test_reusing_before_as_post_is_rejected_with_consistent_inventory(self):
        pr=self.f.case['providers'][3]
        # Mirror a fully self-consistent tampering attempt: drop old after raw
        # files/inventory and reuse the identical preimage descriptor instead.
        old=pr['inputs_after'][0]
        for d in old.values():
            (self.f.root/d['path']).unlink();self.f.files.remove(d)
        pr['inputs_after'][0]=copy.deepcopy(self.f.case['inputs_before']['strided'][0])
        with self.assertRaisesRegex(r.EvidenceError,'distinct filename'):
            self.f.evaluate()

    def test_reusing_identical_guard_path_is_not_new_observation(self):
        pr=self.f.case['providers'][3]
        pr['scores']['suffix_guard']=copy.deepcopy(pr['scores']['prefix_guard'])
        with self.assertRaisesRegex(r.EvidenceError,'distinct filename'):
            self.f.evaluate()

    def test_strict_raw_sha_and_length_before_semantics(self):
        d=self.f.case['providers'][3]['scores']['body']
        path=self.f.root/d['path'];b=path.read_bytes();path.write_bytes(b[:-4])
        self.assert_invalid(self.f.evaluate)

    def test_wrong_glm72_field_order_is_not_accepted_as_dsv4(self):
        self.f.mutate(self.f.case['args72'],0,struct.pack('<I',32))
        self.assert_invalid(self.f.evaluate)

    def test_padded_strides_and_real_full_f32_profile(self):
        with tempfile.TemporaryDirectory() as t:
            f=Fixture(t,1);self.assertTrue(f.evaluate()['all_numerical_passed'])
            p=f.case['providers'][3]
            f.mutate(p['scores']['body'],f.a['n_comp']*4,b'\0')
            self.assert_invalid(f.evaluate)

    def test_signed_zero_and_subnormal_profile_is_actual_not_label(self):
        with tempfile.TemporaryDirectory() as t:
            f=Fixture(t,2);result=f.evaluate()
            self.assertTrue(result['all_numerical_passed'])
            self.assertGreater(result['input_features'][0]['subnormal_count'],0)
            # Change the dense pack, preserving its own declared SHA.
            f.mutate(f.case['inputs_before']['dense'][0]['body'],12,struct.pack('<I',0))
            self.assert_invalid(f.evaluate)

    def test_narrow_scope_rejects_small_prompts_without_gpu(self):
        a=copy.deepcopy(self.f.a)
        for field,value in [('n_tokens',1),('n_tokens',7),('n_comp',512),('n_comp',16385),('ratio',1),('pos0',32761)]:
            b=copy.deepcopy(a);b[field]=value
            self.assert_invalid(lambda:r.scope(b))

    def test_duplicate_stdout_json_key_rejected(self):
        path=self.f.root/'stdout';path.write_text('{"kind":"complete","kind":"timing"}\n')
        self.assert_invalid(lambda:r.read_stdout(path))

    def test_protocol_counter_bool_is_not_integer(self):
        self.assert_invalid(lambda:r.exact(True,1,'count'))


class WindowAndIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.stdout=self.root/'stdout.jsonl'
        self.m=dict(libraries=[dict(index=i,label=('deployed','native_full_tu','probe_full_tu')[i],
               path=f'/tmp/lib{i}',bytes=3,sha256=str(i)*64) for i in range(3)],
               bindings=[dict(provider=i) for i in range(6)],cases=[],all_numerical_passed=True,
               status='correctness_complete_not_adoption')
        self.records=[]
        for lib in self.m['libraries']:self.records.append(dict(kind='loaded_library',**lib))
        for b in self.m['bindings']:self.records.append(dict(kind='pipeline',binding=b))
        for ci,c in enumerate(r.CASES):
            case=dict(args=r.case_description(c),providers=[])
            for p in range(6):
                pr=dict(library_sha256=self.m['libraries'][r.LIBRARIES[p]]['sha256'],gpu_completed=True,
                   numerical_passed=True,tiled_mismatches=0,exact_bridge_mismatches=0 if p==5 else None)
                case['providers'].append(pr)
                self.records.extend([dict(kind='gpu_completion',ordinal=ci*6+p+1,case=ci,provider=p,status=4,
                    completed=True,symbol=r.SYMBOLS[p],library_sha256=pr['library_sha256'],error=None),
                    dict(kind='provider_result',case=ci,provider=p,numerical_passed=True,tiled_mismatches=0,
                    exact_bridge_mismatches=pr['exact_bridge_mismatches'])])
            self.records.append(dict(kind='case_complete',case=ci,args=case['args'],providers=6));self.m['cases'].append(case)
        self.records.append(dict(kind='complete',status='correctness_complete_not_adoption',case_count=12,
           gpu_completion_count=72,all_numerical_passed=True,timing_collected=False,adoption_eligible=False))

    def check(self, passed=True):
        self.stdout.write_text(''.join(json.dumps(x)+'\n' for x in self.records))
        return r.verify_stdout(self.m,self.stdout,passed)

    def test_all72_strict_completion_and_no_timings(self):
        self.assertTrue(self.check())

    def test_missing_completion_or_duplicate_ordinal_is_invalid(self):
        self.records[9]['ordinal']=2
        with self.assertRaises(r.EvidenceError):self.check()

    def test_wrong_actual_symbol_is_invalid_even_if_scores_claim_equal(self):
        self.records[9]['symbol']=r.SYMBOLS[3]
        with self.assertRaises(r.EvidenceError):self.check()

    def test_status_integer_cannot_be_boolean(self):
        self.records[9]['status']=True
        with self.assertRaises(r.EvidenceError):self.check()

    def test_extra_timing_record_is_invalid(self):
        self.records.append(dict(kind='timing',gpu_us=1.0))
        with self.assertRaises(r.EvidenceError):self.check()

    def test_complete_gpu_failure_preserved_without_success(self):
        self.records[9].update(status=5,completed=False,error='fixture GPU fault')
        self.records[10]['numerical_passed']=False
        self.m['cases'][0]['providers'][0].update(gpu_completed=False,numerical_passed=False)
        self.m.update(all_numerical_passed=False,status='numerical_failed')
        self.records[-1].update(all_numerical_passed=False,status='numerical_failed')
        self.assertFalse(self.check(False))

    def test_lying_completion_status_is_invalid(self):
        self.records[9]['status']=5
        with self.assertRaises(r.EvidenceError):self.check()

    def test_build_binary_identity_binds_both_manifests(self):
        binary=self.root/'bench';binary.write_bytes(b'compiled fixture')
        d=dict(path=str(binary),bytes=binary.stat().st_size,sha256=r.sha(binary.read_bytes()))
        good=dict(artifacts=dict(bench_indexer_register_only=d))
        r.verify_compiled_binary(r.Paths(),good,good,d)
        other=self.root/'other';other.write_bytes(b'other binary')
        bad=dict(path=str(other),bytes=other.stat().st_size,sha256=r.sha(other.read_bytes()))
        with self.assertRaises(r.EvidenceError):r.verify_compiled_binary(r.Paths(),good,
            dict(artifacts=dict(bench_indexer_register_only=bad)),d)
        with self.assertRaises(r.EvidenceError):r.verify_compiled_binary(r.Paths(),dict(artifacts={}),good,d)

    def test_complete_host_before_is_bound_not_only_profile_counts(self):
        gpu_dir=self.root/'gpu';host_dir=self.root/'host';gpu_dir.mkdir();host_dir.mkdir()
        gpu=Fixture(gpu_dir,1);host=Fixture(host_dir,1)
        r.compare_host_before(gpu.case,host.case,gpu.payloads(),host.payloads())
        d=gpu.case['inputs_before']['strided'][0]['body']
        gpu.mutate(d,d['bytes']-1,b'\0')
        with self.assertRaises(r.EvidenceError):r.compare_host_before(gpu.case,host.case,gpu.payloads(),host.payloads())

    def test_frozen_host_inputs_have_no_outputs_or_numerical_claim(self):
        host_dir=self.root/'host';host_dir.mkdir();host=Fixture(host_dir,0)
        host.case['providers']=[]
        result=r.review_case(host.case,host.payloads(),r.CASES[0],0,True)
        self.assertFalse(result['numerical_passed'])
        with self.assertRaises(r.EvidenceError):r.exact(json.loads('{"all_numerical_passed":0}')['all_numerical_passed'],False,'actual historical host NSNumber int')

    def test_independent_production_builder_literal_ast_never_executes(self):
        source=self.root/'source';(source/'tools').mkdir(parents=True);(source/'metal').mkdir()
        order=[f'shader{i}.metal' for i in range(29)]
        builder="raise RuntimeError('must never execute')\nSHADER_FILES="+repr(order)+"\nPREAMBLE='pre'\nCOPY_KERNELS='copy'\n"
        (source/'tools/build_metallib.py').write_text(builder)
        expected=b'pre\n'
        for i,name in enumerate(order):
            body=f'shader body {i}'.encode();(source/'metal'/name).write_bytes(body)
            expected+=('\n#line 1 "'+name+'"\n').encode()+body
        expected+=b'copy\n'
        assembled,actual_order=r.assemble_frozen_production(source)
        self.assertEqual(assembled,expected);self.assertEqual(actual_order,order)

    def test_cli_preserves_existing_report(self):
        report=self.root/'report.json';report.write_text('retained evidence\n')
        with mock.patch('sys.argv',['reader','--protocol',str(self.root/'missing'),'--out',str(report)]),contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(r.main(),2)
        self.assertEqual(report.read_text(),'retained evidence\n')


if __name__=='__main__':unittest.main()
