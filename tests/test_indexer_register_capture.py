"""CPU-only contracts; hand-authored raw fixtures, never Metal/model execution."""
import copy
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import types
import unittest

ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
b=load('reg_capture_builder',ROOT/'tools/build_indexer_register_capture.py')
r=load('reg_capture_reader',ROOT/'tools/review_indexer_register_capture.py')
G=b'\xa5'*64


def fixture(directory, *, candidate_diff=False, actual_diff=False, pre_bad=False, guard_bad=False, ids_bad=False):
    args=types.SimpleNamespace(path_map=[],capture_root=None)
    v=r.Review(args);v.cases={};v.metal={'metallibs':{'probe':{'sha256':'probe'}}};v.capture={'loaded_metallib_sha256':'gold'}
    v.manifest={'status':'failed'if candidate_diff or pre_bad or guard_bad or ids_bad else 'correctness_only_complete','score_gpu_completion_count':9}
    v.records=[];v.raw={};ordinal=0;joint=True
    def raw(name,data):
        assert name not in v.raw;p=directory/name;p.write_bytes(data);v.raw[name]=p;return name
    for layer in (2,8,14):
        scores=struct.pack('<513f',*list(range(512,0,-1)),float('-inf'));actual=bytearray(scores)
        if actual_diff:actual[:4]=struct.pack('<I',struct.unpack('<I',actual[:4])[0]+1)
        actual=bytes(actual);ids=struct.pack('<512i',*range(512));inputs={'q-f32':b'Q'*32,'weights-f32':b'W'*16,'cache':b'K'*24}
        counter=dict(prefix_requests=1,prefix_encoded=1,prefix_unavailable=0,nan_fuse_encoded=1,nan_shuffle_encoded=0)
        v.cases[layer]=dict(rows=1,keys=513,pos=1024,ratio=2,metadata=dict(cache_bf16=True,provider='kernel_myds_indexer_score_chunk_exact',invisible_minus_inf_count=1,actual_topk_consumer=counter),payloads=dict(inputs,**{'scores-f32':actual,'selected-i32':ids}))
        for p in range(3):
            v.records.append(dict(kind='real_pipeline',layer=layer,provider=p,symbol=('kernel_myds_indexer_score_chunk_exact','kernel_dsv4_indexer_scores_tiled_bf16',r.SYMBOL)[p],library_sha256='gold'if p==0 else 'probe',argument_bytes=36 if p==0 else 72,threads=[32,4,1],grid=[33 if p==0 else 17,1,1],dynamic_threadgroup_bytes=(5120,21504,0)[p],static_threadgroup_bytes=0))
        def input_records(phase):
            for name,captured in inputs.items():
                post=captured
                if pre_bad and layer==2 and name=='q-f32' and phase=='phase0_pre_dispatch':post=b'!'+post[1:]
                stem=f'layer-{layer}-{phase}-{name}';pr=raw(stem+'-post.bin',post);a=raw(stem+'-guard-prefix.bin',G);z=raw(stem+'-guard-suffix.bin',G)
                ok=post==captured
                v.records.append(dict(kind='real_inputs',layer=layer,phase=phase,input=name,bytes=len(captured),available=True,capture_sha256=r.sha(captured),post_sha256=r.sha(post),guards_intact=True,bytes_equal_capture=ok,post_raw=pr,guard_prefix_raw=a,guard_suffix_raw=z,guard_bytes_per_side=64,guard_prefix_sha256=r.sha(G),guard_suffix_sha256=r.sha(G),passed=ok))
        input_records('phase0_pre_dispatch')
        outputs=[]
        for p in range(3):
            data=actual if p==0 else scores
            if candidate_diff and layer==2 and p==2:
                x=bytearray(data);x[:4]=struct.pack('<I',struct.unpack('<I',x[:4])[0]+1);data=bytes(x)
            outputs.append(data);ordinal+=1
            v.records.append(dict(kind='real_score_completion',ordinal=ordinal,layer=layer,provider=p,status=4,completed=True,error=None))
            prefix=f'layer-{layer}-provider-{p}';raw(prefix+'-scores.f32',data)
            guard=G*2 if not(guard_bad and layer==2 and p==2) else b'!'+(G*2)[1:];raw(prefix+'-guards.bin',guard)
            ad=r.core.diff_words(actual,data)['mismatches'];cd=r.core.diff_words(scores,data)['mismatches']if p==2 else None
            passed=guard==G*2 and (ad==0 if p==0 else p==1 or cd==0);joint &= passed
            v.records.append(dict(kind='real_parity',layer=layer,provider=p,tokens=1,keys=513,bf16=1,captured_actual_provider='kernel_myds_indexer_score_chunk_exact',mismatches_to_actual_capture=ad,candidate_mismatches_to_tiled=cd,invisible_minus_inf_count=1,full_scores=True,guards_intact=guard==G*2,all_written=True,visible_finite=True,causal_tail_valid=True,passed=passed,topk_mask_validated=False,fullmodel_speedup_proven=False))
        for p in range(3):
            data=ids
            if ids_bad and layer==2 and p==2:data=struct.pack('<2i',1,0)+data[8:]
            prefix=f'layer-{layer}-provider-{p}';raw(prefix+'-selected.i32',data);raw(prefix+'-topk-guards.bin',G*4);raw(prefix+'-topk-score-post.f32',outputs[p])
            diff=r.core.diff_words(ids,data)['mismatches'];passed=diff==0;joint &= passed
            v.records.append(dict(kind='real_topk',layer=layer,provider=p,top=512,rows=1,ordered_ids_complete=True,actual_capture_mismatches=diff,candidate_mismatches_to_tiled=diff if p==2 else None,host_encoded=1,GPU_completed=1,guards_intact=True,score_input_unchanged=True,same_consumer_counter_path_as_capture=True,**counter,passed=passed,actual_mask_filter_for_these_layers=False,fullmodel_speedup_proven=False))
        input_records('phase1_post_topk')
    joint &= not pre_bad
    v.records.append(dict(kind='all_layers_numerical_gate',passed=joint,layers=[2,8,14],complete_scores=True,ordered_top512_all_providers_equal_capture=joint,timing_started=False))
    v.manifest['all_layers_numerical_gate_passed']=joint
    return v


class BuilderTests(unittest.TestCase):
    def test_inverse_byte_exact_and_seed_unchanged(self):
        p=ROOT/b.ORIGINAL;original=p.read_bytes();generated,proof=b.generate(original)
        self.assertEqual(b.inverse(generated,proof['exact_edits']),original);self.assertEqual(p.read_bytes(),original);self.assertEqual(proof['edit_count'],12)
    def test_only_new_candidate_has_zero_tg_and_native_topk_helpers_retained(self):
        old=(ROOT/b.ORIGINAL).read_bytes();s,proof=b.generate(old)
        self.assertIn(b'if(provider!=2u)[e setThreadgroupMemoryLength:',s)
        self.assertIn(b'psos[ci][2].staticThreadgroupMemoryLength==0',s)
        self.assertIn(b'@"completed":cb.status==MTLCommandBufferStatusCompleted?@YES:@NO',s)
        for begin,end in [(b'static BOOL real_index_validate',b'/* The exact production TopK'),(b'static BOOL real_index_inputs',b'static real_index_fixture *')]:
            self.assertEqual(s[s.index(begin):s.index(end,s.index(begin))],old[old.index(begin):old.index(end,old.index(begin))])
        self.assertIn(b'myds_v41_gpu_topk_causal',s);self.assertIn(b'phase0_pre_dispatch',s);self.assertIn(b'-topk-score-post.f32',s)
    def test_no_timing_code_option_or_timestamp_survives(self):
        s,_=b.generate((ROOT/b.ORIGINAL).read_bytes())
        for x in (b'@"real_timing"',b'@"real_summary"',b'GPUStartTime',b'GPUEndTime',b'--samples',b'for(unsigned sample='):self.assertNotIn(x,s)
        self.assertTrue(s.endswith(b'myds_metal_cleanup();finish(@"correctness_only_complete");return 0;\n}}\n'))
    def test_changed_seed_and_inverse_edit_refused(self):
        old=(ROOT/b.ORIGINAL).read_bytes()
        with self.assertRaises(ValueError):b.generate(old+b'\n')
        s,p=b.generate(old);p['exact_edits'][0]['after']+='BROKEN'
        with self.assertRaises(ValueError):b.inverse(s,p['exact_edits'])


class ReaderTests(unittest.TestCase):
    def run_fixture(self,**kw):
        td=tempfile.TemporaryDirectory();self.addCleanup(td.cleanup);v=fixture(Path(td.name),**kw);v.check_extra();return v
    def test_all_99_raw_and_55_records_pass(self):
        v=self.run_fixture();self.assertTrue(v.all_numerical_passed);self.assertEqual(len(v.raw),99);self.assertEqual(len(v.records),55);self.assertEqual(v.consumed,set(v.raw))
    def test_actual_exact_and_tiled_may_differ_but_candidate_must_match_tiled(self):
        self.assertTrue(self.run_fixture(actual_diff=True).all_numerical_passed)
    def test_one_ulp_candidate_is_structured_failure(self):
        v=self.run_fixture(candidate_diff=True);self.assertFalse(v.all_numerical_passed);self.assertEqual(v.score_reports[2]['candidate_mismatches_to_tiled']['mismatches'],1)
    def test_changed_guard_is_structured_failure(self):self.assertFalse(self.run_fixture(guard_bad=True).all_numerical_passed)
    def test_wrong_ordered_ids_structured_failure(self):self.assertFalse(self.run_fixture(ids_bad=True).all_numerical_passed)
    def test_before_buffer_change_structured_failure(self):self.assertFalse(self.run_fixture(pre_bad=True).all_numerical_passed)
    def test_topk_score_post_raw_change_with_honest_false_is_failure(self):
        with tempfile.TemporaryDirectory()as td:
            v=fixture(Path(td));row=next(x for x in v.records if x['kind']=='real_topk');name='layer-2-provider-0-topk-score-post.f32';data=v.raw[name].read_bytes();v.raw[name].write_bytes(b'!'+data[1:]);row['score_input_unchanged']=False
            # Producer TopK passed must also reflect its honest failed input witness.
            row['passed']=False;v.records[-1]['passed']=v.records[-1]['ordered_top512_all_providers_equal_capture']=False;v.manifest['all_layers_numerical_gate_passed']=False
            v.check_extra();self.assertFalse(v.all_numerical_passed)
    def test_topk_post_lie_missing_payload_duplicate_or_wrong_pipeline_refused(self):
        for mutation in ('post_lie','missing','duplicate','pipeline','ordinal','status_bool'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory()as td:
                v=fixture(Path(td))
                if mutation=='post_lie':v.raw['layer-2-provider-0-topk-score-post.f32'].write_bytes(b'!'+v.raw['layer-2-provider-0-topk-score-post.f32'].read_bytes()[1:])
                elif mutation=='missing':del v.raw['layer-2-provider-0-topk-score-post.f32']
                elif mutation=='duplicate':v.records.insert(0,copy.deepcopy(v.records[0]))
                elif mutation=='pipeline':v.records[2]['dynamic_threadgroup_bytes']=21504
                elif mutation=='ordinal':next(x for x in v.records if x['kind']=='real_score_completion')['ordinal']=2
                else:next(x for x in v.records if x['kind']=='real_score_completion')['status']=True
                with self.assertRaises(r.EvidenceError):v.check_extra()
    def test_any_timing_record_refused(self):
        for kind in ('real_timing','real_summary'):
            with self.assertRaises(r.EvidenceError):r.no_timing([{'kind':kind}])
    def test_bool_cannot_impersonate_first_ordinal_or_provider_zero(self):
        for kind,field in [('real_score_completion','ordinal'),('real_score_completion','provider'),('real_pipeline','provider'),('real_parity','provider'),('real_topk','provider')]:
            with self.subTest(kind=kind,field=field),tempfile.TemporaryDirectory()as td:
                v=fixture(Path(td));rec=next(x for x in v.records if x['kind']==kind)
                rec[field]=True if field=='ordinal' else False
                with self.assertRaises(r.EvidenceError):v.check_extra()

    def test_synthetic_M2_prerequisite_identity_not_M4_or_other_reader(self):
        m=dict(combined_source_sha256='combined',native_source_sha256='native',metallibs={'probe':{'sha256':'p'},'native':{'sha256':'n'}})
        q=dict(stage='synthetic_correctness',correctness_only=True,timing_allowed=False,expected_gpu_completion_count=72,
               artifacts={'reader':{'sha256':r.METAL_READER_SHA},'probe_library':{'sha256':'p'},'native_library':{'sha256':'n'},'deployed_library':{'sha256':'g'}})
        report=dict(evidence_valid=True,numerical_passed=True,performance_evaluated=False,device='Apple M2 Ultra',gpu_completion_count=72,
                    raw_file_count=1106,case_count=12,reader_sha256=r.METAL_READER_SHA,classification='synthetic_correctness_passed',
                    build_proof=dict(combined_source_sha256='combined',native_source_sha256='native'))
        r.synthetic_gate(report,m,q,'g')
        for k,value in [('device','Apple M4'),('reader_sha256','other'),('numerical_passed',False),('raw_file_count',1105),('classification','evidence_invalid')]:
            with self.subTest(k=k),self.assertRaises(r.EvidenceError):r.synthetic_gate(dict(report,**{k:value}),m,q,'g')
        wrong=copy.deepcopy(q);wrong['artifacts']['probe_library']['sha256']='other'
        with self.assertRaises(r.EvidenceError):r.synthetic_gate(report,m,wrong,'g')
        with self.assertRaises(r.EvidenceError):r.synthetic_gate(report,m,q,'other-gold')

    def test_exact_schema_refuses_bool_int_confusion(self):
        with self.assertRaises(r.EvidenceError):r.exact({'TG':False},{'TG':0},'contract')


if __name__=='__main__':unittest.main()
