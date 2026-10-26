#!/usr/bin/env python3
"""Independent CPU-only real-capture gate for register-only head-control v2.

Frozen old low-level capture, scores, ordered-Top512, input and guard routines
are reused by SHA; their old protocol/window/performance verdict is never used.
No subprocess/Metal calls. Numerical failures remain structured completed
reviews; contradictory/missing provenance is evidence_invalid. No timing or
full-model eligibility is granted by this stage.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys

sys.dont_write_bytecode=True
CORE_SHA='c403c38f2d254d0d4e113afad0e27e32ddf6684d0faeee2d70cac9d9001303f5'
METAL_READER_SHA='d4d41f62d66f4789ee97bf6755cc336e7860c638a3035c9854f529e474f7ae21'
ORIGINAL_SHA='31f63df2a6fcbe3bd0d12a76723828c7e8f1a51cd981fa4f58da2941b669cefe'
CANDIDATE_SHA='a62922c6f5937236bd0e354dd06943b39bfa5f788419c16a8700da793769d616'
V1_CANDIDATE_SHA='a0e6ea89dbe80862593ec0b89b02bdacf75efd83271d53c3a9b0c18f0adbfecb'
SUBJECT='indexer_register_only_head_control_v2'
SYMBOL='kernel_myds_indexer_scores_register_only_bf16'


def sha(b):return hashlib.sha256(b).hexdigest()


def checked_module(path,digest,name):
    p=Path(path)
    if sha(p.read_bytes())!=digest:raise ValueError('frozen dependency SHA: '+str(p))
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


core=checked_module(Path(__file__).with_name('review_indexer_k_capture.py'),CORE_SHA,'register_capture_core')
require=core.require
read_json=core.read_json
file_sha=core.file_sha
EvidenceError=core.EvidenceError


def exact(a,b,label):
    if isinstance(b,dict):
        require(type(a)is dict and set(a)==set(b),label+' keys')
        for k in b:exact(a[k],b[k],label+'.'+k)
    elif isinstance(b,list):
        require(type(a)is list and len(a)==len(b),label+' list')
        for x,y in zip(a,b):exact(x,y,label)
    else:require(type(a)is type(b) and a==b,label+' value/type')


def no_timing(records):
    require(not any(r.get('kind') in ('real_timing','real_summary') for r in records),'correctness-only: no timing records')


def synthetic_gate(s,metal,spr,gold_sha):
    require(s['evidence_valid']is True and s['numerical_passed']is True and s['performance_evaluated']is False and
            s['device']=='Apple M2 Ultra' and s['gpu_completion_count']==72 and s['raw_file_count']==1106 and
            s['case_count']==12 and s['reader_sha256']==METAL_READER_SHA and s['classification']=='synthetic_correctness_passed',
            'M2 synthetic full-payload prerequisite from frozen independent reader')
    require(s['build_proof']['combined_source_sha256']==metal['combined_source_sha256'] and
            s['build_proof']['native_source_sha256']==metal['native_source_sha256'],'synthetic prerequisite same full-TU source')
    require(spr['stage']=='synthetic_correctness' and spr['correctness_only']is True and spr['timing_allowed']is False and
            spr['expected_gpu_completion_count']==72 and spr['artifacts']['reader']['sha256']==METAL_READER_SHA,
            'same frozen synthetic scope and independent reader')
    for k in ('probe','native'):
        require(spr['artifacts'][k+'_library']['sha256']==metal['metallibs'][k]['sha256'],'synthetic same actual library '+k)
    require(spr['artifacts']['deployed_library']['sha256']==gold_sha,'synthetic deployed-gold bridge')


class Review(core.Review):
    def __init__(self,args):
        super().__init__(args)
        self.report.update(reader='independent_register_only_real_capture_reader',reader_sha256=file_sha(__file__),
                           subject=SUBJECT,correctness_only=True,performance_evaluated=False,fullmodel_eligible=False,
                           adoption_eligible=False,evidence_valid=False)

    def desc(self,d):
        require(type(d)is dict and set(d)=={'path','bytes','sha256'},'exact artifact descriptor keys')
        p=self.mapped(d['path']);require(Path(d['path']).is_absolute() and type(d['bytes'])is int and d['bytes']>0,'absolute positive artifact')
        require(p.is_file() and not p.is_symlink() and p.stat().st_size==d['bytes'] and file_sha(p)==d['sha256'],'complete frozen artifact: '+str(p))
        return p

    def check_protocol(self):
        p=self.protocol
        for k,v in dict(schema_version=1,kind='indexer_register_only_real_capture_correctness_protocol',subject=SUBJECT,
                        correctness_only=True,timing_allowed=False,fullmodel_run=False,production_adoption=False,
                        layers=[2,8,14],tokens_used=32768,global_chunk_start=30720,expected_score_completions=9,
                        expected_topk_completions=9,expected_raw_files=99,selected_symbol=SYMBOL,expected_device='Apple M2 Ultra').items():exact(p[k],v,'protocol '+k)
        expected={'reader','core_reader','metal_reader','builder','build_manifest','metal_manifest','candidate_inverse',
                  'capture_verified','deployed_library','initialization_reference','window_wrapper','synthetic_review'}
        require(type(p['artifacts'])is dict and set(p['artifacts'])==expected,'exact complete protocol artifact set')
        self.frozen={k:self.desc(d) for k,d in p['artifacts'].items()}
        require(file_sha(__file__)==p['artifacts']['reader']['sha256'] and p['artifacts']['core_reader']['sha256']==CORE_SHA and
                p['artifacts']['metal_reader']['sha256']==METAL_READER_SHA,'executed reader and frozen low-level dependencies')
        require(file_sha(core.__file__)==self.protocol['artifacts']['core_reader']['sha256'],'executed low-level core bytes equal protocol artifact')
        require(self.frozen['build_manifest'].resolve()==self.args.build_manifest.resolve() and
                self.frozen['metal_manifest'].resolve()==self.args.metal_manifest.resolve() and
                self.frozen['capture_verified'].resolve()==self.args.capture_verified.resolve(),'CLI artifacts equal protocol frozen artifacts')
        for n,h in p['input_sha256'].items():require(file_sha(core.safe_child(self.args.input_root,n))==h,'frozen capture input '+n)
        s=read_json(self.frozen['synthetic_review'])
        # A successful synthetic report names its exact libraries in its frozen protocol.
        sp=self.mapped(p['synthetic_protocol_path']);require(file_sha(sp)==s['protocol_sha256'],'synthetic report/protocol SHA')
        synthetic_gate(s,self.metal,read_json(sp),p['artifacts']['deployed_library']['sha256'])
        require(type(p['canonical_hashes'])is dict and len(p['canonical_hashes'])==3 and
                {Path(n).name for n in p['canonical_hashes']}=={'myds','myds.metallib','myds.conf'} and
                all(Path(n).is_absolute() and type(h)is str and re.fullmatch('[0-9a-f]{64}',h) for n,h in p['canonical_hashes'].items()),'three frozen canonical production artifacts')
        self.report.update(protocol_sha256=file_sha(self.args.protocol),synthetic_review_sha256=file_sha(self.frozen['synthetic_review']),
                           protocol_input_files_verified=len(p['input_sha256']))

    def check_build(self):
        b=self.build;root=self.args.build_manifest.resolve().parent;source=root/'source'
        for k,v in dict(schema_version=1,kind='indexer_register_only_real_capture_build',subject=SUBJECT,selected_symbol=SYMBOL,
                        correctness_only=True,timing_present=False,compile_only_pass=True,gpu_work_submitted=False,
                        binary_executed=False,production_registered=False,topk_api='myds_v41_gpu_topk_causal').items():exact(b[k],v,'replay build '+k)
        inventory=b['source_inventory'];require(type(inventory)is dict and inventory,'complete replay source inventory')
        require(sha(json.dumps(inventory,sort_keys=True,separators=(',',':')).encode())==b['source_inventory_sha256'],'canonical replay source inventory SHA')
        require(set(inventory)=={str(p.relative_to(source))for p in source.rglob('*')if p.is_file()},'complete exact replay source file set')
        for rel,d in inventory.items():
            p=core.safe_child(source,rel);require(p.stat().st_size==d['bytes'] and file_sha(p)==d['sha256'],'replay compiler-visible source '+rel)
        require(inventory['tools/build_indexer_register_capture.py']['sha256']==self.protocol['artifacts']['builder']['sha256'],'builder in source snapshot')
        binary=self.desc(b['binary']);self.binary_sha=b['binary']['sha256']
        proof=read_json(self.desc(b['source_transform']))
        generated=(source/b['generated_source']).read_bytes();original=(source/b['original_seed']).read_bytes();restored=generated
        require(sha(original)==ORIGINAL_SHA==b['original_replay_sha256'] and sha(generated)==b['generated_source_sha256']==proof['generated_sha256'],'frozen seed/generated source hashes')
        require(proof['edit_count']==12 and len(proof['exact_edits'])==12,'exact twelve reversible source edits')
        for e in reversed(proof['exact_edits']):
            a=e['after'].encode();require(a and restored.count(a)==1,'unique replay inverse anchor');restored=restored.replace(a,e['before'].encode(),1)
        require(restored==original and proof['inverse_restored_sha256']==ORIGINAL_SHA and proof['inverse_restores_original_byte_for_byte']is True,'complete replay inverse byte proof')
        require(not any(x in generated for x in (b'@"real_timing"',b'@"real_summary"',b'GPUStartTime',b'GPUEndTime',b'for(unsigned sample=')) and
                generated.endswith(b'myds_metal_cleanup();finish(@"correctness_only_complete");return 0;\n}}\n'),'all timing code absent and explicit return')
        require(len(b['commands'])==7,'six compilation units and one linker')
        units=[]
        for c in b['commands'][:-1]:
            require(c[:2]==['xcrun','clang'] and '-c'in c,'recorded CPU compiler')
            path=self.mapped(c[c.index('-c')+1]).resolve();units.append(path.relative_to(source).as_posix())
        require(units==['src/myds_v41_gpu.c','src/myds_gguf.c','src/myds_log.c','src/myds_configfile.c','src/myds_metal.m',b['generated_source']],'actual production TopK compilation units')
        metal=checked_module(self.frozen['metal_reader'],METAL_READER_SHA,'register_capture_metal_core')
        identity=read_json(self.desc(self.metal['source_identity']));md=self.args.metal_manifest.resolve().parent
        artifacts=dict(native_library=self.metal['metallibs']['native'],probe_library=self.metal['metallibs']['probe'],
                       binary=self.metal['artifacts']['bench_indexer_register_only'],harness=dict(sha256=identity['source_inventory']['tools/bench_indexer_register_only.m']['sha256']))
        maps=[(str(a),str(b))for a,b in self.maps]
        self.report['metal_build_proof']=metal.verify_build(metal.Paths(maps),self.metal,artifacts)
        candidate=(md/'candidate.metal.inc').read_bytes();require(sha(candidate)==CANDIDATE_SHA,'single frozen head-control v2 candidate')
        inv=read_json(self.frozen['candidate_inverse']);r=candidate
        require(len(inv['candidate_edits'])==2,'only two local head epilogues')
        for e in reversed(inv['candidate_edits']):
            a=e['after'].encode();require(r.count(a)==1,'unique candidate inverse anchor');r=r.replace(a,e['before'].encode(),1)
        require(sha(r)==V1_CANDIDATE_SHA and inv['candidate_inverse_restores_v1_exact']is True,'candidate inverse restores failed v1 SHA')
        require(self.protocol['artifacts']['deployed_library']['sha256']==self.capture['loaded_metallib_sha256'],'same deployed actual capture library')
        self.report.update(build_manifest_sha256=file_sha(self.args.build_manifest),metal_build_manifest_sha256=file_sha(self.args.metal_manifest),
                           replay_source_files_verified=len(inventory),binary_sha256=self.binary_sha)

    def check_capture(self):
        super().check_capture()
        for c in self.cases.values():require(c['metadata']['cache_bf16']is True and c['metadata']['provider']=='kernel_myds_indexer_score_chunk_exact','actual exact/BF16 frozen scope')

    def check_window(self):
        directory=self.args.window_dir.resolve(strict=True);plan=read_json(directory/'execution-plan.json');w=read_json(directory/'window.json');monitor=read_json(directory/'monitor.json')
        for k,h in dict(protocol_sha256=file_sha(self.args.protocol),reader_sha256=file_sha(__file__),build_manifest_sha256=file_sha(self.args.build_manifest),
                        metal_build_manifest_sha256=file_sha(self.args.metal_manifest),binary_sha256=self.binary_sha,
                        probe_sha256=self.metal['metallibs']['probe']['sha256'],native_sha256=self.capture['loaded_metallib_sha256'],
                        capture_json_sha256=file_sha(self.args.capture_verified),config_sha256=self.capture['config_sha256'],
                        wrapper_sha256=self.protocol['artifacts']['window_wrapper']['sha256']).items():require(plan[k]==h,'execution-plan frozen '+k)
        require(w['protocol_sha256']==plan['protocol_sha256'] and w['reader_sha256']==plan['reader_sha256'],'window protocol/reader binding')
        exact(plan['MYDS_inherited'],{},'clean inherited MYDS');exact(plan['fullmodel'],False,'component-only window')
        argv=plan['argv'];require(len(argv)==11 and argv[1::2]==['--capture-json','--metallib','--native-metallib','--config','--out-dir'],'exact correctness-only argv (no samples)')
        for raw,h in ((argv[0],self.binary_sha),(argv[2],file_sha(self.args.capture_verified)),(argv[4],plan['probe_sha256']),(argv[6],plan['native_sha256']),(argv[8],plan['config_sha256'])):
            require(file_sha(self.mapped(raw))==h,'executed argv operand full SHA')
        require(self.mapped(argv[0]).resolve()==self.desc(self.build['binary']).resolve(),'exact executed binary')
        require(self.mapped(argv[6]).resolve()==self.frozen['deployed_library'].resolve(),'actual deployed library path')
        self.result_dir=self.args.result_dir.resolve(strict=True) if self.args.result_dir else directory/'outputs'
        require(self.mapped(argv[10]).resolve()==self.result_dir.resolve(),'exact child output directory')
        exact(w['canonical_before'],self.protocol['canonical_hashes'],'frozen canonical before');exact(w['canonical_after'],w['canonical_before'],'canonical after');exact(w['canonical_unchanged'],True,'canonical unchanged')
        child=w['child_pid'];require(type(child)is int and child>0 and type(monitor)is list and monitor,'launched child + nonempty monitor')
        previous=-1.;seen=False;exclusive=not w['recognized_before'] and not w['recognized_after'] and w['abort_reason']is None
        for s in monitor:
            t=s['elapsed_s'];require(type(t)in(int,float) and math.isfinite(t) and t>=previous,'ordered finite monitor');previous=t
            for row in s['recognized']:
                require(type(row['pid'])is int and type(row['comm'])is str,'monitor PID/comm')
                seen|=row['pid']==child and Path(row['comm']).name=='bench_indexer_register_capture';exclusive&=row['pid']==child
        require(plan['started_epoch']<=w['finished_epoch'] and 0<=w['elapsed_s']<=plan['timeout_s']+20,'bounded actual window')
        self.window_gate=self.gate('window','recognized_process_exclusivity',exclusive and seen,child_pid=child,monitor_samples=len(monitor),child_seen=seen)
        self.plan,self.window=plan,w;self.report['window']=self.window_gate
        return True

    def load_outputs(self):
        mp=self.result_dir/'artifact-manifest.json';m=self.manifest=read_json(mp)
        for k,v in dict(schema_version=1,probe='indexer_register_only_real_capture',fixture_kind='actual_fullmodel_prefill',correctness_only=True,
                        timing_collected=False,selected_symbol=SYMBOL,candidate_dynamic_threadgroup_bytes=0,adoption_eligible=False,
                        production_registered=False,fullmodel=False,cold_dram_proven=False,performance_from_capture=False,
                        host_only=False,gpu_work_submitted=True,mode='concurrent',samples=0,warmups=0).items():exact(m[k],v,'actual manifest '+k)
        exact(m['device'],self.protocol['expected_device'],'frozen actual target device')
        for k,h in dict(capture_json_sha256=file_sha(self.args.capture_verified),binary_sha256=self.binary_sha,probe_sha256=self.metal['metallibs']['probe']['sha256'],
                        native_sha256=self.capture['loaded_metallib_sha256'],config_sha256=self.capture['config_sha256']).items():require(m[k]==h,'output manifest artifact '+k)
        require(core.harness_flag(m['topk_replayed_with_actual_helper']) and m['topk_loaded_library_sha256']==m['native_sha256'] and
                self.mapped(m['topk_loaded_library_path']).resolve()==self.frozen['deployed_library'].resolve() and
                file_sha(self.mapped(m['topk_loaded_library_path']))==m['native_sha256'],'actual original GPU TopK loaded exact native library')
        for d in m['files']:
            n=d['path'];require(n not in self.raw,'unique raw filename');p=core.safe_child(self.result_dir,n,single=True)
            require(p.stat().st_size==d['bytes'] and file_sha(p)==d['sha256'],'complete raw SHA '+n);self.raw[n]=p
        require(set(p.name for p in self.result_dir.iterdir())==set(self.raw)|{'artifact-manifest.json'},'complete actual raw directory file set')
        self.jsonl_path=self.args.jsonl or self.args.window_dir/'stdout.jsonl'
        def parse(path):
            info=[];records=[];started=False
            for line in Path(path).read_text().splitlines():
                require(line.strip(),'no blank stdout lines')
                if line.startswith('{'):started=True;records.append(json.loads(line,object_pairs_hook=core.unique_object));continue
                match=re.fullmatch(r'\[\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d{3}\] \[INFO\] (.+)',line)
                require(not started and match is not None,'only pre-JSON known INFO prefix');info.append(match.group(1))
            return info,records
        info,self.records=parse(self.jsonl_path);reference,_=parse(self.frozen['initialization_reference'])
        require(len(info)==len(reference)==21,'exact 21 initialization messages')
        for i,(a,b)in enumerate(zip(info,reference)):
            if i==0:
                mm=re.fullmatch(r'config: 已加载 (.+) —— 生效 (\d+) 条',a);require(mm and int(mm.group(2))==self.capture['config_declared_unique_count'] and file_sha(self.mapped(mm.group(1)))==m['config_sha256'],'actual config log')
            elif i==1:
                mm=re.fullmatch(r'myds_metal: loaded precompiled library from (.+) \(all required V4\.1 kernels verified\)',a);require(mm and self.mapped(mm.group(1)).resolve()==self.frozen['deployed_library'].resolve() and file_sha(self.mapped(mm.group(1)))==m['native_sha256'],'actual library log')
            elif b.startswith('myds_metal: Initialized successfully on GPU: '):require(a=='myds_metal: Initialized successfully on GPU: '+m['device'],'actual GPU log')
            else:require(a==b,'frozen exact initialization message '+str(i))
        no_timing(self.records)
        require(all(r.get('kind')in {'real_pipeline','real_score_completion','real_inputs','real_parity','real_topk','all_layers_numerical_gate'}for r in self.records),'only correctness record types')
        self.report.update(device=m['device'],candidate_library_sha256=m['probe_sha256'],deployed_library_sha256=m['native_sha256'],
                           artifact_manifest_sha256=file_sha(mp),jsonl_sha256=file_sha(self.jsonl_path),raw_files_verified=len(self.raw),known_initialization_info_lines=21)

    def check_extra(self):
        original_records=self.records
        # The old numerical implementation remains exact. New preimages and
        # bindings are checked independently here rather than widening its scope.
        phase1_records=[r for r in original_records if r['kind']not in ('real_pipeline','real_score_completion') and not(r['kind']=='real_inputs' and r['phase']=='phase0_pre_dispatch')]
        # The frozen core's joint gate covers phase1 only. Project that gate
        # from the phase1 declarations, let the core independently validate
        # every declaration from raw, then compare the ORIGINAL new joint gate
        # against phase1 AND preimages below. No raw record is modified.
        raw_gates=[r for r in phase1_records if r['kind']=='all_layers_numerical_gate']
        require(len(raw_gates)==1,'one original joint correctness gate')
        raw_gate=raw_gates[0]
        phase1_claim=all(bool(r['passed']) for r in phase1_records if r['kind']in ('real_parity','real_topk','real_inputs'))
        projection=dict(raw_gate,passed=phase1_claim,ordered_top512_all_providers_equal_capture=phase1_claim)
        self.records=[projection if r is raw_gate else r for r in phase1_records]
        original_manifest=self.manifest
        self.manifest=dict(original_manifest,all_layers_numerical_gate_passed=phase1_claim)
        try:super().check_numerical()
        finally:self.manifest=original_manifest;self.records=original_records
        expected=[];ordinal=0;pre_ok=True;score_completed=True
        for layer in core.LAYERS:
            c=self.cases[layer];keys,rows=c['keys'],c['rows']
            for p in range(3):expected.append(('real_pipeline',layer,p))
            for name in core.INPUTS:expected.append(('real_inputs',layer,'phase0_pre_dispatch',name))
            for p in range(3):expected.extend([('real_score_completion',layer,p),('real_parity',layer,p)])
            for p in range(3):expected.append(('real_topk',layer,p))
            for name in core.INPUTS:expected.append(('real_inputs',layer,'phase1_post_topk',name))
        expected.append(('all_layers_numerical_gate',))
        actual=[]
        for r in original_records:
            kind=r['kind']
            if kind!='all_layers_numerical_gate':require(type(r['layer'])is int,'strict integer record layer')
            if kind not in ('all_layers_numerical_gate','real_inputs'):require(type(r['provider'])is int and 0<=r['provider']<3,'strict integer record provider')
            actual.append((kind,r['layer'],r['phase'],r['input'])if kind=='real_inputs' else (kind,)if kind=='all_layers_numerical_gate' else (kind,r['layer'],r['provider']))
        require(actual==expected,'complete strict per-layer provider/input/score/TopK order with no duplicates or timing')
        for r in original_records:
            kind=r['kind']
            if kind not in ('real_pipeline','real_score_completion','real_inputs','real_topk'):continue
            layer=r['layer'];c=self.cases[layer]
            if kind=='real_pipeline':
                p=r['provider'];grid=[(c['keys']+(15 if p==0 else 31))//(16 if p==0 else 32),(c['rows']+7)//8,1]
                expected_p=dict(kind=kind,layer=layer,provider=p,symbol=('kernel_myds_indexer_score_chunk_exact','kernel_dsv4_indexer_scores_tiled_bf16',SYMBOL)[p],
                                library_sha256=self.capture['loaded_metallib_sha256']if p==0 else self.metal['metallibs']['probe']['sha256'],
                                argument_bytes=36 if p==0 else 72,threads=[32,4,1],grid=grid,dynamic_threadgroup_bytes=(5120,21504,0)[p],static_threadgroup_bytes=0)
                exact(r,expected_p,'actual selected pipeline/ABI/geometry/zeroTG')
            elif kind=='real_score_completion':
                ordinal+=1;require(type(r['status'])is int and r['status']in range(6) and type(r['completed'])is bool and r['completed']==(r['status']==4),'GPU completion/status producer type')
                require(type(r['ordinal'])is int and r['ordinal']==ordinal and (r['error']is None if r['completed']else r['error']is None or type(r['error'])is str),'ordered GPU completion/error')
                score_completed&=r['completed'];self.gate('gpu_completion',f'score-{ordinal}',r['completed'],**r)
            elif kind=='real_inputs' and r['phase']=='phase0_pre_dispatch':
                name=r['input'];captured=c['payloads'][name];stem=f'layer-{layer}-phase0_pre_dispatch-{name}'
                require(r['available']is True and r['bytes']==len(captured) and r['post_raw']==stem+'-post.bin' and
                        r['guard_prefix_raw']==stem+'-guard-prefix.bin' and r['guard_suffix_raw']==stem+'-guard-suffix.bin' and r['guard_bytes_per_side']==64,'complete initial GPU-buffer witness')
                post=self.artifact(r['post_raw']);a=self.artifact(r['guard_prefix_raw']);b=self.artifact(r['guard_suffix_raw']);equal=post==captured;intact=a==b==core.GUARD
                require(r['capture_sha256']==sha(captured) and r['post_sha256']==sha(post) and r['guard_prefix_sha256']==sha(a) and r['guard_suffix_sha256']==sha(b) and
                        r['bytes_equal_capture']==equal and r['guards_intact']==intact and r['passed']==(equal and intact),'raw initial input witness declarations')
                pre_ok&=equal and intact;self.gate('input_preimage',stem,equal and intact,bytes=len(post))
            elif kind=='real_topk':
                p=r['provider'];post=self.artifact(f'layer-{layer}-provider-{p}-topk-score-post.f32');same=post==self.initial_scores[(layer,p)]
                require(r['score_input_unchanged']==same,'consumer score-input runtime witness agrees with retained post raw')
                pre_ok&=same;self.gate('consumer_score_integrity',f'layer-{layer}-provider-{p}',same,bytes=len(post))
        require(self.manifest['score_gpu_completion_count']==ordinal==9,'all nine actual score CB completions')
        require(len(self.raw)==99,'exact complete 99 raw files')
        require(self.consumed==set(self.raw),'every complete raw independently examined')
        self.all_numerical_passed &= pre_ok and score_completed
        require(raw_gate['passed']==self.all_numerical_passed and
                raw_gate['ordered_top512_all_providers_equal_capture']==self.all_numerical_passed and
                self.manifest['all_layers_numerical_gate_passed']==self.all_numerical_passed,
                'original new joint gate equals independently recomputed phase1 + preimage + GPU gate')
        self.report.update(score_gpu_completion_count=ordinal,all_score_gpu_completed=score_completed,initial_GPU_input_bytes_and_guards_passed=pre_ok,
                           topk_completion_count=len(self.id_reports),all_layers_numerical_gate_passed=self.all_numerical_passed)

    def initialize(self):
        self.protocol=read_json(self.args.protocol);self.capture=read_json(self.args.capture_verified);self.build=read_json(self.args.build_manifest);self.metal=read_json(self.args.metal_manifest)
        self.check_protocol();self.check_build();self.check_capture();return self.report

    def run(self):
        self.initialize();self.check_window();self.load_outputs();self.check_extra()
        completed=self.all_numerical_passed and self.window_gate['passed']
        if self.manifest['status']=='correctness_only_complete':require(self.window['exit_code']==0 and self.all_numerical_passed,'successful completion agrees with independent full raw gate')
        else:require(self.manifest['status']=='failed' and self.window['exit_code']!=0 and self.manifest.get('failure'),'structured failed native execution')
        self.report.update(review_completed=True,evidence_valid=True,numerical_passed=bool(self.all_numerical_passed),
                           correctness_passed=bool(completed),outcome='real_capture_correctness_passed'if completed else 'real_capture_correctness_gate_failed',
                           fullmodel_eligible=False,performance_evaluated=False,adoption_eligible=False,failures=self.issues,
                           child_exit_code=self.window['exit_code'],timing_records=0,selected_symbol=SYMBOL,candidate_dynamic_threadgroup_bytes=0,
                           real_capture_and_top512_bridge_completed=bool(completed),complete_score_table_count=len(self.initial_scores),
                           complete_ordered_top512_table_count=len(self.initial_ids),all_topk_GPU_completed=all(x['GPU_completed']for x in self.id_reports),
                           all_input_bytes_and_guards_passed=all(x['passed']for x in self.input_reports) and self.report['initial_GPU_input_bytes_and_guards_passed'],
                           scope='Three retained real layers only; original GPU Top512 consumer, no performance/full-model gate.')
        return self.report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('capture-verified','protocol','build-manifest','metal-manifest','window-dir','input-root','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--capture-root',type=Path);p.add_argument('--result-dir',type=Path);p.add_argument('--jsonl',type=Path)
    p.add_argument('--path-map',action='append',default=[]);p.add_argument('--initialize-only',action='store_true')
    a=p.parse_args();require(not a.out.exists(),'new report output path');r=None;code=0
    try:
        r=Review(a);report=r.initialize()if a.initialize_only else r.run()
        if a.initialize_only:report.update(initialization_passed=True,evidence_valid=True,numerical_passed=False,review_completed=False)
    except (core.EvidenceError,ValueError,KeyError,OSError,TypeError,IndexError,StopIteration)as e:
        code=2;report=r.report if r else {};report.update(review_completed=False,evidence_valid=False,outcome='evidence_invalid',evidence_error=str(e),error_type=type(e).__name__,numerical_passed=False,correctness_passed=False,fullmodel_eligible=False,production_adoption=False)
    a.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({k:report.get(k)for k in ('evidence_valid','initialization_passed','review_completed','numerical_passed','correctness_passed','outcome','evidence_error')},indent=2));raise SystemExit(code)


if __name__=='__main__':main()
