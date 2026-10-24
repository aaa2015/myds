#!/usr/bin/env python3
"""Independently inspect Output_A full-model raw evidence, on CPU only.

No runner analysis routine, inference binary, model execution, Metal API, or
service operation is invoked. A tiny temporary C library accelerates byte FNV;
its only entry point hashes a supplied memory span. Archives are never extracted.
"""
import argparse
import ctypes
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import subprocess
import sys
import tarfile
import tempfile

FLAG='MYDS_ATTN_OUT_A_FIXED_WRAPPER'
FIXED='kernel_myds_outa_q4_v41_fixed_bf16'
PRODUCTION_BINARY='29058d822c95d86ffdc6e84798fe87093d7f8c094b1002f1250177b12066d72b'
PRODUCTION_LIBRARY='c692d821aed43ecce1cd9003f11e19083cbe5ce2923bef44a8df23c1182dfc81'
IDENTITY_KEYS=('argv_raw_hex','myds_env_raw_hex','cwd','log','nice','binary','metallib',
               'binary_sha256','metallib_sha256','config','config_sha256','effective_myds_env')
SCALAR_ORDER=['ctx_pos','rng_state','max_seq_len','max_new_tokens','seed_u32','kv_capacity',
              'pcache_len','anchor_len','continue_from_cache','v41_valid','next_position',
              'engram_position','snapshot_allocated_bytes','prefill_capacity','low_kv_enabled']


def need(value,message):
    if not value: raise ValueError(message)


def load(path):
    def object_pairs(pairs):
        result={}
        for key,value in pairs:
            need(key not in result,'Duplicate JSON key: '+key);result[key]=value
        return result
    return json.loads(path.read_text(),object_pairs_hook=object_pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON: '+x)))


def records(path):
    result=[]
    for line in path.read_text().splitlines():
        if line.strip():
            result.append(json.loads(line,parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x))))
    return result


def sha(path,limit=None):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        while limit is None or limit:
            block=stream.read(min(1048576,limit) if limit is not None else 1048576)
            if not block:break
            value.update(block)
            if limit is not None:limit-=len(block)
    return value.hexdigest()


def finite(value):return type(value) in (int,float) and math.isfinite(value)
def close(a,b,tolerance=1e-8):return finite(a) and finite(b) and abs(a-b)<=tolerance


def actual_environment(actual,expected):
    need({k:v for k,v in actual.items() if v is not None}==expected,
         'Actual complete MYDS environment differs from frozen production plus explicit flag/sinks')


class ByteHashes:
    def __init__(self):
        self.temporary=tempfile.TemporaryDirectory(prefix='myds-fullmodel-cpu-fnv-')
        root=Path(self.temporary.name);source=root/'fnv.c';library=root/('fnv.dylib' if sys.platform=='darwin' else 'fnv.so')
        source.write_text('#include <stdint.h>\n#include <stddef.h>\n'
          'uint64_t hash_span(uint64_t h,const unsigned char *p,size_t n){'
          'for(size_t i=0;i<n;++i){h^=p[i];h*=UINT64_C(1099511628211);}return h;}\n')
        command=['cc','-O3','-shared','-fPIC',str(source),'-o',str(library)]
        done=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        need(done.returncode==0,'CPU byte hashing helper compilation failed: '+done.stderr)
        self.library=ctypes.CDLL(str(library));self.function=self.library.hash_span
        self.function.argtypes=[ctypes.c_uint64,ctypes.c_char_p,ctypes.c_size_t]
        self.function.restype=ctypes.c_uint64;self.cache={}

    def file(self,path):
        st=path.stat();key=(str(path.resolve()),st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
        if key not in self.cache:
            digest=hashlib.sha256();value=14695981039346656037
            with path.open('rb') as stream:
                while True:
                    block=stream.read(1048576)
                    if not block:break
                    digest.update(block);value=self.function(value,block,len(block))
            self.cache[key]=dict(bytes=st.st_size,sha256=digest.hexdigest(),fnv1a64_bytes=format(value,'016x'))
        return self.cache[key]

    def bytes(self,data):return format(self.function(14695981039346656037,data,len(data)),'016x')
    def close(self):self.temporary.cleanup()


def bounded(directory,name):
    need(type(name) is str and name and Path(name).name==name,'Nonflat raw payload path')
    path=(directory/name).resolve(strict=True)
    need(path.parent==directory.resolve() and path.is_file(),'Raw payload escaped its run directory')
    return path


def file_difference(left,right,f32=False):
    first=None;different=0;total=0;float_differences=0
    with left.open('rb') as a,right.open('rb') as b:
        while True:
            x,y=a.read(1048576),b.read(1048576)
            if not x and not y:break
            if x!=y:
                for offset,(u,v) in enumerate(zip(x,y)):
                    if u!=v:
                        if first is None:first=total+offset
                        different+=1
                different+=abs(len(x)-len(y))
                if first is None:first=total+min(len(x),len(y))
                if f32:
                    need(len(x)%4==len(y)%4==0,'Misaligned raw logits')
                    float_differences+=sum(u!=v for u,v in zip(struct.iter_unpack('<I',x),struct.iter_unpack('<I',y)))
                    float_differences+=abs(len(x)-len(y))//4
            total+=max(len(x),len(y))
    return dict(equal=different==0,different_bytes=different,first_different_byte=first,
                f32_bit_mismatches=float_differences if f32 else None)


def state_payload(directory,sample,hashes):
    manifest=load(directory/'manifest.json');metadata=manifest['metadata']
    need(manifest['schema_version']==1 and manifest['scope']=='fresh_context_full_prefill_final_state' and
         manifest['run']==sample['run'] and manifest['complete'] is True,'Incomplete raw state manifest')
    need(metadata['fresh_context'] is True and metadata['byte_order']=='little' and
         metadata['layers']==40 and metadata['window_width']==512 and metadata['window_order']=='chronological_active_rows' and
         metadata['compressed_row_bytes']==1024 and metadata['index_row_bytes']==256 and
         metadata['scalar_order']==SCALAR_ORDER,'Raw state metadata contract differs')
    scalars=struct.unpack('<15Q',bounded(directory,'context-scalars.u64le').read_bytes())
    need(scalars[2:5]==(327680,256,42) and scalars[8] in (0,1) and scalars[9]==1 and scalars[14] in (0,1),
         'Raw context configuration/valid state differs')
    next_pos=scalars[10]
    need(0<next_pos<=327680 and metadata['ctx_pos']==scalars[0] and metadata['next_position']==next_pos and
         metadata['window_rows']==min(next_pos,128),'Raw active state extent differs')
    tail=list(struct.unpack('<3I',bounded(directory,'engram-tail.u32le').read_bytes()))
    need(metadata['rng_state']==format(scalars[1],'016x') and metadata['engram_position']==scalars[11] and
         metadata['engram_tail']==tail,'Raw RNG/Engram metadata differs')
    expected={'context-scalars.u64le':120,'engram-tail.u32le':12,'snapshot-positions.u32le':352,
              'prefix-tokens.i32le':scalars[6]*4,'anchor-tokens.i32le':scalars[7]*4,
              'tokens.i32le':1024,'output.bin':sample['output_bytes']}
    for layer in range(40):expected[f'layer-{layer:02d}-window-active.f32']=min(next_pos,128)*2048
    for owner in range(4):
        if owner<3:
            for kind in ('kv','score'):expected[f'owner-{owner}-previous-{kind}.f32']=2048
        count=next_pos//(1 if owner==3 else 2)
        expected[f'owner-{owner}-compressed-active.bf16']=count*1024
        expected[f'owner-{owner}-index-active.bf16']=count*256
    positions=bounded(directory,'snapshot-positions.u32le').read_bytes()
    need(len(positions)==352,'44 published/staged snapshot pairs required')
    published=[]
    for slot,(position,staged) in enumerate(struct.iter_unpack('<II',positions)):
        need(position<=next_pos and staged<=next_pos and not(position and staged),'Invalid published/staged snapshot pair')
        if position:
            published.append(dict(slot=slot,position=position))
            for layer in range(40):expected[f'snapshot-{slot:02d}-layer-{layer:02d}-window-active.f32']=min(position,128)*2048
            for owner in range(3):
                for kind in ('kv','score'):expected[f'snapshot-{slot:02d}-owner-{owner}-previous-{kind}.f32']=2048
    items=manifest['files'];names=[item['path'] for item in items]
    need(len(set(names))==len(names)==manifest['file_count'] and set(names)==set(expected),
         'Exact complete live/published state file coverage required')
    need(set(p.name for p in directory.iterdir())==set(expected)|{'manifest.json'},'Unrecorded raw state files')
    files={}
    for item in items:
        name=item['path'];path=bounded(directory,name);actual=hashes.file(path)
        need(actual['bytes']==item['bytes']==expected[name] and actual['fnv1a64_bytes']==item['fnv1a64_bytes'],
             'Raw state extent/FNV differs: '+str(path))
        files[name]=actual
    token_bytes=bounded(directory,'tokens.i32le').read_bytes()
    need(len(sample['token_ids'])==256 and token_bytes==struct.pack('<256i',*sample['token_ids']),
         'Complete raw token ID bytes differ')
    need(files['output.bin']['fnv1a64_bytes']==sample['output_bytes_fnv1a64'],'Raw original text bytes differ from callback FNV')
    if 'token_ids_fnv1a64' in sample:need(hashes.bytes(token_bytes)==sample['token_ids_fnv1a64'],'Native token FNV differs')
    need(len(manifest['limitations'])==2 and any('Historical snapshot Engram tails' in x for x in manifest['limitations']) and
         any('allocator padding' in x for x in manifest['limitations']),'Export boundary/limitations missing')
    return dict(metadata=metadata,files=files,published_snapshots=published,limitations=manifest['limitations'])


def verify_sample(sample,directory,logits,setting,prompt_tokens,hashes):
    need(sample['generation_ok'] is True and sample['sample_valid'] is True and sample['fresh_context'] is True and
         sample['output_tokens']==256 and sample['prompt_tokens']==prompt_tokens and sample['cached_tokens']==0 and
         sample['stop_reason']=='length' and sample['warmup']==(sample['run']==0),'Incomplete fresh-context sample')
    for key in ('decode_s','prefill_s','wall_s','first_callback_s','last_callback_s','decode_ms_per_token'):
        need(finite(sample[key]) and sample[key]>0,'Invalid raw timing: '+key)
    need(sample['decode_intervals']==255 and sample['prefill_tokens']==prompt_tokens-1 and
         close(sample['decode_s'],sample['last_callback_s']-sample['first_callback_s'],3e-9) and
         close(sample['decode_ms_per_token'],sample['decode_s']*1000/255) and
         sample['last_callback_s']<=sample['wall_s']+1e-8,'Callback/prefill timing arithmetic differs')
    intervals=sample['callback_intervals_ms']
    need(len(intervals)==255 and all(finite(v) and v>=0 for v in intervals) and
         close(sum(intervals),sample['decode_s']*1000,1e-5),'Complete callback intervals disagree with decode interval')
    raw=logits.read_bytes();actual=hashes.file(logits)
    need(len(raw)==129280*4 and sample['final_logits']['elements']==129280 and sample['final_logits']['finite'] is True and
         all(math.isfinite(v[0]) for v in struct.iter_unpack('<f',raw)) and
         actual['fnv1a64_bytes']==sample['final_logits']['fnv1a64_bytes'],'Full finite raw vocabulary logits/FNV differ')
    candidate=sample['gpu_output_a_fixed_wrapper']
    need(candidate['requests']==candidate['encoded'] and
         candidate['failures']==candidate['fallbacks']==candidate['ineligible']==0 and
         candidate['decode_requests']==candidate['decode_encoded'] and
         candidate['decode_failures']==candidate['decode_fallbacks']==candidate['decode_ineligible']==0 and
         (candidate['encoded']>0)==(candidate['decode_encoded']>0)==bool(setting) and
         (candidate['nr0'],candidate['nsg'],candidate['fc600'])==((2,4,4) if setting else (0,0,0)),
         'Actual fixed wrapper adoption/geometry differs')
    providers={}
    for key in ('gpu_attention_packed_register','gpu_attention_rope_fuse'):
        value=sample[key]
        need(value['requests']==value['encoded']>0 and value['failures']==0 and
             value['decode_requests']==value['decode_encoded']>0 and value['decode_failures']==0,
             'Actual production provider failed: '+key);providers[key]=value
    rf=providers['gpu_attention_rope_fuse']
    if setting:need(candidate['encoded']==rf['encoded'] and candidate['decode_encoded']==rf['decode_encoded'],
                    'Candidate did not cover every ordinary RF output projection')
    state=state_payload(directory,sample,hashes)
    return dict(run=sample['run'],warmup=sample['warmup'],decode_s=sample['decode_s'],prefill_s=sample['prefill_s'],
        wall_s=sample['wall_s'],callback_interval_sum_ms=sum(intervals),logits=actual,state=state,
        candidate=candidate,production_providers=providers,original_text_path=str(directory/'output.bin'),
        original_logits_path=str(logits),state_path=str(directory))


def arm(root,step,plan,environment,hashes):
    label=step['name'];phase,workload,name=label.split('-');setting=int(name in ('on','B'))
    repeats=1 if phase=='correctness' else 3;data=records(root/(label+'.stdout'))
    metadata=[r for r in data if r.get('kind')=='metadata'];loaded=[r for r in data if r.get('kind')=='load']
    samples=[r for r in data if r.get('kind')=='sample']
    need(len(metadata)==len(loaded)==1 and len(data)==2+len(samples),'Unexpected actual harness JSONL records')
    meta,library=metadata[0],loaded[0];directory=root/label
    env=load(directory/'environment.json');expected=dict(environment)
    expected.update({FLAG:str(setting),'MYDS_BENCH_LOGITS_DIR':str(directory/'logits'),'MYDS_BENCH_STATE_DIR':str(directory/'state')})
    need(env==expected,'Requested per-arm environment differs');actual_environment(meta['environment'],expected)
    need(all(meta.get(k)==v for k,v in dict(protocol_version=3,repeats=repeats,warmup_runs=1,max_new_tokens=256,
        max_seq_len=327680,temperature=0,top_p=1,top_k=0,min_p=0,repetition_penalty=1,seed=42,
        clock='CLOCK_MONOTONIC',transport='direct_byte_callback',decode_basis='first_to_last_callback/(tokens-1)',
        prompt_renderer='parse_chat_prompt').items()),'Actual harness protocol differs')
    argv=step['argv'];need(len(argv)==6 and argv[-3:]==[str(repeats),'256','327680'],'Actual harness argv differs')
    stage=Path(argv[0]).parent.parent
    need(Path(argv[0]).name=='bench_output_a_fullmodel' and Path(meta['request_file'])==Path(argv[2]) and
         Path(meta['model'])==Path(argv[1]),'Model/request/harness argv binding differs')
    need(Path(library['actual_metallib_path']).resolve(strict=True)==(stage/'bin/myds.metallib').resolve(strict=True) and
         library['fresh_context_per_sample'] is True and library['cache_configuration_preserved'] is True and
         library['output_a_fixed_wrapper_mode']==setting and
         library['output_a_fixed_wrapper_selected_symbol']==(FIXED if setting else None) and
         library['output_a_fixed_wrapper_api_available']==(name!='gold'),'Actual provider/library identity differs')
    need([r['run'] for r in samples]==list(range(len(samples))) and 0<len(samples)<=repeats+1,'Actual sample order/coverage differs')
    reviewed=[]
    for sample in samples:
        reviewed.append(verify_sample(sample,directory/'state'/f"run-{sample['run']}",
            directory/'logits'/f"run-{sample['run']}.f32",setting,plan['workloads'][workload]['expected_prompt_tokens'],hashes))
    return dict(label=label,phase=phase,workload=workload,name=name,setting=setting,stage=str(stage),
                returncode=step['returncode'],complete=len(samples)==repeats+1,metadata=meta,load=library,samples=reviewed)


def compare_sample(gold,candidate):
    differences=[]
    if gold['state']['metadata']!=candidate['state']['metadata']:differences.append(dict(kind='state_metadata'))
    for name in sorted(set(gold['state']['files'])|set(candidate['state']['files'])):
        a,b=Path(gold['state_path'])/name,Path(candidate['state_path'])/name
        if not a.is_file() or not b.is_file():differences.append(dict(kind='state_file_coverage',path=name));continue
        delta=file_difference(a,b)
        if not delta['equal']:differences.append(dict(kind='state_original_bytes',path=name,**delta))
    delta=file_difference(Path(gold['original_logits_path']),Path(candidate['original_logits_path']),True)
    if not delta['equal']:differences.append(dict(kind='full_vocabulary_f32_bits',**delta))
    return differences


def protocol(plan):
    need(plan['schema_version']==1 and plan['baseline_commit']=='868d94e9e3168aeadac478e75d096d992b6bc92e' and
         plan['candidate_flag']==FLAG and plan['default_off'] is True and plan['production_adoption'] is False and
         plan['max_seq_len']==327680 and plan['max_new_tokens']==256 and plan['temperature']==0 and
         plan['top_p']==1 and plan['seed']==42 and plan['fresh_context_per_sample'] is True,'Frozen plan scope differs')
    correctness=plan['correctness'];performance=plan['performance']
    need(correctness['order']==['gold','off','on'] and correctness['warmup']==1 and correctness['formal_samples']==1 and
         correctness['before_performance'] is True,'Frozen correctness order differs')
    need(performance['order']==['A1','B','A2'] and performance['flags']==dict(A1=0,B=1,A2=0) and
         performance['warmup']==1 and performance['formal_samples']==3 and performance['repeat_limit']==1 and
         performance['maximum_absolute_control_drift_percent']==2 and
         performance['minimum_conservative_decode_time_saved_percent_strict']==2 and
         performance['maximum_conservative_prefill_regression_percent']==2 and performance['all_workloads_required'] is True,
         'Frozen fullmodel performance gate differs')


def performance_group(workload,group):
    decode=[statistics.median(s['decode_s'] for s in a['samples'][1:]) for a in group]
    prefill=[statistics.median(s['prefill_s'] for s in a['samples'][1:]) for a in group]
    drift=(decode[2]/decode[0]-1)*100;saved=(1-decode[1]/min(decode[0],decode[2]))*100
    pf_drift=(prefill[2]/prefill[0]-1)*100;pf_regression=(prefill[1]/min(prefill[0],prefill[2])-1)*100
    return dict(workload=workload,decode_seconds=dict(zip(('A1','B','A2'),decode)),
        prefill_seconds=dict(zip(('A1','B','A2'),prefill)),control_drift_percent=drift,
        conservative_decode_time_saved_percent=saved,prefill_control_drift_percent=pf_drift,
        conservative_prefill_regression_percent=pf_regression,
        gate_pass=abs(drift)<=2 and abs(pf_drift)<=2 and saved>2 and pf_regression<=2)


def archive_bindings(root,source_archives,artifact_archives,arms):
    bindings=load(root/'stage-inputs.json');stages=sorted({Path(v['stage']) for v in arms},key=str)
    need(len(stages)==2,'Exactly independent gold and candidate stages required')
    all_archives=[];index={}
    for kind,archives in (('source',source_archives),('artifacts',artifact_archives)):
        for path in archives:
            count=0
            with tarfile.open(path,'r:*') as archive:
                for member in archive:
                    if member.isdir():continue
                    need(not member.issym() and not member.isdev() and '..' not in Path(member.name).parts,'Unsafe archive entry')
                    stream=archive.extractfile(member);need(stream is not None,'Unreadable archived member')
                    digest=hashlib.sha256();size=0
                    while True:
                        block=stream.read(1048576)
                        if not block:break
                        digest.update(block);size+=len(block)
                    need(member.islnk() or size==member.size,'Archived member extent differs')
                    index.setdefault((kind,digest.hexdigest()),[]).append(member.name);count+=1
            all_archives.append(dict(kind=kind,path=str(path),sha256=sha(path),members_read=count))
    archived=[];external=[]
    for name,digest in bindings.items():
        path=Path(name);matches=[stage for stage in stages if stage in path.parents]
        if matches:
            need(len(matches)==1,'Ambiguous stage source path');relative=str(path.relative_to(matches[0]));kind='artifacts' if relative.startswith('bin/') else 'source'
            candidates=index.get((kind,digest),[])
            need(any(n==relative or n.endswith('/'+relative) for n in candidates),'Frozen source/artifact not archived: '+name)
            archived.append(dict(path=name,sha256=digest,kind=kind))
        else:
            # Frozen approval, leaf reports, and plan are evidence, not executable
            # source. Keep their original paths visible rather than pretending
            # they were members of a source archive.
            need(path.is_file() and sha(path)==digest,'Frozen external evidence differs: '+name)
            external.append(dict(path=name,sha256=digest))
    for record in arms:
        stage=Path(record['stage'])
        required=['Makefile','tools/bench_output_a_fullmodel.c','src/myds_metal.m','src/myds_v41_gpu.c',
                  'bin/bench_output_a_fullmodel','bin/myds.metallib','bin/myds']
        if record['name']!='gold':required+=['tools/run_output_a_fullmodel_window.py','tools/run_iq2_signed4_window.py',
                                           'metal/probe_output_a_fixed.metal']
        for name in required:
            need(str(stage/name) in bindings,'Critical executed source/artifact is not frozen: '+str(stage/name))
        if record['name']=='gold':
            need(bindings[str(stage/'bin/myds')]==PRODUCTION_BINARY and bindings[str(stage/'bin/myds.metallib')]==PRODUCTION_LIBRARY,
                 'Independent gold does not reproduce exact production artifacts')
    return dict(archives=all_archives,archived_stage_inputs=archived,external_evidence_inputs=external,
                all_frozen_stage_inputs_bound=True)


def window_identity(root):
    before,restored,summary=[load(root/name) for name in ('before.json','restored.json','summary.json')]
    need(all(k in before and before[k]==restored[k] for k in IDENTITY_KEYS) and before['health']==restored['health']=={'status':'ok'},
         'Original production identity, config, nice, argv, environment or health differs after restore')
    need(before['binary_sha256']==PRODUCTION_BINARY and before['metallib_sha256']==PRODUCTION_LIBRARY,
         'Actual original production artifact identity differs')
    need(summary['restoration']['restored'] is True and summary['restoration']['pid']==restored['pid'] and
         summary['restoration']['exact_argv_myds_artifacts_config'] is True and summary['production_adoption'] is False and
         summary['fullmodel_run'] is True,
         'Production restoration/adoption summary differs')
    need(sha(root/'production-artifacts/myds')==PRODUCTION_BINARY and
         sha(root/'production-artifacts/myds.metallib')==PRODUCTION_LIBRARY and
         sha(root/'original-myds.conf')==before['config_sha256'] and
         load(root/'effective-env.json')==before['effective_myds_env'],'Original production recovery payload differs')
    events=records(root/'events.jsonl')
    need(events and events[-1]['kind']=='window_complete' and events[-1]['status']==summary['status'] and
         not any(r['kind'] in ('monitor_failure','restoration_failure') for r in events),'Competition/restoration failure event')
    env=before['effective_myds_env']
    need(env.get('MYDS_METAL_CONCURRENT')=='1' and env.get('MYDS_PREFIX_CACHE')=='1' and env.get('MYDS_PLD')=='1',
         'Production concurrent/cache/PLD mode differs')
    need(not any(k.startswith(('MYDS_TRACE','MYDS_SKIP_','MYDS_PROFILE','MYDS_KPROF','MYDS_GAP_PROF','MYDS_CAPTURE_')) for k in env),
         'Diagnostic/skip selectors present in original production environment')
    return before,restored,summary


def approval_inputs(root):
    bindings=load(root/'stage-inputs.json');approvals=[];plans=[]
    for name,digest in bindings.items():
        if not name.endswith('.json'):continue
        path=Path(name)
        if not path.is_file():continue
        value=load(path)
        if type(value) is not dict:continue
        if 'component_and_frozen_two_window_replication_gate_pass' in value:approvals.append((path,digest,value))
        if value.get('experiment')=='Output_A fixed wrapper full-model qualification':plans.append((path,digest,value))
    need(len(approvals)==len(plans)==1,'Unique frozen independent component approval and fullmodel plan required')
    path,digest,approved=approvals[0]
    need(sha(path)==digest and approved['component_and_frozen_two_window_replication_gate_pass'] is True and
         approved['every_window_every_layer_every_control_pass'] is True and
         approved['decision']=='eligible_for_separate_fullmodel_gate','Frozen independent component approval failed')
    groups=approved['all_12_groups']
    need(len(groups)==12 and {(g['window'],g['layer'],g['control_kind']) for g in groups}==
         {(w,layer,kind) for w in (1,2) for layer in (0,19,39) for kind in ('production','probe')} and
         all(g['performance_gate_pass'] is True and abs(g['control_drift_percent'])<=2 and
             g['conservative_time_saved_percent']>2 for g in groups),'Frozen two-window twelve-group prerequisite failed')
    leaves=[]
    for item in approved['reports']+[dict(path=approved['plan_path'],sha256=approved['plan_sha256'])]:
        leaf=(path.parent/item['path']).resolve(strict=True)
        need(path.parent.resolve() in leaf.parents and bindings.get(str(leaf))==item['sha256'] and sha(leaf)==item['sha256'],
             'Frozen component approval leaf report/plan binding differs')
        leaves.append(dict(path=str(leaf),sha256=item['sha256']))
    fullplan,plan_sha,_=plans[0]
    need(sha(fullplan)==plan_sha and load(fullplan)==load(root/'protocol.json'),
         'Executed protocol JSON differs from frozen fullmodel plan')
    return dict(approval_path=str(path),approval_sha256=digest,leaf_reports_and_plan=leaves,
                fullmodel_plan_path=str(fullplan),fullmodel_plan_sha256=plan_sha)


def review(root,model,source_archives,artifact_archives):
    plan=load(root/'protocol.json');protocol(plan);before,restored,summary=window_identity(root)
    prerequisites=approval_inputs(root)
    need(summary['model']['bytes']==model.stat().st_size and summary['model_header_sha256']==sha(model,1048576),
         'Actual model extent/prefix identity differs')
    expected=[f'correctness-{workload}-{name}' for workload in ('short','business') for name in ('gold','off','on')]
    expected +=[f'performance-{workload}-{name}' for workload in ('short','business') for name in ('A1','B','A2')]
    steps=load(root/'steps.json');names=[s['name'] for s in steps]
    need(names==expected[:len(names)] and 0<len(names)<=12,'Strict gold/off/on before A1/B/A2 sequence differs')
    hashes=ByteHashes()
    try:
        arms=[arm(root,step,plan,before['effective_myds_env'],hashes) for step in steps]
        need(all(Path(a['metadata']['model']).resolve()==model.resolve() for a in arms),'Actual harness model path differs')
        bindings=load(root/'stage-inputs.json')
        requests={}
        for workload in ('short','business'):
            paths={Path(a['metadata']['request_file']) for a in arms if a['workload']==workload}
            if not paths:continue
            need(len(paths)==1,'Different actual requests within workload');path=paths.pop()
            need(bindings.get(str(path))==plan['workloads'][workload]['sha256'] and path.is_file() and
                 sha(path)==plan['workloads'][workload]['sha256'],'Frozen workload request SHA differs')
            body=path.read_bytes();requests[workload]=dict(bytes=len(body),sha256=sha(path),fnv1a64=hashes.bytes(body))
            for a in arms:
                if a['workload']==workload:
                    need(a['metadata']['request_bytes']==len(body) and a['metadata']['request_fnv1a64']==requests[workload]['fnv1a64'],
                         'Actual harness parsed different request bytes')
        comparisons=[];oracles={};oracle_loads={};oracle_metadata={}
        for a in arms:
            if a['name']=='gold':
                oracles[a['workload']]=a['samples'][0];oracle_loads[a['workload']]=a['load'];oracle_metadata[a['workload']]=a['metadata']
            need(a['workload'] in oracles,'Missing earlier independent production oracle')
            need(all(a['metadata'][k]==oracle_metadata[a['workload']][k] for k in
                     ('prompt_fnv1a64','prompt_bytes','request_fnv1a64','request_bytes','dspark_model')),
                 'Actual rendered prompt or auxiliary model differs from independent gold')
            modes={k:v for k,v in oracle_loads[a['workload']].items() if k.endswith('_mode') and k!='output_a_fixed_wrapper_mode'}
            need(all(a['load'].get(k)==v for k,v in modes.items()),'Other production provider modes differ from independent gold')
            for sample in a['samples']:
                differences=compare_sample(oracles[a['workload']],sample)
                comparisons.append(dict(arm=a['label'],run=sample['run'],full_original_bytes_equal=not differences,differences=differences))
        correct_arms=[a for a in arms if a['phase']=='correctness']
        correctness=(len(correct_arms)==6 and all(a['complete'] and a['returncode']==0 for a in correct_arms) and
            all(c['full_original_bytes_equal'] for c in comparisons if c['arm'].startswith('correctness-')))
        performance_arms=[a for a in arms if a['phase']=='performance']
        need(correctness or not performance_arms,'Performance was executed before all independent correctness gates passed')
        recomputed=[]
        if performance_arms:
            need(len(performance_arms)==6 and all(a['complete'] and a['returncode']==0 for a in performance_arms),
                 'Incomplete performance arm coverage')
            need(all(c['full_original_bytes_equal'] for c in comparisons),'Performance raw bytes differ from independent correctness gold')
            for workload in ('short','business'):
                group=[next(a for a in performance_arms if a['workload']==workload and a['name']==name) for name in ('A1','B','A2')]
                item=performance_group(workload,group)
                recomputed.append(item)
                published=[p for p in summary['performance'] if p['workload']==workload];need(len(published)==1,'Missing workload summary')
                for key,value in item.items():
                    if key in ('workload','gate_pass'):need(published[0][key]==value,'Published performance decision differs')
                    elif type(value) is dict:need(all(close(published[0][key][n],v) for n,v in value.items()),'Published median differs')
                    else:need(close(published[0][key],value),'Published performance arithmetic differs')
        gate=bool(recomputed) and all(x['gate_pass'] for x in recomputed)
        if correctness:
            need(summary['correctness_pass'] is True,'Correctness summary differs')
        if recomputed:
            need(summary['performance_gate_pass'] is gate and summary['status']==('fullmodel_passed_review_required' if gate else 'fullmodel_rejected_performance'),
                 'Runner fullmodel decision differs from independently recomputed gates')
        else:need(summary['status']=='failed','A partial/correctness-failed window cannot claim fullmodel performance')
        archive=archive_bindings(root,source_archives,artifact_archives,arms)
        return dict(schema_version=1,review_kind='independent_CPU_fullmodel_original_bytes_and_frozen_gates',
            reader_sha256=sha(Path(__file__)),root=str(root),protocol_sha256=sha(root/'protocol.json'),
            gpu_work_submitted_by_reader=False,production_adoption=False,artifact_review_passed=True,
            fullmodel_correctness_pass=correctness,fullmodel_performance_measured=bool(recomputed),
            fullmodel_performance_gate_pass=gate,decision='eligible_for_production_adoption_review' if gate else
                ('retain_default_off_fullmodel_performance_gate_failed' if recomputed else 'retain_default_off_correctness_gate_failed_or_incomplete'),
            original_pid=before['pid'],restored_pid=restored['pid'],restoration_exact_including_nice=True,
            requests=requests,arms=arms,full_original_byte_comparisons=comparisons,
            raw_samples_verified=sum(len(a['samples']) for a in arms),
            state_payload_files_verified=sum(len(s['state']['files']) for a in arms for s in a['samples']),
            recomputed_performance=recomputed,prerequisite_frozen_component_approval=prerequisites,**archive,
            limits=['Fresh-context native harness excludes HTTP queue/transport and context creation.',
                    'Active live state and published snapshots only; historical snapshot Engram tails, persistent disk-cache payload, allocator padding, inactive capacity and scratch are outside export.',
                    'A passing raw review does not itself enable the candidate or modify production.'])
    finally:hashes.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','model','out'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--source-archive',type=Path,action='append',required=True)
    parser.add_argument('--artifacts-archive',type=Path,action='append',required=True)
    a=parser.parse_args()
    try:result=review(a.root.resolve(),a.model.resolve(),a.source_archive,a.artifacts_archive)
    except Exception as exc:
        result=dict(schema_version=1,artifact_review_passed=False,production_adoption=False,
                    gpu_work_submitted_by_reader=False,error=repr(exc),reader_sha256=sha(Path(__file__)))
    with a.out.open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True);stream.write('\n')
    print(json.dumps({k:result[k] for k in ('artifact_review_passed','decision','error') if k in result},sort_keys=True))
    return 0 if result['artifact_review_passed'] else 1


if __name__=='__main__':sys.exit(main())
