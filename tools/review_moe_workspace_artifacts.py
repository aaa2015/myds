#!/usr/bin/env python3
"""Independent CPU review of real prefill MoE workspace component evidence.

No benchmark, model loader, Metal API or service operation is called. The tiny
temporary C helper only checks raw F32 bit patterns, including poison/NaN/Inf.
The retained GGUF is required: complete weight intervals remain model-backed.
"""
import argparse
import ctypes
import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import struct
import subprocess
import sys
import tarfile
import tempfile

LAYERS=(0,19,39)
BOUNDARIES=(1,17,31,32,33)
IDENTITY_KEYS=('argv_raw_hex','myds_env_raw_hex','cwd','log','nice','binary','metallib',
               'binary_sha256','metallib_sha256','config','config_sha256','effective_myds_env')


def need(value,message):
    if not value:raise ValueError(message)


def load(path):
    def pairs(values):
        result={}
        for key,value in values:
            need(key not in result,'Duplicate JSON key: '+key);result[key]=value
        return result
    return json.loads(path.read_text(),object_pairs_hook=pairs,parse_constant=lambda v:(_ for _ in ()).throw(ValueError(v)))


NATIVE_STDOUT_LOG=re.compile(r'^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\] \[(DEBUG|INFO)\] ([^\x00-\x1f\x7f]+)$')


def rows(path,allow_native_log=False,log_evidence=None):
    # myds_log.c sends only DEBUG/INFO to stdout, with this exact timestamp
    # and no ANSI escapes in the frozen non-tty child environment. Retain the
    # physical original log bytes rather than silently dropping arbitrary text.
    data=[];logs=[];levels={}
    for number,raw in enumerate(path.read_bytes().splitlines(keepends=True),1):
        line=raw.decode('utf-8').rstrip('\r\n')
        if not line.strip():continue
        if line.startswith('{'):
            value=json.loads(line,parse_constant=lambda v:(_ for _ in ()).throw(ValueError(v)))
            need(type(value) is dict,'JSONL object required');data.append(value);continue
        match=NATIVE_STDOUT_LOG.fullmatch(line) if allow_native_log else None
        need(match is not None,f'Unrecognized non-JSON original line {path.name}:{number}')
        datetime.datetime.strptime(match[1],'%Y-%m-%d %H:%M:%S.%f')
        logs.append(raw);levels[match[2]]=levels.get(match[2],0)+1
    if log_evidence is not None:
        log_evidence.append(dict(path=str(path),stdout_sha256=sha(path),native_log_lines=len(logs),
            native_log_bytes=sum(map(len,logs)),native_log_raw_sha256=hashlib.sha256(b''.join(logs)).hexdigest(),
            levels=levels,non_json_policy='exact_uncolored_myds_log_DEBUG_INFO_stdout_grammar'))
    return data


def sha(path,offset=0,size=None):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        stream.seek(offset)
        while size is None or size:
            block=stream.read(min(size,1048576) if size is not None else 1048576)
            if not block:break
            value.update(block)
            if size is not None:size-=len(block)
    need(size in (None,0),'Short hashed original interval');return value.hexdigest()


def bounded(base,name):
    need(type(name) is str and Path(name).name==name and name not in ('','.','..'),'Flat relative artifact name required')
    p=base/name;need(not p.is_symlink() and p.is_file() and p.resolve().parent==base.resolve(),'Original bounded payload required')
    return p


def close(a,b,tolerance=1e-7):return type(a) in (int,float) and math.isfinite(a) and abs(a-b)<=tolerance
def yes(value):return value is True or type(value) is int and value==1


class FloatReader:
    def __init__(self):
        self.temp=tempfile.TemporaryDirectory(prefix='moe-workspace-reader-CPU-');root=Path(self.temp.name)
        source=root/'bits.c';library=root/('bits.dylib' if sys.platform=='darwin' else 'bits.so')
        source.write_text('#include <stdint.h>\n#include <stddef.h>\n#include <string.h>\n'
          'int f32ok(const unsigned char*p,size_t n){if(n%4)return 0;for(size_t i=0;i<n;i+=4){uint32_t v;'
          'memcpy(&v,p+i,4);if((v&0x7f800000u)==0x7f800000u||v==0xa5a5a5a5u)return 0;}return 1;}\n')
        run=subprocess.run(['cc','-O3','-shared','-fPIC',str(source),'-o',str(library)],capture_output=True,text=True)
        need(run.returncode==0,'CPU F32 checker compiler failed: '+run.stderr)
        self.library=ctypes.CDLL(str(library));self.check=self.library.f32ok
        self.check.argtypes=[ctypes.c_char_p,ctypes.c_size_t];self.check.restype=ctypes.c_int
    def compare(self,left,right,size,finite=True):
        need(left.stat().st_size==right.stat().st_size==size,'Complete original output extent differs')
        first=None;different=0;offset=0
        with left.open('rb') as a,right.open('rb') as b:
            while True:
                x,y=a.read(1048576),b.read(1048576)
                if not x:break
                if finite:need(self.check(x,len(x)) and self.check(y,len(y)),'Original output nonfinite/poison')
                if x!=y:
                    for i,(u,v) in enumerate(zip(struct.iter_unpack('<I',x),struct.iter_unpack('<I',y))):
                        if u!=v:
                            if first is None:first=offset//4+i
                            different+=1
                offset+=len(x)
        return dict(full_f32_bit_exact=different==0,mismatches=different,first_mismatch=first,elements=size//4,bytes=size)
    def close(self):self.temp.cleanup()


def manifest(directory,phase):
    m=load(directory/'artifact-manifest.json')
    need(m['schema_version']==1 and m['probe']=='real_prefill_moe_workspace' and m['status']=='complete' and
         m['phase']==phase and m['full_f32_bit_exact'] is True and m['weight_payload_exported'] is False and
         m['mode']=='concurrent' and m['samples']==9 and yes(m['repeated_real_fixture_cache_reuse']),
         'Complete real component phase required')
    files={}
    for item in m['files']:
        name=item['path'];need(name not in files,'Duplicate logical artifact')
        p=bounded(directory,name);need(p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],'Raw original SHA/extent differs')
        if name.endswith('.guards'):need(p.read_bytes()==b'\xa5'*512,'Original guard bytes differ')
        files[name]=item
    return m,files


def payload(base,descriptor,size):
    p=bounded(base,descriptor['path']);need(p.stat().st_size==descriptor['bytes']==size and sha(p)==descriptor['sha256'],
       'Captured original payload SHA/extent differs');return p


def case_key(c):return c['layer'],c['source_case'],c['rows'],bool(c['derived_real_prefix'])
def stem(c):return f"layer-{c['layer']}-case-{c['source_case']}-rows-{c['rows']}"


def complete_weights(model,capture,cm,pm):
    names={f'blk.{layer}.ffn_{kind}_exps.weight' for layer in LAYERS for kind in ('gate','up','down')}
    descriptors,start=gguf_descriptors(model,names);result=[]
    need(cm['model_backed_tensors']==pm['model_backed_tensors'] and len(cm['model_backed_tensors'])==9,
         'Same nine complete weight identities required')
    identities={t['name']:t for t in cm['model_backed_tensors']};need(set(identities)==names,'Exact weight tensor name coverage')
    declared={t['name']:t for t in capture['tensors']};need(set(declared)==names,'Observer complete weight descriptors')
    for name in sorted(names):
        d=descriptors[name];t=identities[name];down='down_exps' in name;dims=[2304,5120,384] if down else [5120,2304,384]
        count=dims[0]//256*(84 if down else 66)*dims[1]*dims[2];offset=start+d['offset']
        need(d['type']==(10 if down else 16) and d['dimensions']==dims and t['type']==d['type'] and t['dims']==dims and
             t['offset']==offset and t['bytes']==count and t['model_backed'] is True and t['payload_exported'] is False and
             yes(t['complete_interval_sha_checked']),'Actual complete IQ2/Q2_K interval contract differs')
        need(declared[name]['offset']==offset and declared[name]['bytes']==count and declared[name]['type']==d['type'] and
             declared[name]['dims']==dims and declared[name]['model_backed'] is True,'Observer/model tensor descriptor differs')
        need(offset+count<=model.stat().st_size and sha(model,offset,count)==t['sha256'],'Full retained GGUF interval SHA differs')
        result.append(dict(name=name,offset=offset,bytes=count,sha256=t['sha256'],payload_exported=False,model_backed=True))
    return result


def numeric_capture(root,capture_path,model,cm,pm,cfiles,pfiles,reader):
    capture=load(capture_path);base=capture_path.parent
    need(capture['schema_version']==1 and capture['capture_kind']=='myds_prefill_moe_workspace' and
         capture['model_bytes']==model.stat().st_size and capture['model_header_bytes']==1048576 and
         capture['model_header_sha256']==sha(model,0,1048576),'Real model prefix/extent differs')
    source=capture['source'];need(source['kind']=='actual_native_prefill_routed_moe_batch' and source['actual_tokens']==16401 and
         source['trace_on'] is False and source['performance_from_capture'] is False,'Native real prefill source required')
    need(cm['capture_json_sha256']==pm['capture_json_sha256']==sha(capture_path) and cm['capture_source']==pm['capture_source']==source,
         'Capture/source binding differs')
    observed=capture['cases'];need({c['layer'] for c in observed}==set(LAYERS) and len({c['case'] for c in observed})==len(observed),
         'All actual three-layer batches with unique observer cases required')
    layer0=[c for c in observed if c['layer']==0]
    need(any(c['rows']==16384 and c['actual_mm'] is True for c in layer0) and
         any(c['rows']<32 and c['actual_mm'] is False for c in layer0) and sum(c['rows'] for c in layer0)==16401,
         'Actual layer0 full B MM and complete short-tail coverage required')
    position=0
    for c in layer0:need(c['absolute_start']==position,'Actual layer0 batch ranges overlap or have a hole');position+=c['rows']
    need(sha(base/'capture-records.jsonl')==source['records_sha256'] and rows(base/'capture-records.jsonl')==observed,
         'Observer original record bytes differ')
    summary=load(base/'capture-observer-summary.json')
    need(summary['status']=='complete' and summary['cases']==len(observed) and
         summary['mm_layers_mask']|summary['nonmm_layers_mask']==7 and summary['real_prefill_tokens']==16401,
         'Observer completion/layer coverage differs')
    actual_keys=set();observed_by_key={};boundary=None;outcomes=[]
    for c in observed:
        layer,index,n=c['layer'],c['case'],c['rows'];need(c['capacity']==2048 and c['moe_batch_capacity']==16384 and 0<n<=16384 and
           c['dim']==5120 and c['expert_dim']==2304 and c['routed_experts']==384 and c['top_k']==6 and
           (c['gate_type'],c['up_type'],c['down_type'])==(16,16,10) and c['boundary']=='actual_prefill_routed_moe_batch' and
           c['trace_on'] is False and c['observer_commit_wait'] is True and c['observer_batch_restored'] is True and
           c['performance_from_capture'] is False,'Actual batch shape/observer scope differs')
        x=payload(base,c['input_f32'],n*20480);ids=payload(base,c['ids_i32'],n*24);w=payload(base,c['routeweights_f32'],n*24)
        out=payload(base,c['native_output_f32'],n*20480);key=(layer,index,n,False);actual_keys.add(key);observed_by_key[key]=c
        with x.open('rb') as f:
            for block in iter(lambda:f.read(1048576),b''):need(reader.check(block,len(block)),'Captured full input nonfinite/poison')
        raw_ids=ids.read_bytes();raw_w=w.read_bytes()
        for row in range(n):
            values=struct.unpack_from('<6i',raw_ids,row*24);need(len(set(values))==6 and all(0<=i<384 for i in values),'Actual top6 IDs invalid')
        need(all(math.isfinite(v[0]) and v[0]>=0 for v in struct.iter_unpack('<f',raw_w)),'Actual route weights invalid')
        if boundary is None and layer==0 and n>=33:boundary=(c,x,ids,w)
        reference=bounded(root/'correctness',f'layer-{layer}-case-{index}-rows-{n}-native.f32')
        delta=reader.compare(out,reference,n*20480);need(delta['full_f32_bit_exact'],'Native replay differs from observed real production output')
        outcomes.append(dict(layer=layer,source_case=index,rows=n,derived_real_prefix=False,observed_native_vs_component_native=delta))
    need(boundary is not None,'Actual sufficient layer0 boundary input required')
    wanted=set(actual_keys)
    bc,bx,bi,bw=boundary
    for n in BOUNDARIES:wanted.add((0,bc['case'],n,True))
    by_key={case_key(c):c for c in cm['cases']};need(len(by_key)==len(cm['cases']) and set(by_key)==wanted,'Exact actual plus five derived correctness cases')
    need({case_key(c) for c in pm['cases']}==actual_keys and len(pm['cases'])==len(actual_keys),'Timing must use only all actual captured cases')
    expected_cfiles=set();expected_pfiles=set()
    for key,c in by_key.items():
        layer,index,n,derived=key;need(c['capacity']==2048 and c['moe_batch']==16384,'Frozen full workspace capacity')
        source_record=bc if derived else observed_by_key[key]
        for field,file_key,stride in (('input_sha256','input_f32',20480),('ids_sha256','ids_i32',24),('routeweights_sha256','routeweights_f32',24)):
            path=base/source_record[file_key]['path'];need(c[field]==sha(path,0,n*stride),'Complete original/derived-real input bytes differ')
        name=stem(c);native=bounded(root/'correctness',name+'-native.f32');compact=bounded(root/'correctness',name+'-compact.f32')
        delta=reader.compare(native,compact,n*20480);need(delta['full_f32_bit_exact'],'Native/compact full output mismatch')
        for suffix in ('-native.f32','-compact.f32','-native.guards','-compact.guards','-input.guards','-ids.guards','-routeweights.guards'):expected_cfiles.add(name+suffix)
        if not derived:
            for suffix in ('-input.guards','-ids.guards','-routeweights.guards'):expected_pfiles.add(name+suffix)
        outcomes.append(dict(layer=layer,source_case=index,rows=n,derived_real_prefix=derived,native_vs_compact=delta))
    need(set(cfiles)==expected_cfiles and set(pfiles)==expected_pfiles,'Exact original output/input/guard artifact coverage')
    return capture,outcomes,observed_by_key


def counter_contract(workspace,actual,compact):
    need(workspace['capacity']==2048 and workspace['moe_batch']==16384 and bool(workspace['compact'])==compact and
         workspace['requests']==len(actual) and workspace['mm_encoded']==sum(c['actual_mm'] for c in actual) and
         workspace['mv_encoded']==sum(not c['actual_mm'] for c in actual) and workspace['failures']==0,'Actual complete wrapper adoption differs')
    if compact:
        need(workspace['legacy_gate_bytes']==workspace['legacy_up_bytes']==workspace['legacy_mid_bytes']==0,'Compact retained large legacy pool')
        s=workspace['scratch'];need(s['requests']==len(actual) and s['mm_encoded']==workspace['mm_encoded'] and
           s['mv_encoded']==workspace['mv_encoded'] and s['failures']==0 and s['retired_bytes']>=0 and s['retired_owners']>=0,
           'Actual compact provider counters differ')
        need(s['allocated_bytes']==s['mm_mid_bytes']+s['f32_gate_bytes']+s['f32_up_bytes']+s['f32_mid_bytes']+s['retired_bytes'],
             'Real active/retired scratch byte accounting differs')
    else:need(workspace['legacy_gate_bytes']==workspace['legacy_up_bytes']==workspace['legacy_mid_bytes']==16384*6*2304*4,
              'Original factory must retain exact native F32 pools')


MEMORY_POINTS=('before_create','after_create','after_encode','after_completed','after_free')
MEMORY_METRICS=('metal_current_allocated_size','process_phys_footprint','process_resident_size')


def checkpoint_statistics(timings):
    # Keep absolute task checkpoints and same-arm deltas. Task ledger/RSS can
    # lag driver release across arms; these values are observations, not a
    # proof that the compact workspace reduces a full-model physical peak.
    result={}
    for arm in ('A1','B','A2'):
        samples=[r for r in timings if r['arm']==arm and r['sample']>0]
        need(len(samples)==9,'Nine formal same-arm memory samples required')
        result[arm]=dict(formal_samples=9,checkpoints={})
        for point in MEMORY_POINTS:
            absolute={metric:statistics.median(r[point][metric] for r in samples) for metric in MEMORY_METRICS}
            delta={metric:statistics.median(r[point][metric]-r['before_create'][metric] for r in samples) for metric in MEMORY_METRICS}
            result[arm]['checkpoints'][point]=dict(absolute_median_bytes=absolute,
                median_paired_delta_from_same_sample_before_create_bytes=delta)
    return dict(actual_checkpoint_count=5,checkpoint_order=list(MEMORY_POINTS),arms=result,
        accounting_limits=['Task physical-footprint and RSS ledgers may lag Metal allocation/release across arms.',
          'Within-arm deltas can be negative because the baseline checkpoint can include deferred previous-arm accounting.',
          'Cross-arm absolute physical-footprint differences are checkpoint observations, not an exact attributable memory saving.',
          'No sixth checkpoint or continuously sampled full-model physical peak exists in this component raw data.'])


def performance(root,observed,plan,log_evidence=None):
    data=rows(root/'performance.stdout',allow_native_log=True,log_evidence=log_evidence);timings=[r for r in data if r['kind']=='timing'];summaries=[r for r in data if r['kind']=='summary']
    wanted=[(layer,sample,arm) for layer in LAYERS for sample in range(10) for arm in ('A1','B','A2')]
    need([(r['layer'],r['sample'],r['arm']) for r in timings]==wanted,'Strict global 90-CB interleaved ABA sequence differs')
    need(len(summaries)==3 and len(data)==94 and data[-1]['kind']=='complete' and data[-1]['timing_command_buffers']==90,
         '90 complete original command buffers plus exact summaries required')
    previous=0
    for r in timings:
        need(yes(r['completed_cb']) and yes(r['full_f32_bit_exact']) and yes(r['finite_guards_written']) and r['command_buffers']==1 and
             r['warmup']==(r['sample']==0) and r['gpu_end_s']>r['gpu_start_s']>=previous and
             close(r['gpu_us'],(r['gpu_end_s']-r['gpu_start_s'])*1e6,1e-5),'Original complete serial CB timeline/normalization differs')
        previous=r['gpu_end_s'];actual=[c for c in observed.values() if c['layer']==r['layer']]
        need(r['actual_cases_per_cb']==len(actual),'Timing omitted an actual tail/case')
        for field in ('factory_wall_us','host_encode_us','host_wait_us'):need(math.isfinite(r[field]) and r[field]>=0,'Host timing field invalid')
        for point in ('before_create','after_create','after_encode','after_completed','after_free'):
            m=r[point];need(m['metal_current_allocated_size']>0 and m['process_phys_footprint']>0 and m['process_resident_size']>0,'Actual memory points required')
        counter_contract(r['after_completed']['workspace'],actual,r['arm']=='B')
    out=[];policy=plan['component']
    need(policy['warmup']==1 and policy['formal_samples']==9 and policy['order']==['A1','B','A2'] and
         policy['maximum_absolute_time_control_drift_percent']==2 and policy['maximum_conservative_gpu_time_regression_percent']==2 and
         policy['minimum_actual_device_allocation_reduction_bytes']==536870912,'Frozen component time/memory gates differ')
    for layer in LAYERS:
        arms=[[r for r in timings if r['layer']==layer and r['arm']==a and r['sample']>0] for a in ('A1','B','A2')]
        t=[statistics.median(r['gpu_us'] for r in group) for group in arms]
        dev=[statistics.median(r['after_completed']['metal_current_allocated_size'] for r in group) for group in arms]
        phys=[statistics.median(r['after_completed']['process_phys_footprint'] for r in group) for group in arms]
        drift=(t[2]/t[0]-1)*100;regression=(t[1]/min(t[0],t[2])-1)*100;reduction=min(dev[0],dev[2])-dev[1]
        published=[s for s in summaries if s['layer']==layer];need(len(published)==1,'Exact three layer summaries')
        p=published[0]
        for key,v in zip(('a1_us','b_us','a2_us'),t):need(close(p[key],v),'Published median differs from original GPU CBs')
        need(close(p['control_drift_percent'],drift) and close(p['conservative_time_saved_percent'],-regression),'Published time arithmetic differs')
        out.append(dict(layer=layer,a1_us=t[0],b_us=t[1],a2_us=t[2],control_drift_percent=drift,
            conservative_gpu_time_regression_percent=regression,conservative_gpu_time_saved_percent=-regression,
            completed_device_allocated_size=dict(zip(('A1','B','A2'),dev)),conservative_actual_device_allocation_reduction_bytes=reduction,
            completed_process_phys_footprint=dict(zip(('A1','B','A2'),phys)),
            observed_cross_arm_completed_process_footprint_difference_bytes=min(phys[0],phys[2])-phys[1],
            memory_checkpoint_observations=checkpoint_statistics([r for r in timings if r['layer']==layer]),
            gate_pass=abs(drift)<=2 and regression<=2 and reduction>=536870912))
    return out


def review(root,model,source_archive,artifacts_archive):
    cm,cf=manifest(root/'correctness','correctness');pm,pf=manifest(root/'performance','performance')
    need(pm['correctness_manifest_sha256']==sha(root/'correctness/artifact-manifest.json') and
         cm['binary_sha256']==pm['binary_sha256'] and cm['metallib_sha256']==pm['metallib_sha256'],
         'Exact prerequisite manifest/binary/library binding differs')
    reader=FloatReader()
    try:capture,outputs,observed=numeric_capture(root,root/'raw-capture/capture.json',model,cm,pm,cf,pf,reader)
    finally:reader.close()
    weights=complete_weights(model,capture,cm,pm);plan=load(root/'protocol.json')
    need(plan['capture']['rows']==16401 and plan['capture']['layers']==list(LAYERS) and plan['capture']['derived_boundaries']==list(BOUNDARIES),
         'Frozen real capture/derived distinction differs')
    tokens=Path(capture['source']['tokens_path']);need(sha(tokens)==capture['source']['tokens_sha256'],'Original tokenizer text SHA differs')
    ids=list(map(int,tokens.read_text().split()));need(len(ids)==16401 and hashlib.sha256(struct.pack('<16401i',*ids)).hexdigest()==plan['capture']['raw_ids_sha256'],
         'Actual frozen tokenizer ID bytes differ')
    native_stdout_logs=[]
    recomputed=performance(root,observed,plan,native_stdout_logs)
    parity=rows(root/'correctness.stdout',allow_native_log=True,log_evidence=native_stdout_logs);pr=[r for r in parity if r['kind']=='parity']
    expected={(c['layer'],c['source_case'],c['rows'],bool(c['derived_real_prefix']),provider) for c in cm['cases'] for provider in ('native','compact')}
    need(len(pr)==len(expected) and {(r['layer'],r['source_case'],r['rows'],bool(r['derived_real_prefix']),r['provider']) for r in pr}==expected and
         all(yes(r['full_f32_bit_exact']) and yes(r['finite_guards_written']) for r in pr),'Original full correctness parity coverage differs')
    expected_order=[]
    for layer in LAYERS:
        group=[c for c in cm['cases'] if c['layer']==layer]
        for provider in ('native','compact'):
            sequence=[]
            for c in group:
                key=case_key(c);expected_order.append((*key,provider))
                source=observed.get(key);sequence.append(dict(actual_mm=(c['rows']>=32) if key[3] else source['actual_mm']))
                item=next(r for r in pr if (r['layer'],r['source_case'],r['rows'],bool(r['derived_real_prefix']),r['provider'])==(*key,provider))
                counter_contract(item['workspace'],sequence,provider=='compact')
                if provider=='compact':need(item['workspace']['scratch']['last_provider']==(1 if sequence[-1]['actual_mm'] else 2),
                                            'Actual boundary MM/MV provider differs')
    need([(r['layer'],r['source_case'],r['rows'],bool(r['derived_real_prefix']),r['provider']) for r in pr]==expected_order and
         len([r for r in parity if r['kind']=='allocation'])==6 and len(parity)==len(pr)+7 and
         parity[-1]['kind']=='complete' and parity[-1]['cases']==len(cm['cases']) and parity[-1]['timing_command_buffers']==0,
         'Complete all-layer correctness before timing and original factory records required')
    failures=load(root/'failure-review.json');need(failures.get('scope')=='independent_component_failure_artifact_review' and
         failures.get('evidence_valid') is True and failures.get('functional_failure_gate_pass') is True and
         failures.get('performance_measured') is False and failures.get('fullmodel_cancel_recovery_proven') is False and
         failures['manifest_sha256']==sha(root/'failures/artifact-manifest.json') and failures['groups']==9,
         'Independent complete functional failure review must explicitly pass')
    failure_manifest=load(root/'failures/artifact-manifest.json')
    for item in failure_manifest['files']:
        p=bounded(root/'failures',item['path']);need(p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],
             'Functional failure original payload SHA/extent differs')
        if item['path'].endswith('.guards'):need(p.read_bytes()==b'\xa5'*512,'Functional failure original guard differs')
    steps=load(root/'steps.json')
    need([s['name'] for s in steps]==['allocations','capture','host','correctness','failures','failure-review','performance'] and
         all(s['returncode']==0 for s in steps),'Actual functional-reader gate must precede timing')
    archives=verify_archives(root,source_archive,artifacts_archive,cm,pm)
    before,restored,summary=[load(root/n) for n in ('before.json','restored.json','summary.json')]
    need(all(before[k]==restored[k] for k in IDENTITY_KEYS) and before['health']==restored['health']=={'status':'ok'} and
         summary['restoration']['restored'] is True and summary['restoration']['pid']==restored['pid'] and
         summary['production_adoption'] is False and summary['fullmodel_run'] is False,'Exact production restoration/component scope differs')
    need(cm['metallib_sha256']==before['metallib_sha256'] and
         sha(root/'production-artifacts/myds')==before['binary_sha256'] and
         sha(root/'production-artifacts/myds.metallib')==before['metallib_sha256'] and
         sha(root/'original-myds.conf')==before['config_sha256'] and load(root/'effective-env.json')==before['effective_myds_env'],
         'Frozen original production shader/config/recovery artifacts differ')
    expected_environment=dict(before['effective_myds_env'],MYDS_PREFILL_MOE_WORKSPACE='0')
    need(capture['source']['environment']==expected_environment and expected_environment['MYDS_METAL_CONCURRENT']=='1',
         'Observed capture environment differs from frozen production plus explicit default-off flag')
    gate=all(r['gate_pass'] for r in recomputed)
    return dict(schema_version=1,reader_sha256=sha(Path(__file__)),artifact_review_passed=True,evidence_valid=True,component_gate_pass=gate,
        native_stdout_logs=native_stdout_logs,gpu_work_submitted_by_reader=False,
        production_adoption=False,normal_component_time_memory_gate_pass=gate,fullmodel_run=False,
        decision='qualifies_separate_fullmodel_and_recovery_gates' if gate else 'withhold_fullmodel_component_gate_failed',
        original_pid=before['pid'],restored_pid=restored['pid'],full_original_output_comparisons=outputs,
        complete_model_backed_weight_intervals=weights,weight_payload_exported=False,raw_gpu_command_buffers_verified=90,
        recomputed_performance_and_memory=recomputed,correctness_manifest_sha256=sha(root/'correctness/artifact-manifest.json'),
        performance_manifest_sha256=sha(root/'performance/artifact-manifest.json'),failure_review_sha256=sha(root/'failure-review.json'),
        archives=archives,scope_limits=['Repeated real fixture component; no cold DRAM or fullmodel performance claim.',
          'Task physical-footprint/RSS checkpoints may lag driver release; their differences are not exact attributable savings or a sampled fullmodel peak.',
          'Original GGUF must be retained for complete model-backed weight re-verification; local raw archive does not contain weight payloads.',
          'Functional allocation/PSO proof is separate from real asynchronous cancellation and same-context recovery.'])


def verify_archives(root,source_archive,artifacts_archive,cm,pm):
    bindings=load(root/'stage-inputs.json');index={};reports=[]
    for kind,path in (('source',source_archive),('artifacts',artifacts_archive)):
        count=0
        with tarfile.open(path,'r:*') as tar:
            for member in tar:
                if member.isdir():continue
                need(not member.issym() and not member.isdev() and '..' not in Path(member.name).parts,'Unsafe archived source/artifact')
                f=tar.extractfile(member);need(f is not None,'Unreadable original archive member');h=hashlib.sha256()
                for block in iter(lambda:f.read(1048576),b''):h.update(block)
                index.setdefault((kind,h.hexdigest()),[]).append(member.name);count+=1
        reports.append(dict(kind=kind,path=str(path),sha256=sha(path),members_verified=count))
    roots={Path(p).parent for p in bindings if Path(p).name=='Makefile'}
    for name,digest in bindings.items():
        p=Path(name);stages=[s for s in roots if s in p.parents]
        if stages:
            need(len(stages)==1,'Ambiguous frozen source root');relative=str(p.relative_to(stages[0]));kind='artifacts' if relative.startswith('bin/') else 'source'
            need(any(n==relative or n.endswith('/'+relative) for n in index.get((kind,digest),[])),
                 'Frozen executed source/artifact missing from archives: '+name)
        else:need(p.is_file() and sha(p)==digest,'Frozen external input SHA differs: '+name)
    need(cm['binary_sha256'] in {v for k,v in bindings.items() if k.endswith('/bin/bench_moe_workspace')} and
         cm['metallib_sha256']==pm['metallib_sha256'] and cm['metallib_sha256'] in {v for k,v in bindings.items() if k.endswith('/bin/myds.metallib')},
         'Actual component binary/library not frozen in executed archive')
    return reports


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','model','source-archive','artifacts-archive','out'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    try:result=review(a.root.resolve(),a.model.resolve(),a.source_archive,a.artifacts_archive)
    except Exception as exc:result=dict(schema_version=1,artifact_review_passed=False,evidence_valid=False,component_gate_pass=False,gpu_work_submitted_by_reader=False,production_adoption=False,error=repr(exc),reader_sha256=sha(Path(__file__)))
    with a.out.open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True);stream.write('\n')
    print(json.dumps({k:result[k] for k in ('artifact_review_passed','decision','error') if k in result}))
    return 0 if result['artifact_review_passed'] else 1


def gguf_descriptors(path, wanted):
    """Parse CPU metadata, skip tokenizer arrays, and retain named tensors only."""
    size = Path(path).stat().st_size
    with Path(path).open('rb') as stream:
        def read(n):
            need(0 <= n <= size - stream.tell(), 'GGUF read outside file')
            data = stream.read(n)
            need(len(data) == n, 'Short GGUF read')
            return data
        def integer(fmt):
            return struct.unpack('<' + fmt, read(struct.calcsize('<' + fmt)))[0]
        def skip(n):
            need(0 <= n <= size - stream.tell(), 'GGUF skip outside file')
            stream.seek(n, 1)
        def string():
            n = integer('Q')
            need(n <= 1048576, 'Unbounded GGUF tensor/metadata name')
            return read(n).decode('utf-8')
        primitive = {0:'B', 1:'b', 2:'H', 3:'h', 4:'I', 5:'i', 6:'f', 7:'?', 10:'Q', 11:'q', 12:'d'}
        def discard(kind):
            if kind in primitive:
                skip(struct.calcsize('<' + primitive[kind]))
            elif kind == 8:
                skip(integer('Q'))
            else:
                need(kind == 9, 'Unsupported GGUF metadata type')
                element, count = integer('I'), integer('Q')
                need(count <= 10000000 and element != 9, 'Unbounded/nested GGUF metadata array')
                if element in primitive:
                    skip(count * struct.calcsize('<' + primitive[element]))
                else:
                    need(element == 8, 'Unsupported GGUF array element type')
                    for _ in range(count):
                        skip(integer('Q'))
        need(read(4) == b'GGUF' and integer('I') in (2, 3), 'Expected GGUF v2/v3')
        nt, nk = integer('Q'), integer('Q')
        need(nt < 100000 and nk < 100000, 'Unbounded GGUF descriptor count')
        alignment = 32
        for _ in range(nk):
            key, kind = string(), integer('I')
            if key == 'general.alignment':
                need(kind == 4, 'GGUF alignment must be UINT32')
                alignment = integer('I')
            else:
                discard(kind)
        descriptors = {}
        for _ in range(nt):
            name, nd = string(), integer('I')
            need(0 < nd <= 4, 'Invalid GGUF tensor rank')
            dims = [integer('Q') for _ in range(nd)]
            kind, offset = integer('I'), integer('Q')
            if name in wanted:
                need(name not in descriptors, 'Duplicate target tensor')
                descriptors[name] = dict(type=kind, dimensions=dims, offset=offset)
        need(0 < alignment <= 1048576 and not alignment & (alignment - 1), 'Invalid GGUF alignment')
        data_offset = (stream.tell() + alignment - 1) // alignment * alignment
    need(set(descriptors) == set(wanted), 'Missing real GGUF target descriptor')
    return descriptors, data_offset



if __name__=="__main__":raise SystemExit(main())
