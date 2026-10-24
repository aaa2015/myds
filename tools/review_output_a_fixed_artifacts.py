#!/usr/bin/env python3
"""Independent CPU audit of ordinary-decode Output_A raw evidence.

No runner/harness analyse function is imported; no Metal, inference process,
network request, or benchmark is executed. Exit 0 means consistent evidence,
including a valid numerical/performance rejection. It does not mean adoption.
Use --self-test for synthetic CPU protocol controls, never device evidence.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import sys
import tarfile
import tempfile

LAYERS=(0,19,39)
POSITIONS=(0,1,7,15)
CONTROLS=('production','probe')
ARMS=('A1','B','A2')
NATIVE='kernel_dsv4_attn_out_low_q4_K_bf16'
FIXED='kernel_myds_outa_q4_v41_fixed_bf16'
INPUT_BYTES,OUTPUT_BYTES,WEIGHT_BYTES=131072,32768,18874368
PROBE='decode_output_a_fixed_real_capture'
BOUNDARY='post_inverse_rope_bf16_before_output_a'
IDENTITY_KEYS=('argv_raw_hex','myds_env_raw_hex','cwd','log','nice','binary',
               'binary_sha256','metallib','metallib_sha256','config','config_sha256','effective_myds_env')


def need(value,message):
    if not value:raise ValueError(message)


def sha(path,offset=0,length=None):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        stream.seek(offset)
        while length is None or length:
            data=stream.read(1048576 if length is None else min(length,1048576))
            if not data:break
            value.update(data)
            if length is not None:length-=len(data)
    need(length in (None,0),'Short SHA interval: '+str(path))
    return value.hexdigest()


def digest(data):return hashlib.sha256(data).hexdigest()


def yes(value):
    # Objective-C @(a && b) boxes C int 1, while @(BOOL_variable) boxes a
    # JSON boolean. Both are emitted by the frozen harness for these flags.
    return isinstance(value,(bool,int)) and value==1


def load(path):return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def close(a,b):
    return isinstance(a,(int,float)) and not isinstance(a,bool) and math.isfinite(a) and math.isclose(a,b,rel_tol=1e-8,abs_tol=1e-5)


def uint(value,limit,message):
    need(isinstance(value,int) and not isinstance(value,bool) and 0<=value<=limit,message)
    return value


def path_inside(base,relative):
    need(isinstance(relative,str) and relative and not Path(relative).is_absolute(),'Relative artifact path required')
    p=base/relative
    need(not p.is_symlink() and p.resolve().is_relative_to(base.resolve()) and p.is_file(),
         'Missing, symlinked, or escaped artifact: '+relative)
    return p


def contract():
    return dict(args_bytes=136,buffer_slots=['args136','q4k_weights','heads_f32','low_f32'],
                groups=[128,1,8],threads=[32,4,1],fc600_short=4,nr0=2,dynamic_threadgroup_bytes=256,
                input_bytes=INPUT_BYTES,weight_bytes=WEIGHT_BYTES,output_bytes=OUTPUT_BYTES,bf16_postround=True)


def measurement(root):
    p=load(root/'protocol.json');key=(p.get('samples'),p.get('reps'),p.get('projections_per_cb'))
    choices={(9,8,32):dict(name='original_short_cb_32_projections',samples=9,reps=8,projections=32,cbs=180,formal=162,
                           scope='ordinary_decode_output_a_fixed_wrapper',bench='bench_output_a_fixed',
                           runner='run_output_a_fixed_window.py',required_controls=('production',)),
             (15,256,1024):dict(name='separate_steady_cb_1024_projections',samples=15,reps=256,projections=1024,cbs=288,formal=270,
                               scope='ordinary_decode_output_a_fixed_wrapper_steady',bench='bench_output_a_fixed_steady',
                               runner='run_output_a_fixed_steady_window.py',required_controls=CONTROLS)}
    need(key in choices,'Only frozen (9,8,32,180) or (15,256,1024,288) measurement protocols are supported')
    method=choices[key]
    need(p.get('scope')==method['scope'],'Original/steady protocol scopes must remain separate')
    if method['projections']==1024:
        need(p.get('measurement_method')==method['name'] and p.get('required_control_kinds')==list(CONTROLS),
             'Steady method identity and both-control gate must be explicit')
    else:
        need('measurement_method' not in p and 'required_control_kinds' not in p,
             'Original short-CB protocol cannot silently acquire a different gate/method')
    return method


def args136():
    data=bytearray(136)
    for offset,value in {0:8,4:1,16:4096,20:1024,24:8,56:4096,60:8,64:1,68:1,96:1024,100:8,112:2}.items():
        struct.pack_into('<i',data,offset,value)
    for offset,value in {8:32,32:144,40:2304,48:2359296,72:4,80:16384,88:131072,104:4096}.items():
        struct.pack_into('<Q',data,offset,value)
    return bytes(data)


def f32_state(data,written=True):
    need(len(data)%4==0,'F32 alignment')
    values=[v[0] for v in struct.iter_unpack('<I',data)]
    return dict(elements=len(values),finite=all(v&0x7f800000!=0x7f800000 for v in values),
                written=not written or all(v!=0xa5a5a5a5 for v in values),
                bf16_postround=all(v&0xffff==0 for v in values))


def compare(data,reference):
    need(len(data)==len(reference)==OUTPUT_BYTES,'Complete Output_A output length')
    a=[v[0] for v in struct.iter_unpack('<I',reference)]
    b=[v[0] for v in struct.iter_unpack('<I',data)]
    different=[i for i,(x,y) in enumerate(zip(a,b)) if x!=y]
    maximum_ulp=0;maximum_abs=0.0
    for i in different:
        x,y=a[i],b[i]
        if x&0x7f800000==0x7f800000 or y&0x7f800000==0x7f800000:continue
        def ordered(v):return 0x80000000-(v&0x7fffffff) if v&0x80000000 else 0x80000000+v
        maximum_ulp=max(maximum_ulp,abs(ordered(x)-ordered(y)))
        xf,yf=struct.unpack('<f',struct.pack('<I',x))[0],struct.unpack('<f',struct.pack('<I',y))[0]
        maximum_abs=max(maximum_abs,abs(xf-yf))
    first=different[0] if different else None
    return dict(bytes=OUTPUT_BYTES,elements=OUTPUT_BYTES//4,mismatches=len(different),first_mismatch=first,
                first_expected_bits=None if first is None else a[first],first_actual_bits=None if first is None else b[first],
                first_group=None if first is None else first//1024,first_rank=None if first is None else first%1024,
                max_ulp_finite_pairs=maximum_ulp,max_abs_finite_pairs=maximum_abs,all_words_compared=True)


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


def verify_manifest(directory,status=None):
    method=measurement(directory.parent)
    manifest=load(directory/'artifact-manifest.json');entries=manifest['files'];by_name={};hashes={}
    need(isinstance(entries,list),'Manifest file list')
    for entry in entries:
        name=entry['path'];need(name not in by_name,'Duplicate logical artifact: '+name)
        p=path_inside(directory,name);st=p.stat();key=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
        if key not in hashes:hashes[key]=sha(p)
        need(entry['bytes']==st.st_size and entry['sha256']==hashes[key],'Raw file bytes/SHA: '+name)
        by_name[name]=entry
    if status:need(manifest['status']==status,'Manifest status: '+str(directory))
    need(manifest['probe']==PROBE and manifest['total_cases']==12 and manifest['samples']==method['samples'] and
         manifest['repetitions_per_cb']==method['reps'] and manifest['mode']=='concurrent' and
         manifest['binding_contract']==contract() and manifest['fullmodel'] is False and
         manifest['cold_dram_proven'] is False and manifest['repeated_fixture_cache_reuse'] is True,
         'Frozen Output_A protocol/ABI differs')
    need(path_inside(directory,'args136.bin').read_bytes()==args136(),'Actual args136 bytes differ')
    if manifest['status']!='host_only':
        need(len(manifest['bindings'])==3,'Three actual PSO bindings required')
        for p,b in enumerate(manifest['bindings']):
            need(b['provider']==p and b['symbol']==(FIXED if p==2 else NATIVE) and
                 b['library']==('production' if p==0 else 'probe') and b['contract']==contract() and
                 b['thread_execution_width']==32 and b['max_threads']>=128 and
                 0<=b['static_threadgroup_bytes']<=32768-256,'Actual provider PSO ABI/geometry differs')
    return manifest,by_name,len(hashes)


def verify_stdout_identity(path,manifest):
    data=rows(path);identity=[r for r in data if r.get('kind')=='identity']
    need(len(identity)==1,'Exactly one original harness identity record required')
    initial=identity[0]['manifest']
    for k in ('probe','phase','total_cases','samples','repetitions_per_cb','mode','binding_contract',
              'binary_sha256','probe_metallib_sha256','production_metallib_sha256','capture_json_sha256',
              'capture_source','model','cases','tensors'):
        need(initial[k]==manifest[k],'Harness stdout/manifest initial identity differs: '+k)
    if manifest['status']!='host_only':
        bindings=[r for r in data if r.get('kind')=='bindings']
        need(len(bindings)==1 and bindings[0]['bindings']==manifest['bindings'] and bindings[0]['device']==manifest['device'],
             'Actual PSO stdout/manifest binding differs')


def verify_capture(root,manifest):
    capture=load(root/'capture.json');source=capture['source']
    need(capture['schema_version']==1 and capture['capture_kind']=='myds_decode_output_a' and len(capture['cases'])==12,
         'Ordinary decode capture schema/count')
    need(sha(root/'capture.json')==manifest['capture_json_sha256'] and manifest['capture_source']==source,
         'Component capture identity')
    need(source['kind']=='actual_public_ordinary_decode_post_inverse_rope_bf16' and source['trace_on'] is False and
         source['performance_from_capture'] is False,'Actual ordinary observer scope')
    for key,name in [('request_sha256','request.json'),('token_file_sha256','capture-tokens.txt'),
                     ('effective_environment_sha256','effective-env.json'),('records_sha256','raw-capture/capture-records.jsonl')]:
        need(source[key]==sha(root/name),'Capture request/token/env/records SHA: '+name)
    provider=load(root/'raw-capture/capture-provider.json');observer=load(root/'raw-capture/capture-observer-summary.json')
    need(source['provider']==provider and source['observer']==observer,'Original capture provider/observer identity')
    need(all(provider[k] is v for k,v in dict(public_forward_need_logits=True,ordinary_decode_allowed=True,trace_on=False,
             skip_flags_present=False,provider_contract_passed=True,performance_from_capture=False).items()) and
         provider['forward_tokens_completed']==16 and provider['max_context']==327680 and
         provider['rope_mode']==provider['register_mode']==1 and
         all(provider[k]==640 for k in ('rf_requests','rf_encoded','register_requests','register_encoded')) and
         provider['rf_failures']==provider['register_failures']==0 and
         provider['register_direct_encoded']+provider['register_staged_encoded']==640,'RF/register actual encode counters')
    need(observer['status']=='complete' and observer['capture_kind']=='myds_decode_output_a' and
         observer['observed_layer_calls']==observer['rope_consumed_layer_calls']==640 and observer['selected_cases']==12 and
         observer['selected_mask']==0xfff and observer['forward_tokens']==16 and observer['input_bytes_per_case']==INPUT_BYTES and
         observer['layers']==list(LAYERS) and observer['positions']==list(POSITIONS) and
         all(observer[k] is v for k,v in dict(all_f32_finite=True,all_bf16_bits=True,public_ordinary_decode=True,
                                              trace_on=False,performance_from_capture=False).items()),'Complete actual observer counters')
    original=rows(root/'raw-capture/capture-records.jsonl')
    expected_pairs=[(p,l) for p in POSITIONS for l in LAYERS]
    need([(r['position'],r['layer']) for r in original]==expected_pairs,'Original capture observer order/coverage')
    original_by_pair={(r['layer'],r['position']):r for r in original}
    cases=capture['cases'];need([(c['layer'],c['position']) for c in cases]==[(l,p) for l in LAYERS for p in POSITIONS],
                              'Twelve ordered layer/position inputs')
    reports=[]
    for index,c in enumerate(cases):
        layer,pos=c['layer'],c['position'];o=original_by_pair[(layer,pos)];desc=c['input_f32'];od=o['input_f32']
        need({k:v for k,v in c.items() if k!='input_f32'}=={k:v for k,v in o.items() if k!='input_f32'},
             'Capture JSON metadata differs from original observer record')
        expected_name=f'layer-{layer}-pos-{pos}-input-f32.bin'
        need(Path(desc['path']).name==od['path']==expected_name and desc['bytes']==od['bytes']==INPUT_BYTES and
             desc['sha256']==od['sha256'],'Input descriptor differs from actual observer')
        actual=path_inside(root/'raw-capture',expected_name);data=actual.read_bytes()
        need(len(data)==INPUT_BYTES and sha(actual)==desc['sha256'],'Original heads full bytes/SHA')
        state=f32_state(data,written=False);need(state['finite'] and state['bf16_postround'],'Original heads finite/BF16 bits')
        need(c['call_ordinal']==pos*40+layer+1 and c['need_logits'] is True and c['ordinary_decode_allowed'] is True and
             c['trace_on'] is False and c['attention_rope_consumed'] is True and isinstance(c['compressed'],bool) and
             c['weight_type']==12 and c['boundary']==BOUNDARY and c['all_f32_finite'] is True and c['all_bf16_bits'] is True and
             c['observer_commit_wait'] is True and c['observer_batch_restored'] is True and c['performance_from_capture'] is False,
             'Actual hook metadata does not represent ordinary post-inverse-RoPE heads')
        ci=manifest['cases'][index]
        need(ci==dict(case=index,layer=layer,position=pos,input_bytes=INPUT_BYTES,input_sha256=desc['sha256'],
                     attention_rope_consumed=True,compressed=c['compressed'],boundary=BOUNDARY),'Component captured case identity')
        reports.append(dict(case=index,layer=layer,position=pos,actual_input_sha256=desc['sha256'],
                            full_input_f32_values=32768,finite=True,bf16_postround=True,rf_consumed=True))
    tokens=[int(v) for v in (root/'capture-tokens.txt').read_text().split()];tokenized=(root/'tokenized.tokens.i32le.bin').read_bytes()
    need(len(tokens)==16 and all(v>=0 for v in tokens) and len(tokenized)>=64 and len(tokenized)%4==0 and
         list(struct.unpack('<16i',tokenized[:64]))==tokens,'Sixteen actual tokenizer prefix IDs')
    # Model loading emits ordinary text logs on this driver's stdout. The
    # harness/records remain strict JSONL; only this mixed driver log is filtered.
    complete=[json.loads(line) for line in (root/'capture.stdout').read_text().splitlines()
              if line.lstrip().startswith('{') and json.loads(line).get('kind')=='capture_complete']
    need(len(complete)==1 and complete[0]['cases']==12 and complete[0]['tokens']==16 and
         complete[0]['ordinary_decode'] is True and complete[0]['real_model_forward'] is True and
         complete[0]['performance_from_capture'] is False,'Capture driver completion')
    return capture,reports


def verify_weights(root,manifest,model):
    tensors=manifest['tensors'];wanted={f'blk.{layer}.attn_output_a.weight' for layer in LAYERS}
    need(len(tensors)==3 and {t['name'] for t in tensors}==wanted,'Three real Output_A tensors')
    descriptors,start=gguf_descriptors(model,wanted);st=model.stat();recorded=manifest['model'];capture=load(root/'capture.json')
    need(st.st_size==recorded['bytes']==capture['model_bytes'] and start==recorded['tensor_data_offset'] and
         recorded['header_bytes']==capture['model_header_bytes']==1048576 and
         sha(model,length=1048576)==recorded['header_sha256']==capture['model_header_sha256'],
         'Actual GGUF size/prefix/tensor-data offset')
    reports=[]
    for t in tensors:
        d=descriptors[t['name']];absolute=start+d['offset'];layer=t['layer']
        need(layer in LAYERS and t['name']==f'blk.{layer}.attn_output_a.weight' and
             t['type']==d['type']==12 and t['dimensions']==d['dimensions']==[4096,8192] and
             t['bytes']==WEIGHT_BYTES and t['file_offset']==absolute and 0<=absolute<=st.st_size-WEIGHT_BYTES,
             'Actual GGUF Q4_K tensor dimensions/offset/interval')
        need(sha(model,absolute,WEIGHT_BYTES)==t['sha256'],'Actual full GGUF weight interval SHA')
        reports.append(dict(layer=layer,name=t['name'],type=12,dimensions=[4096,8192],bytes=WEIGHT_BYTES,
                            tensor_file_offset=absolute,sha256=t['sha256'],complete_interval_sha_verified=True))
    source_model=capture['source']['model_metadata'];summary=load(root/'summary.json')
    need(source_model==summary['model'] and source_model['bytes']==st.st_size and
         summary['model_header_sha256']==recorded['header_sha256'],'Capture/runner original model identity')
    same_stat=(source_model['device'],source_model['inode'],source_model['bytes'],source_model['mtime_ns'])==(
        st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
    return reports,same_stat


def verify_component_inputs(directory,manifest,names,capture,require_all_guards):
    need(path_inside(directory,'capture.json').read_bytes()==json.dumps(capture).encode() or
         load(directory/'capture.json')==capture,'Component capture JSON differs')
    need(sha(directory/'capture.json')==manifest['capture_json_sha256'],'Component capture bytes SHA')
    guard_names=[];weight_count=0
    for i,c in enumerate(capture['cases']):
        name=f'capture-case-{i:03d}-input.f32';need(name in names,'Complete archived captured input')
        data=path_inside(directory,name).read_bytes();need(len(data)==INPUT_BYTES and digest(data)==c['input_f32']['sha256'],
                                                         'Archived captured heads differ from observer')
        name=f'input-case-{i:03d}.guards'
        if name in names:
            need(path_inside(directory,name).read_bytes()==b'\xa5'*128,'Archived GPU input guard bytes damaged');guard_names.append(name)
        else:need(not require_all_guards,'Complete successful phase requires input guard raw: '+name)
    for t in manifest['tensors']:
        name=f"weight-layer-{t['layer']}-output-a.q4k"
        if name in names:
            need(names[name]['bytes']==WEIGHT_BYTES and names[name]['sha256']==t['sha256'],'Actual archived weight payload');weight_count+=1
        else:need(not require_all_guards,'Complete successful phase requires all real weight payloads')
        name=f"weight-layer-{t['layer']}-output-a.guards"
        if name in names:
            need(path_inside(directory,name).read_bytes()==b'\xa5'*128,'Archived weight guard bytes damaged');guard_names.append(name)
        else:need(not require_all_guards,'Complete successful phase requires weight guard raw')
    return dict(full_input_payloads_verified=12,archived_weight_tensor_payloads_verified=weight_count,
                archived_input_and_weight_guard_payloads_verified=len(guard_names),
                complete_input_and_weight_guards_raw_verified=len(guard_names)==15)


def verify_input_records(manifest,expected):
    records=manifest['input_integrity'];need(len(records)==len(expected),'Exact input integrity record count')
    tensors={t['layer']:t for t in manifest['tensors']}
    for r,(case,p,ordinal) in zip(records,expected):
        ci=manifest['cases'][case];w=tensors[ci['layer']]['sha256'];x=ci['input_sha256']
        need(r['case']==case and r['provider']==p and r['command_ordinal']==ordinal and
             r['input_sha256']==r['expected_input_sha256']==x and r['weight_sha256']==r['expected_weight_sha256']==w and
             r['input_guards_intact'] is True and r['weight_guards_intact'] is True and yes(r['all_gpu_inputs_unchanged']),
             'Per-CB input/weight SHA and guard record differs from bound actual bytes')
    return len(records)


def verify_correctness(root,manifest,names,complete):
    directory=root/'correctness';out=rows(root/'correctness.stdout');parity=[r for r in out if r.get('kind')=='parity']
    need(not any(r.get('kind') in ('timing','summary') for r in out),'Any timing before complete correctness gate is forbidden')
    need(manifest['timing_command_buffers']==0 and manifest['phase']=='correctness','Correctness phase must contain zero timing CBs')
    need(parity==manifest['parity'] and 0<len(parity)<=36,'Correctness stdout/manifest parity records')
    order=[(case,p,li*3+p+1) for li in range(3) for p in range(3) for case in range(li*4,li*4+4)]
    reports=[];bad=[];expected_input=[]
    for r,(case,p,ordinal) in zip(parity,order):
        ci=manifest['cases'][case];prefix=f'parity-case-{case:03d}-provider-{p}-output';native=f'parity-case-{case:03d}-provider-0-output.f32'
        need(prefix+'.f32' in names and prefix+'.guards' in names and native in names,'Retained complete provider output/guards')
        data=path_inside(directory,prefix+'.f32').read_bytes();reference=path_inside(directory,native).read_bytes()
        diff=compare(data,reference);state=f32_state(data);g=path_inside(directory,prefix+'.guards').read_bytes()==b'\xa5'*128
        ok=diff['mismatches']==0 and all(state[k] for k in ('finite','written','bf16_postround')) and g
        need(r['case']==case and r['provider']==p and r['layer']==ci['layer'] and r['position']==ci['position'] and
             r['command_ordinal']==ordinal and r['phase']=='correctness' and r['mode']=='concurrent' and
             r['output_bytes']==OUTPUT_BYTES and r['mismatches']==diff['mismatches'] and r['first_mismatch']==diff['first_mismatch'] and
             r['output_sha256']==digest(data) and r['full_f32_bit_exact'] is ok and r['finite']==state['finite'] and
             r['written']==state['written'] and r['bf16_postround']==state['bf16_postround'] and r['guards_intact'] is g and
             r['finite_guards_written']==(state['finite'] and state['written'] and g) and
             yes(r['all_gpu_inputs_unchanged']) and r['completed_cb'] is True,'Parity metadata disagrees with complete raw output')
        result=dict(case=case,layer=ci['layer'],position=ci['position'],provider=p,**diff,
                    **{k:v for k,v in state.items() if k!='elements'},guards_intact=g,full_f32_bit_exact=ok)
        reports.append(result);expected_input.append((case,p,ordinal))
        if not ok:bad.append(result)
    verify_input_records(manifest,expected_input)
    actual_outputs={name for name in names if name.startswith('parity-')}
    need(actual_outputs=={f'parity-case-{r["case"]:03d}-provider-{r["provider"]}-output.{suffix}' for r in parity for suffix in ('f32','guards')},
         'Unexpected or missing complete correctness output artifact')
    if complete:
        need(len(parity)==36 and not bad and manifest['status']=='complete' and manifest['full_f32_bit_exact'] is True and
             manifest['output_a_checked'] is True and manifest['all_gpu_inputs_unchanged'] is True,'Complete all-case correctness gate')
        tail=out[-1];need(tail['kind']=='complete' and tail['phase']=='correctness' and tail['cases']==12 and tail['providers']==3 and
             tail['timing_command_buffers']==0 and tail['output_a_checked'] is True and tail['full_f32_bit_exact'] is True,
             'Correctness stdout complete record')
    else:
        need(manifest['status']=='failed' and len(bad)==1 and reports[-1]==bad[0] and bad[0]['mismatches']>0 and
             manifest['failure']=='Output_A full F32 bits/BF16/finite/poison/guards and input identity' and
             not any(r.get('kind')=='complete' for r in out),'Failed correctness must retain the first actual numerical mismatch and stop')
        need(not (root/'performance').exists() and not (root/'performance.stdout').exists(),'Performance present after correctness rejection')
    return reports,bad


def verify_timing(root,manifest,correctness):
    method=measurement(root);samples,reps,projections,cbs=(method[k] for k in ('samples','reps','projections','cbs'))
    directory=root/'performance';out=rows(root/'performance.stdout');timing=[r for r in out if r.get('kind')=='timing'];published=[r for r in out if r.get('kind')=='summary']
    need(manifest['status']=='complete' and manifest['phase']=='performance' and manifest['full_f32_bit_exact'] is True and
         manifest['output_a_checked'] is True and manifest['all_gpu_inputs_unchanged'] is True and manifest['timing_command_buffers']==cbs,
         'Complete performance phase manifest')
    for key in ('binary_sha256','capture_json_sha256','capture_source','probe_metallib_sha256','production_metallib_sha256','model','cases','tensors','binding_contract'):
        need(manifest[key]==correctness[key],'Performance identity differs from the all-case correctness gate: '+key)
    need(manifest['correctness_manifest_sha256']==sha(root/'correctness/artifact-manifest.json')==sha(directory/'correctness-manifest.json') and
         manifest['prior_correctness_providers_verified']==36,'Actual prior correctness gate binding')
    need(len(timing)==cbs and len(published)==6 and not any(r.get('kind') in ('parity','parity_failure') for r in out),
         'Expected exactly the frozen number of complete timing CBs and six summaries')
    expected=[(l,c,s,a) for l in LAYERS for c in CONTROLS for s in range(samples+1) for a in ARMS]
    need([(r['layer'],r['control_kind'],r['sample'],r['arm']) for r in timing]==expected,'Strict full-layer A1/B/A2 interleaving')
    last_end=0;input_expected=[];derived_values={}
    for ordinal,(r,(layer,control,sample,arm)) in enumerate(zip(timing,expected),1):
        p=2 if arm=='B' else (0 if control=='production' else 1);start,end=r['gpu_start_s'],r['gpu_end_s']
        need(all(isinstance(v,(float,int)) and not isinstance(v,bool) and math.isfinite(v) for v in (start,end)) and
             start>0 and end>start and start>=last_end,'Invalid/overlapping complete CB GPU timestamps');last_end=end
        value=(end-start)*1e6/projections;derived_values[(layer,control,sample,arm)]=value
        need(r['provider']==p and r['symbol']==(FIXED if p==2 else NATIVE) and r['command_ordinal']==ordinal and
             r['mode']=='concurrent' and r['cases_in_cb']==4 and r['repetitions']==reps and r['dispatches']==r['projections_per_cb']==projections and
             r['commits']==1 and r['warmup']==(sample==0) and close(r['gpu_us_per_projection'],value) and
             all(r[k] is True for k in ('completed_cb','full_f32_bit_exact','all_gpu_inputs_unchanged','component_only')),
             'Raw timing provider/dispatch/projection normalization differs')
        li=LAYERS.index(layer);input_expected.extend((case,p,ordinal) for case in range(li*4,li*4+4))
    verify_input_records(manifest,input_expected)
    recomputed=[]
    for layer in LAYERS:
        for control in CONTROLS:
            a1,b,a2=[statistics.median(derived_values[(layer,control,s,a)] for s in range(1,samples+1)) for a in ARMS]
            calculated=dict(a1_us=a1,b_us=b,a2_us=a2,control_drift_percent=(a2/a1-1)*100,
                time_saved_percent=(1-b/((a1+a2)/2))*100,conservative_time_saved_percent=(1-b/min(a1,a2))*100)
            pub=[r for r in published if (r['layer'],r['control_kind'])==(layer,control)]
            need(len(pub)==1 and pub[0]['mode']=='concurrent' and pub[0]['control_symbol']==NATIVE and pub[0]['candidate_symbol']==FIXED and
                 all(close(pub[0][k],v) for k,v in calculated.items()),'Published summary differs from original GPU timestamps')
            drift_ok=abs(calculated['control_drift_percent'])<=2;saved_ok=calculated['conservative_time_saved_percent']>2
            recomputed.append(dict(layer=layer,control_kind=control,**calculated,
                control_drift_gate_pass=drift_ok,minimum_improvement_gate_pass=saved_ok,
                candidate_time_ratio_to_faster_control=b/min(a1,a2),performance_gate_pass=drift_ok and saved_ok))
    tail=out[-1];need(tail['kind']=='complete' and tail['phase']=='performance' and tail['cases']==12 and tail['providers']==3 and
         tail['timing_command_buffers']==cbs and tail['output_a_checked'] is True and tail['full_f32_bit_exact'] is True,'Performance stdout complete')
    return recomputed


def archive_file(archive,relative):
    candidates=[m for m in archive.getmembers() if m.name==relative or m.name.endswith('/'+relative)]
    need(len(candidates)==1 and not candidates[0].issym(),'Missing/ambiguous/symlink archive member: '+relative)
    stream=archive.extractfile(candidates[0]);need(stream is not None,'Unreadable archive member: '+relative)
    value=hashlib.sha256()
    for block in iter(lambda:stream.read(1048576),b''):value.update(block)
    return value.hexdigest()


def verify_archives(root,source_archive,artifacts_archive,manifest,capture):
    method=measurement(root)
    bindings=load(root/'stage-inputs.json');roots=[Path(p).parent for p in bindings if Path(p).name=='Makefile']
    need(len(roots)==1,'Unique executed stage root');stage=roots[0];binaries={};source_count=0
    required={'src/myds_v41.c','src/myds_metal.m','metal/moe.metal','metal/probe_output_a_fixed.metal',
              'tools/outa_capture.inc','tools/capture_output_a.c','tools/bench_output_a_fixed.m',
              'tools/build_output_a_fixed_probe.py','tools/run_output_a_fixed_window.py'}
    required|={'tools/'+method['bench']+'.m','tools/'+method['runner']}
    source_names=set()
    with tarfile.open(source_archive,'r:*') as archive:
        for p,expected in bindings.items():
            item=Path(p)
            if not item.is_relative_to(stage):
                need(item.name=='protocol.json' and sha(root/'protocol.json')==expected,'Unexpected external frozen source input');continue
            relative=str(item.relative_to(stage))
            if relative.startswith('bin/'):binaries[item.name]=expected
            else:
                need(archive_file(archive,relative)==expected,'Executed source archive SHA differs: '+relative);source_count+=1;source_names.add(relative)
    need(required<=source_names,'Critical capture/wrapper/harness source is not bound to executed stage')
    need({'count_chat_tokens','capture_output_a',method['bench'],'myds.metallib'}<=set(binaries),'Actual four execution artifacts missing')
    with tarfile.open(artifacts_archive,'r:*') as archive:
        for name,expected in binaries.items():need(archive_file(archive,'bin/'+name)==expected,'Executed artifact archive SHA differs: '+name)
    need(manifest['binary_sha256']==binaries[method['bench']] and manifest['probe_metallib_sha256']==binaries['myds.metallib'] and
         capture['source']['capture_binary_sha256']==binaries['capture_output_a'] and
         capture['source']['capture_metallib_sha256']==binaries['myds.metallib'],'Capture/component binary and probe archive binding')
    executed_roots=[Path(p).parent for p in bindings if not Path(p).is_relative_to(stage) and Path(p).name=='protocol.json']
    need(len(executed_roots)==1,'Original window path must be bound by the frozen protocol input');original=executed_roots[0]
    steps=load(root/'steps.json');model_path=capture['source']['model_metadata']['path']
    for step in steps:
        name=step['name'];argv=step['argv'];expected=('count_chat_tokens' if name=='tokenize' else 'capture_output_a' if name=='capture' else method['bench'])
        need(argv[0]==str(stage/'bin'/expected),'Actual execution argv differs from frozen binary path')
        if name=='tokenize':need(argv[1:]==[model_path,str(original/'request.json'),str(original/'tokenized')],'Tokenizer model/request/output argv')
        elif name=='capture':need(argv[1:]==[model_path,'--tokens-file',str(original/'capture-tokens.txt'),'--out-dir',str(original/'raw-capture')],
                                  'Ordinary capture exact token/output argv')
        else:
            need(argv[1:9]==['--model',model_path,'--capture-json',str(original/'capture.json'),
                 '--probe-metallib',str(stage/'bin/myds.metallib'),'--production-metallib',str(original/'production-artifacts/myds.metallib')],
                 'Harness actual model/capture/production/probe argv')
            tail=argv[9:]
            need(tail[:2]==['--out-dir',str(original/name)],'Harness exact phase output argv')
            wanted=['--host-only'] if name=='host-validation' else ['--correctness-only'] if name=='correctness' else [
                '--correctness-manifest',str(original/'correctness/artifact-manifest.json'),'--correctness-manifest-sha256',
                sha(root/'correctness/artifact-manifest.json'),'--samples',str(method['samples']),'--reps',str(method['reps'])]
            need(tail[2:]==wanted,'Harness explicit phase/gate/repetitions argv')
    return dict(source_archive_sha256=sha(source_archive),source_members_verified=source_count,
                artifacts_archive_sha256=sha(artifacts_archive),executed_artifact_members_verified=len(binaries),
                executed_artifact_sha256=binaries,actual_tokenize_capture_harness_argv_bound=True)


def verify_window(root,correctness,performance,decision):
    method=measurement(root)
    before,restored,summary=[load(root/name) for name in ('before.json','restored.json','summary.json')]
    need(all(k in before and before[k]==restored[k] for k in IDENTITY_KEYS) and
         before['health']==restored['health']=={'status':'ok'},'Exact original production identity/health including nice not restored')
    need(summary['restoration']['restored'] is True and summary['restoration']['pid']==restored['pid'] and
         summary['restoration']['exact_argv_myds_artifacts_config'] is True and
         summary['fullmodel_run'] is False and summary['production_adoption'] is False,'Restoration/full-model scope summary')
    need(sha(root/'original-myds.conf')==before['config_sha256'] and load(root/'effective-env.json')==before['effective_myds_env'],
         'Original config and effective environment archive')
    for kind in ('binary','metallib'):
        need(sha(root/'production-artifacts'/Path(before[kind]['path']).name)==before[kind+'_sha256'],'Original production backup SHA: '+kind)
    need(correctness['production_metallib_sha256']==before['metallib_sha256'],'Native golden was not the archived actual production library')
    env=before['effective_myds_env']
    for k,v in {'MYDS_METAL_CONCURRENT':'1','MYDS_DENSE_Q4_NR0':'2','MYDS_DENSE_Q4_NSG':'4',
                'MYDS_ATTN_OUT_A_FUSE_BF16':'1','MYDS_ATTN_ROPE_FUSE':'1','MYDS_ATTN_PACKED_REGISTER':'1'}.items():
        need(env.get(k,v if k in ('MYDS_DENSE_Q4_NR0','MYDS_DENSE_Q4_NSG','MYDS_ATTN_OUT_A_FUSE_BF16') else None)==v,
             'Actual production Output_A/RF/concurrent provider differs: '+k)
    need(not any(k.startswith(('MYDS_TRACE','MYDS_SKIP_','MYDS_PROFILE','MYDS_KPROF','MYDS_GAP_PROF','MYDS_CAPTURE_')) for k in env),
         'Diagnostic/skip environment outside ordinary source scope')
    failed=decision=='reject_component_correctness';steps=load(root/'steps.json')
    need([r['name'] for r in steps]==(['tokenize','capture','host-validation','correctness'] if failed else
         ['tokenize','capture','host-validation','correctness','performance']) and
         [r['returncode'] for r in steps]==([0,0,0,1] if failed else [0,0,0,0,0]),'Actual execution step order/return codes')
    expected_status='component_rejected_correctness' if failed else ('component_passed_requires_fullmodel' if decision=='eligible_for_fullmodel_gate' else 'component_rejected_performance')
    need(summary['status']==expected_status and summary['correctness_pass'] is (not failed) and
         summary['performance_measured'] is (not failed),'Runner decision disagrees with raw gates')
    if not failed:
        passed=decision=='eligible_for_fullmodel_gate'
        need(summary['performance_gate_pass'] is passed and summary['timing_command_buffers']==method['cbs'] and
             summary['formal_command_buffers']==method['formal'] and summary['warmup_command_buffers']==18,'Runner timing gate accounting')
        need(len(summary['performance'])==6,'Runner six performance groups')
        for r in performance:
            pub=[p for p in summary['performance'] if (p['layer'],p['control_kind'])==(r['layer'],r['control_kind'])]
            need(len(pub)==1 and pub[0]['performance_gate_pass']==r['performance_gate_pass'] and
                 all(close(pub[0][k],r[k]) for k in ('a1_us','b_us','a2_us','control_drift_percent','conservative_time_saved_percent')),
                 'Runner performance values disagree with raw GPU times')
    events=rows(root/'events.jsonl')
    need(events and events[-1]['kind']=='window_complete' and events[-1]['status']==expected_status and
         not any(r['kind'] in ('monitor_failure','restoration_failure','window_failure') for r in events),'Window competition/restoration failure event')
    frozen=load(root/'protocol.json')
    expected=dict(scope=method['scope'],mode='concurrent',layers=list(LAYERS),positions=list(POSITIONS),
        cases=12,providers=3,samples=method['samples'],reps=method['reps'],projections_per_cb=method['projections'],control_symbol=NATIVE,candidate_symbol=FIXED,
        geometry=dict(groups=[128,1,8],threads=[32,4,1],threadgroup_bytes=256,nr=2,nsg=4),
        maximum_absolute_control_drift_percent=2.0,minimum_conservative_time_saved_percent_strict=2.0,
        fullmodel_run=False,production_adoption=False)
    need(all(frozen.get(k)==v for k,v in expected.items()),'Original frozen performance gate differs')
    return dict(restoration_exact_including_nice=True,original_pid=before['pid'],restored_pid=restored['pid'],
                original_production_binary_sha256=before['binary_sha256'],original_production_metallib_sha256=before['metallib_sha256'])


def review(root,model,source_archive,artifacts_archive):
    root=Path(root).resolve();model=Path(model).resolve(strict=True)
    method=measurement(root)
    cm,cn,ci=verify_manifest(root/'correctness');complete=cm['status']=='complete'
    verify_stdout_identity(root/'correctness.stdout',cm)
    need(cm['status'] in ('complete','failed'),'Expected completed correctness or retained numerical rejection')
    capture,cases=verify_capture(root,cm);weights,same_stat=verify_weights(root,cm,model)
    hm,hn,hi=verify_manifest(root/'host-validation','host_only')
    verify_stdout_identity(root/'host-validation.stdout',hm)
    need(hm['capture_json_sha256']==cm['capture_json_sha256'] and hm['binary_sha256']==cm['binary_sha256'] and
         hm['model']==cm['model'] and hm['tensors']==cm['tensors'] and 'device' not in hm,'Pre-GPU host phase identity')
    host_tail=rows(root/'host-validation.stdout')[-1]
    need(host_tail['kind']=='complete' and host_tail['host_only'] is True and host_tail['metal_device_created'] is False,'Host validation did not avoid a Metal device')
    capture_inputs=verify_component_inputs(root/'correctness',cm,cn,capture,complete)
    comparisons,bad=verify_correctness(root,cm,cn,complete)
    phase_reports=dict(correctness=dict(raw_logical_files_verified=len(cn),unique_inodes_hashed=ci,**capture_inputs))
    recomputed=[];timing_count=0;performance_pass=False
    if complete:
        pm,pn,pi=verify_manifest(root/'performance','complete')
        verify_stdout_identity(root/'performance.stdout',pm)
        extra=verify_component_inputs(root/'performance',pm,pn,capture,True)
        recomputed=verify_timing(root,pm,cm);timing_count=method['cbs']
        performance_pass=all(r['performance_gate_pass'] for r in recomputed if r['control_kind'] in method['required_controls'])
        primary=[r for r in recomputed if r['control_kind'] in method['required_controls']]
        decision=('eligible_for_fullmodel_gate' if performance_pass else
                  'reject_component_control_drift' if any(not r['control_drift_gate_pass'] for r in primary) else
                  'reject_component_performance')
        phase_reports['performance']=dict(raw_logical_files_verified=len(pn),unique_inodes_hashed=pi,**extra)
    else:decision='reject_component_correctness'
    restoration=verify_window(root,cm,recomputed,decision)
    archives=verify_archives(root,source_archive,artifacts_archive,cm,capture)
    return dict(schema_version=1,reviewer='independent_cpu_output_a_raw_and_timeline',reviewer_sha256=sha(Path(__file__)),evidence_valid=True,
        correctness_pass=complete,performance_gate_pass=performance_pass,decision=decision,
        decision_scope='Original per-window component protocol; the separately frozen multi-window replication policy is additional',
        single_report_proves_fullmodel_eligibility=False,required_controls_for_window_gate=list(method['required_controls']),
        gpu_work_submitted=False,fullmodel_run=False,production_adoption=False,cold_dram_proven=False,
        repeated_fixture_cache_reuse=True,capture_observed_layer_calls=640,capture_actual_rf_encoded=640,
        capture_scope='12 raw selected inputs; 640-call observer and RF/register counters bound to archived source/binary',
        cases=cases,actual_gguf_weight_intervals_verified=3,weights=weights,current_model_stat_matches_original_capture=same_stat,
        model_identity_scope='File size, first1MiB SHA, independent GGUF offsets and three complete weight-interval SHA; no whole-model SHA',
        full_output_f32_values_verified=len(comparisons)*8192,parity_provider_outputs_verified=len(comparisons),
        full_output_comparisons=comparisons,numerical_rejection=bad,phase_raw_evidence=phase_reports,
        measurement_method=method['name'],samples=method['samples'],repetitions_per_cb=method['reps'],projections_per_cb=method['projections'],
        timing_command_buffers_verified=timing_count,formal_command_buffers_verified=method['formal'] if complete else 0,
        warmup_command_buffers_verified=18 if complete else 0,timing_interleaved_a1_b_a2_verified=complete,
        all_six_groups_replication_gate_pass=complete and all(r['performance_gate_pass'] for r in recomputed),
        control_drift_rejection_present=any(not r['control_drift_gate_pass'] for r in recomputed),
        recomputed_performance=recomputed,**restoration,**archives,
        correctness_manifest_sha256=sha(root/'correctness/artifact-manifest.json'),
        correctness_stdout_sha256=sha(root/'correctness.stdout'),runner_summary_sha256=sha(root/'summary.json'),
        limitations=['Captured observer adds commit/wait and is correctness input evidence only.',
            'All 36 retained correctness outputs are re-compared in full; per-timing-CB output bodies are not retained.',
            'Timing output/input checks are bound to the executed harness and its per-CB records; raw timestamps are independently recomputed.',
            'A correctness rejection can stop before layer-tail input/weight guard dumps; missing guard bytes are explicitly counted.',
            'A component win qualifies a separate full-model gate and does not prove production adoption.'])


def self_test(steady=False):
    """Full synthetic CPU evidence and deliberately rehashed negative controls.

    Fake outputs/counters/timelines exercise this reader only. They must never
    be represented as model, GPU, production restoration, or quality evidence.
    """
    with tempfile.TemporaryDirectory(prefix='myds-outa-reader-cpu-') as temp:
        root=Path(temp);stage=root/'synthetic-stage';stage.mkdir();rawdir=root/'raw-capture';rawdir.mkdir()
        samples,reps,projections,cbs,formal=(15,256,1024,288,270) if steady else (9,8,32,180,162)
        bench_name='bench_output_a_fixed_steady' if steady else 'bench_output_a_fixed'
        runner_name='run_output_a_fixed_steady_window.py' if steady else 'run_output_a_fixed_window.py'
        def write(path,value):
            path.parent.mkdir(parents=True,exist_ok=True)
            if isinstance(value,(dict,list)):path.write_text(json.dumps(value))
            elif isinstance(value,str):path.write_text(value)
            else:path.write_bytes(value)
        def string(value):
            b=value.encode();return struct.pack('<Q',len(b))+b
        header=struct.pack('<IIQQ',0x46554747,3,3,0)
        for i,l in enumerate(LAYERS):header+=string(f'blk.{l}.attn_output_a.weight')+struct.pack('<IQQIQ',2,4096,8192,12,i*WEIGHT_BYTES)
        header+=b'\0'*(-len(header)%32);model=root/'synthetic-cpu-only.gguf'
        with model.open('wb') as f:f.write(header);f.truncate(len(header)+3*WEIGHT_BYTES)
        st=model.stat();model_metadata=dict(path=str(model),bytes=st.st_size,device=st.st_dev,inode=st.st_ino,mtime_ns=st.st_mtime_ns)
        model_identity=dict(path=str(model),bytes=st.st_size,header_bytes=1048576,header_sha256=sha(model,length=1048576),
            tensor_data_offset=len(header),device=st.st_dev,inode=st.st_ino,mtime_sec=st.st_mtime_ns//10**9,mtime_nsec=st.st_mtime_ns%10**9)
        env=dict(MYDS_METAL_CONCURRENT='1',MYDS_DENSE_Q4_NR0='2',MYDS_DENSE_Q4_NSG='4',
                 MYDS_ATTN_OUT_A_FUSE_BF16='1',MYDS_ATTN_ROPE_FUSE='1',MYDS_ATTN_PACKED_REGISTER='1')
        write(root/'effective-env.json',env);write(root/'request.json',dict(fixture_kind='synthetic_CPU_reader_control'))
        write(root/'capture-tokens.txt',' '.join(str(i) for i in range(16))+'\n');write(root/'tokenized.tokens.i32le.bin',struct.pack('<16i',*range(16)))
        provider=dict(schema_version=1,capture_kind='myds_decode_output_a',forward_tokens_completed=16,max_context=327680,
            public_forward_need_logits=True,ordinary_decode_allowed=True,trace_on=False,skip_flags_present=False,rope_mode=1,
            register_mode=1,rf_requests=640,rf_encoded=640,rf_failures=0,register_requests=640,register_encoded=640,
            register_failures=0,register_direct_encoded=640,register_staged_encoded=0,provider_contract_passed=True,performance_from_capture=False)
        observer=dict(status='complete',capture_kind='myds_decode_output_a',observed_layer_calls=640,rope_consumed_layer_calls=640,
            selected_cases=12,selected_mask=4095,forward_tokens=16,input_bytes_per_case=INPUT_BYTES,layers=list(LAYERS),positions=list(POSITIONS),
            all_f32_finite=True,all_bf16_bits=True,public_ordinary_decode=True,trace_on=False,performance_from_capture=False)
        write(rawdir/'capture-provider.json',provider);write(rawdir/'capture-observer-summary.json',observer)
        original=[]
        for p in POSITIONS:
            for l in LAYERS:
                name=f'layer-{l}-pos-{p}-input-f32.bin';data=struct.pack('<I',0x3f800000+l*0x10000)*32768;write(rawdir/name,data)
                original.append(dict(layer=l,position=p,call_ordinal=p*40+l+1,need_logits=True,ordinary_decode_allowed=True,
                    trace_on=False,attention_rope_consumed=True,compressed=l!=0,weight_type=12,boundary=BOUNDARY,
                    all_f32_finite=True,all_bf16_bits=True,observer_commit_wait=True,observer_batch_restored=True,performance_from_capture=False,
                    input_f32=dict(path=name,bytes=INPUT_BYTES,sha256=digest(data))))
        write(rawdir/'capture-records.jsonl',''.join(json.dumps(r)+'\n' for r in original))
        case_objects=[]
        for c in sorted(original,key=lambda r:(r['layer'],r['position'])):
            c=copy.deepcopy(c);c['input_f32']['path']=str(rawdir/c['input_f32']['path']);case_objects.append(c)
        critical=['Makefile','src/myds_v41.c','src/myds_metal.m','metal/moe.metal','metal/probe_output_a_fixed.metal',
                  'tools/outa_capture.inc','tools/capture_output_a.c','tools/bench_output_a_fixed.m',
                  'tools/build_output_a_fixed_probe.py','tools/run_output_a_fixed_window.py']
        if steady:critical+=['tools/'+bench_name+'.m','tools/'+runner_name]
        for n in critical:write(stage/n,'/* synthetic CPU reader fixture, never executable source */\n'+n)
        for n in ['count_chat_tokens','capture_output_a',bench_name,'myds.metallib']:write(stage/'bin'/n,('syntheticCPU:'+n).encode())
        capture=dict(schema_version=1,capture_kind='myds_decode_output_a',model_bytes=st.st_size,model_header_bytes=1048576,
            model_header_sha256=model_identity['header_sha256'],cases=case_objects,source=dict(kind='actual_public_ordinary_decode_post_inverse_rope_bf16',
                model_metadata=model_metadata,provider=provider,observer=observer,request_sha256=sha(root/'request.json'),
                token_file_sha256=sha(root/'capture-tokens.txt'),records_sha256=sha(rawdir/'capture-records.jsonl'),
                effective_environment_sha256=sha(root/'effective-env.json'),capture_binary_sha256=sha(stage/'bin/capture_output_a'),
                capture_metallib_sha256=sha(stage/'bin/myds.metallib'),trace_on=False,performance_from_capture=False))
        write(root/'capture.json',capture);write(root/'capture.stdout',json.dumps(dict(kind='capture_complete',cases=12,tokens=16,
            ordinary_decode=True,real_model_forward=True,performance_from_capture=False))+'\n')
        weight=b'\0'*WEIGHT_BYTES;weight_sha=digest(weight)
        tensors=[dict(layer=l,name=f'blk.{l}.attn_output_a.weight',type=12,dimensions=[4096,8192],bytes=WEIGHT_BYTES,
                      file_offset=len(header)+i*WEIGHT_BYTES,sha256=weight_sha) for i,l in enumerate(LAYERS)]
        case_ids=[dict(case=i,layer=c['layer'],position=c['position'],input_bytes=INPUT_BYTES,input_sha256=c['input_f32']['sha256'],
                      attention_rope_consumed=True,compressed=c['compressed'],boundary=BOUNDARY) for i,c in enumerate(case_objects)]
        config=b'synthetic CPU config\n';write(root/'original-myds.conf',config)
        write(root/'production-artifacts/myds-server',b'synthetic CPU original binary');write(root/'production-artifacts/myds.metallib',b'synthetic CPU original library')
        before=dict(pid=111,argv_raw_hex='00',myds_env_raw_hex='00',cwd='/synthetic',log='/synthetic/log',nice=0,
            binary=dict(path='/synthetic/myds-server'),binary_sha256=sha(root/'production-artifacts/myds-server'),
            metallib=dict(path='/synthetic/myds.metallib'),metallib_sha256=sha(root/'production-artifacts/myds.metallib'),
            config=dict(path='/synthetic/myds.conf'),config_sha256=digest(config),effective_myds_env=env,health=dict(status='ok'))
        restored=dict(before,pid=222);write(root/'before.json',before);write(root/'restored.json',restored)
        def manifest(phase,status):
            m=dict(schema_version=1,probe=PROBE,phase=phase,status=status,total_cases=12,samples=samples,repetitions_per_cb=reps,
                mode='concurrent',binding_contract=contract(),fullmodel=False,cold_dram_proven=False,repeated_fixture_cache_reuse=True,
                binary_sha256=sha(stage/'bin'/bench_name),probe_metallib_sha256=sha(stage/'bin/myds.metallib'),
                production_metallib_sha256=before['metallib_sha256'],capture_json_sha256=sha(root/'capture.json'),capture_source=capture['source'],
                model=model_identity,tensors=tensors,cases=case_ids,timing_command_buffers=0,files=[],parity=[],input_integrity=[])
            if phase!='host_validation':
                m['device']='syntheticCPU-no-device'
                m['bindings']=[dict(provider=p,symbol=FIXED if p==2 else NATIVE,library='production' if p==0 else 'probe',
                    contract=contract(),static_threadgroup_bytes=0,thread_execution_width=32,max_threads=512) for p in range(3)]
            return m
        hm=manifest('host_validation','host_only');cm=manifest('correctness','complete');pm=manifest('performance','complete')
        def artifact(directory,m,name,data):
            write(directory/name,data);m['files'].append(dict(path=name,bytes=len(data),sha256=digest(data)))
        for dirname,m in [('host-validation',hm),('correctness',cm),('performance',pm)]:
            directory=root/dirname;artifact(directory,m,'capture.json',(root/'capture.json').read_bytes());artifact(directory,m,'args136.bin',args136())
            for i,c in enumerate(case_objects):
                artifact(directory,m,f'capture-case-{i:03d}-input.f32',Path(c['input_f32']['path']).read_bytes())
                if dirname!='host-validation':artifact(directory,m,f'input-case-{i:03d}.guards',b'\xa5'*128)
            if dirname!='host-validation':
                for l in LAYERS:
                    artifact(directory,m,f'weight-layer-{l}-output-a.q4k',weight)
                    artifact(directory,m,f'weight-layer-{l}-output-a.guards',b'\xa5'*128)
        write(root/'host-validation/artifact-manifest.json',hm)
        write(root/'host-validation.stdout',json.dumps(dict(kind='identity',manifest=hm))+'\n'+
              json.dumps(dict(kind='complete',host_only=True,metal_device_created=False))+'\n')
        def input_record(case,p,ordinal):return dict(command_ordinal=ordinal,case=case,provider=p,input_sha256=case_ids[case]['input_sha256'],
            expected_input_sha256=case_ids[case]['input_sha256'],weight_sha256=weight_sha,expected_weight_sha256=weight_sha,
            input_guards_intact=True,weight_guards_intact=True,all_gpu_inputs_unchanged=True)
        correct_rows=[dict(kind='identity',manifest=copy.deepcopy(cm)),dict(kind='bindings',bindings=cm['bindings'],device=cm['device'])]
        for li,l in enumerate(LAYERS):
            for p in range(3):
                for c in range(li*4,li*4+4):
                    data=struct.pack('<I',0x3f800000+c*0x10000)*8192;prefix=f'parity-case-{c:03d}-provider-{p}-output'
                    artifact(root/'correctness',cm,prefix+'.f32',data);artifact(root/'correctness',cm,prefix+'.guards',b'\xa5'*128)
                    r=dict(kind='parity',phase='correctness',mode='concurrent',case=c,layer=l,position=case_ids[c]['position'],provider=p,
                        command_ordinal=li*3+p+1,output_bytes=OUTPUT_BYTES,mismatches=0,first_mismatch=None,output_sha256=digest(data),
                        full_f32_bit_exact=True,finite=True,written=True,bf16_postround=True,guards_intact=True,finite_guards_written=True,
                        all_gpu_inputs_unchanged=True,completed_cb=True)
                    correct_rows.append(r);cm['parity'].append(r);cm['input_integrity'].append(input_record(c,p,li*3+p+1))
        cm.update(full_f32_bit_exact=True,output_a_checked=True,all_gpu_inputs_unchanged=True)
        correct_rows.append(dict(kind='complete',phase='correctness',cases=12,providers=3,timing_command_buffers=0,output_a_checked=True,full_f32_bit_exact=True))
        write(root/'correctness.stdout',''.join(json.dumps(r)+'\n' for r in correct_rows));write(root/'correctness/artifact-manifest.json',cm)
        perf_rows=[dict(kind='identity',manifest=copy.deepcopy(pm)),dict(kind='bindings',bindings=pm['bindings'],device=pm['device'])];clock=10.;ordinal=0;performance=[]
        for li,l in enumerate(LAYERS):
            for control in CONTROLS:
                values={a:[] for a in ARMS}
                for s in range(samples+1):
                    for arm in ARMS:
                        p=2 if arm=='B' else (0 if control=='production' else 1);us=90. if p==2 else 100.
                        start=clock;end=start+us*projections/1e6;clock=end+0.001;actual=(end-start)*1e6/projections;ordinal+=1
                        if s:values[arm].append(actual)
                        perf_rows.append(dict(kind='timing',layer=l,control_kind=control,arm=arm,sample=s,provider=p,symbol=FIXED if p==2 else NATIVE,
                            command_ordinal=ordinal,mode='concurrent',cases_in_cb=4,repetitions=reps,dispatches=projections,projections_per_cb=projections,commits=1,
                            warmup=s==0,gpu_start_s=start,gpu_end_s=end,gpu_us_per_projection=actual,completed_cb=True,full_f32_bit_exact=True,
                            all_gpu_inputs_unchanged=True,component_only=True))
                        pm['input_integrity'].extend(input_record(c,p,ordinal) for c in range(li*4,li*4+4))
                a1,b,a2=[statistics.median(values[a]) for a in ARMS];r=dict(layer=l,control_kind=control,a1_us=a1,b_us=b,a2_us=a2,
                    control_drift_percent=(a2/a1-1)*100,time_saved_percent=(1-b/((a1+a2)/2))*100,conservative_time_saved_percent=(1-b/min(a1,a2))*100)
                performance.append(dict(r,performance_gate_pass=True));perf_rows.append(dict(r,kind='summary',mode='concurrent',control_symbol=NATIVE,candidate_symbol=FIXED))
        pm.update(full_f32_bit_exact=True,output_a_checked=True,all_gpu_inputs_unchanged=True,timing_command_buffers=cbs,
                  correctness_manifest_sha256=sha(root/'correctness/artifact-manifest.json'),prior_correctness_providers_verified=36)
        artifact(root/'performance',pm,'correctness-manifest.json',(root/'correctness/artifact-manifest.json').read_bytes())
        perf_rows.append(dict(kind='complete',phase='performance',cases=12,providers=3,timing_command_buffers=cbs,output_a_checked=True,full_f32_bit_exact=True))
        write(root/'performance.stdout',''.join(json.dumps(r)+'\n' for r in perf_rows));write(root/'performance/artifact-manifest.json',pm)
        summary=dict(status='component_passed_requires_fullmodel',correctness_pass=True,performance_measured=True,performance_gate_pass=True,
            performance=performance,timing_command_buffers=cbs,formal_command_buffers=formal,warmup_command_buffers=18,fullmodel_run=False,production_adoption=False,
            model=model_metadata,model_header_sha256=model_identity['header_sha256'],restoration=dict(restored=True,pid=222,exact_argv_myds_artifacts_config=True))
        write(root/'summary.json',summary)
        common_argv=[str(stage/'bin'/bench_name),'--model',str(model),'--capture-json',str(root/'capture.json'),
            '--probe-metallib',str(stage/'bin/myds.metallib'),'--production-metallib',str(root/'production-artifacts/myds.metallib')]
        steps=[dict(name='tokenize',returncode=0,argv=[str(stage/'bin/count_chat_tokens'),str(model),str(root/'request.json'),str(root/'tokenized')]),
            dict(name='capture',returncode=0,argv=[str(stage/'bin/capture_output_a'),str(model),'--tokens-file',str(root/'capture-tokens.txt'),'--out-dir',str(rawdir)])]
        for phase in ('host-validation','correctness','performance'):
            tail=['--host-only'] if phase=='host-validation' else ['--correctness-only'] if phase=='correctness' else [
                '--correctness-manifest',str(root/'correctness/artifact-manifest.json'),'--correctness-manifest-sha256',sha(root/'correctness/artifact-manifest.json'),
                '--samples',str(samples),'--reps',str(reps)]
            steps.append(dict(name=phase,returncode=0,argv=common_argv+['--out-dir',str(root/phase)]+tail))
        write(root/'steps.json',steps)
        write(root/'events.jsonl',json.dumps(dict(kind='window_complete',status=summary['status']))+'\n')
        frozen=dict(scope='ordinary_decode_output_a_fixed_wrapper_steady' if steady else 'ordinary_decode_output_a_fixed_wrapper',
            mode='concurrent',layers=list(LAYERS),positions=list(POSITIONS),cases=12,providers=3,
            samples=samples,reps=reps,projections_per_cb=projections,control_symbol=NATIVE,candidate_symbol=FIXED,geometry=dict(groups=[128,1,8],threads=[32,4,1],threadgroup_bytes=256,nr=2,nsg=4),
            maximum_absolute_control_drift_percent=2.0,minimum_conservative_time_saved_percent_strict=2.0,fullmodel_run=False,production_adoption=False)
        if steady:frozen.update(measurement_method='separate_steady_cb_1024_projections',required_control_kinds=list(CONTROLS))
        write(root/'protocol.json',frozen)
        bindings={str(p):sha(p) for p in stage.rglob('*') if p.is_file()};bindings[str(root/'protocol.json')]=sha(root/'protocol.json');write(root/'stage-inputs.json',bindings)
        sa,ba=root/'source.tar.gz',root/'artifacts.tar.gz'
        for path,binaries in [(sa,False),(ba,True)]:
            with tarfile.open(path,'w:gz') as archive:
                for p in stage.rglob('*'):
                    if p.is_file() and str(p.relative_to(stage)).startswith('bin/')==binaries:archive.add(p,arcname=str(p.relative_to(stage)),recursive=False)
        result=review(root,model,sa,ba);need(result['performance_gate_pass'],'Valid complete CPU fixture rejected')
        negative=[]
        def reject(name,change,restore):
            change()
            try:review(root,model,sa,ba)
            except (ValueError,KeyError,TypeError) as exc:negative.append(dict(case=name,rejected=True,reason=str(exc)))
            else:raise ValueError('Negative CPU reader control accepted: '+name)
            finally:restore()
        def save_cm():write(root/'correctness/artifact-manifest.json',cm)
        target=root/'correctness/parity-case-000-provider-2-output.f32';original_data=target.read_bytes()
        reject('raw_SHA_corruption',lambda:write(target,b'\1'+original_data[1:]),lambda:write(target,original_data))
        def rehash(name,data):
            write(root/'correctness'/name,data)
            for r in cm['files']:
                if r['path']==name:r['sha256']=digest(data);r['bytes']=len(data)
            save_cm()
        reject('full_output_difference_after_rehash',lambda:rehash(target.name,struct.pack('<I',0x40000000)+original_data[4:]),lambda:rehash(target.name,original_data))
        guard='input-case-000.guards'
        reject('input_guard_corruption_after_rehash',lambda:rehash(guard,b'\0'+b'\xa5'*127),lambda:rehash(guard,b'\xa5'*128))
        guard='weight-layer-0-output-a.guards'
        reject('weight_guard_corruption_after_rehash',lambda:rehash(guard,b'\0'+b'\xa5'*127),lambda:rehash(guard,b'\xa5'*128))
        def save_perf():write(root/'performance.stdout',''.join(json.dumps(r)+'\n' for r in perf_rows))
        t=next(r for r in perf_rows if r.get('kind')=='timing');saved=t['gpu_us_per_projection']
        reject('projection_normalization_tampered',lambda:(t.update(gpu_us_per_projection=saved+1),save_perf()),lambda:(t.update(gpu_us_per_projection=saved),save_perf()))
        first=perf_rows[2];second=perf_rows[3]
        reject('strict_ABA_order_changed',lambda:(perf_rows.__setitem__(2,second),perf_rows.__setitem__(3,first),save_perf()),
               lambda:(perf_rows.__setitem__(2,first),perf_rows.__setitem__(3,second),save_perf()))
        s=next(r for r in perf_rows if r.get('kind')=='summary');saved=s['b_us']
        reject('median_tampered',lambda:(s.update(b_us=saved+1),save_perf()),lambda:(s.update(b_us=saved),save_perf()))
        old=dict(restored)
        reject('restoration_nice_changed',lambda:write(root/'restored.json',dict(restored,nice=1)),lambda:write(root/'restored.json',old))
        old=provider['rf_encoded']
        reject('actual_RF_counter_changed',lambda:(provider.update(rf_encoded=639),write(rawdir/'capture-provider.json',provider)),
               lambda:(provider.update(rf_encoded=old),write(rawdir/'capture-provider.json',provider)))
        reject('unsupported_measurement_tuple',lambda:write(root/'protocol.json',dict(frozen,samples=samples+1)),lambda:write(root/'protocol.json',frozen))
        reject('short_steady_scope_mixed',lambda:write(root/'protocol.json',dict(frozen,scope='ordinary_decode_output_a_fixed_wrapper' if steady else
               'ordinary_decode_output_a_fixed_wrapper_steady')),lambda:write(root/'protocol.json',frozen))
        secondary_gate_rejection_accepted=False
        if steady:
            reject('steady_both_controls_missing',lambda:write(root/'protocol.json',dict(frozen,required_control_kinds=['production'])),
                   lambda:write(root/'protocol.json',frozen))
            baseline_perf_rows=copy.deepcopy(perf_rows);baseline_summary=copy.deepcopy(summary)
            clock=10.
            for r in perf_rows:
                if r.get('kind')!='timing':continue
                us=105. if (r['layer'],r['control_kind'],r['arm'])==(39,'probe','B') else r['gpu_us_per_projection']
                start=clock;end=start+us*projections/1e6;clock=end+0.001
                r.update(gpu_start_s=start,gpu_end_s=end,gpu_us_per_projection=(end-start)*1e6/projections)
            new_performance=[]
            for s in perf_rows:
                if s.get('kind')!='summary':continue
                group=[r for r in perf_rows if r.get('kind')=='timing' and r['layer']==s['layer'] and
                       r['control_kind']==s['control_kind'] and not r['warmup']]
                a1,b,a2=[statistics.median(r['gpu_us_per_projection'] for r in group if r['arm']==a) for a in ARMS]
                s.update(a1_us=a1,b_us=b,a2_us=a2,control_drift_percent=(a2/a1-1)*100,
                         time_saved_percent=(1-b/((a1+a2)/2))*100,conservative_time_saved_percent=(1-b/min(a1,a2))*100)
                new_performance.append({**{k:v for k,v in s.items() if k not in ('kind','mode','control_symbol','candidate_symbol')},
                                        'performance_gate_pass':abs(s['control_drift_percent'])<=2 and s['conservative_time_saved_percent']>2})
            summary.update(status='component_rejected_performance',performance_gate_pass=False,performance=new_performance)
            save_perf();write(root/'summary.json',summary);write(root/'events.jsonl',json.dumps(dict(kind='window_complete',status=summary['status']))+'\n')
            secondary=review(root,model,sa,ba)
            need(not secondary['performance_gate_pass'] and all(r['performance_gate_pass'] for r in secondary['recomputed_performance']
                 if r['control_kind']=='production') and secondary['decision']=='reject_component_performance',
                 'Steady probe-only failure incorrectly accepted because production primary passed')
            secondary_gate_rejection_accepted=True
            reject('steady_wrong_primary_only_summary',
                lambda:(write(root/'summary.json',dict(summary,status='component_passed_requires_fullmodel',performance_gate_pass=True)),
                        write(root/'events.jsonl',json.dumps(dict(kind='window_complete',status='component_passed_requires_fullmodel'))+'\n')),
                lambda:(write(root/'summary.json',summary),write(root/'events.jsonl',json.dumps(dict(kind='window_complete',status=summary['status']))+'\n')))
            perf_rows[:]=baseline_perf_rows;summary.clear();summary.update(baseline_summary);save_perf();write(root/'summary.json',summary)
            write(root/'events.jsonl',json.dumps(dict(kind='window_complete',status=summary['status']))+'\n')
        # A numerical rejection is valid evidence, with no timing permitted.
        # Match the real early-stop shape: native/probe four cases each, then
        # fixed provider's first case; later weight/guard dumps are absent.
        failed=copy.deepcopy(cm);failed.update(status='failed',failure='Output_A full F32 bits/BF16/finite/poison/guards and input identity')
        for k in ('full_f32_bit_exact','output_a_checked','all_gpu_inputs_unchanged'):failed.pop(k,None)
        failed['parity']=copy.deepcopy(cm['parity'][:9]);failed['input_integrity']=copy.deepcopy(cm['input_integrity'][:9])
        last=failed['parity'][-1];changed=struct.pack('<I',0x3f810000)+original_data[4:];write(target,changed)
        last.update(mismatches=1,first_mismatch=0,output_sha256=digest(changed),full_f32_bit_exact=False)
        retained={'capture.json','args136.bin','weight-layer-0-output-a.q4k'}|{f'capture-case-{i:03d}-input.f32' for i in range(12)}
        retained|={f'parity-case-{r["case"]:03d}-provider-{r["provider"]}-output.{suffix}' for r in failed['parity'] for suffix in ('f32','guards')}
        failed['files']=[r for r in failed['files'] if r['path'] in retained]
        for r in failed['files']:
            if r['path']==target.name:r['sha256']=digest(changed)
        write(root/'correctness/artifact-manifest.json',failed)
        failed_rows=correct_rows[:2]+failed['parity'];write(root/'correctness.stdout',''.join(json.dumps(r)+'\n' for r in failed_rows))
        (root/'performance').rename(root/'held-performance');(root/'performance.stdout').rename(root/'held-performance.stdout')
        fail_summary=dict(summary,status='component_rejected_correctness',correctness_pass=False,performance_measured=False)
        write(root/'summary.json',fail_summary);steps=steps[:-1];steps[-1]['returncode']=1;write(root/'steps.json',steps)
        write(root/'events.jsonl',json.dumps(dict(kind='window_complete',status=fail_summary['status']))+'\n')
        rejection=review(root,model,sa,ba)
        need(rejection['decision']=='reject_component_correctness' and rejection['parity_provider_outputs_verified']==9 and
             rejection['timing_command_buffers_verified']==0 and rejection['numerical_rejection'][0]['mismatches']==1 and
             not rejection['phase_raw_evidence']['correctness']['complete_input_and_weight_guards_raw_verified'],
             'Valid retained CPU numerical rejection rejected or overstated')
        reject('timing_after_numerical_rejection',lambda:(root/'held-performance').rename(root/'performance'),
               lambda:(root/'performance').rename(root/'held-performance'))
        return dict(schema_version=1,passed=True,fixture_kind='synthetic_CPU_protocol_evidence_only_not_GPU_or_quality',
                    measurement_method='steady1024' if steady else 'original32',
                    gpu_work_submitted=False,valid_full_fixture_accepted=True,valid_retained_numerical_rejection_accepted=True,
                    steady_secondary_only_rejection_accepted=secondary_gate_rejection_accepted if steady else None,
                    negative_controls=negative)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','model','source-archive','artifacts-archive','out'):parser.add_argument('--'+name,type=Path)
    parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    try:
        if args.self_test:
            need(not any((args.root,args.model,args.source_archive,args.artifacts_archive)),'Self-test is separate from actual evidence review')
            short=self_test(False);steady=self_test(True)
            result=dict(schema_version=1,passed=True,gpu_work_submitted=False,
                        fixture_kind='synthetic_CPU_protocol_evidence_only_not_GPU_or_quality',methods=[short,steady])
        else:
            need(all((args.root,args.model,args.source_archive,args.artifacts_archive)),'--root --model --source-archive --artifacts-archive required')
            result=review(args.root,args.model,args.source_archive,args.artifacts_archive)
        rc=0
    except (ValueError,KeyError,TypeError,OSError,tarfile.TarError,struct.error) as error:
        result=dict(schema_version=1,evidence_valid=False,gpu_work_submitted=False,error=str(error));rc=2
    text=json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n'
    if args.out:
        with args.out.open('x') as stream:stream.write(text)
    print(text,end='');return rc


if __name__=='__main__':sys.exit(main())



