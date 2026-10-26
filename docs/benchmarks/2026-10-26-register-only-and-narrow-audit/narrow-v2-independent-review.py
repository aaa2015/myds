#!/usr/bin/env python3
"""Independent saved-evidence review; does not import or run the audit tool."""
import copy, hashlib, json, math, re, struct, tarfile
from pathlib import Path, PurePosixPath

BASE=Path('/private/tmp/myds-narrow-cpu-tool-v2-20261026/narrow-v2')
ARCHIVE=BASE.parent/'narrow-v2-complete.tar.gz'
DURABLE=Path('/Users/fangqingyuan/Projects/myds/docs/benchmarks/2026-10-26-register-only-and-narrow-audit')
REMOTE='/private/tmp/myds-register-only-20261026/narrow-v2'
OUT=Path(__file__).resolve().parent
PINNED={
 'tools/audit_v41_narrow_weights.py':'f1e9a600b732a26f01042babe0b5fe4b672a2842196d0f60c3f3c042d3cdcc55',
 'tools/v41_narrow_weights_worker.c':'712f41c2d89d9835bf2e0d4ebaf6db1eab02df3233eb73cd7f7281cc856e1fa7',
 'tests/test_v41_narrow_weights_audit.py':'72d9039a2dec0d6e12a00ea77ace2cbf73be0c730b020175aaca6f8ed2dcfa35'}
PATCHES={
 '7d9903b2ed6bd74c14aed06a89a591a14d613e9e':'c64a447efd6b41f815ba48be37db742da43cd0dfa1a37509c275ee427ef9db6a',
 '55b1e75e64ec66b458e5cd5cf0944bec20590ae8':'17883c1499b92b18bf02f47d9a886a77a1b647acb74357dd769185613aef37fb'}
INDEX=[2,8,14,20,24,28,32,36]
SPEC=([('router',f'blk.{i}.ffn_gate_inp.weight',[5120,384],0) for i in range(40)] +
      [('index_qb',f'blk.{i}.indexer.attn_q_b.weight',[1280,4096],1) for i in INDEX] +
      [('engram',f'blk.{i}.engram_kv.weight',[6144,25600],1) for i in (1,14)])
ZERO_COUNTS='failed_blocks values_without_accepted_block unencodable_at_all_scales roundtrip_mismatches shader_f32_mismatches infinity nan failed_scale_trials invalid_scales under_witness_scale_mismatch_count blocks_without_finite_witness_scale'.split()

def need(value,message):
    if not value: raise ValueError(message)
def digest(data): return hashlib.sha256(data).hexdigest()
def descriptor(path):
    p=Path(path);raw=p.read_bytes()
    return {'path':str(p),'bytes':len(raw),'sha256':digest(raw)}
def hashed_string(value): return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value)
def mapped(remote):
    need(remote.startswith(REMOTE+'/'),'descriptor outside run root')
    rel=PurePosixPath(remote[len(REMOTE)+1:])
    need(not rel.is_absolute() and '..' not in rel.parts,'unsafe descriptor')
    return BASE.joinpath(*rel.parts)
def verify_descriptor(d):
    p=mapped(d['path']);raw=p.read_bytes()
    need(len(raw)==d['bytes'] and digest(raw)==d['sha256'],'descriptor mismatch: '+str(p))
    return descriptor(p)

def check_report(r):
    need(r['schema_version']==1 and r['kind']=='v41_narrow_weights_CPU_bit_reversibility_audit','wrong report schema/kind')
    need(r['qualification_passed'] is True and r['audit_completed'] is True and r['exit_code']==0 and r['errors']==[],'report not completed/pass')
    need(r['GPU_work_submitted'] is False and r['model_modified'] is False and r['arithmetic_equivalence_or_performance_proven'] is False,'scope inflation')
    need(r['full_model_sha256'] is None,'must not claim full model hash')
    expected=[{'family':f,'name':n,'dims':d,'source_type':'F32' if t==0 else 'F16'} for f,n,d,t in SPEC]
    need(r['expected_targets']==expected,'expected target list mismatch')
    ts=r['tensors'];need(len(ts)==50 and len({t['name'] for t in ts})==50,'incomplete/duplicate target set')
    header=r['model']['headers'][0]
    need(r['model']['kind']=='single' and len(r['model']['headers'])==1,'unexpected model split mode')
    need(header['version']==3 and header['alignment']==16384 and header['data_offset']==5832704,'actual header identity mismatch')
    need(header['tensor_count']==1046 and header['metadata_count']==67,'actual header counts mismatch')
    need(len(header['metadata_value_identities'])==67 and hashed_string(header['header_sha256']),'incomplete header identities')
    m,mt,mi=header['metadata'],header['metadata_types'],header['metadata_value_identities']
    need(m['general.architecture']=='deepseek41' and mt['general.architecture']==8,'architecture mismatch')
    sb=r['scope_binding']
    need(sb['architecture']=={'key':'general.architecture','value':'deepseek41','gguf_type':8},'architecture binding mismatch')
    for label,key,n in [('layers','deepseek41.num_hidden_layers',40),('width','deepseek41.hidden_size',5120)]:
        expected_sha=digest(struct.pack('<I',n)); b=sb[label]
        need(m[key]==n and mt[key]==4 and mi[key]['sha256']==expected_sha and mi[key]['bytes']==4,'actual dimension metadata identity')
        need(b['value']==n and b['selected_key']==key and b['selected_gguf_type']==4,'selected scope key/type mismatch')
        need(b['present_aliases']==[{'key':key,'value':n,'gguf_type':4,'metadata_value_sha256':expected_sha}],'alias binding mismatch')
    for key,values in [('deepseek41.index_source_layer_ids',INDEX),('deepseek41.engram.layer_ids',[1,14]),('deepseek41.kv_source_layer_ids',[2,8,14,20])]:
        raw=struct.pack('<IQ',4,len(values))+struct.pack('<'+'I'*len(values),*values)
        need(m[key]==values and mt[key]==9 and header['metadata_array_types'][key]==4,'source layer IDs mismatch')
        need(mi[key]['sha256']==digest(raw) and mi[key]['bytes']==len(raw),'source layer IDs raw identity mismatch')
    ids=r['model_file_identity'];need(len(ids)==1,'unexpected model file count')
    identity=ids[0];need(identity['unchanged'] is True and identity['before']==identity['after_fd']==identity['after_path'],'model stat identity changed')
    model_size=identity['before']['st_size'];need(model_size==362142629888,'wrong actual model size')
    totals={};families={};spans=[]
    for (family,name,dims,ttype),t in zip(SPEC,ts):
        elements=math.prod(dims);width=4 if ttype==0 else 2;blocks=0 if ttype==0 else elements//1024
        need((t['family'],t['name'],t['dims'],t['type'])==(family,name,dims,ttype),'actual name/shape/type mismatch')
        need(t['expected_dims']==dims and t['expected_type']==('F32' if ttype==0 else 'F16') and t['actual_type_name']==t['expected_type'],'expected/actual type fields disagree')
        need(all(t[k] is True for k in ('qualification_passed','bits_reversible','finite_weights','streams_form_valid_complete_representation')),'tensor not independently marked finite/reversible')
        need(t['qualification_rejection_reasons']==[] and t['first_failure'] is None and t['first_nonfinite'] is None and 'gate_error' not in t,'hidden tensor failure')
        nbytes=elements*width;code_bytes=elements*(2 if ttype==0 else 1);scale_bytes=blocks*4
        need(t['bytes']==t['source_bytes_scanned']==nbytes and t['code_payload_bytes']==code_bytes and t['scale_payload_bytes']==scale_bytes,'tensor byte budget mismatch')
        need(t['part_index']==0 and t['file_offset']==header['data_offset']+t['offset'] and t['offset']%header['alignment']==0,'tensor span/alignment mismatch')
        need(header['data_offset']<=t['file_offset'] and t['file_offset']+nbytes<=model_size,'tensor outside actual model file')
        spans.append((t['file_offset'],t['file_offset']+nbytes,name))
        c=t['counts'];need(all(type(v) is int and v>=0 for v in c.values()),'bad count types')
        need(c['elements']==elements and c['blocks']==blocks,'counted shape mismatch')
        need(all(c[k]==0 for k in ZERO_COUNTS),'nonzero failure or nonfinite counter')
        need(sum(c[k] for k in ('positive_zero','negative_zero','subnormal','normal','infinity','nan'))==elements,'classification not exhaustive')
        need(c['witness_scale_values_tested']==(0 if ttype==0 else elements),'witness did not cover full projection')
        for k in ('source_sha256','reconstructed_stream_sha256','codes_stream_sha256'): need(hashed_string(t[k]),'malformed tensor SHA')
        need(t['source_sha256']==t['reconstructed_stream_sha256'],'source/reconstructed SHA differ')
        if ttype==0:
            need(t['scales_stream_sha256'] is None and t['block_valid_mask_sha256'] is None and t['valid_scales'] is None,'router carries FP8-only streams')
        else:
            need(hashed_string(t['scales_stream_sha256']),'invalid scale SHA')
            need(t['block_valid_mask_sha256']==digest(b'\1'*blocks),'all-valid block mask SHA mismatch')
            scales=t['valid_scales'];need(scales['count']==blocks,'wrong accepted scale count')
            for value in (scales['min'],scales['max']): need(math.isfinite(value) and value>0 and math.frexp(value)[0]==0.5,'scale not finite positive power of two')
            need(scales['min']<=scales['max'],'scale bounds inverted')
        families.setdefault(family,{'records':[],'source_bytes':0,'code_bytes':0,'scale_bytes':0,'counts':{}})
        f=families[family];f['records'].append(t);f['source_bytes']+=nbytes;f['code_bytes']+=code_bytes;f['scale_bytes']+=scale_bytes
        for k,v in c.items(): f['counts'][k]=f['counts'].get(k,0)+v;totals[k]=totals.get(k,0)+v
    ordered=sorted(spans)
    need(all(a[1]<=b[0] for a,b in zip(ordered,ordered[1:])),'target spans overlap')
    result={}
    for family,f in families.items():
        actual=r['families'][family];records=f['records']
        need(actual['tensor_count']==actual['qualified_tensor_count']==len(records) and actual['all_tensors_qualify'] is True,'family qualification mismatch')
        need(actual['counts']==f['counts'] and actual['source_bytes_scanned']==f['source_bytes'],'family count sum mismatch')
        need(actual['first_rejection'] is None and actual['gate_failures']==[],'family hidden reject')
        need(actual['potential_code_payload_bytes']==actual['qualified_code_payload_bytes']==f['code_bytes'],'family code budget mismatch')
        need(actual['potential_scale_payload_bytes']==actual['qualified_scale_payload_bytes']==f['scale_bytes'],'family scale budget mismatch')
        manifest=[{'name':t['name'],'type':t['type'],'dims':t['dims'],'source_sha256':t['source_sha256']} for t in records]
        need(actual['source_tensor_manifest_sha256']==digest(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()),'family source manifest SHA mismatch')
        result[family]={'tensor_count':len(records),'actual_source_type':'F32' if family=='router' else 'F16','source_bytes':f['source_bytes'],'code_bytes':f['code_bytes'],'scale_bytes':f['scale_bytes'],'elements':f['counts']['elements'],'source_tensor_manifest_sha256':actual['source_tensor_manifest_sha256'],'counts':f['counts']}
    return {'families':result,'totals':totals,'source_bytes':sum(f['source_bytes'] for f in families.values()),'code_bytes':sum(f['code_bytes'] for f in families.values()),'scale_bytes':sum(f['scale_bytes'] for f in families.values()),'matched_source_reconstructed_hashes':50,'target_span_overlap':False,'scope_binding':sb}

def encoder_body(text):
    s=text.index('{',text.index('bool ds41_e4m3_code('));depth=0
    for i in range(s,len(text)):
        depth+=(text[i]=='{')-(text[i]=='}')
        if depth==0:return text[s:i+1]
    raise ValueError('missing encoder brace')

def main():
    r=json.loads((BASE/'audit.json').read_text());result=check_report(r)
    verified=[]
    for d in json.loads((BASE/'delivery.json').read_text())['files']:
        p=BASE/'source'/d['path'];raw=p.read_bytes()
        need(len(raw)==d['bytes'] and digest(raw)==d['sha256'],'delivery input differs')
        expected=PINNED.get(d['path'])
        if expected: need(d['sha256']==expected,'source not frozen v2')
        else: need(d['sha256']==PATCHES[p.stem],'patch not pinned original')
        verified.append(descriptor(p))
    manifest=json.loads((BASE/'artifacts/artifact-manifest.json').read_text())
    need(manifest['GPU_work_submitted'] is False,'artifact scope wrong')
    declared=set()
    for d in manifest['files']:
        rel=PurePosixPath(d['relative_path']);need(not rel.is_absolute() and '..' not in rel.parts,'unsafe artifact name')
        need(str(rel) not in declared,'duplicate artifact');declared.add(str(rel))
        p=BASE/'artifacts'/rel;raw=p.read_bytes();need(len(raw)==d['bytes'] and digest(raw)==d['sha256'],'artifact byte identity mismatch')
        verified.append(descriptor(p))
    actual={str(p.relative_to(BASE/'artifacts')) for p in (BASE/'artifacts').rglob('*') if p.is_file() and p.name!='artifact-manifest.json'}
    need(actual==declared and len(actual)==9,'artifact inventory incomplete/extra')
    verify_descriptor(r['artifact_manifest']);verify_descriptor(r['tool'])
    w=r['worker']
    for k in ('source','original_source','binary','compile_command','compile_stdout','compile_stderr'):verify_descriptor(w[k])
    need(w['source']['sha256']==w['original_source']['sha256']==PINNED['tools/v41_narrow_weights_worker.c'],'worker not frozen source')
    for d in w['upstream_patches']:
        verify_descriptor(d);need(d['sha256']==PATCHES[d['commit']],'worker patch identity wrong')
    worker_source=(BASE/'artifacts/v41_narrow_weights_worker.c').read_text()
    patch=(BASE/'artifacts/upstream-commits/7d9903b2ed6bd74c14aed06a89a591a14d613e9e.patch').read_text()
    added='\n'.join(line[1:] for line in patch.splitlines() if line.startswith('+') and not line.startswith('+++'))
    need(encoder_body(worker_source)==encoder_body(added),'actual worker encoder differs from pinned official body')
    need(digest(encoder_body(added).encode())==w['encoder_body_sha256'],'encoder body SHA wrong')
    command=json.loads((BASE/'artifacts/compile-command.json').read_text())
    need(command['argv']==w['argv'] and command['environment']==w['compile_environment'],'compile command/env not bound')
    expected_command=['/usr/bin/clang','-std=c11','-O2','-fno-fast-math','-ffp-contract=off','-fPIC','-dynamiclib',REMOTE+'/artifacts/v41_narrow_weights_worker.c','-o',REMOTE+'/artifacts/worker.dylib','-lm']
    need(command['argv']==expected_command,'unexpected compiler arguments')
    need(command['environment']['DEVELOPER_DIR']=='/Applications/Xcode.app/Contents/Developer','unbound Xcode selection')
    need(command['environment']['SDKROOT']=='/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk','actual CLT SDKROOT not recorded')
    need((BASE/'artifacts/compile-stdout.txt').read_bytes()==(BASE/'artifacts/compile-stderr.txt').read_bytes()==b'','compiler diagnostics not empty')
    need((BASE/'artifacts/compiler-version.txt').read_text()==w['compiler_version'],'compiler version file not bound')
    binary=(BASE/'artifacts/worker.dylib').read_bytes();h=struct.unpack_from('<8I',binary);need(h[0]==0xfeedfacf and h[1]==0x0100000c and h[3]==6,'worker not arm64 Mach-O dylib')
    offset=32;deps=[];build=[]
    for _ in range(h[4]):
        cmd,size=struct.unpack_from('<II',binary,offset);need(size>=8 and offset+size<=len(binary),'malformed Mach-O')
        if cmd in (12,0x80000018,0x8000001f,0x80000023):
            start=struct.unpack_from('<I',binary,offset+8)[0];deps.append(binary[offset+start:offset+size].split(b'\0')[0].decode())
        if cmd==0x32:build.append(list(struct.unpack_from('<4I',binary,offset+8)))
        offset+=size
    need(deps==['/usr/lib/libSystem.B.dylib'],'worker links unexpected framework/library')
    execution=json.loads((BASE/'execution.json').read_text())
    need(execution['returncode']==0 and execution['model_write_authorized'] is False and execution['GPU_work'] is False,'execution not read-only CPU success')
    need(execution['argv'][:2]==['python3','-B'] and execution['argv'][2:]==r['argv'],'execution argv not bound')
    need(execution['end']>execution['start'] and execution['end']-execution['start']>=r['elapsed_seconds'],'invalid execution window')
    need((BASE/'stdout').read_bytes()==b'','unexpected stdout')
    lines=(BASE/'stderr').read_text().splitlines();need(lines==[f'{f} {n}: PASS' for f,n,_,_ in SPEC],'stderr lacks exact 50 ordered PASS records')
    durable=[]
    for local,name in [('audit.json','narrow-v2-audit.json'),('execution.json','narrow-v2-execution.json'),('delivery.json','narrow-v2-delivery.json'),('stdout','narrow-v2-stdout'),('stderr','narrow-v2-stderr')]:
        need((BASE/local).read_bytes()==(DURABLE/name).read_bytes(),'durable copy mismatch');durable.append(descriptor(DURABLE/name))
    tar_files=0;tar_names=set()
    with tarfile.open(ARCHIVE) as tf:
        for m in tf.getmembers():
            name=PurePosixPath(m.name);need(not name.is_absolute() and '..' not in name.parts and m.name not in tar_names,'unsafe/duplicate archive name');tar_names.add(m.name)
            need(m.isdir() or m.isfile(),'archive has link/device')
            if m.isfile():
                need(name.parts[0]=='narrow-v2','wrong archive root')
                p=BASE.joinpath(*name.parts[1:]);need(tf.extractfile(m).read()==p.read_bytes(),'archive/extracted mismatch');tar_files+=1
    negatives=[]
    def reject(name,mutate):
        altered=copy.deepcopy(r);mutate(altered)
        try:check_report(altered)
        except (ValueError,KeyError,TypeError):negatives.append(name);return
        raise ValueError('negative fixture was accepted: '+name)
    reject('missing_tensor',lambda x:x['tensors'].pop())
    reject('wrong_router_type',lambda x:x['tensors'][0].update(type=1))
    reject('wrong_index_shape',lambda x:x['tensors'][40].update(dims=[1280,4095]))
    reject('source_reconstructed_SHA_mismatch',lambda x:x['tensors'][48].update(reconstructed_stream_sha256='0'*64))
    reject('nonzero_numeric_failure',lambda x:x['tensors'][41]['counts'].update(roundtrip_mismatches=1))
    reject('wrong_block_valid_mask',lambda x:x['tensors'][40].update(block_valid_mask_sha256='0'*64))
    reject('family_sum_mismatch',lambda x:x['families']['router']['counts'].update(elements=0))
    reject('scale_budget_omitted',lambda x:x['tensors'][48].update(scale_payload_bytes=0))
    reject('scope_alias_mismatch',lambda x:x['scope_binding']['width'].update(selected_key='deepseek41.embedding_length'))
    reject('tensor_span_overlap',lambda x:x['tensors'][1].update(file_offset=x['tensors'][0]['file_offset'],offset=x['tensors'][0]['offset']))
    report={'schema_version':1,'kind':'independent_saved_CPU_narrow_audit_report_and_artifact_review','verdict':'PASS_CPU_representation_evidence_consistency','reviewer_imported_or_executed_original_tool':False,'model_reread':False,'GPU_work_submitted':False,'frozen_files_modified':False,
      'review_source':descriptor(__file__),'audit':descriptor(BASE/'audit.json'),'execution':descriptor(BASE/'execution.json'),'complete_archive':descriptor(ARCHIVE),
      'verified_regular_files_in_archive':tar_files,'artifact_manifest_file_count':len(declared),'source_and_artifact_files_verified':verified,'durable_copies_verified':durable,
      'recomputed':result,'total_narrow_bytes_including_scales':result['code_bytes']+result['scale_bytes'],'total_narrow_MiB_including_scales':(result['code_bytes']+result['scale_bytes'])/1048576,
      'additional_allocation_if_originals_retained':True,'metadata_count':67,'original_GGUF_tensor_count':1046,'audited_target_count':50,'block_scale_count':348160,
      'worker_binary':descriptor(BASE/'artifacts/worker.dylib'),'worker_MachO':{'CPU':'arm64','type':'dylib','load_dylibs':deps,'LC_BUILD_VERSION_raw':build},
      'actual_compile_environment':command['environment'],'negative_fixtures_rejected':negatives,
      'limitations':['Original target payloads and GGUF header raw bytes are not included in the small evidence bundle; their SHA values are recorded by the pinned CPU audit, not recomputed from the model by this review.',
      'All retained source/binary/patch/command/log/manifest/archive/durable-copy file identities were independently rehashed; each reported source/reconstruction SHA pair, target coverage, counts, all-valid mask SHA, family manifest SHA and byte budget were independently checked.',
      'This establishes saved CPU format-reversibility evidence consistency only. It does not establish GPU GEMV reduction/FMA/rounding equivalence, runtime lifetime safety, additional physical memory headroom, decode speed or production adoption.',
      'The actual compiler is Xcode clang; the recorded SDKROOT is the CommandLineTools MacOSX.sdk. No claim of an Xcode SDK build is made.',
      '362GB whole-model content SHA remains unavailable by design; file stat identity and audited header/target content scope are explicit.']}
    out=OUT/'review.json';out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'review':descriptor(out),'verdict':report['verdict'],'recomputed':{k:v for k,v in result.items() if k not in ('families','scope_binding')},'code_bytes':result['code_bytes'],'scale_bytes':result['scale_bytes'],'total_narrow_bytes':report['total_narrow_bytes_including_scales'],'negative_fixtures':len(negatives)},indent=2))

if __name__=='__main__':main()
