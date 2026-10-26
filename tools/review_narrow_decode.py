#!/usr/bin/env python3
"""Independent CPU-only full-payload narrow GPU decode correctness gate.

No subprocess, Metal, model loader or performance interpretation. All declared
weights, guarded pre/post images, original provider outputs and GPU reconstruction
payloads are read. A completed numerical failure remains valid evidence.
"""
import argparse
import array
import ast
import hashlib
import json
from pathlib import Path
import re
import struct
import sys
sys.dont_write_bytecode=True
GUARD=b'\xa5'*64
MARKER=b'\n#line 1 "myds_narrow_decode.metal.inc"\n'
ROLES=('weight','codes','scales','input')
ARTIFACTS={'harness','reader','binary','build_identity','deployed_library','native_library','probe_library','fixtures','host_fixture_manifest'}
class EvidenceError(ValueError):pass
def need(ok,msg):
    if not ok:raise EvidenceError(msg)
def integer(x,lo=0,hi=(1<<64)-1):
    need(type(x)is int and lo<=x<=hi,'strict bounded integer');return x
def exact(a,b,label):
    if type(b)is dict:
        need(type(a)is dict and set(a)==set(b),label+' keys')
        for k,v in b.items():exact(a[k],v,label+'.'+k)
    elif type(b)is list:
        need(type(a)is list and len(a)==len(b),label+' list')
        for x,y in zip(a,b):exact(x,y,label)
    else:need(type(a)is type(b)and a==b,label+' exact value/type')
def pairs(items):
    d={}
    for k,v in items:need(k not in d,'duplicate JSON key');d[k]=v
    return d
def load(path):
    return json.loads(Path(path).read_text(),object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in()).throw(EvidenceError('nonfinite JSON')))
def digest(b):return hashlib.sha256(b).hexdigest()
def file_sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def alias(s):return '/private/tmp/'+s[5:]if s.startswith('/tmp/')else s
class Paths:
    def __init__(self,maps=()):self.maps=sorted(((alias(a.rstrip('/')),Path(b))for a,b in maps),key=lambda x:-len(x[0]))
    def path(self,s):
        need(type(s)is str and Path(s).is_absolute(),'absolute frozen path');s=alias(s)
        for a,b in self.maps:
            if s==a or s.startswith(a+'/'):return b/s[len(a):].lstrip('/')
        return Path(s)
    def desc(self,d,empty=False):
        need(type(d)is dict and set(d)=={'path','bytes','sha256'},'exact artifact descriptor');integer(d['bytes'],0 if empty else 1);need(re.fullmatch('[0-9a-f]{64}',d['sha256'])is not None,'SHA256 text')
        p=self.path(d['path']);need(p.is_file()and not p.is_symlink(),'regular frozen artifact '+str(p));need(p.stat().st_size==d['bytes']and file_sha(p)==d['sha256'],'frozen artifact bytes/SHA '+str(p));return p

def safe(root,name):
    need(type(name)is str and name==Path(name).name and name not in('', '.', '..')and not name.startswith('.'),'single raw filename');p=Path(root)/name;need(p.is_file()and not p.is_symlink(),'regular complete raw '+name);return p
class Raw:
    def __init__(self,root,records):
        self.root=Path(root);self.records={};self.used=set();cache={}
        need(type(records)is list,'raw file list')
        for d in records:
            need(type(d)is dict and set(d)in({'path','bytes','sha256'},{'path','bytes','sha256','storage_alias_of'}),'exact raw descriptor');n=d['path'];need(n not in self.records,'unique raw inventory');p=safe(root,n);integer(d['bytes']);st=p.stat();need(st.st_size==d['bytes'],'raw length');key=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns);h=cache.get(key)
            if h is None:h=file_sha(p);cache[key]=h
            need(h==d['sha256'],'complete raw SHA '+n);self.records[n]=d
            if 'storage_alias_of'in d:
                a=d['storage_alias_of'];need(a in self.records and a!=n,'prior hardlink alias');q=safe(root,a);need(q.stat().st_ino==st.st_ino and q.stat().st_dev==st.st_dev,'actual retained hardlink')
        need({p.name for p in self.root.iterdir()}==set(self.records)|{'artifact-manifest.json'},'exact output raw directory closure')
    def use(self,d,name=None):
        need(type(d)is dict and 'path'in d and d['path']in self.records and d==self.records[d['path']],'raw descriptor equals complete inventory')
        if name is not None:exact(d['path'],name,'deterministic raw path')
        need(d['path']not in self.used,'distinct raw pathname per declaration');self.used.add(d['path']);return self.root/d['path']
    def guarded(self,d,stem,expected=None):
        need(type(d)is dict and set(d)=={'body','prefix_guard','suffix_guard'},'guarded image keys')
        b=self.use(d['body'],stem+'.body.raw');a=self.use(d['prefix_guard'],stem+'.prefix-guard.raw');z=self.use(d['suffix_guard'],stem+'.suffix-guard.raw')
        ok=a.read_bytes()==z.read_bytes()==GUARD
        equal=True if expected is None else same_files(b,expected)
        return b,ok,equal
    def finish(self):need(self.used==set(self.records),'every retained raw file accounted for')
def same_files(a,b):
    if Path(a).stat().st_size!=Path(b).stat().st_size:return False
    with Path(a).open('rb')as x,Path(b).open('rb')as y:
        while True:
            c=x.read(1<<20);d=y.read(1<<20)
            if c!=d:return False
            if not c:return True
def bits_half(h):
    sign=(h&0x8000)<<16;e=(h>>10)&31;m=h&1023
    if e==31:return sign|0x7f800000|(m<<13)
    if e:return sign|((e+112)<<23)|(m<<13)
    if not m:return sign
    shift=m.bit_length()-1;return sign|((shift+103)<<23)|((m-(1<<shift))<<(23-shift))
HALF_TABLE=tuple(struct.pack('<I',bits_half(h))for h in range(65536))
def weight_f32_chunks(path,router):
    with Path(path).open('rb')as f:
        while True:
            b=f.read((1<<18)*(4 if router else 2))
            if not b:return
            if router:yield b
            else:
                need(len(b)%2==0,'half whole chunk');a=array.array('H');a.frombytes(b)
                if sys.byteorder!='little':a.byteswap()
                yield b''.join(HALF_TABLE[x]for x in a)
def f32_weight_identity(path,router):
    h=hashlib.sha256();n=0
    for b in weight_f32_chunks(path,router):h.update(b);n+=len(b)
    return dict(bytes=n,sha256=h.hexdigest())
def expanded_equal(raw,source,router):
    with Path(raw).open('rb')as f:
        for b in weight_f32_chunks(source,router):
            if f.read(len(b))!=b:return False
        return f.read(1)==b''
def finite_file(path,bf16=False):
    with Path(path).open('rb')as f:
        for b in iter(lambda:f.read(1<<20),b''):
            if len(b)%4:return False
            for x, in struct.iter_unpack('<I',b):
                if x&0x7f800000==0x7f800000 or bf16 and x&65535:return False
    return True

def written_file(path):
    with Path(path).open('rb')as f:
        for b in iter(lambda:f.read(1<<20),b''):
            if len(b)%4 or any(x==0xa5a5a5a5 for x, in struct.iter_unpack('<I',b)):return False
    return True


def fixture_cases(f,paths,actual=False):
    need(type(f)is dict and f.get('schema_version')==1 and f.get('kind')=='narrow_decode_fixture_set'and f.get('correctness_only')is True and type(f.get('cases'))is list and f['cases'],'frozen fixture scope')
    cases=[];ids=set()
    for c in f['cases']:
        need(type(c)is dict and {'id','family','columns','rows','profile','outputs',*ROLES}<=set(c),'complete fixture case');ident=c['id'];need(type(ident)is str and re.fullmatch('[A-Za-z0-9_-]+',ident)and ident not in ids,'unique safe fixture id');ids.add(ident);family=c['family'];need(family in('router','index_qb','engram'),'exact family');k=integer(c['columns'],32,6144);n=integer(c['rows'],32,25600);need(k%32==0 and n%32==0,'block-multiple component scope');need(not actual or(k,n)=={'router':(5120,384),'index_qb':(1280,4096),'engram':(6144,25600)}[family],'strict actual-family full tensor geometry');router=family=='router';exact(c['outputs'],['f32']if router else['f32','bf16'],'frozen output modes');need(type(c['profile'])is str and c['profile'],'input profile')
        source={role:paths.desc(c[role],empty=role=='scales'and router)for role in ROLES};sizes=dict(weight=k*n*(4 if router else 2),codes=k*n*(2 if router else 1),scales=0 if router else(k//32)*(n//32)*4,input=k*4)
        need(all(c[role]['bytes']==size for role,size in sizes.items()),'complete source role extents')
        key=(family,k,n,*(c[x]['sha256']for x in('weight','codes','scales')))
        cases.append(dict(spec=c,source=source,sizes=sizes,router=router,key=key,f32=f32_weight_identity(source['weight'],router)))
    return cases

def verify_actual_sources(f,cases,paths):
    expected={**{f'blk.{i}.ffn_gate_inp.weight':('router',0)for i in range(40)},**{f'blk.{i}.indexer.attn_q_b.weight':('index_qb',1)for i in(2,8,14,20,24,28,32,36)},**{f'blk.{i}.engram_kv.weight':('engram',1)for i in(1,14)}}
    need(len(cases)==50,'all fifty actual tensor cases');observed={};audit=None
    for case in cases:
        c=case['spec'];v=c['provenance'];need(v['synthetic']is False and v['CPU_exact_qualification']is True,'actual prior-qualified source, not synthetic')
        name=v['tensor_name'];need(name in expected and name not in observed and(c['family'],v['type'])==expected[name],'complete unique actual target name/type')
        p=paths.desc(v['prior_audit']);need(file_sha(p)=='495064f318033e61a48ed73e0f01c7453edecd462b44922ac3942b3de8456dbb','frozen full50 CPU qualification')
        if audit is None:audit=load(p)
        t=next(t for t in audit['tensors']if t['name']==name)
        need(t['qualification_passed']is True and t['source_sha256']==c['weight']['sha256']and t['codes_stream_sha256']==c['codes']['sha256']and v['file_offset']==t['file_offset']and t['dims']==[c['columns'],c['rows']],'actual full tensor source/code and offset bound to prior audit')
        need((t['scales_stream_sha256']if t['scales_stream_sha256']is not None else digest(b''))==c['scales']['sha256'],'actual scale stream bound to prior audit');observed[name]=True
    need(set(observed)==set(expected)and f['scope_binding']==audit['scope_binding']and f['whole_model_payload_sha256_verified']is False,'exact fullarchitecture scope; no whole-model SHA claim')
    return dict(actual_target_tensors=50,prior_audit_sha256='495064f318033e61a48ed73e0f01c7453edecd462b44922ac3942b3de8456dbb',complete_source_code_scale_streams_bound=True)


def mv_bytes(k,n,width,nr):
    a=bytearray(112)
    for off,val in ((0,k),(4,n),(8,1),(48,k),(52,1),(56,1),(96,n),(100,1),(104,nr)):struct.pack_into('<i',a,off,val)
    for off,val in ((16,width),(24,k*width),(32,k*width*n),(40,k*width*n),(64,4),(72,k*4),(80,k*4),(88,k*4)):struct.pack_into('<Q',a,off,val)
    struct.pack_into('<hh',a,108,1,1);return bytes(a)

def verify_build(build_path,paths,artifacts):
    b=load(build_path);root=build_path.parent
    for k,v in dict(schema_version=1,kind='narrow_decode_complete_library_build',subject='exact_narrow_decode_weights',compile_only_pass=True,compilation_skipped=False,gpu_work_submitted=False,default_enabled=False,production_adoption=False,production_registered=False,runtime_connected=False,production_shader_count=29).items():exact(b[k],v,'complete build '+k)
    exact(b['compile_options'],['-mmacosx-version-min=15.0','-O3','-ffast-math','-DDS4_METAL_HC_STABLE=1','-DDS4_METAL_NORM_RSQRT_DISABLE=1'],'same immutable original compile options')
    need(len(b['commands'])==len(b['compilation_records'])==5,'all original native/probe compile/link plus harness compiler records')
    for command,record in zip(b['commands'],b['compilation_records']):
        exact(record['argv'],command,'actual compiler argv');exact(record['returncode'],0,'successful compiler command');paths.desc(record['stdout'],True);paths.desc(record['stderr'],True)
    for key in('native','probe'):paths.desc(b['air'][key])
    identity=load(paths.desc(b['source_identity']));need(identity['source_inventory']==b['source_inventory']and identity['source_inventory_sha256']==b['source_inventory_sha256'],'compiler-visible identity inventory equal build')
    paths.desc(b['source_transform'])
    inv=b['source_inventory'];need(type(inv)is dict and inv,'full compiler source inventory');source=root/'source';need(set(inv)=={str(p.relative_to(source))for p in source.rglob('*')if p.is_file()},'exact compiler source closure')
    for rel,d in inv.items():
        p=source/rel;need('..'not in Path(rel).parts and p.is_file()and not p.is_symlink()and p.stat().st_size==d['bytes']and file_sha(p)==d['sha256'],'compiler input full bytes '+rel)
    need(digest(json.dumps(inv,sort_keys=True,separators=(',',':')).encode())==b['source_inventory_sha256'],'compiler inventory canonical SHA')
    literals={}
    for node in ast.parse((source/'tools/build_metallib.py').read_text()).body:
        if isinstance(node,ast.Assign)and len(node.targets)==1 and isinstance(node.targets[0],ast.Name)and node.targets[0].id in('PREAMBLE','SHADER_FILES','COPY_KERNELS'):literals[node.targets[0].id]=ast.literal_eval(node.value)
    need(set(literals)=={'PREAMBLE','SHADER_FILES','COPY_KERNELS'}and len(literals['SHADER_FILES'])==29,'literal production29 shader assembly')
    native=(literals['PREAMBLE']+'\n').encode()
    for name in literals['SHADER_FILES']:native+=('\n#line 1 "'+name+'"\n').encode()+(source/'metal'/name).read_bytes()
    native+=(literals['COPY_KERNELS']+'\n').encode();np=paths.desc(b['source_files']['native']);pp=paths.desc(b['source_files']['probe']);need(np.read_bytes()==native and digest(native)==b['native_source_sha256'],'native TU exactly original29')
    candidate=(source/'tools/narrow_decode.metal.inc').read_bytes();combined=pp.read_bytes();need(combined==native+MARKER+candidate and digest(combined)==b['combined_source_sha256'],'append-only candidate TU exact inverse')
    for name in('native','probe'):need(b['metallibs'][name]==artifacts[name+'_library'],'executed library is actual complete build '+name)
    need(b['artifacts']['bench_narrow_decode']==artifacts['binary'],'actual executed binary built from full frozen harness');need(inv['tools/bench_narrow_decode.m']['sha256']==artifacts['harness']['sha256'],'actual harness compiler source identity')
    for name in('native','probe'):paths.desc(b['metallibs'][name])
    return dict(source_files=len(inv),production_shader_count=29,native_source_sha256=digest(native),combined_source_sha256=digest(combined),candidate_sha256=digest(candidate))

class Review:
    def __init__(self,args):
        self.args=args;self.paths=Paths(tuple(x.split('=',1))for x in args.path_map);self.report=dict(schema_version=1,reader='independent_narrow_decode_full_payload',reader_sha256=file_sha(__file__),review_completed=False,evidence_valid=False,numerical_passed=False,performance_evaluated=False,fullmodel_eligible=False,production_adopted=False)
    def initialize(self):
        p=self.protocol=load(self.args.protocol);exact(p['schema_version'],1,'protocol version');exact(p['kind'],'narrow_decode_component_protocol','protocol kind');need(p['stage']in('synthetic_correctness','actual_weight_correctness'),'known prospective stage');exact(p['correctness_only'],True,'correctness only');exact(p['timing_allowed'],False,'no timing');need(type(p['expected_device'])is str and p['expected_device'],'frozen targetdevice');need(set(p['artifacts'])==ARTIFACTS,'exact protocol artifacts')
        self.frozen={k:self.paths.desc(d)for k,d in p['artifacts'].items()};need(file_sha(__file__)==p['artifacts']['reader']['sha256'],'executed frozen reader');self.fixture=load(self.frozen['fixtures']);self.cases=fixture_cases(self.fixture,self.paths,p['stage']=='actual_weight_correctness');self.report['actual_source_binding']=verify_actual_sources(self.fixture,self.cases,self.paths)if p['stage']=='actual_weight_correctness'else None;counts={f:len({c['key']for c in self.cases if c['spec']['family']==f})for f in('router','index_qb','engram')};unique=sum(counts.values());need(p['expected_case_count']==len(self.cases)and type(p['expected_case_count'])is int and p['expected_unique_weight_count']==unique and type(p['expected_unique_weight_count'])is int,'prospective exact case/unique coverage');exact(p['expected_family_unique_weights'],counts,'prospective per-family unique coverage');expected_completions=sum(4*len(c['spec']['outputs'])for c in self.cases)+counts['router']+2*(counts['index_qb']+counts['engram']);exact(p['expected_gpu_completion_count'],expected_completions,'prospective completion coverage');self.expected_raw_count=3+sum(12+62*len(c['spec']['outputs'])for c in self.cases)+10*(counts['router']+2*(counts['index_qb']+counts['engram']));exact(p['expected_raw_file_count'],self.expected_raw_count,'prospective complete raw coverage');self.build_proof=verify_build(self.frozen['build_identity'],self.paths,p['artifacts']);self.report.update(stage=p['stage'],protocol_sha256=file_sha(self.args.protocol),build_proof=self.build_proof,case_count=len(self.cases))
        self.check_manifest(self.frozen['host_fixture_manifest'].parent,True)
        self.report.update(initialization_passed=True);return self.report
    def check_manifest(self,root,host):
        m=load(Path(root)/'artifact-manifest.json');p=self.protocol;need(type(m)is dict,'manifest object')
        for k,v in dict(schema_version=1,probe='narrow_decode_correctness',correctness_only=True,timing_collected=False,fullmodel=False,production_registered=False,adoption_eligible=False,host_only=host,gpu_work_submitted=not host,guard_bytes=64,poison_byte=165,case_count=len(self.cases)).items():exact(m[k],v,'manifest '+k)
        for key,artifact in(('fixture','fixtures'),('binary','binary'),('build_identity','build_identity'),('harness_source','harness')):need(m[key+'_sha256']==p['artifacts'][artifact]['sha256']and self.paths.path(m[key+'_path']).resolve()==self.frozen[artifact].resolve(),'actual executed artifact '+key)
        libs=p['artifacts'];expectedlibs=[libs[x+'_library']for x in('deployed','native','probe')];need(len(m['libraries'])==3,'three actual libraries')
        for i,(actual,expected)in enumerate(zip(m['libraries'],expectedlibs)):
            need(actual['index']==i and actual['bytes']==expected['bytes']and actual['sha256']==expected['sha256']and self.paths.path(actual['path']).resolve()==self.frozen[('deployed','native','probe')[i]+'_library'].resolve(),'loaded full library identity')
        if not host:exact(m['device'],p['expected_device'],'actual targetdevice')
        raw=Raw(root,m['files']);need(raw.use(m['fixtures_copy'],'fixtures.json').read_bytes()==self.frozen['fixtures'].read_bytes(),'complete frozen fixture copy');need(raw.use(m['build_identity_copy'],'build-identity.json').read_bytes()==self.frozen['build_identity'].read_bytes(),'complete build identity copy');need(raw.use(m['harness_source_copy'],'harness-source.m').read_bytes()==self.frozen['harness'].read_bytes(),'complete executed harness copy');need(len(m['cases'])==len(self.cases),'exact actual case order')
        first={};completed=0;allok=True;rows=[]
        for ci,(record,case)in enumerate(zip(m['cases'],self.cases)):
            c=case['spec'];k,n=c['columns'],c['rows'];router=case['router'];stem=f'case{ci:03d}';exact(record['case'],ci,'case index');exact(record['id'],c['id'],'case id');exact(record['family'],c['family'],'case family');exact(record['columns'],k,'case K');exact(record['rows'],n,'case N');exact(record['fixture'],c,'complete actual fixture');exact(record['expected_f32_weights'],case['f32'],'independent integer half-widen SHA')
            for role in ROLES:
                _,guard,equal=raw.guarded(record['inputs_before'][role],stem+'.'+role+'.before',case['source'][role]);need(guard and equal,'actual device input preimage equals frozen fixture '+role)
            reference={};providers=record['providers'];need(len(providers)==4*len(c['outputs']),'complete provider catalog');index=0
            for mode in c['outputs']:
                bf=mode=='bf16';nr=2 if bf else 1;native='kernel_mul_mv_f16_f32_4_bf16_nr2'if bf else'kernel_mul_mv_f32_f32_4'if router else'kernel_mul_mv_f16_f32_4';candidate='kernel_myds_narrow_router_bf16_nr1'if router else'kernel_myds_narrow_e4m3_bf16_nr2'if bf else'kernel_myds_narrow_e4m3_f32_nr1';native_args=mv_bytes(k,n,4 if router else 2,nr);narrow_args=mv_bytes(k,n,2 if router else 1,nr);need(raw.use(record[mode+'_args'],stem+'.'+mode+'.args112.raw').read_bytes()==native_args,'complete native ABI112 byte layout');need(raw.use(record[mode+'_candidate_args'],stem+'.'+mode+'.candidate-args112.raw').read_bytes()==narrow_args,'complete narrow ABI112 byte strides')
                for provider in range(4):
                    row=providers[index];index+=1;li=provider if provider<3 else 2;symbol=native if provider<3 else candidate
                    expected=dict(provider=provider,mode=mode,symbol=symbol,library_index=li,library_sha256=expectedlibs[li]['sha256'],args_bytes=112,args_kind='narrow'if provider==3 else'original',args_sha256=digest(narrow_args if provider==3 else native_args),nr0=nr,nsg=8,dynamic_threadgroup_bytes=128*nr,threads=[32,8,1],grid=[(n+nr-1)//nr,1,1],buffers=['args','codes','input','output','scales']if provider==3 else['args','weight','input','output'])
                    for key,value in expected.items():exact(row[key],value,'provider binding '+key)
                    if host:need(set(row)==set(expected),'host-only contains noGPUclaims');continue
                    completed+=1;self.pipeline(row,128*nr);ok=self.completion(row,completed);post=row['inputs_after'];need(set(post)==set(ROLES),'all actual input postimages');inputok=True
                    for role in ROLES:
                        _,guard,equal=raw.guarded(post[role],stem+f'.{mode}.provider{provider}.{role}.after',case['source'][role]);inputok&=guard and equal
                    exact(row['inputs_unchanged'],inputok,'native input declaration');ok&=inputok
                    out,guard,_=raw.guarded(row['output'],stem+f'.{mode}.provider{provider}.output');need(out.stat().st_size==n*4,'complete output extent');ref=reference.setdefault(mode,out);same=same_files(out,ref);finite=finite_file(out);roundok=not bf or finite_file(out,True);written=written_file(out);ok&=guard and same and finite and roundok and written;exact(row['matches_deployed'],same,'actual original provider byte match');exact(row['finite'],finite,'all output values finite');exact(row['bf16_output_bits_valid'],roundok,'requested BF16 output bits');exact(row['all_output_words_written'],written,'all output words actually written');exact(row['guards_intact'],guard,'output guard claim');exact(row['numerical_passed'],ok,'provider complete result');rows.append(dict(case=ci,provider=provider,mode=mode,passed=ok));allok&=ok
            key=case['key'];witness=first.setdefault(key,ci);exact(record['reconstruction_case'],witness,'first unique representation witness');names=[]if witness!=ci else['kernel_myds_narrow_expand_router_f32']if router else['kernel_myds_narrow_expand_e4m3_f16','kernel_myds_narrow_expand_e4m3_f32'];need(len(record['expansions'])==len(names),'once-per-unique-weight all reconstruction modes')
            for row,symbol in zip(record['expansions'],names):
                f32=symbol.endswith('f32');expected=dict(symbol=symbol,args_bytes=16,threads=[256,1,1],grid=[(k*n+255)//256,1,1],output_kind='F32'if f32 else'F16',library_sha256=expectedlibs[2]['sha256'])
                for a,b in expected.items():exact(row[a],b,'expansion binding '+a)
                need(raw.use(row['args'],stem+'.'+symbol+'.args16.raw').read_bytes()==struct.pack('<IIQ',k,n,k*n),'complete reconstruction ABI16')
                if host:need(set(row)==set(expected)|{'args'},'host expansion noGPUclaims');continue
                completed+=1;self.pipeline(row,0);ok=self.completion(row,completed);after=True
                for role in('codes','scales'):
                    _,guard,equal=raw.guarded(row[role+'_after'],stem+'.'+symbol+'.'+role+'.after',case['source'][role]);after&=guard and equal
                out,guard,_=raw.guarded(row['output'],stem+'.'+symbol+'.output');need(out.stat().st_size==k*n*(4 if f32 else 2),'complete reconstruction extent');same=expanded_equal(out,case['source']['weight'],router)if f32 else same_files(out,case['source']['weight']);ok&=after and guard and same;exact(row['inputs_unchanged'],after,'reconstruction inputs');exact(row['guards_intact'],guard,'reconstruction output guard');exact(row['matches_original_bits'],same,'complete reconstructed weight original bit match');exact(row['numerical_passed'],ok,'actual reconstruction result');rows.append(dict(case=ci,symbol=symbol,passed=ok));allok&=ok
        raw.finish();exact(m['gpu_completion_count'],completed,'all GPUcompletion count');exact(m['all_numerical_passed'],False if host else allok,'complete numerical declaration');exact(m['status'],'host_only_complete_not_numerical'if host else'correctness_complete_not_adoption'if allok else'numerical_failed','completion classification')
        if not host:need(len(raw.records)==self.expected_raw_count,'full prospective raw filecount');self.report['family_results']={f:dict(passed=any(c['spec']['family']==f for c in self.cases)and all(r['passed']for r in rows if self.cases[r['case']]['spec']['family']==f),case_count=sum(c['spec']['family']==f for c in self.cases),unique_weight_count=len({c['key']for c in self.cases if c['spec']['family']==f}))for f in('router','index_qb','engram')};self.report.update(device=m['device'],numerical_passed=allok,gpu_completion_count=completed,raw_file_count=len(raw.records),unique_weight_reconstruction_count=len(first),case_results=rows,artifact_manifest_sha256=file_sha(Path(root)/'artifact-manifest.json'))
        return m
    def pipeline(self,row,tg):
        exact(row['thread_execution_width'],32,'actual SIMD width');need(integer(row['max_threads'],256)>=256,'actual thread capacity');integer(row['static_threadgroup_bytes']);need(row['static_threadgroup_bytes']==0,'nohidden static TG memory')
    def completion(self,row,ordinal):
        exact(row['completion_ordinal'],ordinal,'strict serial completion order');integer(row['gpu_status'],0,5);ok=row['gpu_status']==4 and row['gpu_error']is None;exact(row['gpu_completed'],ok,'actual Metal completion');return ok
    def run(self):
        self.initialize();m=self.check_manifest(self.args.artifact_root,False);records=[]
        for line in Path(self.args.stdout).read_text().splitlines():
            need(line.startswith('{'),'strict native JSONL only');records.append(json.loads(line,object_pairs_hook=pairs))
        expected=[];ordinal=0
        for c in m['cases']:
            for p in c['providers']:ordinal+=1;expected.append(dict(kind='provider_complete',case=c['case'],provider=p['provider'],mode=p['mode'],ordinal=ordinal,gpu_status=p['gpu_status'],numerical_passed=p['numerical_passed']))
            for e in c['expansions']:ordinal+=1;expected.append(dict(kind='expansion_complete',case=c['case'],symbol=e['symbol'],ordinal=ordinal,gpu_status=e['gpu_status'],numerical_passed=e['numerical_passed']))
            expected.append(dict(kind='case_complete',case=c['case'],id=c['id'],host_only=False))
        expected.append(dict(kind='complete',case_count=len(self.cases),gpu_completion_count=ordinal,all_numerical_passed=self.report['numerical_passed'],status=m['status'],timing_collected=False,adoption_eligible=False));exact(records,expected,'complete ordered actual stdout records');self.report.update(review_completed=True,evidence_valid=True,classification='component_numerical_passed'if self.report['numerical_passed']else'completed_numerical_failure',stdout_sha256=file_sha(self.args.stdout));return self.report

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--artifact-root',type=Path);ap.add_argument('--stdout',type=Path);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--initialize-only',action='store_true');ap.add_argument('--path-map',action='append',default=[]);a=ap.parse_args();need(not a.out.exists(),'new independent report; retained evidence never overwritten');run=Review(a);code=0
    try:
        report=run.initialize()if a.initialize_only else run.run();report['evidence_valid']=True
        if a.initialize_only:report['gpu_work_submitted_by_reader']=False
    except(EvidenceError,OSError,ValueError,TypeError,KeyError,IndexError)as e:report=run.report;report.update(evidence_error=str(e),classification='evidence_invalid',evidence_valid=False,numerical_passed=False);code=2
    a.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({k:report.get(k)for k in('review_completed','evidence_valid','numerical_passed','classification','evidence_error')},indent=2));return code
if __name__=='__main__':raise SystemExit(main())
