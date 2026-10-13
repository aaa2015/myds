#!/usr/bin/env python3
"""CPU-only complete payload/timestamp/source audit of attention+rope fixture."""
import argparse,hashlib,json,math,statistics,struct
from pathlib import Path
def require(v,message):
    if not v:raise ValueError(message)
def sha(data):return hashlib.sha256(data).hexdigest()
def parse(data):
    def pairs(items):
        d={}
        for k,v in items:require(k not in d,'duplicate JSON key');d[k]=v
        return d
    def bad(x):raise ValueError('nonfinite JSON literal '+x)
    obj=json.loads(data,object_pairs_hook=pairs,parse_constant=bad)
    def finite(v):
        if type(v) is float:require(math.isfinite(v),'nonfinite JSON number')
        elif type(v) is dict:
            for x in v.values():finite(x)
        elif type(v) is list:
            for x in v:finite(x)
    finite(obj);return obj
def read(p):
    require(p.is_file() and not p.is_symlink(),'missing/unsafe file '+str(p));return p.read_bytes()
def bit(v):require(type(v) is bool or type(v) is int and v in (0,1),'invalid native boolean');return bool(v)
def extract(source,name):
    start=source.index('static inline void '+name+'(');brace=source.index('{',start);depth=1;i=brace+1
    while depth:
        if source[i]=='{':depth+=1
        elif source[i]=='}':depth-=1
        i+=1
    return source[start:i]
def review(a):
    m=parse(read(a.raw/'artifact-manifest.json'));s=parse(read(a.source_identity))
    require(m['schema_version']==1 and m['status']=='complete','native manifest incomplete')
    require(m['fullmodel_speedup_proven'] is False and m['gpu_physical_overlap_proven'] is False,'component overclaim')
    for field,path in [('golden_metallib_sha256',a.golden_metallib),('candidate_metallib_sha256',a.candidate_metallib),('binary_sha256',a.binary),('source_identity_sha256',a.source_identity)]:
        require(sha(read(path))==m[field],'external identity mismatch '+field)
    for name,h in s['input_sha256'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,'unsafe source identity path')
        require(sha(read(a.source_root/name))==h,'frozen source mismatch '+name)
    require(s['output_sha256']['myds.metallib']==m['candidate_metallib_sha256'] and s['output_sha256']['test_attention_rope_fuse']==m['binary_sha256'],'builder output identity mismatch')
    source=read(a.source_root/'metal/flash_attn.metal').decode()
    orig=extract(source,'ds4_flash_attn_vec_packed_register_reduce_f16_512');cand=extract(source,'myds_attention_register_rope_fuse_impl')
    store='''                const uint oc = (uint)quadrant * (uint)NW + (uint)out_lane;
                const float4 value = reduced * reduce_inv;
                dst4[oc] = myds_attention_rope_store(value, oc, rope);'''
    restored=cand.replace('myds_attention_register_rope_fuse_impl','ds4_flash_attn_vec_packed_register_reduce_f16_512',1).replace('        constant myds_attention_rope_fuse_args & rope,\n','',1).replace(store,'                dst4[quadrant * NW + out_lane] = reduced * reduce_inv;')
    require(restored==orig and sha(orig.encode())==s['original_helper_sha256'] and sha(cand.encode())==s['candidate_helper_sha256'],'reducer clone differs outside finalstore')
    files=m['files'];require(set(p.name for p in a.raw.iterdir())==set(files)|{'artifact-manifest.json'},'raw inventory mismatch')
    data={}
    for name,f in files.items():
        require(Path(name).name==name,'unsafe raw path');b=read(a.raw/name)
        require(type(f['bytes']) is int and len(b)==f['bytes'] and sha(b)==f['sha256'],'payload size/SHA mismatch '+name);data[name]=b
    def guarded(name,n,bf16=False):
        b=data[name];require(len(b)==n+32 and b[:16]==b[-16:]==bytes([0xa5])*16,'guarded payload extent/canary '+name)
        payload=b[16:-16];require(all(math.isfinite(v[0]) for v in struct.iter_unpack('<f',payload)),'nonfinite output '+name)
        require(all(v[0]!=0xa5a5a5a5 and (not bf16 or v[0]&65535==0) for v in struct.iter_unpack('<I',payload)),'poison/BF16 '+name)
        return payload
    records=[parse(line) for line in read(a.stdout).splitlines() if line.startswith(b'{')]
    require([r for r in records if r.get('kind')=='parity']==m['cases'] and [r for r in records if r.get('kind')=='timing']==m['timings'] and [r for r in records if r.get('kind')=='rejection']==m['rejections'],'stdout/native record binding differs')
    complete=[r for r in records if r.get('kind')=='complete'];require(len(complete)==1 and complete[0]['pass'] is True and complete[0]['raw_files']==len(files),'completion record mismatch')
    require(len(m['rejections'])==11,'negative-control count')
    negative_lengths=[131072,262144,256,19456,4,2048,131104,655360,65600,4210688,1280]
    for i,r in enumerate(m['rejections']):
        require(r['dispatch_delta']==r['encoded_delta']==0 and r['requests_delta']==r['failures_delta']==1 and r['all_observed_buffers_full_bytes_unchanged'] is True,'predispatch rejection contract')
        for j in range(11):
            before=data[f'reject{i}-buffer{j}-before.raw'];after=data[f'reject{i}-buffer{j}-after.raw']
            require(len(before)==negative_lengths[j] and before==after,'negative complete buffer extent/bytes differ')
    quick=bit(m['quick']);require(len(m['cases'])==(16 if quick else 120),'case cardinality')
    expected={(p,c,d,k) for p in ([0,11] if quick else [0,11,127,128,8191,32768]) for c in (False,True) for d in (False,True) for k in range(2 if quick else 5)}
    observed=set();golden={}
    for i,r in enumerate(m['cases']):
        require(r['case']==i and all(r[k] is True for k in ('full_output_same','partial_stats_full_bytes_same','finite','guards','production_golden')),'case flags/index')
        key=(r['pos'],bit(r['compressed']),bit(r['direct']),r['mask_mode']);require(key in expected and key not in observed,'duplicate/unexpected case');observed.add(key)
        require(r['keys']==dict(zip([0,11,127,128,8191,32768],[1,33,128,129,639,640]))[r['pos']] and r['n_raw']==min(r['keys'],128) and r['raw_start']==127 and r['selected']==r['keys']-r['n_raw'] and r['source_rows']==(r['selected']*2+17 if r['selected'] else 0),'case staging geometry')
        pre=f'case{i}';golden[i]=guarded(pre+'-out0.guarded',131072,True)
        require(guarded(pre+'-out1.guarded',131072)==guarded(pre+'-production-unrotated.guarded',131072),'private baseline vs actual production unrotated')
        for n in (2,3):require(guarded(pre+f'-out{n}.guarded',131072,True)==golden[i],'fused full output differs')
        for kind,n in [('partial',4194304),('stats',24576)]:require(guarded(pre+f'-{kind}0.guarded',n)==guarded(pre+f'-{kind}1.guarded',n),'complete intermediate bytes differ')
        for suffix,n in [('q.f32',131072),('window.f32',262144),('sinks.f32',256),('frequencies.f32',128)]:
            require(len(data[pre+'-'+suffix])==n,'complete input extent')
            require(all(math.isfinite(v[0]) for v in struct.iter_unpack('<f',data[pre+'-'+suffix])),'input nonfinite')
        require(len(data[pre+'-mask.f16'])==1280,'mask extent')
        if r['selected']:
            require(len(data[pre+'-comp.bf16'])==r['source_rows']*1024 and len(data[pre+'-gathered.f32'])==r['selected']*2048 and len(data[pre+'-ids.i32'])==r['selected']*4,'selected input extent')
    require(observed==expected,'missing case coverage')
    samples,reps=m['samples'],m['repetitions'];require(type(samples) is int and 3<=samples<=31 and type(reps) is int and 1<=reps<=64,'sampling contract')
    times={}
    for r in m['timings']:
        key=(r['case'],r['arm'],r['sample']);require(key not in times and key[0] in golden and key[1] in ('A1','B','A2') and type(key[2]) is int and 0<=key[2]<=samples,'unexpected/duplicate timing')
        require(bit(r['warmup'])==(key[2]==0) and r['repetitions']==reps and r['fused_encoded']==(reps if key[1]=='B' else 0) and all(r[k] is True for k in ('full_output_same','guards','finite')),'actual timing adoption gate')
        case=m['cases'][key[0]];base=(2 if bit(case['direct']) else 2+(case['n_raw']>128-case['raw_start']))+(case['selected']>0)+(case['keys']%32!=0)
        require(r['dispatches']==reps*(base+(key[1]!='B')),'complete pair actual dispatch count')
        require(type(r['gpu_start_s']) in (int,float) and type(r['gpu_end_s']) in (int,float) and r['gpu_end_s']>r['gpu_start_s'],'invalid GPU timestamps')
        us=(r['gpu_end_s']-r['gpu_start_s'])*1e6/reps;require(math.isclose(us,r['gpu_us'],rel_tol=1e-12,abs_tol=1e-9) and us>0,'timing normalization')
        require(guarded(f"case{key[0]}-timing-{key[1]}-s{key[2]}.guarded",131072,True)==golden[key[0]],'timing full output differs');times[key]=us
    require(set(times)==({(i,arm,sample) for i in golden for arm in ('A1','B','A2') for sample in range(samples+1)} if bit(m['timing_enabled']) else set()),'missing timing samples')
    summaries=[]
    if times:
        for i in golden:
            med=[statistics.median(times[i,arm,s] for s in range(1,samples+1)) for arm in ('A1','B','A2')]
            summaries.append({'case':i,'median_complete_pair_gpu_us':med,'time_saved_pct':100*(1-med[1]/((med[0]+med[2])/2)),'control_drift_pct':100*(med[2]/med[0]-1),'faster_than_both_controls':med[1]<min(med[0],med[2])})
    return {'pass':True,'cpu_only':True,'cases':len(golden),'negative_controls':11,'raw_files':len(files),'raw_bytes':sum(len(b) for b in data.values()),'all_payload_sha_recomputed':True,'full_output_and_partial_stats_bytes_equal':True,'source_clone_only_finalstore_difference':True,'timing_records':len(times),'summaries':summaries,'stdout_sha256':sha(read(a.stdout)),'manifest_sha256':sha(read(a.raw/'artifact-manifest.json')),'reviewer_sha256':sha(read(Path(__file__))),'fullmodel_speedup_proven':False,'gpu_physical_overlap_proven':False,'scope':'Saved seeded component fixture, complete attention+inverse-RoPE pair; not real model hidden-state or cold-DRAM/full-model performance proof'}
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('raw','stdout','source-identity','source-root','binary','golden-metallib','candidate-metallib','out'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();require(not a.out.exists(),'output must be fresh');r=review(a);a.out.write_text(json.dumps(r,indent=2,sort_keys=True,allow_nan=False)+'\n');print(json.dumps({'pass':True,'report':str(a.out),'cases':r['cases']}))
if __name__=='__main__':main()
