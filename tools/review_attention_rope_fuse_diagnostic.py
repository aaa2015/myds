#!/usr/bin/env python3
"""CPU audit of failed/isolated attention diagnostics. Never accepts a candidate."""
import argparse,hashlib,json,math,struct
from pathlib import Path
from review_attention_rope_fuse import parse,read,require,bit
def sha(b):return hashlib.sha256(b).hexdigest()
def diff(a,b):
    require(len(a)==len(b) and len(a)%4==0,'diff extent')
    aa=[v[0] for v in struct.iter_unpack('<I',a)];bb=[v[0] for v in struct.iter_unpack('<I',b)]
    ix=[i for i,(x,y) in enumerate(zip(aa,bb)) if x!=y]
    return {'same':not ix,'different_words':len(ix),'first_index':ix[0] if ix else -1,'reference_bits':f'{aa[ix[0]]:08x}' if ix else 'none','candidate_bits':f'{bb[ix[0]]:08x}' if ix else 'none'}
def review(a):
    m=parse(read(a.raw/'artifact-manifest.json'))
    require(m['status'] in ('failed_parity','diagnostic_complete'),'not a completed isolated diagnostic or saved parity failure')
    require(m['diagnostic_timings_accepted'] is False,'diagnostic performance overclaim')
    for field,path in [('golden_metallib_sha256',a.golden_metallib),('candidate_metallib_sha256',a.candidate_metallib),('binary_sha256',a.binary),('source_identity_sha256',a.source_identity)]:require(sha(read(path))==m[field],'external identity '+field)
    source=parse(read(a.source_identity))
    require(source['output_sha256']['myds.metallib']==m['candidate_metallib_sha256'] and source['output_sha256']['test_attention_rope_fuse']==m['binary_sha256'],'builder output identities')
    reused=source.get('diagnostic_reused_measured_library')
    require(reused and reused['library_sha256']==m['candidate_metallib_sha256'] and reused['shader_sources_and_debug_probe_unchanged'] is True,'diagnostic must preserve measured library')
    data={}
    require({p.name for p in a.raw.iterdir()}==set(m['files'])|{'artifact-manifest.json'},'raw inventory')
    for name,f in m['files'].items():
        require(Path(name).name==name,'unsafe raw name');b=read(a.raw/name)
        require(len(b)==f['bytes'] and sha(b)==f['sha256'],'raw bytes/SHA '+name);data[name]=b
    records=[parse(line) for line in read(a.stdout).splitlines() if line.startswith(b'{')]
    require([r for r in records if r.get('kind')=='diagnostic_case']==m['diagnostics'],'diagnostic stdout/manifest differ')
    require(m['diagnostics'],'no diagnostic case')
    cases=[]
    for r in m['diagnostics']:
        i=r['case'];prefix=f'case{i}';valid=True
        for suffix,n in [('q.f32',131072),('window.f32',262144),('sinks.f32',256),('frequencies.f32',128),('mask.f16',1280)]:
            require(len(data[prefix+'-'+suffix])==n,'complete input extent '+suffix)
            if suffix.endswith('.f32'):require(all(math.isfinite(v[0]) for v in struct.iter_unpack('<f',data[prefix+'-'+suffix])),'input nonfinite')
        def output(suffix,n,bf16=False):
            nonlocal valid
            b=data[prefix+'-'+suffix];require(len(b)==n+32,'full guarded extent '+suffix)
            valid=valid and b[:16]==b[-16:]==bytes([0xa5])*16
            payload=b[16:-16]
            valid=valid and all(math.isfinite(v[0]) for v in struct.iter_unpack('<f',payload))
            valid=valid and all(v[0]!=0xa5a5a5a5 and (not bf16 or v[0]&65535==0) for v in struct.iter_unpack('<I',payload))
            return payload
        unrotated=output('production-unrotated.guarded',131072)
        out=[output(f'out{j}.guarded',131072,j!=1) for j in range(4)]
        partial=[output(f'partial{j}.guarded',4194304) for j in range(2)]
        stats=[output(f'stats{j}.guarded',24576) for j in range(2)]
        comparisons={'reference_debug_unrotated':diff(unrotated,out[1]),'production_fused':diff(out[0],out[3]),'debug_fused':diff(out[0],out[2]),'partial':diff(*partial),'stats':diff(*stats)}
        for key,value in comparisons.items():
            require(value=={**r[key],'same':bit(r[key]['same'])},'native diagnostic mismatch '+key)
        exact=valid and all(x['same'] for x in comparisons.values())
        require(bit(r['case_exact_gate_passed'])==exact and bit(r['guards_finite_poison_bf16_valid'])==valid,'native exact gate differs')
        require(r['candidate_accepted'] is False and r['diagnostic_timings_accepted'] is False and r['actual_production_fused_ran_before_debug'] is True,'diagnostic adoption/run-order declaration')
        if r['mask_mode']==4:require(r['mask_description']=='unmasked_no_effective_sink','mode4 mislabeled')
        tail=[]
        for key in ('production_fused','debug_fused'):
            d=comparisons[key];index=d['first_index']
            if index>=0:
                head,col=divmod(index,512);row={'provider':key,'head':head,'column':col,'pair':(col-448)//2 if col>=448 else None,'reference_bits':d['reference_bits'],'candidate_bits':d['candidate_bits']}
                if col>=448:
                    pair=(col-448)//2;rawbits=struct.unpack_from('<II',unrotated,(head*512+448+2*pair)*4)
                    def bf(u):return (u if u&0x7f800000==0x7f800000 else u+0x7fff+((u>>16)&1))&0xffff0000
                    row.update(first_bf16_reference_inputs=[f'{bf(u):08x}' for u in rawbits],frequency_bits=f"{struct.unpack_from('<I',data[prefix+'-frequencies.f32'],pair*4)[0]:08x}",position=r['pos'])
                tail.append(row)
        cases.append({'case':i,'pos':r['pos'],'keys':r['keys'],'mask_mode':r['mask_mode'],'compressed':bit(r['compressed']),'direct':bit(r['direct']),'full_bytes_comparisons':comparisons,'guards_finite_poison_bf16_valid':valid,'case_exact_gate_passed':exact,'first_tail_difference_inputs':tail})
    return {'status':'diagnostic_recomputed','cpu_only':True,'http_requests_sent':0,'GPU_work_submitted':False,'candidate_accepted':False,'component_performance_accepted':False,'manifest_status':m['status'],'raw_files':len(data),'raw_bytes':sum(len(b) for b in data.values()),'all_payload_SHA_recomputed':True,'cases':cases,'manifest_sha256':sha(read(a.raw/'artifact-manifest.json')),'stdout_sha256':sha(read(a.stdout)),'reader_sha256':sha(read(Path(__file__))),'scope':'Full-byte saved diagnostics, strict original production oracle; a mismatch remains rejection, timings are not accepted evidence'}
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('raw','stdout','source-identity','binary','golden-metallib','candidate-metallib','out'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();require(not a.out.exists(),'out must be fresh');r=review(a);a.out.write_text(json.dumps(r,indent=2,sort_keys=True,allow_nan=False)+'\n');print(json.dumps({'status':r['status'],'candidate_accepted':False,'report':str(a.out)}))
if __name__=='__main__':main()
