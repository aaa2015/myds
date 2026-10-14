#!/usr/bin/env python3
"""CPU-only full Shared Q4 component reader. No GPU/runtime activity.
Deployment capture/freeze bind the candidate library's actual role externally;
component parity/timing do not prove native full-model adoption or decode speed.
"""
import argparse,hashlib,json,math,statistics,struct
from pathlib import Path
G=64;W=6635520;D=5120;M=2304
GOLDEN_SHA="e17018b547d24e558c64c61b3493d343ac3bd595397e207bb1a763d9ad544d57"
SYMS=['kernel_myds_v41_shared_gate_up_swiglu_q4','kernel_mul_mv_q4_K_dense_f32_bf16','kernel_myds_v41_shared_gate_up_swiglu_q4_v41_nb','kernel_myds_q4_v41_shared_down_nb_bf16']
def need(x,m):
 if not x:raise ValueError(m)
def pairs(v):
 d={}
 for k,x in v:need(k not in d,'duplicate JSON key');d[k]=x
 return d
def reject_const(x):raise ValueError('nonfinite JSON constant '+x)
def parse(b):return json.loads(b,object_pairs_hook=pairs,parse_constant=reject_const)
def h(p):
 x=hashlib.sha256()
 with Path(p).open('rb') as f:
  while b:=f.read(1048576):x.update(b)
 return x.hexdigest()
def integer(x):return type(x)is int
def flag(x):
 need(type(x)is bool or type(x)is int and x in (0,1),"strict native boolean/0/1")
 return bool(x)
def close(x,y):return math.isfinite(x) and abs(x-y)<=max(1e-8,abs(y)*1e-8)
def floats(b,bf16):
 need(len(b)%4==0,'float payload length')
 for u, in struct.iter_unpack('<I',b):
  need(u!=0xa5a5a5a5 and (u&0x7f800000)!=0x7f800000,'unwritten/nonfinite output')
  if bf16:need(u&0xffff==0,'non BF16 output')
def function(s,name):
 p=s.index(name);a=s.rfind('\n',0,p)+1;b=s.index('{',p);d=1;e=b+1
 while d:d+=(s[e]=='{')-(s[e]=='}');e+=1
 return s[a:e]
def source_math(root):
 gu=(root/'metal/myds_v41_moe.metal').read_text();down=(root/'metal/myds_q4_v41.metal').read_text()
 need(hashlib.sha256(gu.encode()[:54576]).hexdigest()=='4e54f4c000c45cb0a802c1acb6d497b79e830bb44f45841ec37b636aae7f6eaf','original GU source prefix')
 need(hashlib.sha256(down.encode()[:5294]).hexdigest()=='3f9feae495f022cdbf22004c397c2cfa9a6fa9005389e6d18589620e32a67cde','original Q4 body prefix')
 old=function(gu,'void kernel_myds_v41_shared_gate_up_swiglu_q4_impl(');new=function(gu,'static inline void myds_shared_q4_v41_nb_gu_impl(')
 new=new.replace('static inline void myds_shared_q4_v41_nb_gu_impl(', 'void kernel_myds_v41_shared_gate_up_swiglu_q4_impl(').replace('    constexpr short NR0 = 2;\n','')
 new=new.replace('const int nb = 20;', 'const int nb = (int)args.in_dim / QK_K;').replace('    const int first_row = (tgpig.x * NSG + sgitg) * NR0;', '    const int first_row = (tgpig.x * NSG + sgitg) * NR0;\n    if (first_row >= (int)args.out_dim) return;')
 new=new.replace('const uint offset0 = uint(first_row + row) * 2880u;', 'const uint64_t offset0 = (uint64_t)(first_row + row) * args.row_bytes;\n        const bool active = (first_row + row < (int)args.out_dim);').replace('src_gate + offset0','src_gate + (active ? offset0 : 0)').replace('src_up   + offset0','src_up   + (active ? offset0 : 0)')
 new=new.replace('        FOR_UNROLL (short row = 0; row < NR0; ++row) {\n', '        FOR_UNROLL (short row = 0; row < NR0; ++row) {\n            if (first_row + row >= (int)args.out_dim) break;\n').replace('for (short row = 0; row < NR0; ++row) {\n        float tot_g', 'for (short row = 0; row < NR0 && first_row + row < (int)args.out_dim; ++row) {\n        float tot_g')
 need(new==old and hashlib.sha256(old.encode()).hexdigest()=='8db55f0c8a211cd8eb81e38d52a82401ce68077c68d94697bd4ff8223cb402a8','exact GU FP/source restoration')
 wrapper=function(down,'kernel void kernel_myds_q4_v41_shared_down_nb_bf16(')
 need('myds_q4_v41_body<5120,2304,true,true>(args,w,x,y,group,lane,sg); (void)scratch;' in wrapper,'Down unchanged math-body/shape identity')
 return True
def seeded_input(c):
 mask=(1<<64)-1;rng=88172645463325252^(((c+1)*0x9e3779b97f4a7c15)&mask);out=bytearray()
 for _ in range(D):
  rng=(rng^(rng<<13))&mask;rng^=rng>>7;rng=(rng^(rng<<17))&mask
  u=struct.unpack('<I',struct.pack('<f',((rng%20001)-10000)/4096))[0]
  if c%2==0:u=(u+0x7fff+((u>>16)&1))&0xffff0000
  out.extend(struct.pack('<I',u))
 return bytes(out)
def main():
 p=argparse.ArgumentParser()
 for k in ['raw','stdout','golden-metallib','candidate-metallib','binary','source-identity','source-root','deployment-identity','deployment-freeze','out']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();need(not a.out.exists(),'refuse overwrite report');root=a.raw.resolve();m=parse((root/'artifact-manifest.json').read_bytes())
 need(m['schema_version']==1 and m['probe']=='shared_q4_v41_nb' and m['status']=='complete','full completed schema required')
 need(m['fullmodel'] is False and m['cold_dram_proven'] is False and m['deployment_identity_externally_required'] is True,'scope claims')
 for field,path in [('binary_sha256',a.binary),('source_identity_sha256',a.source_identity),('golden_metallib_sha256',a.golden_metallib),('candidate_metallib_sha256',a.candidate_metallib)]:need(m[field]==h(path),'actual '+field)
 need(m['golden_metallib_sha256']==GOLDEN_SHA,'immutable actual post467 production golden SHA')
 need(m['golden_metallib_sha256']!=m['candidate_metallib_sha256'],'separate golden/candidate libraries')
 freeze=parse(a.deployment_freeze.read_bytes());capture=parse(a.deployment_identity.read_bytes())
 need(freeze['metallib_sha256']==capture['metallib_sha256']==m['candidate_metallib_sha256'],'actual deployed candidate SHA')
 need(freeze['production_binary_sha256']==capture['binary_sha256'],'actual deployed binary capture/freeze')
 source=freeze['source_sha256'];need(type(source)is dict and len(source)>200,'full frozen source set')
 for name,sha in source.items():
  n=Path(name);need(not n.is_absolute() and '..' not in n.parts,'source path escape');f=a.source_root/n;need(f.is_file() and not f.is_symlink() and h(f)==sha,'full frozen source '+name)
 for name in ['metal/myds_v41_moe.metal','metal/myds_q4_v41.metal','src/myds_metal.m','src/myds_decode_candidates.inc','src/myds_v41_moe.c','include/myds_metal.h','tools/bench_shared_q4_v41.m']:need(name in source,'required source identity '+name)
 need(source_math(a.source_root),'source math contract')
 si=parse(a.source_identity.read_bytes());need(si['cpu_compile_only'] is True and si['gpu_run'] is False and si['private_metallib_compiled'] is False and si['deployment_authority_externally_required'] is True,'actual standalone builder contract')
 need(si['deployed_metallib_sha256']==m['candidate_metallib_sha256'] and si['output_sha256']['bench_shared_q4_v41']==m['binary_sha256'],'actual builder library/binary binding')
 for name,sha in si['input_sha256'].items():need(name in source and source[name]==sha,'builder input vs full production source '+name)
 need('tools/bench_shared_q4_v41.m' in si['input_sha256'] and 'src/myds_gguf.c' in si['input_sha256'] and 'include/myds_gguf.h' in si['input_sha256'],'actual producer inputs')
 layers=m['layers'];need(type(layers)is list and 1<=len(layers)<=40,'layers');ids=[int(x) for x in layers];need(len(set(ids))==len(ids) and all(0<=x<40 for x in ids),'numeric layer uniqueness')
 cases=m['cases_per_layer'];samples=m['samples'];reps=m['repetitions'];need(all(integer(x) for x in [cases,samples,reps,m['total_cases']]) and 1<=cases<=32 and 3<=samples<=31 and 1<=reps<=64,'probe counts')
 count=len(ids)*cases;need(m['total_cases']==count,'total cases')
 descriptors=m['tensors'];need(len(descriptors)==len(ids)*3,'all tensors')
 inventory=m['files'];need(type(inventory)is list,'raw inventory');expected=set()
 for layer in ids:
  for j in range(3):
   for pre in ['weight','final-weight']:expected.add(f'{pre}-layer-{layer:02d}-kind-{j}.raw')
 for c in range(count):
  expected.update([f'input-case-{c:03d}.raw',f'final-input-case-{c:03d}.raw'])
  for arm in range(3):
   for phase in ['parity','final']:
    for part in ['gu','down']:expected.add(f'{phase}-arm-{arm}-case-{c:03d}-{part}.raw')
 need(len(inventory)==len(expected),'complete raw payload count');files={}
 for e in inventory:
  n=e['path'];need(type(n)is str and n in expected and n not in files,'raw name/uniqueness');f=root/n;need(f.is_file() and not f.is_symlink() and integer(e['bytes']) and f.stat().st_size==e['bytes'] and h(f)==e['sha256'],'raw file size/SHA');files[n]=f
 need(set(files)==expected and {str(x.relative_to(root)) for x in root.rglob('*') if x.is_file()}==expected|{'artifact-manifest.json'},'exact file inventory')
 def body(n,size):
  b=files[n].read_bytes();need(len(b)==size+2*G and b[:G]==bytes([0xa5])*G and b[-G:]==bytes([0xa5])*G,'whole payload/guards '+n);return b[G:-G]
 for li,layer in enumerate(ids):
  for j in range(3):
   td=descriptors[3*li+j];suffix=['ffn_gate_shexp.weight','ffn_up_shexp.weight','ffn_down_shexp.weight'][j]
   need(td['name']==f'blk.{layer}.{suffix}' and integer(td['layer']) and td['layer']==layer and td['shape']==([D,M] if j==2 else [M,D]) and td['ggml_type']==12 and td['bytes']==W and integer(td['absolute_offset']) and td['absolute_offset']>=m['model']['header_bytes'],'GGUF descriptor')
   wb=body(f'weight-layer-{layer:02d}-kind-{j}.raw',W);need(hashlib.sha256(wb).hexdigest()==td['sha256'] and wb==body(f'final-weight-layer-{layer:02d}-kind-{j}.raw',W),'whole real tensor bytes/hash unchanged')
 for c in range(count):
  b=body(f'input-case-{c:03d}.raw',D*4);need(b==seeded_input(c%cases),'complete seeded original input');need(b==body(f'final-input-case-{c:03d}.raw',D*4),'whole input bytes unchanged');floats(b,False)
  for part,size in [('gu',M*4),('down',D*4)]:
   golden=body(f'parity-arm-0-case-{c:03d}-{part}.raw',size);floats(golden,True)
   for phase in ['parity','final']:
    for arm in range(3):
     b=body(f'{phase}-arm-{arm}-case-{c:03d}-{part}.raw',size);floats(b,True);need(b==golden,'whole cold golden output parity')
 records=[parse(x) for x in a.stdout.read_bytes().splitlines() if x];need(records and records[0]['kind']=='identity' and records[1]['kind']=='bindings','stdout identity sequence')
 initial={k:v for k,v in m.items() if k not in ['status','files','bindings','device']};need(records[0]['manifest']==initial,'original identity stdout/manifest')
 need(records[1]['bindings']==m['bindings'] and records[1]['device']==m['device'],'original PSO bindings stdout')
 bindings=m['bindings'];need(len(bindings)==4,'four PSO bindings')
 for i,b in enumerate(bindings):
  down=i%2==1;need(b['symbol']==SYMS[i] and b['provider']==('immutable_golden' if i<2 else 'requested_candidate_library') and b['args_bytes']==(112 if down else 16),'actual symbol/provider ABI')
  need(b['threads']==([32,4,1] if down else [128,1,1]) and b['groups']==([640,1,1] if down else [288,1,1]) and b['dynamic_threadgroup_bytes']==(256 if down else 0) and b['thread_execution_width']==32 and b['max_threads']>=128,'actual geometry')
  need(b['ordered_constants']==([{'index':600,'is_bool':False,'value':4}] if down else []),'exact ordered constants')
 pos=2
 for arm in range(3):
  for c in range(count):
   r=records[pos];pos+=1;need(r['kind']=='parity' and all(integer(r[k]) for k in ['arm','case','layer','seed']) and r['arm']==arm and r['case']==c and r['layer']==ids[c//cases] and r['seed']==c%cases,'unique ordered whole parity records')
   clamp=0 if c%cases==0 else 1e-6 if c%cases==1 else 1e-5 if c%cases==2 else 10;need(abs(r['clamp']-clamp)<=max(1e-12,abs(clamp)*1e-7),'native clamp')
   need(flag(r['completed_cb']) and flag(r['guards_finite_bf16_written']) and flag(r['full_bytes_exact']),'producer actual parity flags')
 times=[[],[],[]];last_wall=0
 for arm in range(3):
  for sample in range(samples+1):
   r=records[pos];pos+=1;need(r['kind']=='timing' and all(integer(r[k]) for k in ['arm','sample','pairs','dispatches']) and r['arm']==arm and r['sample']==sample and flag(r['warmup'])==(sample==0) and flag(r['completed_cb']) and r['component_only'] is True,'ordered warm/formal ABA')
   n=count*reps;need(r['pairs']==n and r['dispatches']==2*n and r['gpu_start_s']>0 and r['gpu_end_s']>r['gpu_start_s'] and r['wall_start_s']>=last_wall and r['wall_end_s']>r['wall_start_s'],'completed monotonic timestamps/dispatch geometry');last_wall=r['wall_end_s']
   need(close(r['gpu_us_per_pair'],(r['gpu_end_s']-r['gpu_start_s'])*1e6/n) and close(r['wall_us_per_pair'],(r['wall_end_s']-r['wall_start_s'])*1e6/n),'raw timestamp timing arithmetic')
   if sample:times[arm].append(r['gpu_us_per_pair'])
 need(pos==len(records)-1 and records[-1]=={'kind':'complete','cases':count,'arms':3,'formal_samples_per_arm':samples,'full_bytes_exact':True,'fullmodel':False},'complete unique original stdout')
 med=list(map(statistics.median,times));report={'pass':True,'cpu_only_review':True,'raw_files':len(files),'complete_cases':count,'complete_parity_records':3*count,'whole_inputs_weights_gu_down_exact':True,'actual_deployed_candidate_library_verified':True,'golden_metallib_sha256':m['golden_metallib_sha256'],'candidate_metallib_sha256':m['candidate_metallib_sha256'],'full_source_files_rehashed':len(source),'formal_gpu_median_us_per_pair':med,'control_drift_percent':100*(med[2]-med[0])/med[0],'candidate_time_saved_vs_faster_control_percent':100*(min(med[0],med[2])-med[1])/min(med[0],med[2]),'fullmodel_adoption_proven':False,'decode_speedup_proven':False,'cold_dram_proven':False,'reader_sha256':h(Path(__file__)),'artifact_sha256':{'manifest':h(root/'artifact-manifest.json'),'stdout':h(a.stdout),'source_identity':h(a.source_identity),'deployment_identity':h(a.deployment_identity),'deployment_freeze':h(a.deployment_freeze)}}
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,sort_keys=True))
if __name__=='__main__':main()
