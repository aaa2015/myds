#!/usr/bin/env python3
"""Compile-only private attention+rope probe with complete intermediate taps."""
import argparse,hashlib,importlib.util,json,os,subprocess,shutil
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out-dir',required=True)
p.add_argument('--reuse-metallib',type=Path,help='Diagnostic host rebuild: preserve the already measured private library bytes')
p.add_argument('--reuse-source-identity',type=Path,help='Required original library source/build identity for diagnostic reuse')
a=p.parse_args()
root=Path(__file__).resolve().parents[1];out=Path(a.out_dir).resolve()
if root==out or root in out.parents:raise SystemExit('Use a private directory outside repo')
out.mkdir(parents=True,exist_ok=True)
s=(root/'metal/flash_attn.metal').read_text()
def extract(name):
 start=s.index('static inline void '+name+'(');brace=s.index('{',start);depth=1;i=brace+1
 while depth:
  if s[i]=='{':depth+=1
  elif s[i]=='}':depth-=1
  i+=1
 return s[start:i]
original=extract('ds4_flash_attn_vec_packed_register_reduce_f16_512')
candidate=extract('myds_attention_register_rope_fuse_impl')
store='''                const uint oc = (uint)quadrant * (uint)NW + (uint)out_lane;
                const float4 value = reduced * reduce_inv;
                dst4[oc] = myds_attention_rope_store(value, oc, rope);'''
restored=candidate.replace('myds_attention_register_rope_fuse_impl','ds4_flash_attn_vec_packed_register_reduce_f16_512',1).replace('        constant myds_attention_rope_fuse_args & rope,\n','',1).replace(store,'                dst4[quadrant * NW + out_lane] = reduced * reduce_inv;')
assert restored==original,'Reducer clone differs outside finalstore/signature'
def debug(body,name):
 body=body.replace(body.split('(')[0].split()[-1],name,1)
 body=body.replace('        device       char * dst,','        device       char * dst,\n        device char *partial_dump, device char *stats_dump,',1)
 body=body.replace('            sink_scale[(uint)iwg] = out_scale;','''            sink_scale[(uint)iwg] = out_scale;
            device float *debug_stats=(device float *)stats_dump;
            const uint at=((uint)head*32u+(uint)iwg)*3u;
            debug_stats[at]=S; debug_stats[at+1u]=M; debug_stats[at+2u]=out_scale;''',1)
 body=body.replace('            partial_plane[(uint)iwg * 33u + (uint)lane] = lo;','''            partial_plane[(uint)iwg * 33u + (uint)lane] = lo;
            ((device float4 *)partial_dump)[((uint)head*32u+(uint)iwg)*128u+(uint)quadrant*32u+(uint)lane]=lo;''',1)
 return body
wrappers=[]
for variant,name in [('reference','probe_attention_rope_reference_impl'),('fused','probe_attention_rope_fused_impl')]:
 wrappers.append(f'''kernel void probe_attention_rope_{variant}(
 constant myds_attention_rope_fuse_args &fused [[buffer(0)]],
 device const char *q [[buffer(1)]], device const char *k [[buffer(2)]],
 device const char *v [[buffer(3)]], device const char *mask [[buffer(4)]],
 device const char *sinks [[buffer(5)]], device const char *pad [[buffer(6)]],
 device char *out [[buffer(7)]], device char *partial [[buffer(8)]], device char *stats [[buffer(9)]],
 threadgroup char *scratch [[threadgroup(0)]], uint head [[threadgroup_position_in_grid]],
 ushort lane [[thread_index_in_simdgroup]],ushort sg [[simdgroup_index_in_threadgroup]]) {{
 {name}(fused.attention,q,k,v,mask,sinks,pad,out,partial,stats,{"fused," if variant=="fused" else ""}scratch,head,lane,sg);
}}
''')
probe=debug(original,'probe_attention_rope_reference_impl')+'\n'+debug(candidate,'probe_attention_rope_fused_impl')+'\n'+''.join(wrappers)
(out/'probe_attention_rope.metal').write_text(probe)
spec=importlib.util.spec_from_file_location('bm',root/'tools/build_metallib.py');bm=importlib.util.module_from_spec(spec);spec.loader.exec_module(bm)
combined=bm.PREAMBLE+'\n'+''.join('\n#line 1 "'+n+'"\n'+(root/'metal'/n).read_text() for n in bm.SHADER_FILES)+bm.COPY_KERNELS+'\n'+probe
(out/'myds_all.metal').write_text(combined)
env=dict(os.environ);env.setdefault('DEVELOPER_DIR','/Applications/Xcode.app/Contents/Developer')
commands=[['xcrun','-sdk','macosx','metal','-mmacosx-version-min=15.0','-O3','-ffast-math','-DDS4_METAL_HC_STABLE=1','-DDS4_METAL_NORM_RSQRT_DISABLE=1','-c',str(out/'myds_all.metal'),'-o',str(out/'myds_all.air')],
 ['xcrun','-sdk','macosx','metallib',str(out/'myds_all.air'),'-o',str(out/'myds.metallib')]]
reused=None
if a.reuse_metallib:
 assert a.reuse_source_identity,'Reuse requires the original source identity'
 old=json.loads(a.reuse_source_identity.read_text())
 sha=lambda b:hashlib.sha256(b).hexdigest()
 assert sha(a.reuse_metallib.read_bytes())==old['output_sha256']['myds.metallib'],'Reused library SHA differs'
 assert old['original_helper_sha256']==sha(original.encode()) and old['candidate_helper_sha256']==sha(candidate.encode())
 assert old['output_sha256']['probe_attention_rope.metal']==sha(probe.encode()),'Debug shader has changed'
 for name,h in old['input_sha256'].items():
  if name.startswith('metal/') or name=='tools/build_metallib.py':assert sha((root/name).read_bytes())==h,'Library shader source changed'
 if a.reuse_metallib.resolve()!=(out/'myds.metallib').resolve():shutil.copyfile(a.reuse_metallib,out/'myds.metallib')
 reused={'library_sha256':sha(a.reuse_metallib.read_bytes()),'original_source_identity_sha256':sha(a.reuse_source_identity.read_bytes()),'shader_sources_and_debug_probe_unchanged':True}
 commands=[]
for n in ('myds_gguf','myds_log'):
 commands.append(['clang','-O3','-Wall','-Wextra','-std=c11','-Iinclude','-Ithird_party','-c','src/'+n+'.c','-o',str(out/(n+'.o'))])
commands.append(['clang','-O3','-Wall','-Wextra','-fobjc-arc','-Iinclude','-Isrc','-Itests','-Ithird_party','tests/test_attention_rope_fuse.m',str(out/'myds_gguf.o'),str(out/'myds_log.o'),'-framework','Foundation','-framework','Metal','-lm','-lpthread','-o',str(out/'test_attention_rope_fuse')])
with (out/'build.log').open('w') as log:
 for c in commands:
  log.write(json.dumps(c)+'\n');log.flush();subprocess.run(c,cwd=root,env=env,stdout=log,stderr=log,check=True)
sha=lambda b:hashlib.sha256(b).hexdigest()
paths=['metal/flash_attn.metal','metal/dsv41.metal','src/myds_v41_gpu.c','src/myds_metal.m','src/myds_decode_candidates.inc','include/myds_v41_gpu.h','include/myds_metal.h','tests/test_attention_rope_fuse.m','tests/test_attention_packed.m','tools/build_attention_rope_fuse_probe.py','tools/build_metallib.py',*['metal/'+n for n in bm.SHADER_FILES]]
identity={'cpu_compile_only':True,'gpu_run':False,'original_helper_sha256':sha(original.encode()),'candidate_helper_sha256':sha(candidate.encode()),
 'clone_reconstructed_original_exact':True,'original_helper_lines':len(original.splitlines()),'candidate_helper_lines':len(candidate.splitlines()),
 'only_clone_difference':'function signature and final float4 store; partial/stats/reductions/barriers byte-identical after normalization',
 'input_sha256':{n:sha((root/n).read_bytes()) for n in sorted(set(paths))},'commands':commands,
 'output_sha256':{n:sha((out/n).read_bytes()) for n in ('myds.metallib','test_attention_rope_fuse','probe_attention_rope.metal')}}
if reused:identity['diagnostic_reused_measured_library']=reused
(out/'source-identity.json').write_text(json.dumps(identity,indent=2,sort_keys=True)+'\n')
print(json.dumps({'built':True,'cpu_compile_only':True,'out':str(out),'clone_reconstructed_original_exact':True}))
