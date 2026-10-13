#!/usr/bin/env python3
"""Private source-frozen component build; never edits standard shader lists or production outputs."""
from pathlib import Path
import hashlib, json, importlib.util, subprocess, os
import argparse
parser=argparse.ArgumentParser(description='Build a private packed-attention register/barrier probe; submits no GPU work')
parser.add_argument('--out-dir',required=True,help='Private output directory outside repository')
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
out=Path(args.out_dir).expanduser().resolve()
if out==root or root in out.parents: raise SystemExit('Output must be outside repository')
out.mkdir(parents=True,exist_ok=True)
source=(root/'metal/flash_attn.metal').read_text()
start=source.index('static inline void ds4_flash_attn_vec_packed8_reduce_f16_512(')
end=source.index('// Host-visible fused packed-reduce decode FlashAttention',start)
original=source[start:end]
def debug_helper(body,name):
    body=body.replace('static inline void ds4_flash_attn_vec_packed8_reduce_f16_512(',f'template <bool CAPTURE>\nstatic inline void {name}(')
    body=body.replace('device       char * dst,','device       char * dst,\n        device       char * partial_dump,\n        device       char * stats_dump,')
    body=body.replace('sink_scale[(uint)iwg] = out_scale;','sink_scale[(uint)iwg] = out_scale;\n            if (CAPTURE) {\n                device float * dump = (device float *)stats_dump;\n                const uint at = ((uint)head * 32u + (uint)iwg) * 3u;\n                dump[at+0] = S; dump[at+1] = M; dump[at+2] = out_scale;\n            }')
    body=body.replace('partial_plane[(uint)iwg * 33u + (uint)lane] = lo;','partial_plane[(uint)iwg * 33u + (uint)lane] = lo;\n            if (CAPTURE) {\n                device float4 * dump = (device float4 *)partial_dump;\n                dump[((uint)head * 32u + (uint)iwg) * 128u + (uint)quadrant * 32u + (uint)lane] = lo;\n            }')
    return body
reference=debug_helper(original,'probe_attn_tg_impl')
candidate=debug_helper(original,'probe_attn_register_impl')
candidate=candidate.replace('/* 24,448 dynamic bytes: shared Q, eight score/mask banks, all 32','/* 22,400 dynamic bytes: shared Q, lane-local score/mask registers, all 32')
candidate=candidate.replace('    threadgroup half *score_banks =\n        (threadgroup half *)(q_shared + DK4);\n    threadgroup volatile float *weights =\n        (threadgroup volatile float *)(score_banks + NSG * SH);','    threadgroup volatile float *weights =\n        (threadgroup volatile float *)(q_shared + DK4);')
candidate=candidate.replace('    threadgroup half *bank = score_banks + (short)sgitg * SH;\n    threadgroup float *ss = (threadgroup float *)bank;\n    threadgroup half *sm = bank + 2 * C;\n','')
candidate=candidate.replace('        ss[lane] = 0.0f;\n        sm[lane] = (half)0.0h;\n','')
candidate=candidate.replace('            sm[lane] = pm[ic + lane];\n            if (simd_max(sm[lane]) > -MAXHALF) {','            const half lane_mask = pm[ic + lane];\n            if (simd_max(lane_mask) > -MAXHALF) {')
candidate=candidate.replace('                ss[lane] = fma(lane_mqk, args.scale,\n                               (float)sm[lane]);','                const float lane_score = fma(lane_mqk, args.scale,\n                                             (float)lane_mask);')
candidate=candidate.replace('                const float score = ss[lane];','                const float score = lane_score;')
candidate=candidate.replace('                ss[lane] = vs;\n','')
candidate=candidate.replace('                weights[(uint)iwg * C + (uint)lane] = ss[lane];','                weights[(uint)iwg * C + (uint)lane] = vs;')
candidate=candidate.replace('        simdgroup_barrier(mem_flags::mem_threadgroup);\n','').replace('                simdgroup_barrier(mem_flags::mem_threadgroup);\n','')
assert 'ss[' not in candidate and 'sm[' not in candidate and 'score_banks' not in candidate
assert original.count('simdgroup_barrier')==4 and candidate.count('simdgroup_barrier')==0
assert original.count('threadgroup_barrier')==4 and candidate.count('threadgroup_barrier')==4
entry_start=source.index('kernel void kernel_myds_flash_attn_vec_packed8_reduce_f16_dk512_dv512(',end)
entry_end=source.index('/* Opt-in physical-layout candidates.',entry_start)
entry=source[entry_start:entry_end]
wrappers=[]
for variant,helper in [('tg','probe_attn_tg_impl'),('register','probe_attn_register_impl')]:
    for capture in (False,True):
        name='probe_attention_'+variant+('_debug' if capture else '')
        body=entry.replace('kernel_myds_flash_attn_vec_packed8_reduce_f16_dk512_dv512',name)
        body=body.replace('        threadgroup  char * shmem', '        device       char * partial_dump              [[buffer(8)]],\n        device       char * stats_dump                [[buffer(9)]],\n        threadgroup  char * shmem')
        body=body.replace('ds4_flash_attn_vec_packed8_reduce_f16_512(',helper+('<true>(' if capture else '<false>('))
        body=body.replace('args, q, k, v, mask, sinks, pad, dst, shmem,','args, q, k, v, mask, sinks, pad, dst, partial_dump, stats_dump, shmem,')
        wrappers.append(body)
probe='/* Private probe. Not included in official SHADER_FILES.\n * Original arithmetic and logical splits remain fixed; debug adds full partial/stat exports. */\n'+reference+candidate+'\n'.join(wrappers)
tracked=root/'metal/probe_attention_register.metal'
if tracked.read_text()!=probe: raise SystemExit('Tracked probe no longer matches exact current packed8 helper reconstruction')
(out/'probe_attention_register.metal').write_text(probe)
spec=importlib.util.spec_from_file_location('bm',root/'tools/build_metallib.py')
bm=importlib.util.module_from_spec(spec);spec.loader.exec_module(bm)
combined=bm.PREAMBLE+'\n'+''.join(f'\n#line 1 "{name}"\n'+(root/'metal'/name).read_text() for name in bm.SHADER_FILES)+bm.COPY_KERNELS+'\n'+probe
(out/'probe_all.metal').write_text(combined)
identity={'source_head':subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'original_helper_sha256':hashlib.sha256(original.encode()).hexdigest(),'reference_helper_sha256':hashlib.sha256(reference.encode()).hexdigest(),'candidate_helper_sha256':hashlib.sha256(candidate.encode()).hexdigest(),'probe_shader_sha256':hashlib.sha256(probe.encode()).hexdigest(),'shared_memory_bytes':{'reference':24448,'candidate':22400},'simdgroup_barriers_per_virtual_split':{'reference':4,'candidate':0},'threadgroup_barriers_lexical':4,'remaining_threadgroup_barrier_sites':4,'GPU_run':False}
(out/'source_identity.json').write_text(json.dumps(identity,indent=2)+'\n')
inputs=[root/'tools/bench_attention_register.m',root/'tests/test_attention_packed.m',root/'src/myds_metal.m',root/'src/myds_v41_gpu.c',root/'src/myds_gguf.c',root/'src/myds_log.c',*(root/'metal'/name for name in bm.SHADER_FILES),tracked]
identity['inputs_sha256']={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
commands=[
 ['xcrun','-sdk','macosx','metal','-mmacosx-version-min=15.0','-O3','-ffast-math','-DDS4_METAL_HC_STABLE=1','-DDS4_METAL_NORM_RSQRT_DISABLE=1','-c',str(out/'probe_all.metal'),'-o',str(out/'probe_all.air')],
 ['xcrun','-sdk','macosx','metallib',str(out/'probe_all.air'),'-o',str(out/'myds.metallib')],
 ['clang','-O3','-Wall','-Wextra','-std=c11','-Iinclude','-Ithird_party','-c','src/myds_gguf.c','-o',str(out/'myds_gguf.o')],
 ['clang','-O3','-Wall','-Wextra','-std=c11','-Iinclude','-Ithird_party','-c','src/myds_log.c','-o',str(out/'myds_log.o')],
 ['clang','-O3','-Wall','-Wextra','-fobjc-arc','-Iinclude','-Isrc',str(root/'tools/bench_attention_register.m'),str(out/'myds_gguf.o'),str(out/'myds_log.o'),'-framework','Foundation','-framework','Metal','-lm','-lpthread','-o',str(out/'bench_attention_register')],
]
environment=dict(os.environ);environment.setdefault('DEVELOPER_DIR','/Applications/Xcode.app/Contents/Developer')
with (out/'build.log').open('w') as log:
    for command in commands:
        log.write(json.dumps(command)+'\n');log.flush()
        subprocess.run(command,cwd=root,env=environment,stdout=log,stderr=log,check=True)
identity['build_commands']=commands
identity['outputs_sha256']={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in ('bench_attention_register','myds.metallib')}
(out/'source_identity.json').write_text(json.dumps(identity,indent=2)+'\n')
print(json.dumps({'built':True,'out_dir':str(out),'GPU_run':False,'sha256':identity['outputs_sha256']}))
