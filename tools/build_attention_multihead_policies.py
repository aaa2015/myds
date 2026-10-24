#!/usr/bin/env python3
"""Generate independent uninstrumented arithmetic-policy attention probes.

Reads the frozen initial H2/H4 shader, never edits it or production sources.
Output may be appended to the diagnostic builder's combined Metal source.
--compile performs AIR/metallib compilation only; no GPU work is submitted.
All policies retain the original 200B ABI, eight resources, logical NSG/NWG1,
H2 physical SIMD mapping and 23040B scratch. They are diagnostic candidates,
not numerical fixes until real full-output 0-ULP gates pass.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

EXPECTED = 'c0ad53e0dd6e97344eb8dc29d5f249c222af37b8665a88b731af085a79221ad3'
PREFIX = 'kernel_myds_prefill_attention_head2_'
SUFFIX = '_f16_dk512_dv512'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def once(text, old, new):
    require(text.count(old) == 1, 'Nonunique policy anchor: '+old[:80])
    return text.replace(old,new,1)


def generate(original):
    text=original[:original.index('typedef decltype(')]
    text=text.replace('myds_prefill_stage_half16','probe_mh_policy_stage_half16')
    text=once(text,'template<ushort HEADS>\nkernel void kernel_myds_prefill_attention_multihead(',
              'template<ushort HEADS, ushort POLICY>\nkernel void probe_mh_policy_impl(')
    text=once(text,'        {\n            threadgroup const half4 * pq4 = sq4 + tx;',
              '        float policy_ms = 0.0f;\n        {\n            threadgroup const half4 * pq4 = sq4 + tx;')
    qk_start='            FOR_UNROLL (short pass = 0; pass < 2; ++pass) {'
    qk_end='            ss[tx] = fma(mqk[tx], args.scale, (float) sm[tx]);'
    at=text.index(qk_start);end=text.index(qk_end,at)
    tiled=text[at:end]
    direct='''            if constexpr ((POLICY & 8u) != 0) {
                device const half4 * pk4 = (device const half4 *)(k + ic*args.nb11);
                pk4 += tx;
                FOR_UNROLL (short cc = 0; cc < C; ++cc) {
                    FOR_UNROLL (short ii = 0; ii < DK4/NL; ++ii) {
                        mqk[cc] += dot((float4)pk4[cc*128 + ii*NL], (float4)pq4[ii*NL]);
                    }
                    mqk[cc] = simd_sum(mqk[cc]);
                }
            } else {
'''+tiled+'''            }
'''
    text=text[:at]+direct+text[end:]
    # Bit4 controls both ordinary online-softmax S and the final sink update.
    update='S = S*ms + simd_sum(vs);'
    require(text.count(update)==2,'Expected block and sink S updates')
    text=text.replace(update,'''if constexpr ((POLICY & 4u) != 0) {
                S = precise::fma(S, ms, simd_sum(vs));
            } else {
                S = S*ms + simd_sum(vs);
            }''')
    block_scale='''            FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                so4[ii*NL] *= ms;
            }
        }
        simdgroup_barrier'''
    text=once(text,block_scale,'''            if constexpr ((POLICY & 1u) != 0) {
                policy_ms = ms;
            } else {
                FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                    so4[ii*NL] *= ms;
                }
            }
        }
        simdgroup_barrier''')
    at=text.index(qk_start,text.index('            const auto sst = ss;'))
    end=text.index('            FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {\n                so4[ii*NL] += lo[ii];',at)
    pv=text[at:end]
    product='''                        lo[ii] += float4(float4(pv4[cc*128 + ii*NL])*
                                        float4(sst[pass*16 + cc]));'''
    pv=once(pv,product,'''                        if constexpr ((POLICY & 2u) != 0) {
                            lo[ii] = precise::fma(float4(pv4[cc*128 + ii*NL]),
                                                float4(sst[pass*16 + cc]),lo[ii]);
                        } else {
                            lo[ii] += float4(float4(pv4[cc*128 + ii*NL])*
                                            float4(sst[pass*16 + cc]));
                        }''')
    direct='''            if constexpr ((POLICY & 16u) != 0) {
                device const half4 * pv4 = (device const half4 *)(v + ic*args.nb21);
                pv4 += tx;
                FOR_UNROLL (short cc = 0; cc < C; ++cc) {
                    FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                        lo[ii] += float4(float4(pv4[cc*128 + ii*NL])*float4(sst[cc]));
                    }
                }
            } else {
'''+pv+'''            }
'''
    text=text[:at]+direct+text[end:]
    text=once(text,'                so4[ii*NL] += lo[ii];',
              '''                if constexpr ((POLICY & 1u) != 0) {
                    so4[ii*NL] = precise::fma(so4[ii*NL],float4(policy_ms),lo[ii]);
                } else {
                    so4[ii*NL] += lo[ii];
                }''')
    configurations=[(f'policy{i}',i) for i in range(1,8)]+[('direct_k',8),('direct_v',16),('direct_kv',24)]
    text='// Generated diagnostic-only policies; original formal shader is untouched.\n'+text
    text+='\ntypedef decltype(probe_mh_policy_impl<2,1>) probe_mh_policy_t;\n'
    for label,value in configurations:
        text+=f'template [[host_name("{PREFIX+label+SUFFIX}")]]\nkernel probe_mh_policy_t probe_mh_policy_impl<2,{value}>;\n'
    return text,configurations


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',required=True,type=Path)
    parser.add_argument('--out-dir',required=True,type=Path)
    parser.add_argument('--compile',action='store_true')
    args=parser.parse_args()
    source=args.source_root.resolve(strict=True)
    original=source/'metal/myds_prefill_attention_multihead.metal'
    require(sha(original)==EXPECTED,'Source is not the frozen initial H2/H4 shader')
    args.out_dir.mkdir(parents=True,exist_ok=False)
    generated,configurations=generate(original.read_text())
    shader=args.out_dir/'attention_multihead_policies.metal'
    shader.write_text(generated)
    report=dict(schema_version=1,source_root=str(source),original_shader_sha256=EXPECTED,
                generated_shader_path=str(shader),generated_shader_sha256=sha(shader),gpu_work_submitted=False,
                numerical_parity_proven=False,abi_bytes=200,buffer_indices=list(range(8)),
                dynamic_threadgroup_bytes=23040,threads=[32,2,1],groups=[32,'rows',1],logical_nsg=1,logical_nwg=1,
                policies={label:dict(symbol=PREFIX+label+SUFFIX,bits=value) for label,value in configurations},
                semantics={'bit1':'Keep old so4 until PV lo complete, then precise::fma(old,ms,lo); sink scaling unchanged',
                           'bit2':'PV lo accumulation uses precise::fma(v,weight,lo) in cc order',
                           'bit4':'S online and sink accumulation uses precise::fma(S,ms,simd_sum(vs))',
                           'bit8':'Direct device K QK ablation; V remains shared tile',
                           'bit16':'Direct device V PV ablation; K remains shared tile'})
    if args.compile:
        spec=importlib.util.spec_from_file_location('frozen_build_metallib',source/'tools/build_metallib.py')
        build=importlib.util.module_from_spec(spec);spec.loader.exec_module(build)
        combined=args.out_dir/'policies_all.metal'
        combined.write_text(build.PREAMBLE+'\n'+(source/'metal/flash_attn.metal').read_text()+'\n'+generated)
        air,library=args.out_dir/'policies.air',args.out_dir/'policies.metallib'
        env=dict(os.environ,DEVELOPER_DIR='/Applications/Xcode.app/Contents/Developer')
        commands=[['xcrun','-sdk','macosx','metal','-mmacosx-version-min=26.0','-O3','-ffast-math',
                   '-DDS4_METAL_HC_STABLE=1','-DDS4_METAL_NORM_RSQRT_DISABLE=1','-c',str(combined),'-o',str(air)],
                  ['xcrun','-sdk','macosx','metallib',str(air),'-o',str(library)]]
        with (args.out_dir/'compile.log').open('wb') as log:
            for command in commands:
                subprocess.run(command,check=True,env=env,stdout=log,stderr=subprocess.STDOUT)
        report.update(compile_only_pass=True,compile_commands=commands,metallib_path=str(library),metallib_sha256=sha(library))
    (args.out_dir/'policies.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps(report,indent=2,sort_keys=True))


if __name__=='__main__':
    main()
