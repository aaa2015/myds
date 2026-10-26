#!/usr/bin/env python3
"""Freeze/CPU-compile a complete-library exact narrow-weight decode probe.

No GPU submission or model execution. The native29-shader translation unit is
byte-exact; seven independent probe symbols are appended. No registration,
runtime flag, global optimization or production file is changed.
"""
import argparse
import ast
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys

sys.dont_write_bytecode = True
MARKER=b'\n#line 1 "myds_narrow_decode.metal.inc"\n'
GEMV={
 'kernel_myds_narrow_router_bf16_nr1':dict(dtype='BF16',nr0=1,round_bf16=False),
 'kernel_myds_narrow_e4m3_f32_nr1':dict(dtype='E4M3',nr0=1,round_bf16=False),
 'kernel_myds_narrow_e4m3_f32_nr2':dict(dtype='E4M3',nr0=2,round_bf16=False),
 'kernel_myds_narrow_e4m3_bf16_nr2':dict(dtype='E4M3',nr0=2,round_bf16=True)}
EXPAND=('kernel_myds_narrow_expand_router_f32','kernel_myds_narrow_expand_e4m3_f32','kernel_myds_narrow_expand_e4m3_f16')
DENSE_FIELDS=[('ne00','int32_t',0),('ne01','int32_t',4),('ne02','int32_t',8),('nb00','uint64_t',16),
 ('nb01','uint64_t',24),('nb02','uint64_t',32),('nb03','uint64_t',40),('ne10','int32_t',48),
 ('ne11','int32_t',52),('ne12','int32_t',56),('nb10','uint64_t',64),('nb11','uint64_t',72),
 ('nb12','uint64_t',80),('nb13','uint64_t',88),('ne0','int32_t',96),('ne1','int32_t',100),
 ('nr0','int32_t',104),('r2','int16_t',108),('r3','int16_t',110)]
FAMILIES={'router':dict(cols=5120,rows=384,dtype='BF16'),
          'engram':dict(cols=6144,rows=25600,dtype='E4M3'),
          'index_qb':dict(cols=1280,rows=4096,dtype='E4M3')}

def require(ok,message):
    if not ok:raise ValueError(message)
def sha(b):return hashlib.sha256(b).hexdigest()
def descriptor(p):
    p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=sha(b))
def write(p,j):p.write_text(json.dumps(j,indent=2,sort_keys=True)+'\n')
def block(s,anchor):
    require(s.count(anchor)==1,'unique source anchor: '+anchor)
    a=s.index(anchor);b=s.index('{',a);n=1;c=b+1
    while n and c<len(s):n+=(s[c]=='{')-(s[c]=='}');c+=1
    require(n==0,'complete source block: '+anchor);return s[a:c]

def literals(raw):
    tree=ast.parse(raw);wanted={'PREAMBLE','SHADER_FILES','COPY_KERNELS'};out={}
    for node in tree.body:
        if isinstance(node,ast.Assign)and len(node.targets)==1 and isinstance(node.targets[0],ast.Name)and node.targets[0].id in wanted:
            n=node.targets[0].id;require(n not in out,'unique literal '+n);out[n]=ast.literal_eval(node.value)
    require(set(out)==wanted and type(out['SHADER_FILES'])is list and len(out['SHADER_FILES'])==29 and len(set(out['SHADER_FILES']))==29,'actual29shader production literal inventory')
    require(type(out['PREAMBLE'])is str and type(out['COPY_KERNELS'])is str,'literal production source fragments')
    return out

def assemble_native(root):
    raw=(root/'tools/build_metallib.py').read_bytes();p=literals(raw);payloads={'tools/build_metallib.py':raw};records=[];native=bytearray(p['PREAMBLE'].encode()+b'\n')
    for ordinal,n in enumerate(p['SHADER_FILES']):
        rel=Path(n);require(type(n)is str and not rel.is_absolute()and '..'not in rel.parts,'safe production shader path')
        f=root/'metal'/rel;require(f.is_file()and not f.is_symlink(),'regular native source');b=f.read_bytes();b.decode();require(b'\r'not in b,'native source newline bytes preserved')
        native.extend(f'\n#line 1 "{n}"\n'.encode());start=len(native);native.extend(b)
        records.append(dict(index=ordinal,path='metal/'+n,bytes=len(b),sha256=sha(b),source_begin=start,source_end=len(native)));payloads['metal/'+n]=b
    native.extend(p['COPY_KERNELS'].encode()+b'\n');native=bytes(native)
    require(all(native[x['source_begin']:x['source_end']]==payloads[x['path']]for x in records),'complete byte-exact native source prefix')
    return native,payloads,records

def e4m3_f32_bits(code,scale_bits):
    """CPU spelling of integer shader decode, limited to finite normal F32 weights."""
    require(type(code)is int and 0<=code<256 and type(scale_bits)is int and 0<scale_bits<0x7f800000 and scale_bits&0x7fffff==0,'positive normal power-of-two scale and byte code')
    sign=(code&128)<<24;mag=code&127
    if not mag:return sign
    if mag==127:return sign|0x7fc00000
    se=scale_bits>>23;e=mag>>3;m=mag&7
    if e:exp=se+e-7;mant=m<<20
    else:
        shift=2 if m>=4 else 1 if m>=2 else 0;exp=se+shift-9;mant=(m-(1<<shift))<<(23-shift)
    require(0<exp<255,'qualified F16-valued result is normal finite F32')
    return sign|(exp<<23)|mant

def exact_f16_bits(bits):
    sign=(bits>>16)&0x8000;e=(bits>>23)&255;m=bits&0x7fffff
    if bits&0x7fffffff==0:return sign
    if 113<=e<=142 and m&0x1fff==0:return sign|((e-112)<<10)|(m>>13)
    if 103<=e<=112:
        shift=126-e;significand=m|0x800000
        if significand&((1<<shift)-1)==0:return sign|(significand>>shift)
    return sign|0x7e00

def cpu_codec_contract():
    checked=0;half_checked=0
    # Every code including both signed zeros/NaN codes, across the complete
    # finite-F16 relevant power-of-two scale envelope. No source model read.
    for power in range(-30,17):
        scale=math.ldexp(1.,power);sb=struct.unpack('<I',struct.pack('<f',scale))[0]
        for c in range(256):
            got=e4m3_f32_bits(c,sb);m=c&127;e=m>>3;f=m&7
            if m==127:
                require(got&0x7fffffff==0x7fc00000,'NaN code diagnostic');checked+=1;continue
            value=(math.ldexp(1+f/8,e-7)if e else math.ldexp(f,-9))*scale
            if c&128:value=-value
            ref=struct.unpack('<I',struct.pack('<f',value))[0];require(got==ref,'integer code+scale reconstructs exact F32 bits');checked+=1
            try:h=struct.pack('<e',value)
            except OverflowError:continue
            back=struct.unpack('<e',h)[0]
            if struct.pack('<f',back)!=struct.pack('<f',value):continue
            require(exact_f16_bits(got)==struct.unpack('<H',h)[0],'qualified F16 signed-zero/subnormal/normal exact packing');half_checked+=1
    return dict(code_scale_pairs_checked=checked,all256_codes_checked=True,scale_power_range_inclusive=[-30,16],qualified_F16_pairs_checked=half_checked,integer_reconstruction_passed=True,model_or_GPU_used=False)

def validate_problem(family,symbol,args,buffer_bytes=None):
    require(family in FAMILIES and symbol in GEMV,'known family and selected GEMV symbol')
    f=FAMILIES[family];v=GEMV[symbol];require(v['dtype']==f['dtype'],'family symbol encoding')
    require(all(type(args[n])is int for n,_,_ in DENSE_FIELDS),'strict integer ABI fields')
    cols,rows=f['cols'],f['rows'];elem=2 if v['dtype']=='BF16'else 1
    expected=dict(ne00=cols,ne01=rows,ne02=1,nb00=elem,nb01=cols*elem,nb02=cols*rows*elem,nb03=cols*rows*elem,
        ne10=cols,ne11=1,ne12=1,nb10=4,nb11=cols*4,nb12=cols*4,nb13=cols*4,ne0=rows,ne1=1,nr0=v['nr0'],r2=1,r3=1)
    require(args==expected,'exact single-token contiguous scoped native ABI with narrow byte strides')
    require(family!='router'or not v['round_bf16'],'router output keeps native F32')
    spans=dict(codes=cols*rows*elem,input=cols*4,output=rows*4,scales=0 if elem==2 else(cols//32)*(rows//32)*4)
    if buffer_bytes is not None:require(set(buffer_bytes)==set(spans)and all(type(buffer_bytes[k])is int and buffer_bytes[k]>=n for k,n in spans.items()),'complete buffer extents')
    return dict(required_buffer_bytes=spans,grid=[(rows+v['nr0']-1)//v['nr0'],1,1],threads=[32,8,1],dynamic_threadgroup_bytes=32*v['nr0']*4,requires_native_buffer_alignment=dict(codes=16,input=16,output=4,scales=4),base_address_and_writer_alias_validation_required=True)

def source_contract(payloads,candidate):
    dense=payloads['metal/dense.metal'].decode();args=block(dense,'struct ds4_metal_args_mul_mv {');fields=re.findall(r'\b(int|short|ulong)\s+(\w+)\s*;',args)
    kinds={'int32_t':'int','int16_t':'short','uint64_t':'ulong'}
    require(fields==[(kinds[t],n)for n,t,_ in DENSE_FIELDS],'current native Metal ABI fields/order')
    types={'int32_t':ctypes.c_int32,'int16_t':ctypes.c_int16,'uint64_t':ctypes.c_uint64}
    class A(ctypes.Structure):_fields_=[(n,types[t])for n,t,_ in DENSE_FIELDS]
    require(ctypes.sizeof(A)==112 and ctypes.alignment(A)==8 and all(getattr(A,n).offset==off for n,_,off in DENSE_FIELDS),'C native ABI112 offsets')
    native=block(dense,'void kernel_mul_mv_t_t_4_impl(');fused=block(dense,'void kernel_mul_mv_f16_f32_bf16_nr2_impl(');reduce=block(dense,'static inline void helper_mv_reduce_and_write(')
    for token in ('const short NSG = FC_mul_mv_nsg;','constexpr short NB  = 32;','constexpr short NF  = 16;','sumq += dot(float4(xb4[i]), float4(yl4[i]));','sumf[row] += sumq;','yb4 += NSG*NF*NW/4;'):
        require(token in native and token in fused,'original vector4/native fused order: '+token)
    require(re.findall(r'kernel void (\w+)\(',candidate)==list(EXPAND)+['NAME'],'exact seven-kernel macro catalog; no extra entries')
    require(set(re.findall(r'MYDS_NARROW_MV\((kernel_\w+),',candidate))==set(GEMV),'exact GEMV symbol catalog')
    require('constexpr short NSG = 8;'in candidate and 'FC_mul_mv_nsg'not in candidate and 'sumq += dot(float4(w4[i]), float4(yl4[i]));'in candidate and 'sumf[row] += sumq;'in candidate and 'yb4 += NSG * NF * NW / 4;'in candidate,'candidate fixed NSG8 same K walk and arithmetic order')
    require('helper_mv_reduce_and_write<NR0, ROUND_BF16>'in candidate and '#pragma clang fp'not in candidate,'unchanged native reduction and global math settings')
    require('scale *'not in candidate and '* scale'not in candidate and ' * s256'not in candidate,'integer weight materialization has no floating dequant scale to reassociate')
    require(all(n not in dense for n in (*GEMV,*EXPAND)),'fresh independent candidate symbols')
    return dict(dense_abi=dict(type='ds4_metal_args_mul_mv',bytes=112,alignment=8,fields=[dict(name=n,type=t,offset=o)for n,t,o in DENSE_FIELDS]),
        expand_abi=dict(type='myds_narrow_expand_args',bytes=16,alignment=8,fields=[dict(name='cols',type='uint32_t',offset=0),dict(name='rows',type='uint32_t',offset=4),dict(name='count',type='uint64_t',offset=8)]),
        native_raw_template_sha256=sha(native.encode()),native_fused_template_sha256=sha(fused.encode()),native_reduction_sha256=sha(reduce.encode()),
        exact_reduction_reused=True,native_tail_zero_iterations_for_scope=True,candidate_function_constants=[],native_oracle_function_constants={'600':8},
        decode_only_admission_required=True,representation_qualification_required=True,scope=FAMILIES,CPU_codec=cpu_codec_contract(),GPU_numerical_identity_proven=False)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--source-root',required=True,type=Path);ap.add_argument('--out-dir',required=True,type=Path);ap.add_argument('--candidate-source',type=Path);ap.add_argument('--harness-source',type=Path);ap.add_argument('--with-harness',action='store_true');ap.add_argument('--emit-only',action='store_true');ap.add_argument('--metal-minver',default='15.0');ap.add_argument('--metal-std',default='');a=ap.parse_args()
    require(not a.out_dir.is_symlink(),'output symlink refused')
    root=a.source_root.resolve(strict=True);out=a.out_dir.resolve();require(not out.exists()and not out.is_relative_to(root),'new private output outside source root')
    candidate_path=a.candidate_source or Path(__file__).with_name('narrow_decode.metal.inc');require(candidate_path.is_file()and not candidate_path.is_symlink(),'regular candidate fragment');candidate_raw=candidate_path.read_bytes();require(b'\r'not in candidate_raw,'candidate LF bytes')
    native,payloads,shaders=assemble_native(root);candidate=candidate_raw.decode();contract=source_contract(payloads,candidate)
    require(MARKER not in native and all(n.encode()not in native for n in (*GEMV,*EXPAND)),'fresh append')
    combined=native+MARKER+candidate_raw;proof=dict(kind='complete_TU_append_inverse',complete_native_prefix_byte_exact=True,inverse_remove_only_recorded_suffix_restores_native_byte_for_byte=True,native_bytes=len(native),native_sha256=sha(native),append_marker=MARKER.decode(),candidate_source_begin=len(native)+len(MARKER),candidate_source_bytes=len(candidate_raw),candidate_source_sha256=sha(candidate_raw),combined_bytes=len(combined),combined_sha256=sha(combined))
    require(combined[:proof['native_bytes']]==native and combined[proof['native_bytes']:]==MARKER+candidate_raw,'complete suffix-removal native inverse')
    require(re.fullmatch(r'[0-9]+(?:\.[0-9]+){0,2}',a.metal_minver)is not None and (not a.metal_std or re.fullmatch(r'[A-Za-z0-9_.-]+',a.metal_std)),'safe explicit compiler options')
    require(not any(os.environ.get(k)for k in('MYDS_METALLIB_DEFINES','MYDS_METAL_MINVER','MYDS_METAL_STD')),'explicit compiler flags only; no inherited Metal overrides')
    opts=['-mmacosx-version-min='+a.metal_minver]+(['-std='+a.metal_std]if a.metal_std else[])+['-O3','-ffast-math','-DDS4_METAL_HC_STABLE=1','-DDS4_METAL_NORM_RSQRT_DISABLE=1']
    harness=a.harness_source or(Path(__file__).with_name('bench_narrow_decode.m')if a.with_harness else None)
    if harness is not None:require(harness.is_file()and not harness.is_symlink(),'regular standalone harness')
    frozen=dict(payloads);frozen['tools/build_narrow_decode_probe.py']=Path(__file__).read_bytes();frozen['tools/narrow_decode.metal.inc']=candidate_raw
    if harness is not None:frozen['tools/bench_narrow_decode.m']=harness.read_bytes()
    out.mkdir(parents=True);source=out/'source'
    for name,b in frozen.items():f=source/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(b)
    for name,b in [('native.metal',native),('all.metal',combined),('candidate.metal.inc',candidate_raw)]: (out/name).write_bytes(b)
    write(out/'source-transform.json',proof);inventory={n:dict(bytes=len(b),sha256=sha(b))for n,b in frozen.items()}
    commands=[]
    for n,m in [('native','native.metal'),('probe','all.metal')]:commands.extend([['xcrun','-sdk','macosx','metal']+opts+['-c',str(out/m),'-o',str(out/(n+'.air'))],['xcrun','-sdk','macosx','metallib',str(out/(n+'.air')),'-o',str(out/(n+'.metallib'))]])
    if harness is not None:commands.append(['xcrun','clang','-O2','-fobjc-arc','-Wno-deprecated-declarations','-mmacosx-version-min=26.0','-framework','Foundation','-framework','Metal',str(source/'tools/bench_narrow_decode.m'),'-o',str(out/'bench_narrow_decode')])
    env=dict(os.environ);env.setdefault('DEVELOPER_DIR','/Applications/Xcode.app/Contents/Developer')
    identity=dict(schema_version=1,kind='narrow_decode_full_library_source_identity',subject='exact_narrow_decode_weights',source_root=str(root),source_inventory=inventory,source_inventory_sha256=sha(json.dumps(inventory,sort_keys=True,separators=(',',':')).encode()),source_file_count=len(inventory),shader_records=shaders,production_shader_count=29,native_source_sha256=sha(native),combined_source_sha256=sha(combined),candidate_source_original=str(candidate_path.resolve()),harness_source_original=str(harness.resolve())if harness else None,production_registered=False,runtime_connected=False,default_enabled=False,production_adoption=False,gpu_work_submitted=False,contract=contract)
    write(out/'source-identity.json',identity)
    manifest=dict(identity,kind='narrow_decode_complete_library_build',source_identity=descriptor(out/'source-identity.json'),source_transform=descriptor(out/'source-transform.json'),commands=commands,compile_options=opts,compile_environment={k:env.get(k)for k in('DEVELOPER_DIR','SDKROOT','MACOSX_DEPLOYMENT_TARGET','TOOLCHAINS')},compile_only_pass=False,compilation_skipped=a.emit_only,metallibs={},artifacts={},source_files={'native':descriptor(out/'native.metal'),'probe':descriptor(out/'all.metal')},native_source=descriptor(out/'native.metal'),probe_source=descriptor(out/'all.metal'),selected_symbols=list(GEMV),expand_symbols=list(EXPAND),GEMV_catalog=GEMV,native_oracle_symbols=['kernel_mul_mv_f32_f32_4','kernel_mul_mv_f16_f32_4','kernel_mul_mv_f16_f32_4_bf16_nr2'])
    write(out/'build-manifest.json',manifest)
    if not a.emit_only:
        statuses=[]
        for i,cmd in enumerate(commands):
            with(out/f'compile-{i}.stdout').open('wb')as stdout,(out/f'compile-{i}.stderr').open('wb')as stderr:proc=subprocess.run(cmd,env=env,stdout=stdout,stderr=stderr)
            statuses.append(dict(argv=cmd,returncode=proc.returncode,stdout=descriptor(out/f'compile-{i}.stdout'),stderr=descriptor(out/f'compile-{i}.stderr')))
            manifest['compilation_records']=statuses;write(out/'build-manifest.json',manifest);require(proc.returncode==0,'CPU compilation failed; preserve complete logs')
        manifest['metallibs']={n:descriptor(out/(n+'.metallib'))for n in('native','probe')};manifest['air']={n:descriptor(out/(n+'.air'))for n in('native','probe')}
        if harness is not None:manifest['artifacts']['bench_narrow_decode']=descriptor(out/'bench_narrow_decode')
        actual={str(p.relative_to(source)):dict(bytes=p.stat().st_size,sha256=sha(p.read_bytes()))for p in sorted(source.rglob('*'))if p.is_file()}
        require(actual==inventory and (out/'native.metal').read_bytes()==native and (out/'all.metal').read_bytes()==combined and (out/'candidate.metal.inc').read_bytes()==candidate_raw,'compiler-visible sources unchanged before and after compilation')
        manifest['compiler_visible_source_prepost_unchanged']=True
        manifest['compile_only_pass']=True;write(out/'build-manifest.json',manifest)
    print(json.dumps(dict(build_manifest=descriptor(out/'build-manifest.json'),source_files=len(inventory),production_shader_count=29,compile_only_pass=manifest['compile_only_pass'],gpu_work_submitted=False,CPU_codec=contract['CPU_codec']),indent=2))
if __name__=='__main__':
    try:main()
    except Exception as error:print(type(error).__name__+': '+str(error),file=sys.stderr);raise SystemExit(1)
