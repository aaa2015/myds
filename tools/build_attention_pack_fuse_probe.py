#!/usr/bin/env python3
"""Compile-only isolated direct-pack probe; no GPU or service operations."""
import argparse, hashlib, importlib.util, json, os, subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out-dir',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[1];out=a.out_dir.resolve();assert root!=out and root not in out.parents
out.mkdir(parents=True,exist_ok=True)
env=dict(os.environ);env.setdefault('DEVELOPER_DIR','/Applications/Xcode.app/Contents/Developer')
resolve=lambda q:subprocess.check_output(['xcrun','--sdk','macosx',q],env=env,text=True).strip()
sdk=resolve('--show-sdk-path');clang=subprocess.check_output(['xcrun','--sdk','macosx','--find','clang'],env=env,text=True).strip()
spec=importlib.util.spec_from_file_location('builder',root/'tools/build_metallib.py');bm=importlib.util.module_from_spec(spec);spec.loader.exec_module(bm)
paths=['tools/build_metallib.py','tools/build_attention_pack_fuse_probe.py','tools/bench_attention_pack_fuse.m','tests/test_attention_packed.m','src/myds_v41_gpu.c','src/myds_metal.m','src/myds_decode_candidates.inc','src/myds_gguf.c','src/myds_log.c',*[str(n.relative_to(root))for n in(root/'include').glob('*.h')],*['metal/'+n for n in bm.SHADER_FILES]]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();before={n:sha(root/n)for n in sorted(set(paths))}
commands=[['python3',str(root/'tools/build_metallib.py'),'--bin-dir',str(out),'--build-dir',str(out/'build')]]
for n in ('myds_gguf','myds_log'):commands.append([clang,'-isysroot',sdk,'-O3','-Wall','-Wextra','-std=c11','-Iinclude','-Ithird_party','-c','src/'+n+'.c','-o',str(out/(n+'.o'))])
commands.append([clang,'-isysroot',sdk,'-O3','-Wall','-Wextra','-fobjc-arc','-Iinclude','-Isrc','-Itests','-Ithird_party','tools/bench_attention_pack_fuse.m',str(out/'myds_gguf.o'),str(out/'myds_log.o'),'-framework','Foundation','-framework','Metal','-lm','-lpthread','-o',str(out/'bench_attention_pack_fuse')])
with(out/'build.log').open('w')as log:
 log.write(json.dumps({'sdk':sdk,'clang':clang,'developer_dir':env['DEVELOPER_DIR']})+'\n');log.flush()
 for c in commands:log.write(json.dumps(c)+'\n');log.flush();subprocess.run(c,cwd=root,env=env,stdout=log,stderr=log,check=True)
after={n:sha(root/n)for n in before};assert after==before,'Build source mutated; reject freeze'
identity={'compile_only':True,'GPU_run':False,'toolchain':{'sdk_path':sdk,'sdk_version':resolve('--show-sdk-version'),'clang':clang,'compiler_version':subprocess.check_output([clang,'--version'],env=env,text=True).strip(),'developer_dir':env['DEVELOPER_DIR'],'all_clang_isysroot_explicit':True},'input_sha256':before,'commands':commands,'output_sha256':{n:sha(out/n)for n in ('myds.metallib','bench_attention_pack_fuse')}}
(out/'source-identity.json').write_text(json.dumps(identity,indent=2)+'\n');print(json.dumps({'pass':True,'compile_only':True,'out_dir':str(out)}))
