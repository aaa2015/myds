#!/usr/bin/env python3
"""CPU-only standalone Shared Q4 probe builder. Never compiles a private Metal
library, creates a Metal device, submits GPU work or initializes the runtime.
The supplied deployed library SHA is recorded; actual deployment authority must
come from the enclosing frozen window's production capture.
"""
import argparse,hashlib,importlib.util,json,os,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(cmd):
 p=subprocess.run(cmd,cwd=R,text=True,capture_output=True)
 if p.returncode:raise RuntimeError(json.dumps({'argv':cmd,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}))
 return p
INPUTS=['tools/build_shared_q4_v41_probe.py','tools/bench_shared_q4_v41.m','tools/review_shared_q4_v41.py','tools/test_shared_q4_v41_nb_cpu.py','src/myds_gguf.c','include/myds_gguf.h','src/myds_metal.m','src/myds_decode_candidates.inc','src/myds_v41_moe.c','include/myds_metal.h','metal/myds_v41_moe.metal','metal/myds_q4_v41.metal','tools/build_metallib.py','Makefile']
def main():
 p=argparse.ArgumentParser();p.add_argument('--out-dir',type=Path,required=True);p.add_argument('--deployed-metallib',type=Path,required=True);a=p.parse_args();out=a.out_dir.resolve();lib=a.deployed_metallib.resolve(strict=True)
 if out.exists():raise ValueError('new output directory required')
 inputs={n:sha(R/n) for n in INPUTS};library_sha=sha(lib)
 spec=importlib.util.spec_from_file_location('shared_source_proof',R/'tools/test_shared_q4_v41_nb_cpu.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);proof=m.source_check()
 sdk=run(['xcrun','--sdk','macosx','--show-sdk-path']).stdout.strip();clang=run(['xcrun','--sdk','macosx','--find','clang']).stdout.strip();version=run([clang,'--version']).stdout;out.mkdir(parents=True)
 commands=[[clang,'-isysroot',sdk,'-O3','-Wall','-Wextra','-std=c11','-Iinclude','-c','src/myds_gguf.c','-o',str(out/'myds_gguf.o')],[clang,'-isysroot',sdk,'-O3','-Wall','-Wextra','-fobjc-arc','-Iinclude','tools/bench_shared_q4_v41.m',str(out/'myds_gguf.o'),'-framework','Foundation','-framework','Metal','-o',str(out/'bench_shared_q4_v41')]]
 logs=[]
 for c in commands:
  cp=run(c);logs.append({'argv':c,'returncode':cp.returncode,'stdout':cp.stdout,'stderr':cp.stderr})
 if inputs!={n:sha(R/n) for n in INPUTS} or sha(lib)!=library_sha:raise ValueError('source/library changed during build')
 report={'cpu_compile_only':True,'gpu_run':False,'private_metallib_compiled':False,'deployment_authority_externally_required':True,'deployed_metallib_path':str(lib),'deployed_metallib_sha256':library_sha,'deployed_metallib_bytes':lib.stat().st_size,'input_sha256':inputs,'source_contract':proof,'toolchain':{'developer_dir':os.environ.get('DEVELOPER_DIR'),'sdk_path':sdk,'sdk_version':run(['xcrun','--sdk','macosx','--show-sdk-version']).stdout.strip(),'clang':clang,'clang_version':version},'commands':commands,'command_results':logs,'output_sha256':{'myds_gguf.o':sha(out/'myds_gguf.o'),'bench_shared_q4_v41':sha(out/'bench_shared_q4_v41')}}
 (out/'source-identity.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'pass':True,'cpu_compile_only':True,'out_dir':str(out),'source_identity_sha256':sha(out/'source-identity.json'),'probe_binary_sha256':report['output_sha256']['bench_shared_q4_v41']},sort_keys=True))
if __name__=='__main__':main()
