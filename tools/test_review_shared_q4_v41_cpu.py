#!/usr/bin/env python3
"""Focused fabricated raw/protocol mutations for Shared Q4 reader, CPU only.
Positive data are fabricated zero-Q4 tensors and outputs, not GPU evidence.
"""
import argparse,contextlib,copy,hashlib,importlib.util,io,json,os,struct,sys,tempfile
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--golden-metallib',type=Path,required=True);p.add_argument('--candidate-metallib',type=Path,required=True);p.add_argument('--binary',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise ValueError('new CPU report required')
 sp=importlib.util.spec_from_file_location('review_shared',R/'tools/review_shared_q4_v41.py');r=importlib.util.module_from_spec(sp);sp.loader.exec_module(r)
 assert r.h(a.golden_metallib)==r.GOLDEN_SHA and r.h(a.candidate_metallib)!=r.GOLDEN_SHA
 with tempfile.TemporaryDirectory(prefix='shared-q4-reader-cpu-') as t:
  d=Path(t);raw=d/'raw';raw.mkdir();source={}
  for n in ['Makefile']+[str(p.relative_to(R)) for part in ['include','src','metal','tools','tests','third_party'] for p in (R/part).rglob('*') if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts]:source[n]=r.h(R/n)
  si={'cpu_compile_only':True,'gpu_run':False,'private_metallib_compiled':False,'deployment_authority_externally_required':True,'deployed_metallib_sha256':r.h(a.candidate_metallib),'input_sha256':{n:source[n] for n in ['tools/bench_shared_q4_v41.m','src/myds_gguf.c','include/myds_gguf.h']},'output_sha256':{'bench_shared_q4_v41':r.h(a.binary)}}
  (d/'source.json').write_text(json.dumps(si));cap={'metallib_sha256':r.h(a.candidate_metallib),'binary_sha256':'a'*64};fr={'metallib_sha256':r.h(a.candidate_metallib),'production_binary_sha256':'a'*64,'source_sha256':source};(d/'capture.json').write_text(json.dumps(cap));(d/'freeze.json').write_text(json.dumps(fr))
  m={'schema_version':1,'probe':'shared_q4_v41_nb','status':'complete','fullmodel':False,'cold_dram_proven':False,'deployment_identity_externally_required':True,'cases_per_layer':3,'samples':3,'repetitions':1,'source_identity_sha256':r.h(d/'source.json'),'binary_sha256':r.h(a.binary),'golden_metallib_sha256':r.h(a.golden_metallib),'candidate_metallib_sha256':r.h(a.candidate_metallib),'layers':['0'],'total_cases':3,'model':{'header_bytes':32},'tensors':[],'bindings':[],'device':'fabricated CPU-only','files':[]}
  def save(n,b):
   (raw/n).write_bytes(b);m['files'].append({'path':n,'bytes':len(b),'sha256':sha(b)})
  guard=bytes([0xa5])*64;weights=guard+bytes(r.W)+guard
  for j,part in enumerate(['ffn_gate_shexp.weight','ffn_up_shexp.weight','ffn_down_shexp.weight']):
   m['tensors'].append({'name':'blk.0.'+part,'layer':0,'shape':[r.D,r.M] if j==2 else [r.M,r.D],'ggml_type':12,'bytes':r.W,'absolute_offset':32+j*r.W,'sha256':sha(bytes(r.W))})
   for phase in ['weight','final-weight']:save(f'{phase}-layer-00-kind-{j}.raw',weights)
  for c in range(3):
   for phase in ['input','final-input']:save(f'{phase}-case-{c:03d}.raw',guard+r.seeded_input(c)+guard)
   for phase in ['parity','final']:
    for arm in range(3):
     for part,size in [('gu',r.M*4),('down',r.D*4)]:save(f'{phase}-arm-{arm}-case-{c:03d}-{part}.raw',guard+bytes(size)+guard)
  for i,n in enumerate(r.SYMS):
   down=i%2==1;m['bindings'].append({'symbol':n,'provider':'immutable_golden' if i<2 else 'requested_candidate_library','args_bytes':112 if down else 16,'ordered_constants':[{'index':600,'is_bool':False,'value':4}] if down else [],'threads':[32,4,1] if down else [128,1,1],'groups':[640,1,1] if down else [288,1,1],'dynamic_threadgroup_bytes':256 if down else 0,'thread_execution_width':32,'static_threadgroup_bytes':0,'max_threads':512})
  initial={k:v for k,v in m.items() if k not in ['status','files','bindings','device']};rows=[{'kind':'identity','manifest':initial},{'kind':'bindings','bindings':m['bindings'],'device':m['device']}]
  for arm in range(3):
   for c in range(3):rows.append({'kind':'parity','arm':arm,'case':c,'layer':0,'seed':c,'clamp':0 if c==0 else 1e-6 if c==1 else 1e-5,'guards_finite_bf16_written':True,'full_bytes_exact':True,'completed_cb':True})
  for arm in range(3):
   for sample in range(4):
    start=1000+arm*10+sample;stop=start+.001;rows.append({'kind':'timing','arm':arm,'warmup':sample==0,'sample':sample,'completed_cb':True,'gpu_start_s':start,'gpu_end_s':stop,'wall_start_s':start,'wall_end_s':stop+.0001,'gpu_us_per_pair':(stop-start)*1e6/3,'wall_us_per_pair':(stop+.0001-start)*1e6/3,'pairs':3,'dispatches':6,'component_only':True})
  rows.append({'kind':'complete','cases':3,'arms':3,'formal_samples_per_arm':3,'full_bytes_exact':True,'fullmodel':False});stdout=d/'stdout'
  def write(mm=m,rr=rows):
   (raw/'artifact-manifest.json').write_text(json.dumps(mm));stdout.write_text(''.join(json.dumps(x)+'\n' for x in rr))
  write()
  argv=['reader','--raw',str(raw),'--stdout',str(stdout),'--golden-metallib',str(a.golden_metallib.resolve()),'--candidate-metallib',str(a.candidate_metallib.resolve()),'--binary',str(a.binary.resolve()),'--source-identity',str(d/'source.json'),'--source-root',str(R),'--deployment-identity',str(d/'capture.json'),'--deployment-freeze',str(d/'freeze.json'),'--out',str(d/'report.json')]
  def call(expect):
   (d/'report.json').unlink(missing_ok=True);old=sys.argv;sys.argv=argv
   try:
    with contextlib.redirect_stdout(io.StringIO()):r.main()
   except (ValueError,KeyError,IndexError,TypeError,AssertionError) as e:
    if expect:raise
    return str(e)
   finally:sys.argv=old
   if not expect:raise AssertionError('mutation accepted')
   return 'PASS fabricated complete raw/protocol'
  call(True);failures=[]
  def mutate(name,fn):
   mm=copy.deepcopy(m);rr=copy.deepcopy(rows);fn(mm,rr);write(mm,rr);reason=call(False);failures.append({'name':name,'rejection':reason});write()
  mutate('missing full parity record',lambda mm,rr:rr.pop(4))
  mutate('duplicate case ID',lambda mm,rr:rr[3].update(case=0))
  mutate('wrong native flag2',lambda mm,rr:rr[12].update(warmup=2))
  mutate('bool case ID',lambda mm,rr:rr[3].update(case=True))
  mutate('false completion',lambda mm,rr:rr[2].update(completed_cb=False))
  mutate('forged timing summary old raw timestamp',lambda mm,rr:rr[12].update(gpu_us_per_pair=1))
  mutate('wrong golden SHA',lambda mm,rr:mm.update(golden_metallib_sha256='b'*64))
  mutate('wrong ABI',lambda mm,rr:mm['bindings'][2].update(args_bytes=112))
  mutate('duplicate raw inventory name',lambda mm,rr:mm['files'].append(mm['files'][0]))
  mutate('incomplete raw inventory',lambda mm,rr:mm['files'].pop())
  mutate('failed manifest',lambda mm,rr:mm.update(status='failed'))
  name='parity-arm-1-case-002-gu.raw';b=(raw/name).read_bytes();(raw/name).write_bytes(b[:-1]);write();failures.append({'name':'truncated full output','rejection':call(False)});(raw/name).write_bytes(b)
  b=bytearray((raw/name).read_bytes());old=bytes(b);b[64:68]=struct.pack('<I',0x7fc00000);(raw/name).write_bytes(b)
  mm=copy.deepcopy(m);next(e for e in mm['files'] if e['path']==name)['sha256']=sha(b);write(mm,rows);failures.append({'name':'NaN payload consistent SHA','rejection':call(False)});(raw/name).write_bytes(old);write()
  name='input-case-001.raw';old=(raw/name).read_bytes();changed=guard+bytes(r.D*4)+guard;(raw/name).write_bytes(changed);mm=copy.deepcopy(m);next(e for e in mm['files'] if e['path']==name)['sha256']=sha(changed);write(mm,rows);failures.append({'name':'zeroed finite original seed input consistent SHA','rejection':call(False)});(raw/name).write_bytes(old);write()
  (raw/'unlisted.raw').write_bytes(b'x');failures.append({'name':'unlisted full payload','rejection':call(False)});(raw/'unlisted.raw').unlink()
  cap['metallib_sha256']='c'*64;(d/'capture.json').write_text(json.dumps(cap));failures.append({'name':'actual KERN candidate SHA mismatch','rejection':call(False)})
  report={'pass':True,'cpu_only':True,'fabricated_raw_positive':True,'real_gpu_evidence':False,'decode_speedup_proven':False,'reader_sha256':r.h(R/'tools/review_shared_q4_v41.py'),'test_sha256':r.h(Path(__file__)),'negative_cases':failures,'negative_count':len(failures)}
  a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,sort_keys=True))
if __name__=='__main__':main()
