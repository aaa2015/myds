#!/usr/bin/env python3
"""CPU-only, read-only model export for exact narrow-reader GPU qualification.
Creates a NEW fixture directory; never rewrites GGUF or runs a Metal device.
The frozen CPU auditor and encoder are reused byte-for-byte. Every actual
payload/code/scale SHA must match the prior complete50-tensor qualification.
"""
import argparse, ctypes, hashlib, importlib.util, json, math, os
from pathlib import Path
import struct, sys
sys.dont_write_bytecode=True
AUDITOR_SHA='f1e9a600b732a26f01042babe0b5fe4b672a2842196d0f60c3f3c042d3cdcc55'
WORKER_SHA='712f41c2d89d9835bf2e0d4ebaf6db1eab02df3233eb73cd7f7281cc856e1fa7'
AUDIT_SHA='495064f318033e61a48ed73e0f01c7453edecd462b44922ac3942b3de8456dbb'
FAMILIES=('router','index_qb','engram')

def require(ok,message):
 if not ok:raise ValueError(message)
def file_sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def descriptor(p):return dict(path=str(Path(p).resolve()),bytes=Path(p).stat().st_size,sha256=file_sha(p))
def dump(p,x):
 with Path(p).open('x')as f:json.dump(x,f,indent=2,sort_keys=True);f.write('\n')
def input_bytes(columns,profile,seed):
 result=bytearray();u=seed&0xffffffff
 for i in range(columns):
  u=(1664525*u+1013904223)&0xffffffff;sign=u&0x80000000
  bits=sign|((120+(u%9))<<23)|(u&0x7fffff)
  if profile==0:bits&=0xffff0000
  if i%37==0:bits=sign
  elif profile==2 and i%53==0:bits=sign|(u&0x7fffff or 1)
  result.extend(struct.pack('<I',bits))
 return bytes(result)
def synthetic_stripe(columns,row,family):
 # Only1024values per scale are built in Python, then tile bytes; no huge
 # per-element Python loop or model-sized temporary allocation.
 scale_exp=(-16,-10,-3,0,6);cache={}
 def block(e):
  if e not in cache:
   a=bytearray()
   for i in range(1024):
    code=(i*37+(row//32)*11)%256
    if code&127==127:code&=0xfe
    if family=='router':
     special=(0x00000000,0x80000000,0x00010000,0x80010000,0x3f800000,0xbf800000)
     bits=special[i%len(special)]if i%7==0 else ((code&128)<<24)|((112+(code%24))<<23)|((code&127)<<16)
     a.extend(struct.pack('<I',bits))
    else:
     exp=(code>>3)&15;mant=code&7
     v=math.ldexp(mant,-9)if exp==0 else math.ldexp(8+mant,exp-10)
     if code&128:v=-v
     a.extend(struct.pack('<e',math.ldexp(v,e)))
   cache[e]=bytes(a)
  return cache[e]
 width=4 if family=='router'else 2;row_bytes=32*width
 return b''.join(block(scale_exp[(bc+row//32)%5])[ri*row_bytes:(ri+1)*row_bytes]for ri in range(32)for bc in range(columns//32))
def encode_matrix(out,case_id,family,columns,rows,stripe,lib,auditor,provenance=None,expected=None):
 require(family in FAMILIES and type(columns)is int and type(rows)is int and columns>0 and columns%32==0 and rows>0 and rows%32==0,'whole32x32 qualification shape')
 d=out/'payloads'/case_id;d.mkdir(parents=True,exist_ok=False);projection=family!='router';width=2 if projection else 4
 paths={n:d/(n+'.bin')for n in('weight','codes','scales')};totals={k:0 for k in auditor.STAT_FIELDS if not k.startswith('first_')};restored_hash=hashlib.sha256()
 with paths['weight'].open('xb')as fw,paths['codes'].open('xb')as fc,paths['scales'].open('xb')as fs:
  for row in range(0,rows,32):
   raw=stripe(row);require(len(raw)==32*columns*width,'complete original source stripe')
   src=ctypes.create_string_buffer(raw,len(raw));n=32*columns;code=ctypes.create_string_buffer(n if projection else n*2);back=ctypes.create_string_buffer(n*width);stats=auditor.Stats()
   if projection:
    scales=ctypes.create_string_buffer(columns//32*4);mask=ctypes.create_string_buffer(columns//32)
    rc=lib.v41_audit_e4m3(src,columns,code,scales,back,mask,ctypes.byref(stats));require(mask.raw==b'\1'*(columns//32),'every block must qualify; failed partial export retained')
    for value, in struct.iter_unpack('<f',scales.raw):require(math.isfinite(value)and value>0 and math.frexp(value)[0]==.5,'positive finite power2 scale')
    fs.write(scales.raw)
   else:rc=lib.v41_audit_router(src,n,code,back,ctypes.byref(stats))
   require(rc==0 and not any(getattr(stats,k)for k in('failed_blocks','roundtrip_mismatches','shader_f32_mismatches','infinity','nan')),'finite exact CPU qualification required')
   require(back.raw==raw,'complete reconstructed source bytes')
   fw.write(raw);fc.write(code.raw);restored_hash.update(back.raw)
   for k in totals:totals[k]+=getattr(stats,k)
 desc={n:descriptor(p)for n,p in paths.items()};require(desc['weight']['sha256']==restored_hash.hexdigest(),'complete source/restored stream identity')
 if expected:
  require(expected['qualification_passed']is True and expected['name']==provenance['tensor_name'],'prior qualified tensor identity')
  for n,key in(('weight','source_sha256'),('codes','codes_stream_sha256')):require(desc[n]['sha256']==expected[key],'fresh export matches prior audited stream: '+n)
  if projection:require(desc['scales']['sha256']==expected['scales_stream_sha256'],'fresh scale stream matches prior audit')
 profile=int(case_id.split('-')[-1])%3 if case_id.split('-')[-1].isdigit()else 0
 x=d/'input.bin';x.write_bytes(input_bytes(columns,profile,0x12345678+rows+columns))
 return dict(id=case_id,family=family,columns=columns,rows=rows,profile=('BF16representable','fullF32_signedzero','fullF32_signedzero_subnormal')[profile],outputs=['f32']if not projection else['f32','bf16'],weight=desc['weight'],codes=desc['codes'],scales=desc['scales'],input=descriptor(x),provenance=dict(provenance or{},CPU_exact_qualification=True,counts=totals,reconstructed_source_sha256=restored_hash.hexdigest()))
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--auditor',type=Path,required=True);p.add_argument('--upstream-patches',type=Path,required=True);p.add_argument('--out-root',type=Path,required=True);p.add_argument('--model',type=Path);p.add_argument('--prior-audit',type=Path);p.add_argument('--synthetic',action='store_true');p.add_argument('--cc',default='/usr/bin/clang');args=p.parse_args()
 require(args.synthetic != bool(args.model),'choose synthetic or model, exclusively')
 require(file_sha(args.auditor)==AUDITOR_SHA and file_sha(args.auditor.with_name('v41_narrow_weights_worker.c'))==WORKER_SHA,'frozen CPU auditor/worker SHA')
 require(args.out_root.is_absolute() and not args.out_root.exists() and not args.out_root.is_symlink(),'new absolute fixture directory')
 if args.model:require(args.prior_audit and file_sha(args.prior_audit)==AUDIT_SHA,'actual completed prior50tensor qualification required')
 args.out_root.mkdir();art=args.out_root/'worker';art.mkdir();(args.out_root/'payloads').mkdir()
 spec=importlib.util.spec_from_file_location('frozen_narrow_auditor',args.auditor);a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a);lib,worker=a.build_worker(art,args.cc,args.upstream_patches);model=None;cases=[]
 try:
  if args.synthetic:
   for family in FAMILIES:
    for i,(cols,rows)in enumerate(((128,64),({'router':5120,'index_qb':1280,'engram':6144}[family],{'router':384,'index_qb':4096,'engram':25600}[family]),(256,96))):
     cases.append(encode_matrix(args.out_root,'synthetic-'+family+'-'+str(i),family,cols,rows,lambda row,c=cols,f=family:synthetic_stripe(c,row,f),lib,a,dict(synthetic=True,source='deterministic exact BF16 / E4M3 representable F16, no model source')))
   model_identity=None;binding=None
  else:
   prior=json.loads(args.prior_audit.read_text());require(prior['audit_completed']is True and prior['qualification_passed']is True,'prior audit pass')
   before={t['name']:t for t in prior['tensors']};model=a.open_model(args.model);binding=a.validate_scope(model)
   for i,(family,name,dims,kind)in enumerate(a.targets()):
    t=model['tensors'][name];require(t['dims']==dims and t['type']==kind,'exact actual tensor type/shape')
    width=4 if kind==0 else 2;cols,rows=dims
    read=lambda row,t=t,c=cols,w=width:t['reader'].read(t['file_offset']+row*c*w,32*c*w)
    cases.append(encode_matrix(args.out_root,'actual-'+family+'-'+str(i),family,cols,rows,read,lib,a,dict(synthetic=False,tensor_name=name,file_offset=t['file_offset'],type=kind,prior_audit=descriptor(args.prior_audit)),before[name]))
   model_identity=[item for reader in model['readers']for item in reader.finish_identity()];require(all(x['unchanged']for x in model_identity),'model remained readonly/unchanged')
  manifest=dict(schema_version=1,kind='narrow_decode_fixture_set',correctness_only=True,case_count=len(cases),cases=cases,model_identity=model_identity,whole_model_payload_sha256_verified=False,scope_binding=binding,tool=descriptor(Path(__file__)),auditor=descriptor(args.auditor),worker=worker,GPU_work_submitted=False,qualification_passed=True,source_original_bytes=sum(c['weight']['bytes']for c in cases),narrow_payload_and_scales_bytes=sum(c['codes']['bytes']+c['scales']['bytes']for c in cases))
  dump(args.out_root/'fixtures.json',manifest);print(json.dumps(dict(fixtures=descriptor(args.out_root/'fixtures.json'),case_count=len(cases),GPU_work_submitted=False)))
 finally:
  if model:
   for f in model['readers']:f.close()
if __name__=='__main__':
 try:main()
 except Exception as e:print(type(e).__name__+': '+str(e),file=sys.stderr);sys.exit(2)
